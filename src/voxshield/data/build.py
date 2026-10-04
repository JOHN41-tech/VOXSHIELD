"""Dataset build orchestration: corpus in, manifests and a build report out.

One function does the whole thing, :func:`build_dataset`, and it does it in one
fixed order. The order is the point, not an implementation detail:

1. **discover** -- walk the registry, hash, drop byte-identical duplicates;
2. **validate** -- decide which files may become training samples;
3. **split** -- assign whole speakers or sources to sides, on the *source* set;
4. **segment** -- cut accepted sources into windows, inheriting their split;
5. **gate** -- check every identity axis for cross-boundary overlap;
6. **write** -- emit manifests, statistics, and reports.

Splitting before segmentation is what makes a speaker-disjoint claim true. If
windows were cut first, a whole speaker would land on both sides of the boundary
and every downstream number would be quietly optimistic; the order is enforced
here and again inside :func:`~voxshield.data.preprocess.preprocess_sources`,
which refuses a source with no assignment.

Every stage is deterministic given the same configuration and the same corpus.
Nothing in this module invents a value: counts come from the stage that
produced them, and a claim the corpus cannot support is reported as
unavailable rather than assumed.
"""

from __future__ import annotations

import dataclasses
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voxshield.data import discovery, leakage, preprocess, splitting
from voxshield.data import manifest as manifest_io
from voxshield.data.cache import PreprocessingCache
from voxshield.data.config import DataConfig
from voxshield.data.errors import DatasetBuildError
from voxshield.data.gates import GateReport, assert_gates_pass, evaluate_leakage_gates
from voxshield.data.manifest import DatasetStatistics
from voxshield.data.paths import DataPaths
from voxshield.data.preprocess import PreprocessedSource
from voxshield.data.registry import DatasetRegistry
from voxshield.data.schema import SampleRecord, SourceRecord
from voxshield.data.splitting import SplitAssignment
from voxshield.data.validation import ValidationResult, validate_sources


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Everything one build produced.

    Attributes:
        build_id: The build identifier every row and manifest header carries.
        rows: Manifest rows, in manifest order.
        manifests: Written file paths, keyed by manifest name.
        statistics: Corpus summary.
        discovery: Inventory outcome, including duplicates and unreadable files.
        validation: Admission outcome, including every rejection with its reason.
        assignment: The source-to-split assignment and the axis holdouts.
        preprocessing: One entry per accepted source, plus any segmentation
            rejection.
        gate_report: Go/no-go for the configured gates.
        leakage: Cross-boundary overlap per identity axis.
        reports: Written report paths.
        elapsed_seconds: Wall time for the whole build.
        audio_seconds: Source audio fed to the segmenter.
        cache_hits: Sources whose canonical audio came from the cache.
    """

    build_id: str
    rows: tuple[SampleRecord, ...]
    manifests: dict[str, str]
    statistics: DatasetStatistics
    discovery: discovery.DiscoveryResult
    validation: ValidationResult
    assignment: splitting.SplitAssignment
    preprocessing: tuple[preprocess.PreprocessedSource, ...]
    gate_report: GateReport
    leakage: leakage.LeakageReport
    reports: tuple[Path, ...] = ()
    elapsed_seconds: float = 0.0
    audio_seconds: float = 0.0
    cache_hits: int = 0

    @property
    def gate_passed(self) -> bool:
        """Whether every mandatory gate passed."""
        return self.gate_report.passed

    @property
    def real_time_factor(self) -> float | None:
        """Audio seconds per wall second, or ``None`` with no audio processed.

        Below ``1.0`` means the build reads faster than playback; the value is
        reported so a corpus too large for a rebuild is visible before it is
        discovered the hard way.
        """
        if self.elapsed_seconds <= 0.0 or self.audio_seconds <= 0.0:
            return None
        return self.audio_seconds / self.elapsed_seconds

    def to_dict(self) -> dict[str, Any]:
        """A JSON-friendly summary for the build report."""
        return {
            "build_id": self.build_id,
            "rows": len(self.rows),
            "segments_by_split": self.statistics.by_split,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "audio_seconds": round(self.audio_seconds, 3),
            "real_time_factor": (
                None if self.real_time_factor is None else round(self.real_time_factor, 3)
            ),
            "cache_hits": self.cache_hits,
            "gates_passed": self.gate_passed,
            "gate_failures": list(self.gate_report.failures),
            "manifests": dict(sorted(self.manifests.items())),
            "reports": [str(path) for path in self.reports],
        }


def jsonable(value: Any) -> Any:
    """Make a dataclass tree JSON-serialisable.

    Public because it is the only place that knows how a stage report's types
    reduce to JSON, and the CLI's ``voxshield data`` commands print those same
    reports. A second implementation would drift and print something subtly
    different from what ``data/reports/*.json`` holds.

    Args:
        value: Any value from a stage report.

    Returns:
        The same structure with paths stringified and enums reduced to values.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {key: jsonable(item) for key, item in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, payload: Any) -> Path:
    """Write one report as sorted, indented JSON.

    Args:
        path: Destination, created with parents.
        payload: The report body.

    Returns:
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return path


def dataset_provenance(config: DataConfig) -> dict[str, Any]:
    """Per-corpus provenance for the manifest header.

    Args:
        config: Build configuration.

    Returns:
        One entry per configured dataset, keyed by id, with the licence recorded
        as published. A dataset with an undetermined licence keeps
        ``"unknown"`` rather than gaining a guess.
    """
    return {entry.dataset_id: dataclasses.asdict(entry) for entry in config.enabled_datasets()}


def discover_corpus(
    config: DataConfig,
    paths: DataPaths | None = None,
    *,
    registry: DatasetRegistry | None = None,
) -> tuple[discovery.DiscoveryResult, tuple[SourceRecord, ...], DatasetRegistry]:
    """Inventory every configured corpus and drop byte-identical duplicates.

    Args:
        config: Build configuration.
        paths: Resolved data layout.
        registry: A pre-built registry, to avoid re-resolving adapters.

    Returns:
        The inventory, the surviving source records in deterministic order, and
        the registry used.

    Raises:
        DatasetBuildError: No configured corpus is usable, so there is nothing
            to build from. This is a configuration or install problem, and
            saying so is more useful than an empty manifest set.
    """
    data_paths = paths if paths is not None else config.data_paths()
    reg = registry if registry is not None else DatasetRegistry(config)
    records, skips, _descriptions = reg.discover_all()
    if not records:
        detail = "; ".join(f"{item.dataset_id}: {item.reason}" for item in skips) or "none"
        msg = (
            "no usable source files found. Check the dataset paths under "
            f"{data_paths.root} and the registry (skipped: {detail})"
        )
        raise DatasetBuildError(msg)
    result = discovery.discover(
        records,
        data_paths.root,
        max_file_bytes=config.validation.max_file_bytes,
    )
    kept_ids = result.kept_ids()
    surviving = tuple(record for record in records if record.sample_id in kept_ids)
    return result, surviving, reg


def validate_corpus(
    records: tuple[SourceRecord, ...],
    config: DataConfig,
    paths: DataPaths,
    *,
    inventory: discovery.DiscoveryResult | None = None,
) -> ValidationResult:
    """Decide which discovered files may become training samples.

    Args:
        records: Surviving source records.
        config: Build configuration.
        paths: Resolved data layout.
        inventory: Discovery's measurements, reused so the corpus is not hashed
            twice.

    Returns:
        The admission outcome, rejections included.
    """
    return validate_sources(
        records,
        paths.root,
        config.validation,
        audio_config=config.audio_config(),
        inventory=inventory,
    )


def split_corpus(
    records: tuple[SourceRecord, ...],
    config: DataConfig,
    *,
    seed: int | None = None,
) -> splitting.SplitAssignment:
    """Assign whole groups to splits, on the source set.

    Args:
        records: Accepted source records.
        config: Build configuration.
        seed: Determinism seed. ``None`` uses ``config.random_seed``, which is
            what the configuration documents itself as the split seed and what
            the build id is computed from. Passing a different seed therefore
            changes both the split and the build id.

    Returns:
        The assignment, including the cross-axis holdouts.
    """
    return splitting.assign_splits(
        records, config.split, seed=config.random_seed if seed is None else seed
    )


def segment_corpus(
    records: tuple[SourceRecord, ...],
    config: DataConfig,
    paths: DataPaths,
    assignment: splitting.SplitAssignment,
    *,
    build_id: str,
    cache: PreprocessingCache | None = None,
) -> tuple[preprocess.PreprocessedSource, ...]:
    """Cut accepted sources into windows that inherit their split.

    Args:
        records: Accepted source records.
        config: Build configuration.
        paths: Resolved data layout.
        assignment: Source-to-split assignment.
        build_id: Build stamp for the rows.
        cache: Preprocessing cache. ``None`` uses the one
            :attr:`~voxshield.data.config.DataConfig.cache` describes; pass a
            cache with ``enabled=False`` to force a cold run that also writes
            nothing.

    Returns:
        One outcome per source, accepted or rejected.
    """
    return tuple(
        preprocess.preprocess_sources(
            records,
            config=config,
            split_of=assignment.splits,
            paths=paths,
            cache=cache,
            dataset_build_id=build_id,
        )
    )


def _rows_from(outcomes: tuple[preprocess.PreprocessedSource, ...]) -> tuple[SampleRecord, ...]:
    """Collect manifest rows from segmentation outcomes, in a stable order."""
    return tuple(
        sorted(
            (row for outcome in outcomes for row in outcome.segments),
            key=lambda row: row.sample_id,
        )
    )


def _accepted_sources(
    outcomes: tuple[preprocess.PreprocessedSource, ...],
) -> tuple[SourceRecord, ...]:
    """Source records that produced at least one segment."""
    return tuple(outcome.source for outcome in outcomes if outcome.accepted)


def write_rejected_samples(
    rejections: Sequence[dict[str, Any]],
    paths: DataPaths,
) -> Path:
    """Append every sample that did not survive the build to one JSONL file.

    A rejection is the record of a decision a human may want to revisit, so it
    is written rather than logged. The file is appended to rather than
    overwritten: a rejected sample is a fact about the corpus, and rebuilding
    after a configuration change must not erase what was excluded last time.

    Args:
        rejections: One dict per rejected sample, with at least ``sample_id``,
            ``stage``, ``code``, and ``reason``.
        paths: Resolved data layout.

    Returns:
        The path written.
    """
    target = paths.rejected_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for item in rejections:
            handle.write(json.dumps(jsonable(item), sort_keys=True) + "\n")
    return target


def write_build_reports(
    result: BuildResult,
    config: DataConfig,
    paths: DataPaths,
) -> tuple[Path, ...]:
    """Persist the per-stage reports a build produced.

    Discovery, validation, split, gate, and summary reports are written under
    ``<reports>/``, so a build that was refused still leaves behind the evidence
    for the refusal. A build that fails a gate does not publish manifests, and a
    report that only exists on success cannot explain a failure.

    The build report names the other reports, so it is written last. The caller
    receives the full list and is expected to attach it with
    ``dataclasses.replace``; :class:`BuildResult` is frozen, so this function
    cannot attach the list itself.

    Args:
        result: The build outcome.
        config: Build configuration.
        paths: Resolved data layout.

    Returns:
        The report paths written.
    """
    reports: list[Path] = []
    reports.append(
        _write_json(
            paths.inventory_path(),
            {
                "registry": _registry_report(config),
                "discovery": result.discovery,
            },
        )
    )
    reports.append(
        _write_json(
            paths.validation_path(),
            {
                "stats": result.validation.stats,
                "rejected": list(result.validation.rejected),
                "accepted": [item.sample_id for item in result.validation.accepted],
            },
        )
    )
    reports.append(
        splitting.write_split_report(
            splitting.build_split_report(
                _accepted_sources(result.preprocessing), result.assignment
            ),
            paths.split_report_path(),
        )
    )
    reports.append(_write_json(paths.quality_path(), result.gate_report))
    reports.append(_write_json(paths.leakage_path(), result.leakage))
    reports.append(write_rejected_samples(_rejections(result), paths))

    # The build report lists the evidence the build produced, so it is written
    # last, against a result that already knows the list. Otherwise it serialises
    # its own inventory as empty and a reader has to guess where the other six
    # reports went. Its own path is known in advance, so the list is complete
    # before the file that names it exists.
    reports.append(paths.build_report_path())
    summary = dataclasses.replace(result, reports=tuple(reports))
    _write_json(paths.build_report_path(), summary.to_dict())
    return tuple(reports)


def _registry_report(config: DataConfig) -> dict[str, Any]:
    """Per-corpus provenance, including the ones that were excluded and why."""
    return {
        "datasets": {
            entry.dataset_id: dataclasses.asdict(entry) for entry in config.enabled_datasets()
        },
        "admissible": [dataclasses.asdict(entry) for entry in config.admissible_datasets()],
        "excluded": [
            {"dataset_id": entry.dataset_id, "reason": reason}
            for entry, reason in config.excluded_datasets()
        ],
    }


def _rejections(result: BuildResult) -> list[dict[str, Any]]:
    """Every sample that did not become a row, with the stage that said so.

    Discovery-time unreadable files, validation rejections, and segmentation
    rejections are different findings and are labelled as such, so a reader is
    never left guessing which policy excluded a file.
    """
    items: list[dict[str, Any]] = [
        {
            "sample_id": entry.sample_id,
            "stage": "discovery",
            "code": "unreadable",
            "reason": entry.reason,
        }
        for entry in result.discovery.unreadable
    ]
    items.extend(
        {
            "sample_id": entry.sample_id,
            "stage": "validation",
            "code": entry.code,
            "reason": entry.reason,
        }
        for entry in result.validation.rejected
    )
    items.extend(
        {
            "sample_id": outcome.source.sample_id,
            "stage": "segmentation",
            "code": outcome.rejected_code,
            "reason": outcome.rejected_reason,
        }
        for outcome in result.preprocessing
        if not outcome.accepted
    )
    return sorted(items, key=lambda item: (item["stage"], item["sample_id"]))


def _write_failure_evidence(
    config: DataConfig,
    paths: DataPaths,
    admission: ValidationResult,
    inventory: discovery.DiscoveryResult | None = None,
    assignment: SplitAssignment | None = None,
    preprocessing: Sequence[PreprocessedSource] | None = None,
) -> tuple[Path, ...]:
    """Write what is known about a build that could not finish.

    A refusal that leaves no file behind is indistinguishable from a refusal
    that never ran, so the reports a partial build *can* produce are written
    before the error is raised. The stages that never ran are simply absent
    rather than faked as empty.

    Args:
        config: Build configuration.
        paths: Resolved data layout.
        admission: What validation decided.
        inventory: Discovery outcome, when discovery completed.
        assignment: Split assignment, when splitting completed.
        preprocessing: Segmentation outcomes, when segmentation completed.

    Returns:
        The report paths written.
    """
    reports: list[Path] = []
    if inventory is not None:
        reports.append(
            _write_json(
                paths.inventory_path(),
                {"registry": _registry_report(config), "discovery": inventory},
            )
        )
    reports.append(
        _write_json(
            paths.validation_path(),
            {
                "stats": admission.stats,
                "rejected": list(admission.rejected),
                "accepted": [item.sample_id for item in admission.accepted],
            },
        )
    )
    if assignment is not None:
        reports.append(
            splitting.write_split_report(
                # The split report counts *sources*. An ``Acceptance`` is a decision
                # wrapper around a source, so reporting the wrappers would count
                # rows the splitter never saw.
                splitting.build_split_report(
                    (item.record for item in admission.accepted),
                    assignment,
                ),
                paths.split_report_path(),
            )
        )
    rejections: list[dict[str, Any]] = []
    if inventory is not None:
        rejections.extend(
            {
                "sample_id": entry.sample_id,
                "stage": "discovery",
                "code": "unreadable",
                "reason": entry.reason,
            }
            for entry in inventory.unreadable
        )
    rejections.extend(
        {
            "sample_id": entry.sample_id,
            "stage": "validation",
            "code": entry.code,
            "reason": entry.reason,
        }
        for entry in admission.rejected
    )
    # A segmentation refusal is the case where this matters most: the sources
    # all passed validation, so "why is there no data" is answered only by the
    # per-file segmentation reasons. Losing them leaves the operator with a
    # validation report that says everything was fine.
    if preprocessing is not None:
        rejections.extend(
            {
                "sample_id": outcome.source.sample_id,
                "stage": "segmentation",
                "code": outcome.rejected_code or "no_windows",
                "reason": outcome.rejected_reason or "no contiguous speech window",
            }
            for outcome in preprocessing
            if not outcome.accepted
        )
    reports.append(write_rejected_samples(rejections, paths))
    return tuple(reports)


def build_dataset(
    config: DataConfig,
    *,
    paths: DataPaths | None = None,
    registry: DatasetRegistry | None = None,
    split_seed: int | None = None,
    cache: PreprocessingCache | None = None,
    enforce_gates: bool = True,
    overwrite: bool = True,
) -> BuildResult:
    """Run one full build, in the one order that keeps the claims true.

    Args:
        config: Build configuration.
        paths: Resolved data layout. Taken from the config when omitted.
        registry: A pre-built registry.
        split_seed: Determinism seed for the non-temporal split order. ``None`` uses
            ``config.random_seed``.
        cache: Preprocessing cache. ``None`` builds the cache
            ``config.cache`` describes -- enabled or not, per that setting -- so a
            rebuild of unchanged audio does not re-decode it. Pass
            ``PreprocessingCache(paths, enabled=False)`` to force a cold build,
            which is how a "does the cache change the output" question is
            answered rather than assumed.
        enforce_gates: Refuse to publish manifests when a mandatory gate fails.
            The reports are still written. A build that stops here has not
            published a corpus, so nothing downstream can cite it.
        overwrite: Permit replacing existing manifests.

    Returns:
        The build outcome, with manifests published only if the gates passed.

    Raises:
        DatasetBuildError: No corpus is usable, or segmentation produced nothing.
        DatasetConfigError: The evaluation split is too small to be a
            measurement, when the gates are being enforced.
        DatasetLeakageError: A mandatory gate failed while ``enforce_gates`` is
            set. Every stage report is on disk first, so the failure can be
            diagnosed without re-running the build.
    """
    started = time.perf_counter()
    data_paths = paths if paths is not None else config.data_paths()
    active_cache = cache if cache is not None else PreprocessingCache.for_config(config)
    inventory, surviving, _registry = discover_corpus(config, data_paths, registry=registry)

    admission = validate_corpus(surviving, config, data_paths, inventory=inventory)
    accepted = tuple(item.record for item in admission.accepted)
    if not accepted:
        evidence = _write_failure_evidence(config, data_paths, admission, inventory)
        raise DatasetBuildError(
            "every discovered file was rejected by validation; see the validation "
            f"report under {data_paths.reports} for the per-file reasons "
            f"(written: {', '.join(str(path) for path in evidence)})"
        )

    assignment = split_corpus(accepted, config, seed=split_seed)
    build_id = manifest_io.dataset_build_id(config)
    outcomes = segment_corpus(
        accepted,
        config,
        data_paths,
        assignment,
        build_id=build_id,
        cache=active_cache,
    )
    rows = _rows_from(outcomes)
    if not rows:
        codes = sorted({o.rejected_code for o in outcomes if o.rejected_code})
        evidence = _write_failure_evidence(
            config,
            data_paths,
            admission,
            inventory,
            assignment=assignment,
            preprocessing=outcomes,
        )
        msg = (
            "segmentation produced no windows from any accepted source "
            f"(rejection codes: {', '.join(codes) or 'none'}). The sources passed "
            "validation, so the VAD found no contiguous speech to cut "
            f"(evidence written: {', '.join(str(path) for path in evidence)})"
        )
        raise DatasetBuildError(msg)

    report = evaluate_leakage_gates(
        rows,
        config=config.gates,
        split_config=config.split,
    )
    audio_seconds = sum(float(item.duration_seconds) for item in admission.accepted)
    cache_hits = sum(1 for outcome in outcomes if outcome.from_cache)
    provisional = BuildResult(
        build_id=build_id,
        rows=rows,
        manifests={},
        statistics=manifest_io.compute_statistics(rows, dataset_build_id=build_id),
        discovery=inventory,
        validation=admission,
        assignment=assignment,
        preprocessing=outcomes,
        gate_report=report,
        leakage=report.leakage,
        audio_seconds=audio_seconds,
        cache_hits=cache_hits,
    )

    written: dict[str, str] = {}
    if report.passed or not enforce_gates:
        written = manifest_io.write_manifest_set(
            rows,
            data_paths,
            config=config,
            holdouts=assignment.holdouts,
            datasets=dataset_provenance(config),
            overwrite=overwrite,
        )

    # Reports are written before the gate refusal so a refused build is
    # diagnosable from disk, and after the manifest write so the build report
    # names the manifests it actually published.
    result = dataclasses.replace(
        provisional,
        manifests=written,
        elapsed_seconds=time.perf_counter() - started,
    )
    result = dataclasses.replace(result, reports=write_build_reports(result, config, data_paths))

    if enforce_gates and not report.passed:
        assert_gates_pass(report, config=config.gates)
    return result


@dataclass(frozen=True, slots=True)
class BuildSummary:
    """A build's numbers, without the audio.

    Attributes:
        sources: Source files admitted by validation.
        segments: Manifest rows written.
        segments_by_split: Rows per split.
        labels: Rows per label.
        gates_passed: Whether every mandatory gate passed.
        real_time_factor: Audio seconds per wall second, or ``None``.
    """

    sources: int
    segments: int
    segments_by_split: dict[str, int] = field(default_factory=dict)
    labels: dict[str, int] = field(default_factory=dict)
    gates_passed: bool = True
    real_time_factor: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return jsonable(
            {
                "sources": self.sources,
                "segments": self.segments,
                "segments_by_split": self.segments_by_split,
                "labels": self.labels,
                "gates_passed": self.gates_passed,
                "real_time_factor": self.real_time_factor,
            }
        )


def summarise(result: BuildResult) -> BuildSummary:
    """Reduce a build to the numbers a report needs.

    Args:
        result: A build outcome.

    Returns:
        The summary.
    """
    return BuildSummary(
        sources=len(result.validation.accepted),
        segments=len(result.rows),
        segments_by_split=dict(result.statistics.by_split),
        labels=dict(result.statistics.by_label),
        gates_passed=result.gate_passed,
        real_time_factor=result.real_time_factor,
    )
