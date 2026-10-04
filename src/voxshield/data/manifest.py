"""The versioned manifest: the only thing a training run is allowed to read.

Everything upstream of this module is plumbing. Discovery decides which files
exist, validation decides which are usable, splitting decides where the boundary
goes, and preprocessing decides what the audio becomes. None of that survives a
process restart. What survives is a manifest: a self-describing, ordered,
content-addressed list of analysis segments that states which build produced it,
which configuration produced that build, and which licence each contributing
corpus admitted under.

Three properties make it a provenance record rather than a list of paths.

**It is versioned and refuses to be read by the wrong version.** The first line
is a header carrying a schema version, the configuration hash, and the dataset
build id. A loader written against a different manifest layout fails loudly
instead of silently defaulting every missing field to something plausible, which
is the failure mode that turns a dataset change into an unattributable metric
regression.

**It is append-only.** A source removed from the corpus keeps its row, annotated
with ``removed_reason`` and ``removed_at``. The rule is in
``docs/dataset-manifest.md``: a dataset that can be quietly reweighted after a
disappointing result is not reproducible. Rewriting the file to *drop* a row is
therefore a different operation from writing a new one, and gets its own
function (:func:`retire_samples`) so the intent is explicit at the call site.

**It carries no audio.** ``audio_path`` is relative to the data root and the
repository commits manifests, never audio. A manifest is small enough to review
in a pull request; a corpus is not.

Statistics are written beside the manifest rather than inside it. A manifest
answers "what is in the corpus"; statistics answer "what shape is it", and mixing
them means a statistics fix rewrites a provenance artefact.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voxshield import __version__
from voxshield.data.errors import ManifestError
from voxshield.data.labels import BONA_FIDE
from voxshield.data.schema import UNKNOWN, SampleRecord

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "SPLIT_MANIFESTS",
    "DatasetStatistics",
    "Distribution",
    "Manifest",
    "ManifestEntry",
    "ManifestHeader",
    "compute_statistics",
    "dataset_build_id",
    "evaluation_view_names",
    "read_manifest",
    "retire_samples",
    "write_manifest",
    "write_manifest_set",
    "write_statistics",
]

#: Layout version of the manifest file itself. Bumped when a row changes shape or
#: a header field is removed, and deliberately *not* bumped for additive header
#: fields, so that a reader can tolerate a newer build that only added
#: provenance.
#:
#: Version 1 is the layout this module defines.
MANIFEST_SCHEMA_VERSION: int = 1

#: Manifest file names by split. ``"all"`` is the whole corpus in one file, which
#: is what a dataset loader reads when it has not been told to filter.
SPLIT_MANIFESTS: Mapping[str, str] = {
    "all": "all.jsonl",
    "train": "train.jsonl",
    "dev": "dev.jsonl",
    "test": "test.jsonl",
}

#: Record type tags. A manifest is a stream of JSON objects with no schema
#: enforcement, so each line states what it is rather than relying on position.
_HEADER_TYPE = "manifest_header"
_ENTRY_TYPE = "sample"

#: Manifests written per evaluation axis, beyond the whole test split. Each is a
#: subset of ``test``, and they overlap on purpose: a cross-language segment that
#: is also in-domain is not a contradiction, it is just the row answering two
#: questions. The axis names match :class:`~voxshield.data.config.SplitConfig`'s
#: holdout fields so a claim in a report can be traced to the configuration that
#: authorised it.
_EVALUATION_VIEWS: Mapping[str, str] = {
    "generator": "test_cross_generator",
    "language": "test_cross_language",
    "codec": "test_cross_codec",
}

#: The record attribute each evaluation axis reads.
_AXIS_ATTRIBUTE: Mapping[str, str] = {
    "generator": "generator_id",
    "language": "language",
    "codec": "codec",
}


def _now() -> str:
    """Current UTC time, ISO-8601, second resolution."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def dataset_build_id(config: Any) -> str:
    """The build id stamped on every row of a build.

    Derived from :meth:`~voxshield.data.config.DataConfig.content_fingerprint`,
    which covers the settings that can change a sample or a split and excludes
    the ones that only change where output lands. Two builds of the same corpus
    on two machines therefore share an id, which is what lets a result cite the
    dataset it was produced from without also pinning a filesystem layout.

    Args:
        config: A :class:`~voxshield.data.config.DataConfig`.

    Returns:
        A short, human-quotable identifier, e.g. ``"vs-1f0c..."``.
    """
    return f"vs-{config.content_fingerprint()[:16]}"


@dataclass(frozen=True, slots=True)
class ManifestHeader:
    """Line 1 of a manifest: what produced it, and under what authority.

    Attributes:
        schema_version: :data:`MANIFEST_SCHEMA_VERSION` of the row layout.
        dataset_build_id: The build these rows belong to.
        config_hash: Full configuration hash, including paths. Cites *this run*.
        content_fingerprint: Configuration hash of only the corpus-affecting
            settings. Equal across machines.
        created_at: When the manifest was written, ISO-8601 UTC.
        voxshield_version: The package version that wrote it.
        split: The split this file holds, or ``"all"``. Lets a reader reject a
            file whose rows disagree with its own name.
        n_rows: Rows on disk, including retired ones.
        n_active: Rows that are not retired.
        datasets: Per-corpus provenance: id, name, version, licence, licence
            status, and the train/feature/redistribution permissions each
            corpus admitted under. Recorded here rather than per row because it
            is a property of the corpus, and repeating it on 400k rows would
            make the file unreadable for no additional information.
    """

    dataset_build_id: str
    config_hash: str
    content_fingerprint: str
    created_at: str
    split: str
    n_rows: int
    n_active: int
    voxshield_version: str = __version__
    schema_version: int = MANIFEST_SCHEMA_VERSION
    record_type: str = _HEADER_TYPE
    datasets: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "dataset_build_id": self.dataset_build_id,
            "config_hash": self.config_hash,
            "content_fingerprint": self.content_fingerprint,
            "created_at": self.created_at,
            "voxshield_version": self.voxshield_version,
            "split": self.split,
            "n_rows": self.n_rows,
            "n_active": self.n_active,
            "datasets": {key: self.datasets[key] for key in sorted(self.datasets)},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ManifestHeader:
        try:
            return cls(
                schema_version=int(payload["schema_version"]),
                record_type=str(payload.get("record_type", _HEADER_TYPE)),
                dataset_build_id=str(payload["dataset_id"])
                if "dataset_id" in payload
                else str(payload["dataset_build_id"]),
                config_hash=str(payload["config_hash"]),
                content_fingerprint=str(payload["content_fingerprint"]),
                created_at=str(payload["created_at"]),
                split=str(payload["split"]),
                n_rows=int(payload["n_rows"]),
                n_active=int(payload["n_active"]),
                voxshield_version=str(payload.get("voxshield_version", "unknown")),
                datasets=dict(payload.get("datasets") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            msg = f"manifest header is malformed: {exc}"
            raise ManifestError(msg) from exc


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One row: a sample, plus the manifest's own bookkeeping about it.

    The removal fields are here rather than on :class:`SampleRecord` because they
    are statements *about the corpus*, not about the audio. A retired segment is
    still a real segment with a real hash; what changed is whether this dataset
    includes it, and that is the manifest's claim to make.
    """

    record: SampleRecord
    removed_reason: str = ""
    removed_at: str = ""

    @property
    def removed(self) -> bool:
        return bool(self.removed_reason)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"record_type": _ENTRY_TYPE}
        payload.update(self.record.to_dict())
        if self.removed:
            payload["removed_reason"] = self.removed_reason
            payload["removed_at"] = self.removed_at
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ManifestEntry:
        return cls(
            record=SampleRecord.from_dict(payload),
            removed_reason=str(payload.get("removed_reason") or ""),
            removed_at=str(payload.get("removed_at") or ""),
        )


@dataclass(frozen=True, slots=True)
class Distribution:
    """Summary statistics for one numeric field, over a set of rows.

    Reported as a shape rather than a total because the failure this guards
    against is a corpus that is technically large and practically useless: a
    ``speech_seconds`` total of 900 across 400 segments says nothing, while a
    mean of 2.2s with a maximum of 4.0s says the segments are short on purpose.
    """

    count: int
    total: float
    minimum: float
    maximum: float
    mean: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "total": round(self.total, 4),
            "min": round(self.minimum, 6),
            "max": round(self.maximum, 6),
            "mean": round(self.mean, 6),
        }

    @classmethod
    def from_values(cls, values: Sequence[float]) -> Distribution:
        if not values:
            return cls(count=0, total=0.0, minimum=0.0, maximum=0.0, mean=0.0)
        total = float(sum(values))
        return cls(
            count=len(values),
            total=total,
            minimum=float(min(values)),
            maximum=float(max(values)),
            mean=total / len(values),
        )


def _counter(mapping: Mapping[str, int]) -> dict[str, int]:
    """Sort a counter's keys, so two runs produce byte-identical statistics."""
    return {key: mapping[key] for key in sorted(mapping)}


def _tally(rows: Iterable[SampleRecord], attribute: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = getattr(row, attribute, UNKNOWN) or UNKNOWN
        out[value] = out.get(value, 0) + 1
    return _counter(out)


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    """What shape is this corpus, in enough detail to catch a bad build.

    Counts per split, label, corpus, attack family, generator, language, and
    codec; the duration and coverage distributions; padding; and the duplicate
    picture. The two duplicate counters are deliberately different measurements:
    ``duplicate_content_hashes_within_split`` is a corpus-hygiene number, while
    :attr:`cross_split_content_duplicates` is a leak, and conflating them is how
    a genuine duplicate gets dismissed as noise.
    """

    dataset_build_id: str
    n_rows: int
    n_active: int
    n_retired: int
    n_sources: int
    n_speakers: int
    by_split: Mapping[str, int]
    by_label: Mapping[str, int]
    by_dataset: Mapping[str, int]
    by_attack_family: Mapping[str, int]
    by_generator: Mapping[str, int]
    by_language: Mapping[str, int]
    by_codec: Mapping[str, int]
    by_channel: Mapping[str, int]
    by_source_split: Mapping[str, int]
    duration: Distribution
    speech: Distribution
    coverage: Distribution
    n_padded: int
    n_distinct_content_hashes: int
    duplicate_content_hashes_within_split: int
    cross_split_content_duplicates: int

    @property
    def class_ratio(self) -> float:
        """Fraction of active rows that are bona fide.

        Reported because a 50/50 benchmark says nothing about deployment: real
        traffic is overwhelmingly genuine, and a corpus that drifted to 50/50
        will produce a ``FAR`` that is optimistic by construction.
        """
        total = self.n_active
        if not total:
            return 0.0
        return self.by_label.get(BONA_FIDE, 0) / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_build_id": self.dataset_build_id,
            "n_rows": self.n_rows,
            "n_active": self.n_active,
            "n_retired": self.n_retired,
            "n_sources": self.n_sources,
            "n_speakers": self.n_speakers,
            "class_ratio": round(self.class_ratio, 6),
            "by_split": dict(self.by_split),
            "by_label": dict(self.by_label),
            "by_dataset": dict(self.by_dataset),
            "by_attack_family": dict(self.by_attack_family),
            "by_generator": dict(self.by_generator),
            "by_language": dict(self.by_language),
            "by_codec": dict(self.by_codec),
            "by_channel": dict(self.by_channel),
            "by_source_split": dict(self.by_source_split),
            "duration": self.duration.to_dict(),
            "speech": self.speech.to_dict(),
            "coverage": self.coverage.to_dict(),
            "n_padded": self.n_padded,
            "n_distinct_content_hashes": self.n_distinct_content_hashes,
            "duplicate_content_hashes_within_split": self.duplicate_content_hashes_within_split,
            "cross_split_content_duplicates": self.cross_split_content_duplicates,
        }


def _duplicates(entries: Sequence[ManifestEntry]) -> tuple[int, int]:
    """Count within-split duplicate content and cross-split duplicate content.

    The two are different findings and are never summed. A digest that repeats
    inside one split is corpus hygiene -- the same audio filed twice, which
    silently over-weights it in training. A digest that appears in two splits is
    a leak, and is the number that must block a build.

    Args:
        entries: Manifest rows.

    Returns:
        ``(within_split, cross_split)``, each counting the occurrences beyond
        the first for its digest. A digest with no recorded content hash is
        excluded entirely: an absent measurement is not evidence of a
        duplicate, and counting it as one would block builds over missing
        provenance rather than over a real conflict.
    """
    occurrences: dict[str, list[str]] = {}
    for entry in entries:
        if entry.removed:
            continue
        digest = entry.record.content_hash
        if not digest or digest == UNKNOWN:
            continue
        occurrences.setdefault(digest, []).append(entry.record.split)

    within = 0
    cross = 0
    for splits in occurrences.values():
        if len(splits) < 2:
            continue
        distinct = set(splits)
        if len(distinct) > 1:
            cross += len(splits) - 1
        else:
            within += len(splits) - 1
    return within, cross


def compute_statistics(
    entries: Sequence[ManifestEntry | SampleRecord],
    *,
    dataset_build_id: str = "",
) -> DatasetStatistics:
    """Summarise a manifest's rows.

    Retired rows are excluded from every count except :attr:`n_retired`, because
    they are provenance, not corpus.

    Args:
        entries: Manifest rows, or bare sample records.
        dataset_build_id: Build id for the report. Inferred from the rows when
            omitted.

    Returns:
        A :class:`DatasetStatistics`.
    """
    rows = [
        entry if isinstance(entry, ManifestEntry) else ManifestEntry(entry) for entry in entries
    ]
    active = [entry.record for entry in rows if not entry.removed]

    build_id = dataset_build_id or next(
        (row.dataset_build_id for row in active if row.dataset_build_id),
        "",
    )
    speakers = {row.speaker_id for row in active if row.speaker_id and row.speaker_id != UNKNOWN}
    parents = {row.parent_id for row in active if row.parent_id}
    digests = {
        row.content_hash for row in active if row.content_hash and row.content_hash != UNKNOWN
    }
    within, cross = _duplicates(rows)

    return DatasetStatistics(
        dataset_build_id=build_id,
        n_rows=len(rows),
        n_active=len(active),
        n_retired=len(rows) - len(active),
        n_sources=len(parents),
        n_speakers=len(speakers),
        by_split=_tally(active, "split"),
        by_label=_tally(active, "label"),
        by_dataset=_tally(active, "dataset_id"),
        by_attack_family=_tally(active, "attack_family"),
        by_generator=_tally(active, "generator_id"),
        by_language=_tally(active, "language"),
        by_codec=_tally(active, "codec"),
        by_channel=_tally(active, "channel"),
        by_source_split=_tally(active, "source_split"),
        duration=Distribution.from_values([row.duration_seconds for row in active]),
        speech=Distribution.from_values([row.speech_seconds for row in active]),
        coverage=Distribution.from_values([row.coverage for row in active]),
        n_padded=sum(1 for row in active if row.is_padded),
        n_distinct_content_hashes=len(digests),
        duplicate_content_hashes_within_split=within,
        cross_split_content_duplicates=cross,
    )


def evaluation_view_names(
    record: SampleRecord,
    holdouts: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Which cross-axis test manifests a test row also belongs to.

    Only ``test`` rows qualify, and a row is in a cross-axis view when its value
    on that axis is inside the configured holdout. An unknown value never
    qualifies: claiming a cross-generator test on a row whose generator was
    never published produces a set that is not cross-generator, which is the
    exact claim the axis exists to support.

    Args:
        record: One manifest row.
        holdouts: Axis to allowed values, from
            :attr:`~voxshield.data.splitting.SplitAssignment.holdouts`.

    Returns:
        View names in a stable order, excluding ``"test_indomain"``.
    """
    if record.split != "test":
        return ()
    names: list[str] = []
    for axis, attribute in _AXIS_ATTRIBUTE.items():
        allowed = holdouts.get(axis) or ()
        if not allowed:
            continue
        value = getattr(record, attribute, UNKNOWN)
        if value and value != UNKNOWN and value in allowed:
            names.append(_EVALUATION_VIEWS[axis])
    return tuple(names)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> Path:
    """Write ``text`` to ``path`` atomically.

    A manifest is written in one shot, so a process killed mid-write would leave
    a file that is valid JSONL for most of its lines and truncated at the end --
    readable, and quietly short a training run. Writing to a sibling temporary
    file and renaming means a reader sees either the previous manifest or the
    complete new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def _existing_build_id(path: Path) -> str | None:
    """The build id of an existing manifest, or ``None`` if absent/unreadable."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as stream:
            first = stream.readline()
    except OSError:
        return None
    if not first.strip():
        return None
    try:
        payload = json.loads(first)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("record_type") != _HEADER_TYPE:
        return None
    build = payload.get("dataset_build_id")
    return str(build) if build else None


def _provenance(datasets: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalise dataset provenance for the header."""
    if not datasets:
        return {}
    out: dict[str, Any] = {}
    for key in sorted(datasets):
        value = datasets[key]
        out[key] = dict(value) if isinstance(value, Mapping) else {"value": value}
    return out


def _normalise(
    entries: Sequence[ManifestEntry | SampleRecord],
) -> tuple[ManifestEntry, ...]:
    """Accept bare records or entries, and return entries in a stable order.

    Sorted by ``sample_id`` so that two builds of the same corpus produce
    byte-identical manifests, which is what makes the file reviewable in a diff
    and makes "did anything change?" answerable by hash.
    """
    rows = [
        entry if isinstance(entry, ManifestEntry) else ManifestEntry(entry) for entry in entries
    ]
    rows.sort(key=lambda entry: entry.record.sample_id)
    return tuple(rows)


def _validate(
    rows: Sequence[ManifestEntry],
    *,
    split: str,
    require_active: bool,
) -> None:
    """Refuse to write a manifest that would misdescribe the corpus."""
    if not rows:
        msg = f"refusing to write an empty {split} manifest"
        raise ManifestError(msg)

    if require_active and all(entry.removed for entry in rows):
        msg = (
            f"every row in the {split} manifest is retired; writing it would "
            "publish a dataset with no samples"
        )
        raise ManifestError(msg)

    seen: set[str] = set()
    duplicates: list[str] = []
    for entry in rows:
        sample_id = entry.record.sample_id
        if sample_id in seen:
            duplicates.append(sample_id)
        seen.add(sample_id)
    if duplicates:
        msg = f"duplicate sample_id(s) in the {split} manifest: " + ", ".join(
            sorted(set(duplicates))[:5]
        )
        raise ManifestError(msg)

    stray = sorted({entry.record.split for entry in rows} - {split})
    if split != "all" and stray:
        msg = (
            f"the {split} manifest contains rows assigned to {stray}; a "
            "split-specific file whose rows disagree with its name is worse than "
            "no file, because a loader filtering on the filename will disagree "
            "with the loader filtering on the row"
        )
        raise ManifestError(msg)

    builds = {entry.record.dataset_build_id for entry in rows if entry.record.dataset_build_id}
    if len(builds) > 1:
        msg = f"the {split} manifest mixes dataset builds {sorted(builds)}"
        raise ManifestError(msg)

    _within, cross = _duplicates(rows)
    if cross:
        msg = (
            f"{cross} content hash(es) appear on both sides of a split boundary "
            f"in the {split} manifest; the same audio on both sides of a "
            "train/test boundary is leakage, not redundancy"
        )
        raise ManifestError(msg)


def _serialise(header: ManifestHeader, rows: Sequence[ManifestEntry]) -> str:
    """Render a manifest to JSONL text.

    Compact separators and sorted keys, so two builds of the same corpus
    produce byte-identical files and a manifest is reviewable in a diff. One
    line per record, because a truncated final line is the only corruption
    worth designing for and a line-oriented format makes it detectable.
    """
    lines = [json.dumps(header.to_dict(), sort_keys=True, separators=(",", ":"))]
    lines.extend(
        json.dumps(entry.to_dict(), sort_keys=True, separators=(",", ":")) for entry in rows
    )
    return "\n".join(lines) + "\n"


def write_manifest(
    entries: Sequence[ManifestEntry | SampleRecord],
    path: str | Path,
    *,
    split: str = "all",
    dataset_build_id: str = "",
    config_hash: str = "",
    content_fingerprint: str = "",
    datasets: Mapping[str, Any] | None = None,
    created_at: str | None = None,
    overwrite: bool = False,
    require_active: bool = True,
) -> Path:
    """Write a versioned JSONL manifest.

    Args:
        entries: Rows to write. Sorted here, so caller order does not affect
            the output.
        path: Destination file.
        split: The split this file holds. Must match the rows unless ``"all"``.
        dataset_build_id: Build stamp. Inferred from the rows when omitted.
        config_hash: Full configuration hash for the header.
        content_fingerprint: Corpus-affecting configuration hash for the header.
        datasets: Per-corpus provenance for the header.
        created_at: Header timestamp. Defaults to now; pass a value to make the
            output byte-reproducible.
        overwrite: Permit replacing an existing manifest. Replacing a manifest
            of the *same* build is always allowed, so an interrupted build can
            be re-run without a flag.
        require_active: Refuse a manifest whose every row is retired.

    Returns:
        The path written.

    Raises:
        ManifestError: Empty, self-contradictory, cross-split-duplicated, or
            cross-split-contaminated rows; or an existing manifest that
            ``overwrite`` does not permit replacing.
    """
    target = Path(path)
    rows = _normalise(entries)
    _validate(rows, split=split, require_active=require_active)

    existing = _existing_build_id(target)
    if existing is not None and not overwrite:
        new_build = dataset_build_id or next(
            (entry.record.dataset_build_id for entry in rows if entry.record.dataset_build_id),
            "",
        )
        if existing != new_build:
            msg = (
                f"{target} already holds dataset build {existing!r} and this "
                f"build is {new_build!r}; pass overwrite=True to replace it. "
                "Two datasets sharing a path is how a result ends up citing the "
                "wrong corpus."
            )
            raise ManifestError(msg)

    active = [entry for entry in rows if not entry.removed]
    header = ManifestHeader(
        dataset_build_id=dataset_build_id or (rows[0].record.dataset_build_id if rows else ""),
        config_hash=config_hash,
        content_fingerprint=content_fingerprint,
        created_at=created_at or _now(),
        split=split,
        n_rows=len(rows),
        n_active=len(active),
        datasets=_provenance(datasets),
    )

    return _atomic_write(target, _serialise(header, rows))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Manifest:
    """A parsed manifest: its header and its rows.

    Attributes:
        header: Line 1.
        entries: Every row, including retired ones, in file order.
        path: Where it was read from, or ``None`` for a manifest built in memory.
    """

    header: ManifestHeader
    entries: tuple[ManifestEntry, ...] = ()
    path: Path | None = None

    @property
    def samples(self) -> tuple[SampleRecord, ...]:
        """Active rows only, which is what a loader wants."""
        return tuple(entry.record for entry in self.entries if not entry.removed)

    @property
    def retired(self) -> tuple[ManifestEntry, ...]:
        return tuple(entry for entry in self.entries if entry.removed)

    def by_split(self, split: str) -> tuple[SampleRecord, ...]:
        return tuple(row for row in self.samples if row.split == split)

    def for_split(self, split: str) -> Manifest:
        """A view restricted to one split, keeping the header.

        The view *excludes* rows rather than retiring them, so the header's row
        count becomes the number actually present.
        """
        selected = self.by_split(split)
        return Manifest(
            header=_recount(self.header, n_rows=len(selected), n_active=len(selected), split=split),
            entries=tuple(entry for entry in self.entries if entry.record.split == split),
            path=self.path,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[SampleRecord]:
        return iter(self.samples)


def _recount(
    header: ManifestHeader,
    *,
    n_rows: int,
    n_active: int,
    split: str | None = None,
) -> ManifestHeader:
    """Copy a header with corrected row counts, and optionally a new split.

    The two counts are separate because they answer different questions.
    ``n_rows`` is what is on disk; ``n_active`` is what the dataset contains.
    Retirement lowers the second and leaves the first alone -- that difference
    *is* the append-only record, and collapsing the pair would make a retired
    row indistinguishable from a row that was never written.
    """
    updated = replace(header, n_rows=n_rows, n_active=n_active)
    if split is None or split == updated.split:
        return updated
    return replace(updated, split=split)


def _parse_line(raw: str, *, path: Path, lineno: int) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        msg = f"{path}:{lineno}: manifest line is not valid JSON: {exc}"
        raise ManifestError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"{path}:{lineno}: manifest line is not a JSON object"
        raise ManifestError(msg)
    return payload


def read_manifest(path: str | Path) -> Manifest:
    """Read a manifest, refusing anything this build cannot interpret.

    Args:
        path: The manifest file.

    Returns:
        A :class:`Manifest`.

    Raises:
        ManifestError: Missing file, missing or foreign header, an unsupported
            schema version, or a malformed row. Every failure names the file and
            line, because the alternative is a training run that trained on
            whatever the parser defaulted a malformed field to.
    """
    target = Path(path)
    if not target.is_file():
        msg = f"no manifest at {target}"
        raise ManifestError(msg)

    header: ManifestHeader | None = None
    entries: list[ManifestEntry] = []
    with target.open("r", encoding="utf-8") as stream:
        for lineno, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            payload = _parse_line(raw, path=target, lineno=lineno)

            if header is None:
                if payload.get("record_type") != _HEADER_TYPE:
                    msg = (
                        f"{target}:{lineno}: the first line must be a "
                        f"{_HEADER_TYPE!r} record; found "
                        f"{payload.get('record_type')!r}. This file was not "
                        "written by this pipeline, or its header was lost."
                    )
                    raise ManifestError(msg)
                header = ManifestHeader.from_dict(payload)
                if header.schema_version > MANIFEST_SCHEMA_VERSION:
                    msg = (
                        f"{target} is manifest schema version {header.schema_version}, "
                        f"but this build understands at most {MANIFEST_SCHEMA_VERSION}; "
                        "upgrade VoxShield or rebuild the dataset"
                    )
                    raise ManifestError(msg)
                continue

            if payload.get("record_type") != _ENTRY_TYPE:
                msg = (
                    f"{target}:{lineno}: expected a {_ENTRY_TYPE!r} record, got "
                    f"{payload.get('record_type')!r}"
                )
                raise ManifestError(msg)
            try:
                entries.append(ManifestEntry.from_dict(payload))
            except (KeyError, TypeError, ValueError) as exc:
                msg = f"{target}:{lineno}: malformed manifest row: {exc}"
                raise ManifestError(msg) from exc

    if header is None:
        msg = f"{target} has no {_HEADER_TYPE!r} first line; it is not a manifest"
        raise ManifestError(msg)
    if header.n_active != sum(1 for entry in entries if not entry.removed):
        msg = (
            f"{target} header claims {header.n_active} active row(s) but the file "
            f"holds {sum(1 for entry in entries if not entry.removed)}; the file was "
            "truncated or edited without updating the header"
        )
        raise ManifestError(msg)

    return Manifest(header=header, entries=tuple(entries), path=target)


def retire_samples(
    path: str | Path,
    sample_ids: Sequence[str],
    reason: str,
    *,
    removed_at: str | None = None,
) -> int:
    """Annotate rows as removed, keeping them on disk.

    The append-only rule from ``docs/dataset-manifest.md``: a removed source
    keeps its row, because a dataset that can be quietly reweighted after a
    disappointing result is not reproducible. This rewrites the file with the
    rows still present and their ``removed_reason`` and ``removed_at`` filled in,
    and updates the header's active count.

    Args:
        path: The manifest to rewrite.
        sample_ids: Rows to retire. Ids that are not present are ignored.
        reason: Why the rows were removed.
        removed_at: Removal timestamp. Defaults to now.

    Returns:
        The number of rows newly retired.

    Raises:
        ManifestError: No ``reason``, or the manifest cannot be read.
    """
    if not reason.strip():
        msg = "retiring a row without a reason defeats the purpose of the append-only rule"
        raise ManifestError(msg)

    manifest = read_manifest(path)
    targets = set(sample_ids)
    stamp = removed_at or _now()

    updated: list[ManifestEntry] = []
    changed = 0
    for entry in manifest.entries:
        if entry.record.sample_id in targets and not entry.removed:
            updated.append(
                ManifestEntry(
                    record=entry.record,
                    removed_reason=reason,
                    removed_at=stamp,
                )
            )
            changed += 1
        else:
            updated.append(entry)

    if not changed:
        return 0

    rows = _normalise(updated)
    _validate(rows, split=manifest.header.split, require_active=False)
    # n_rows stays at the full on-disk row count; only n_active falls, because the
    # retired rows are still on disk and that is the point of the append-only rule.
    header = _recount(
        manifest.header,
        n_rows=len(rows),
        n_active=sum(1 for entry in rows if not entry.removed),
    )
    _atomic_write(Path(path), _serialise(header, rows))
    return changed


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def write_statistics(
    statistics: DatasetStatistics,
    path: str | Path,
    *,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write dataset statistics as JSON.

    Args:
        statistics: The computed summary.
        path: Destination file.
        extra: Additional top-level keys, such as the gate report or the
            manifests' relative paths.

    Returns:
        The path written.
    """
    payload: dict[str, Any] = statistics.to_dict()
    if extra:
        payload.update({key: extra[key] for key in sorted(extra)})
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    return _atomic_write(Path(path), text)


def write_manifest_set(
    entries: Sequence[ManifestEntry | SampleRecord],
    paths: Any,
    *,
    config: Any | None = None,
    holdouts: Mapping[str, Sequence[str]] | None = None,
    datasets: Mapping[str, Any] | None = None,
    created_at: str | None = None,
    overwrite: bool = True,
) -> dict[str, str]:
    """Write every manifest for one build, plus its statistics.

    Writes ``all.jsonl``, the per-split manifests, and one manifest per
    cross-axis evaluation view, then ``dataset_statistics.json`` covering the
    whole corpus. Evaluation views that select nothing are skipped rather than
    written empty: an empty cross-generator set is a configuration that produced
    no such measurement, and publishing a zero-row file for it invites a
    trainer to report a perfect score on nothing.

    Args:
        entries: All rows of the build.
        paths: A :class:`~voxshield.data.paths.DataPaths`.
        config: Build configuration, for the header hashes and build id.
        holdouts: Axis holdouts, for the evaluation views.
        datasets: Per-corpus provenance.
        created_at: Header timestamp, shared across files.
        overwrite: Permit replacing existing manifests.

    Returns:
        Relative path to written file, keyed by manifest name. ``"all"``,
        ``"train"``, ``"dev"``, ``"test"``, any evaluation view, and
        ``"statistics"``.

    Raises:
        ManifestError: A per-split manifest fails validation, which means the
            split assignment and the rows disagree.
    """
    rows = _normalise(entries)
    stamp = created_at or _now()
    build_id = dataset_build_id(config) if config is not None else ""
    config_hash = config.config_hash() if config is not None else ""
    fingerprint = config.content_fingerprint() if config is not None else ""
    axis_holdouts = dict(holdouts or {})

    if build_id:
        # The header and the rows must agree on which build they belong to. A
        # header citing a build the rows do not carry is a provenance claim that
        # resolves to nothing, and it is exactly what a result cites.
        rows = tuple(
            ManifestEntry(
                record=entry.record.with_build(build_id),
                removed_reason=entry.removed_reason,
                removed_at=entry.removed_at,
            )
            for entry in rows
        )

    written: dict[str, str] = {}
    for name, filename in SPLIT_MANIFESTS.items():
        selected = tuple(entry for entry in rows if name == "all" or entry.record.split == name)
        target = paths.manifests / filename
        write_manifest(
            selected,
            target,
            split=name,
            dataset_build_id=build_id,
            config_hash=config_hash,
            content_fingerprint=fingerprint,
            datasets=datasets,
            created_at=stamp,
            overwrite=overwrite,
        )
        written[name] = str(target)

    for view in _EVALUATION_VIEWS.values():
        selected = tuple(
            entry for entry in rows if view in evaluation_view_names(entry.record, axis_holdouts)
        )
        if not selected:
            continue
        target = paths.manifests / f"{view}.jsonl"
        write_manifest(
            selected,
            target,
            split="test",
            dataset_build_id=build_id,
            config_hash=config_hash,
            content_fingerprint=fingerprint,
            datasets=datasets,
            created_at=stamp,
            overwrite=overwrite,
        )
        written[view] = str(target)

    statistics = compute_statistics(rows, dataset_build_id=build_id)
    stats_path = paths.statistics_path()
    write_statistics(
        statistics,
        stats_path,
        extra={
            "manifests": {name: SPLIT_MANIFESTS[name] for name in ("all", "train", "dev", "test")},
            "evaluation_views": sorted(name for name in written if name not in SPLIT_MANIFESTS),
        },
    )
    written["statistics"] = str(stats_path)
    return written
