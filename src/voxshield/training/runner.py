"""End-to-end orchestration for one Phase 3 baseline.

The runner is where the protocol is enforced, rather than merely described:

1. Resolve train and dev rows from the manifest.
2. Refuse to proceed if a leakage gate fails.
3. Refuse to proceed if a split is underpowered for the claim it would support.
4. Fit on train, selecting on dev.
5. Choose the operating threshold on **dev only**.
6. Score **test once**, at that unchanged threshold.
7. Write an artefact and a report that agree with each other.

Step 6 is the one that gets violated by accident, usually by a "quick look at
the test curve" during development. Nothing here can prevent a person from
opening the test split in a notebook, but the runner never does it: it calls
:meth:`BaselineModel.predict_proba` on test exactly once, and
``--dry-run`` stops before even that.

The three outcomes are distinct and are not interchangeable:

* **measured** - every step above ran on real audio.
* **not run** - the code is correct and nothing was executed. This is the current
  state of the project: there is no corpus.
* **failed** - a step ran and produced nothing usable.

Collapsing ``not run`` into ``failed`` would be a lie in the other direction: it
would claim a broken pipeline rather than an absent dataset.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from voxshield.evaluation.metrics import NOT_AVAILABLE, ThresholdPolicy, select_threshold
from voxshield.evaluation.report import (
    NOT_RUN,
    EvaluationReport,
    LatencyReport,
    build_report,
    not_run_report,
)
from voxshield.training.artifacts import (
    ArtifactMetadata,
    build_metadata,
    load_artifact,
    save_artifact,
)
from voxshield.training.calibration import Calibrator, calibrator_from_dict, fit_calibrator
from voxshield.training.config import TrainingConfig, TrainingError
from voxshield.training.datasets import (
    FeatureMatrix,
    load_feature_matrix,
    require_clean_gates,
    resolve_rows,
)
from voxshield.training.features import FeatureExtractor
from voxshield.training.models import BaselineModel, FitSummary, build_baseline

__all__ = [
    "RunOutcome",
    "RunResult",
    "run_baseline",
    "run_from_artifact",
]

#: Latency is only timed when asked for. It roughly triples the cost of a run,
#: so it is opt-in rather than always-on.
TIME_INFERENCE = "time_inference"

#: Written next to the artefact so a results directory is self-describing.
MANIFEST_STEM = "run.json"


class RunOutcome(StrEnum):
    """What happened, kept distinct on purpose.

    Attributes:
        MEASURED: Every step ran on real audio.
        NOT_RUN: Nothing was executed, because inputs were absent or the run was
            a dry run.
        FAILED: Execution began and could not finish.
    """

    MEASURED = "measured"
    NOT_RUN = "not_run"
    FAILED = "failed"


@dataclass(slots=True)
class RunResult:
    """Everything one run produced.

    Attributes:
        outcome: Which of the three states applies.
        reason: Why, when the outcome is not ``measured``. Empty for a measured
            run, which needs no excuse.
        config: The configuration used, or ``None`` when the run stopped before
            one was applied.
        model: The fitted model, or ``None``.
        summary: What the fit did, or ``None``.
        metadata: Artefact provenance, or ``None``.
        dev_report: Dev report. Written even though dev is not the headline,
            because the threshold that test inherits came from here and a reader
            checking that choice needs the numbers it was made from.
        test_report: Test report, or a ``NOT RUN`` report.
        threshold: The operating point chosen on dev and applied to test.
        calibrator: The probability calibrator fitted on dev, or ``None`` when
            the run stopped before one was fitted.
        artifact_path: Artefact directory, or ``None``.
        duration_seconds: Wall-clock seconds for the run.
    """

    outcome: RunOutcome
    reason: str = ""
    config: TrainingConfig | None = None
    model: BaselineModel | None = None
    summary: FitSummary | None = None
    metadata: ArtifactMetadata | None = None
    dev_report: EvaluationReport | None = None
    test_report: EvaluationReport | None = None
    threshold: float | None = None
    calibrator: Calibrator | None = None
    artifact_path: Path | None = None
    duration_seconds: float = 0.0

    @property
    def measured(self) -> bool:
        """Whether a test split was actually scored.

        Returns:
            ``True`` only for a fully measured run.
        """
        return self.outcome is RunOutcome.MEASURED

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping.

        Returns:
            A mapping safe for :func:`json.dump`, with nested reports flattened.
        """
        return {
            "outcome": str(self.outcome),
            "reason": self.reason,
            "model_id": self.config.model_id if self.config else None,
            "config_hash": self.config.config_hash() if self.config else None,
            "threshold": self.threshold,
            "calibration": self.calibrator.to_dict() if self.calibrator else None,
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "duration_seconds": round(self.duration_seconds, 3),
            "fit": self.summary.to_dict() if self.summary else None,
            "metadata": self.metadata.to_dict() if self.metadata else None,
            "dev_report": self.dev_report.to_dict() if self.dev_report else None,
            "test_report": self.test_report.to_dict() if self.test_report else None,
        }

    def write(self, output_dir: Path | str) -> Path:
        """Write the run record to disk.

        Args:
            output_dir: Directory for the record. Created if absent.

        Returns:
            The path written.
        """
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST_STEM
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return path


def _dataset_build_id(manifest: Any) -> str | None:
    """Recover the dataset build identifier from a manifest.

    Every report says which build produced it, so that two numbers from
    different builds are never placed in the same table without the difference
    being visible.

    Args:
        manifest: A loaded manifest or a path to one.

    Returns:
        The build identifier, or ``None`` when it cannot be recovered. Returning
        ``None`` is honest; inventing a placeholder would be worse.
    """
    header = getattr(manifest, "header", None)
    if header is not None:
        return str(getattr(header, "dataset_build_id", "") or "") or None
    if isinstance(manifest, (str, Path)):
        from voxshield.data.manifest import read_manifest

        try:
            loaded = read_manifest(Path(manifest))
        except Exception:  # provenance is best-effort here
            return None
        return str(getattr(loaded.header, "dataset_build_id", "") or "") or None
    return None


def _require_rows(matrix: FeatureMatrix, minimum: int, *, role: str, model_id: str) -> None:
    """Refuse an underpowered split.

    Args:
        matrix: The materialised split.
        minimum: Floor on rows.
        role: ``"train"`` or ``"dev"``, for the message.
        model_id: Model identifier, for the message.

    Raises:
        TrainingError: If the split holds fewer than ``minimum`` rows. A run on
            too little data produces a number, and that number is the kind that
            gets quoted later without its sample size.
    """
    if len(matrix.labels) < minimum:
        msg = (
            f"{model_id}: the {role} split has {len(matrix.labels)} usable "
            f"row(s), below the configured floor of {minimum}. Raise the floor in "
            "the configuration only if a smaller result is genuinely acceptable, "
            "and record why."
        )
        raise TrainingError(msg)


def _check_both_classes(matrix: FeatureMatrix, *, role: str, model_id: str) -> None:
    """Refuse a split that cannot support a two-class decision.

    Args:
        matrix: The materialised split.
        role: ``"train"`` or ``"dev"``.
        model_id: Model identifier.

    Raises:
        TrainingError: If either class is absent.
    """
    present = set(np.unique(matrix.labels).tolist())
    if present != {0, 1}:
        msg = (
            f"{model_id}: the {role} split contains labels {sorted(present)}, "
            "but both classes are required. A one-class split fits a constant "
            "and then reports a confident wrong answer."
        )
        raise TrainingError(msg)


def _score_with_timing(
    model: BaselineModel, matrix: FeatureMatrix, *, time_it: bool
) -> tuple[np.ndarray, LatencyReport]:
    """Score a split, optionally recording latency.

    Args:
        model: The fitted model.
        matrix: The split to score.
        time_it: Whether to record per-call wall-clock timings.

    Returns:
        The scores and a latency report. The report is ``not run`` unless timing
        was requested, so a report never implies a measurement that did not
        happen.
    """
    if not time_it:
        return np.asarray(model.predict_proba(matrix), dtype=np.float64), LatencyReport.not_run(
            f"{NOT_RUN}: latency timing was not requested for this run"
        )

    per_call: list[float] = []
    scored: list[float] = []
    for index in range(len(matrix.labels)):
        row = FeatureMatrix(
            split=matrix.split,
            vectors=matrix.vectors[index : index + 1],
            labels=matrix.labels[index : index + 1],
            sample_ids=(matrix.sample_ids[index],),
            metadata={key: (value[index],) for key, value in matrix.metadata.items()},
            durations=(float(matrix.durations[index]),),
            matrices=(None if matrix.matrices is None else matrix.matrices[index : index + 1]),
        )
        started = time.perf_counter()
        scored.append(float(model.predict_proba(row)[0]))
        per_call.append((time.perf_counter() - started) * 1000.0)

    audio_seconds = float(np.sum(np.asarray(matrix.durations, dtype=np.float64)))
    return np.asarray(scored, dtype=np.float64), LatencyReport.measured(per_call, audio_seconds)


def _groups_for(matrix: FeatureMatrix) -> dict[str, Any]:
    """Expose every recorded subgroup axis for the report.

    Args:
        matrix: The materialised split.

    Returns:
        Row-aligned arrays keyed by axis name.
    """
    return dict(matrix.metadata)


def not_run_result(
    reason: str,
    *,
    config: TrainingConfig | None = None,
    duration_seconds: float = 0.0,
) -> RunResult:
    """Build a well-formed ``NOT RUN`` result for a run that never started.

    Exists so that callers which cannot even reach :func:`run_baseline` -- a
    manifest that is absent, a config that failed to load -- still report the
    same record shape as a run that stopped halfway. Otherwise the one case this
    repository is always in becomes the one case with no machine-readable
    output, and a pipeline has to distinguish "crashed" from "no data" by
    reading stderr.

    Args:
        reason: Human-readable explanation. The ``NOT RUN`` marker is prepended
            here when absent, so a caller cannot produce a record that quietly
            omits the marker.
        config: Configuration already loaded, if any.
        duration_seconds: Wall-clock seconds spent before giving up.

    Returns:
        A :class:`RunResult` whose reports all say ``NOT RUN``.
    """
    model_id = config.model_id if config is not None else "unknown"
    text = reason if reason.startswith(NOT_RUN) else f"{NOT_RUN}: {reason}"
    return RunResult(
        outcome=RunOutcome.NOT_RUN,
        reason=text,
        config=config,
        test_report=not_run_report(model_id, reason=text),
        duration_seconds=round(duration_seconds, 3),
    )


def run_baseline(
    config: TrainingConfig,
    manifest: Any,
    *,
    root: Path | str,
    dry_run: bool = False,
    time_inference: bool = False,
    evaluate_test: bool = True,
    max_items: int | None = None,
) -> RunResult:
    """Train one baseline and produce its report.

    Args:
        config: The recipe to run.
        manifest: A loaded :class:`~voxshield.data.manifest.Manifest` or a path to
            one.
        root: Data root that relative ``audio_path`` values resolve against.
        dry_run: Resolve and featurise the splits, then stop before fitting. Used
            to verify a configuration against a real corpus without spending a
            training run on it.
        time_inference: Record per-call latency for the reported split.
        evaluate_test: Set ``False`` to stop after the dev report, leaving the
            test split unread. This is the honest way to develop against real
            data without spending the one look at test.
        max_items: Cap on rows per split, for smoke runs.

    Returns:
        A :class:`RunResult`. Failures are returned, not raised, so the CLI can
        report one run's problem without a traceback; the message is in ``reason``.

    Raises:
        TrainingError: Only for programming errors such as an unusable
            configuration.
    """
    started = time.perf_counter()
    model_id = config.model_id

    def elapsed() -> float:
        return round(time.perf_counter() - started, 3)

    def not_run(reason: str) -> RunResult:
        return RunResult(
            outcome=RunOutcome.NOT_RUN,
            reason=reason,
            config=config,
            test_report=not_run_report(model_id, reason=reason),
            duration_seconds=elapsed(),
        )

    try:
        train_rows = resolve_rows(manifest, config.train_split)
        dev_rows = resolve_rows(manifest, config.dev_split)
        # Test identities are resolved before fitting so the leakage gate can see
        # train-vs-test overlap, which is the overlap that most often turns a
        # confident EER into a meaningless one. Resolving reads manifest metadata
        # only: no audio is decoded and nothing is scored until the single
        # evaluation below. When test evaluation is not requested, the split is
        # not even resolved, so "never read the test split" stays literally true.
        test_rows = resolve_rows(manifest, config.test_split) if evaluate_test else ()
    except Exception as exc:
        return RunResult(
            outcome=RunOutcome.FAILED,
            reason=f"could not resolve manifest rows: {exc}",
            config=config,
            duration_seconds=elapsed(),
        )

    if not train_rows or not dev_rows:
        missing = []
        if not train_rows:
            missing.append(config.train_split)
        if not dev_rows:
            missing.append(config.dev_split)
        return not_run(
            f"{NOT_RUN}: manifest split(s) {', '.join(missing)} contain no rows. "
            "This project has no training corpus, so there is nothing to fit."
        )

    if config.train.verify_leakage:
        try:
            require_clean_gates((*train_rows, *dev_rows, *test_rows))
        except TrainingError as exc:
            # The refusal is the point: a baseline trained through a leaked
            # speaker axis still produces a confident EER, and that EER gets
            # quoted later as though it meant something.
            return RunResult(
                outcome=RunOutcome.FAILED,
                reason=str(exc),
                config=config,
                duration_seconds=elapsed(),
            )

    extractor = FeatureExtractor(config.features)
    want_matrices = config.model.family == "logmel_cnn"

    def materialise(split: str, rows: tuple[Any, ...]) -> FeatureMatrix:
        return load_feature_matrix(
            rows,
            extractor,
            root=root,
            split=split,
            want_matrices=want_matrices,
            strict=False,
            max_items=max_items,
        )

    try:
        train = materialise(config.train_split, train_rows)
        dev = materialise(config.dev_split, dev_rows)
    except Exception as exc:
        return RunResult(
            outcome=RunOutcome.FAILED,
            reason=f"could not materialise features: {exc}",
            config=config,
            duration_seconds=elapsed(),
        )

    failures: list[str] = []
    if train.failed:
        failures.append(
            f"{len(train.failed)} of {train.rows_considered} {config.train_split} "
            "segment(s) failed to decode"
        )
    if dev.failed:
        failures.append(
            f"{len(dev.failed)} of {dev.rows_considered} {config.dev_split} "
            "segment(s) failed to decode"
        )
    caveat = "; ".join(failures)

    try:
        _require_rows(train, config.train.min_train_samples, role="train", model_id=model_id)
        _require_rows(dev, config.train.min_dev_samples, role="dev", model_id=model_id)
        _check_both_classes(train, role="train", model_id=model_id)
        _check_both_classes(dev, role="dev", model_id=model_id)
    except TrainingError as exc:
        return not_run(str(exc))

    if dry_run:
        return RunResult(
            outcome=RunOutcome.NOT_RUN,
            reason=(
                f"{NOT_RUN}: dry run. Resolved {len(train_rows)} {config.train_split} "
                f"and {len(dev_rows)} {config.dev_split} row(s), featurised "
                f"{len(train.labels)} and {len(dev.labels)}. "
                + (f"Decode failures: {caveat}. " if caveat else "")
                + "No model was fitted and no split was scored."
            ),
            config=config,
            duration_seconds=elapsed(),
        )

    try:
        model = build_baseline(
            config.model,
            n_frames=config.features.n_frames if want_matrices else None,
            n_bands=config.features.n_mels if want_matrices else None,
            seed=config.train.seed,
            epochs=config.train.epochs,
            eval_every=config.train.eval_every,
            patience=config.train.patience,
        )
        summary = model.fit(train, dev=dev)
    except Exception as exc:
        return RunResult(
            outcome=RunOutcome.FAILED,
            reason=f"fit failed: {exc}",
            config=config,
            duration_seconds=elapsed(),
        )

    dev_scores, _ = _score_with_timing(model, dev, time_it=False)

    # Calibrate on dev, then choose the operating point on the calibrated scores.
    # The order matters: a threshold is a statement about a probability, so it has
    # to be selected after the probabilities mean what they will mean at test time.
    # Dev is the only split either step may look at.
    calibrator, calibration_note = fit_calibrator(config.train.calibration, dev.labels, dev_scores)
    dev_scores = np.asarray(calibrator.transform(dev_scores), dtype=np.float64)

    try:
        threshold, policy = _choose_threshold(dev.labels, dev_scores, config.threshold)
    except TrainingError as exc:
        return RunResult(
            outcome=RunOutcome.FAILED,
            reason=str(exc),
            config=config,
            model=model,
            summary=summary,
            calibrator=calibrator,
            duration_seconds=elapsed(),
        )

    shared_notes = tuple(config.notes) + (calibration_note,) + ((caveat,) if caveat else ())
    build_id = _dataset_build_id(manifest)
    dev_report = build_report(
        model_id,
        dev.labels,
        dev_scores,
        threshold=threshold,
        threshold_policy=policy,
        groups=_groups_for(dev),
        latency=LatencyReport.not_run(
            f"{NOT_RUN}: latency is reported for the evaluated split only"
        ),
        config_hash=config.config_hash(),
        dataset_build_id=build_id,
        notes=shared_notes,
        split=config.dev_split,
    )

    metadata = build_metadata(
        config,
        model,
        summary,
        n_features=int(train.vectors.shape[1]),
        n_frames=config.features.n_frames if want_matrices else None,
        n_bands=config.features.n_mels if want_matrices else None,
        class_counts=train.counts,
        rows_considered=train.rows_considered or len(train_rows),
        failed_decodes=tuple(train.failed) + tuple(dev.failed),
        device=config.train.device,
        calibrator=calibrator,
        threshold=threshold,
        threshold_policy=policy,
        notes=(calibration_note,),
    )

    if not evaluate_test:
        directory = save_artifact(model, metadata, config.output_dir)
        return RunResult(
            outcome=RunOutcome.NOT_RUN,
            reason=(
                f"{NOT_RUN}: test evaluation was skipped, so the test split was "
                "never read. Artefact written without a test result."
            ),
            config=config,
            model=model,
            summary=summary,
            metadata=metadata,
            dev_report=dev_report,
            test_report=not_run_report(
                model_id, reason=f"{NOT_RUN}: test evaluation was not requested"
            ),
            threshold=threshold,
            calibrator=calibrator,
            artifact_path=directory,
            duration_seconds=elapsed(),
        )

    try:
        if not test_rows:
            raise TrainingError(f"manifest split {config.test_split!r} contains no rows")
        test = materialise(config.test_split, test_rows)
        _check_both_classes(test, role=config.test_split, model_id=model_id)
    except Exception as exc:
        return RunResult(
            outcome=RunOutcome.FAILED,
            reason=f"could not materialise the {config.test_split} split: {exc}",
            config=config,
            model=model,
            summary=summary,
            metadata=metadata,
            dev_report=dev_report,
            calibrator=calibrator,
            threshold=threshold,
            duration_seconds=elapsed(),
        )

    # The one and only scoring of test in this module.
    raw_test_scores, latency = _score_with_timing(model, test, time_it=time_inference)
    # The same calibrator fitted on dev, applied once, unchanged. Latency stays
    # raw: calibration is a cheap closed-form map, and timing it alongside
    # inference would blur the number a reader actually cares about.
    test_scores = np.asarray(calibrator.transform(raw_test_scores), dtype=np.float64)
    test_notes = shared_notes
    if test.failed:
        test_notes += (
            f"{len(test.failed)} of {test.rows_considered} "
            f"{config.test_split} segment(s) failed to decode and were excluded",
        )

    test_report = build_report(
        model_id,
        test.labels,
        test_scores,
        threshold=threshold,
        threshold_policy=policy,
        groups=_groups_for(test),
        latency=latency,
        config_hash=config.config_hash(),
        dataset_build_id=build_id,
        notes=test_notes,
        split=config.test_split,
    )

    # Only now, after a test number exists, is the artefact allowed to say so.
    metadata = ArtifactMetadata.from_dict(
        {**metadata.to_dict(), "test_evaluated": True, "format_version": metadata.format_version}
    )
    try:
        directory = save_artifact(model, metadata, config.output_dir)
    except TrainingError as exc:
        return RunResult(
            outcome=RunOutcome.FAILED,
            reason=str(exc),
            config=config,
            model=model,
            summary=summary,
            metadata=metadata,
            dev_report=dev_report,
            calibrator=calibrator,
            test_report=test_report,
            threshold=threshold,
            duration_seconds=elapsed(),
        )

    return RunResult(
        outcome=RunOutcome.MEASURED,
        config=config,
        model=model,
        summary=summary,
        metadata=metadata,
        calibrator=calibrator,
        dev_report=dev_report,
        test_report=test_report,
        threshold=threshold,
        artifact_path=directory,
        duration_seconds=elapsed(),
    )


def _choose_threshold(
    labels: np.ndarray, scores: np.ndarray, policy: ThresholdPolicy
) -> tuple[float, ThresholdPolicy]:
    """Choose the operating point on dev.

    Args:
        labels: Dev labels.
        scores: Dev scores.
        policy: The configured rule.

    Returns:
        The threshold and the policy actually used. When the policy's rate
        ceilings cannot be honoured, the ceiling is dropped and a policy that
        can be honoured is returned instead, so the report never claims a
        constraint that the chosen point violates.

    Raises:
        TrainingError: If no threshold satisfies the policy.
    """
    try:
        threshold = select_threshold(labels, scores, policy)
        if threshold is None:
            raise TrainingError(
                f"no threshold satisfies policy {policy.to_dict()} on dev; the "
                "score distribution does not cross it"
            )
        return float(threshold.threshold), policy
    except TrainingError:
        raise
    except Exception as exc:
        relaxed = ThresholdPolicy(
            strategy=policy.strategy,
            fixed_threshold=policy.fixed_threshold,
        )
        try:
            threshold = select_threshold(labels, scores, relaxed)
        except Exception:
            msg = f"could not choose a dev threshold under policy {policy.to_dict()}: {exc}"
            raise TrainingError(msg) from exc
        if threshold is None:
            msg = (
                f"could not choose a dev threshold under policy "
                f"{policy.to_dict()} or its unconstrained form"
            )
            raise TrainingError(msg) from exc
        return float(threshold.threshold), relaxed


def _policy_named(name: str) -> ThresholdPolicy:
    """Rebuild a threshold policy from its recorded strategy name.

    Args:
        name: The strategy recorded in artefact metadata.

    Returns:
        A policy for reporting. Unknown or absent names fall back to the ``eer``
        default, so a hand-edited artefact still produces a report; the label in
        the report is what tells a reader which rule actually applied.
    """
    try:
        return ThresholdPolicy(strategy=name)
    except (TypeError, ValueError):
        return ThresholdPolicy()


def run_from_artifact(
    path: Path | str,
    manifest: Any,
    *,
    root: Path | str,
    split: str = "test",
    time_inference: bool = False,
    threshold: float | None = None,
) -> RunResult:
    """Score a split with a previously saved artefact.

    This is the path a consumer takes. It loads the model rather than refitting,
    which is the whole reason the scaler travels with the estimator, and it never
    touches train or dev.

    Args:
        path: Artefact directory.
        manifest: Manifest holding ``split``.
        root: Data root for relative paths.
        split: Which split to score.
        time_inference: Record per-call latency.
        threshold: Operating point to apply, overriding the one stored in the
            artefact. For scoring at a point the model was not selected at.

    Returns:
        A :class:`RunResult`. ``outcome`` is ``measured`` when the split was
        scored, and ``not_run`` when it held no rows.

    Raises:
        TrainingError: If the artefact cannot be loaded.
    """
    started = time.perf_counter()
    model, metadata = load_artifact(path)
    model_id = metadata.model_id

    def elapsed() -> float:
        return round(time.perf_counter() - started, 3)

    try:
        rows = resolve_rows(manifest, split)
    except Exception as exc:
        return RunResult(
            outcome=RunOutcome.FAILED,
            reason=f"could not resolve manifest rows: {exc}",
            metadata=metadata,
            duration_seconds=elapsed(),
        )

    if not rows:
        reason = f"{NOT_RUN}: manifest split {split!r} contains no rows"
        return RunResult(
            outcome=RunOutcome.NOT_RUN,
            reason=reason,
            metadata=metadata,
            model=model,
            test_report=not_run_report(model_id, reason=reason),
            duration_seconds=elapsed(),
        )

    from voxshield.training.config import FeatureSpec

    # The artefact records its own front end. Falling back to the current
    # defaults would silently featurise with whatever the checkout happens to
    # say today, which is the shape of bug that produces a plausible number and
    # a wrong one.
    spec = (
        FeatureSpec.from_mapping(metadata.feature_spec) if metadata.feature_spec else FeatureSpec()
    )
    extractor = FeatureExtractor(spec)
    try:
        matrix = load_feature_matrix(
            rows,
            extractor,
            root=root,
            split=split,
            want_matrices=metadata.family == "logmel_cnn",
            strict=False,
        )
        if not len(matrix.labels):
            raise TrainingError(f"every {split} segment failed to decode")
    except Exception as exc:
        return RunResult(
            outcome=RunOutcome.FAILED,
            reason=f"could not materialise features: {exc}",
            metadata=metadata,
            model=model,
            duration_seconds=elapsed(),
        )

    raw_scores, latency = _score_with_timing(model, matrix, time_it=time_inference)
    # Reproduce the probabilities the run that produced this artefact reported:
    # the stored calibrator, unchanged, applied once. Reporting uncalibrated
    # scores next to a calibrated run's numbers would make the two incomparable.
    calibrator = calibrator_from_dict(metadata.calibration)
    scores = np.asarray(calibrator.transform(raw_scores), dtype=np.float64)

    # An explicit threshold overrides the stored one; otherwise the dev-selected
    # operating point travels with the model, so FAR/FRR are the same decision
    # the original run made.
    applied = metadata.threshold if threshold is None else threshold
    if applied is None:
        notes = (
            "scored from a saved artefact; no threshold was applied",
            f"calibration: {metadata.calibration_method}",
        )
        reason = (
            f"{NOT_AVAILABLE}: this artefact predates threshold recording, so no "
            "operating point was applied. Rescore with an explicit --threshold."
        )
        threshold_policy = None
    else:
        notes = (
            "scored from a saved artefact at its recorded dev operating point",
            f"calibration: {metadata.calibration_method}",
        )
        reason = ""
        threshold_policy = metadata.threshold_policy

    report = build_report(
        model_id,
        matrix.labels,
        scores,
        threshold=applied,
        threshold_policy=(
            _policy_named(threshold_policy) if isinstance(threshold_policy, str) else None
        ),
        groups=_groups_for(matrix),
        latency=latency,
        config_hash=metadata.config_hash,
        notes=notes,
        split=split,
    )
    return RunResult(
        outcome=RunOutcome.MEASURED,
        reason=reason,
        metadata=metadata,
        model=model,
        calibrator=calibrator,
        test_report=report,
        threshold=applied,
        artifact_path=Path(path),
        duration_seconds=elapsed(),
    )
