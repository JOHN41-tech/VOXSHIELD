"""Phase 3 evaluation: score a detector, and say plainly what was not measured.

This package answers three questions that Phase 2 deliberately did not: how good
is a detector, how good is it on each subgroup, and how fast is it. The third
module boundary matters as much as the first two.

A report from this package has to survive being read by someone who will make a
decision about routing cases to a human analyst. That reader needs to know which
numbers came from data and which are placeholders, because the two are easy to
confuse and only one of them is evidence. So the report carries an explicit
status field -- ``"measured"``, ``"not_run"``, or ``"unavailable"`` -- rather than
relying on a missing key or a null. Three states, three different meanings:

* ``measured`` -- computed from real scored samples, with the sample count
  attached so a reader can judge the precision of the estimate.
* ``not_run`` -- the experiment was never executed. This is the state of every
  metric in this repository today, because no corpus is present. It is not an
  error to report and must not be rendered as one.
* ``unavailable`` -- the experiment was attempted and could not be completed:
  a subgroup with one class, a corpus too small for the statistic, a missing
  probability channel. The reason is carried alongside.

Collapsing ``not_run`` into ``unavailable`` would hide the difference between
"we did not do this" and "we tried and could not", which is exactly the
distinction a reader needs before trusting any number that is present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from voxshield.data.labels import BONA_FIDE, SPOOF
from voxshield.evaluation.metrics import (
    NOT_AVAILABLE,
    BinaryMetrics,
    ConfusionCounts,
    OperatingPoint,
    ThresholdPolicy,
    _binary_arrays,
    brier_score,
    confusion_at,
    detect,
    expected_calibration_error,
    operating_points,
    pr_curve_points,
    reliability_bins,
    roc_curve_points,
    score_bands,
    select_threshold,
    subgroup_scores,
)

__all__ = [
    "NOT_RUN",
    "EvaluationReport",
    "EvaluationStatus",
    "LatencyReport",
    "SubgroupReport",
    "build_report",
    "calibration_summary",
    "describe_operating_point",
    "not_run_report",
    "operating_point_at",
]

#: Timings below this many calls are reported as unavailable. A p95 over three
#: samples is one of the three samples, and printing it would invite exactly the
#: reading it cannot support.
_MIN_TIMING_SAMPLES = 5

#: A slice needs this many samples of *each* class before its two-class metrics
#: are treated as an estimate rather than an anecdote.
_MIN_SUBGROUP_SAMPLES = 20


class EvaluationStatus(StrEnum):
    """Whether a reported figure was measured, skipped, or impossible.

    :class:`enum.StrEnum` rather than a ``str`` mixin, because the whole point of
    these values is to survive a round trip through JSON: a mixin enum
    serialises as ``"EvaluationStatus.MEASURED"`` under a naive encoder and
    compares unequal to the plain string ``"measured"``, which is precisely the
    kind of quiet mismatch that makes a report's status field untrustworthy.
    ``StrEnum`` serialises as its own value, compares equal to it, and needs no
    hand-written ``__str__``.
    """

    MEASURED = "measured"
    NOT_RUN = "not_run"
    UNAVAILABLE = "unavailable"


#: The reason attached to any experiment that was never executed. Constant so
#: that a grep for it finds every not-run result in the repository.
NOT_RUN = "not_run"


@dataclass(frozen=True, slots=True)
class LatencyReport:
    """Wall-clock cost of scoring, with the sample size attached.

    Latency is reported as a distribution rather than a mean because the tail is
    the only part that a reviewer feels: a detector whose 95th percentile is
    three times its median will produce visible latency on real traffic even when
    the average looks acceptable. Counts accompany the percentiles because a
    percentile over four samples is not an estimate of anything, and a report
    that omits the count invites exactly that misreading.

    Attributes:
        status: :attr:`EvaluationStatus.MEASURED` only when samples were timed.
        n_samples: Timed scoring calls.
        mean_ms: Mean wall-clock milliseconds per call.
        median_ms: Median milliseconds per call.
        p95_ms: 95th percentile milliseconds per call.
        p99_ms: 99th percentile milliseconds per call.
        max_ms: Slowest observed call.
        real_time_factor: Audio seconds processed per wall-clock second, computed
            as total audio over total wall time. Above 1.0 is faster than real
            time.
        audio_seconds: Total audio duration behind the timing, when known.
        reason: Why the figures are absent.
    """

    status: EvaluationStatus
    n_samples: int = 0
    mean_ms: float | None = None
    median_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None
    max_ms: float | None = None
    real_time_factor: float | None = None
    audio_seconds: float | None = None
    reason: str | None = None

    @classmethod
    def not_run(cls, reason: str = NOT_RUN) -> LatencyReport:
        """A timing result for an experiment that was never executed."""
        return cls(status=EvaluationStatus.NOT_RUN, reason=reason)

    @classmethod
    def unavailable(cls, reason: str) -> LatencyReport:
        """A timing result for an experiment that could not be completed."""
        return cls(status=EvaluationStatus.UNAVAILABLE, reason=reason)

    @classmethod
    def measured(
        cls,
        durations_ms: Any,
        audio_seconds: float | None = None,
    ) -> LatencyReport:
        """Summarise observed timings.

        Args:
            durations_ms: Per-call wall-clock milliseconds.
            audio_seconds: Total audio duration behind those calls.

        Returns:
            A :class:`LatencyReport`. Fewer than :data:`_MIN_TIMING_SAMPLES`
            samples yields ``unavailable`` rather than a percentile computed from
            too few points to mean anything.
        """
        values = np.asarray(list(durations_ms), dtype=np.float64)
        if values.size == 0:
            return cls.unavailable("no timings were recorded")
        if values.size < _MIN_TIMING_SAMPLES:
            return cls.unavailable(
                f"{values.size} timing sample(s) is below the "
                f"{_MIN_TIMING_SAMPLES} needed for a percentile"
            )
        factor: float | None = None
        if audio_seconds is not None:
            total_wall_s = float(values.sum()) / 1000.0
            if total_wall_s > 0.0:
                # Total audio over total wall time. Both sides must be totals:
                # dividing total audio by a *median* latency compares an aggregate
                # against a per-call statistic, which reports a figure inflated
                # by the number of calls -- 20 one-second clips at 10 ms each is
                # 100x real time, not 2000x.
                factor = float(audio_seconds / total_wall_s)
        return cls(
            status=EvaluationStatus.MEASURED,
            n_samples=int(values.size),
            mean_ms=float(values.mean()),
            median_ms=float(np.median(values)),
            p95_ms=float(np.percentile(values, 95)),
            p99_ms=float(np.percentile(values, 99)),
            max_ms=float(values.max()),
            real_time_factor=factor,
            audio_seconds=audio_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "status": str(self.status),
            "n_samples": self.n_samples,
            "mean_ms": _round(self.mean_ms),
            "median_ms": _round(self.median_ms),
            "p95_ms": _round(self.p95_ms),
            "p99_ms": _round(self.p99_ms),
            "max_ms": _round(self.max_ms),
            "real_time_factor": _round(self.real_time_factor),
            "audio_seconds": _round(self.audio_seconds),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SubgroupReport:
    """Metrics for one slice of the test set, with its own sample counts.

    A subgroup with a handful of samples is the main way an evaluation becomes
    misleading: an AUC over four spoof samples is not a measurement, but it will
    render as a number on a chart. This report therefore carries
    :attr:`reliable`, which is false below :data:`_MIN_SUBGROUP_SAMPLES` on
    either side, so a chart can show the slice without implying it is supported.

    Attributes:
        name: Group value, e.g. ``"generator=waveglow"``.
        axis: Which attribute was grouped on, e.g. ``"generator"``.
        status: Whether the slice was scored at all.
        metrics: The scored metrics, when available.
        n_samples: Rows in the slice.
        n_positive: Spoof rows in the slice.
        n_negative: Bona fide rows in the slice.
        reliable: Whether both classes meet :data:`_MIN_SUBGROUP_SAMPLES`.
        reason: Why the slice could not be scored.
    """

    name: str
    axis: str
    status: EvaluationStatus
    metrics: BinaryMetrics | None = None
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0
    reliable: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "name": self.name,
            "axis": self.axis,
            "status": str(self.status),
            "reliable": self.reliable,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "reason": self.reason,
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """A complete, honest record of one evaluation run.

    Attributes:
        model_id: Identifier of the evaluated model, matching the registry.
        split: Which split was scored. Always the test split in a real report.
        status: Overall status. ``NOT_RUN`` means no metric in this report came
            from data.
        metrics: Whole-split metrics, when scored.
        threshold: The operating point, chosen on dev and applied unchanged.
        threshold_policy: The rule used to choose it, for reproducibility.
        confusion: Confusion at the operating point.
        curves: Named curves, e.g. ``"roc"`` and ``"pr"``.
        calibration_bins: Reliability bins, when probabilities were available.
        subgroups: Per-slice reports.
        latency: Timing result.
        dataset_build_id: The dataset the run was produced from, so a figure can
            be invalidated when the corpus changes.
        config_hash: Hash of the training configuration.
        notes: Free-text caveats, e.g. placeholder thresholds.
        reason: Why the report carries no measurements.
    """

    model_id: str
    split: str = "test"
    status: EvaluationStatus = EvaluationStatus.NOT_RUN
    metrics: BinaryMetrics | None = None
    threshold: float | None = None
    threshold_policy: ThresholdPolicy | None = None
    confusion: ConfusionCounts | None = None
    curves: dict[str, Any] = field(default_factory=dict)
    calibration_bins: tuple[Any, ...] = ()
    subgroups: tuple[SubgroupReport, ...] = ()
    latency: LatencyReport = field(default_factory=LatencyReport.not_run)
    dataset_build_id: str | None = None
    config_hash: str | None = None
    notes: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def measured(self) -> bool:
        """Whether any metric in this report came from real samples."""
        return self.status is EvaluationStatus.MEASURED

    def subgroup(self, name: str) -> SubgroupReport | None:
        """Look up one subgroup by name.

        Args:
            name: The ``"axis=value"`` key the report uses.

        Returns:
            The report, or ``None`` when the axis had no such value.
        """
        for item in self.subgroups:
            if item.name == name:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping, safe to print when nothing was run.

        ``NOT AVAILABLE`` marks each individual metric that could not be
        computed, so a reader scanning the JSON sees per-metric absence rather
        than having to correlate a null with the status field to know which.
        """
        return {
            "model_id": self.model_id,
            "split": self.split,
            "status": str(self.status),
            "reason": self.reason,
            "measured": self.measured,
            "dataset_build_id": self.dataset_build_id,
            "config_hash": self.config_hash,
            "threshold": self.threshold,
            "threshold_policy": (
                None if self.threshold_policy is None else self.threshold_policy.to_dict()
            ),
            "metrics": None if self.metrics is None else self.metrics.to_dict(),
            "confusion": None if self.confusion is None else self.confusion.to_dict(),
            "curves": self.curves,
            "calibration_bins": [item.to_dict() for item in self.calibration_bins],
            "subgroups": [item.to_dict() for item in self.subgroups],
            "latency": self.latency.to_dict(),
            "notes": list(self.notes),
            "positive_class": SPOOF,
            "negative_class": BONA_FIDE,
        }

    def summary_line(self) -> str:
        """One line for a table or a log, safe when nothing was run.

        Returns a ``NOT RUN`` line rather than an exception or a row of dashes
        when unmeasured, so that a table containing several models stays aligned
        whether or not any of them have been trained.
        """
        if not self.measured:
            return f"{self.model_id}: NOT RUN ({self.reason or NOT_RUN})"
        parts = [f"{self.model_id}"]
        if self.metrics is not None and self.metrics.roc_auc is not None:
            parts.append(f"EER={self.metrics.eer:.4f}")
        if self.metrics is not None and self.metrics.eer is not None:
            parts.append(f"AUC={self.metrics.roc_auc:.4f}")
        if self.threshold is not None:
            parts.append(f"threshold={self.threshold:.4f}")
        return " ".join(parts)


def _round(value: float | None) -> float | str:
    """Serialise an optional measurement, marking absence explicitly."""
    return NOT_AVAILABLE if value is None else round(float(value), 6)


def not_run_report(
    model_id: str,
    *,
    reason: str = NOT_RUN,
    split: str = "test",
    notes: tuple[str, ...] = (),
) -> EvaluationReport:
    """A report for an experiment that was never executed.

    This is the correct output of ``voxshield ml evaluate`` against a repository
    with no data. Returning it with ``status=NOT_RUN`` -- rather than raising,
    and rather than fabricating a zero-valued metric block -- is what lets the
    command run, print an auditable artefact, and exit with a distinct code that
    a pipeline can act on without having to parse prose.

    Args:
        model_id: Identifier of the model that would have been evaluated.
        reason: Why nothing was run.
        split: The split that would have been scored.
        notes: Caveats to record alongside the absent measurements.

    Returns:
        An :class:`EvaluationReport` carrying no measurements.
    """
    return EvaluationReport(
        model_id=model_id,
        split=split,
        status=EvaluationStatus.NOT_RUN,
        reason=reason,
        notes=notes,
        latency=LatencyReport.not_run(reason),
    )


def _subgroup_report(
    name: str,
    axis: str,
    labels: Any,
    scores: Any,
    threshold: float | None,
) -> SubgroupReport:
    """Score one slice, marking thin slices as unreliable rather than absent."""
    slice_metrics = detect(labels, scores, threshold=threshold)
    n_pos = slice_metrics.n_positive
    n_neg = slice_metrics.n_negative
    reliable = (
        slice_metrics.available
        and n_pos >= _MIN_SUBGROUP_SAMPLES
        and n_neg >= _MIN_SUBGROUP_SAMPLES
    )
    if not slice_metrics.available:
        status = EvaluationStatus.UNAVAILABLE
        reason: str | None = slice_metrics.reason
    elif reliable:
        status = EvaluationStatus.MEASURED
        reason = None
    else:
        status = EvaluationStatus.UNAVAILABLE
        reason = (
            f"{n_pos} spoof / {n_neg} bona fide samples is below "
            f"{_MIN_SUBGROUP_SAMPLES} per class; metrics are shown but are not "
            "an estimate"
        )
    return SubgroupReport(
        name=name,
        axis=axis,
        status=status,
        metrics=slice_metrics,
        n_samples=slice_metrics.n_samples,
        n_positive=n_pos,
        n_negative=n_neg,
        reliable=reliable,
        reason=reason,
    )


def build_report(
    model_id: str,
    labels: Any,
    scores: Any,
    *,
    threshold: float | None = None,
    threshold_policy: ThresholdPolicy | None = None,
    groups: dict[str, Any] | None = None,
    latency: LatencyReport | None = None,
    dataset_build_id: str | None = None,
    config_hash: str | None = None,
    notes: tuple[str, ...] = (),
    split: str = "test",
) -> EvaluationReport:
    """Assemble a full report from scored samples.

    Args:
        model_id: Identifier of the evaluated model.
        labels: Labels for the split.
        scores: Detector scores for the split. Values in ``[0, 1]`` are also
            treated as probabilities, which enables the calibration figures.
        threshold: Operating point chosen on dev and applied unchanged.
        threshold_policy: The rule used to choose :attr:`threshold`.
        groups: Row-aligned attribute arrays for subgroup analysis, e.g.
            ``{"generator": [...], "language": [...]}``.
        latency: A measured timing result, or ``None`` for not-run.
        dataset_build_id: Dataset the run was produced from.
        config_hash: Hash of the training configuration.
        notes: Caveats to record.
        split: Which split was scored.

    Returns:
        An :class:`EvaluationReport`. Subgroups that cannot support a two-class
        metric are included with their own status and reason rather than being
        dropped, so the reader can see that the slice was attempted.
    """
    metrics = detect(labels, scores, threshold=threshold)
    confusion: ConfusionCounts | None = None
    if threshold is not None:
        ys, ss, problem = _binary_arrays(labels, scores)
        if problem is None and len(ss):
            confusion = confusion_at(ys, ss, float(threshold))

    curves: dict[str, Any] = {}
    if metrics.available:
        curves["roc"] = roc_curve_points(labels, scores).to_dict()
        try:
            curves["pr"] = pr_curve_points(labels, scores).to_dict()
        except ValueError:
            # No spoof samples: precision is undefined. Recorded as absent rather
            # than as a curve of zeros.
            pass

    bins: tuple[Any, ...] = ()
    if metrics.available:
        try:
            bins = reliability_bins(labels, scores)
        except ValueError:
            # Raw scores outside [0, 1] are not probabilities, so there is no
            # calibration curve to draw. This is a property of the model, not an
            # evaluation failure.
            bins = ()

    subgroups: tuple[SubgroupReport, ...] = ()
    if groups:
        buckets = subgroup_scores(labels, scores, groups)
        subgroups = tuple(
            _subgroup_report(name, name.split("=", 1)[0], bucket_labels, bucket_scores, threshold)
            for name, (bucket_labels, bucket_scores) in buckets.items()
        )

    status = (
        EvaluationStatus.MEASURED
        if metrics.available and metrics.n_samples > 0
        else EvaluationStatus.UNAVAILABLE
    )

    return EvaluationReport(
        model_id=model_id,
        split=split,
        status=status,
        metrics=metrics,
        threshold=threshold,
        threshold_policy=threshold_policy,
        confusion=confusion,
        curves=curves,
        calibration_bins=bins,
        subgroups=subgroups,
        latency=latency or LatencyReport.not_run(),
        dataset_build_id=dataset_build_id,
        config_hash=config_hash,
        notes=notes,
    )


def describe_operating_point(
    labels: Any,
    scores: Any,
    policy: ThresholdPolicy | None = None,
) -> dict[str, Any]:
    """Sweep every threshold and report the selectable ones.

    Useful when the operating point is a policy decision rather than a modelling
    one: the sweep shows what each constraint would cost, so a reviewer can see
    what ``max_frr=0.01`` buys and what it gives up before it is adopted.

    Args:
        labels: Labels to sweep, which must be the dev split.
        scores: Dev scores.
        policy: Optional policy; only its rate ceilings are applied here.

    Returns:
        A mapping with the selectable points, the chosen point, and a note that
        the choice must be made on dev and applied unchanged.
    """
    chosen_policy = policy or ThresholdPolicy()
    ys, ss, problem = _binary_arrays(labels, scores)
    if problem is not None:
        return {"error": problem, "points": [], "chosen": None}
    selectable: list[dict[str, Any]] = []
    for point in operating_points(ys, ss):
        if not point.feasible:
            continue
        if chosen_policy.max_far is not None and (
            point.counts.far is None or point.counts.far > chosen_policy.max_far
        ):
            continue
        if chosen_policy.max_frr is not None and (
            point.counts.frr is None or point.counts.frr > chosen_policy.max_frr
        ):
            continue
        selectable.append(point.to_dict())

    chosen = select_threshold(ys, ss, chosen_policy)
    return {
        "points": selectable,
        "chosen": None if chosen is None else chosen.to_dict(),
        "note": (
            "Select the operating point on dev and apply it unchanged to test; "
            "choosing on test and reporting the result is not an evaluation."
        ),
    }


def calibration_summary(labels: Any, scores: Any, n_bins: int = 10) -> dict[str, Any]:
    """Summarise how well predicted probabilities track observed rates.

    Args:
        labels: Labels for the split.
        scores: Probabilities in ``[0, 1]``.
        n_bins: Bin count.

    Returns:
        The Brier score, the skill score relative to a constant forecast, the
        expected calibration error, and the risk-band distribution.
    """
    return {
        "brier_score": _round(brier_score(labels, scores)),
        "expected_calibration_error": _round(
            expected_calibration_error(labels, scores, n_bins=n_bins)
        ),
        "risk_band_counts": score_bands(scores),
        "bands_note": (
            "Band edges are the unvalidated placeholders in voxshield.policy; "
            "they are counts under a notional split, not operating points."
        ),
    }


def operating_point_at(
    labels: Any,
    scores: Any,
    threshold: float,
) -> OperatingPoint:
    """Confusion at one fixed threshold.

    Args:
        labels: Labels for the split.
        scores: Scores for the split.
        threshold: The operating point.

    Returns:
        The :class:`OperatingPoint`.

    Raises:
        ValueError: Labels and scores are not readable as binary pairs.
    """
    ys, ss, problem = _binary_arrays(labels, scores)
    if problem is not None:
        raise ValueError(problem)
    return OperatingPoint(float(threshold), confusion_at(ys, ss, float(threshold)))
