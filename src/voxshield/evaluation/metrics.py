"""Binary detection metrics for the Phase 3 baselines.

Three decisions live here, and each of them exists because the obvious
alternative produces numbers that look valid and are not.

**The positive class is spoof.** Every rate in this module is stated from the
synthetic-speech side. A detector's false reject rate is how often *real* speech
is called synthetic; its false accept rate is how often *synthetic* speech is
called real. Writing ``far`` and ``frr`` without saying which class they are
about is how a report ends up claiming a 2% false-positive rate that is really a
2% false-reject rate.

**Metrics are computed here, in NumPy, rather than imported.** ``sklearn`` is a
real dependency of the training layer, but metrics are the part of Phase 3 that
must be computable in a minimal environment -- a CI job that only wants to check
that ``eer`` is computed correctly, a reader checking a published figure, or a
future Phase 4 risk engine that needs a calibration curve without pulling in
torch. A metric that can only be checked by running a full training stack is a
metric nobody checks.

**An undefined metric is ``None``, never a number.** One-class splits, absent
subgroups, and thresholds that were never selected produce ``None`` with a
stated reason. This matters more here than in most detection work: the corpus
that would make an EER well defined is the corpus this project does not have
yet, and the most likely failure mode for the first real report is a silent
``0.0`` where the honest answer is "not measurable".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from scipy.special import ndtri

from voxshield.data.labels import BONA_FIDE, SPOOF, encode_label

__all__ = [
    "NOT_AVAILABLE",
    "BinaryMetrics",
    "ConfusionCounts",
    "Curve",
    "OperatingPoint",
    "ReliabilityBin",
    "ThresholdPolicy",
    "average_precision",
    "brier_score",
    "brier_skill_score",
    "confusion_at",
    "describe_labels",
    "detect",
    "eer",
    "expected_calibration_error",
    "operating_points",
    "pr_curve_points",
    "reliability_bins",
    "roc_auc",
    "roc_curve_points",
    "score_bands",
    "select_threshold",
    "subgroup_scores",
]

#: The literal written wherever a metric could not be computed. It is a string
#: rather than ``None`` in the serialised form so that a JSON report is
#: self-describing: a reader who only ever sees the JSON cannot mistake an
#: absent measurement for a measured zero.
NOT_AVAILABLE: Final = "NOT AVAILABLE"

#: Guard against float thresholds that cannot order any score. ``np.inf`` is the
#: conventional ROC sentinel but is useless as an operating point: it rejects
#: every sample, so its confusion counts are real numbers describing a model
#: that never fires.
_MAX_THRESHOLD: Final = 1.0 + 1e-9

#: Selection rules :class:`ThresholdPolicy` understands. Declared at module scope
#: rather than in the class body: under ``slots=True`` an annotated class-level
#: constant becomes a dataclass field, and a ``Final`` mapping is not one.
_STRATEGIES: Final[frozenset[str]] = frozenset(
    {"eer", "youden", "max_recall", "min_mistakes", "fixed"}
)


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """Integer confusion counts for one operating point.

    Named from the spoof-positive side, so ``true_positive`` counts synthetic
    speech correctly labelled and ``false_negative`` counts synthetic speech
    missed.

    Attributes:
        true_positive: Spoof scored at or above threshold.
        false_positive: Bona fide scored at or above threshold.
        true_negative: Bona fide scored below threshold.
        false_negative: Spoof scored below threshold.
    """

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    @property
    def total(self) -> int:
        """Every sample at this operating point."""
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative

    @property
    def support_positive(self) -> int:
        """Spoof samples, the number of genuine detections available."""
        return self.true_positive + self.false_negative

    @property
    def support_negative(self) -> int:
        """Bona fide samples, the number of genuine releases available."""
        return self.false_positive + self.true_negative

    @property
    def far(self) -> float | None:
        """False accept rate: bona fide called synthetic."""
        if self.support_negative == 0:
            return None
        return self.false_positive / self.support_negative

    @property
    def frr(self) -> float | None:
        """False reject rate: synthetic called bona fide. The missed-detection rate."""
        if self.support_positive == 0:
            return None
        return self.false_negative / self.support_positive

    @property
    def accuracy(self) -> float | None:
        """Fraction of all samples classified correctly."""
        if self.total == 0:
            return None
        return (self.true_positive + self.true_negative) / self.total

    @property
    def balanced_accuracy(self) -> float | None:
        """Mean of the two class recalls.

        Preferred over :attr:`accuracy` here because the spoof/bona fide ratio in
        any usable corpus is lopsided enough that raw accuracy is close to the
        majority-class rate. Reporting accuracy alone would let a model that
        never fires score well.
        """
        far, frr = self.far, self.frr
        if far is None or frr is None:
            return None
        return ((1.0 - frr) + (1.0 - far)) / 2.0

    @property
    def precision(self) -> float | None:
        """Positive predictive value: of those called synthetic, how many were."""
        called = self.true_positive + self.false_positive
        if called == 0:
            return None
        return self.true_positive / called

    @property
    def recall(self) -> float | None:
        """True positive rate: of synthetic samples, how many were caught."""
        return None if self.frr is None else 1.0 - self.frr

    @property
    def specificity(self) -> float | None:
        """True negative rate: of bona fide samples, how many passed."""
        return None if self.far is None else 1.0 - self.far

    @property
    def f1(self) -> float | None:
        """Harmonic mean of precision and recall, ``None`` if both are zero."""
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or (precision + recall) == 0.0:
            return None
        return 2.0 * precision * recall / (precision + recall)

    @property
    def misclassification_count(self) -> int:
        """Absolute number of errors, for tie-breaking a threshold."""
        return self.false_positive + self.false_negative

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping, with :data:`NOT_AVAILABLE` for undefined rates."""
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "far": _number(self.far),
            "frr": _number(self.frr),
            "accuracy": _number(self.accuracy),
            "balanced_accuracy": _number(self.balanced_accuracy),
            "precision": _number(self.precision),
            "recall": _number(self.recall),
            "specificity": _number(self.specificity),
            "f1": _number(self.f1),
        }


@dataclass(frozen=True, slots=True)
class OperatingPoint:
    """One threshold together with the confusion it produces.

    Attributes:
        threshold: Score at or above which a sample is called spoof. Chosen on
            dev, applied unchanged to test.
        counts: Confusion at this threshold.
    """

    threshold: float
    counts: ConfusionCounts

    @property
    def feasible(self) -> bool:
        """Whether the threshold calls any sample at all.

        A threshold above every score produces a perfectly clean confusion matrix
        that means nothing. It is retained in sweeps rather than dropped so that
        the shape of the curve stays inspectable, and filtered at selection time.
        """
        return self.counts.true_positive + self.counts.false_positive > 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {"threshold": self.threshold, **self.counts.to_dict()}


@dataclass(frozen=True, slots=True)
class Curve:
    """A sampled detection curve.

    Axis names are stored rather than inferred, because three curves share one
    class and their axes are not interchangeable: a saved DET curve labelled
    ``false_positive_rate`` on an axis that actually holds false *reject* rate in
    deviate units is not a labelling slip, it is a plot of a detector inverted.

    Attributes:
        x: Horizontal coordinates.
        y: Vertical coordinates.
        thresholds: Score at each vertex, descending. :data:`numpy.inf` first,
            following the usual convention: a threshold above every score.
        x_name: What ``x`` holds, e.g. ``"false_positive_rate"``.
        y_name: What ``y`` holds, e.g. ``"true_positive_rate"``.
        units: Transform applied to the axes. ``"rate"`` or ``"normal_deviate"``.
        positive_class: Always ``"spoof"``. Recorded so a saved curve cannot be
            read with the classes silently swapped.
    """

    x: np.ndarray
    y: np.ndarray
    thresholds: np.ndarray
    x_name: str = "false_positive_rate"
    y_name: str = "true_positive_rate"
    units: str = "rate"
    positive_class: str = SPOOF

    def __post_init__(self) -> None:
        """Reject a curve whose axes disagree in length."""
        if not (len(self.x) == len(self.y) == len(self.thresholds)):
            msg = "curve axes must have equal length"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping with parallel point lists."""
        return {
            "x_name": self.x_name,
            "y_name": self.y_name,
            "units": self.units,
            "positive_class": self.positive_class,
            "points": [
                {
                    "threshold": _number(float(thr)),
                    "x": _number(float(x)),
                    "y": _number(float(y)),
                }
                for thr, x, y in zip(self.thresholds, self.x, self.y, strict=True)
            ],
        }


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One bin of a calibration curve.

    Attributes:
        lower: Inclusive lower edge of predicted probability.
        upper: Exclusive upper edge, except in the final bin.
        count: Samples in the bin.
        mean_predicted: Mean predicted probability in the bin.
        observed_fraction: Fraction of the bin that was actually spoof.
    """

    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_fraction: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_predicted": self.mean_predicted,
            "observed_fraction": self.observed_fraction,
        }


@dataclass(frozen=True, slots=True)
class ThresholdPolicy:
    """How an operating point is chosen, and under what constraints.

    The threshold is always chosen on dev and applied unchanged to test. This
    policy exists to make that choice reproducible rather than incidental: two
    people picking "a reasonable threshold" from the same dev scores should get
    the same number, and the report should say which number and why.

    Attributes:
        strategy: ``"eer"`` minimises the equal error rate, ``"youden"``
            maximises ``tpr - fpr``, ``"max_recall"`` catches as much synthetic
            speech as any threshold can while ignoring the bona fide cost,
            ``"min_mistakes"`` minimises the absolute error count, and
            ``"fixed"`` takes :attr:`fixed_threshold` verbatim.
        max_frr: Optional ceiling on the false reject rate. A candidate above it
            is not selectable, which is how a product states "a missed synthetic
            clip is worse than a false alarm".
        max_far: Optional ceiling on the false accept rate.
        fixed_threshold: The threshold used when ``strategy`` is ``"fixed"``.
    """

    strategy: str = "eer"
    max_frr: float | None = None
    max_far: float | None = None
    fixed_threshold: float | None = None

    def __post_init__(self) -> None:
        """Validate the strategy and the rate ceilings."""
        if self.strategy not in _STRATEGIES:
            msg = f"unknown threshold strategy {self.strategy!r}"
            raise ValueError(msg)
        if self.strategy == "fixed" and self.fixed_threshold is None:
            msg = "strategy 'fixed' requires fixed_threshold"
            raise ValueError(msg)
        for name, value in (("max_frr", self.max_frr), ("max_far", self.max_far)):
            if value is not None and not 0.0 <= value <= 1.0:
                msg = f"{name} must lie in [0, 1], got {value!r}"
                raise ValueError(msg)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "strategy": self.strategy,
            "max_frr": self.max_frr,
            "max_far": self.max_far,
            "fixed_threshold": self.fixed_threshold,
        }


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """The full metric picture for one split at one operating point.

    Curve-derived metrics (``roc_auc``, ``average_precision``, ``eer``) are
    available whenever both classes are present. Operating-point metrics are
    available only when a threshold was supplied, because choosing one from the
    same scores being scored is how a test set quietly becomes a tuning set.

    Attributes:
        n_samples: Rows scored.
        n_positive: Spoof rows.
        n_negative: Bona fide rows.
        threshold: The applied operating point, if one was applied.
        counts: Confusion at :attr:`threshold`, if applied.
        roc_auc: Area under the ROC curve.
        average_precision: Step-wise area under the precision/recall curve.
        eer: Equal error rate.
        eer_threshold: Interpolated threshold at the ROC crossing.
        brier: Mean squared error of the probability, lower is better.
        brier_skill: Brier score relative to a constant-base-rate forecast. Zero
            is no better than always predicting the base rate.
        expected_calibration_error: Mean absolute gap between predicted
            probability and observed rate, over populated bins.
        reason: Why this report carries no numbers, when it carries none.
    """

    n_samples: int
    n_positive: int
    n_negative: int
    threshold: float | None = None
    counts: ConfusionCounts | None = None
    roc_auc: float | None = None
    average_precision: float | None = None
    eer: float | None = None
    eer_threshold: float | None = None
    brier: float | None = None
    brier_skill: float | None = None
    expected_calibration_error: float | None = None
    reason: str | None = None

    @property
    def available(self) -> bool:
        """Whether the scores were usable at all."""
        return self.reason is None

    @property
    def far(self) -> float | None:
        """False accept rate at the applied threshold."""
        return None if self.counts is None else self.counts.far

    @property
    def frr(self) -> float | None:
        """False reject rate at the applied threshold."""
        return None if self.counts is None else self.counts.frr

    @property
    def accuracy(self) -> float | None:
        """Accuracy at the applied threshold."""
        return None if self.counts is None else self.counts.accuracy

    @property
    def balanced_accuracy(self) -> float | None:
        """Balanced accuracy at the applied threshold."""
        return None if self.counts is None else self.counts.balanced_accuracy

    @property
    def precision(self) -> float | None:
        """Positive predictive value at the applied threshold."""
        return None if self.counts is None else self.counts.precision

    @property
    def recall(self) -> float | None:
        """True positive rate at the applied threshold."""
        return None if self.counts is None else self.counts.recall

    @property
    def f1(self) -> float | None:
        """F1 at the applied threshold."""
        return None if self.counts is None else self.counts.f1

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping; undefined metrics serialise as :data:`NOT_AVAILABLE`."""
        return {
            "available": self.available,
            "reason": self.reason,
            "n_samples": self.n_samples,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "positive_class": SPOOF,
            "roc_auc": _number(self.roc_auc),
            "average_precision": _number(self.average_precision),
            "eer": _number(self.eer),
            "eer_threshold": _number(self.eer_threshold),
            "brier_score": _number(self.brier),
            "brier_skill_score": _number(self.brier_skill),
            "expected_calibration_error": _number(self.expected_calibration_error),
            "operating_point": (
                None
                if self.threshold is None
                else OperatingPoint(self.threshold, self.counts or _empty_counts()).to_dict()
            ),
        }


def _empty_counts() -> ConfusionCounts:
    """All-zero confusion, for a report that has a threshold but no rows."""
    return ConfusionCounts(0, 0, 0, 0)


def _number(value: float | None) -> float | str:
    """Serialise an optional measurement, marking absence explicitly."""
    return NOT_AVAILABLE if value is None else round(float(value), 6)


def _binary_arrays(
    labels: Sequence[int | str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Normalise labels and scores to parallel binary arrays.

    Args:
        labels: ``0``/``1`` indices, or ``"bona_fide"``/``"spoof"`` strings.
        scores: Detector scores. Higher means more likely spoof.

    Returns:
        ``(ys, ss, problem)``. ``ys`` is ``int8``, ``ss`` is ``float64``, and
        ``problem`` is a human-readable reason when the pair cannot be scored.

    Raises:
        ValueError: A label is neither a known index nor a known label string.
    """
    raw_labels = list(np.asarray(labels, dtype=object).ravel())
    raw_scores = np.asarray(scores, dtype=np.float64).ravel()

    if len(raw_labels) != len(raw_scores):
        return (
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.float64),
            f"length mismatch: {len(raw_labels)} labels vs {len(raw_scores)} scores",
        )

    ys = np.empty(len(raw_labels), dtype=np.int8)
    for position, raw in enumerate(raw_labels):
        if isinstance(raw, (bool, np.bool_)):
            return (
                np.empty(0, dtype=np.int8),
                np.empty(0, dtype=np.float64),
                f"row {position}: boolean labels are ambiguous; use 0/1 or label names",
            )
        if isinstance(raw, (int, np.integer)):
            if int(raw) not in (0, 1):
                return (
                    np.empty(0, dtype=np.int8),
                    np.empty(0, dtype=np.float64),
                    f"row {position}: label index {int(raw)} is not 0 (bona fide) or 1 (spoof)",
                )
            ys[position] = int(raw)
        elif isinstance(raw, str):
            try:
                ys[position] = encode_label(raw)
            except ValueError as exc:
                return (np.empty(0, dtype=np.int8), np.empty(0, dtype=np.float64), str(exc))
        else:
            return (
                np.empty(0, dtype=np.int8),
                np.empty(0, dtype=np.float64),
                f"row {position}: label of type {type(raw).__name__} is not a label",
            )

    if not np.all(np.isfinite(raw_scores)):
        bad = int(np.flatnonzero(~np.isfinite(raw_scores))[0])
        return (
            np.empty(0, dtype=np.int8),
            np.empty(0, dtype=np.float64),
            f"row {bad}: score is not finite; a NaN here would sort as a real measurement",
        )

    return ys, raw_scores, None


def _both_classes(ys: np.ndarray) -> str | None:
    """Reason the scores cannot support a two-class metric, if any."""
    n_pos = int(ys.sum())
    n_neg = len(ys) - n_pos
    if len(ys) == 0:
        return "no samples were scored"
    if n_pos == 0:
        return "split contains no spoof samples; a two-class metric is undefined"
    if n_neg == 0:
        return "split contains no bona fide samples; a two-class metric is undefined"
    return None


def confusion_at(
    ys: np.ndarray,
    ss: np.ndarray,
    threshold: float,
) -> ConfusionCounts:
    """Confusion counts for one threshold.

    Args:
        ys: Binary labels, ``1`` meaning spoof.
        ss: Scores. Higher means more likely spoof.
        threshold: Score at or above which a sample is called spoof.

    Returns:
        The four integer counts.
    """
    called = ss >= threshold
    positive = ys == 1
    return ConfusionCounts(
        true_positive=int(np.count_nonzero(called & positive)),
        false_positive=int(np.count_nonzero(called & ~positive)),
        true_negative=int(np.count_nonzero(~called & ~positive)),
        false_negative=int(np.count_nonzero(~called & positive)),
    )


def operating_points(
    labels: Sequence[int | str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> tuple[OperatingPoint, ...]:
    """Every distinct operating point, one per distinct score.

    Sweeping the distinct scores rather than a fixed grid is what makes
    selection exact: the confusion at each candidate is a real set of integer
    counts, so two runs on the same data cannot disagree because of grid
    resolution.

    Args:
        labels: ``0``/``1`` indices, or ``"bona_fide"``/``"spoof"`` strings.
        scores: Detector scores.

    Returns:
        Candidates sorted by descending threshold, each with its confusion.

    Raises:
        ValueError: Labels and scores cannot be read as binary pairs.
    """
    ys, ss, problem = _binary_arrays(labels, scores)
    if problem is not None:
        raise ValueError(problem)
    if len(ss) == 0:
        return ()
    return tuple(
        OperatingPoint(float(threshold), confusion_at(ys, ss, float(threshold)))
        for threshold in np.unique(ss)[::-1]
    )


def _sweep_axis(ys: np.ndarray, ss: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared swept-curve core: ``(tps, fps, thresholds)`` at each distinct score."""
    order = np.argsort(-ss, kind="stable")
    sorted_scores = ss[order]
    sorted_labels = ys[order]
    # Collapse ties so a curve vertex appears once per distinct score, not once
    # per sample. Otherwise a run of identical scores produces a staircase whose
    # area depends on how many identical samples there were.
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    ends = np.concatenate([boundaries, [len(sorted_scores)]])
    true_positives = np.cumsum(sorted_labels)[ends - 1]
    false_positives = ends - true_positives
    return true_positives, false_positives, sorted_scores[ends - 1]


def roc_curve_points(
    labels: Sequence[int | str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> Curve:
    """The receiver operating characteristic.

    Args:
        labels: Binary labels.
        scores: Detector scores.

    Returns:
        A :class:`Curve` of false positive rate against true positive rate,
        starting at the origin and ending at the top-right corner.

    Raises:
        ValueError: Scores cannot be read, or the split holds only one class.
    """
    ys, ss, problem = _binary_arrays(labels, scores)
    if problem is not None:
        raise ValueError(problem)
    single = _both_classes(ys)
    if single is not None:
        raise ValueError(single)

    n_pos, n_neg = int(ys.sum()), int((ys == 0).sum())
    tps, fps, thresholds = _sweep_axis(ys, ss)
    return Curve(
        x=np.concatenate([[0.0], fps / n_neg]),
        y=np.concatenate([[0.0], tps / n_pos]),
        thresholds=np.concatenate([[np.inf], thresholds]),
    )


def pr_curve_points(
    labels: Sequence[int | str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> Curve:
    """The precision/recall curve.

    Preferred over ROC when the spoof class is rare, which it is in any corpus
    built to resemble real traffic.

    Args:
        labels: Binary labels.
        scores: Detector scores.

    Returns:
        A :class:`Curve` of recall against precision.

    Raises:
        ValueError: Scores cannot be read, or the split holds no spoof samples.
    """
    ys, ss, problem = _binary_arrays(labels, scores)
    if problem is not None:
        raise ValueError(problem)
    if int(ys.sum()) == 0:
        raise ValueError("split contains no spoof samples; recall is undefined")

    n_pos = int(ys.sum())
    tps, fps, thresholds = _sweep_axis(ys, ss)
    precision = tps / np.maximum(tps + fps, 1)
    return Curve(
        x=np.concatenate([[0.0], tps / n_pos]),
        y=np.concatenate([[1.0], precision]),
        thresholds=np.concatenate([[np.inf], thresholds]),
        x_name="recall",
        y_name="precision",
    )


def det_curve_points(
    labels: Sequence[int | str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> Curve:
    """A detection error tradeoff curve in normal deviate coordinates.

    The linear ROC compresses the region that matters operationally -- the
    low-error corner -- into the last few pixels. Mapping each axis through the
    inverse normal CDF gives the axes in standard deviations, which is how the
    DET literature and every plotting library expect to read one.

    Args:
        labels: Binary labels.
        scores: Detector scores.

    Returns:
        A :class:`Curve` of false reject rate against false accept rate, both
        mapped through the inverse normal CDF. The extreme endpoints are clipped
        to keep the transform finite.

    Raises:
        ValueError: Scores cannot be read, or the split holds only one class.
    """
    ys, ss, problem = _binary_arrays(labels, scores)
    if problem is not None:
        raise ValueError(problem)
    single = _both_classes(ys)
    if single is not None:
        raise ValueError(single)

    n_pos, n_neg = int(ys.sum()), int((ys == 0).sum())
    tps, fps, thresholds = _sweep_axis(ys, ss)
    false_reject = 1.0 - tps / n_pos
    false_accept = fps / n_neg
    return Curve(
        x=_to_deviate(false_reject),
        y=_to_deviate(false_accept),
        thresholds=np.concatenate([[np.inf], thresholds]),
        x_name="false_reject_rate",
        y_name="false_accept_rate",
        units="normal_deviate",
    )


def _to_deviate(rates: np.ndarray) -> np.ndarray:
    """Map rates onto normal deviate axes, clipping the degenerate endpoints.

    ``0.0`` and ``1.0`` have no finite deviate, and a DET curve's endpoints are
    exactly those, so they are pulled in to ``+/-6`` standard deviations. That is
    the usual convention and it keeps the axis finite without affecting any
    measurement, which lives in the interior.
    """
    eps = 1e-6
    return ndtri(np.clip(rates, eps, 1.0 - eps))


def roc_auc(
    labels: Sequence[int | str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float | None:
    """Area under the ROC curve, computed from average ranks.

    The rank formulation (Mann-Whitney U) is used instead of trapezoids over a
    sampled curve because it is exact on tied scores. Two spoof samples that
    both score ``0.90`` must not be separated by trapezoid geometry that depends
    on their order.

    Args:
        labels: Binary labels.
        scores: Detector scores.

    Returns:
        The AUC, or ``None`` when the split holds only one class.
    """
    ys, ss, problem = _binary_arrays(labels, scores)
    if problem is not None or _both_classes(ys) is not None:
        return None

    n_pos, n_neg = int(ys.sum()), int((ys == 0).sum())
    order = np.argsort(ss, kind="stable")
    sorted_scores = ss[order]
    sorted_labels = ys[order]
    ranks = _average_ranks(sorted_scores)
    positive_rank_sum = float(ranks[sorted_labels == 1].sum())
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_ranks(sorted_scores: np.ndarray) -> np.ndarray:
    """Ranks of an ascending array, ties sharing their mean rank."""
    total = len(sorted_scores)
    ranks = np.arange(1, total + 1, dtype=np.float64)
    if total < 2:
        return ranks
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [total]])
    for start, end in zip(starts, ends, strict=True):
        ranks[start:end] = (start + end + 1) / 2.0
    return ranks


def average_precision(
    labels: Sequence[int | str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> float | None:
    """Step-wise area under the precision/recall curve.

    Args:
        labels: Binary labels.
        scores: Detector scores.

    Returns:
        Average precision, or ``None`` when the split holds no spoof samples.
    """
    ys, ss, problem = _binary_arrays(labels, scores)
    if problem is not None or int(ys.sum()) == 0:
        return None

    n_pos = int(ys.sum())
    tps, fps, _ = _sweep_axis(ys, ss)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / n_pos
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def eer(
    labels: Sequence[int | str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
) -> tuple[float | None, float | None]:
    """Equal error rate and the threshold that achieves it.

    The ROC and its mirror cross between vertices, so the crossing is located by
    linear interpolation rather than by reporting the nearer vertex. Reporting a
    vertex would inflate the EER and quietly understate the false reject rate.

    Args:
        labels: Binary labels.
        scores: Detector scores.

    Returns:
        ``(eer, threshold)``, or ``(None, None)`` when the split holds only one
        class or the curve never crosses.
    """
    ys, ss, problem = _binary_arrays(labels, scores)
    if problem is not None or _both_classes(ys) is not None:
        return (None, None)

    curve = roc_curve_points(ys, ss)
    false_accept = curve.x
    false_reject = 1.0 - curve.y
    gap = false_reject - false_accept

    # An exact tie is the *common* case for a well-separated detector, and it
    # must not be missed: a perfect separator has gap == 0 at the vertex where
    # every spoof is caught and no bona fide is caught, and requiring a strict
    # sign change would report "no EER available" for a model with zero errors.
    # That is the most reassuring result available and therefore the one least
    # likely to be questioned if it is wrong.
    tolerance = 1e-12
    ties = np.flatnonzero(np.abs(gap) <= tolerance)
    if len(ties):
        index = int(ties[0])
        return (
            float(false_accept[index]),
            float(min(curve.thresholds[index], _MAX_THRESHOLD)),
        )

    crossings = np.flatnonzero(np.sign(gap[:-1]) * np.sign(gap[1:]) < 0)
    if len(crossings) == 0:
        # One error rate dominates the other at every threshold, so there is no
        # equal-error point to report. Distinct from an EER of 0.
        return (None, None)

    index = int(crossings[0])
    lower, upper = float(gap[index]), float(gap[index + 1])
    fraction = 0.0 if upper == lower else -lower / (upper - lower)
    rate = float(false_accept[index] + fraction * (false_accept[index + 1] - false_accept[index]))
    # The sentinel at index 0 is not a usable operating point, so the reported
    # threshold is the finer of the two bracketing finite scores.
    bracket = [
        float(value)
        for value in (curve.thresholds[index], curve.thresholds[index + 1])
        if np.isfinite(value)
    ]
    if not bracket:
        return (rate, None)
    return (rate, min(bracket))


def brier_score(
    labels: Sequence[int | str] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> float | None:
    """Mean squared error of predicted spoof probabilities.

    Args:
        labels: Binary labels.
        probabilities: Values in ``[0, 1]`` meaning P(spoof). Unlike a score,
            a probability here is required, because the metric is only meaningful
            when the number claims to be one.

    Returns:
        The Brier score, or ``None`` for empty input.

    Raises:
        ValueError: A value lies outside ``[0, 1]``.
    """
    ys, ps, problem = _binary_arrays(labels, probabilities)
    if problem is not None:
        raise ValueError(problem)
    if len(ps) == 0:
        return None
    if np.any((ps < 0.0) | (ps > 1.0)):
        msg = "calibration metrics require probabilities in [0, 1]; a raw score is not one"
        raise ValueError(msg)
    return float(np.mean(np.square(ps - ys.astype(np.float64))))


def brier_skill_score(
    labels: Sequence[int | str] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> float | None:
    """Brier score relative to a constant base-rate forecast.

    Args:
        labels: Binary labels.
        probabilities: Values in ``[0, 1]`` meaning P(spoof).

    Returns:
        ``1 - brier / brier_reference``, or ``None`` when the reference is
        degenerate. Zero means no better than always predicting the base rate;
        negative means worse, which is worth reporting rather than hiding.
    """
    observed = brier_score(labels, probabilities)
    if observed is None:
        return None
    ys, ps, _ = _binary_arrays(labels, probabilities)
    reference = float(np.mean(np.square(ps.mean() - ys.astype(np.float64))))
    if reference == 0.0:
        return None
    return 1.0 - observed / reference


def reliability_bins(
    labels: Sequence[int | str] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> tuple[ReliabilityBin, ...]:
    """Group predicted probabilities into equal-width bins.

    Args:
        labels: Binary labels.
        probabilities: Values in ``[0, 1]`` meaning P(spoof).
        n_bins: Number of equal-width bins.

    Returns:
        Bins in ascending probability order, including empty ones so that a gap
        in the predictions is visible rather than silently compressed out.

    Raises:
        ValueError: Fewer than one bin is requested, or a value leaves ``[0, 1]``.
    """
    if n_bins < 1:
        msg = f"n_bins must be positive, got {n_bins}"
        raise ValueError(msg)
    ys, ps, problem = _binary_arrays(labels, probabilities)
    if problem is not None:
        raise ValueError(problem)
    if len(ps) == 0:
        return ()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # The last bin is closed on the right so that a probability of exactly 1.0
    # is counted rather than dropped by a strict upper-edge comparison.
    assignment = np.clip(np.digitize(ps, edges[1:-1], right=False), 0, n_bins - 1)
    bins: list[ReliabilityBin] = []
    for index in range(n_bins):
        selected = assignment == index
        count = int(np.count_nonzero(selected))
        if count == 0:
            mean_predicted = (edges[index] + edges[index + 1]) / 2.0
            observed = float("nan")
        else:
            mean_predicted = float(ps[selected].mean())
            observed = float(ys[selected].mean())
        bins.append(
            ReliabilityBin(
                lower=float(edges[index]),
                upper=float(edges[index + 1]),
                count=count,
                mean_predicted=mean_predicted,
                observed_fraction=observed,
            )
        )
    return tuple(bins)


def expected_calibration_error(
    labels: Sequence[int | str] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> float | None:
    """Sample-weighted mean gap between predicted and observed rate.

    Args:
        labels: Binary labels.
        probabilities: Values in ``[0, 1]`` meaning P(spoof).
        n_bins: Number of equal-width bins.

    Returns:
        The error, or ``None`` when there is nothing to score.
    """
    bins = reliability_bins(labels, probabilities, n_bins=n_bins)
    populated = [item for item in bins if item.count > 0]
    if not populated:
        return None
    total = sum(item.count for item in populated)
    return float(
        sum(item.count * abs(item.mean_predicted - item.observed_fraction) for item in populated)
        / total
    )


def score_bands(
    probabilities: Sequence[float] | np.ndarray,
    boundaries: Sequence[float] = (0.5, 0.8),
) -> dict[str, int]:
    """Count predictions in each policy risk band.

    The bands come from :mod:`voxshield.policy.evaluator`, where they are
    placeholders rather than calibrated operating points. Reporting the counts
    makes it visible how much traffic would be routed to review once a threshold
    is actually chosen.

    Args:
        probabilities: Values in ``[0, 1]`` meaning P(spoof).
        boundaries: Ascending band edges, excluding the implicit 0 and 1.

    Returns:
        Counts keyed ``"low"``, ``"medium"``, ``"high"``.

    Raises:
        ValueError: Boundaries are not strictly ascending, or a value leaves
            ``[0, 1]``.
    """
    edges = np.asarray(boundaries, dtype=np.float64)
    if edges.size and not np.all(np.diff(edges) > 0):
        msg = f"band boundaries must ascend strictly, got {boundaries!r}"
        raise ValueError(msg)
    values = np.asarray(probabilities, dtype=np.float64).ravel()
    if values.size and np.any((values < 0.0) | (values > 1.0)):
        msg = "risk bands require probabilities in [0, 1]"
        raise ValueError(msg)
    index = np.digitize(values, edges, right=False)
    names = ["low", "medium", "high"]
    return {names[position]: int(np.count_nonzero(index == position)) for position in range(3)}


def select_threshold(
    labels: Sequence[int | str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    policy: ThresholdPolicy | None = None,
) -> OperatingPoint | None:
    """Choose an operating point from dev scores under an explicit policy.

    Call this on dev and pass the resulting threshold to
    :func:`detect` unchanged. Selecting here and scoring there is the only
    supported way to produce a test-set operating point.

    Args:
        labels: Dev labels.
        scores: Dev scores.
        policy: Selection rule and constraints. Defaults to minimising EER.

    Returns:
        The chosen point, or ``None`` when nothing is selectable -- an empty
        split, or one where every candidate violates a stated ceiling. Returning
        ``None`` rather than the least-bad candidate is deliberate: a threshold
        that misses its own constraint must not be applied silently.
    """
    chosen_policy = policy or ThresholdPolicy()
    ys, ss, problem = _binary_arrays(labels, scores)
    if problem is not None or len(ss) == 0:
        return None

    if chosen_policy.strategy == "fixed":
        pinned = chosen_policy.fixed_threshold
        if pinned is None:
            # Unreachable: __post_init__ rejects this combination. Guarded rather
            # than asserted so library code carries no assertion that a stripped
            # build could remove.
            return None
        return OperatingPoint(float(pinned), confusion_at(ys, ss, float(pinned)))

    candidates = [point for point in operating_points(ys, ss) if point.feasible]
    if not candidates:
        return None

    feasible = []
    for point in candidates:
        far, frr = point.counts.far, point.counts.frr
        if chosen_policy.max_far is not None and (far is None or far > chosen_policy.max_far):
            continue
        if chosen_policy.max_frr is not None and (frr is None or frr > chosen_policy.max_frr):
            continue
        feasible.append(point)

    if not feasible:
        return None

    if chosen_policy.strategy == "eer":
        equal = _nearest_to_eer(feasible)
        if equal is not None:
            return equal
    if chosen_policy.strategy == "youden":
        return _best(feasible, lambda p: -(p.counts.recall - p.counts.specificity))
    if chosen_policy.strategy == "max_recall":
        return _best(feasible, lambda p: -(p.counts.recall or 0.0))
    return _best(feasible, lambda p: p.counts.misclassification_count)


def _nearest_to_eer(candidates: Sequence[OperatingPoint]) -> OperatingPoint | None:
    """The feasible candidate closest to ``far == frr``, lowest FAR breaking ties."""
    scored = [
        (abs((point.counts.far or 0.0) - (point.counts.frr or 0.0)), point.counts.far or 0.0, point)
        for point in candidates
    ]
    scored = [row for row in scored if row[2].counts.far is not None]
    if not scored:
        return None
    return min(scored, key=lambda row: (round(row[0], 12), row[1]))[2]


def _best(
    candidates: Sequence[OperatingPoint],
    cost: Any,
) -> OperatingPoint:
    """Pick the lowest-cost candidate, highest threshold breaking ties.

    Preferring the higher threshold on a tie keeps the rule stable and biases
    towards fewer false alarms, which is the cheaper error to make for an
    advisory tool.
    """
    return min(candidates, key=lambda point: (cost(point), -point.threshold))


def detect(
    labels: Sequence[int | str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    threshold: float | None = None,
    n_calibration_bins: int = 10,
) -> BinaryMetrics:
    """Score one split.

    Args:
        labels: Labels for the split.
        scores: Detector scores, or probabilities when calibration is wanted.
        threshold: Operating point chosen on dev. Omit it for a curve-only view;
            operating-point fields then report ``None`` rather than being derived
            from the split under test.
        n_calibration_bins: Bin count for the calibration error.

    Returns:
        A :class:`BinaryMetrics`. Unusable input yields a report with a reason
        and no numbers, rather than an exception, so that one bad subgroup
        cannot abort a whole evaluation.
    """
    ys, ss, problem = _binary_arrays(labels, scores)
    if problem is not None:
        return BinaryMetrics(
            n_samples=0,
            n_positive=0,
            n_negative=0,
            reason=problem,
        )

    n_pos, n_neg = int(ys.sum()), int((ys == 0).sum())
    if len(ss) == 0:
        return BinaryMetrics(
            n_samples=0,
            n_positive=0,
            n_negative=0,
            reason="no samples were scored",
        )

    single_class = _both_classes(ys)
    if single_class is not None:
        # One class is not an error: it is a real finding about a real split.
        # The confusion matrix is still meaningful, so the report keeps the
        # counts and the operating point and withholds only the two-class
        # metrics, which is what an EER of "how often this one generator is
        # caught" would quietly be.
        counts = None if threshold is None else confusion_at(ys, ss, float(threshold))
        return BinaryMetrics(
            n_samples=len(ss),
            n_positive=n_pos,
            n_negative=n_neg,
            threshold=threshold,
            counts=counts,
            reason=single_class,
        )

    counts = None if threshold is None else confusion_at(ys, ss, float(threshold))
    equal_rate, equal_threshold = eer(ys, ss)
    calibration_error: float | None
    brier: float | None
    skill: float | None
    try:
        calibration_error = expected_calibration_error(ys, ss, n_bins=n_calibration_bins)
        brier = brier_score(ys, ss)
        skill = brier_skill_score(ys, ss)
    except ValueError:
        # Raw scores are not probabilities. That is normal for a model reporting
        # a margin, so calibration is reported as absent rather than the whole
        # evaluation failing.
        calibration_error, brier, skill = None, None, None

    return BinaryMetrics(
        n_samples=len(ss),
        n_positive=n_pos,
        n_negative=n_neg,
        threshold=threshold,
        counts=counts,
        roc_auc=roc_auc(ys, ss),
        average_precision=average_precision(ys, ss),
        eer=equal_rate,
        eer_threshold=equal_threshold,
        brier=brier,
        brier_skill=skill,
        expected_calibration_error=calibration_error,
    )


def subgroup_scores(
    labels: Sequence[int | str] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    groups: Mapping[str, Sequence[Any]],
) -> dict[str, tuple[list[Any], list[Any]]]:
    """Align group labels with scores, dropping rows whose group is absent.

    Args:
        labels: Binary labels.
        scores: Detector scores.
        groups: Row-aligned attribute values, e.g. ``generator_id``. Missing
            values may be ``None``.

    Returns:
        ``{value: (labels, scores)}`` for each value present, in sorted order so
        that a report is byte-identical across runs. Values absent from a row are
        excluded rather than bucketed as an ``"unknown"`` subgroup, because an
        absent attribute says nothing about the audio.
    """
    ys, ss, problem = _binary_arrays(labels, scores)
    if problem is not None:
        return {}

    usable: dict[str, list[int]] = {}
    for name, values in groups.items():
        aligned = list(values)
        if len(aligned) != len(ss):
            msg = f"group {name!r} has {len(aligned)} values for {len(ss)} scores"
            raise ValueError(msg)
        usable[name] = [index for index, value in enumerate(aligned) if value is not None]

    buckets: dict[str, tuple[list[Any], list[Any]]] = {}
    for name, indices in usable.items():
        aligned = list(groups[name])
        for index in indices:
            key = f"{name}={aligned[index]}"
            existing_labels, existing_scores = buckets.get(key, ([], []))
            existing_labels.append(ys[index].item())
            existing_scores.append(float(ss[index]))
            buckets[key] = (existing_labels, existing_scores)
    return dict(sorted(buckets.items()))


def describe_labels(labels: Sequence[int | str] | np.ndarray) -> dict[str, Any]:
    """Summarise a label vector, for inclusion next to its metrics.

    Args:
        labels: Binary labels or label names.

    Returns:
        Counts and the spoof prevalence, which is the number every reader needs
        in order to judge an accuracy figure.
    """
    ys, _, problem = _binary_arrays(labels, np.zeros(len(list(labels))))
    if problem is not None:
        return {"error": problem}
    n_pos = int(ys.sum())
    n_neg = len(ys) - n_pos
    return {
        "n_samples": len(ys),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "spoof_prevalence": _number(n_pos / len(ys) if len(ys) else None),
        "positive_class": SPOOF,
        "negative_class": BONA_FIDE,
    }
