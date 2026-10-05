"""Phase 3 evaluation: metrics, operating points, and honest reporting.

Three modules, in dependency order:

* :mod:`~voxshield.evaluation.metrics` -- the measurements themselves. Pure
  NumPy and SciPy, no model dependency, so a metric can be checked without a
  trained model existing.
* :mod:`~voxshield.evaluation.report` -- assembling measurements into a report
  that distinguishes *measured*, *not run*, and *unavailable*.

The distinction this package exists to preserve: a metric that was never
computed and a metric that does not exist both render as absent, and collapsing
them would let a report imply evidence that was never gathered. Every figure
carries a status, and this repository's current status for every figure is
``not_run``, because no corpus is present.
"""

from __future__ import annotations

from voxshield.evaluation.metrics import (
    NOT_AVAILABLE,
    BinaryMetrics,
    ConfusionCounts,
    Curve,
    OperatingPoint,
    ReliabilityBin,
    ThresholdPolicy,
    average_precision,
    brier_score,
    brier_skill_score,
    confusion_at,
    describe_labels,
    detect,
    eer,
    expected_calibration_error,
    operating_points,
    pr_curve_points,
    reliability_bins,
    roc_auc,
    roc_curve_points,
    score_bands,
    select_threshold,
    subgroup_scores,
)
from voxshield.evaluation.report import (
    NOT_RUN,
    EvaluationReport,
    EvaluationStatus,
    LatencyReport,
    SubgroupReport,
    build_report,
    calibration_summary,
    describe_operating_point,
    not_run_report,
    operating_point_at,
)

__all__ = [
    "NOT_AVAILABLE",
    "NOT_RUN",
    "BinaryMetrics",
    "ConfusionCounts",
    "Curve",
    "EvaluationReport",
    "EvaluationStatus",
    "LatencyReport",
    "OperatingPoint",
    "ReliabilityBin",
    "SubgroupReport",
    "ThresholdPolicy",
    "average_precision",
    "brier_score",
    "brier_skill_score",
    "build_report",
    "calibration_summary",
    "confusion_at",
    "describe_labels",
    "describe_operating_point",
    "detect",
    "eer",
    "expected_calibration_error",
    "not_run_report",
    "operating_point_at",
    "operating_points",
    "pr_curve_points",
    "reliability_bins",
    "roc_auc",
    "roc_curve_points",
    "score_bands",
    "select_threshold",
    "subgroup_scores",
]
