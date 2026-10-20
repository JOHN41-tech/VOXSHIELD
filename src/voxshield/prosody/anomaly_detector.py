"""Explainable deviation analysis relative to an enrolled baseline."""

from __future__ import annotations

from dataclasses import dataclass

from voxshield.prosody.baseline_service import BehavioralBaseline
from voxshield.prosody.config import ProsodyConfig
from voxshield.prosody.schema import BaselineComparisonStatus, ProsodyFeatures


@dataclass(frozen=True, slots=True)
class FeatureDeviation:
    feature_name: str
    observed_value: float
    reference_median: float | None
    reference_mean: float | None
    deviation: float | None
    outside_reference: bool
    confidence: str
    explanation: str

    def as_dict(self) -> dict[str, object]:
        return {
            "feature_name": self.feature_name,
            "observed_value": self.observed_value,
            "reference_median": self.reference_median,
            "reference_mean": self.reference_mean,
            "deviation": self.deviation,
            "outside_reference": self.outside_reference,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class DeviationAnalysis:
    status: BaselineComparisonStatus
    baseline_id: str | None
    deviations: tuple[FeatureDeviation, ...]
    warnings: tuple[str, ...]
    summary: str


def compare_to_baseline(
    features: ProsodyFeatures,
    baseline: BehavioralBaseline,
    config: ProsodyConfig | None = None,
) -> DeviationAnalysis:
    cfg = config or ProsodyConfig()
    if not baseline.compatible_with(features, cfg):
        return DeviationAnalysis(
            BaselineComparisonStatus.INCONCLUSIVE,
            baseline.baseline_id,
            (),
            ("INCOMPATIBLE_BASELINE",),
            "Baseline and current feature schema or extraction configuration are incompatible.",
        )
    deviations: list[FeatureDeviation] = []
    missing = 0
    outside = 0
    for name, observed in sorted(features.valid_values.items()):
        reference = baseline.distributions.get(name)
        if reference is None:
            missing += 1
            continue
        if reference.count < cfg.minimum_feature_observations:
            continue
        scale = max(reference.std, 1.4826 * reference.mad, 1e-9)
        robust_deviation = abs(float(observed) - reference.median) / scale
        is_outside = robust_deviation >= cfg.deviation_mad_threshold
        if is_outside:
            outside += 1
        deviations.append(
            FeatureDeviation(
                feature_name=name,
                observed_value=float(observed),
                reference_median=reference.median,
                reference_mean=reference.mean,
                deviation=robust_deviation,
                outside_reference=is_outside,
                confidence="HIGH" if reference.count >= 10 else "LIMITED",
                explanation=(
                    f"{name} differs from this speaker's enrolled baseline."
                    if is_outside
                    else f"{name} is within the observed reference variation."
                ),
            )
        )
    if not deviations:
        return DeviationAnalysis(
            BaselineComparisonStatus.INCONCLUSIVE,
            baseline.baseline_id,
            (),
            ("NO_COMPARABLE_FEATURES",),
            "Insufficient reference data to assess the available features.",
        )
    warnings = ("MISSING_BASELINE_FEATURES",) if missing else ()
    summary = (
        f"{outside} of {len(deviations)} evaluated features are outside the configured "
        "reference variation; this indicates behavioral difference, not intent."
    )
    return DeviationAnalysis(
        BaselineComparisonStatus.AVAILABLE,
        baseline.baseline_id,
        tuple(deviations),
        warnings,
        summary,
    )
