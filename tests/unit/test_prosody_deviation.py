from dataclasses import replace

from voxshield.prosody.anomaly_detector import compare_to_baseline
from voxshield.prosody.baseline_service import enroll_baseline
from voxshield.prosody.config import ProsodyConfig
from voxshield.prosody.schema import BaselineComparisonStatus, FeatureValue, ProsodyFeatures


def _features(pitch: float) -> ProsodyFeatures:
    return ProsodyFeatures(
        values={
            "pitch_median_hz": FeatureValue(pitch, "Hz"),
            "energy_median_rms": FeatureValue(0.1, "linear RMS"),
            "speech_activity_ratio": FeatureValue(0.7, "ratio"),
        },
    )


def test_outlier_deviation_is_explainable() -> None:
    config = ProsodyConfig(minimum_baseline_samples=3)
    baseline = enroll_baseline([_features(120), _features(121), _features(119)], config)
    result = compare_to_baseline(_features(180), baseline, config)
    assert result.status is BaselineComparisonStatus.AVAILABLE
    pitch = next(item for item in result.deviations if item.feature_name == "pitch_median_hz")
    assert pitch.outside_reference is True
    assert "enrolled baseline" in pitch.explanation
    assert "cloning" not in result.summary.lower()


def test_incompatible_schema_is_inconclusive() -> None:
    config = ProsodyConfig(minimum_baseline_samples=3)
    baseline = enroll_baseline([_features(120), _features(121), _features(119)], config)
    result = compare_to_baseline(
        _features(120), replace(baseline, extraction_fingerprint="different"), config
    )
    assert result.status is BaselineComparisonStatus.INCONCLUSIVE
    assert "INCOMPATIBLE_BASELINE" in result.warnings


def test_small_reference_is_not_treated_as_anomaly() -> None:
    config = ProsodyConfig(minimum_baseline_samples=3, minimum_feature_observations=10)
    baseline = enroll_baseline(
        [_features(120), _features(121), _features(119)], ProsodyConfig(minimum_baseline_samples=3)
    )
    result = compare_to_baseline(_features(180), baseline, config)
    assert result.status is BaselineComparisonStatus.INCONCLUSIVE
