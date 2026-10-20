import math

import pytest

from voxshield.prosody.config import ProsodyConfig
from voxshield.prosody.schema import FeatureValue, ProsodyFeatures


def test_feature_value_is_json_safe() -> None:
    feature = FeatureValue(120.0, "Hz")
    assert feature.as_dict()["value"] == 120.0


def test_feature_value_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        FeatureValue(math.nan, "Hz")


def test_unavailable_feature_requires_reason() -> None:
    with pytest.raises(ValueError):
        FeatureValue(None, "Hz", valid=False)


def test_features_expose_only_valid_numeric_values() -> None:
    features = ProsodyFeatures(
        values={
            "pitch": FeatureValue(120.0, "Hz"),
            "jitter": FeatureValue(None, "ratio", valid=False, reason="not reliable"),
        }
    )
    assert features.valid_values == {"pitch": 120.0}
    assert features.as_dict()["features"]["jitter"]["value"] is None


def test_configuration_rejects_invalid_frame_order() -> None:
    with pytest.raises(ValueError, match="hop_ms"):
        ProsodyConfig(frame_ms=10.0, hop_ms=20.0)


def test_configuration_fingerprint_is_stable() -> None:
    assert "sr=16000" in ProsodyConfig().extraction_fingerprint
