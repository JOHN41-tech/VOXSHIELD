from dataclasses import replace

import pytest

from voxshield.prosody.baseline_service import (
    BaselineNotFoundError,
    LocalBaselineStore,
    enroll_baseline,
)
from voxshield.prosody.config import ProsodyConfig
from voxshield.prosody.schema import FeatureValue, ProsodyFeatures


def _features(offset: float = 0.0) -> ProsodyFeatures:
    names = {
        "pitch_median_hz": (120.0 + offset, "Hz"),
        "pitch_std_hz": (10.0 + offset, "Hz"),
        "energy_median_rms": (0.1 + offset / 1000, "linear RMS"),
        "speech_activity_ratio": (0.7, "ratio"),
    }
    return ProsodyFeatures(values={name: FeatureValue(value, unit) for name, (value, unit) in names.items()})


def test_enrollment_requires_multiple_samples() -> None:
    with pytest.raises(ValueError, match="at least"):
        enroll_baseline([_features(), _features(1.0)])


def test_baseline_store_lifecycle_is_pseudonymous(tmp_path) -> None:
    config = ProsodyConfig(minimum_baseline_samples=3)
    baseline = enroll_baseline([_features(), _features(1.0), _features(-1.0)], config)
    store = LocalBaselineStore(tmp_path)
    store.create(baseline)
    loaded = store.get(baseline.baseline_id)
    assert loaded.distributions["pitch_median_hz"].count == 3
    assert next(iter(tmp_path.glob("*.json"))).read_text().find("raw") == -1
    inactive = store.deactivate(baseline.baseline_id)
    assert inactive.active is False
    assert store.delete(baseline.baseline_id) is True
    with pytest.raises(BaselineNotFoundError):
        store.get(baseline.baseline_id)


def test_baseline_updates_require_version_increment(tmp_path) -> None:
    baseline = enroll_baseline([_features(), _features(1.0), _features(-1.0)])
    store = LocalBaselineStore(tmp_path)
    store.create(baseline)
    with pytest.raises(Exception, match="increment"):
        store.update(replace(baseline, limitations=("changed",)))
