import numpy as np
import pytest

from voxshield.prosody.feature_extractor import extract_prosodic_features


def _tone(seconds: float = 1.2, frequency: float = 140.0) -> np.ndarray:
    t = np.arange(round(16_000 * seconds), dtype=np.float32) / 16_000
    return (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_extracts_finite_pitch_and_energy_features() -> None:
    result = extract_prosodic_features(_tone(), 16_000)
    values = result.features.valid_values
    assert result.features.duration_seconds == pytest.approx(1.2, abs=0.01)
    assert 100.0 < values["pitch_median_hz"] < 180.0
    assert values["energy_median_rms"] > 0.0
    assert all(np.isfinite(float(value)) for value in values.values())


def test_empty_audio_is_reported_without_fabricated_pitch() -> None:
    result = extract_prosodic_features(np.zeros(0, dtype=np.float32), 16_000)
    assert result.features.values["pitch_median_hz"].valid is False
    assert "INSUFFICIENT_VOICED_SPEECH" in result.features.warnings


def test_rejects_stereo_and_wrong_rate() -> None:
    with pytest.raises(ValueError, match="mono"):
        extract_prosodic_features(np.zeros((2, 100), dtype=np.float32), 16_000)
    with pytest.raises(ValueError, match="requires 16000"):
        extract_prosodic_features(_tone(), 8_000)


def test_silence_does_not_create_valid_pitch() -> None:
    result = extract_prosodic_features(np.zeros(16_000, dtype=np.float32), 16_000)
    assert not result.features.values["pitch_mean_hz"].valid
    assert result.features.values["speech_activity_ratio"].value == 0.0
