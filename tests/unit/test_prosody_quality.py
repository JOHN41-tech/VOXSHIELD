import numpy as np

from voxshield.prosody.feature_extractor import extract_prosodic_features
from voxshield.prosody.quality_checks import assess_prosody_quality
from voxshield.prosody.schema import QualityStatus


def test_silence_is_inconclusive() -> None:
    extraction = extract_prosodic_features(np.zeros(16_000, dtype=np.float32), 16_000)
    quality = assess_prosody_quality(extraction)
    assert quality.status is QualityStatus.INCONCLUSIVE
    assert "SILENCE_ONLY" in quality.warnings


def test_short_signal_is_inconclusive() -> None:
    samples = np.sin(2 * np.pi * 130 * np.arange(2_000) / 16_000).astype(np.float32)
    extraction = extract_prosodic_features(samples, 16_000)
    quality = assess_prosody_quality(extraction)
    assert quality.status is QualityStatus.INCONCLUSIVE
    assert "SHORT_RECORDING" in quality.warnings


def test_clipping_is_warning_when_other_evidence_exists() -> None:
    samples = np.sin(2 * np.pi * 130 * np.arange(20_000) / 16_000).astype(np.float32)
    samples[::100] = 1.0
    extraction = extract_prosodic_features(samples, 16_000)
    quality = assess_prosody_quality(extraction)
    assert "CLIPPING" in quality.warnings
    assert quality.status in {QualityStatus.WARN, QualityStatus.INCONCLUSIVE}
