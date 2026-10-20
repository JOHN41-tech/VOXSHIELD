import numpy as np

from voxshield.prosody.analysis_service import analyze_prosody
from voxshield.prosody.baseline_service import LocalBaselineStore, enroll_baseline
from voxshield.prosody.config import ProsodyConfig
from voxshield.prosody.schema import AnalysisStatus, BaselineComparisonStatus


def _tone(frequency: float = 130.0, seconds: float = 1.2) -> np.ndarray:
    t = np.arange(round(16_000 * seconds), dtype=np.float32) / 16_000
    return (0.25 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_baseline_free_analysis_is_completed_without_comparison(tmp_path) -> None:
    result = analyze_prosody(_tone(), 16_000)
    assert result.status is AnalysisStatus.COMPLETED
    assert result.baseline_comparison_status is BaselineComparisonStatus.NOT_REQUESTED
    assert result.features is not None
    assert result.latency_ms >= 0.0
    assert result.as_dict()["features"] is not None


def test_explicit_baseline_is_loaded_and_compared(tmp_path) -> None:
    config = ProsodyConfig(minimum_baseline_samples=3)
    store = LocalBaselineStore(tmp_path)
    samples = [
        analyze_prosody(_tone(130 + shift), 16_000, config=config).features
        for shift in (-2, 0, 2)
    ]
    baseline = enroll_baseline([item for item in samples if item is not None], config)
    store.create(baseline)
    result = analyze_prosody(
        _tone(180), 16_000, baseline_id=baseline.baseline_id, config=config, baseline_store=store
    )
    assert result.baseline_comparison_status is BaselineComparisonStatus.AVAILABLE
    assert result.baseline_id == baseline.baseline_id


def test_silence_is_inconclusive_not_failed() -> None:
    result = analyze_prosody(np.zeros(16_000, dtype=np.float32), 16_000)
    assert result.status is AnalysisStatus.INCONCLUSIVE
    assert result.error is None


def test_wrong_sample_rate_is_operational_failure() -> None:
    result = analyze_prosody(_tone(), 8_000)
    assert result.status is AnalysisStatus.FAILED
    assert result.error is not None
