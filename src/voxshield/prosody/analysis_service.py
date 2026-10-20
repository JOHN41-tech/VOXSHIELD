"""Stable orchestration interface for Phase 6 prosody analysis."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from voxshield.prosody.anomaly_detector import DeviationAnalysis, compare_to_baseline
from voxshield.prosody.baseline_service import (
    BaselineError,
    BaselineNotFoundError,
    BehavioralBaseline,
    LocalBaselineStore,
)
from voxshield.prosody.config import ProsodyConfig
from voxshield.prosody.feature_extractor import extract_prosodic_features
from voxshield.prosody.quality_checks import assess_prosody_quality
from voxshield.prosody.schema import (
    AnalysisStatus,
    BaselineComparisonStatus,
    ProsodyFeatures,
    QualityStatus,
)


@dataclass(frozen=True, slots=True)
class ProsodyAnalysisResult:
    analysis_id: str
    status: AnalysisStatus
    quality_status: QualityStatus
    feature_schema_version: str
    features: ProsodyFeatures | None
    baseline_comparison_status: BaselineComparisonStatus
    baseline_id: str | None
    feature_deviations: tuple[dict[str, object], ...]
    deviation_summary: str | None
    warnings: tuple[str, ...]
    latency_ms: float
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis_id": self.analysis_id,
            "status": self.status.value,
            "quality_status": self.quality_status.value,
            "feature_schema_version": self.feature_schema_version,
            "features": None if self.features is None else self.features.as_dict(),
            "baseline_comparison": {
                "status": self.baseline_comparison_status.value,
                "baseline_id": self.baseline_id,
                "deviation_summary": self.deviation_summary,
            },
            "feature_deviations": list(self.feature_deviations),
            "warnings": list(self.warnings),
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


def _empty_result(
    started: float,
    status: AnalysisStatus,
    quality: QualityStatus,
    warnings: tuple[str, ...],
    error: str | None = None,
) -> ProsodyAnalysisResult:
    return ProsodyAnalysisResult(
        analysis_id=str(uuid.uuid4()),
        status=status,
        quality_status=quality,
        feature_schema_version="1.0",
        features=None,
        baseline_comparison_status=BaselineComparisonStatus.NOT_REQUESTED,
        baseline_id=None,
        feature_deviations=(),
        deviation_summary=None,
        warnings=warnings,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        error=error,
    )


def analyze_prosody(
    audio: np.ndarray,
    sample_rate: int,
    speaker_profile_id: str | None = None,
    baseline_id: str | None = None,
    *,
    config: ProsodyConfig | None = None,
    baseline_store: LocalBaselineStore | None = None,
) -> ProsodyAnalysisResult:
    """Analyze one in-memory waveform without enrolling or storing raw audio."""
    del speaker_profile_id  # Reserved correlation key; Phase 5 embeddings are separate.
    started = time.perf_counter()
    cfg = config or ProsodyConfig()
    try:
        arr = np.asarray(audio)
        if arr.ndim != 1:
            return _empty_result(
                started,
                AnalysisStatus.FAILED,
                QualityStatus.INCONCLUSIVE,
                ("MONO_AUDIO_REQUIRED",),
                "audio must be a mono 1-D waveform",
            )
        if sample_rate != cfg.sample_rate_hz:
            return _empty_result(
                started,
                AnalysisStatus.FAILED,
                QualityStatus.INCONCLUSIVE,
                ("INVALID_SAMPLE_RATE",),
                f"expected sample rate {cfg.sample_rate_hz} Hz",
            )
        if not np.all(np.isfinite(arr)):
            return _empty_result(
                started,
                AnalysisStatus.INCONCLUSIVE,
                QualityStatus.INCONCLUSIVE,
                ("NON_FINITE_SAMPLES",),
                "audio quality prevents reliable feature extraction",
            )
        extraction = extract_prosodic_features(arr, sample_rate, cfg)
        quality = assess_prosody_quality(extraction, cfg)
        warnings = list(quality.warnings)
        comparison_status = BaselineComparisonStatus.NOT_REQUESTED
        loaded_baseline: BehavioralBaseline | None = None
        comparison: DeviationAnalysis | None = None
        if baseline_id is not None:
            if baseline_store is None:
                comparison_status = BaselineComparisonStatus.NOT_AVAILABLE
                warnings.append("BASELINE_STORE_NOT_CONFIGURED")
            else:
                try:
                    loaded_baseline = baseline_store.get(baseline_id)
                except BaselineNotFoundError:
                    comparison_status = BaselineComparisonStatus.NOT_AVAILABLE
                    warnings.append("BASELINE_NOT_FOUND")
                except BaselineError:
                    comparison_status = BaselineComparisonStatus.INCONCLUSIVE
                    warnings.append("BASELINE_LOAD_FAILED")
                else:
                    comparison = compare_to_baseline(extraction.features, loaded_baseline, cfg)
                    comparison_status = comparison.status
                    warnings.extend(comparison.warnings)
        if quality.status is QualityStatus.INCONCLUSIVE:
            status = AnalysisStatus.INCONCLUSIVE
        elif comparison_status is BaselineComparisonStatus.INCONCLUSIVE:
            status = AnalysisStatus.INCONCLUSIVE
        else:
            status = AnalysisStatus.COMPLETED
        return ProsodyAnalysisResult(
            analysis_id=str(uuid.uuid4()),
            status=status,
            quality_status=quality.status,
            feature_schema_version=extraction.features.schema_version,
            features=extraction.features,
            baseline_comparison_status=comparison_status,
            baseline_id=None if loaded_baseline is None else loaded_baseline.baseline_id,
            feature_deviations=()
            if comparison is None
            else tuple(item.as_dict() for item in comparison.deviations),
            deviation_summary=None if comparison is None else comparison.summary,
            warnings=tuple(dict.fromkeys(warnings)),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
    except Exception as exc:
        return _empty_result(
            started,
            AnalysisStatus.FAILED,
            QualityStatus.INCONCLUSIVE,
            ("ANALYSIS_FAILURE",),
            type(exc).__name__,
        )
