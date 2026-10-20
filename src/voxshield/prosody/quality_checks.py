"""Quality gates for prosody measurements."""

from __future__ import annotations

from dataclasses import dataclass

from voxshield.prosody.config import ProsodyConfig
from voxshield.prosody.feature_extractor import ExtractionResult
from voxshield.prosody.schema import QualityStatus


@dataclass(frozen=True, slots=True)
class ProsodyQuality:
    status: QualityStatus
    warnings: tuple[str, ...]
    usable_feature_count: int

    @property
    def is_comparable(self) -> bool:
        return self.status is not QualityStatus.INCONCLUSIVE and self.usable_feature_count >= 3


def assess_prosody_quality(
    extraction: ExtractionResult,
    config: ProsodyConfig | None = None,
) -> ProsodyQuality:
    cfg = config or ProsodyConfig()
    warnings = list(extraction.features.warnings)
    if extraction.quality.is_silent:
        warnings.append("SILENCE_ONLY")
    if extraction.quality.has_clipping:
        warnings.append("CLIPPING")
    if extraction.quality.non_finite_ratio > 0.0:
        warnings.append("NON_FINITE_SAMPLES")
    if extraction.features.duration_seconds < cfg.min_duration_seconds:
        warnings.append("SHORT_RECORDING")
    if extraction.speech.speech_seconds < cfg.min_voiced_seconds:
        warnings.append("INSUFFICIENT_VOICED_SPEECH")
    valid_count = len(extraction.features.valid_values)
    if extraction.quality.is_silent or extraction.quality.non_finite_ratio > 0.0:
        status = QualityStatus.INCONCLUSIVE
    elif valid_count < 3 or extraction.speech.speech_seconds < cfg.min_voiced_seconds:
        status = QualityStatus.INCONCLUSIVE
    elif extraction.quality.has_clipping or extraction.quality.warnings:
        status = QualityStatus.WARN
    else:
        status = QualityStatus.PASS
    return ProsodyQuality(status, tuple(dict.fromkeys(warnings)), valid_count)
