"""Typed contracts for consented speaker verification.

Profiles hold vectors, never recordings. Vectors are biometric data and must be
handled as sensitive information even though they cannot be played as audio.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class Decision(StrEnum):
    MATCH = "MATCH"
    NON_MATCH = "NON_MATCH"
    INCONCLUSIVE = "INCONCLUSIVE"


class QualityStatus(StrEnum):
    ACCEPTABLE = "ACCEPTABLE"
    INSUFFICIENT = "INSUFFICIENT"


@dataclass(frozen=True, slots=True)
class Embedding:
    vector: tuple[float, ...]
    encoder_id: str
    encoder_version: str
    preprocessing_version: str


@dataclass(frozen=True, slots=True)
class SpeakerProfile:
    speaker_profile_id: str
    profile_version: int
    encoder_id: str
    encoder_version: str
    embedding_dimension: int
    reference_embedding: tuple[float, ...]
    enrollment_sample_count: int
    preprocessing_version: str
    created_at: str
    consent_reference: str
    quality_summary: dict[str, Any]
    expires_at: str | None = None
    active: bool = True


@dataclass(frozen=True, slots=True)
class SampleQuality:
    acceptable: bool
    rejection_reasons: tuple[str, ...]
    duration_seconds: float
    voiced_seconds: float
    voiced_ratio: float
    silence_ratio: float
    clipping_ratio: float
    rms_dbfs: float
    snr_db: float | None
    preprocessing_ms: float


@dataclass(frozen=True, slots=True)
class EnrollmentSampleResult:
    sample_index: int
    accepted: bool
    rejection_reasons: tuple[str, ...]
    quality: SampleQuality | None


@dataclass(frozen=True, slots=True)
class EnrollmentResult:
    speaker_profile_id: str
    profile_version: int
    accepted_sample_count: int
    samples: tuple[EnrollmentSampleResult, ...]
    profile_created: bool


@dataclass(frozen=True, slots=True)
class VerificationResult:
    verification_id: str
    speaker_profile_id: str
    decision: Decision
    similarity_score: float | None
    verification_score: float | None
    threshold: float
    threshold_version: str
    quality_status: QualityStatus
    quality_reasons: tuple[str, ...]
    quality_metrics: dict[str, float | None]
    encoder_id: str
    encoder_version: str
    profile_version: int
    latency_ms: float
    calibration_status: str = "NOT_CALIBRATED"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready result; cosine similarity is not a probability."""
        return {
            "verification_id": self.verification_id,
            "speaker_profile_id": self.speaker_profile_id,
            "decision": self.decision.value,
            "similarity_score": self.similarity_score,
            "verification_score": self.verification_score,
            "threshold": self.threshold,
            "threshold_version": self.threshold_version,
            "quality_status": self.quality_status.value,
            "quality_reasons": list(self.quality_reasons),
            "quality_metrics": self.quality_metrics,
            "encoder_id": self.encoder_id,
            "encoder_version": self.encoder_version,
            "profile_version": self.profile_version,
            "latency_ms": self.latency_ms,
            "calibration_status": self.calibration_status,
        }
