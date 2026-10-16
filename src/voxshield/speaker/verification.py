"""Speaker-profile verification, distinct from synthetic-speech detection."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Protocol

import numpy as np

from voxshield.audio.loader import AudioSource
from voxshield.speaker.embeddings.encoder import (
    PREPROCESSING_VERSION,
    SpeakerEncoder,
    normalized_embedding,
)
from voxshield.speaker.exceptions import (
    EncoderCompatibilityError,
    ProfileExpiredError,
    ProfileInactiveError,
)
from voxshield.speaker.quality import SpeakerQualityPolicy, assess_sample
from voxshield.speaker.schemas import Decision, QualityStatus, VerificationResult
from voxshield.speaker.storage import ProfileStore, validate_profile_id


@dataclass(frozen=True, slots=True)
class ThresholdPolicy:
    """Cosine threshold. The default project recipe is provisional, not calibrated."""

    threshold: float = 0.75
    version: str = "development-uncalibrated-v1"

    def __post_init__(self) -> None:
        if not -1.0 <= self.threshold <= 1.0:
            raise ValueError("cosine threshold must be within [-1, 1]")
        if not self.version.strip():
            raise ValueError("threshold policy version must be non-empty")


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity in [-1, 1], never a probability."""
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim != 1 or b.ndim != 1 or a.size == 0 or a.shape != b.shape:
        raise ValueError("cosine similarity requires non-empty vectors of equal dimensions")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("cosine similarity requires finite vectors")
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        raise ValueError("cosine similarity is undefined for a zero-norm vector")
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


@dataclass(frozen=True, slots=True)
class VerificationAuditEvent:
    verification_id: str
    speaker_profile_id: str
    decision: str
    similarity_score: float | None
    threshold: float
    profile_version: int
    encoder_id: str
    encoder_version: str
    created_at: str
    latency_ms: float
    quality_reasons: tuple[str, ...]


class VerificationAuditSink(Protocol):
    def append(self, event: VerificationAuditEvent) -> None: ...


class InMemoryVerificationAuditSink:
    """Minimal metadata-only local audit sink; no audio or embedding fields exist."""

    def __init__(self) -> None:
        self._events: list[VerificationAuditEvent] = []
        self._lock = Lock()

    def append(self, event: VerificationAuditEvent) -> None:
        with self._lock:
            self._events.append(event)

    def events(self) -> tuple[VerificationAuditEvent, ...]:
        with self._lock:
            return tuple(self._events)


class VerificationService:
    def __init__(
        self,
        encoder: SpeakerEncoder,
        store: ProfileStore,
        *,
        threshold_policy: ThresholdPolicy | None = None,
        audit_sink: VerificationAuditSink | None = None,
        quality_policy: SpeakerQualityPolicy | None = None,
        minimum_voiced_seconds: float | None = None,
    ) -> None:
        if quality_policy is not None and minimum_voiced_seconds is not None:
            raise ValueError("provide either quality_policy or minimum_voiced_seconds, not both")
        self.encoder = encoder
        self.store = store
        self.threshold_policy = threshold_policy or ThresholdPolicy()
        self.audit_sink = audit_sink or InMemoryVerificationAuditSink()
        self.quality_policy = quality_policy or SpeakerQualityPolicy(
            minimum_voiced_seconds=(
                2.0 if minimum_voiced_seconds is None else minimum_voiced_seconds
            )
        )

    def verify(
        self, *, audio: AudioSource, speaker_profile_id: str, sample_rate: int | None = None
    ) -> VerificationResult:
        """Compare one suitable sample with exactly the requested active profile."""
        started = time.perf_counter()
        profile_id = validate_profile_id(speaker_profile_id)
        profile = self.store.get(profile_id)
        if not profile.active:
            raise ProfileInactiveError("requested speaker profile is inactive")
        if profile.expires_at is not None:
            expires_at = datetime.fromisoformat(profile.expires_at.replace("Z", "+00:00"))
            if expires_at <= datetime.now(UTC):
                raise ProfileExpiredError("requested speaker profile has expired")
        if (
            profile.encoder_id != self.encoder.encoder_id
            or profile.encoder_version != self.encoder.encoder_version
            or profile.preprocessing_version != PREPROCESSING_VERSION
            or profile.embedding_dimension != self.encoder.embedding_dimension
        ):
            raise EncoderCompatibilityError(
                "profile encoder/preprocessing version is incompatible; "
                "re-enroll with the configured encoder"
            )

        assessed = assess_sample(audio, sample_rate=sample_rate, quality_policy=self.quality_policy)
        similarity: float | None = None
        if not assessed.quality.acceptable:
            decision = Decision.INCONCLUSIVE
            quality_status = QualityStatus.INSUFFICIENT
            quality_reasons = assessed.quality.rejection_reasons
        else:
            try:
                test_embedding = self.encoder.encode(assessed.samples, assessed.sample_rate)
                if (
                    test_embedding.encoder_id != self.encoder.encoder_id
                    or test_embedding.encoder_version != self.encoder.encoder_version
                    or test_embedding.preprocessing_version != PREPROCESSING_VERSION
                ):
                    raise ValueError("encoder returned incompatible embedding metadata")
                test_vector = normalized_embedding(
                    np.asarray(test_embedding.vector), dimension=self.encoder.embedding_dimension
                )
                similarity = cosine_similarity(
                    np.asarray(profile.reference_embedding, dtype=np.float64), test_vector
                )
                decision = (
                    Decision.MATCH
                    if similarity >= self.threshold_policy.threshold
                    else Decision.NON_MATCH
                )
                quality_status = QualityStatus.ACCEPTABLE
            except Exception:
                decision = Decision.INCONCLUSIVE
                quality_status = QualityStatus.INSUFFICIENT
                similarity = None
                quality_reasons = ("EMBEDDING_FAILURE",)
            finally:
                assessed.samples = np.empty(0, dtype=np.float32)
            if similarity is not None:
                quality_reasons = assessed.quality.rejection_reasons
        if assessed.samples.size:
            assessed.samples = np.empty(0, dtype=np.float32)

        latency = (time.perf_counter() - started) * 1000.0
        result = VerificationResult(
            verification_id=str(uuid.uuid4()),
            speaker_profile_id=profile_id,
            decision=decision,
            similarity_score=similarity,
            verification_score=None,
            threshold=self.threshold_policy.threshold,
            threshold_version=self.threshold_policy.version,
            quality_status=quality_status,
            quality_reasons=quality_reasons,
            quality_metrics={
                "duration_seconds": assessed.quality.duration_seconds,
                "voiced_seconds": assessed.quality.voiced_seconds,
                "voiced_ratio": assessed.quality.voiced_ratio,
                "silence_ratio": assessed.quality.silence_ratio,
                "clipping_ratio": assessed.quality.clipping_ratio,
                "rms_dbfs": assessed.quality.rms_dbfs,
                "snr_db": assessed.quality.snr_db,
            },
            encoder_id=self.encoder.encoder_id,
            encoder_version=self.encoder.encoder_version,
            profile_version=profile.profile_version,
            latency_ms=latency,
            calibration_status="NOT_CALIBRATED",
        )
        self.audit_sink.append(
            VerificationAuditEvent(
                verification_id=result.verification_id,
                speaker_profile_id=profile_id,
                decision=result.decision.value,
                similarity_score=result.similarity_score,
                threshold=result.threshold,
                profile_version=result.profile_version,
                encoder_id=result.encoder_id,
                encoder_version=result.encoder_version,
                created_at=datetime.now(UTC).isoformat(),
                latency_ms=result.latency_ms,
                quality_reasons=result.quality_reasons,
            )
        )
        return result
