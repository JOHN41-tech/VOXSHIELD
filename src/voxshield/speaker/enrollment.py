"""Explicit, consent-gated enrollment service."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from voxshield.audio.loader import AudioSource
from voxshield.speaker.embeddings.encoder import (
    PREPROCESSING_VERSION,
    SpeakerEncoder,
    normalized_embedding,
)
from voxshield.speaker.exceptions import ConsentRequiredError
from voxshield.speaker.quality import SpeakerQualityPolicy, assess_sample
from voxshield.speaker.schemas import (
    Embedding,
    EnrollmentResult,
    EnrollmentSampleResult,
    SpeakerProfile,
)
from voxshield.speaker.storage import ProfileStore, validate_profile_id


def _fingerprint(source: AudioSource) -> str | None:
    """Hash explicit source material transiently to reject exact duplicate inputs."""
    try:
        if isinstance(source, (bytes, bytearray, memoryview)):
            raw = bytes(source)
            digest = hashlib.sha256(raw).hexdigest()
        elif isinstance(source, (str, Path)):
            hasher = hashlib.sha256()
            with Path(source).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        else:
            return None
        return digest
    except OSError:
        return None


class EnrollmentService:
    """Create an enrolled profile only after explicit authorization and consent."""

    def __init__(
        self,
        encoder: SpeakerEncoder,
        store: ProfileStore,
        *,
        quality_policy: SpeakerQualityPolicy | None = None,
        minimum_voiced_seconds: float | None = None,
    ):
        if quality_policy is not None and minimum_voiced_seconds is not None:
            raise ValueError("provide either quality_policy or minimum_voiced_seconds, not both")
        self.encoder = encoder
        self.store = store
        self.quality_policy = quality_policy or SpeakerQualityPolicy(
            minimum_voiced_seconds=(
                2.0 if minimum_voiced_seconds is None else minimum_voiced_seconds
            )
        )

    def enroll(
        self,
        *,
        speaker_profile_id: str,
        samples: Sequence[AudioSource],
        consent_confirmed: bool,
        consent_reference: str,
        authorized: bool,
        sample_rates: Sequence[int | None] | None = None,
    ) -> EnrollmentResult:
        """Enroll one or more explicitly supplied recordings; raw audio is not stored."""
        profile_id = validate_profile_id(speaker_profile_id)
        if not authorized or not consent_confirmed or not consent_reference.strip():
            raise ConsentRequiredError(
                "enrollment requires authorization, explicit consent, and a consent reference"
            )
        if not samples:
            raise ValueError("at least one enrollment sample is required")
        if sample_rates is not None and len(sample_rates) != len(samples):
            raise ValueError("sample_rates must have one entry per sample")

        unique_hashes: set[str] = set()
        vectors: list[np.ndarray] = []
        qualities: list[dict[str, float | None]] = []
        outcomes: list[EnrollmentSampleResult] = []
        for index, source in enumerate(samples):
            fingerprint = _fingerprint(source)
            if fingerprint is not None and fingerprint in unique_hashes:
                outcomes.append(EnrollmentSampleResult(index, False, ("DUPLICATE_SAMPLE",), None))
                continue
            if fingerprint is not None:
                unique_hashes.add(fingerprint)
            assessed = assess_sample(
                source,
                sample_rate=None if sample_rates is None else sample_rates[index],
                quality_policy=self.quality_policy,
            )
            quality = assessed.quality
            if not quality.acceptable:
                assessed.samples = np.empty(0, dtype=np.float32)
                outcomes.append(
                    EnrollmentSampleResult(index, False, quality.rejection_reasons, quality)
                )
                continue
            try:
                embedding: Embedding = self.encoder.encode(assessed.samples, assessed.sample_rate)
                if (
                    embedding.encoder_id != self.encoder.encoder_id
                    or embedding.encoder_version != self.encoder.encoder_version
                    or embedding.preprocessing_version != PREPROCESSING_VERSION
                ):
                    raise ValueError("encoder returned incompatible embedding metadata")
                vector = normalized_embedding(
                    np.asarray(embedding.vector), dimension=self.encoder.embedding_dimension
                )
            except Exception:
                del assessed.samples
                outcomes.append(
                    EnrollmentSampleResult(index, False, ("EMBEDDING_FAILURE",), quality)
                )
                continue
            finally:
                # The temporary canonical waveform must not be retained by this service.
                if "assessed" in locals():
                    assessed.samples = np.empty(0, dtype=np.float32)

            if any(float(np.dot(vector, existing)) >= 0.99999 for existing in vectors):
                outcomes.append(
                    EnrollmentSampleResult(index, False, ("DUPLICATE_SAMPLE",), quality)
                )
                continue
            vectors.append(vector)
            qualities.append(
                {
                    "duration_seconds": quality.duration_seconds,
                    "voiced_seconds": quality.voiced_seconds,
                    "voiced_ratio": quality.voiced_ratio,
                    "silence_ratio": quality.silence_ratio,
                    "clipping_ratio": quality.clipping_ratio,
                    "rms_dbfs": quality.rms_dbfs,
                    "snr_db": quality.snr_db,
                }
            )
            outcomes.append(EnrollmentSampleResult(index, True, (), quality))

        if not vectors:
            return EnrollmentResult(profile_id, 0, 0, tuple(outcomes), False)
        centroid = normalized_embedding(
            np.mean(np.stack(vectors), axis=0), dimension=self.encoder.embedding_dimension
        )
        profile = SpeakerProfile(
            speaker_profile_id=profile_id,
            profile_version=1,
            encoder_id=self.encoder.encoder_id,
            encoder_version=self.encoder.encoder_version,
            embedding_dimension=self.encoder.embedding_dimension,
            reference_embedding=tuple(float(value) for value in centroid),
            enrollment_sample_count=len(vectors),
            preprocessing_version=PREPROCESSING_VERSION,
            created_at=datetime.now(UTC).isoformat(),
            consent_reference=consent_reference.strip(),
            quality_summary={
                "accepted_sample_count": len(vectors),
                "mean_duration_seconds": float(
                    np.mean([item["duration_seconds"] for item in qualities])
                ),
                "mean_voiced_seconds": float(
                    np.mean([item["voiced_seconds"] for item in qualities])
                ),
            },
            active=True,
        )
        self.store.create(profile)
        return EnrollmentResult(profile_id, 1, len(vectors), tuple(outcomes), True)
