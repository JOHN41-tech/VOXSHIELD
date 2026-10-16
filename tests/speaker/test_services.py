from __future__ import annotations

import uuid
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from voxshield.speaker.embeddings.encoder import PREPROCESSING_VERSION
from voxshield.speaker.enrollment import EnrollmentService
from voxshield.speaker.exceptions import (
    ConsentRequiredError,
    EncoderCompatibilityError,
    ProfileExpiredError,
)
from voxshield.speaker.schemas import Embedding
from voxshield.speaker.storage import LocalProfileStore
from voxshield.speaker.verification import (
    Decision,
    InMemoryVerificationAuditSink,
    ThresholdPolicy,
    VerificationService,
)


class FakeEncoder:
    encoder_id = "test-encoder"
    encoder_version = "1"
    sample_rate = 16_000
    embedding_dimension = 3

    def encode(self, audio: np.ndarray, sample_rate: int) -> Embedding:
        assert sample_rate == self.sample_rate
        # A fixed fixture vector makes this an interface test, not a model claim.
        return Embedding(
            (1.0, 2.0, 3.0), self.encoder_id, self.encoder_version, PREPROCESSING_VERSION
        )


def test_enrollment_requires_authorization_and_consent(
    tmp_path: Path, speech_samples: np.ndarray
) -> None:
    service = EnrollmentService(FakeEncoder(), LocalProfileStore(tmp_path))
    with pytest.raises(ConsentRequiredError):
        service.enroll(
            speaker_profile_id=str(uuid.uuid4()),
            samples=[speech_samples],
            sample_rates=[16_000],
            consent_confirmed=False,
            consent_reference="consent-ref",
            authorized=True,
        )


def test_enrollment_and_verification_round_trip_privacy_and_versions(
    tmp_path: Path, speech_samples: np.ndarray
) -> None:
    profile_id = str(uuid.uuid4())
    store = LocalProfileStore(tmp_path / "profiles")
    encoder = FakeEncoder()
    enrollment = EnrollmentService(encoder, store).enroll(
        speaker_profile_id=profile_id,
        samples=[speech_samples, speech_samples],
        sample_rates=[16_000, 16_000],
        consent_confirmed=True,
        consent_reference="opaque-consent-ref",
        authorized=True,
    )
    assert enrollment.profile_created
    assert enrollment.accepted_sample_count == 1
    assert enrollment.samples[1].rejection_reasons == ("DUPLICATE_SAMPLE",)
    profile = store.get(profile_id)
    assert profile.enrollment_sample_count == 1
    assert profile.preprocessing_version == PREPROCESSING_VERSION
    assert len(profile.reference_embedding) == 3
    assert (tmp_path / "profiles" / f"{profile_id}.json").stat().st_mode & 0o777 == 0o600

    audit = InMemoryVerificationAuditSink()
    verifier = VerificationService(
        encoder, store, threshold_policy=ThresholdPolicy(0.8, "test-threshold-v2"), audit_sink=audit
    )
    result = verifier.verify(
        audio=speech_samples, sample_rate=16_000, speaker_profile_id=profile_id
    )
    assert result.decision is Decision.MATCH
    assert result.similarity_score == pytest.approx(1.0)
    assert result.verification_score is None
    assert result.calibration_status == "NOT_CALIBRATED"
    assert result.quality_metrics["duration_seconds"] > 0
    assert len(audit.events()) == 1
    assert "embedding" not in audit.events()[0].__dataclass_fields__
    assert store.delete(profile_id)
    assert not store.delete(profile_id)


def test_insufficient_quality_returns_inconclusive(
    tmp_path: Path, speech_samples: np.ndarray
) -> None:
    profile_id = str(uuid.uuid4())
    store = LocalProfileStore(tmp_path)
    service = EnrollmentService(FakeEncoder(), store)
    service.enroll(
        speaker_profile_id=profile_id,
        samples=[speech_samples],
        sample_rates=[16_000],
        consent_confirmed=True,
        consent_reference="ref",
        authorized=True,
    )
    result = VerificationService(FakeEncoder(), store).verify(
        audio=np.zeros(16000, dtype=np.float32), sample_rate=16_000, speaker_profile_id=profile_id
    )
    assert result.decision is Decision.INCONCLUSIVE
    assert result.similarity_score is None
    assert result.quality_reasons == ("INVALID_AUDIO",)


def test_incompatible_encoder_is_rejected(tmp_path: Path, speech_samples: np.ndarray) -> None:
    profile_id = str(uuid.uuid4())
    store = LocalProfileStore(tmp_path)
    EnrollmentService(FakeEncoder(), store).enroll(
        speaker_profile_id=profile_id,
        samples=[speech_samples],
        sample_rates=[16_000],
        consent_confirmed=True,
        consent_reference="ref",
        authorized=True,
    )

    class NewEncoder(FakeEncoder):
        encoder_version = "2"

    with pytest.raises(EncoderCompatibilityError):
        VerificationService(NewEncoder(), store).verify(
            audio=speech_samples, sample_rate=16_000, speaker_profile_id=profile_id
        )


def test_expired_profile_is_rejected(tmp_path: Path, speech_samples: np.ndarray) -> None:
    profile_id = str(uuid.uuid4())
    store = LocalProfileStore(tmp_path)
    EnrollmentService(FakeEncoder(), store).enroll(
        speaker_profile_id=profile_id,
        samples=[speech_samples],
        sample_rates=[16_000],
        consent_confirmed=True,
        consent_reference="ref",
        authorized=True,
    )
    profile = store.get(profile_id)
    store.update(replace(profile, profile_version=2, expires_at="2000-01-01T00:00:00+00:00"))
    with pytest.raises(ProfileExpiredError):
        VerificationService(FakeEncoder(), store).verify(
            audio=speech_samples, sample_rate=16_000, speaker_profile_id=profile_id
        )
