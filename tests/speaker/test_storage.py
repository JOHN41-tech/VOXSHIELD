from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from voxshield.speaker.exceptions import ProfileInactiveError, ProfileNotFoundError
from voxshield.speaker.schemas import SpeakerProfile
from voxshield.speaker.storage import LocalProfileStore


def _profile(profile_id: str) -> SpeakerProfile:
    return SpeakerProfile(
        speaker_profile_id=profile_id,
        profile_version=1,
        encoder_id="test",
        encoder_version="1",
        embedding_dimension=2,
        reference_embedding=(0.6, 0.8),
        enrollment_sample_count=1,
        preprocessing_version="prep-v1",
        created_at="2026-01-01T00:00:00+00:00",
        consent_reference="opaque-reference",
        quality_summary={"accepted_sample_count": 1},
    )


def test_deactivate_versions_and_delete_only_named_profile(tmp_path: Path) -> None:
    store = LocalProfileStore(tmp_path)
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    store.create(_profile(first))
    store.create(_profile(second))
    deactivated = store.deactivate(first)
    assert deactivated.profile_version == 2
    assert not deactivated.active
    with pytest.raises(ProfileInactiveError):
        store.deactivate(first)
    assert store.get(second).active
    assert store.delete(first)
    with pytest.raises(ProfileNotFoundError):
        store.get(first)
    assert store.get(second).active


def test_profile_id_rejects_path_traversal(tmp_path: Path) -> None:
    store = LocalProfileStore(tmp_path)
    with pytest.raises(ValueError, match="UUID"):
        store.get("../../other")
