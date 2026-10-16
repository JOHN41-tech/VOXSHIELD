"""Biometric speaker-profile storage with explicit deletion and atomic writes."""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from voxshield.speaker.exceptions import (
    ProfileAlreadyExistsError,
    ProfileInactiveError,
    ProfileIntegrityError,
    ProfileNotFoundError,
)
from voxshield.speaker.schemas import SpeakerProfile


class ProfileStore(Protocol):
    """Interface required by enrollment and verification services."""

    def create(self, profile: SpeakerProfile) -> None: ...
    def get(self, profile_id: str) -> SpeakerProfile: ...
    def update(self, profile: SpeakerProfile) -> None: ...
    def list_active(self) -> tuple[SpeakerProfile, ...]: ...
    def deactivate(self, profile_id: str) -> SpeakerProfile: ...
    def delete(self, profile_id: str) -> bool: ...


def validate_profile_id(profile_id: str) -> str:
    """Require UUID-shaped pseudonymous IDs to prevent path traversal/PII labels."""
    try:
        return str(uuid.UUID(profile_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("speaker_profile_id must be a pseudonymous UUID") from exc


def _profile_from_dict(value: dict[str, object]) -> SpeakerProfile:
    expected = set(SpeakerProfile.__dataclass_fields__)
    if set(value) != expected:
        raise ProfileIntegrityError("stored profile fields do not match the schema")
    try:
        value["reference_embedding"] = tuple(value["reference_embedding"])  # type: ignore[arg-type]
        profile = SpeakerProfile(**value)  # type: ignore[arg-type]
        normalized_id = validate_profile_id(profile.speaker_profile_id)
        if normalized_id != profile.speaker_profile_id:
            raise ValueError("profile identifier is not normalized")
        if profile.profile_version < 1 or profile.embedding_dimension != len(
            profile.reference_embedding
        ):
            raise ValueError("profile version or embedding dimension is invalid")
        if profile.embedding_dimension < 1 or not profile.consent_reference.strip():
            raise ValueError("profile dimension or consent reference is invalid")
        if profile.enrollment_sample_count < 1 or (
            not profile.active and profile.profile_version < 2
        ):
            raise ValueError("profile lifecycle metadata is invalid")
        if any(
            not isinstance(x, (float, int)) or not float("-inf") < float(x) < float("inf")
            for x in profile.reference_embedding
        ):
            raise ValueError("profile embedding is non-finite")
        vector_norm = math.sqrt(sum(float(value) ** 2 for value in profile.reference_embedding))
        if not math.isclose(vector_norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError("profile embedding must be L2-normalized")
        created_at = datetime.fromisoformat(profile.created_at.replace("Z", "+00:00"))
        if created_at.tzinfo is None:
            raise ValueError("profile creation timestamp must include a timezone")
        if profile.expires_at is not None:
            expiry = datetime.fromisoformat(profile.expires_at.replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                raise ValueError("profile expiry must include a timezone")
    except Exception as exc:
        if isinstance(exc, ProfileIntegrityError):
            raise
        raise ProfileIntegrityError("stored profile failed validation") from exc
    return profile


class LocalProfileStore:
    """Development store. Files contain sensitive vectors; protect the directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass
        self._lock = threading.RLock()

    def _path(self, profile_id: str) -> Path:
        return self.root / f"{validate_profile_id(profile_id)}.json"

    def _write(self, profile: SpeakerProfile, *, create: bool = False) -> None:
        path = self._path(profile.speaker_profile_id)
        _profile_from_dict(asdict(profile))
        payload = json.dumps(asdict(profile), sort_keys=True, separators=(",", ":"))
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.root, delete=False
        ) as handle:
            temporary = Path(handle.name)
            try:
                os.chmod(temporary, 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                if create and path.exists():
                    raise ProfileAlreadyExistsError("a profile already exists for this ID")
                os.replace(temporary, path)
                os.chmod(path, 0o600)
            finally:
                temporary.unlink(missing_ok=True)

    def create(self, profile: SpeakerProfile) -> None:
        with self._lock:
            self._write(profile, create=True)

    def get(self, profile_id: str) -> SpeakerProfile:
        path = self._path(profile_id)
        with self._lock:
            if not path.is_file():
                raise ProfileNotFoundError("requested speaker profile was not found")
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("profile root is not an object")
                profile = _profile_from_dict(value)
                if profile.speaker_profile_id != validate_profile_id(profile_id):
                    raise ProfileIntegrityError("profile identifier does not match its storage key")
            except ProfileIntegrityError:
                raise
            except Exception as exc:
                raise ProfileIntegrityError("stored profile could not be read") from exc
        return profile

    def update(self, profile: SpeakerProfile) -> None:
        with self._lock:
            previous = self.get(profile.speaker_profile_id)
            if profile.profile_version != previous.profile_version + 1:
                raise ProfileIntegrityError("profile update must increment version by exactly one")
            self._write(profile)

    def list_active(self) -> tuple[SpeakerProfile, ...]:
        with self._lock:
            files = sorted(self.root.glob("*.json"))
        profiles = (self.get(path.stem) for path in files)
        now = datetime.now(UTC)
        return tuple(
            profile
            for profile in profiles
            if profile.active
            and (
                profile.expires_at is None
                or datetime.fromisoformat(profile.expires_at.replace("Z", "+00:00")) > now
            )
        )

    def deactivate(self, profile_id: str) -> SpeakerProfile:
        from dataclasses import replace

        current = self.get(profile_id)
        if not current.active:
            raise ProfileInactiveError("requested speaker profile is already inactive")
        updated = replace(current, profile_version=current.profile_version + 1, active=False)
        self.update(updated)
        return updated

    def delete(self, profile_id: str) -> bool:
        path = self._path(profile_id)
        with self._lock:
            if not path.exists():
                return False
            path.unlink()
            return True
