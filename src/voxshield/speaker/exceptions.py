"""Stable exceptions for the speaker-verification subsystem."""

from __future__ import annotations


class SpeakerVerificationError(Exception):
    """Base error for speaker-verification operations."""


class ConsentRequiredError(SpeakerVerificationError):
    """Enrollment was not explicitly authorized and consented."""


class ProfileNotFoundError(SpeakerVerificationError):
    """The exact requested profile is absent or deleted."""


class ProfileInactiveError(SpeakerVerificationError):
    """The requested profile has been deactivated."""


class ProfileExpiredError(SpeakerVerificationError):
    """The requested profile has passed its configured expiration time."""


class ProfileIntegrityError(SpeakerVerificationError):
    """A stored profile is malformed or fails integrity validation."""


class ProfileAlreadyExistsError(SpeakerVerificationError):
    """A profile already exists for this pseudonymous identifier."""


class EncoderCompatibilityError(SpeakerVerificationError):
    """Stored profile and active encoder/preprocessing versions differ."""


class EmbeddingExtractionError(SpeakerVerificationError):
    """A speaker embedding could not be extracted safely."""
