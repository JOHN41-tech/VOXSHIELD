"""VoxShield: advisory synthetic-speech detection.

VoxShield answers one question -- *does this audio contain synthetic speech?* --
and reports it as a risk band plus a recommended next step. It does not block
transactions, does not freeze accounts, and does not report callers. The
strongest action it can recommend is routing a case to a human analyst.

Phase 0 ships the audio preprocessing contract, the metadata-only audit trail,
and the API surface. It ships **no model**: :class:`UnavailableDetector` is
installed by default, and the API returns a ``null`` score with status
``UNSCORED`` rather than a plausible-looking guess. A fabricated score can
trigger a step-up verification against a real customer; an absent one cannot.

Quick start::

    from voxshield import prepare

    prepared = prepare(uploaded_bytes)
    print(prepared.speech_seconds, prepared.n_segments)
    print(prepared.metadata())          # safe to log
"""

from __future__ import annotations

from voxshield.audio.features import compute_log_mel
from voxshield.audio.pipeline import (
    PreparedAudio,
    allowed_upload_formats,
    prepare,
    prepare_from_bytes,
)
from voxshield.config import AudioConfig, FeatureConfig, VadConfig, load_audio_config
from voxshield.errors import (
    AudioDecodeError,
    AudioIntakeError,
    AudioQualityError,
    AudioTooLargeError,
    InsufficientSpeechError,
    InvalidAudioSignalError,
    ModelContractError,
    ModelUnavailableError,
    PrivacyPolicyError,
    UnsupportedAudioFormatError,
    VoxShieldError,
)
from voxshield.models.interface import (
    SegmentScore,
    SynthSpeechDetector,
    UnavailableDetector,
)
from voxshield.policy import (
    ActionType,
    PolicyEngine,
    RecommendedAction,
    RiskBand,
)
from voxshield.storage import (
    AuditRecord,
    AuditStore,
    InMemoryAuditStore,
    JsonlAuditStore,
    purge_expired,
)

__version__ = "0.1.0"

__all__ = [
    "ActionType",
    "AudioConfig",
    "AudioDecodeError",
    "AudioIntakeError",
    "AudioQualityError",
    "AudioTooLargeError",
    "AuditRecord",
    "AuditStore",
    "FeatureConfig",
    "InMemoryAuditStore",
    "InsufficientSpeechError",
    "InvalidAudioSignalError",
    "JsonlAuditStore",
    "ModelContractError",
    "ModelUnavailableError",
    "PolicyEngine",
    "PreparedAudio",
    "PrivacyPolicyError",
    "RecommendedAction",
    "RiskBand",
    "SegmentScore",
    "SynthSpeechDetector",
    "UnavailableDetector",
    "UnsupportedAudioFormatError",
    "VadConfig",
    "VoxShieldError",
    "__version__",
    "allowed_upload_formats",
    "compute_log_mel",
    "load_audio_config",
    "prepare",
    "prepare_from_bytes",
    "purge_expired",
]
