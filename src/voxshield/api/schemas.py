"""Request and response models for the public API.

Response shapes are declared here rather than assembled ad hoc in the route so
that a change to the contract is a visible, reviewable diff, and so that
OpenAPI is generated from the same definitions the code validates against.

The one rule the models encode: **absence is explicit.** Fields that may have no
value are typed ``| None`` and are always present in the payload. A caller must
never have to distinguish "this field was omitted" from "this field has no
value", because that distinction is exactly where a missing detector could be
mistaken for a clean bill of health.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AnalyzeFileResponse",
    "AudioMetadataResponse",
    "DetectorResponse",
    "ErrorResponse",
    "HealthResponse",
    "RecommendedActionResponse",
    "SupportedFormatsResponse",
]

# Stable machine-readable outcome codes. The API returns these rather than
# prose, so a client can branch without parsing English. Values are part of the
# public contract and must not be renamed without a version bump.
STATUS_ANALYZED = "ANALYZED"
STATUS_UNSCORED = "UNSCORED"
STATUS_INSUFFICIENT_SPEECH = "INSUFFICIENT_SPEECH"


class _StrictModel(BaseModel):
    """Base model that rejects unknown fields.

    A silently-ignored field is how a caller ends up believing they supplied
    ``callback_url`` when nothing read it. Rejecting unknown input makes that
    failure loud.
    """

    model_config = ConfigDict(extra="forbid")


class DetectorResponse(BaseModel):
    """Detector output for one clip.

    With no model loaded, every probability is ``None`` and ``status`` is
    ``"unavailable"``. The fields are still present.
    """

    status: str = Field(description="`ok` when a model scored the clip, else `unavailable`.")
    model_version: str | None = Field(
        default=None,
        description="Version of the weights that produced the score. Null when no model is loaded.",
    )
    score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Aggregate synthetic-speech probability. Null when no model is loaded.",
    )
    synthetic_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    bona_fide_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    n_segments_scored: int = Field(
        default=0, ge=0, description="Number of analysis windows the detector scored."
    )
    risk_band: str = Field(
        default="unknown",
        description="Coarse band: unknown, low, medium, high. Not a conclusion about a person.",
    )
    reason_code: str = Field(
        default="MODEL_UNAVAILABLE",
        description="Stable code explaining why this is scored or unscored.",
    )


class RecommendedActionResponse(BaseModel):
    """What VoxShield recommends the caller do next.

    Advisory only. The strongest available action is ``escalate_to_analyst``;
    nothing in VoxShield blocks a transaction.
    """

    action: str = Field(
        description="none, log_only, challenge_mfa, verified_callback, escalate_to_analyst."
    )
    risk_band: str
    rationale: str = Field(
        description="Why this action. Never contains content derived from the audio."
    )
    requires_human_review: bool


class AudioMetadataResponse(BaseModel):
    """Metadata about the submitted audio.

    Timings, counts, and signal statistics. No samples, no transcript, no
    identity-bearing field.
    """

    model_config = ConfigDict(extra="allow")

    container: str | None = None
    sample_rate_hz: int | None = None
    channels: int | None = None
    duration_seconds: float | None = None
    canonical_sample_rate_hz: int | None = None
    speech_seconds: float | None = None
    speech_ratio: float | None = None
    n_segments: int | None = None
    vad_threshold_dbfs: float | None = None
    gain_db_applied: float | None = None
    clipped: bool | None = None


class AnalyzeFileResponse(BaseModel):
    """Successful analysis response.

    Returned for both ``ANALYZED`` and ``UNSCORED``. An unscored response is a
    normal success: the audio was understood, and no model was available to
    judge it. That is different from an error and different from a low risk
    result.
    """

    request_id: str = Field(description="Server-generated identifier for this analysis.")
    session_id: str = Field(description="Caller-supplied pseudonymous session identifier.")
    status: str = Field(description="ANALYZED or UNSCORED.")
    created_at: datetime
    policy_version: str
    detector: DetectorResponse
    audio_metadata: AudioMetadataResponse
    recommended_action: RecommendedActionResponse
    audio_retained: bool = Field(
        default=False,
        description="Always false. Present so a client can assert the guarantee from the response.",
    )


class HealthResponse(BaseModel):
    """Liveness and readiness."""

    status: str
    model_loaded: bool
    model_version: str | None
    policy_version: str


class SupportedFormatsResponse(BaseModel):
    """Intake contract, so a client can check before uploading."""

    allowed_containers: list[str]
    allowed_subtypes: list[str]
    max_upload_bytes: int
    max_duration_seconds: float
    max_channels: int
    canonical_sample_rate_hz: int
    min_speech_seconds: float


class ErrorResponse(_StrictModel):
    """Uniform error body.

    ``code`` is stable and machine-readable. ``message`` is for humans and
    never echoes caller-supplied data. ``request_id`` lets an operator find the
    corresponding audit record.
    """

    error: str
    code: str
    message: str
    request_id: str
