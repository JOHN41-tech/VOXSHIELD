"""HTTP routes.

Thin by design. Each route validates, delegates, records, and returns. Business
logic lives in :mod:`voxshield.audio.pipeline` and
:mod:`voxshield.policy.evaluator`, so it can be tested without a request and so
the same code path serves a future CLI or a batch job.

The route is responsible for three things the pipeline cannot do for itself:

* bounding the request *before* it is buffered,
* converting typed errors into stable HTTP responses, and
* writing the audit record that makes a decision reviewable later.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import UTC, datetime

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile

from voxshield.api.schemas import (
    STATUS_ANALYZED,
    STATUS_UNSCORED,
    AnalyzeFileResponse,
    AudioMetadataResponse,
    DetectorResponse,
    HealthResponse,
    RecommendedActionResponse,
    SupportedFormatsResponse,
)
from voxshield.audio.pipeline import PreparedAudio, allowed_upload_formats, prepare
from voxshield.config import AudioConfig
from voxshield.errors import AudioTooLargeError, PrivacyPolicyError
from voxshield.models.interface import SynthSpeechDetector
from voxshield.monitoring import get_logger, safe_metadata
from voxshield.policy.evaluator import POLICY_VERSION, PolicyEngine
from voxshield.storage.audit import (
    AuditStore,
    build_record,
    new_request_id,
    validate_session_id,
)

__all__ = ["router"]

logger = get_logger("api")

# Read the upload in chunks so a caller cannot force a single large allocation
# even after the Content-Length check passes.
_UPLOAD_CHUNK_BYTES = 64 * 1024


def _get_config(request: Request) -> AudioConfig:
    return request.app.state.audio_config


def _get_store(request: Request) -> AuditStore:
    return request.app.state.audit_store


def _get_detector(request: Request) -> SynthSpeechDetector:
    return request.app.state.detector


def _read_bounded(upload: UploadFile, limit: int) -> bytes:
    """Read an upload, refusing to buffer more than ``limit`` bytes.

    ``Content-Length`` is checked in :func:`analyze_file` before this runs, but
    it is a client-supplied header. This function enforces the limit against
    bytes actually received, so a lying or absent ``Content-Length`` cannot get
    an oversized payload into memory.

    Args:
        upload: The uploaded file.
        limit: Maximum bytes to accept.

    Returns:
        The raw bytes.

    Raises:
        AudioTooLargeError: More than ``limit`` bytes were received.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = upload.file.read(_UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            msg = f"upload exceeds the {limit}-byte limit"
            raise AudioTooLargeError(msg)
        chunks.append(chunk)
    return b"".join(chunks)


def _register_request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if not request_id:
        request_id = new_request_id()
        request.state.request_id = request_id
    return request_id


def _check_declared_length(request: Request, limit: int) -> None:
    """Reject an oversized upload from its declared Content-Length.

    Cheap first line of defence: it refuses the request before the body is
    buffered, so an oversized upload costs the server a header read rather than
    a memory allocation.
    """
    raw = request.headers.get("content-length")
    if raw is None:
        return
    try:
        declared = int(raw)
    except ValueError:
        return
    if declared > limit:
        msg = f"upload declares {declared} bytes, above the {limit}-byte limit"
        raise AudioTooLargeError(msg)


def _persist(
    *,
    store: AuditStore,
    request_id: str,
    session_id: str,
    status: str,
    model_version: str,
    latency_ms: float,
    metadata: dict[str, object],
    reason_codes: tuple[str, ...],
) -> None:
    """Write an audit record, never letting a logging failure break the request.

    A response that reaches the caller without an audit record would be
    un-reviewable, so a store failure is logged loudly and re-raised as a
    privacy policy violation: the service must not appear to work while silently
    dropping the record of what it decided.
    """
    record = build_record(
        request_id=request_id,
        session_id=session_id,
        status=status,
        model_version=model_version,
        policy_version=POLICY_VERSION,
        latency_ms=latency_ms,
        metadata=metadata,
        reason_codes=reason_codes,
    )
    try:
        store.append(record)
    except Exception as exc:
        logger.error(
            "failed to persist audit record",
            extra=safe_metadata({"request_id": request_id, "error": type(exc).__name__}),
        )
        msg = "audit record could not be persisted; refusing to return an unrecorded decision"
        raise PrivacyPolicyError(msg) from exc


router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    """Liveness and readiness, including whether a model is actually loaded."""
    detector = _get_detector(request)
    # UnavailableDetector reports the sentinel version "unavailable"; any other
    # string identifies loaded weights.
    loaded = detector.model_version != "unavailable"
    return HealthResponse(
        status="ok",
        model_loaded=loaded,
        model_version=detector.model_version if loaded else None,
        policy_version=POLICY_VERSION,
    )


@router.get("/v1/meta/formats", response_model=SupportedFormatsResponse, tags=["ops"])
def supported_formats(request: Request) -> SupportedFormatsResponse:
    """Publish the intake contract so a client can validate before uploading."""
    cfg = _get_config(request)
    return SupportedFormatsResponse(
        allowed_containers=allowed_upload_formats(cfg),
        allowed_subtypes=sorted(cfg.allowed_subtypes),
        max_upload_bytes=cfg.max_upload_bytes,
        max_duration_seconds=cfg.max_duration_seconds,
        max_channels=cfg.max_channels,
        canonical_sample_rate_hz=cfg.target_sample_rate,
        min_speech_seconds=cfg.min_speech_seconds,
    )


@router.post("/v1/analyze/file", response_model=AnalyzeFileResponse, tags=["analysis"])
def analyze_file(
    request: Request,
    audio: UploadFile = File(..., description="Encoded audio. WAV only in MVP."),
    session_id: str = Form(..., description="Pseudonymous UUID identifying the session."),
) -> AnalyzeFileResponse:
    """Analyse an uploaded audio file.

    Returns 200 for both a scored and an unscored analysis. An unavailable model
    is a normal success with a ``null`` score, not an error: the service is
    healthy, the audio was understood, and no judgement was made.
    """
    request_id = _register_request_id(request)
    cfg = _get_config(request)
    started = time.perf_counter()

    # Validate identity before touching the audio: reject a request that is not
    # pseudonymous regardless of what it also contains.
    session = validate_session_id(session_id)
    _check_declared_length(request, cfg.max_upload_bytes)

    try:
        payload = _read_bounded(audio, cfg.max_upload_bytes)
    finally:
        # Close eagerly. Starlette would close it too, but doing it here means
        # the file handle is released even if the pipeline raises.
        audio.file.close()

    if not payload:
        msg = "uploaded file is empty"
        raise AudioTooLargeError(msg)

    prepared = prepare(payload, cfg)

    # Capture metadata *before* releasing buffers. drop_audio_references() clears
    # the segment list, so reading metadata afterwards reports n_segments=0 and
    # silently corrupts the audit record. Metadata holds no audio, so hoisting it
    # is safe.
    metadata = prepared.metadata()

    try:
        features: list[np.ndarray] = [f for _, f in prepared.iter_segment_features()]
        engine = PolicyEngine()
        action, assessment = engine.evaluate(
            detector=_get_detector(request),
            features=features,
            speech_seconds=prepared.speech_seconds,
        )
    finally:
        # Release the audio-derived buffers as soon as inference returns, rather
        # than holding them for the rest of the request lifetime.
        prepared.drop_audio_references()

    latency_ms = (time.perf_counter() - started) * 1000.0
    status = STATUS_ANALYZED if assessment is not None else STATUS_UNSCORED

    _persist(
        store=_get_store(request),
        request_id=request_id,
        session_id=session,
        status=status,
        model_version=assessment.model_version if assessment else "unavailable",
        latency_ms=latency_ms,
        metadata=metadata,
        reason_codes=(assessment.band.value,) if assessment else ("MODEL_UNAVAILABLE",),
    )

    logger.info(
        "analyzed upload",
        extra=safe_metadata(
            {
                "request_id": request_id,
                "status": status,
                "latency_ms": round(latency_ms, 2),
                "speech_seconds": metadata.get("speech_seconds"),
                "n_segments": metadata.get("n_segments"),
                "action": action.action.value,
            }
        ),
    )

    return AnalyzeFileResponse(
        request_id=request_id,
        session_id=session,
        status=status,
        created_at=datetime.now(UTC),
        policy_version=POLICY_VERSION,
        detector=DetectorResponse(**engine.detector_payload(assessment)),
        audio_metadata=AudioMetadataResponse.model_validate(metadata),
        recommended_action=RecommendedActionResponse(**engine.action_payload(action)),
        audio_retained=False,
    )


def iter_metadata(prepared: PreparedAudio) -> Iterator[tuple[str, object]]:
    """Yield metadata key/value pairs. Provided for tests and batch callers."""
    yield from prepared.metadata().items()
