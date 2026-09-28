"""Typed-error to HTTP translation.

The rule this module enforces: **untrusted input never produces a 500.** Every
error raised inside the audio pipeline maps to a deliberate status code and a
stable machine-readable code, so a hostile file yields a 4xx with a useful
message rather than a stack trace.

The security property matters more than the ergonomics. An endpoint that can be
made to 500 by a malformed file is an endpoint whose error path is less tested
than its happy path, and that is where availability bugs live.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from voxshield.errors import (
    AudioDecodeError,
    AudioIntakeError,
    AudioTooLargeError,
    InsufficientSpeechError,
    InvalidAudioSignalError,
    ModelContractError,
    ModelUnavailableError,
    PrivacyPolicyError,
    UnsupportedAudioFormatError,
    VoxShieldError,
)

__all__ = ["error_body", "install_exception_handlers"]

# HTTP status and stable code for each typed error. Chosen so a client can act
# on the code alone: retrying a 413 or a 415 will never help, but retrying a 503
# might.
_STATUS_MAP: tuple[tuple[type[VoxShieldError], int, str], ...] = (
    (AudioTooLargeError, 413, "AUDIO_TOO_LARGE"),
    (UnsupportedAudioFormatError, 415, "UNSUPPORTED_AUDIO_FORMAT"),
    (AudioDecodeError, 422, "AUDIO_DECODE_FAILED"),
    (InvalidAudioSignalError, 422, "INVALID_AUDIO_SIGNAL"),
    (InsufficientSpeechError, 422, "INSUFFICIENT_SPEECH"),
    (ModelUnavailableError, 503, "MODEL_UNAVAILABLE"),
    (ModelContractError, 500, "MODEL_CONTRACT_VIOLATION"),
    (PrivacyPolicyError, 400, "PRIVACY_POLICY_VIOLATION"),
    (AudioIntakeError, 422, "AUDIO_INTAKE_FAILED"),
)


def _mapping_for(error: Exception) -> tuple[int, str]:
    for error_type, status, code in _STATUS_MAP:
        if isinstance(error, error_type):
            return status, code
    return 500, "INTERNAL_ERROR"


def error_body(
    error: Exception,
    request_id: str,
    *,
    expose_message: bool = True,
) -> dict[str, object]:
    """Build the JSON body for an error response.

    Args:
        error: The raised exception.
        request_id: Request identifier for correlation.
        expose_message: When ``False``, the message is replaced with a generic
            string. Used for unexpected errors, whose messages can contain
            internal detail.

    Returns:
        A dict matching :class:`~voxshield.api.schemas.ErrorResponse`.
    """
    status, code = _mapping_for(error)
    if status == 500:
        expose_message = False
    message = str(error) if expose_message else "An internal error occurred."
    return {
        "error": "analysis_failed",
        "code": code,
        "message": message,
        "request_id": request_id,
    }


def install_exception_handlers(app: FastAPI) -> None:
    """Register handlers that turn typed errors into JSON responses.

    Args:
        app: The FastAPI application.
    """
    from voxshield.storage.audit import new_request_id

    @app.exception_handler(InsufficientSpeechError)
    async def _insufficient(request: Request, exc: InsufficientSpeechError) -> JSONResponse:
        # An abstention is a legitimate, informative outcome, not a client
        # mistake. 422 keeps it out of the 5xx alerting path while still telling
        # the caller to provide more audio.
        request_id = getattr(request.state, "request_id", None) or new_request_id()
        body = error_body(exc, request_id)
        body["error"] = "insufficient_speech"
        return JSONResponse(status_code=422, content=body)

    @app.exception_handler(VoxShieldError)
    async def _voxshield(request: Request, exc: VoxShieldError) -> JSONResponse:
        request_id = getattr(request.state, "request_id", None) or new_request_id()
        body = error_body(exc, request_id)
        if isinstance(exc, (AudioTooLargeError, UnsupportedAudioFormatError)):
            body["error"] = "audio_rejected"
        elif isinstance(exc, (AudioDecodeError, InvalidAudioSignalError)):
            body["error"] = "audio_unusable"
        return JSONResponse(status_code=_mapping_for(exc)[0], content=body)

    @app.exception_handler(ValueError)
    async def _value_error(request: Request, exc: ValueError) -> JSONResponse:
        # The API layer raises ValueError for contract violations such as a
        # session_id that is not a UUID. That is a caller error, not a bug, but
        # it must still not escape as a 500.
        request_id = getattr(request.state, "request_id", None) or new_request_id()
        return JSONResponse(
            status_code=400,
            content={
                "error": "invalid_request",
                "code": "INVALID_REQUEST",
                "message": str(exc),
                "request_id": request_id,
            },
        )
