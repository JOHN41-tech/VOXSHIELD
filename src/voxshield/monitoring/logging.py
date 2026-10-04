"""Structured logging that cannot leak audio.

The privacy boundary is enforced by what the logging layer is *able* to accept,
not by remembering to be careful at each call site. :func:`safe_metadata` copies
a caller-supplied mapping through a screen that drops forbidden keys and scrubs
value shapes that look like direct identifiers, then
:func:`get_logger` returns a logger whose standard fields are already screened.

Application logs are a second copy of the audit trail, and they fan out to
aggregators, error trackers, and support tooling. An unscreened log line is the
easiest way for a payload to escape the boundary that
:mod:`voxshield.storage.audit` enforces for the record store.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Mapping
from typing import Any

__all__ = [
    "configure_logging",
    "get_logger",
    "safe_metadata",
    "scrub_mapping",
]

_LOGGER_NAMESPACE = "voxshield"

# Keys that must never be logged. Matched case-insensitively against whole
# ``_``-delimited tokens. Token boundaries matter: a bare ``sample`` pattern
# would also redact ``sample_rate_hz``, which is a number worth keeping.
_FORBIDDEN_LOG_KEY = re.compile(
    r"(?:^|_)(?:"
    r"audio|waveform|transcripts?|transcription|"
    r"raw_bytes|raw_samples|sample_data|sample_values|pcm_data|pcm|"
    r"payloads?|blobs?|"
    r"recording|recording_id|"
    r"phones?|phone_numbers?|msisdn|account_numbers?|accountnumbers?|iban|"
    r"full_names?|customer_names?|emails?|email_addresses?|ssn|passports?"
    r")(?:$|_)",
    re.IGNORECASE,
)

_SCRUBBED = "[redacted]"

# A forbidden token followed by a unit or a measure is a *metric about* audio,
# not the audio. ``audio_seconds`` is a duration in the same category as the
# ``duration_seconds`` and ``sample_rate_hz`` beside it, and redacting it would
# hide exactly the number an operator needs to size a clip. A payload cannot
# carry a measure suffix, so this narrows the screen without opening a hole.
#
# ``_bytes`` is deliberately absent: ``audio_bytes`` names a payload, and a byte
# count is a size of that payload rather than a property of the signal.
_METRIC_KEY_SUFFIX = re.compile(
    r"(?:_seconds|_milliseconds|_ms|_hz|_samples?_(?:count|per_second)|"
    r"_count|_ratio|_percent|_frames|_db|_dbfs)$",
    re.IGNORECASE,
)


def _is_forbidden_key(key: str) -> bool:
    """Whether ``key`` may not appear in a log record.

    A forbidden token makes the key forbidden unless the key ends in a unit or
    measure suffix, which marks it as a derived statistic.
    """
    return bool(_FORBIDDEN_LOG_KEY.search(key)) and not _METRIC_KEY_SUFFIX.search(key)


# Sequences longer than this are treated as payloads rather than diagnostics.
_MAX_LOG_SEQUENCE = 32

_IDENTIFIER_VALUE_PATTERNS = (
    # A leading "(" is permitted because "(555) 123-4567" is a routine way to
    # write a number and must not be the one form that escapes screening.
    re.compile(r"^\+?\(?\d[\d\s().-]{7,}$"),
    re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,}$"),
    re.compile(r"^\d{3}-\d{2}-\d{4}$"),
    re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$"),
)

# Substrings that reveal a filesystem path to a user directory. Redacted so that
# log aggregation does not turn into a directory map of the host.
_PATH_PATTERN = re.compile(r"[A-Za-z]:\\[^\s\"']+|/(?:home|Users)/[^\s\"']+")


def _looks_like_identifier(value: str) -> bool:
    return any(pattern.match(value) for pattern in _IDENTIFIER_VALUE_PATTERNS)


def scrub_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a loggable copy of a mapping.

    Forbidden keys are dropped, values that look like direct identifiers or
    absolute host paths are replaced with ``[redacted]``, and non-JSON-serialisable
    values are replaced with their type name rather than raising.

    Args:
        payload: Candidate fields. ``None`` is treated as empty.

    Returns:
        A new dict safe to pass to a logger.
    """
    if not payload:
        return {}

    clean: dict[str, Any] = {}
    for key, value in payload.items():
        key_str = str(key)
        if _is_forbidden_key(key_str.lower()):
            clean[key_str] = _SCRUBBED
            continue
        clean[key_str] = _scrub_value(value)
    return clean


def _scrub_value(value: Any, *, depth: int = 0) -> Any:
    """Recursively scrub a single value, bounding recursion depth."""
    if depth > 6:
        return "[truncated]"

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _looks_like_identifier(value.strip()):
            return _SCRUBBED
        return _PATH_PATTERN.sub(_SCRUBBED, value)
    if isinstance(value, Mapping):
        return scrub_mapping(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        # A long numeric sequence in a log field is a payload, not a diagnostic.
        # Truncating it would still leak a fragment of the audio, so drop it whole.
        if len(value) > _MAX_LOG_SEQUENCE:
            return f"[{type(value).__name__} len={len(value)} redacted]"
        return [_scrub_value(item, depth=depth + 1) for item in value]

    # Bytes and bytearray are the shape an encoded payload takes. Never let one
    # reach a log sink, even truncated.
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"[{type(value).__name__} redacted]"

    # numpy scalars and arrays appear in audio metadata. Scalars are reduced to
    # plain Python numbers so they stay readable, but an array is *never*
    # expanded: ``tolist()`` would copy the samples themselves into the log line
    # and inflate a single event to megabytes. The shape is the part worth
    # keeping -- it answers "how long was the clip" without carrying any audio.
    if type(value).__module__.startswith("numpy"):
        if getattr(value, "ndim", None) == 0 and hasattr(value, "item"):
            return value.item()
        shape = getattr(value, "shape", None)
        if shape is not None:
            return f"[ndarray shape={tuple(shape)} dtype={getattr(value, 'dtype', '?')} redacted]"
        return f"[{type(value).__name__} redacted]"

    return f"[{type(value).__name__}]"


def safe_metadata(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Screen a metadata mapping for logging.

    Args:
        payload: Candidate metadata.

    Returns:
        A scrubbed copy.
    """
    return scrub_mapping(payload)


# Keywords that belong to the stdlib logging call rather than to the record's
# structured fields.
_LOGRECORD_PASSTHROUGH = frozenset({"exc_info", "stack_info", "stacklevel"})

# ``logging`` raises if an ``extra`` key would overwrite an existing record
# attribute, so a caller-supplied field name must never reach it unchecked.
_RESERVED_RECORD_KEYS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


class _ScreenedLogger:
    """Logger proxy that screens every log call's structured fields.

    Wrapping the logger, rather than trusting call sites, is deliberate: the
    first person to add a ``logger.info("decoded %s", header)`` should not be
    able to leak a payload by accident.
    """

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _render(
        self, message: str, kwargs: dict[str, Any]
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        """Split a call into message, screened fields, and stdlib passthrough.

        ``extra={...}`` and bare keyword fields are merged into one mapping that
        is handed to the stdlib logger *as* its ``extra``. Re-wrapping it under a
        literal ``"extra"`` key instead would bury every field one level down,
        where neither the JSON formatter nor a plain formatter can reach it --
        the fields would be silently missing from every emitted line.
        """
        fields: dict[str, Any] = {}
        passthrough: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in _LOGRECORD_PASSTHROUGH:
                passthrough[key] = value
            elif key == "extra" and isinstance(value, Mapping):
                fields.update(value)
            else:
                fields[key] = value

        message = _PATH_PATTERN.sub(_SCRUBBED, str(message))
        if not message.isprintable() and _looks_binary(message):
            message = "[non-printable message redacted]"

        screened = {
            key: item
            for key, item in scrub_mapping(fields).items()
            if key not in _RESERVED_RECORD_KEYS
        }
        return message, screened, passthrough

    def debug(self, message: str, **kwargs: Any) -> None:
        rendered, extra, passthrough = self._render(message, kwargs)
        self._logger.debug(rendered, extra=extra or None, **passthrough)

    def info(self, message: str, **kwargs: Any) -> None:
        rendered, extra, passthrough = self._render(message, kwargs)
        self._logger.info(rendered, extra=extra or None, **passthrough)

    def warning(self, message: str, **kwargs: Any) -> None:
        rendered, extra, passthrough = self._render(message, kwargs)
        self._logger.warning(rendered, extra=extra or None, **passthrough)

    def error(self, message: str, **kwargs: Any) -> None:
        rendered, extra, passthrough = self._render(message, kwargs)
        self._logger.error(rendered, extra=extra or None, **passthrough)

    def exception(self, message: str, **kwargs: Any) -> None:
        rendered, extra, passthrough = self._render(message, kwargs)
        self._logger.exception(rendered, extra=extra or None, **passthrough)


def _looks_binary(message: str) -> bool:
    return any(ord(ch) < 32 and ch not in "\t\n\r" for ch in message)


def get_logger(name: str | None = None) -> _ScreenedLogger:
    """Return a logger whose structured fields are screened.

    Args:
        name: Optional child name, e.g. ``"pipeline"``.

    Returns:
        A screening logger.
    """
    suffix = f".{name}" if name else ""
    return _ScreenedLogger(logging.getLogger(f"{_LOGGER_NAMESPACE}{suffix}"))


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per line, so log shippers can parse it."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in logging.LogRecord("", 0, "", 0, "", (), None).__dict__:
                continue
            payload[key] = _scrub_value(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        try:
            return json.dumps(payload, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            return json.dumps({"level": record.levelname, "message": "[unserialisable log record]"})


def configure_logging(level: int = logging.INFO, *, json_output: bool = True) -> None:
    """Install the VoxShield logging configuration.

    Idempotent: repeated calls replace the handler rather than stacking them,
    so a reloaded app does not duplicate every line.

    Args:
        level: Minimum level to emit.
        json_output: Emit JSON lines instead of plain text.
    """
    logger = logging.getLogger(_LOGGER_NAMESPACE)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(
        _JsonFormatter() if json_output else logging.Formatter("%(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
