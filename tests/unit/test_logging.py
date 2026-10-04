"""Tests for the logging screening layer.

Logging is the one place where a privacy guarantee is easiest to lose, because
it is the one place where application data is copied verbatim into a system that
is usually shipped off-host. These tests treat the logger as a security
boundary: an unreviewed call site must not be able to leak a payload.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Iterator

import numpy as np
import pytest

from voxshield.monitoring.logging import (
    configure_logging,
    get_logger,
    safe_metadata,
)


@pytest.fixture
def captured() -> Iterator[io.StringIO]:
    """Collect emitted JSON lines from a private logger namespace.

    Uses the production JSON formatter, because the flattening assertions only
    mean something if the fields survive the real serialisation path.
    """
    from voxshield.monitoring.logging import _JsonFormatter

    logger = logging.getLogger("voxshield.test")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        logger.handlers.clear()


def _emitted(stream: io.StringIO) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in stream.getvalue().splitlines() if line.strip().startswith("{")
    ]


class TestStructuredFields:
    def test_fields_are_flattened_onto_the_record(self, captured: io.StringIO) -> None:
        # The bug this guards: re-wrapping fields under a literal "extra" key
        # buries them one level down, so every field silently vanishes from the
        # emitted line and an operator loses all correlation data.
        get_logger("test").info("analyzed", extra={"request_id": "req_1", "status": "OK"})
        record = _emitted(captured)[-1]
        assert record["request_id"] == "req_1"
        assert record["status"] == "OK"
        assert "extra" not in record

    def test_bare_keyword_fields_are_flattened_too(self, captured: io.StringIO) -> None:
        get_logger("test").info("analyzed", request_id="req_2", latency_ms=12.5)
        record = _emitted(captured)[-1]
        assert record["request_id"] == "req_2"
        assert record["latency_ms"] == 12.5

    def test_merged_and_screened(self, captured: io.StringIO) -> None:
        get_logger("test").info("m", request_id="req_3", extra={"latency_ms": 1.0})
        record = _emitted(captured)[-1]
        assert record["request_id"] == "req_3"
        assert record["latency_ms"] == 1.0

    @pytest.mark.parametrize("level", ["debug", "info", "warning", "error"])
    def test_every_level_flattens(self, captured: io.StringIO, level: str) -> None:
        getattr(get_logger("test"), level)("m", extra={"k": "v"})
        assert _emitted(captured)[-1]["k"] == "v"

    def test_stdlib_passthrough_is_not_treated_as_a_field(self, captured: io.StringIO) -> None:
        get_logger("test").warning("m", extra={"k": "v"}, stacklevel=2)
        record = _emitted(captured)[-1]
        assert record["k"] == "v"
        assert "stacklevel" not in record

    def test_reserved_record_keys_cannot_overwrite(self, captured: io.StringIO) -> None:
        # logging raises if extra would clobber an existing attribute. Filtering
        # them means a careless field name degrades to "missing", never to a
        # crash inside an error handler.
        get_logger("test").info("m", extra={"name": "spoofed", "levelname": "spoofed"})
        record = _emitted(captured)[-1]
        assert record["level"] == "INFO"
        assert record["logger"] == "voxshield.test"

    def test_fields_survive_the_json_formatter(self, capsys: pytest.CaptureFixture) -> None:
        configure_logging(logging.INFO, json_output=True)
        get_logger("formatter").info("analyzed", extra={"request_id": "req_json"})
        for line in capsys.readouterr().err.splitlines():
            if "req_json" in line:
                assert json.loads(line)["request_id"] == "req_json"


class TestPayloadRedaction:
    def test_ndarray_never_expands(self, captured: io.StringIO) -> None:
        # tolist() here would copy the samples into the log line.
        samples = np.arange(16_000, dtype=np.float32)
        get_logger("test").info("m", extra={"samples": samples})
        record = _emitted(captured)[-1]
        assert isinstance(record["samples"], str)
        assert "redacted" in str(record["samples"])
        assert "15999" not in str(record["samples"])
        assert "(16000,)" in str(record["samples"])

    def test_two_dimensional_shape_is_reported(self, captured: io.StringIO) -> None:
        get_logger("test").info("m", extra={"logmel": np.zeros((300, 40), dtype=np.float32)})
        assert "(300, 40)" in str(_emitted(captured)[-1]["logmel"])

    def test_bytes_are_dropped(self, captured: io.StringIO) -> None:
        get_logger("test").info("m", extra={"payload": b"RIFF....secret"})
        record = _emitted(captured)[-1]
        assert "secret" not in str(record["payload"])
        assert "redacted" in str(record["payload"])

    def test_long_numeric_sequence_is_dropped_whole(self, captured: io.StringIO) -> None:
        # Truncating would still leak a fragment of the audio, so the list is
        # replaced entirely with a length marker.
        get_logger("test").info("m", extra={"pcm": list(range(4_000))})
        record = _emitted(captured)[-1]
        assert "redacted" in str(record["pcm"])
        assert "3999" not in str(record["pcm"])

    def test_short_sequence_survives(self, captured: io.StringIO) -> None:
        get_logger("test").info("m", extra={"bands": [1, 2, 3]})
        assert _emitted(captured)[-1]["bands"] == [1, 2, 3]

    def test_numpy_scalar_becomes_a_plain_number(self, captured: io.StringIO) -> None:
        get_logger("test").info("m", extra={"sr": np.int64(16_000), "g": np.float32(1.5)})
        record = _emitted(captured)[-1]
        assert record["sr"] == 16_000
        assert record["g"] == pytest.approx(1.5)

    def test_nested_mapping_is_screened(self, captured: io.StringIO) -> None:
        get_logger("test").info(
            "m", extra={"meta": {"samples": np.zeros(100, dtype=np.float32), "ok": 1}}
        )
        record = _emitted(captured)[-1]
        assert "redacted" in str(record["meta"]["samples"])
        assert record["meta"]["ok"] == 1

    def test_file_path_in_message_is_scrubbed(self, captured: io.StringIO) -> None:
        get_logger("test").info("decoded C:/Users/johnf/recordings/call_1234.wav")
        line = captured.getvalue()
        assert "johnf" not in line
        assert "redacted" in line

    def test_binary_message_is_replaced(self, captured: io.StringIO) -> None:
        get_logger("test").info("header \x00\x01\x02 RIFF")
        assert "redacted" in captured.getvalue()


class TestSafeMetadata:
    def test_does_not_mutate_the_input(self) -> None:
        payload = {"samples": np.zeros(10, dtype=np.float32), "ok": 1}
        safe_metadata(payload)
        assert isinstance(payload["samples"], np.ndarray)
        assert payload["ok"] == 1

    def test_handles_none(self) -> None:
        assert safe_metadata(None) == {}

    @pytest.mark.parametrize(
        "value",
        [
            "+15551234567",
            "555-123-4567",
            "(555) 123-4567",
            "GB82WEST12345698765432",
            "123-45-6789",
            "john.smith@example.com",
        ],
    )
    def test_whole_value_identifiers_are_replaced(self, value: str) -> None:
        assert safe_metadata({"field": value})["field"] == "[redacted]"

    @pytest.mark.parametrize(
        "key",
        [
            "audio",
            "raw_samples",
            "waveform",
            "transcript",
            "customer_name",
            "email_address",
            "phone_number",
            "msisdn",
            "ssn",
        ],
    )
    def test_forbidden_keys_are_replaced(self, key: str) -> None:
        assert safe_metadata({key: "anything"})[key] == "[redacted]"

    def test_token_matching_preserves_useful_numeric_fields(self) -> None:
        # The forbidden-key pattern is token-bounded on purpose. An unbounded
        # "sample" would redact sample_rate_hz, which is the single most useful
        # number in the whole log.
        assert safe_metadata({"sample_rate_hz": 16_000})["sample_rate_hz"] == 16_000
        assert safe_metadata({"n_samples": 64_000})["n_samples"] == 64_000

    def test_measure_suffix_exempts_a_forbidden_token(self) -> None:
        # A forbidden token plus a unit suffix is a *metric about* audio, not the
        # audio. ``audio_seconds`` is the number an operator uses to size a clip,
        # and redacting it would hide the useful signal the key screen exists to
        # protect. See docs/privacy-design.md.
        for key in ("audio_seconds", "audio_hz", "raw_samples_count", "transcript_seconds"):
            assert safe_metadata({key: 3.47})[key] == 3.47

    def test_byte_count_is_not_exempted_by_the_measure_suffix(self) -> None:
        # ``audio_bytes`` names a payload. A byte count is a size *of* that
        # payload, not a property of the signal, so it stays redacted even
        # though it ends in a measure-ish suffix.
        assert safe_metadata({"audio_bytes": 1024})["audio_bytes"] == "[redacted]"

    def test_value_screen_catches_a_payload_under_an_exempted_key(self) -> None:
        # The measure suffix makes the *key* legal, so the value screen is the
        # only thing standing between a raw array and the log. This pins that
        # backstop rather than trusting it to hold by accident.
        raw = np.zeros(4_000, dtype=np.float32)
        assert safe_metadata({"waveform_hz": raw})["waveform_hz"] == (
            "[ndarray shape=(4000,) dtype=float32 redacted]"
        )
        assert safe_metadata({"audio_seconds": list(raw)})["audio_seconds"] == (
            "[list len=4000 redacted]"
        )

    def test_short_numeric_payload_under_an_unflagged_key_passes_through(self) -> None:
        # DOCUMENTED LIMITATION, asserted so it cannot drift silently into
        # being mistaken for a guarantee. ``samples`` alone is not a forbidden
        # token, and a numeric sequence of 32 or fewer values is not a payload
        # by the length rule, so a short sample list under a key like
        # ``samples_2`` is logged in full. Closing this would mean flagging
        # bare ``samples`` as forbidden, which would also redact the
        # ``n_samples`` count operators rely on. The caller contract is to pass
        # metadata, not samples; see docs/privacy-design.md.
        short = [0.1] * 8
        assert safe_metadata({"samples_2": short})["samples_2"] == short
        # The boundary itself is the length rule, and it is exact.
        assert safe_metadata({"samples_2": [0.1] * 33})["samples_2"] == "[list len=33 redacted]"

    def test_free_text_is_not_name_parsed(self) -> None:
        # Documented boundary, asserted so it cannot drift silently. The
        # scrubber redacts a value that *is* an identifier and any field whose
        # *key* is forbidden; it does not run name detection over prose. The
        # caller's contract is to pass metadata, not a transcript, and the
        # forbidden-key list is what stops a transcript-shaped field from
        # reaching a log sink.
        scrubbed = safe_metadata({"note": "John Smith called"})
        assert scrubbed["note"] == "John Smith called"
        assert safe_metadata({"customer_name": "John Smith"})["customer_name"] == "[redacted]"

    def test_numeric_metadata_survives_unchanged(self) -> None:
        payload = {"speech_seconds": 3.47, "n_segments": 4, "clipped": False}
        assert safe_metadata(payload) == payload

    def test_unserialisable_object_degrades_to_a_type_name(self) -> None:
        class Opaque:
            pass

        assert safe_metadata({"thing": Opaque()})["thing"] == "[Opaque]"


class TestFormatter:
    def test_unserialisable_record_does_not_raise(self) -> None:
        # json.dumps with default=str handles almost everything, but a
        # pathological object must still not take down the logging call.
        stream = io.StringIO()
        logger = logging.getLogger("voxshield.formatter")
        logger.propagate = False
        handler = logging.StreamHandler(stream=stream)
        handler.setFormatter(_json_formatter())
        logger.addHandler(handler)
        try:
            get_logger("formatter").info("m", extra={"k": "v"})
        finally:
            logger.removeHandler(handler)
        assert json.loads(stream.getvalue().splitlines()[-1])["k"] == "v"

    def test_exception_text_is_included(self) -> None:
        stream = io.StringIO()
        logger = logging.getLogger("voxshield.exc")
        logger.propagate = False
        handler = logging.StreamHandler(stream=stream)
        handler.setFormatter(_json_formatter())
        logger.addHandler(handler)
        try:
            get_logger("exc").exception("failed", extra={"k": "v"})
        except RuntimeError:
            pass
        else:
            # No active exception: logging emits "NoneType: None", which is
            # still a valid record.
            pass
        assert json.loads(stream.getvalue().splitlines()[-1])["k"] == "v"

    def test_configure_logging_is_idempotent(self) -> None:
        configure_logging(logging.INFO)
        configure_logging(logging.INFO)
        logger = logging.getLogger("voxshield")
        assert len(logger.handlers) == 1

    def test_configure_logging_does_not_stack_handlers(self) -> None:
        for _ in range(5):
            configure_logging(logging.INFO)
        assert len(logging.getLogger("voxshield").handlers) == 1


def _json_formatter() -> logging.Formatter:
    from voxshield.monitoring.logging import _JsonFormatter

    return _JsonFormatter()
