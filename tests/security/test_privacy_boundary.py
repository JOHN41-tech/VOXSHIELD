"""Tests that the privacy boundary actually holds.

The central claim of VoxShield is that raw audio is not retained. These tests
attack that claim from every direction available to a careless future
contributor: a new metadata field, a log call, an error message, and the
serialised audit file itself.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Callable

import numpy as np
import pytest

from voxshield.monitoring.logging import (
    _ScreenedLogger,
    configure_logging,
    get_logger,
    safe_metadata,
    scrub_mapping,
)
from voxshield.storage import InMemoryAuditStore, JsonlAuditStore

_SESSION = str(uuid.UUID("b875c030-7664-4d7e-ad3f-b0714c09c345"))


class TestLogScrubbing:
    @pytest.mark.parametrize(
        "key",
        [
            "audio",
            "raw_audio",
            "audio_bytes",
            "waveform",
            "transcript",
            "raw_samples",
            "payload",
            "blob",
            "phone",
            "email",
        ],
    )
    def test_forbidden_keys_redacted(self, key: str) -> None:
        assert scrub_mapping({key: "secret"})[key] == "[redacted]"

    def test_sample_rate_not_redacted(self) -> None:
        # Regression guard for over-broad matching.
        assert scrub_mapping({"sample_rate_hz": 16000})["sample_rate_hz"] == 16000
        assert scrub_mapping({"source_sample_rate_hz": 44100})["source_sample_rate_hz"] == 44100

    @pytest.mark.parametrize(
        "key",
        [
            "audio_seconds",
            "duration_seconds",
            "audio_frames",
            "audio_count",
            "recording_seconds",
            "pcm_count",
            "silence_ratio",
            "peak_dbfs",
        ],
    )
    def test_measure_suffixes_are_not_treated_as_payloads(self, key: str) -> None:
        """A duration or count *about* audio is a metric, not the audio.

        Over-broad matching here is a real cost: redacting ``audio_seconds``
        hides the number an operator uses to size a clip, and a screen that
        cries wolf gets disabled.
        """
        assert scrub_mapping({key: 12.5})[key] == 12.5

    @pytest.mark.parametrize(
        "key",
        ["audio", "audio_bytes", "raw_samples", "pcm_data", "my_audio_blob"],
    )
    def test_the_narrowing_did_not_open_a_hole(self, key: str) -> None:
        """Payload names stay redacted even though a neighbouring metric does not."""
        assert scrub_mapping({key: "secret"})[key] == "[redacted]"

    @pytest.mark.parametrize("value", [b"\x00\x01", bytearray(b"x"), memoryview(b"y")])
    def test_binary_values_never_survive(self, value) -> None:
        out = scrub_mapping({"benign": value})["benign"]
        assert "redacted" in out
        assert not isinstance(out, (bytes, bytearray, memoryview))

    def test_numpy_array_is_never_expanded(self) -> None:
        # ``tolist()`` would copy the samples into the log line. The shape is the
        # diagnostic; the samples are not.
        out = scrub_mapping({"benign": np.zeros(16_000, dtype=np.float32)})["benign"]
        assert isinstance(out, str)
        assert "redacted" in out
        assert "16000" in out
        assert "0.0" not in out

    def test_numpy_scalar_becomes_a_plain_number(self) -> None:
        out = scrub_mapping({"peak": np.float32(0.25)})["peak"]
        assert out == pytest.approx(0.25)
        assert isinstance(out, float)

    def test_long_numeric_sequence_is_dropped_whole(self) -> None:
        # A list of samples is a payload whether or not it arrived as an array.
        out = scrub_mapping({"summary": [0.01 * i for i in range(4_000)]})["summary"]
        assert "redacted" in out
        assert "0.01" not in out

    def test_short_sequence_is_preserved(self) -> None:
        assert scrub_mapping({"codes": ["A", "B", "C"]})["codes"] == ["A", "B", "C"]

    def test_home_paths_are_redacted(self) -> None:
        out = scrub_mapping({"note": r"C:\Users\somebody\secret.wav"})["note"]
        assert "somebody" not in out

    def test_identifier_values_redacted(self) -> None:
        assert scrub_mapping({"note": "caller@example.com"})["note"] == "[redacted]"

    def test_nested_structures_screened(self) -> None:
        out = scrub_mapping({"outer": {"inner": {"audio": "x"}}})
        assert out["outer"]["inner"]["audio"] == "[redacted]"

    def test_safe_metadata_of_none(self) -> None:
        assert safe_metadata(None) == {}

    def test_non_serialisable_does_not_raise(self) -> None:
        class Exotic:
            pass

        assert "Exotic" in str(scrub_mapping({"thing": Exotic()})["thing"])


class TestScreenedLogger:
    @staticmethod
    def _capture(name: str) -> tuple[_ScreenedLogger, list[dict]]:
        captured: list[dict] = []

        class Sink(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured.append(record.__dict__)

        base = logging.getLogger(name)
        base.handlers.clear()
        base.addHandler(Sink())
        base.setLevel(logging.DEBUG)
        base.propagate = False
        return _ScreenedLogger(base), captured

    def test_extra_is_screened(self) -> None:
        logger, captured = self._capture("voxshield.test-scrub")
        logger.info("hello", extra={"audio": "leak", "speech_seconds": 3.2})

        record = captured[-1]
        # The payload is replaced, not dropped: the field is still present so a
        # reader can see that something was withheld.
        assert record["audio"] == "[redacted]"
        assert record["speech_seconds"] == 3.2
        assert "leak" not in record["msg"]

    def test_every_level_screens(self) -> None:
        logger, captured = self._capture("voxshield.test-levels")
        for level in ("debug", "info", "warning", "error"):
            getattr(logger, level)("m", extra={"raw_audio": "leak"})
        assert len(captured) == 4
        for record in captured:
            assert record["raw_audio"] == "[redacted]"

    def test_path_in_message_redacted(self) -> None:
        logger, captured = self._capture("voxshield.test-path")
        logger.info(r"failed reading C:\Users\johnf\audio.wav")
        assert "johnf" not in captured[-1]["msg"]

    def test_non_printable_message_redacted(self) -> None:
        logger, captured = self._capture("voxshield.test-binary")
        logger.info("head: \x00\x01\x02\x03")
        assert "redacted" in captured[-1]["msg"]

    def test_levels_available(self) -> None:
        logger = get_logger("unit-test")
        for level in ("debug", "info", "warning", "error", "exception"):
            assert callable(getattr(logger, level))


class TestNoAudioOnDisk:
    def test_pipeline_writes_nothing(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch, wav: Callable, samples: Callable
    ) -> None:
        from voxshield import prepare

        payload = wav(samples(4.0))
        monkeypatch.chdir(tmp_path)
        prepared = prepare(payload)
        assert prepared.speech_seconds > 0
        # The working directory is the most likely accidental write target, and
        # the cleanest place to assert that nothing appeared at all.
        assert list(tmp_path.rglob("*")) == []

    def test_metadata_contains_no_arrays_or_bytes(self, wav: Callable, samples: Callable) -> None:
        from voxshield import prepare

        metadata = prepare(wav(samples(4.0))).metadata()
        for key, value in metadata.items():
            assert not isinstance(value, (bytes, bytearray, memoryview)), key
            assert not isinstance(value, np.ndarray), key
        json.dumps(metadata)  # must be serialisable

    def test_audit_file_contains_no_samples(
        self, tmp_path, wav: Callable, samples: Callable
    ) -> None:
        from voxshield import prepare
        from voxshield.storage.audit import build_record, new_request_id

        prepared = prepare(wav(samples(4.0)))
        path = tmp_path / "audit.jsonl"
        store = JsonlAuditStore(path)
        store.append(
            build_record(
                request_id=new_request_id(),
                session_id=_SESSION,
                status="UNSCORED",
                model_version="unavailable",
                policy_version="mvp-1.0.0",
                latency_ms=1.0,
                metadata=prepared.metadata(),
            )
        )
        text = path.read_text(encoding="utf-8")
        assert "speech_seconds" in text
        # No long run of digits that could be encoded PCM.
        longest = max((len(run) for run in re.findall(r"\d+", text)), default=0)
        assert longest < 12


class TestDropAudioReferences:
    def test_releases_samples_and_segments(self, wav: Callable, samples: Callable) -> None:
        from voxshield import prepare

        prepared = prepare(wav(samples(4.0)))
        assert prepared.n_segments > 0
        assert prepared.preprocessed.samples.size > 0

        prepared.drop_audio_references()
        assert prepared.preprocessed.samples.size == 0
        assert prepared.n_segments == 0

    def test_metadata_survives_drop(self, wav: Callable, samples: Callable) -> None:
        # The audit record needs metadata after the buffers are released, which
        # is why routes.py captures metadata before dropping.
        from voxshield import prepare

        prepared = prepare(wav(samples(4.0)))
        prepared.drop_audio_references()
        assert "speech_seconds" in prepared.metadata()


class TestIdentifierRefusalAtApi:
    def test_non_uuid_session_rejected(self, client, wav: Callable, samples: Callable) -> None:
        response = client.post(
            "/v1/analyze/file",
            files={"audio": ("c.wav", wav(samples(4.0)), "audio/wav")},
            data={"session_id": "555-123-4567"},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_REQUEST"

    def test_name_as_session_rejected(self, client, wav: Callable, samples: Callable) -> None:
        response = client.post(
            "/v1/analyze/file",
            files={"audio": ("c.wav", wav(samples(4.0)), "audio/wav")},
            data={"session_id": "John Smith"},
        )
        assert response.status_code == 400

    def test_audit_store_never_receives_identifier(
        self, client, store: InMemoryAuditStore, wav: Callable, samples: Callable
    ) -> None:
        client.post(
            "/v1/analyze/file",
            files={"audio": ("c.wav", wav(samples(4.0)), "audio/wav")},
            data={"session_id": "555-123-4567"},
        )
        # A rejected request must not leave a partial record behind.
        assert len(store) == 0


class TestLoggingConfiguration:
    def test_configure_is_idempotent(self) -> None:
        configure_logging(logging.INFO)
        configure_logging(logging.INFO)
        assert len(logging.getLogger("voxshield").handlers) == 1

    def test_json_output_is_parseable(self, capsys) -> None:
        configure_logging(logging.INFO, json_output=True)
        get_logger("test").info("structured", extra={"speech_seconds": 1.5})
        # The handler writes to stderr.
        line = capsys.readouterr().err.strip().splitlines()[-1]
        parsed = json.loads(line)
        assert parsed["message"] == "structured"
        assert parsed["level"] == "INFO"
        assert parsed["speech_seconds"] == 1.5

    def test_json_output_scrubs_structured_fields(self, capsys) -> None:
        configure_logging(logging.INFO, json_output=True)
        get_logger("test").info("leaky", extra={"audio": "secret-payload"})
        line = capsys.readouterr().err.strip().splitlines()[-1]
        assert "secret-payload" not in line
        assert json.loads(line)["audio"] == "[redacted]"
