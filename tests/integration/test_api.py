"""End-to-end tests for the HTTP contract.

These tests are written from a caller's point of view: what a client sends,
what it gets back, and what ends up in the audit trail. The recurring theme is
honesty under uncertainty -- with no model loaded the service must say so
plainly, and must never return a score it did not compute.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable

import numpy as np
import pytest

_SESSION = str(uuid.uuid4())
_SESSION2 = str(uuid.uuid4())


def _post(client, wav: Callable, samples: Callable, seconds: float = 4.0, **data):
    return client.post(
        "/v1/analyze/file",
        files={"audio": ("call.wav", wav(samples(seconds)), "audio/wav")},
        data={"session_id": data.pop("session_id", _SESSION), **data},
    )


class TestHealth:
    def test_reports_no_model_loaded(self, client) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        # The MVP has no weights. Reporting a version here would be a lie, and a
        # caller that believes it would trust a score that is never computed.
        assert body["model_loaded"] is False
        assert body["model_version"] is None
        assert body["policy_version"]

    def test_reports_a_loaded_model(self, client_for) -> None:
        from tests.conftest import StubDetector

        client = client_for(detector=StubDetector(score=0.5))
        body = client.get("/health").json()
        assert body["model_loaded"] is True
        assert body["model_version"] == "stub-1.0.0"


class TestFormatMetadata:
    def test_publishes_the_intake_contract(self, client) -> None:
        body = client.get("/v1/meta/formats").json()
        assert "WAV" in body["allowed_containers"]
        assert "FLAC" in body["allowed_containers"]
        assert "PCM_16" in body["allowed_subtypes"]
        assert body["canonical_sample_rate_hz"] == 16_000
        assert body["max_upload_bytes"] > 0
        assert body["min_speech_seconds"] > 0

    def test_contract_matches_the_config_in_use(self, client_for, config) -> None:
        client = client_for(audit_store=None, **{})
        body = client.get("/v1/meta/formats").json()
        assert body["max_duration_seconds"] == config.max_duration_seconds


class TestAnalyzeWithoutModel:
    def test_returns_success_with_a_null_score(self, client, wav: Callable, samples: Callable) -> None:
        response = _post(client, wav, samples)
        # An unavailable model is a normal success: the service is healthy and
        # the audio was understood. A 5xx here would push callers to retry a
        # request that can never succeed.
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "UNSCORED"
        assert body["detector"]["status"] == "unavailable"
        assert body["detector"]["score"] is None
        assert body["detector"]["model_version"] is None
        assert body["detector"]["reason_code"] == "MODEL_UNAVAILABLE"
        assert body["recommended_action"]["action"] == "none"
        assert body["recommended_action"]["risk_band"] == "unknown"

    def test_never_claims_to_have_retained_audio(self, client, wav: Callable, samples: Callable) -> None:
        body = _post(client, wav, samples).json()
        assert body["audio_retained"] is False

    def test_echoes_the_pseudonymous_session(self, client, wav: Callable, samples: Callable) -> None:
        body = _post(client, wav, samples, session_id=_SESSION2).json()
        assert body["session_id"] == _SESSION2

    def test_response_is_json_serialisable(self, client, wav: Callable, samples: Callable) -> None:
        json.dumps(_post(client, wav, samples).json())

    def test_metadata_describes_the_clip(self, client, wav: Callable, samples: Callable) -> None:
        meta = _post(client, wav, samples, seconds=4.0).json()["audio_metadata"]
        assert meta["canonical_sample_rate_hz"] == 16_000
        assert meta["speech_seconds"] > 0
        assert meta["n_segments"] > 0
        assert meta["speech_ratio"] > 0.7
        assert meta["clipped"] is False
        # Absolute duration is what a caller needs to reconcile the clip with
        # their own records; it is derived from decoded frames, not the header.
        assert meta["duration_seconds"] == pytest.approx(4.0, abs=0.2)

    def test_metadata_carries_no_arrays(self, client, wav: Callable, samples: Callable) -> None:
        meta = _post(client, wav, samples).json()["audio_metadata"]
        for key, value in meta.items():
            assert not isinstance(value, (bytes, bytearray, np.ndarray, list)), key


class TestAnalyzeWithModel:
    def test_high_score_escalates(self, client_for, wav: Callable, samples: Callable) -> None:
        from tests.conftest import StubDetector

        client = client_for(detector=StubDetector(score=0.95))
        body = _post(client, wav, samples).json()
        assert body["status"] == "ANALYZED"
        assert body["detector"]["status"] == "ok"
        assert body["detector"]["score"] == pytest.approx(0.95)
        assert body["recommended_action"]["risk_band"] == "high"
        assert body["recommended_action"]["action"] == "escalate_to_analyst"

    def test_low_score_only_logs(self, client_for, wav: Callable, samples: Callable) -> None:
        from tests.conftest import StubDetector

        client = client_for(detector=StubDetector(score=0.02))
        body = _post(client, wav, samples).json()
        assert body["detector"]["score"] == pytest.approx(0.02)
        assert body["recommended_action"]["risk_band"] == "low"
        assert body["recommended_action"]["action"] == "log_only"

    def test_action_is_advisory_only(self, client_for, wav: Callable, samples: Callable) -> None:
        # Even the strongest action is a recommendation to a human. Nothing in
        # the response may read as an instruction to a downstream system.
        from tests.conftest import StubDetector

        client = client_for(detector=StubDetector(score=1.0))
        action = _post(client, wav, samples).json()["recommended_action"]
        assert action["action"] == "escalate_to_analyst"
        assert "block" not in json.dumps(action).lower()
        assert "freeze" not in json.dumps(action).lower()

    def test_broken_detector_yields_unscored_not_an_error(
        self, client_for, wav: Callable, samples: Callable
    ) -> None:
        # An operational failure is not evidence. It must not surface as a risk
        # assessment, and it must not be a 500 either.
        from tests.conftest import BrokenDetector

        client = client_for(detector=BrokenDetector())
        response = _post(client, wav, samples)
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "UNSCORED"
        assert body["detector"]["score"] is None
        assert body["recommended_action"]["action"] == "none"


class TestErrorTranslation:
    def test_missing_file_is_422(self, client) -> None:
        response = client.post("/v1/analyze/file", data={"session_id": _SESSION})
        assert response.status_code == 422

    def test_missing_session_id_is_422(self, client, wav: Callable, samples: Callable) -> None:
        response = client.post(
            "/v1/analyze/file",
            files={"audio": ("call.wav", wav(samples(4.0)), "audio/wav")},
        )
        assert response.status_code == 422

    @pytest.mark.parametrize(
        "session_id", ["", "not-a-uuid", "John Smith", "555-123-4567", "'; DROP TABLE audit; --"]
    )
    def test_non_uuid_session_rejected(
        self, client, wav: Callable, samples: Callable, session_id: str
    ) -> None:
        # Only a pseudonymous UUID is accepted. A real name, a phone number, or
        # a SQL fragment must never reach the audit store.
        response = _post(client, wav, samples, session_id=session_id)
        assert response.status_code in (400, 422)
        if response.status_code == 400:
            assert response.json()["code"] == "INVALID_REQUEST"

    def test_empty_file_rejected(self, client) -> None:
        response = client.post(
            "/v1/analyze/file",
            files={"audio": ("call.wav", b"", "audio/wav")},
            data={"session_id": _SESSION},
        )
        assert response.status_code in (400, 413, 422)

    def test_garbage_bytes_rejected(self, client) -> None:
        response = client.post(
            "/v1/analyze/file",
            files={"audio": ("call.wav", b"not audio at all" * 100, "audio/wav")},
            data={"session_id": _SESSION},
        )
        # 422, not 400: the request was well formed, its content is not
        # processable. Retrying will never help, which the code conveys.
        assert response.status_code == 422
        assert response.json()["code"] == "AUDIO_DECODE_FAILED"

    def test_error_message_hides_internal_detail(self, client) -> None:
        # The libsndfile message names internal types and object addresses. An
        # operator gets it from the chained exception in the log; a caller does
        # not get a map of our runtime.
        response = client.post(
            "/v1/analyze/file",
            files={"audio": ("call.wav", b"not audio at all" * 100, "audio/wav")},
            data={"session_id": _SESSION},
        )
        message = response.json()["message"]
        assert "0x" not in message
        assert "BytesIO" not in message
        assert "Error opening" not in message

    def test_unsupported_container_rejected(
        self, client, wav: Callable, samples: Callable
    ) -> None:
        response = client.post(
            "/v1/analyze/file",
            files={"audio": ("call.flac", wav(samples(4.0), fmt="FLAC"), "audio/flac")},
            data={"session_id": _SESSION},
        )
        # FLAC is in the default allow-list, so this must succeed; the rejection
        # path is covered by the AIFF case below.
        assert response.status_code == 200

    def test_disallowed_container_rejected(
        self, client, wav: Callable, samples: Callable
    ) -> None:
        # 415 is the honest status here: the client's media type is wrong, and
        # the /v1/meta/formats response told them so.
        response = client.post(
            "/v1/analyze/file",
            files={"audio": ("call.aiff", wav(samples(4.0), fmt="AIFF"), "audio/aiff")},
            data={"session_id": _SESSION},
        )
        assert response.status_code == 415
        assert response.json()["code"] == "UNSUPPORTED_AUDIO_FORMAT"

    def test_silence_abstains_rather_than_scoring(
        self, client_for, wav: Callable
    ) -> None:
        # Silence has no verdict. Reporting one would be an accusation against
        # an empty recording.
        from tests.conftest import StubDetector

        client = client_for(detector=StubDetector(score=0.9))
        response = client.post(
            "/v1/analyze/file",
            files={"audio": ("call.wav", wav(np.zeros(4 * 16_000, dtype=np.float32)), "audio/wav")},
            data={"session_id": _SESSION},
        )
        assert response.status_code == 422
        assert response.json()["code"] in {"INVALID_AUDIO_SIGNAL", "INSUFFICIENT_SPEECH"}

    def test_short_clip_abstains(self, client, wav: Callable, samples: Callable) -> None:
        response = _post(client, wav, samples, seconds=0.4)
        assert response.status_code == 422
        assert response.json()["code"] == "INSUFFICIENT_SPEECH"

    def test_error_body_has_a_stable_shape(self, client, wav: Callable, samples: Callable) -> None:
        body = _post(client, wav, samples, session_id="nope").json()
        assert set(body) >= {"code", "message", "request_id"}
        assert isinstance(body["code"], str)
        # A parseable code lets a client branch; a prose message lets an
        # operator act. Neither may carry the payload back.
        assert "nope" not in json.dumps(body)

    def test_unsupported_route_is_404(self, client) -> None:
        assert client.get("/v1/analyze/stream").status_code == 404


class TestAuditTrail:
    def test_every_analysis_is_recorded(
        self, client, store, wav: Callable, samples: Callable
    ) -> None:
        _post(client, wav, samples)
        records = list(store.iter_all())
        assert len(records) == 1
        record = records[0]
        assert record.session_id == _SESSION
        assert record.status == "UNSCORED"
        assert record.model_version == "unavailable"
        assert "MODEL_UNAVAILABLE" in record.reason_codes

    def test_rejected_requests_are_not_recorded(
        self, client, store, wav: Callable, samples: Callable
    ) -> None:
        # A rejected request must not leave a partial record that looks like an
        # analysis somebody acted on.
        _post(client, wav, samples, session_id="John Smith")
        assert len(store) == 0

    def test_record_metadata_matches_the_response(
        self, client, store, wav: Callable, samples: Callable
    ) -> None:
        body = _post(client, wav, samples).json()
        record = next(iter(store.iter_all()))
        assert record.request_id == body["request_id"]
        assert record.metadata["speech_seconds"] == pytest.approx(
            body["audio_metadata"]["speech_seconds"]
        )
        assert record.metadata["n_segments"] == body["audio_metadata"]["n_segments"]

    def test_no_audio_reaches_the_store(
        self, client, store, wav: Callable, samples: Callable
    ) -> None:
        _post(client, wav, samples)
        serialised = json.dumps(next(iter(store.iter_all())).as_dict(), default=str)
        assert "samples" not in serialised
        assert "waveform" not in serialised

    def test_record_asserts_no_raw_audio(self, client, store, wav: Callable, samples: Callable) -> None:
        # The guarantee is legible from the record itself, so an auditor can
        # confirm retention without reading the implementation.
        _post(client, wav, samples)
        assert next(iter(store.iter_all())).raw_audio_persisted is False

    def test_record_carries_an_expiry(self, client, store, wav: Callable, samples: Callable) -> None:
        _post(client, wav, samples)
        record = next(iter(store.iter_all()))
        assert record.expires_at > record.created_at

    def test_unique_request_id_per_call(self, client, store, wav: Callable, samples: Callable) -> None:
        _post(client, wav, samples)
        _post(client, wav, samples, session_id=_SESSION2)
        records = list(store.iter_all())
        assert len({r.request_id for r in records}) == 2
        assert records[0].session_id != records[1].session_id

    def test_latency_is_recorded(self, client, store, wav: Callable, samples: Callable) -> None:
        _post(client, wav, samples)
        assert next(iter(store.iter_all())).latency_ms > 0

    def test_high_risk_band_is_recorded(
        self, client_for, store, wav: Callable, samples: Callable
    ) -> None:
        from tests.conftest import StubDetector

        client = client_for(detector=StubDetector(score=0.95), audit_store=store)
        _post(client, wav, samples)
        assert next(iter(store.iter_all())).reason_codes == ("high",)

    def test_scored_record_names_the_model(
        self, client_for, store, wav: Callable, samples: Callable
    ) -> None:
        from tests.conftest import StubDetector

        client = client_for(detector=StubDetector(score=0.95), audit_store=store)
        _post(client, wav, samples)
        record = next(iter(store.iter_all()))
        assert record.status == "ANALYZED"
        assert record.model_version == "stub-1.0.0"

    def test_failed_detector_does_not_record_a_risk_band(
        self, client_for, store, wav: Callable, samples: Callable
    ) -> None:
        # An operational failure must not leave a band an operator could act on.
        from tests.conftest import BrokenDetector

        client = client_for(detector=BrokenDetector(), audit_store=store)
        _post(client, wav, samples)
        record = next(iter(store.iter_all()))
        assert record.status == "UNSCORED"
        assert "high" not in record.reason_codes
