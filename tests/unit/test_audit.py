"""Unit tests for the metadata-only audit store and its privacy guarantees.

These are the tests that matter most for VoxShield's central claim. If raw
audio or a customer identifier can reach the audit store, the product's
premise fails regardless of how good the detector is.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from voxshield.storage.audit import (
    AuditRecord,
    InMemoryAuditStore,
    JsonlAuditStore,
    assert_no_direct_identifiers,
    build_record,
    new_request_id,
    purge_expired,
    validate_session_id,
)

_SESSION = "b875c030-7664-4d7e-ad3f-b0714c09c345"


def _record(**overrides) -> AuditRecord:
    payload = {
        "request_id": new_request_id(),
        "session_id": _SESSION,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
        "expires_at": datetime(2026, 1, 31, tzinfo=UTC),
        "status": "UNSCORED",
        "model_version": "unavailable",
        "policy_version": "mvp-1.0.0",
        "latency_ms": 12.5,
        "metadata": {"speech_seconds": 3.4, "n_segments": 1},
        "reason_codes": ("MODEL_UNAVAILABLE",),
    }
    payload.update(overrides)
    return AuditRecord(**payload)


class TestIdentifierScreening:
    @pytest.mark.parametrize(
        "key",
        [
            "audio",
            "raw_audio",
            "audio_bytes",
            "waveform",
            "transcript",
            "transcription",
            "pcm_data",
            "raw_samples",
            "sample_data",
            "raw_bytes",
            "payload",
            "blob",
            "phone",
            "phone_number",
            "msisdn",
            "account_number",
            "iban",
            "full_name",
            "customer_name",
            "email",
            "ssn",
            "passport",
        ],
    )
    def test_rejects_forbidden_keys(self, key: str) -> None:
        with pytest.raises(ValueError, match="forbidden"):
            assert_no_direct_identifiers({key: "x"})

    @pytest.mark.parametrize(
        "value",
        [
            "+1 555 123 4567",
            "+441632960123",
            "GB82WEST12345698765432",
            "123-45-6789",
            "caller@example.com",
        ],
    )
    def test_rejects_identifier_shaped_values(self, value: str) -> None:
        with pytest.raises(ValueError, match="direct identifier"):
            assert_no_direct_identifiers({"notes": value})

    @pytest.mark.parametrize(
        "value",
        [
            b"\x00\x01\x02",
            bytearray(b"abc"),
            memoryview(b"abcd"),
            np.zeros(16000, dtype=np.float32),
            np.zeros((10, 80), dtype=np.float32),
        ],
    )
    def test_rejects_payload_values_regardless_of_key(self, value) -> None:
        # An innocuously named field must not be able to carry a buffer.
        with pytest.raises(ValueError, match="may not carry audio"):
            assert_no_direct_identifiers({"totally_benign": value})

    def test_rejects_a_sample_buffer_disguised_as_a_list(self) -> None:
        # ``numpy_array.tolist()`` is the obvious way to hand a payload to code
        # that only inspects for arrays and bytes. Length is the only signal
        # left, so a long flat numeric sequence must be refused.
        smuggled = [0.01 * i for i in range(16_000)]
        with pytest.raises(ValueError, match="numeric sequence"):
            assert_no_direct_identifiers({"summary": smuggled})
        with pytest.raises(ValueError, match="numeric sequence"):
            assert_no_direct_identifiers({"outer": {"summary": tuple(smuggled)}})

    def test_allows_short_numeric_sequences(self) -> None:
        # The guard must not become a blanket ban on lists: a short tuple of
        # numbers is legitimate metadata, not audio.
        assert_no_direct_identifiers({"channel_gains": [0.5, 0.25, 1.0]})
        assert_no_direct_identifiers({"bands": [1, 2, 3, 4, 5, 6, 7, 8]})

    def test_allows_sequences_of_mixed_types(self) -> None:
        # Non-numeric lists are not sample buffers and must pass through.
        assert_no_direct_identifiers({"codes": [f"CODE_{i}" for i in range(64)]})
        assert_no_direct_identifiers({"flags": [True, False] * 40})

    @pytest.mark.parametrize(
        "key",
        [
            "sample_rate_hz",
            "source_sample_rate_hz",
            "canonical_sample_rate_hz",
            "n_segments",
            "speech_seconds",
            "n_frames",
            "frames",
            "duration_seconds",
        ],
    )
    def test_allows_numeric_metadata(self, key: str) -> None:
        # Regression guard: a bare "sample" pattern would wrongly reject
        # sample_rate_hz, which is legitimate numeric metadata.
        assert_no_direct_identifiers({key: 16000})

    def test_screens_nested_structures(self) -> None:
        with pytest.raises(ValueError, match="forbidden"):
            assert_no_direct_identifiers({"outer": {"inner": {"audio": "x"}}})
        with pytest.raises(ValueError, match="forbidden"):
            assert_no_direct_identifiers({"items": [{"phone_number": "555"}]})


class TestAuditRecord:
    def test_rejects_raw_audio_persisted_true(self) -> None:
        with pytest.raises(ValueError, match="raw_audio_persisted"):
            _record(raw_audio_persisted=True)

    def test_defaults_to_no_audio_persisted(self) -> None:
        assert _record().raw_audio_persisted is False

    def test_has_no_field_capable_of_holding_audio(self) -> None:
        # Structural guarantee: there is nowhere to put samples, so there is
        # nothing to forget to redact.
        fields = set(AuditRecord.__dataclass_fields__)
        assert not fields & {"audio", "samples", "payload", "transcript", "waveform"}

    def test_as_dict_round_trips_through_json(self) -> None:
        payload = _record().as_dict()
        encoded = json.dumps(payload)
        assert "speech_seconds" in encoded
        assert payload["created_at"].startswith("2026-01-01")

    def test_as_dict_does_not_reject_own_guarantee_field(self) -> None:
        # raw_audio_persisted is a legitimate field of the envelope; screening
        # the whole dict would reject it via the "audio" token match.
        assert "raw_audio_persisted" in _record().as_dict()


class TestSessionId:
    def test_accepts_uuid(self) -> None:
        assert validate_session_id(_SESSION) == _SESSION

    def test_normalises_case(self) -> None:
        assert validate_session_id(_SESSION.upper()) == _SESSION

    @pytest.mark.parametrize(
        "value",
        ["+1 555 123 4567", "call-12345", "John Smith", "", "   ", "GB82WEST12345698765432"],
    )
    def test_rejects_non_uuid(self, value: str) -> None:
        with pytest.raises(ValueError, match="session_id"):
            validate_session_id(value)


class TestInMemoryStore:
    def test_append_and_get(self, store: InMemoryAuditStore) -> None:
        record = _record()
        store.append(record)
        assert store.get(record.request_id) == record

    def test_rejects_duplicate_request_id(self, store: InMemoryAuditStore) -> None:
        record = _record()
        store.append(record)
        with pytest.raises(ValueError, match="duplicate"):
            store.append(record)

    def test_empty_store_is_falsy_but_must_still_be_injectable(self) -> None:
        # Regression guard: InMemoryAuditStore defines __len__, so an empty
        # instance is falsy. Dependency injection must use `is None`, never
        # `or`, or an injected empty store is silently discarded.
        assert len(InMemoryAuditStore()) == 0
        assert not InMemoryAuditStore()

    def test_delete_removes_only_targets(self, store: InMemoryAuditStore) -> None:
        keep = _record()
        drop = _record()
        store.append(keep)
        store.append(drop)
        assert store.delete([drop.request_id]) == 1
        assert store.get(drop.request_id) is None
        assert store.get(keep.request_id) is not None


class TestJsonlStore:
    def test_round_trip(self, tmp_path) -> None:
        store = JsonlAuditStore(tmp_path / "audit.jsonl")
        record = _record()
        store.append(record)
        assert store.get(record.request_id) == record

    def test_creates_parent_directory(self, tmp_path) -> None:
        store = JsonlAuditStore(tmp_path / "nested" / "deep" / "audit.jsonl")
        store.append(_record())
        assert store.path.exists()

    def test_skips_corrupt_lines(self, tmp_path) -> None:
        path = tmp_path / "audit.jsonl"
        store = JsonlAuditStore(path)
        good = _record()
        store.append(good)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("{not json\n")
            handle.write("\n")
        assert store.get(good.request_id) == good

    def test_delete_rewrites_file(self, tmp_path) -> None:
        path = tmp_path / "audit.jsonl"
        store = JsonlAuditStore(path)
        keep = _record()
        drop = _record()
        store.append(keep)
        store.append(drop)
        assert store.delete([drop.request_id]) == 1
        assert store.get(drop.request_id) is None
        assert store.get(keep.request_id) == keep

    def test_file_contains_no_sample_values(self, tmp_path) -> None:
        path = tmp_path / "audit.jsonl"
        store = JsonlAuditStore(path)
        store.append(_record(metadata={"speech_seconds": 3.4, "frames": 64_000}))
        text = path.read_text(encoding="utf-8")

        assert "speech_seconds" in text
        # The real guarantee is structural, not textual: every value that lands in
        # the file must be a short scalar. A raw PCM payload would show up here
        # either as a long sequence or as a very long line.
        for line in text.splitlines():
            record = json.loads(line)
            for value in record["metadata"].values():
                assert not isinstance(value, (list, tuple, dict, bytes))
                assert len(str(value)) <= 64
        assert len(text.splitlines()) == 1

    def test_record_metadata_is_screened_again_on_write(self, tmp_path) -> None:
        # Screening at construction is not enough on its own: a store can be fed a
        # record built by another process. as_dict() re-screens on the way out.
        path = tmp_path / "audit.jsonl"
        store = JsonlAuditStore(path)
        record = _record()
        object.__setattr__(record, "metadata", {"audio_samples": list(range(4096))})
        with pytest.raises(ValueError, match="forbidden"):
            store.append(record)


class TestRetention:
    def test_default_expiry_is_30_days(self) -> None:
        record = build_record(
            request_id=new_request_id(),
            session_id=_SESSION,
            status="UNSCORED",
            model_version="unavailable",
            policy_version="mvp-1.0.0",
            latency_ms=1.0,
        )
        delta = record.expires_at - record.created_at
        assert delta == timedelta(days=30)

    def test_explicit_retention_override(self) -> None:
        record = build_record(
            request_id=new_request_id(),
            session_id=_SESSION,
            status="UNSCORED",
            model_version="unavailable",
            policy_version="mvp-1.0.0",
            latency_ms=1.0,
            retention_days=7,
        )
        assert record.expires_at - record.created_at == timedelta(days=7)

    def test_purge_expired_removes_only_stale(self, store: InMemoryAuditStore) -> None:
        now = datetime(2026, 6, 1, tzinfo=UTC)
        stale = build_record(
            request_id=new_request_id(),
            session_id=_SESSION,
            status="UNSCORED",
            model_version="unavailable",
            policy_version="mvp-1.0.0",
            latency_ms=1.0,
            retention_days=1,
            now=now - timedelta(days=2),
        )
        fresh = build_record(
            request_id=new_request_id(),
            session_id=_SESSION,
            status="UNSCORED",
            model_version="unavailable",
            policy_version="mvp-1.0.0",
            latency_ms=1.0,
            retention_days=30,
            now=now,
        )
        store.append(stale)
        store.append(fresh)
        assert purge_expired(store, now=now) == 1
        assert store.get(stale.request_id) is None
        assert store.get(fresh.request_id) == fresh

    def test_malformed_env_retention_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VOXSHIELD_AUDIT_RETENTION_DAYS", "not-a-number")
        record = build_record(
            request_id=new_request_id(),
            session_id=_SESSION,
            status="UNSCORED",
            model_version="unavailable",
            policy_version="mvp-1.0.0",
            latency_ms=1.0,
        )
        assert record.expires_at - record.created_at == timedelta(days=30)

    def test_zero_retention_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            build_record(
                request_id=new_request_id(),
                session_id=_SESSION,
                status="UNSCORED",
                model_version="unavailable",
                policy_version="mvp-1.0.0",
                latency_ms=1.0,
                retention_days=0,
            )
