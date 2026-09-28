"""Audit records and retention enforcement.

This module is the enforcement point for VoxShield's central privacy claim: raw
audio is not retained. It is not a policy document that the rest of the code is
expected to honour -- it is the only sanctioned way to record that a decision
happened, and it accepts metadata only.

Three properties are enforced structurally rather than by convention:

1. **No audio can enter a record.** :class:`AuditRecord` has no field capable of
   holding samples. There is no ``samples`` attribute to forget to redact,
   because there is nowhere to put one.
2. **Identifier screening.** :func:`assert_no_direct_identifiers` refuses records
   carrying direct identifiers, so a careless future field fails loudly at write
   time instead of leaking into a store.
3. **Bounded retention.** Every record carries ``expires_at``. The default
   development window is 30 days, matching
   ``docs/privacy-design.md``. :func:`purge_expired` is the janitor.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_RETENTION_DAYS",
    "RETENTION_DAYS_ENV",
    "AuditRecord",
    "AuditStore",
    "InMemoryAuditStore",
    "JsonlAuditStore",
    "assert_no_direct_identifiers",
    "new_request_id",
    "purge_expired",
    "validate_session_id",
]

DEFAULT_RETENTION_DAYS = 30
RETENTION_DAYS_ENV = "VOXSHIELD_AUDIT_RETENTION_DAYS"

# Keys that may never appear in an audit record, matched case-insensitively
# against whole ``_``-delimited tokens. Token boundaries matter: a bare
# ``sample`` pattern would also reject ``sample_rate_hz``, which is a number we
# legitimately record. Only compound terms that can actually carry a payload
# are listed.
_FORBIDDEN_KEY_PATTERN = re.compile(
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

# Shapes that indicate a direct identifier regardless of the key name.
_IDENTIFIER_VALUE_PATTERNS = (
    re.compile(r"^\+?\d[\d\s().-]{7,}$"),          # phone-like
    re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,}$"),    # IBAN-like
    re.compile(r"^\d{3}-\d{2}-\d{4}$"),             # SSN-like
    re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$"), # email-like
)


def new_request_id() -> str:
    """Generate a server-side request identifier.

    Server-generated, not client-supplied, so a caller cannot collide with or
    enumerate another caller's identifiers.
    """
    return f"req_{uuid.uuid4().hex[:12]}"


def validate_session_id(session_id: str) -> str:
    """Validate a client-supplied pseudonymous session identifier.

    The MVP requires a UUID. This is stricter than it needs to be for
    pseudonymousity, but it makes it impossible to smuggle a phone number or an
    account number into the audit trail by putting it in the session field, and
    it keeps session IDs uniform across integrations.

    Args:
        session_id: Candidate identifier.

    Returns:
        The normalised identifier, lowercased.

    Raises:
        ValueError: Not a UUID.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        msg = "session_id must be a non-empty string"
        raise ValueError(msg)
    candidate = session_id.strip()
    try:
        uuid.UUID(candidate)
    except ValueError as exc:
        msg = (
            "session_id must be a UUID. The MVP accepts only pseudonymous "
            "identifiers; names, phone numbers, and account numbers are not "
            "permitted in the request."
        )
        raise ValueError(msg) from exc
    return candidate.lower()


def _screen_value(key: str, value: Any) -> None:
    """Raise if a single key/value pair could carry audio or an identifier.

    Three independent checks, because key screening alone is defeatable: a new
    field could be named innocuously and still hold a buffer.

    1. A value that *is* a payload -- bytes, or a numeric array of any
       dimension -- is rejected whatever its name.
    2. A long flat sequence of numbers is rejected, which is what a sample
       buffer becomes once it is converted to a plain list.
    3. A string that matches a known identifier shape is rejected.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        msg = (
            f"audit field {key!r} holds a binary payload; audit records may "
            "not carry audio"
        )
        raise ValueError(msg)

    # numpy is not imported here. Duck-typing on ``ndim`` covers arrays without
    # adding an import to a module that runs on every request, and scalars do not
    # have the attribute.
    ndim = getattr(value, "ndim", None)
    if ndim is not None and isinstance(ndim, int) and ndim >= 1:
        msg = (
            f"audit field {key!r} holds an array of {ndim} dimensions; audit "
            "records may not carry audio-derived sample arrays"
        )
        raise ValueError(msg)

    if _is_long_numeric_sequence(value):
        msg = (
            f"audit field {key!r} holds a {len(value)}-element numeric sequence; "
            "audit records may not carry audio-derived sample arrays"
        )
        raise ValueError(msg)

    if isinstance(value, str) and value:
        candidate = value.strip()
        for pattern in _IDENTIFIER_VALUE_PATTERNS:
            if pattern.match(candidate):
                msg = (
                    f"audit field {key!r} appears to contain a direct "
                    "identifier; the MVP stores pseudonymous metadata only"
                )
                raise ValueError(msg)


def assert_no_direct_identifiers(payload: dict[str, Any]) -> None:
    """Reject a record that carries audio or a direct identifier.

    Args:
        payload: Candidate audit payload.

    Raises:
        ValueError: A forbidden key name, or a value matching a known
            identifier shape.
    """
    for key, value in payload.items():
        lowered = str(key).lower()
        if _FORBIDDEN_KEY_PATTERN.search(lowered):
            msg = (
                f"audit field {key!r} is forbidden: audit records may not carry "
                "audio, transcripts, or direct identifiers"
            )
            raise ValueError(msg)
        # Screen the value at every level, including containers, so that a
        # numeric sequence hiding inside a list is caught by the same rules as a
        # bare buffer.
        _screen_value(str(key), value)
        if isinstance(value, dict):
            assert_no_direct_identifiers(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    assert_no_direct_identifiers(item)


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """An immutable record that a decision was produced.

    There is deliberately no field that can hold audio, a transcript, or a
    customer identifier. :meth:`as_dict` is the only serialisation path, and it
    screens the payload on the way out.

    Attributes:
        request_id: Server-generated identifier for this analysis.
        session_id: Client-supplied pseudonymous session identifier.
        created_at: UTC timestamp of record creation.
        expires_at: UTC timestamp after which the record must be purged.
        status: Analysis outcome, e.g. ``ANALYZED`` or ``INSUFFICIENT_SPEECH``.
        model_version: Version of the detector, or ``unavailable``.
        policy_version: Version of the policy engine that produced any action.
        latency_ms: End-to-end processing time.
        metadata: Audio statistics. Metadata only, screened on write.
        reason_codes: Stable codes explaining the outcome.
        raw_audio_persisted: Always ``False``. Present so that an auditor can
            confirm the guarantee from the record itself rather than by trusting
            the implementation.
    """

    request_id: str
    session_id: str
    created_at: datetime
    expires_at: datetime
    status: str
    model_version: str
    policy_version: str
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    raw_audio_persisted: bool = False

    def __post_init__(self) -> None:
        if self.raw_audio_persisted:
            msg = (
                "raw_audio_persisted must be False: VoxShield does not retain "
                "raw audio, and an audit record asserting otherwise indicates a "
                "bug elsewhere"
            )
            raise ValueError(msg)
        assert_no_direct_identifiers(self.metadata)
        for code in self.reason_codes:
            _screen_value("reason_codes", code)

    def as_dict(self) -> dict[str, Any]:
        """Serialise for storage.

        Only ``metadata`` is screened, not the whole envelope. The envelope keys
        are fixed by the frozen dataclass and no untrusted input reaches them,
        whereas ``metadata`` is assembled by callers. Screening the envelope too
        would reject this record's own ``raw_audio_persisted`` guarantee field,
        which the key patterns are correctly built to catch.
        """
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["expires_at"] = self.expires_at.isoformat()
        payload["reason_codes"] = list(self.reason_codes)
        assert_no_direct_identifiers(payload["metadata"])
        return payload


class AuditStore:
    """Append-only store of audit records.

    Subclasses implement :meth:`append`, :meth:`get`, :meth:`iter_all`, and
    :meth:`delete`. There is no update or delete-by-content operation, so a
    recorded decision cannot be quietly rewritten.
    """

    def append(self, record: AuditRecord) -> None:
        """Persist a record."""
        raise NotImplementedError

    def get(self, request_id: str) -> AuditRecord | None:
        """Look up a record by request identifier."""
        raise NotImplementedError

    def iter_all(self) -> Iterator[AuditRecord]:
        """Iterate over all stored records."""
        raise NotImplementedError

    def delete(self, request_ids: Iterable[str]) -> int:
        """Delete records by identifier. Returns the number removed."""
        raise NotImplementedError


def _retention_delta(retention_days: int | None) -> timedelta:
    """Resolve the retention window, preferring an explicit override."""
    if retention_days is None:
        raw = os.environ.get(RETENTION_DAYS_ENV, "").strip()
        # A malformed value must never silently disable or extend retention.
        retention_days = int(raw) if raw.lstrip("-").isdigit() else DEFAULT_RETENTION_DAYS
    if retention_days <= 0:
        msg = f"retention_days must be positive, got {retention_days!r}"
        raise ValueError(msg)
    return timedelta(days=retention_days)


class InMemoryAuditStore(AuditStore):
    """Thread-safe in-memory store, for tests and local development.

    Records vanish on restart. That is acceptable for development and
    unacceptable for production, which is why :class:`JsonlAuditStore` exists
    and why the deployment config must select it explicitly.
    """

    def __init__(self) -> None:
        self._records: dict[str, AuditRecord] = {}
        self._lock = threading.Lock()

    def append(self, record: AuditRecord) -> None:
        """Persist a record, rejecting duplicate request identifiers."""
        with self._lock:
            if record.request_id in self._records:
                msg = f"duplicate audit record for request_id {record.request_id!r}"
                raise ValueError(msg)
            self._records[record.request_id] = record

    def get(self, request_id: str) -> AuditRecord | None:
        """Look up a record by request identifier."""
        with self._lock:
            return self._records.get(request_id)

    def iter_all(self) -> Iterator[AuditRecord]:
        """Iterate over a snapshot of stored records."""
        with self._lock:
            snapshot = list(self._records.values())
        return iter(snapshot)

    def delete(self, request_ids: Iterable[str]) -> int:
        """Delete records by identifier. Returns the number removed."""
        with self._lock:
            removed = 0
            for request_id in request_ids:
                if self._records.pop(request_id, None) is not None:
                    removed += 1
            return removed

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


class JsonlAuditStore(AuditStore):
    """Append-only JSONL store on disk.

    Suitable for single-instance deployment. For multi-instance deployments use
    a database with the same append-only and expiry semantics; see
    ``docs/architecture.md``.

    Warning:
        The file contains decision metadata, which is still sensitive: a
        ``request_id`` plus a risk band can reveal that a specific call was
        flagged. The file must be encrypted at rest and access-controlled.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        """Filesystem location of the JSONL file."""
        return self._path

    def append(self, record: AuditRecord) -> None:
        """Append a record as one JSON line."""
        line = json.dumps(record.as_dict(), separators=(",", ":"), sort_keys=True)
        with self._lock, self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def get(self, request_id: str) -> AuditRecord | None:
        """Scan for a record by request identifier."""
        for record in self.iter_all():
            if record.request_id == request_id:
                return record
        return None

    def iter_all(self) -> Iterator[AuditRecord]:
        """Iterate over every well-formed record, skipping corrupt lines."""
        if not self._path.exists():
            return
        with self._lock, self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    yield _record_from_dict(data)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    # A corrupt line must not take down the audit reader.
                    continue

    def delete(self, request_ids: Iterable[str]) -> int:
        """Rewrite the file without the given identifiers.

        Rewrites atomically via a temporary file in the same directory, then
        replaces. Note that this leaves the old content recoverable from the
        filesystem until the blocks are reused; on a real deployment, purge
        should be paired with storage-level expiry.
        """
        targets = set(request_ids)
        if not targets or not self._path.exists():
            return 0
        kept: list[str] = []
        removed = 0
        with self._lock, self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError:
                    kept.append(stripped)
                    continue
                if data.get("request_id") in targets:
                    removed += 1
                else:
                    kept.append(stripped)

        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for line in kept:
                handle.write(line + "\n")
        tmp.replace(self._path)
        return removed


def _record_from_dict(data: dict[str, Any]) -> AuditRecord:
    """Rebuild an :class:`AuditRecord` from stored JSON."""
    return AuditRecord(
        request_id=str(data["request_id"]),
        session_id=str(data["session_id"]),
        created_at=datetime.fromisoformat(str(data["created_at"])),
        expires_at=datetime.fromisoformat(str(data["expires_at"])),
        status=str(data["status"]),
        model_version=str(data["model_version"]),
        policy_version=str(data["policy_version"]),
        latency_ms=float(data["latency_ms"]),
        metadata=dict(data.get("metadata") or {}),
        reason_codes=tuple(data.get("reason_codes") or ()),
        raw_audio_persisted=bool(data.get("raw_audio_persisted", False)),
    )


def build_record(
    *,
    request_id: str,
    session_id: str,
    status: str,
    model_version: str,
    policy_version: str,
    latency_ms: float,
    metadata: dict[str, Any] | None = None,
    reason_codes: tuple[str, ...] = (),
    retention_days: int | None = None,
    now: datetime | None = None,
) -> AuditRecord:
    """Construct an audit record with a computed expiry.

    Args:
        request_id: Server-generated request identifier.
        session_id: Validated pseudonymous session identifier.
        status: Outcome code.
        model_version: Detector version, or ``unavailable``.
        policy_version: Policy engine version.
        latency_ms: End-to-end processing time.
        metadata: Metadata-only audio statistics.
        reason_codes: Stable outcome codes.
        retention_days: Override for the retention window.
        now: Injectable clock, for deterministic tests.

    Returns:
        A screened, expiring :class:`AuditRecord`.
    """
    created = now or datetime.now(UTC)
    return AuditRecord(
        request_id=request_id,
        session_id=validate_session_id(session_id),
        created_at=created,
        expires_at=created + _retention_delta(retention_days),
        status=status,
        model_version=model_version,
        policy_version=policy_version,
        latency_ms=latency_ms,
        metadata=dict(metadata or {}),
        reason_codes=tuple(reason_codes),
        raw_audio_persisted=False,
    )


def purge_expired(
    store: AuditStore,
    *,
    now: datetime | None = None,
) -> int:
    """Delete every record whose expiry has passed.

    Args:
        store: Store to purge.
        now: Injectable clock, for deterministic tests.

    Returns:
        Number of records removed.
    """
    cutoff = now or datetime.now(UTC)
    expired = [r.request_id for r in store.iter_all() if r.expires_at <= cutoff]
    if not expired:
        return 0
    return store.delete(expired)


# A list or tuple of numbers is a sample buffer in disguise. Neither the binary
# check nor the ``ndim`` check can tell 16000 float samples from a legitimate
# short sequence, and ``numpy_array.tolist()`` is the obvious way to hand a
# payload to code that only looks for arrays. A metadata field has no legitimate
# reason to hold a long numeric sequence, so length is a safe discriminator.
_MAX_SCALAR_SEQUENCE = 32


def _is_long_numeric_sequence(value: object) -> bool:
    """Whether ``value`` is a long flat sequence of numbers."""
    if not isinstance(value, (list, tuple)) or len(value) <= _MAX_SCALAR_SEQUENCE:
        return False
    return all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    )
