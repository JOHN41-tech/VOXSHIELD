# Privacy design

## Principle

VoxShield processes a voice that may belong to a person who never consented to
being analysed. The design consequence is that **the safe default is to know as
little as possible, for as short as possible, about as many people as possible.**

Every control below exists to make the safe path the default path, rather than
to be switched on by an integrator who remembers.

## The four guarantees

### 1. Raw audio is never persisted

Audio exists in memory, in `numpy` arrays, for the duration of a request.

`PreparedAudio.drop_audio_references()` is called in a `finally` block in
`api/routes.py`, immediately after inference returns. The decoded array and the
normalised array become collectable at that point rather than living for the
request's full lifetime — and, critically, rather than being pinned by an
exception traceback in a failure path that nobody thought about.

`AuditRecord` has no field that can hold audio. `raw_audio_persisted: False` is
present as a field specifically so that an auditor can confirm the guarantee
**from the record itself** rather than by trusting the implementation.

### 2. Only metadata leaves the process

`PreparedAudio.metadata()` is the single representation of an analysed call that
may leave the process. It contains container, sample rate, channel count,
duration, speech seconds, speech ratio, segment count, the VAD threshold, applied
gain, and a clipped flag.

It contains no samples, no transcript, and no identity-bearing field. The
response field `audio_retained: false` states this to the caller.

A shape descriptor is deliberately *not* included for the sample array. Array
*shape* answers "how long was the clip", which is genuinely useful, and the same
value is available from `duration_seconds` without inviting the question of why
one array dimension survived redaction and another did not.

### 3. Identity is pseudonymous at the boundary

`session_id` must be a UUID, validated by `validate_session_id` before the audio
is touched — identity is checked first, so a request that is not pseudonymous is
rejected regardless of what else it contains.

A name, a phone number, or a SQL fragment never reaches the audit store. The
request is rejected with `400` and no record is written, so a rejected request
cannot leave a partial artefact that looks like an analysis somebody acted on.

Binding a pseudonymous session to a real customer is the integrating
organisation's responsibility, using data it already legitimately holds. VoxShield
never asks for that mapping.

### 4. Retention is enforced, not documented

Records carry `expires_at` and are removed by `purge_expired` after 30 days by
default. The retention window is a function parameter, not a comment.

## Logging is a security boundary

Logging is where a privacy guarantee is easiest to lose, because it is the one
place application data is copied verbatim into a system that is usually shipped
off-host.

`voxshield.monitoring.logging` therefore screens at the logger, not at the call
site. `get_logger()` returns a proxy that filters every call. The reasoning is
that the first person to add `logger.info("decoded %s", header)` should not be
able to leak a payload by accident.

What is removed:

| Input | Result | Why |
| --- | --- | --- |
| `ndarray` | `[ndarray shape=(16000,) dtype=float32 redacted]` | `tolist()` would copy the samples into the log line and inflate one event to megabytes |
| `bytes` / `bytearray` / `memoryview` | `[bytes redacted]` | The shape an encoded payload takes. Never truncate — a fragment still leaks |
| Numeric sequence longer than 32 | `[list len=4000 redacted]` | A long numeric sequence is a payload, not a diagnostic. Truncating would leak a fragment |
| Key matching the forbidden list | `[redacted]` | `audio`, `waveform`, `transcript`, `raw_samples`, `customer_name`, `phone_number`, `ssn`, … |
| Value that *is* a phone / IBAN / SSN / email | `[redacted]` | Whole-value patterns |
| Absolute path under a user directory | `[redacted]` | Log aggregation must not become a directory map of the host |
| Non-printable message | `[non-printable message redacted]` | A binary blob reaching a text sink |

Key matching is **token-bounded** (`_`-delimited whole tokens). This is
deliberate: an unbounded `sample` pattern would also redact `sample_rate_hz`,
which is the single most useful number in the whole log. That regression is
pinned by `tests/unit/test_logging.py::TestSafeMetadata::test_token_matching_preserves_useful_numeric_fields`.

### The measure suffix is an exception, and exceptions are risk

A forbidden token is normally redacted, but a key that ends in a **unit or
measure suffix** is exempt:

| Key | Key screen | Why |
| --- | --- | --- |
| `audio_seconds` | allowed | A duration. `audio_bytes` describes a payload; `audio_seconds` is a property of the signal |
| `audio_hz` | allowed | A rate |
| `raw_samples_count` | allowed | A count of samples, not the samples |
| `transcript_seconds` | allowed | A duration |
| `sample_rate_hz`, `n_samples` | allowed | Not forbidden tokens at all |
| `audio_bytes` | **redacted** | Names a payload; a byte count is a size *of* the payload |
| `raw_samples`, `waveform` | **redacted** | Forbidden tokens |

This is a real bypass surface, not a free win. `waveform_hz` is a legal key, so
a raw array logged under it passes the key screen. What stops it is the
**value** screen: arrays, byte strings, and numeric sequences over 32 entries
are redacted regardless of key. Pinned by
`test_value_screen_catches_a_payload_under_an_exempted_key`.

### Documented limitation: a short numeric payload under an unflagged key

`audio_bytes` is redacted, but bare `samples` is **not** a forbidden token, and
a numeric sequence of 32 or fewer values is not long enough to trip the length
rule. A short sample list therefore survives both screens:

```python
safe_metadata({"samples_2": [0.1] * 8})  # -> {"samples_2": [0.1, ... , 0.1]}
safe_metadata({"samples_2": [0.1] * 33})  # -> {"samples_2": "[list len=33 redacted]"}
```

This is not fixed, deliberately. Adding bare `samples` to the forbidden list
would also redact `n_samples`, the sample count an operator needs to size a
clip, and would not close the class of hole anyway — `audio_0` would still get
through. The honest framing is that the key screen is a **narrow net for
accidents**, and the load-bearing control is the caller's contract to pass
metadata rather than samples.

Asserted in
`tests/unit/test_logging.py::TestSafeMetadata::test_short_numeric_payload_under_an_unflagged_key_passes_through`
so the limitation is pinned rather than drifting into being mistaken for a
guarantee.

### Documented limitation: free text is not name-parsed

The scrubber redacts a value that *is* an identifier, and any field whose *key* is
forbidden. It does **not** run name detection over prose. A note field reading
`"John Smith called"` passes through unchanged.

This is a deliberate boundary, not an oversight. Name detection in a log scrubber
is unreliable enough that relying on it would create false confidence — the
failure would be silent and would look like a working control. The caller's
contract is to pass metadata, not a transcript, and the forbidden-key list is
what stops a transcript-shaped field from reaching a sink at all.

It is asserted in `tests/unit/test_logging.py::test_free_text_is_not_name_parsed`
so that the boundary cannot drift silently into being mistaken for a guarantee.

## Error responses do not leak internals

`api/errors.py` has an `expose_message` flag. Unexpected errors return
`"An internal error occurred."` — their messages can contain internal detail.

The decode path needed the same treatment: it was interpolating the raw
`libsndfile` exception into a client-facing message, which leaked an internal
object repr and a memory address. Those messages are now clean sentences, and
the original exception is preserved via `from exc` for the log.

## Why the strongest action is escalation

A false accusation against a genuine customer is a serious harm, not a UX
problem. The policy engine's most severe output is `escalate_to_analyst`. There
is no code path that blocks a transaction, freezes an account, or publicly
labels a caller as fraudulent.

This is a product constraint expressed in code, not a policy statement. See
`docs/decision-log/ADR-001-privacy-and-advisory-posture.md`.

## Not yet addressed

These are real gaps, stated rather than implied:

- **No encryption at rest or in transit in this repository.** The JSONL audit
  store writes plaintext. Deployment must provide disk encryption and TLS.
- **No authentication or authorisation on any endpoint.** `/v1/analyze/file` is
  unauthenticated. Phase 0 is not deployable on a network.
- **No per-customer consent model**, because there is no speaker verification and
  nothing to consent to yet.
- **No data-processing agreement or retention policy documentation** for
  compliance review. The technical controls are here; the governance artefacts
  are not.
- **The audit store is not tamper-evident.** An operator with filesystem access
  can edit `AuditRecord` lines. Defending decisions after the fact — the stated
  goal in the README — needs an append-only sink or a signature chain.
