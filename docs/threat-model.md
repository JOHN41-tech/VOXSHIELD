# Threat model

## Scope

Phase 0 accepts a file upload and returns an advisory response. This model
covers that surface. Streaming, WebRTC, the dashboard, speaker verification, and
context fusion are out of scope because they do not exist yet.

The asset being protected is not the audio. It is the **integrity of a decision
about a person** — specifically, the risk that a real customer is falsely
accused of fraud, or that a cloned voice passes unchallenged, and that either
error is presented to a human as a measured fact.

## Adversaries

| # | Adversary | Capability | Goal |
| --- | --- | --- | --- |
| A1 | Fraudster | Controls the audio on the call | Have a cloned voice pass as genuine |
| A2 | Fraudster | Controls the audio and tunes it to the detector | Evade a known detector |
| A3 | Hostile caller | Crafts the upload | Crash, hang, or exhaust the service |
| A4 | Hostile caller | Crafts the upload | Reach a code path that logs or stores audio |
| A5 | Insider | Has log or audit-store read access | Recover audio or identity from the record |
| A6 | Insider | Has log or audit-store write access | Alter a decision record after the fact |
| A7 | Well-meaning integrator | Misconfigures the deployment | Break a privacy or honesty guarantee |
| A8 | Automated client | No malice | Act on a response that is not what it appears to be |

## Threats and mitigations

### T1 — False accusation of a genuine caller

**A1 / model error.** The highest-severity threat in the product. A real
customer is labelled a fraudster.

Mitigations in place: aggregation uses the **mean** of window scores, not the
max, precisely so one spurious window cannot become a call-level verdict. No
action blocks a transaction. The strongest output is `escalate_to_analyst`.

**Not yet mitigated:** the risk bands in `policy/actions.py` are uncalibrated.
They are not measured false-positive rates and must not be presented as such.
Until `docs/evaluation-protocol.md` is executed, no end user should see a band
at all.

### T2 — Absent model reported as a negative result

**A8.** A caller sees an unavailable model and treats "no score" as "no risk".

This is the failure mode most likely to be shipped by accident, because every
naive implementation returns `0.0` or `"low"` when it has nothing to say.

Mitigations in place: `UNSCORED` status, `score: null`,
`risk_band: "unknown"`, `action: "none"`, `reason_code: MODEL_UNAVAILABLE`,
`model_version: null` on `/health`. An unavailable model returns `200`, not `5xx`,
so a client does not retry a request that can never succeed and does not mistake
the state for an outage.

### T3 — Operational failure rendered as a risk assessment

**A2, by degrading the detector.** A corrupt weights file, an OOM during
inference, a timeout.

Mitigation in place: a raising detector is caught and produces the same
`UNSCORED` path as an absent one. A failure must not surface as a risk band,
because a record saying "high" that was actually produced by an exception is
worse than no record.

### T4 — Resource exhaustion via upload

**A3.** A large or long file to exhaust memory or CPU.

Mitigations in place: a byte ceiling (10 MiB) checked against the declared
`Content-Length` *before* the body is buffered, and again while reading, with a
hard stop. Bounded decode with no full-file read. A declared-duration and
channel-count ceiling checked at header parse time, before sample decoding. The
one gap is CPU: there is no timeout on `sf.read` itself, and a
compression-bomb-shaped file inside the byte ceiling could still be slow.

### T5 — Malformed or hostile audio

**A3 / A4.** Crafted headers, absurd channel counts, zero-length frames, NaN and
Inf in the payload.

Mitigations in place: every header field is validated before use. `sanitize()`
replaces non-finite samples. `detect_speech` sanitises its own input as well —
without that, a single NaN poisons `np.percentile` for the whole file, collapses
the seed arm, drops the threshold to the absolute floor, and can mark near-silence
as speech. Containers and subtypes are checked against an allow-list.

### T6 — Audio or identity reaching a log sink

**A4 / A5.**

Mitigations in place: screening at the logger proxy rather than at call sites, so
a new call site cannot leak by accident. Forbidden keys, whole-value identifier
patterns, array/bytes/long-sequence redaction, and host-path redaction. See
`docs/privacy-design.md` for the full table and the documented free-text
limitation.

### T7 — Internal detail in client-facing errors

**A3.** Fingerprinting the runtime through error text.

Mitigation in place: `expose_message=False` for unexpected errors. The decode
path's raw `libsndfile` exception text — which contained an object repr and a
memory address — was removed from the response and kept on the chained exception
for logs.

### T8 — Non-pseudonymous identity reaching the audit store

**A4 / A5.** A request carrying a real name or account number.

Mitigations in place: `validate_session_id` runs before the audio is touched and
requires a UUID. A rejected request writes no record.
`assert_no_direct_identifiers` screens the metadata payload, and long numeric
sequences are treated as payloads rather than diagnostics.

### T9 — Decision record altered after the fact

**A6.** The README's stated goal is that outcomes can be explained and defended
later. A mutable plaintext JSONL file does not support that.

**Not mitigated.** The audit store is not tamper-evident. Closing this needs an
append-only sink or a signature chain. Listed here because it is a real gap in the
stated value proposition, not because it is fixed.

### T10 — Misconfiguration silently weakening a guarantee

**A7.** An integrator sets an environment variable that turns a hard gate into a
no-op, or points the audit store somewhere unprotected.

Mitigations in place: configuration is validated on load and fails loudly at
startup rather than producing a subtly wrong pipeline. dBFS values accept
negatives, so a floor like `-55.0` is expressible rather than being silently
clamped. `InMemoryAuditStore` warns at startup that records are lost on restart.

**Not mitigated:** no authentication on any endpoint, and no encryption at rest
or in transit in this repository. Phase 0 must not be exposed to a network.

## Accepted risks

- **A1 can currently succeed.** Phase 0 has no detector. This is the defining
  limitation of the milestone and is stated in `docs/architecture.md` rather than
  papered over.
- **False-positive cost is unquantified.** The policy thresholds are
  placeholders. Shipping them to end users without calibration would misrepresent
  them.
- **Detector evasion is unstudied.** No adversarial-robustness evaluation exists.
  A detector that is trivially evaded is worse than none, because it manufactures
  confidence.

## Verification

| Threat | Test |
| --- | --- |
| T1 | `tests/unit/test_segmentation.py::TestAggregateSegmentScores::test_uses_the_mean_not_the_max` |
| T2 | `tests/integration/test_api.py::TestAnalyzeWithoutModel::*`, `::TestHealth::test_reports_no_model_loaded` |
| T3 | `tests/integration/test_api.py::TestAnalyzeWithModel::test_broken_detector_yields_unscored_not_an_error` |
| T4 | `tests/unit/test_decode.py` — size and duration limits |
| T5 | `tests/unit/test_decode.py` — hostile input; `tests/unit/test_vad.py::TestDetectSpeech::test_non_finite_input_does_not_crash` |
| T6 | `tests/unit/test_logging.py::TestPayloadRedaction::*` |
| T7 | `tests/integration/test_api.py::TestErrorTranslation::test_error_message_hides_internal_detail` |
| T8 | `tests/integration/test_api.py::TestErrorTranslation::test_non_uuid_session_rejected`, `tests/security/test_privacy_boundary.py` |
| T9, T10 | Open — see above |
