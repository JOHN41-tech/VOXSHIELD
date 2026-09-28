# ADR-001: Privacy and advisory posture

- **Status:** Accepted
- **Date:** 2026-09-27
- **Deciders:** project owner
- **Affects:** `api/routes.py`, `api/schemas.py`, `policy/actions.py`,
  `policy/evaluator.py`, `storage/audit.py`, `monitoring/logging.py`

## Context

VoxShield analyses a voice that may belong to a person who never consented to
being analysed, in order to influence a decision about money or authority.

Two questions are architectural rather than implementable, because they change
what the code is allowed to do rather than how it does it:

1. May the system's output act on a person directly?
2. What may be retained about that person?

The obvious engineering default — return a score, persist the audio, act on a
high score — maximises a demo and is wrong for this product. A false accusation
against a real customer is a serious harm, and a raw-audio retention policy
turns a detection tool into a voice-recording archive that inherits every
lawful-intercept, data-protection, and breach-notification obligation in the
deployment.

The tension is real: the more conservative the output, the less impressive the
product appears. That pressure is precisely why the choice must be recorded as a
decision rather than left to be re-litigated in each endpoint.

## Decision

**1. No output may act on a person.** The response is advisory and the strongest
action is `escalate_to_analyst`. There is no code path that blocks a
transaction, freezes an account, or publicly labels a caller as fraudulent. A
human makes the decision; the system informs it.

**2. Raw audio is never persisted.** It exists in memory for the request
lifetime and `PreparedAudio.drop_audio_references()` is called in a `finally`
block, so buffers are released on the error path too.

**3. Only metadata leaves the process.** `PreparedAudio.metadata()` is the sole
exported representation. `AuditRecord` has no field that can hold audio, and
carries `raw_audio_persisted: False` so the guarantee is auditable from the
record itself rather than by trusting the implementation.

**4. Identity is pseudonymous at the boundary.** `session_id` must be a UUID,
validated before the audio is touched. A rejected request writes no record.

**5. Retention is enforced, not documented.** Records carry `expires_at` and are
purged after 30 days by default, by `purge_expired`.

**6. Uncertainty is never rendered as a low risk.** An absent or failing
detector yields `status: UNSCORED`, `score: null`, `risk_band: "unknown"`,
`action: "none"`. Never `0.0`, never `"low"`.

**7. Unscored is a success.** The response is `200`, not `5xx`. A model that is
not deployed is a normal state, not an outage, and a client must not be told to
retry a request that can never succeed.

## Alternatives considered

**Block automatically above a critical threshold.** Rejected. It converts a
detector error into a direct harm to a named customer, and no accuracy
measurement yet justifies the autonomy. A high rate of false accusation ends the
deployment, and a block is the one action a customer experiences immediately and
irreversibly.

**Persist raw audio for later review and re-training.** Rejected for Phase 0.
The value is real and non-obvious, which is why it is a decision rather than a
rejection of the idea. It requires an explicit consent basis, a retention limit,
and a lawful-basis analysis per jurisdiction, none of which exist. `FAR`/`FRR`
measurement can proceed on features without retaining audio. If governance later
approves a consented research corpus, that is a new ADR, not a quiet default.

**Return `503` when no model is loaded.** Rejected. It conflates "not
deployed" with "broken", and would drive clients into retry and alerting loops
for a condition that will not resolve on retry.

**Return `0.0` with band `low` when no model is loaded.** Rejected outright. It
is the failure mode most likely to ship by accident, because it is the easiest
thing to return. A caller treating "unknown" as "fine" gets exactly the outcome
this product exists to prevent.

**Let the caller pass a real identity string.** Rejected. A UUID keeps the
pseudonymisation in the one place that can enforce it. Binding a session to a
customer is the integrating organisation's job, using data it already holds.

## Consequences

- The response surface is bounded and cannot be made to look more capable than
  it is. This is intentional, and it is why Phase 0 can be demonstrated
  honestly.
- Analysts receive a band and a rationale to act on, so the conservative
  posture does not make the tool useless — it moves the work to the point where a
  human is already involved.
- The audit record is defensible: it explains what was decided, on which model
  and policy version, without retaining the voice.
- Per-customer consent modelling, lawful-basis governance, and tamper-evident
  audit storage remain open. `docs/privacy-design.md` lists them as gaps rather
  than deferring them silently.
- `MEDIUM_THRESHOLD` and `HIGH_THRESHOLD` in `policy/evaluator.py` are
  uncalibrated placeholders. Under this decision they must not reach an end user
  until `docs/evaluation-protocol.md` is satisfied.
