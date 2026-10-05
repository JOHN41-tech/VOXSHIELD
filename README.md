# VoxShield

**AI-powered, privacy-preserving detection and prevention of voice-cloning impersonation attacks on live and near-real-time calls.**

> Status: **Phase 0 — pipeline complete, no model.** A working audio-analysis path ships in this repository. It accepts audio, measures it, and reports `UNSCORED` with `risk_band: "unknown"` because no detection model has been trained or loaded.
>
> **VoxShield cannot currently detect voice cloning.** The detection capability is Phase 1, gated on the evaluation protocol below. What Phase 0 does deliver is the whole path that carries an answer, plus the contract that keeps the system honest when it has none: no raw-audio retention, no blocked transactions, and no risk band without a model behind it.

---

## 1. Project summary

VoxShield is a platform that detects signs of synthetic or manipulated speech in telephone and VoIP calls, scores the impersonation risk, and recommends safe verification actions *before* a high-risk decision is made.

It is designed to be embedded as an API and dashboard inside existing bank, enterprise, and telecom systems — not as a standalone consumer app that a fraudster or a legitimate caller interacts with directly.

---

## 2. Problem

Modern voice-cloning and speech-synthesis tools can generate highly realistic voices from very short audio samples (seconds of audio is often enough). This collapses the assumption that "the voice on the phone is the person I know."

Fraudsters use cloned voices to impersonate:

- **Bank customers** — to authorize fund transfers or disclose account details
- **Senior executives** — to issue fraudulent instructions to staff
- **Government officials** — to extract sensitive information or unlock processes
- **Trusted contacts** — family, colleagues, and known counterparties

The attack surface is the live call itself, which means traditional controls are insufficient:

| Traditional control | Why it fails |
| --- | --- |
| Caller ID / caller name | Trivially spoofed; carries no liveness or identity guarantee |
| "Does the voice sound familiar?" | Human ears are now a weak signal; familiarity is exactly what cloning exploits |
| Manual callback procedure | Slow, inconsistently applied, and a known target for social engineering of the callback itself |

### What is actually missing

Organizations currently have no way to answer, in real time on a live call:

1. Is this speech synthetic, replayed, or genuinely live?
2. Does this voice match a *verified* profile for the person it claims to be?
3. How confident are we?
4. Given the answer, what is the **safest next action** — without disrupting a legitimate call and without falsely accusing a real person?

---

## 3. Proposed solution

VoxShield processes live or uploaded audio through an AI-based **voice-integrity pipeline** and turns the result into a calibrated risk score plus a recommended action.

### 3.1 Signals analyzed

**Acoustic and spectral analysis**
Detect artifacts and statistical fingerprints characteristic of neural vocoders and TTS output — unnatural spectral envelopes, over-smoothed formant structure, inconsistent noise floors, phase relationships that do not occur in live speech.

**Speech rhythm and prosody**
Human conversational speech has irregular timing, breath groups, and micro-variation. Synthetic speech tends toward metronomic, over-regular prosody. Detecting this is often possible even when timbre has been cloned convincingly.

**Speaker similarity against a verified profile** *(only where explicit consent and enrollment exist)*
Where a person has enrolled a verified voice profile, the claimed speaker is compared against it. This is an opt-in capability and is never applied to a voice the user has not consented to enroll.

**Permitted contextual signals**
Voice evidence is combined with transaction context that the integrating organization already legitimately holds — for example, whether the call involves a high-value transfer, a new or previously unseen beneficiary, or an unusual request pattern. Context raises or lowers risk; it never substitutes for voice evidence.

### 3.2 Risk scoring

The pipeline emits a **calibrated impersonation-risk score**. Calibration matters: a score is only useful if a given threshold corresponds to a real, measured false-positive / false-negative trade-off for the deployment.

### 3.3 Recommended actions (not automatic blocks)

VoxShield deliberately **does not automatically block a transaction and does not accuse a caller.** It recommends a proportionate, safe action:

| Risk level | Recommended action |
| --- | --- |
| **Low** | Continue the call; log the interaction for later review |
| **Medium** | Request multi-factor authentication before proceeding |
| **High** | Require a verified callback to a number on file before proceeding |
| **Critical** | Escalate to a fraud or security analyst for human decision |

The design principle: *the system informs and de-risks the human decision, it does not replace it.* A false accusation against a genuine customer is a serious harm in its own right.

### 3.4 Interfaces

- **Secure APIs** for real-time integration with bank, enterprise, and telecom call flows
- **Analyst dashboard** for reviewing flagged interactions, risk signals, and model decisions

---

## 4. Privacy and security principles

Privacy protection is a core design constraint, not a compliance afterthought.

- **In-memory processing** — audio is analyzed in memory and is not written to raw-audio storage by default
- **No raw-audio retention by default** — raw recordings are not persisted unless explicitly required for a specific, consented purpose
- **Pseudonymous session identifiers** — sessions are keyed by pseudonymous IDs so that analysis and audit trails are not inherently tied to a person's identity
- **Encryption** — data in transit and at rest is encrypted
- **Auditable model and decision records** — every model version, decision, and recommended action is recorded so outcomes can be explained and defended after the fact

---

## 5. Non-goals

These are explicitly out of scope, and are called out because they are the failure modes that would destroy trust in the product:

- ❌ Automatically blocking transactions
- ❌ Publicly or automatically accusing a caller of fraud
- ❌ Building a general-purpose voice-cloning tool
- ❌ Storing raw audio by default
- ❌ Performing speaker verification against enrolled profiles without explicit consent
- ❌ Replacing the human fraud or security analyst

---

## 6. Intended users

| User | Need |
| --- | --- |
| **Banks / financial institutions** | Prevent account takeover and unauthorized transfer authorization during calls |
| **Enterprises** | Protect high-trust internal and customer phone workflows (payments, credentials, approvals) |
| **Telecom operators / UCaaS platforms** | Offer voice-integrity scoring as an embedded network service |
| **Fraud & security analysts** | Triage flagged calls with the signal detail and audit history behind each decision |

---

## 7. Architecture (conceptual)

```
Live / uploaded audio
        │
        ▼
┌───────────────────────┐
│  Audio intake         │  in-memory, no raw-audio persistence by default
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Voice-integrity      │  acoustic/spectral · prosody · speaker similarity*
│  pipeline             │  (* opt-in, consent + enrollment only)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Context fusion       │  permitted transaction context (value, new beneficiary…)
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Calibrated risk      │  low · medium · high · critical
│  scoring              │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Action recommender   │  log · MFA · verified callback · analyst escalation
└───────────┬───────────┘
            ▼
   API response  +  analyst dashboard  +  auditable decision record
```

---

## 8. Open questions

Carried forward from the concept stage. Several are now resolved or have been
converted into a documented protocol — see `docs/`.

### Resolved or converted

- [x] **Detection model approach** — deferred to Phase 1, gated by
      [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md), which must
      be satisfied before any model is connected to the response surface.
      `docs/dataset-manifest.md` specifies the corpus it will be measured on.
- [x] **Calibration methodology and FP/FN ownership** — defined in
      [`docs/evaluation-protocol.md`](docs/evaluation-protocol.md). The current
      thresholds in `policy/evaluator.py` are uncalibrated placeholders and must
      not be shown to an end user.
- [x] **Identity model** — resolved by
      [`ADR-001`](docs/decision-log/ADR-001-privacy-and-advisory-posture.md):
      VoxShield accepts only a pseudonymous UUID session id. Binding it to a
      customer record is the integrating organisation's responsibility.
- [x] **Compliance scope** — technical controls are documented in
      [`docs/privacy-design.md`](docs/privacy-design.md); the governance
      artefacts (DPA, retention policy) remain open and are listed as gaps.
- [x] **Latency budget and degradation** — the degradation strategy is decided:
      abstain with `UNSCORED`. The budget itself is still open.

### Still open

- [ ] Which channels are in scope for v1 — PSTN/SIP only, or also WebRTC and existing UCaaS platforms?
- [ ] Reference speaker verification stack and its consent/enrollment/revocation model
- [ ] Where the pipeline runs — telco edge, customer VPC/on-prem, or cloud
- [ ] Language and accent coverage, which materially affects detector generalisation

---

## 9. Repository status

```
voxshield/
├── README.md
├── pyproject.toml
├── appsmoke.py              # end-to-end smoke check
├── src/voxshield/
│   ├── audio/               # loader · decode · preprocess · quality · vad
│   │                        # segmentation · features · pipeline · process · streaming
│   ├── api/                 # FastAPI routes, schemas, error translation
│   ├── data/                # config · schema · manifest · gates · build · adapters
│   ├── evaluation/          # metrics · reports · threshold policies
│   ├── training/            # config · features · datasets · models · artefacts · runner
│   ├── models/              # detector protocol + UnavailableDetector
│   ├── policy/              # risk bands and the advisory action ladder
│   ├── storage/             # metadata-only audit store with retention
│   ├── monitoring/          # JSON logging with privacy screening
│   └── config.py
├── tests/                   # unit, integration, privacy-boundary
├── configs/
│   ├── data.yaml
│   └── ml/                  # mfcc_logreg · mfcc_xgboost · logmel_cnn recipes
└── docs/
    ├── architecture.md            # what is actually implemented
    ├── privacy-design.md          # the four privacy guarantees
    ├── threat-model.md            # adversaries, mitigations, accepted risks
    ├── evaluation-protocol.md     # the gate a model must pass
    ├── dataset-manifest.md        # required corpus and provenance schema
    ├── phase3-baseline-models.md  # the three baselines and their method
    └── decision-log/              # ADR-001, ADR-002
```

`voxshield ml train` is implemented and exits `2` with a `not_run` record,
because no corpus exists. See
[`docs/phase3-baseline-models.md`](docs/phase3-baseline-models.md).

Not yet created: `sdk/`, `dashboard/`, `services/`. The target architecture in
section 7 is conceptual; [`docs/architecture.md`](docs/architecture.md) describes
what exists and explicitly lists what it does not contain.

### Analysing one file

`voxshield process` runs the full offline path and prints a metadata-only
report. It writes no audio, sends nothing off-host, and reports audio
measurements only — never a risk verdict, because the risk bands in
`policy/actions.py` are uncalibrated placeholders.

```powershell
python -m voxshield.cli process recording.wav
python -m voxshield.cli process recording.wav --hop 250 --short-policy keep --compact
```

`voxshield.audio.process` is also runnable directly, and delegates to the same
command rather than keeping a second copy of the exit-code contract:

```powershell
python -m voxshield.audio.process recording.wav --compact
```

After `pip install -e .` the same command is available as `voxshield process`.

Exit codes are part of the contract, not incidental:

| Code | Meaning |
| --- | --- |
| `0` | Analysed and scorable. A `CLIPPED` warning still exits 0 — the caveat travels in `quality_warning_issues` |
| `1` | Refused, never analysed: undecodable bytes, unsupported container, an audio budget exceeded, or a signal with nothing to measure (e.g. digital silence) |
| `2` | The path is not a file |
| `3` | Analysed but unscorable: an abstention, such as too little speech to fill one analysis window |

A caller can branch on refusal without parsing output. The `0`/`3` split matters
most: collapsing them teaches an operator that "send more audio" and "this file
is broken" are the same event.

### Verifying

```powershell
python -m pytest -q          # 674 tests
python -m ruff check src tests
python -m mypy src
python -X utf8 appsmoke.py 2>&1 | Select-String -NotMatch '^\{"ts"' | Select-Object -Last 8
```

Requires Python 3.12 or newer.

---

## 10. License

TBD.
