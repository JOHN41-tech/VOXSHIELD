# Architecture

## Scope

This document describes what VoxShield **actually implements today**. It is not
the target architecture. Where the README describes an intended capability that
is not built, this document says so explicitly rather than implying it exists.

## What is built

Phase 0 delivers a complete, honest audio-analysis path with **no model**. The
system accepts audio, understands it, measures it, and says plainly that it
cannot judge it.

That is a deliberate milestone, not a stub. A voice-cloning detector whose
failure mode is a confident wrong answer is worse than one that abstains, so the
first thing to get right is the path that carries an answer: intake, decoding,
preprocessing, segmentation, feature extraction, and the reporting contract
around them.

| Module | Responsibility |
| --- | --- |
| `voxshield.config` | `AudioConfig` and its nested `VadConfig` / `FeatureConfig` / `QualityConfig` / `NormalizationConfig`, loaded from the environment with validation |
| `voxshield.audio.decode` | Bounded, container-checked decode of an upload into float32 samples |
| `voxshield.audio.loader` | `load_audio`: one entry point for encoded bytes, files, streams, and in-memory arrays |
| `voxshield.audio.preprocess` | Mono-fold, resample to 16 kHz, de-click, loudness normalise, sanitise |
| `voxshield.audio.quality` | Scalar quality report: level, peak, clipping, DC, crest factor, measurable SNR, stable issue codes |
| `voxshield.audio.vad` | Frame-based speech detection with an absolute floor and adaptive seed threshold |
| `voxshield.audio.segmentation` | Overlapping analysis windows over detected speech, configurable hop, short-window policy |
| `voxshield.audio.features` | STFT, mel filterbank, log-mel, per-utterance CMVN |
| `voxshield.audio.pipeline` | `prepare()` orchestration; owns the audio buffers and releases them |
| `voxshield.audio.process` | `process_audio()`: one offline run end to end, returning a scalar-only `ProcessResult` |
| `voxshield.audio.streaming` | `StreamingProcessor`: rolling window buffer over chunks, reusing the offline window and feature code |
| `voxshield.models.interface` | `SynthSpeechDetector` protocol, `SegmentScore`, `UnavailableDetector` |
| `voxshield.policy.evaluator` | Window scores + speech duration → risk band + recommended action |
| `voxshield.storage.audit` | Metadata-only `AuditRecord`, in-memory and JSONL stores, retention purge |
| `voxshield.monitoring.logging` | JSON logging with structured-field screening |
| `voxshield.api` | `FastAPI` app: `/health`, `/v1/meta/formats`, `/v1/analyze/file` |
| `voxshield.cli` | `serve`, `inspect`, `formats`, and `process` (analyse one file, print a metadata-only report) |

## Data flow

```
upload bytes
    │
    ▼
loader.load_audio ────────────────► DecodedAudio(samples, sample_rate, source)
    │                                 guards: container allow-list, subtype
    │                                 allow-list, channel count, declared
    │                                 duration, byte ceiling
    ▼
quality.assess_quality ──────────► QualityReport + blocking/warning issues
    │                                 (measured on decoded samples, BEFORE
    │                                  DC removal / resample / normalise)
    ▼
preprocess.preprocess_decoded ──► mono @ 16 kHz, RMS −23 dBFS,
    │                            gain_db_applied reported
    ▼
vad.detect_speech ──────────────► SpeechMask(is_speech, threshold_dbfs,
    │                             speech_seconds, speech_ratio, sample_mask)
    ▼
segmentation.segment_speech ────► list[Segment] (4 s windows, configurable hop)
    │
    ▼
features.compute_log_mel ───────► per-window log-mel (80 mels, CMVN applied)
    │
    ▼
process.process_audio ──────────► ProcessResult (scalars only, no samples)
    │
    ▼
policy.PolicyEngine.evaluate(detector, features, speech_seconds)
    │
    ├─── detector is UnavailableDetector ──► assessment=None
    │                                          status=UNSCORED, score=null,
    │                                          band="unknown", action="none"
    │
    └─── detector returns scores ──────────► risk band + advisory action
    │
    ▼
AuditRecord(metadata only) + JSON log line
```

Quality assessment runs **before** normalisation on purpose. Normalising a
clipped signal towards −23 dBFS would scale the clipped samples back into
range and erase the evidence that the recording was clipped. The same ordering
reasoning applies to DC offset and to the resampler.

## Two things that are deliberately different between the offline and streaming paths

`process_audio()` and `StreamingProcessor` share the window geometry and the
feature function, but they are not interchangeable, and pretending otherwise
would be the more convenient choice.

**1. Normalisation.** `normalize_loudness()` with the default `rms` strategy
measures RMS across the whole buffer. A stream has no whole buffer, so
`StreamingProcessor` does not attempt to reproduce it: it accepts a caller
supplied `gain_db` instead. A streaming deployment must measure level
upstream. Batch and stream feature parity is therefore only exact with
normalisation disabled, and the tests assert parity only in that
configuration rather than asserting it in general.

**2. The tail.** Offline segmentation anchors its last window to the end of
the clip, so a long clip ends with a full-width window. `flush()` cannot do
that without unbounded buffering; it emits **at most one** zero-filled final
window and does not stride over a short remainder. A caller that needs the
offline geometry must buffer, or must accept the one-window difference and
account for it.

Resampling per chunk introduces boundary artifacts at chunk seams, because a
polyphase filter has no history from the previous chunk. This is why
`StreamingProcessor` is infrastructure, not a drop-in replacement for the
offline path.

`StreamingProcessor` is **not** incremental speech detection. VAD is frame
based and buffer oriented; the streaming class windows audio, it does not
decide what is speech. The honest summary is a rolling buffer that reuses the
offline window and feature code, not a real-time pipeline.

## An abstention must name its own cause

`InsufficientSpeechError` is one exception carrying one stable reason code
(`INSUFFICIENT_SPEECH`), because the caller's action is the same in all three
cases: ask for more audio. The *message*, however, distinguishes them, and that
distinction is load-bearing. Reporting all three as "not enough speech" produced

```
Detected 2.98s of speech, below the 2.00s minimum required for a verdict.
```

on a clip that held 2.98 s of perfectly contiguous speech — a message that
contradicts its own numbers and sends an operator to solve the wrong problem.

| `reason` | Cause | What the operator should do |
| --- | --- | --- |
| `total` | Total speech below `min_speech_seconds` | Get more speech in the recording |
| `contiguous` | Enough total speech, but no window is dense enough | Get more *continuous* speech |
| `short_window` | Enough contiguous speech, but the clip is shorter than one `segment_seconds` window and `short_segment_policy` is `drop` | Send a longer recording, or score the short one with `pad` / `keep` |

The `short_window` case is the one that is easy to misdiagnose: the recording
already contains enough speech, and the refusal comes from a *policy* decision
about window geometry. All three branches are pinned in
`tests/unit/test_process.py::TestConfigurationIsHonoured`.

Note that the default `drop` policy is deliberate — it is the only policy that
never shows a model a window containing audio it did not receive — so
`short_window` abstentions are an expected consequence of a correct default, not
a defect.

## Quality and abstention

`assess_quality()` produces scalars and stable issue codes, and the codes
separate two different questions that a single "bad audio" label would
conflate:

- **Blocking** issues mean there is nothing to score. Silence below the
  amplitude floor, and non-finite samples, are refusals rather than low scores.
  A silent clip returning `scorable: true` would be the same mistake as
  rendering uncertainty as a low risk.
- **Warning** issues mean audio is scoreable but suspect. Clipping, high DC
  offset, and low measured SNR land here: the analysis is worth returning, and
  the caveat travels with it as an issue code rather than as prose.

`ProcessResult.scorable` is derived from the quality report, never set
independently. A clip that is analyzed but not scorable is an abstention with
a reason attached, which is the entire point of running quality assessment
before inference.

Note that `preprocess` also reports a `clipped` flag, and it does **not** mean
the same thing. That flag records that normalisation pushed the signal past
the peak ceiling and the result had to be attenuated to fit. Input clipping is
`quality.ISSUE_CLIPPED`. Keeping the two separate matters: conflating them
would let an attenuation report be read as evidence of a clipped recording.

## The reporting contract

This is the most important interface in Phase 0, because every later phase
depends on it and because it is what the service promises when it has nothing
useful to say.

**An absent model is a success, not an error.** `/v1/analyze/file` returns
`200` with `status: "UNSCORED"` when no detector is loaded. The alternative —
a `5xx` — would tell clients to retry a request that can never succeed, and
would conflate "the model is not deployed" with "VoxShield is broken".

`/health` reports `model_loaded: false` and `model_version: null` rather than a
placeholder string. A caller that believes it has a model version will trust a
score that is never computed.

**Uncertainty is never rendered as a low risk.** The unscored path returns
`risk_band: "unknown"`, not `"low"`. A caller that treats "unknown" as "fine"
gets exactly the failure this product exists to prevent.

**Every action is advisory.** The strongest action the policy engine can emit is
`escalate_to_analyst`. There is no code path that blocks a transaction, freezes
an account, or accuses a caller. See `docs/privacy-design.md` and
`ADR-001`.

**Silence has no verdict.** Audio below the usable amplitude floor, or with less
than `min_speech_seconds` of speech, returns `422` with a stable reason code
rather than a score. A score for an empty recording would be an accusation
against nobody.

## Why there is one feature extractor

`librosa` and `torchaudio` are deliberately absent from the dependency list.
Resampling and mel filtering are implemented in `voxshield.audio` directly.

The reasoning is a production-failure argument rather than a preference: a model
trained on one feature representation and served with another is a silent
defect. The training-time and serving-time extractors must be the same code, and
the cheapest way to guarantee that is to own the implementation. It also removes
a heavyweight transitive dependency from the API path.

`hz_to_mel` follows the HTK formulation with no `scale` parameter. The presence
or absence of that argument is not cosmetic — it changes the mapping, so it is
named explicitly in code and pinned by test.

## Configuration

`AudioConfig` is constructed from environment variables and validated on load.
Invalid values fail loudly at startup rather than producing a subtly wrong
pipeline. dBFS values accept negatives, which is required for a floor like
`absolute_floor_dbfs = -55.0`.

Defaults that shape behaviour:

| Setting | Value | Reason |
| --- | --- | --- |
| `target_sample_rate` | 16 kHz | Wideband-telephony canonical rate; also the rate a voice model expects |
| `target_rms_dbfs` | −23 | Broadcast/call loudness convention |
| `max_upload_bytes` | 10 MiB | Bounds memory before decode |
| `max_duration_seconds` | 30 | Bounds work per request |
| `max_channels` | 8 | Rejects multi-mic arrays that would defeat mono-fold |
| `segment_seconds` | 4 | Window long enough to carry prosody, short enough to localise |
| `min_segment_seconds` | 2 | Prevents padding a near-empty window to satisfy the window length |
| `allowed_formats` | WAV, FLAC | Lossless containers; deliberately excludes anything with a lossy transcode path |

`/v1/meta/formats` publishes these at runtime so a client can validate before
uploading, rather than discovering a limit after a 10 MiB transfer.

## Operational posture

- **Audit store.** `InMemoryAuditStore` warns on startup that records are lost on
  restart. Production deployments set `VOXSHIELD_AUDIT_STORE_PATH` to a
  `JsonlAuditStore`.
- **Retention.** Records carry `expires_at` and are purged after 30 days by
  default. Retention is enforced by `purge_expired`, not by convention.
- **Buffer lifetime.** `PreparedAudio.drop_audio_references()` is called in a
  `finally` block, so decoded and normalised arrays are released as soon as
  inference returns rather than living for the request's full lifetime or being
  pinned by an exception traceback.
- **Logging.** JSON lines to stderr, one object per line, with structured-field
  screening (see `docs/privacy-design.md`).

## What Phase 0 deliberately does not contain

- No trained model, no weights file, no `torch` dependency on the API path.
- No speaker verification, and no enrollment or consent store to hold profiles.
- No context fusion from the integrating organisation's transaction data.
- No calibration, and therefore no false-positive or false-negative measurement.
  The risk bands in `policy/actions.py` are uncalibrated placeholders and must not
  be presented to an end user as measured rates. See
  `docs/evaluation-protocol.md`.
- No WebRTC or live transport. `StreamingProcessor` is a rolling buffer over
  chunks a caller already has; there is no realtime ingest path, and the
  streaming class does not do incremental VAD.
- No dashboard.
- No calibrated detection. `process` in the CLI reports audio measurements and
  quality, never a risk verdict about a person.

The honest consequence: **the system cannot detect voice cloning.** It can
measure audio, and it can refuse to guess. The detection capability is gated on
the evaluation protocol in `docs/evaluation-protocol.md`, and the risk bands
that exist today are uncalibrated placeholders.
