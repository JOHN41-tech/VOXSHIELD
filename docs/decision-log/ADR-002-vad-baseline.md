# ADR-002: Deterministic VAD baseline for Phase 0

- **Status:** Accepted
- **Date:** 2026-09-27
- **Deciders:** project owner
- **Affects:** `audio/vad.py`, `config.py` (`VadConfig`), `audio/pipeline.py`
- **Superseded by:** nothing. Superseded *when* a neural VAD lands — see
  Consequences.

## Context

Phase 0 needs to know which parts of an uploaded call contain speech, in order
to segment it and to refuse to score near-silence. The VAD therefore sits on
the critical path: a false "speech" region produces segments, features, and
ultimately a verdict about a recording that contained nothing.

It is also the first component most likely to be quietly wrong, because its
failure is a silent reduction in analysed content rather than an error.

Two properties matter more than raw accuracy at this stage.

**Gain invariance.** Callers submit audio captured on unknown devices, and
loudness normalisation to −23 dBFS happens upstream. A fixed absolute dBFS
threshold would therefore be tuned to the loudest recording in the test set and
would fail on the quietest. Telephony audio is routinely quiet.

**Failure in the safe direction.** When uncertain, the VAD must report *no*
speech. The pipeline then surfaces `INSUFFICIENT_SPEECH` and abstains. It must
never fabricate a speech region to justify a verdict, because the two errors are
not symmetric: a missed speech region yields an abstention that a human
re-examines, while a fabricated one yields a fabricated accusation.

## Decision

Implement a deterministic frame-based classifier — frame energy with an adaptive
seed threshold, plus zero-crossing rate and spectral flatness as guards — and
**not** a neural VAD.

Concretely:

- Frames of 30 ms with a 10 ms hop, via `sliding_window_view`.
- An **absolute floor** at `absolute_floor_dbfs = -55.0`. Nothing above it can be
  speech, however loud the rest of the file. This bounds the "adapt to a file
  that is 90% noise" failure mode.
- An **adaptive seed** from the signal's own frame-energy distribution: a high
  percentile of frames establishes the speech level, and a low percentile
  establishes the noise floor. The threshold sits a margin above the noise
  reference, clamped to the absolute floor.
- **Guards**: `max_zero_crossing_rate` rejects broadband noise, which has high
  zero-crossing rate at high energy; `max_spectral_flatness` rejects
  noise-like frames that pass the energy test.
- **Hysteresis** on both speech and silence duration, so a single ambiguous
  frame cannot flip a region.
- `min_speech_ratio` as a final plausibility gate.
- **`detect_speech` sanitises its own input** through
  `voxshield.audio.preprocess.sanitize`.

That last point is not incidental and is recorded here because the failure is
counter-intuitive. Without sanitisation, one `NaN` in the input poisons
`np.percentile` for the entire file, which collapses the seed arm, drops the
threshold to the absolute floor, and can mark near-silence as speech. The
pipeline already sanitises, so a defensive call inside `detect_speech` is
redundant for the normal path — and that is exactly why it is there: it makes
the function correct in isolation, which is also what makes it testable.

## Alternatives considered

**Silero-VAD.** The best available option and the right long-term choice, and it
is scheduled as a Phase 1 upgrade. Rejected for Phase 0 because it adds a model
dependency and ~1.8 MB of weights to a milestone whose explicit purpose is to
prove the pipeline with **no model**, adds a `torch` dependency to the API path,
and makes the VAD's behaviour unreproducible without the weights file. A
deterministic VAD can be reasoned about from its source; that is worth more than
a few points of accuracy before the pipeline is trusted at all.

**Pure energy threshold.** Rejected as too fragile. Gain invariance requires
adapting to the recording, and a pure fixed threshold fails exactly where
telephony audio lives.

**Energy with no guards.** Rejected. Broadband noise passes an energy test
reliably; the zero-crossing and flatness guards are what stop a noisy call being
read as speech.

**Neural VAD from the start.** Rejected for the same reason as Silero, and
because it would make the Phase 0 claim — "no model dependency" — false.

## Consequences

- Zero model dependencies and full reproducibility. Output is a pure function of
  the input, so tests assert exact values.
- Failure modes are inspectable by reading ~150 lines rather than by inspecting
  a checkpoint.
- **Accuracy is modest and unmeasured.** The absolute floor and percentile
  settings are heuristics. There is no measured `VAD false-negative rate` and
  none is claimed.
- Edge cases are handled by the safe direction: a low `speech_ratio` triggers
  `INSUFFICIENT_SPEECH` and abstention rather than a confident result.
- `sanitize` is called on every call, adding a redundant pass over the samples
  in the normal path. Accepted for correctness in isolation.
- The thresholds are environment variables, so they can be tuned per deployment
  without a code change — which also means they can be misconfigured. A
  misconfiguration presents as reduced analysis, not as a crash.
- **`SpeechMask.regions()` bounds are frame-quantised, not sample-exact.** A run
  of frames is reported as `(start * hop, end * hop)`, so a single detected frame
  reports `hop` seconds rather than the `frame_seconds` it actually covers. The
  error is bounded by one frame and is acceptable for the display and
  attribution uses it serves — the pipeline's "longest contiguous stretch"
  figure in an `INSUFFICIENT_SPEECH` message, for instance. It is **not**
  accurate enough to use as a measured duration in a report or a threshold
  comparison; use `speech_seconds` or `sample_mask` for those.
- **This ADR is expected to be superseded.** A neural VAD is a Phase 1 upgrade.
  When that happens, `docs/evaluation-protocol.md` applies to the VAD as a
  component, including the requirement that silence and near-silence must still
  route to abstention rather than to a band.
