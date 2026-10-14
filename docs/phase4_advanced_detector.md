# Phase 4 — advanced synthetic voice-cloning detector

## Status

**RUN, on a real public corpus. Measured. Negative result.** Unlike Phase 3, this
phase obtained a corpus and executed the full path, so the numbers below are
results rather than a protocol awaiting one. Two of the intended axes are **NOT
AVAILABLE** and are reported as such rather than omitted; one portability defect is
reported rather than hidden.

The headline is that the advanced model **did not beat the cheap floor**: EER
0.1414 against `mfcc_logreg`'s 0.0909. That is the honest outcome and it is
reported as the result, not as a footnote to an architecture description. What the
phase *does* demonstrate is that a fourth, structurally different family travels
the whole runner → artefact → registry path with measured, calibrated,
protocol-respecting numbers — and that the result can be a null one.

The advanced family is `logmel_crnn`: a convolutional-recurrent network over
log-mel frames, with squeeze-excitation, attention pooling, an optional
bidirectional GRU, and spec-augment. It is wired through the same runner, artefact,
and registry path as the Phase 3 baselines — not a parallel package — because a
second training path is a second place for the protocol to be violated.

## What this phase adds over Phase 3

Phase 3 answered "is a cheap model a useful floor?" with three families over
pooled MFCCs and a plain log-mel CNN. Phase 4 asks the next question: does
modelling *where* a cue sits in the spectrogram, and *which* frequency content
matters, buy anything a flat average over the time axis throws away?

| Addition | Why it is here |
| --- | --- |
| Squeeze-excitation (`SEBlock`) | Global average pooling treats every channel as equally informative, so a channel that fires on silence counts as much as one that fires on vocoder artefacts. SE learns the weighting. |
| Attention pooling (`AttentionPool`) | A spoofing cue is often localised — a handful of frames carry the fingerprint. A flat mean over 300 frames dilutes it by the frames that carry nothing. With equal weights, attention reduces to the flat mean, so it can only help in the representational sense. |
| Bidirectional GRU | Reads the frequency-collapsed frame sequence in both directions, so a cue's position in time is modelled rather than averaged away. Optional (`rnn_hidden: 0` disables it). |
| Spec-augment | Frequency and time masking during training only. The standard regulariser for a catalogue of a few thousand clips. Applied only when `self.training` — never at scoring. |

The model is deliberately small — 18,137 parameters — because the binding
constraint on this hardware is CPU time, not capacity. A larger model would not
finish inside the budget, and reporting the resulting number would be reporting a
hardware limit as a modelling result.

## Architecture

```
(B, T, F)  log-mel, e.g. (batch, 300, 80)
   │
   ├─ SpecAugment            masks freq/time bands — TRAINING ONLY
   │
   ├─ conv block ×2          Conv2d(3×3) → BatchNorm → ReLU → SEBlock → MaxPool(2×2)
   │   channels [16, 32]     (time ÷4, frequency ÷4)
   │
   ├─ mean over frequency    (B, C, T', F') → (B, T', C)
   ├─ optional bi-GRU        batch_first, hidden 32/direction
   ├─ AttentionPool          softmax over time, weighted sum
   ├─ Dropout
   └─ Linear → 2 logits
```

Collapsing frequency *before* the recurrent layer is what makes the GRU
affordable: a recurrent pass over the flattened `(time, frequency)` grid would
step thousands of inputs per frame, whereas stepping over 32 channels per frame
keeps the parameter count in the tens of thousands.

Parameter budget by configuration, measured:

| Configuration | Parameters |
| --- | --- |
| `widths=(16,32)`, SE 8, bi-GRU 32 | 18,137 |
| `widths=(16,32)`, no SE, no GRU | ~5.9k |

The `widths=(32,64)` variant used by `logmel_cnn` is larger; the CRNN's smaller
widths are a deliberate CPU-cost choice, recorded here so the comparison is not
read as a like-for-like capacity comparison.

## Protocol — non-negotiable, and enforced in code

The same contract Phase 3 established applies unchanged:

1. **train** fits parameters.
2. **dev** chooses the checkpoint epoch (dev log loss), the calibrator, and the
   operating threshold (EER). Nothing else.
3. **test** is scored exactly once, at the threshold dev selected, unmodified.

Augmentation is training-only: `SpecAugment.forward` returns the input unchanged
when `self.training` is false, so a scoring forward pass sees the clean
spectrogram. A test asserts that two consecutive score calls on one fitted model
agree exactly, which is what fails if a mask ever leaks into scoring.

Calibration, leakage gates, and sample floors are unchanged from Phase 3 and are
documented in `docs/phase3-baseline-models.md` and
`docs/evaluation-protocol.md`.

## Corpus

| Property | Value |
| --- | --- |
| Source | Hugging Face `garystafford/deepfake-audio-detection` |
| License (asserted by the dataset card) | CC BY 4.0 — recorded as `REQUIRES_VERIFICATION` |
| Emitted by | `voxshield data build`, adapter `wavefake` |
| On disk | `data/raw/wavefake/{real,fake}` — 933 + 933 FLAC, 0 read failures |
| Segments (manifest rows) | 2313 |
| Build id | `vs-3690c7cf8f068430` |
| Splits | train 1547 / dev 293 / test 473 |
| Labels | bona fide 1645 / spoof 668 |
| Audio | 6247.8 s total |

### NOT AVAILABLE: cross-generator, cross-language, cross-codec

The corpus carries no `generator_id`, so it is `UNKNOWN` for every row. Every
provenance axis derived from it — `by_generator`, `by_source_split` — is therefore
`unknown`, and the by-generator, cross-language, and cross-codec slices **cannot be
computed for this corpus**. They are reported as `NOT AVAILABLE`, not silently
dropped, because the absence is a property of the data and a reader who sees a
results table without those rows should be told why.

> A model evaluated only on a split it was tuned near tells you about that split.
> Generalisation to an unseen generator is the claim that matters and the one this
> corpus cannot support. Treat every EER below as an in-corpus number.

The leakage report for the axes that *can* be checked: parent (1327), speaker (94),
and file (3640) are all available and all show `leaked=False`. Session, generator,
language, channel, and device are unavailable and reported as such.

## Results

### Advanced family, full corpus

Measured test-split result for the full-corpus run. Every number was produced by
the runner; none is estimated. `data/records/run.json` is the authoritative
record — read it directly rather than quoting a copy of this table.

| Metric | `logmel_crnn` (this phase) | `mfcc_logreg` (Phase 3 floor) |
| --- | --- | --- |
| EER | **0.1414** | 0.0909 |
| ROC AUC | 0.9426 | 0.9658 |
| Average precision | 0.8079 | 0.9195 |
| Brier score | 0.0778 | 0.0506 |
| Brier skill score | 0.5298 | 0.6943 |
| Expected calibration error | 0.0387 | 0.0349 |
| EER threshold (calibrated) | 0.2396 | 0.372 |
| FAR / FRR at the EER point | 0.080 / 0.263 | — |
| Test n (positive) | 473 (99) | 473 (99) |
| Train / dev n | 1547 / 293 | 1547 / 293 |
| Epochs run (best) | 10 (4) | 1 |
| Fit + score wall time | 264 s | — |
| Inference, mean / P95 / RTF | 7.8 ms / 10.6 ms / 513× | — |

**The advanced model lost.** The CRNN's EER (0.1414) is worse than the
transformed-MFCC logistic regression floor's (0.0909), and it is worse on every
secondary metric. This is the result, and it is reported as such rather than
buried under the architecture description.

The most plausible reading, stated as a hypothesis rather than a finding: at 2313
segments and 668 spoof rows, a recurrent model with an attention head has far more
freedom than a standardised-MFCC vector under a linear model, and the extra
degrees of freedom cost more than the temporal structure they buy. The dev loss
reaching its best at epoch 4 of 10 is consistent with that — the model had stopped
improving well before it stopped training. Confirming it would need a
cross-generator holdout, which this corpus cannot provide.

What this phase demonstrates is that the *pipeline* carries a fourth, structurally
different family end-to-end with measured, calibrated, protocol-respecting numbers.
It does not demonstrate that the advanced architecture is better, because it is not.

### Reproducing

```powershell
# Full corpus (CPU-slow; run in the background)
python -m voxshield.cli ml train `
  --config configs/ml/logmel_crnn.yaml `
  --manifest data/manifests/all.jsonl `
  --root data --record-dir data/records --time

# Balanced subset for a quick plumbing check
python -m voxshield.cli ml train `
  --config configs/ml/logmel_crnn.yaml `
  --manifest data/manifests/smoke_balanced.jsonl `
  --root data --record-dir data/records --time
```

Do **not** use `--max-items` for a capped run on this manifest. The manifest is
written label-sorted, so the first N rows are all one class; the runner refuses
them with "the train split contains labels 1, but both classes are required", and
loosening that guard to make a capped run pass would be loosening the guard that
catches a real one. `data/manifests/smoke_balanced.jsonl` exists precisely because
a capped smoke run needs an interleaved subset.

## CPU cost

`logmel_cnn` timed out at 30 minutes on this corpus (1547 train rows). The CRNN
finished the same corpus in **264 seconds** — roughly 7× inside the budget the CNN
exceeded — because it is smaller and stops when dev loss stops improving:
`epochs: 20` with `patience: 6` stopped it at epoch 10, with the best checkpoint at
epoch 4. Inference is 7.8 ms mean / 10.6 ms P95 per 4-second clip, an RTF of 513×
on CPU.

The contrast with `logmel_cnn` is instructive and worth stating plainly: the
recurrent model is *cheaper* than the plain convolutional one, because the
frequency collapse keeps the recurrent step narrow and early stopping cuts the
epoch budget short. "More advanced" did not mean "more expensive" here.

A serious capacity comparison would still need a GPU or a much longer budget.
The 18k-parameter size is a CPU-budget decision, and the loss to the floor suggests
the budget was spent on the wrong axis — more capacity, not less, is what a
negative result like this argues for.

## Finding — the colon in `audio_path` is a Windows alternate data stream

This is a genuine defect found during Phase 4 and is reported rather than buried.

The manifest's `audio_path` for this corpus contains a colon:

```
data/processed/segments/deepfake_audio_detection/deepfake_audio_detection:<stem>__s0000.wav
```

On NTFS, `:` is the alternate-data-stream separator. Every one of the 2313 segments
is therefore an ADS (128 044 bytes — 4 s of 16 kHz PCM16 mono plus a WAV header) of
a single **0-byte container file**, `deepfake_audio_detection`. Reading works:
`soundfile` opens the colon path and `read_segment` returns `(64000,) float32`,
and all 2313 rows featurised with no decode failure. Training works in place.

It is still wrong, for reasons that do not show up on this machine:

- **Not portable.** The segments exist only as ADSs on NTFS. Copying the tree to a
  non-NTFS filesystem, zipping it, or mounting it through a network share loses
  them; `robocopy` needs `/copyall` to preserve them at all.
- **Not trackable.** Git cannot version an ADS. The corpus cannot be committed,
  which is acceptable for fetched data but means the artefacts of the build are not
  reproducible from the repository alone.
- **Fragile.** A tool that rewrites the container file can drop every stream at
  once, and the loss is silent until a decode fails.

**Recommended fix, before this phase ships:** use a filesystem-safe separator in
the on-disk filename (e.g. `deepfake_audio_detection__<stem>__s0000.wav`) and keep
the colon only in the opaque `sample_id`, which is an identifier rather than a
path. The fix belongs in segmentation/path construction, not in the manifest
reader.

## Artefacts

A `logmel_crnn` and a `logmel_cnn` artefact share the layout:

| File | Contents |
| --- | --- |
| `metadata.json` | provenance: config hash, build id, counts, fit facts, calibration, threshold |
| `weights.pt` | torch state dict |
| `state.json` | frame geometry **and** shape-affecting architecture |

`state.json` now carries `hidden_channels`, `dropout`, `se_reduction`, and
`rnn_hidden` alongside `n_frames`/`n_bands`. Every one is read back with a default
matching its pre-Phase-4 value, so a `logmel_cnn` artefact written by an earlier
build — with no `se_reduction` key — still loads. The rebuild goes through
`build_baseline`, the same factory the runner uses, so `TORCH_FAMILIES` and the
factory cannot disagree about which families exist.

## Tests

| Area | What is asserted |
| --- | --- |
| `tests/unit/test_training_models.py` | CRNN construction needs a frame shape and declares matrices; one-column spoof-positive output; separable ordering; determinism (i.e. no augmentation at score time); dev epoch selection and no-dev honesty; patience stop; `describe` parameter count; disabled-block variants build |
| `tests/unit/test_training_artifacts_runner.py` | `logmel_crnn` round-trips to identical scores; `state.json` key set is exact; a CRNN rebuilds with identical parameter count; a legacy artefact without the new keys still loads |
| `tests/unit/test_training_config_families.py` | family sets are mutually consistent and every torch family builds; new fields round-trip and validate; unknown keys still rejected; old configurations load with defaults |

Full suite: `python -m pytest` — green.

## Known limitations

1. **No cross-generator holdout.** `generator_id` is `UNKNOWN`, so the axes that
   would test generalisation are `NOT AVAILABLE`. This is the single most important
   caveat on every number here.
2. **CPU-only, small model.** Torch is `2.13.0+cpu`, `cuda.is_available()` is
   `False`. The architecture is sized to the hardware, so the result is a
   lower-bound-shaped statement about capacity, not a capacity claim.
3. **ADS portability.** See the finding above.
4. **Positive class is spoof.** FAR is a real clip called synthetic; FRR is a
   synthetic clip called real. This is stated in `docs/evaluation-protocol.md` and
   applies here unchanged.
