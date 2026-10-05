# Phase 3 baseline models

## Status

**NOT RUN.** No baseline has been trained. This repository ships no audio, no
manifests, and no cached artefacts, so every performance figure below is a
protocol to be executed rather than a result that was obtained.

What *does* exist is the full training and evaluation path: three model
families, a manifest-backed data layer, a protocol-enforcing runner, artefact
persistence, and a CLI. That code is verified against synthetic fixtures, which
proves it wires together correctly and says nothing whatsoever about detection
accuracy. Those two statements are kept separate throughout this document
because conflating them is the most likely way this phase goes wrong.

| Figure | Value | Why |
| --- | --- | --- |
| EER | NOT RUN | Requires a corpus with disjoint speakers across splits |
| ROC AUC | NOT RUN | Requires the same |
| FAR / FRR at an operating point | NOT RUN | Requires the same |
| Latency, RTF | NOT RUN | Requires real audio of realistic duration |
| Calibration error | NOT RUN | Requires the same |
| Subgroup deltas | NOT RUN | Requires subgroup-bearing manifests |

`voxshield ml train` exits `2` on this repository as it stands, prints a
well-formed `not_run` record, and invents nothing. That is the intended
behaviour, not a defect.

## Why three trivial models first

Three families are in scope, and the point of running all three is attribution,
not a leaderboard:

| Family | Front end | Model | Role |
| --- | --- | --- | --- |
| `mfcc_logreg` | MFCC | Logistic regression | The floor. Everything else must beat it to justify existing. |
| `mfcc_xgboost` | MFCC | Gradient-boosted trees | Tests whether nonlinearity helps at all on pooled coefficients. |
| `logmel_cnn` | Log-mel | 1-D CNN | Tests whether keeping the time axis is worth the cost. |

The two tabular recipes share a byte-identical `FeatureSpec`. That is
deliberate: it means any difference between their numbers is attributable to the
model family and not to two different front ends. Changing that to make one look
better would destroy the only useful thing this comparison provides.

A separate argument for starting here: pooled MFCCs are expected to perform
poorly against modern general-purpose TTS. They are being run anyway, because a
cheap honest floor is worth more than an expensive number with an unexamined
preprocessing pipeline behind it.

## Method

### Split discipline

Three splits, each disjoint by construction at the manifest layer and enforced
again at training time:

- **train** fits parameters.
- **dev** chooses the operating point. Nothing else touches it.
- **test** is scored exactly once, at the threshold dev selected.

`run_baseline` resolves and materialises train and dev, then scores test once.
The threshold is computed on dev and applied to test unmodified; there is no
code path that re-tunes on test, and `tests/unit/test_training_artifacts_runner.py`
asserts that the reported test threshold equals the dev-chosen one.

The leakage gate runs over train, dev, and test identities together before any
model is fitted. It checks speaker, parent, session, generator, file hash, and
content hash. A value appearing on both sides of a boundary stops the run.
`UNKNOWN` on a provenance axis cannot prove disjointness and is reported as such
rather than silently accepted.

> A baseline trained through a leaked speaker axis still produces a confident EER.
> That EER gets quoted. This is the most common way a number in a paper turns out
> to be meaningless, and it is why the gate refuses rather than warns.

### Front end

Canonical log-mel, matching `src/voxshield/audio/features.py`:

- `n_fft=400`, `hop_length=160`, `n_mels=80`, HTK mel scale, 20–7600 Hz.
- No `librosa` and no `torchaudio`, so the front end cannot drift with a
  dependency upgrade.
- Pre-emphasis and per-utterance cepstral mean normalisation run before
  coefficients are taken.
- CNN inputs are padded and cropped to a fixed frame count with `target_frames`,
  and carry the feature geometry in the artefact so the front end can be
  reconstructed exactly.

### Models

Tabular models use an explicit train-fitted `StandardScaler` and attach the dev
split through per-family mechanism, because `sklearn.pipeline.Pipeline.fit`
cannot route XGBoost's `eval_set`:

- `LogisticBaseline` scales, fits, and exposes the scaler for serialisation.
- `XGBoostBaseline` receives `early_stopping_rounds` only when a dev `eval_set`
  exists.
- `LogMelCNN` selects on dev loss each `eval_every` epochs and keeps the best
  weights.

Torch is optional. Importing `voxshield.training` does not import it, so a
tabular-only installation trains without it.

### Sample floors

A run refuses to proceed below the configured minimum per split, and below the
minimum per class, because a small split yields a constant model that then
reports confident numbers. The refusal message says which floor was hit and
that raising it is a decision to be recorded, not a default to be nudged.

## Results

**NOT RUN.** Nothing to report.

Once a corpus exists, record per family: EER, ROC AUC, average precision, the
operating point with its FAR and FRR, calibration bins, latency distribution and
RTF, and subgroup deltas with their sample counts. Use
`docs/evaluation-protocol.md` as the contract for that report and
`src/voxshield/evaluation/report.py` to emit it.

Two constraints on how those numbers may be presented:

1. Subgroup slices need at least 20 samples per class or they are suppressed as
   `NOT AVAILABLE`. A four-sample slice is noise presented as a finding.
2. Timing needs at least five calls. A single call measures process warm-up.

## Calibration

A run fits a probability calibrator on the dev split and then chooses its
operating point on the *calibrated* scores, in that order. A threshold is a
statement about a probability, so it has to be selected after the probabilities
mean what they will mean at test time. Test is scored exactly once, from raw
scores, calibrated once.

`train.calibration` selects the method, and the default is `platt`:

| Method | Behaviour |
| --- | --- |
| `platt` | Logistic recalibration; the default, and the sane choice at dev sizes that will actually occur. |
| `isotonic` | Stepwise recalibration. Needs at least 100 dev samples, because below that it is memorisation wearing a curve. |
| `none` | Raw model probabilities. |

The calibrator, the method and the selected threshold travel in
`metadata.json`, so a rescore from the artefact reproduces the run's
probabilities and its operating point rather than a fresh guess. Registry loads
apply them too: `ModelRegistry.load()` returns a bundle whose scores are already
calibrated, and `run_from_artifact(..., threshold=...)` scores at another point
without editing the artefact.

A calibrator that cannot be supported by the dev split falls back to `none` and
says so in the report notes *and* in the artefact's own notes. It does not fail
the run -- EER does not depend on calibration -- but it is recorded rather than
silent, because an artefact showing `calibration_method: "none"` should not be
indistinguishable from one where the recipe asked for calibration and the split
was too small to honour it.

See the calibration section of `docs/evaluation-protocol.md` for why this
matters more than accuracy.

## Artefacts

A saved artefact is a directory holding `metadata.json` and the model itself:

| Family | Weight file | Also stored |
| --- | --- | --- |
| `mfcc_logreg` | `model.joblib` | scaler and estimator in one joblib payload |
| `mfcc_xgboost` | `model.joblib` | scaler and estimator in one joblib payload |
| `logmel_cnn` | `weights.pt` | frame geometry and architecture in `state.json` |

Metadata records the config hash, the full `feature_spec`, class counts, decode
failures with per-sample reasons, selection facts, library versions, and
`test_evaluated`. That last flag is `false` until test is actually scored, so an
artefact cannot claim a test result it does not have. It also records the
calibrator, its method, the dev-selected threshold and the policy that chose it,
so the operating point is a property of the model rather than of whoever scored
it next.

Artefacts refuse to load if the format version is unknown, if metadata carries
unrecognised fields, or if the recorded family disagrees with the stored weights.
Round-trip fidelity is asserted for all three families; the CNN reproduces scores
to within float32 tolerance, the tabular families exactly.

## Limitations

These are properties of the phase, stated so they are not discovered later:

- Pooled MFCCs discard the time axis and are expected to underperform on
  general-purpose TTS.
- No augmentation, no speaker verification, no embeddings, no fine-tuning of any
  pretrained front end.
- A trained model applies a fixed threshold from a dev split of one corpus. That
  does not transfer across corpora, languages, or recording conditions, and no
  claim is made that it does.
- Calibration is fitted on one dev split and frozen into the artefact. A
  calibrator does not transfer between corpora either, and a model rescored on
  new material is being scored by a curve fitted elsewhere. Re-fit on the target
  corpus before trusting its probabilities.
- The `MEDIUM_THRESHOLD` and `HIGH_THRESHOLD` constants in
  `src/voxshield/policy/evaluator.py` remain unvalidated placeholders and are
  unrelated to any threshold these baselines select.

## Reproducing a run

Validate a recipe without touching audio:

```bash
voxshield ml config --config configs/ml/mfcc_logreg.yaml
```

Train, selecting on dev and scoring test once:

```bash
voxshield ml train \
  --config configs/ml/mfcc_xgboost.yaml \
  --manifest data/manifests/all.jsonl \
  --root data/raw \
  --time
```

Confirm the split resolves and the front end runs, having fitted nothing:

```bash
voxshield ml train --config configs/ml/mfcc_logreg.yaml --dry-run
```

Rescore a split with a saved artefact, without refitting:

```bash
voxshield ml evaluate --artifact data/models/mfcc_logreg \
  --manifest data/manifests/all.jsonl --split test
```

List what has been trained, with the calibration and threshold each artefact
carries:

```bash
voxshield ml models --root data/models
voxshield ml models --root data/models --resolve mfcc_logreg
voxshield ml models --root data/models --compact
```

`ml models` reads metadata only and never loads a model, so it stays cheap on a
full model zoo. A directory that is not an artefact is skipped with a reason
rather than failing the scan. Exit codes: `0` a number was produced, `1` the run
failed, `2` nothing ran and
nothing could have. The third is this repository's current state, and a CI job
should treat it as distinct from `1` so an absent corpus is never recorded as a
code defect.

## Model registry

`ModelRegistry` enumerates the artefacts under a root and resolves a `model_id`
to something scorable. It scans the immediate children of the directory it is
given, so point it at the recipe's `output_dir`, where each artefact lives as
`<model_id>/metadata.json` -- not at the corpus root above it, which would sweep
in `raw`, `manifests` and `processed` as stray directories and find nothing.

Resolution is strict. An unknown `model_id` raises, and so does a score request
whose front end disagrees with the one the model was trained on, because a
probability computed from the wrong features is not a degraded answer but a
wrong one. An artefact that cannot be read is skipped with its reason instead,
so one corrupt directory does not hide the models that are fine.

Loading returns a `ScoringBundle` carrying the model, its calibrator, its
threshold and its metadata together. The calibrator is applied inside the
bundle, which is what stops the registry handing back probabilities calibrated
against nothing.

## Scope

Out of scope for this phase: serving integration, streaming or real-time
inference, any UI or risk-band display, and deployment. An artefact is
loadable; nothing here ships it to a detector.
