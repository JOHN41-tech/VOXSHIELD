# Evaluation protocol

## Status

**No evaluation has been run.** There is no model, no dataset, and no measured
performance figure for this repository.

This document defines the protocol that must be executed before any VoxShield
risk band is shown to an end user. It exists now, before a model exists, so that
the bar is set in advance rather than negotiated once a number is already on a
slide.

The thresholds currently in `src/voxshield/policy/evaluator.py` —
`MEDIUM_THRESHOLD = 0.50`, `HIGH_THRESHOLD = 0.80` — are **unvalidated
placeholders**. They are not calibrated, and they are not false-positive rates.
Presenting them as either would be the single most damaging thing this project
could do.

## Why calibration is the whole game

A detector produces a number. A bank needs a decision. The gap between them is
calibration, and the only thing that closes it is a measured
false-positive/false-negative trade-off at a chosen operating point.

The asymmetry drives every decision in this protocol:

- A **false positive** accuses a real customer of fraud. It is a direct, visible
  harm to an innocent person, and it is the failure that ends the product's
  credibility in the first deployment.
- A **false negative** lets one cloned call through. It is serious, but it is a
  miss within a system that retains a human in the loop anyway.

Therefore the operating point is chosen by the cost ratio, not by whichever
number looks better. `docs/privacy-design.md` explains why the response surface
is advisory; this document explains why the advice itself must be measured.

## Definitions

| Term | Meaning |
| --- | --- |
| **Genuine** | Human speech, either live or human-recorded, including genuine calls under adverse conditions |
| **Synthetic** | Neural-vocoder or TTS output, including re-encoded and post-processed variants |
| **Window decision** | Per-`Segment` binary or score output |
| **Call decision** | Aggregation of window decisions into one assessment for the uploaded call |
| **FRR** | False Reject Rate — genuine speech judged synthetic |
| **FAR** | False Accept Rate — synthetic speech judged genuine |
| **Attack** | Any synthetic sample. All attacks count as positives |

`FAR` is the number that matters most and is the hardest to measure honestly,
because synthetic-speech corpora are overwhelmingly public TTS samples while real
attacks arrive over telephony codecs at unknown levels with unknown devices.
Evaluating only on a public corpus produces a number that will not survive
contact with a real call.

## Dataset requirements

Composition is specified in `docs/dataset-manifest.md`. The evaluation split must
satisfy four constraints that corpora routinely fail:

1. **Speaker-disjoint.** No speaker appears in both train and test. Without this,
   a model can memorise a voice and the split measures memorisation.
2. **Source-disjoint.** Different TTS vendors, different corpora, different
   codecs between train and test. A model that learns "this vendor's artefacts"
   will look excellent in-house and fail against a vendor it never saw.
3. **Channel-disjoint.** Telephony band, wideband, and mobile codecs must all
   appear, and must not be concentrated in one split.
4. **Device-disjoint.** Microphone and handset populations should differ between
   train and test.

A split that violates any of these must be reported as such, and its result must
not be used to set an operating point.

## Metrics

Reported per tier, and reported for `FRR`/`FAR` at **each** candidate operating
point rather than only at the chosen one:

- `FRR` and `FAR` at the call level, with 95% confidence intervals.
- Precision and recall, because they depend on prevalence and are only
  interpretable alongside the deployment's expected synthetic-speech rate.
- ROC-AUC, for model comparison only. **AUC is not a deployment number** — it
  hides the operating point entirely, and citing it alone is a way of
  overselling a model.
- Detection latency per call, against the in-call verdict budget.
- Performance per channel, per device class, and per acoustic condition, so that
  a good aggregate cannot hide a subgroup at `FAR` 0.30.

That last point is the one most often skipped. An aggregate `FAR` of 0.02 is
consistent with `FAR` 0.15 for low-bitrate mobile calls, which is the segment a
telecom deployment cares about most.

## Required test conditions

Aggregate metrics over clean data are not sufficient. The evaluation must report
each condition separately:

- Clean, quiet, studio-quality speech.
- Telephony narrowband (8 kHz) and wideband (16 kHz).
- Mobile and VoIP codecs at multiple bitrates.
- Low SNR; background speech; music; keyboard; line noise.
- Reverberant and near-field recordings.
- Level extremes: very quiet and very loud, testing the `−23 dBFS`
  normalisation path and the VAD's gain invariance.
- Non-speech and near-silence, to confirm the abstention path is used rather
  than producing a band.

## Adversarial robustness

A detector that a fraudster can tune around is worse than no detector, because it
manufactures confidence in exactly the calls that are fraudulent. The
evaluation must include, at minimum:

- Post-processing the attack: re-encoding, resampling, mild filtering, gain
  change, background noise mixing, and splicing.
- Adversarial perturbation search over the attack, reporting the perturbation
  budget at which the detector fails.
- **Over-the-air** testing. Every attack passes through a real codec path.
  A perturbation that survives a 20 ms frame but is removed by G.711 is not a
  real attack, and one that survives G.711 is more than an academic result.

The finding that matters: report the evasion rate *at the deployed operating
point*. A detector evadable by simple resampling is not a detector.

## Latency and degradation

Report p50, p95, and p99 end-to-end latency on the target hardware.

A stated degradation strategy is required, not optional. When the latency budget
is missed, the system must have a defined behaviour. The Phase 0 default is
already the right one: abstain with `UNSCORED`. Whatever replaces the `max`
aggregation — the known accuracy cost of 4 s windows overlapping by 50% — is a
latency/accuracy trade-off to be measured, not assumed.

### The RTF instrument now exists, and is not yet a result

`ProcessResult` reports `wall_seconds` and `rtf` (wall time over audio duration)
for the offline path, and `n_segments` / `n_padded_windows` alongside them. That
makes the preprocessing cost of a clip measurable without a model present.

Two cautions, both of which matter more than the number:

- A single clip's RTF is not a latency distribution. Collect p50/p95/p99 over a
  corpus on the target hardware, and report the clip duration alongside, because
  RTF is strongly length-dependent — a 2 s clip pays fixed per-window cost over
  very little audio.
- `rtf` covers intake through feature extraction **only**. It excludes model
  inference and the policy engine, so it is a floor on end-to-end cost, never a
  substitute for it. Report it as the preprocessing share, labelled as such.

Feature extraction is currently the dominant preprocessing cost, and
`--no-features` exists in the CLI to measure the earlier stages in isolation.
Numbers from that flag bound the decode/VAD/segmentation share.

## Calibration procedure

1. Train on the train split only. Freeze the model.
2. Sweep the decision threshold across its full range on a held-out
   **calibration** split, disjoint from test.
3. Choose the operating point by the cost ratio, with the `FRR` ceiling set by
   the integrating organisation, not by the model team.
4. Re-derive the risk bands in `policy/actions.py` from that operating point.
   The bands are outputs of calibration, not inputs to it.
5. Evaluate **once** on the test split, at the frozen operating point.
6. Bump `POLICY_VERSION` and record the measured numbers in the ADR.
7. Re-run on any change to features, the model, or the thresholds. A feature
   change invalidates calibration even when accuracy looks unchanged.

## Reporting standard

- Report the number **and** the dataset it came from, with sample counts.
  "FAR 0.01" without `n` is not a result.
- Report the confidence interval. At these corpus sizes the interval is wide
  enough to change the conclusion.
- Report the conditions under which the system abstains. An abstention rate is a
  property of the deployment, and a high one is a capacity finding, not a
  rounding error.
- Never report a metric on a split that violates the disjointness constraints
  without labelling it as such.
- Never cite ROC-AUC as a user-facing capability claim.

## Exit criteria for Phase 1

A model may be connected to the response surface only when all of the following
hold. These are the gate, and they are deliberately strict:

- [ ] Disjoint splits, with the disjointness verified and documented.
- [ ] `FRR` and `FAR` measured at the chosen operating point, with CIs.
- [ ] A `FRR` ceiling agreed in writing with a named owner at the integrating
      organisation.
- [ ] Per-channel and per-device breakdowns published, with no subgroup above the
      agreed `FAR` ceiling.
- [ ] Adversarial and over-the-air evaluation completed, with the evasion rate
      reported.
- [ ] Latency measured on target hardware against a stated budget.
- [ ] Calibration re-derived from the frozen model, and the risk bands in
      `policy/actions.py` replaced with measured values.
- [ ] `docs/dataset-manifest.md` populated with the corpus actually used,
      including its known gaps.
- [ ] A shadow or advisory deployment completed with an analyst reviewing every
      flagged call, before any band reaches a customer-facing surface.

## Implementation

The protocol above is coded, and deliberately so: it is easier to violate a
document than a guard.

- `src/voxshield/training/runner.py` enforces split discipline. Dev selects the
  operating point, test is scored once at that threshold, and there is no code
  path that re-tunes on test.
- `src/voxshield/data/gates.py` refuses leakage across speaker, parent, session,
  generator, file hash, and content hash.
- `src/voxshield/evaluation/report.py` serialises unmeasured work as `NOT RUN`
  and undefined figures as `NOT AVAILABLE`, so a missing number is visibly
  missing rather than absent.
- Sample floors and per-class minimums refuse small splits, because a constant
  model fitted on four samples still reports confident metrics.

The three baselines to be measured under this protocol, and their method, are in
[`docs/phase3-baseline-models.md`](phase3-baseline-models.md). Their current
state is **NOT RUN**: no corpus exists.

Until every box is checked, the correct response from VoxShield remains
`UNSCORED` — which is exactly what Phase 0 does.
