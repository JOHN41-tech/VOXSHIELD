# Dataset manifest

## Status

**This manifest describes no data.** VoxShield currently holds no audio, no
features, no samples, and no labels. The manifest is a specification of the
corpus required to satisfy `docs/evaluation-protocol.md`, written before
collection so that the collection cannot be shaped to flatter a model.

No data has been collected. There is no `data/` directory, no download script, and
no cached artefact. Anyone looking for training data should be told plainly that
there is none.

The `data/` directory now exists as an empty scaffold of ignored subdirectories,
and `voxshield ml train` reads its manifest from `data/manifests/all.jsonl`. On a
fresh checkout that path does not exist, so the command exits `2` and prints a
`not_run` record. That is the correct behaviour, not a defect to work around; see
[`docs/phase3-baseline-models.md`](phase3-baseline-models.md).

## Legal and ethical gate

This is a prerequisite to every row below, not a section to fill in later.

Every source must be recorded with its licence, whether it permits model
training, whether it permits derived features, and whether it permits
redistribution. A row with an unverified licence status cannot be admitted.

Corpora of real human speech carry consent obligations that differ by
jurisdiction. Public availability on the internet is **not** consent to train a
voice-cloning detector on someone's voice, and treating it as such is both a legal
risk and a reputational one for a project whose entire premise is not misusing
voices.

**Do not include live call recordings.** Real telephony audio from an integrating
organisation requires an explicit lawful basis, a retention limit, and a
data-processing agreement. This is the hardest data to obtain and the easiest to
collect without proper authority. Treat it as out of reach until governance
approves it in writing; the evaluation is designed to work without it.

## Required composition

| Slice | Purpose | Notes |
| --- | --- | --- |
| **Genuine — conversational** | Real call-like speech, matched in length and channel to attacks | The most important and hardest slice |
| **Genuine — read speech** | Neutral reference | Frequently over-represented; do not let it dominate |
| **Genuine — adverse** | Low SNR, background speech, music, keyboard, line noise, reverb | Where `FAR` is usually won or lost |
| **Synthetic — public TTS** | Vendor and corpus diversity across train/test | Diversity of vendors matters more than total hours |
| **Synthetic — voice cloning** | Few-shot and zero-shot cloning output | The actual threat |
| **Synthetic — post-processed** | Attacks after re-encode, filter, gain, noise, splice | Measures the evasion rate |
| **Non-speech** | Silence, noise, music only, DTMF | Confirms the abstention path |
| **Level extremes** | Very quiet, very loud, near-clipping | Exercises normalisation and VAD gain invariance |

## Split constraints

These are the constraints from `docs/evaluation-protocol.md`, restated because
they are the ones a manifest most often fails to record:

- **Speaker-disjoint** across train / calibration / test.
- **Source-disjoint** across TTS vendor and corpus, not just file.
- **Channel-disjoint** across narrowband, wideband, and mobile codecs.
- **Device-disjoint** across handset and microphone population.
- **Contributor-disjoint** where a source can be traced to individuals, to
  prevent identity-linked leakage between a person's train and test audio.

A split record is required per split, naming the axes it was verified on and who
verified it. "Disjoint" without a recorded check is an assumption.

## Provenance schema

One JSON object per source. No unrecorded source may be admitted.

| Field | Purpose |
| --- | --- |
| `id` | Stable identifier, referenced by every derived artefact |
| `kind` | `genuine` / `synthetic` / `non_speech` |
| `provenance` | Where it came from, and how |
| `licence` | Licence identifier and URL |
| `permits_training` | Boolean, verified not assumed |
| `permits_features` | Boolean, verified not assumed |
| `permits_redistribution` | Boolean |
| `consent_basis` | Consent or lawful basis, with reference |
| `vendor` | TTS vendor, or `n/a` for genuine |
| `cloning_method` | Few-shot / zero-shot / full, or `n/a` |
| `speakers` | Speaker identifiers, for the disjointness check |
| `language` | Language code |
| `channel` | `narrowband` / `wideband` / `mobile` / `studio` / `unknown` |
| `codec_bitrate` | Encoding detail, or `n/a` |
| `device` | Capture device class, or `unknown` |
| `duration_seconds` | Total |
| `n_speakers` | Distinct speaker count |
| `licence_verified_by` | Who verified it, and when |
| `notes` | Known defects, consent limits, anything unusual |

`provenance` and `consent_basis` are the fields that make an audit possible later.
A manifest that records only counts is a shopping list, not provenance.

## Known gaps in the planned corpus

Stated now rather than discovered during evaluation:

- **No real-call genuine slice** until governance approves one. If it cannot be
  obtained, the `FAR` estimate will be optimistic and must be labelled so.
- **Language coverage is undecided.** A detector's behaviour on a language or
  accent absent from training is unknown, and telephony attacks are not
  anglophone. This is an open decision, not an oversight.
- **No adversarial perturbation set** committed. It must be generated, which
  means the generation code is itself part of the deliverable.
- **Codec coverage is aspirational.** Real G.711 and AMR-WB traffic needs
  either a partner or a licensed capture. Proceeding without it leaves a gap
  exactly where telephony deployments live.
- **Class balance will be artificial.** A real deployment sees a low rate of
  synthetic speech. A 50/50 benchmark `FAR` says nothing about a deployment at
  0.1% prevalence, which is why precision and recall are reported alongside
  `FAR`/`FRR`.

## Rules

- No raw audio is committed to this repository, and none may be. Derived
  features and labels are referenceable by `id`; the audio stays in controlled
  storage. This is the same rule `docs/privacy-design.md` applies at runtime.
- Manifests are append-only. A removed source keeps its row, with
  `removed_reason` and `removed_at`, because a dataset that can be quietly
  reweighted after a disappointing result is not reproducible.
- A manifest entry without a verified licence and consent basis is not admitted,
  regardless of how much data it would add.
- Every evaluation run records the manifest revision it used. A result without
  one cannot be reproduced.
