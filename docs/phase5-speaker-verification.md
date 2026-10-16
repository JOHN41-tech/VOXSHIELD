# Phase 5 — speaker verification and voice identity matching

## Task plan and delivery status

- [x] Verify the prior-phase code and repository state before coding. The Phase 3 training runner, artifact handling, and registry are present. Phase 4 has a log-mel CRNN implementation and measured report. The README/architecture overview is stale and describes the earlier foundation; no Phase 5 code or speaker-labeled corpus was present at start.
- [x] Reuse the validated shared audio loader, canonical preprocessing, signal-quality report, and VAD.
- [x] Add replaceable speaker-embedding interface and local-only pretrained ECAPA-TDNN adapter.
- [x] Add consent/authorization-gated, multi-sample enrollment with quality rejection, duplicate suppression, and normalized-mean profile construction.
- [x] Add profile schema, atomic local storage, version compatibility checks, deactivation and targeted deletion.
- [x] Add cosine scoring and typed MATCH / NON_MATCH / INCONCLUSIVE verification output, with quality status and metadata-only auditing.
- [x] Add deterministic trial construction, speaker-disjoint validation, FAR/FRR/EER/ROC-AUC/confusion metrics, dev-threshold selection, robustness transforms, and latency summary utilities.
- [x] Add CLI and unit/integration/privacy tests.
- [x] Add explicit model/config setup notes and record unavailable measurements honestly.

## Model choice and setup

The optional encoder is SpeechBrain's ECAPA-TDNN VoxCeleb model (`speechbrain/spkrec-ecapa-voxceleb`). The upstream model card identifies Apache-2.0, 16 kHz single-channel audio, a 192-dimensional embedding, VoxCeleb 1/2 training data, and an EER of 0.80% on the cleaned VoxCeleb1 test set. That benchmark result is not a VoxShield result and does not establish performance on telephone, multilingual, synthetic, or this project's data. See the model card and SpeechBrain inference tutorial in References.

VoxShield does not fetch weights. Install the optional dependencies with `pip install -e '.[speaker]'`, then provision/download the upstream model into a local directory through an approved process. Pass that local path as `--model-dir`. The adapter accepts no model-hub name and requires a local YAML hyperparameter file, so ordinary enrollment/verification cannot silently trigger a large download. The model weights may be several hundred megabytes depending on upstream packaging; measure and record the actual file size when provisioning. CPU is supported; GPU is optional when the installed PyTorch build and hardware support it.

ECAPA-TDNN's published score and general model are not a guarantee of use quality. The deployed threshold must be fit/evaluated on appropriately licensed, consented, speaker-labeled development trials for the target setting. The checked-in `0.75` value is a provisional plumbing default only, labeled `NOT_CALIBRATED`; it is not a production policy. VoxShield returns `verification_score: null` until calibration is implemented and validated on a disjoint development set.

The model card does not establish a universal minimum speech duration. The checked-in 2-second voiced-speech gate is a configurable starting quality rule, not a validated optimum. Adjust `--minimum-voiced-seconds`, `--maximum-silence-ratio`, and `--maximum-clipping-ratio` for the selected model/domain and validate those settings experimentally.

The evaluation CLI supports a development-only EER-balanced threshold, a target-FAR threshold (prioritize fewer false accepts), or a target-FRR threshold (preserve genuine-speaker acceptance). It reports resulting trade-offs at the frozen test threshold. Dataset, test-session, and test-duration slices plus genuine/impostor score quantiles are included when supported. `--robustness` adds deterministic clean, −6 dB gain, and approximately 26 dB SNR additive-noise test-side conditions; it does not claim codec, language, microphone, speaking-style, or clone robustness.

## API and privacy behavior

Use `EnrollmentService.enroll(...)` only for an explicitly authorized request with `consent_confirmed=True` and a non-empty consent reference. It accepts one or more recordings, rejects unacceptable samples individually, drops exact/near-identical duplicates, and refuses to create an empty profile. The local profile store persists only the normalized reference embedding and metadata, never raw audio. JSON profile files are atomically replaced, with restrictive owner-only file/directory modes where supported. Profile IDs are UUID-shaped pseudonyms.

`VerificationService.verify(...)` resolves only the requested profile. Missing, inactive, expired (when `expires_at` is set), corrupt, or encoder-incompatible profiles raise specific errors; there is no fallback profile. An unacceptable sample or failed embedding returns `INCONCLUSIVE`. A score is cosine similarity in `[-1, 1]`, not a probability. An in-memory audit sink records the decision and permitted metadata only; it has no field for recordings or vectors. A persistent production audit/profile backend and its retention/deletion guarantees remain deployment work.

Profile deactivation increments the version. `LocalProfileStore.delete(profile_id)` removes only the named profile file; backups/caches are not introduced. Deletion from an external backup is outside this development store and must be handled by any future storage adapter. Embeddings remain sensitive biometric data. The implementation does not establish legal compliance or real-world identity.

A speaker match does not mean the audio is genuine. A cloned voice can match its target. Phase 4 spoof detection remains separate, and no combined impersonation-risk score, automated financial action, live-call interception, or dashboard is added here.

## Evaluation contract and current measurements

`build_trials` constructs same/different-speaker trials from distinct recordings with a reproducible seed and prefers cross-session genuine pairs when session metadata supports it. Evaluation metadata carries split, dataset, speaker, session, recording, duration, and quality information; labels are used to construct and evaluate trials, never passed into the embedding encoder. `validate_disjoint_splits` rejects dev/test speaker overlap. EER, target-FAR, and target-FRR operating points are selected on development scores only; test metrics are computed at the frozen threshold. Reports include score distributions, FAR, FRR, approximate EER, ROC-AUC, confusion counts, latency summaries, and dataset/session/duration slices when supported. Metrics return `NOT_AVAILABLE` instead of fabricated values when either class is absent.

No speaker-labeled corpus/trial manifest is in this checkout. Therefore Phase 5 has **no measured VoxShield FAR, FRR, EER, ROC-AUC, calibration quality, robustness results, or model latency distribution**. Unit fixtures validate deterministic plumbing only and do not count as speaker-verification evidence. Calibration is intentionally unavailable until a sufficient, appropriately licensed trial set exists. The CLI can run deterministic gain/noise robustness trials and latency summaries when a manifest and local model are supplied; neither is claimed as measured here.

## Commands

The CLI command group is `voxshield speaker`:

```bash
voxshield speaker enroll --input sample-a.wav --input sample-b.wav --profile-id <uuid> \
  --consent-reference <opaque-reference> --authorized --consent-confirmed \
  --model-dir /path/to/local/spkrec-ecapa-voxceleb
voxshield speaker verify --input new-sample.wav --profile-id <uuid> \
  --model-dir /path/to/local/spkrec-ecapa-voxceleb
voxshield speaker profile --profile-id <uuid> --store data/speaker-profiles
voxshield speaker profile --profile-id <uuid> --delete --store data/speaker-profiles
voxshield speaker evaluate --manifest data/speaker-recordings.jsonl --model-dir /path/to/local/model
voxshield speaker evaluate --manifest data/speaker-recordings.jsonl --model-dir /path/to/local/model \
  --trials-output reports/speaker-trials.jsonl --threshold-policy target_far --target-far 0.01 --robustness
```

The profile deletion command names exactly one UUID profile and does not recursively delete the store. Evaluation input is JSONL with `recording_id`, `path`, `speaker_id`, `dataset_id`, and `split` (`dev` or `test`); optionally include `session_id`, `duration_seconds`, and `quality_status`. Relative audio paths resolve against the manifest directory. Store speaker IDs as pseudonyms. Do not place legal names, phone numbers, or account IDs in manifests. `--trials-output` writes the seed and explicit trial metadata (pseudonymous labels and local paths); keep it in an appropriately restricted directory.

## References

[1]: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb "SpeechBrain ECAPA-TDNN VoxCeleb model card"
[2]: https://speechbrain.readthedocs.io/en/latest/tutorials/advanced/pre-trained-models-and-fine-tuning-with-huggingface.html "SpeechBrain pretrained-model inference tutorial"
[3]: https://github.com/speechbrain/speechbrain/blob/develop/recipes/VoxCeleb/SpeakerRec/README.md "SpeechBrain VoxCeleb speaker-recognition recipe"
