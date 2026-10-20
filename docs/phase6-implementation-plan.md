# Phase 6 implementation plan

## Task breakdown and commit policy

Phase 6 is implemented as small, reviewable commits. Each task is completed with focused tests and pushed before the next task starts.

1. **Integration contract and plan** — document the existing audio contract, module boundaries, non-goals, and task order.
2. **Feature schema and configuration** — add versioned typed result models and validated prosody configuration.
3. **Feature extraction** — implement deterministic pitch, energy, timing, rhythm, and quality-aware extraction over validated mono waveforms.
4. **Quality handling** — define explicit PASS/WARN/INCONCLUSIVE behavior for empty, silent, short, clipped, non-finite, and insufficiently voiced inputs.
5. **Behavioral baseline management** — add explicit enrollment, compatibility metadata, local privacy-conscious storage, update, deactivation, and deletion.
6. **Deviation analysis** — compare compatible observations using transparent robust statistics and produce explainable per-feature deviations.
7. **Analysis service** — provide the stable `analyze_prosody` interface, baseline-free mode, latency, and failure/inconclusive separation.
8. **Evaluation pipeline** — add deterministic fixture/manifests and metrics that report `NOT AVAILABLE` when external labeled data is absent.
9. **CLI workflow** — add analysis and evaluation commands with structured JSON output and no implicit enrollment.
10. **Unit/integration/regression tests** — cover mathematical behavior, invalid inputs, lifecycle, compatibility, serialization, and Phase 0–5 compatibility.
11. **Benchmarks** — measure validation, extraction, comparison, and end-to-end latency on executable fixtures.
12. **Documentation and final verification** — document algorithms, schemas, privacy, limitations, commands, benchmark results, and run the complete regression gates.

## Existing repository contract

- `voxshield.audio.pipeline.prepare` is the canonical encoded-audio path. It decodes, converts to mono 16 kHz, normalizes, runs the deterministic VAD, and returns metadata-safe speech regions.
- `voxshield.audio.vad.detect_speech` is reused for array inputs. Prosody does not introduce a second VAD implementation.
- `voxshield.audio.quality.assess_quality` remains the source of clipping, silence, level, and non-finite quality signals.
- Phase 5 speaker embeddings and speaker profiles remain separate from prosodic behavioral baselines. A speaker profile ID is only an optional correlation key; it does not supply a prosody baseline.
- Phase 4 synthetic-voice output and future Phase 8 risk fusion are not imported or combined by this service.
- Raw audio is processed in memory and is not persisted by the Phase 6 baseline store. Stored records contain pseudonymous IDs, aggregate feature statistics, schema/configuration fingerprints, lifecycle metadata, and quality notes.

## Engineering posture

The first implementation is descriptive and baseline-relative. It does not claim that prosodic deviation proves cloning, deception, emotion, intent, or maliciousness. Baseline-free analysis reports measurements and data quality only. External dataset evaluation is only reported when the repository actually contains suitable data and labels.
