# Phase 6 — Prosody and Behavioral Analysis

**Status: IMPLEMENTED AND TESTED for the Phase 6 service and fixtures.** This layer reports measurable vocal behavior and optional deviation from an explicitly enrolled behavioral baseline. It does not produce a final impersonation-risk score, synthetic-speech verdict, policy decision, or claim about honesty, emotion, intent, or maliciousness.

## Objectives and boundaries

Phase 6 measures fundamental frequency, energy, timing, pauses, speech continuity, temporal variation, signal quality, and optional speaker-specific behavioral deviation. It is deliberately separate from the Phase 4 synthetic-voice detector, the Phase 5 speaker-embedding profile, and future Phase 8 risk fusion.

The existing canonical contract is reused: mono floating-point audio at 16 kHz, with the repository's deterministic energy/ZCR/spectral-flatness VAD and quality reporting. The Phase 6 extractor does not add a second VAD implementation.

## Architecture

| Module | Responsibility |
| --- | --- |
| `prosody/config.py` | Validated extraction, quality, baseline, and deviation defaults; extraction fingerprint. |
| `prosody/schema.py` | Versioned `FeatureValue`, `ProsodyFeatures`, and status types; finite JSON-safe values. |
| `prosody/feature_extractor.py` | Pitch, energy, timing, pause, speech-segment, and rhythm features. |
| `prosody/quality_checks.py` | PASS/WARN/INCONCLUSIVE classification and quality warnings. |
| `prosody/baseline_service.py` | Explicit enrollment, robust aggregate distributions, local storage, update, deactivate, delete. |
| `prosody/anomaly_detector.py` | Per-feature robust deviations and explainable summaries. |
| `prosody/analysis_service.py` | Stable orchestration interface, latency, response construction, and error separation. |
| `evaluation/prosody/evaluate_features.py` | Deterministic manifests and extraction-quality evaluation. |
| `scripts/benchmark_prosody.py` | Offline stage benchmark with mean, median, and p95. |

## Feature definitions and units

Features are versioned as schema `1.0`. Each value contains `value`, `unit`, `valid`, and either a reason or quality notes. Unavailable measurements use `value: null` and never serialize NaN or infinity.

- Pitch features are estimated with windowed normalized autocorrelation over 50–500 Hz. Median, mean, standard deviation, range, and median frame-to-frame variation are in Hz. Unvoiced and weakly periodic frames are excluded. Absolute pitch is not treated as identity evidence.
- `voiced_frame_ratio` and `valid_pitch_frame_ratio` are ratios. The latter reports valid pitch frames among detected speech frames.
- Energy features use linear waveform RMS, not calibrated dB SPL. Median, standard deviation, P10, P90, and frame-change rate are reported only over detected speech frames.
- Timing features include total duration, detected speech duration, estimated valid-pitch duration, silence duration, speech-activity ratio, pause count, median and standard deviation of pauses, segment count, segment-duration variation, and inter-segment energy variation. Pause boundaries inherit the VAD frame/hop resolution; this is not word or syllable alignment.
- Jitter, shimmer, HNR, syllable rate, and word rate are not reported because this implementation does not have sufficiently validated alignment or voice-quality prerequisites for them.

## Quality and failure behavior

`PASS` means the sample has sufficient usable speech and no elevated quality warning. `WARN` means features are available but clipping or other recoverable quality caveats are present. `INCONCLUSIVE` is returned for silence-only input, insufficient voiced speech, too few valid features, non-finite values, or other conditions where an anomaly claim would be unsafe. Invalid mono/sample-rate contracts are operational failures, not anomalies.

The service distinguishes:

- `COMPLETED`: feature extraction completed; baseline comparison is either not requested, available, or unavailable without invalidating the measured features.
- `INCONCLUSIVE`: the audio or requested comparison has insufficient evidence.
- `FAILED`: the input contract or an operational extraction failure prevented reliable analysis.

Clipping, low level, silence, non-finite data, and VAD limitations are surfaced as warnings. Poor quality never fabricates an anomaly.

## Stable service interface

```python
from voxshield.prosody.analysis_service import analyze_prosody

result = analyze_prosody(
    audio,                         # in-memory mono numpy waveform
    sample_rate=16_000,
    speaker_profile_id=None,       # correlation only; does not supply a baseline
    baseline_id=None,              # explicit, previously enrolled baseline only
    config=config,
    baseline_store=store,
)
```

`result.as_dict()` returns an `analysis_id`, status, quality status, schema version, structured features, baseline comparison status, per-feature deviations, warnings, latency, and an operational error field. Raw audio is not present in the response.

## Baseline enrollment and comparison

Enrollment is explicit and requires multiple suitable `ProsodyFeatures` samples. The default minimum is three samples. Unsuitable samples are counted as rejected; features are retained only when they have enough observations. The stored baseline contains pseudonymous UUID identifiers, feature schema/configuration identity, sample counts, mean, standard deviation, median, MAD, lifecycle metadata, and limitations. It does not contain raw audio.

Comparison requires both schema and extraction fingerprint compatibility. Each feature receives an observed value, reference median/mean, robust deviation, outside-reference flag, confidence label, and plain-language explanation. The calculation uses the larger of standard deviation and `1.4826 * MAD`, with a configured robust threshold. Zero-variance data is handled by a finite floor; low-count references produce `INCONCLUSIVE` rather than an alert.

A deviation means measurable difference from a reference distribution. It does not mean that the speaker is lying, that the recording is synthetic, or that malicious behavior occurred.

## Privacy and lifecycle

Baseline records are stored under a directory with mode `0700`; files are written atomically with mode `0600`. The local store supports retrieval, versioned update, deactivation, and deletion of the explicitly requested baseline. Raw audio is processed in memory and is never persisted by Phase 6. Authorization controls are not yet part of the repository, so callers must place enrollment and lifecycle operations behind their own consent/authorization workflow.

The optional `speaker_profile_id` is not used as an embedding or automatic enrollment trigger. Phase 5 speaker profiles and Phase 6 behavioral baselines remain separate artifacts.

## CLI

```bash
python -m voxshield.prosody --help
python scripts/analyze_prosody.py analyze --audio path/to/sample.wav
python scripts/analyze_prosody.py analyze --audio path/to/sample.wav --baseline-id <uuid>
python scripts/analyze_prosody.py enroll --audio sample1.wav sample2.wav sample3.wav
python scripts/analyze_prosody.py baseline get --baseline-id <uuid>
python scripts/analyze_prosody.py baseline deactivate --baseline-id <uuid>
python scripts/analyze_prosody.py baseline delete --baseline-id <uuid>
python scripts/analyze_prosody.py evaluate --dataset path/to/wav-directory
python scripts/benchmark_prosody.py --runs 10
```

The comparison command loads an existing baseline and never silently enrolls one. The evaluation command returns `NOT AVAILABLE` when no dataset is present.

## Evaluation and split integrity

The implemented evaluator creates a sorted path/SHA-256 manifest and records seed, sample count, success count, quality rejections, failures, and mean latency. It does not calculate classification metrics without independent labels. No suitable external prosody-labeled dataset was present in this checkout, so population-level, session-disjoint baseline, condition-sensitivity, and real-vs-synthetic experiments are **NOT RUN**. Mathematical fixtures are explicitly not human-speech performance evidence.

Phase 6 tests actually run:

- `28 passed` across schema, extraction, quality, baseline lifecycle, deviation, service, evaluation, and integration/privacy tests.
- Benchmark: `3` runs on Linux x86_64, 16 kHz, 1.25-second mathematical tone fixture.
- Measured validation: mean `0.0393 ms`, median `0.0379 ms`, p95 `0.0423 ms`.
- Measured quality: mean `0.0294 ms`, median `0.0261 ms`, p95 `0.0355 ms`.
- Measured extraction: mean `18.6679 ms`, median `17.9414 ms`, p95 `20.1623 ms`.
- Measured baseline-comparison service path: mean `18.4102 ms`, median `18.2538 ms`, p95 `18.9677 ms`.
- Measured total benchmark loop: mean `18.4473 ms`, median `18.2807 ms`, p95 `19.0137 ms`.

These are offline measurements on a mathematical fixture and do not establish Phase 7 streaming or production latency.

## Regression status and blockers

The Phase 6-specific suite is green. The complete repository suite was attempted and produced `1055 passed, 1 skipped, 89 failed, 102 errors`. The failures/errors are environment or fixture blockers outside Phase 6: `python-multipart` is absent for existing API tests; scikit-learn/XGBoost/joblib/Torch are absent for existing training tests; and the repository's external dataset fixtures are not present. Phase 4/5 source files were not modified by Phase 6.

## Integration with future phases

- **Phase 7:** can call `analyze_prosody` on rolling windows once streaming orchestration supplies validated mono 16 kHz frames. Phase 6 does not claim streaming support.
- **Phase 8:** may consume `ProsodyAnalysisResult` as one independently interpretable signal. It must keep prosody, synthetic-voice probability, and speaker verification separate until a separately specified and evaluated fusion layer exists.
- **Phase 9:** no policy or automated response is emitted by this service.

## Known limitations

Pitch tracking can fail under noise, compression, short utterances, expressive speech, and octave ambiguity. VAD boundary errors affect pause and speech-ratio estimates. Energy values are recording-level waveform measurements, not calibrated SPL. The current baseline statistics are descriptive engineering defaults, not a validated population model or impersonation detector. Cross-language, accent, microphone, codec, and session generalization remain **NOT RUN** pending representative data.
