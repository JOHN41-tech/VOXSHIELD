import json

import numpy as np
import soundfile as sf

from voxshield.prosody.analysis_service import analyze_prosody
from voxshield.prosody.baseline_service import LocalBaselineStore, enroll_baseline
from voxshield.prosody.config import ProsodyConfig


def _tone(frequency: float) -> np.ndarray:
    t = np.arange(20_000, dtype=np.float32) / 16_000
    return (0.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def test_wav_fixture_enrollment_and_comparison(tmp_path) -> None:
    paths = []
    for index, frequency in enumerate((125, 130, 135)):
        path = tmp_path / f"sample-{index}.wav"
        sf.write(path, _tone(frequency), 16_000)
        paths.append(path)
    config = ProsodyConfig(minimum_baseline_samples=3)
    features = []
    for path in paths:
        audio, rate = sf.read(path, dtype="float32")
        features.append(analyze_prosody(audio, rate, config=config).features)
    baseline = enroll_baseline([item for item in features if item is not None], config)
    store = LocalBaselineStore(tmp_path / "baselines")
    store.create(baseline)
    current, rate = sf.read(paths[0], dtype="float32")
    result = analyze_prosody(
        current,
        rate,
        baseline_id=baseline.baseline_id,
        config=config,
        baseline_store=store,
    )
    json.dumps(result.as_dict(), allow_nan=False)
    assert result.features is not None
    assert result.baseline_id == baseline.baseline_id


def test_no_raw_audio_in_response() -> None:
    result = analyze_prosody(_tone(130), 16_000)
    serialized = json.dumps(result.as_dict(), allow_nan=False)
    assert "samples" not in serialized
    assert "raw_audio" not in serialized
