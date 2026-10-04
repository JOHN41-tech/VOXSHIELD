"""End-to-end smoke: generate a small corpus and run the whole build.

Not a test -- a way to see the pipeline run. Writes real WAV files into a
temporary data root, then calls ``build_dataset`` exactly as the CLI will.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from voxshield.data.build import build_dataset, summarise
from voxshield.data.config import load_data_config
from voxshield.data.torch_dataset import build_dataloader, load_split, summarise_batches

ROOT = Path(os.environ.get("TEMP", ".")) / "voxshield_e2e"
SPEAKERS = tuple(f"spk{index:03d}" for index in range(1, 11))
RATE = 16000


def harmonic(seed: int, seconds: float) -> np.ndarray:
    """A voiced-speech-shaped signal: harmonics under a formant envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(RATE * seconds), dtype=np.float64) / RATE
    f0 = 110.0 + 20.0 * rng.random()
    signal = np.zeros_like(t)
    for harmonic in range(1, 26):
        signal += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t + rng.random())
    # Syllabic amplitude modulation, so the VAD has something to find.
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    signal *= envelope
    signal += 0.01 * rng.standard_normal(t.size)
    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak * 0.6).astype(np.float32)


def _clean(path: Path) -> None:
    """Remove a tree, retrying briefly: Windows can still hold a handle."""
    if not path.exists():
        return
    for attempt in range(5):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.4 * (attempt + 1))


def generate() -> int:
    """Write the corpus. Returns the file count."""
    target = ROOT / "raw" / "voxshield_local"
    _clean(target)
    target.mkdir(parents=True)
    count = 0
    for speaker_index, speaker in enumerate(SPEAKERS):
        for take in range(3):
            audio = harmonic(seed=speaker_index * 100 + take, seconds=6.0)
            path = target / f"{speaker}_{take:02d}.wav"
            sf.write(path, audio, RATE, subtype="PCM_16")
            count += 1
    return count


def main() -> int:
    """Run the build and the loader over it."""
    _clean(ROOT)
    count = generate()
    print(f"wrote {count} wav files under {ROOT}")

    config = load_data_config(
        "configs/data.yaml",
        overrides={"root": str(ROOT)},
    )
    config = config.__class__(
        **{
            **{
                field: getattr(config, field)
                for field in config.__dataclass_fields__
            },
            "root": str(ROOT),
            "gates": type(config.gates)(min_test_sources=1),
        }
    )

    result = build_dataset(config)
    summary = summarise(result)
    print("summary:", summary.to_dict())
    print("build_id:", result.build_id)
    print("manifests:", {k: str(v) for k, v in sorted(result.manifests.items())})
    print("reports:", [str(p) for p in result.reports])
    print("gates passed:", result.gate_passed)
    print("failures:", list(result.gate_report.failures))
    print("cache hits:", result.cache_hits)

    paths = config.data_paths()
    for split in ("train", "dev", "test"):
        dataset = load_split(split, paths=paths, config=config)
        print(f"{split}: {len(dataset)} rows, labels {dataset.label_counts()}")
        loader = build_dataloader(dataset, batch_size=4, seed=7)
        print("   ", summarise_batches(dataset, loader, max_batches=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
