"""Reproducible feature-extraction evaluation; no fabricated classification metrics."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from voxshield.prosody.config import ProsodyConfig
from voxshield.prosody.feature_extractor import extract_prosodic_features
from voxshield.prosody.quality_checks import assess_prosody_quality


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    status: str
    seed: int
    sample_count: int
    success_count: int
    quality_rejection_count: int
    failure_count: int
    mean_latency_ms: float | None
    manifest: tuple[dict[str, str], ...]
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "success_count": self.success_count,
            "quality_rejection_count": self.quality_rejection_count,
            "failure_count": self.failure_count,
            "mean_latency_ms": self.mean_latency_ms,
            "manifest": list(self.manifest),
            "notes": list(self.notes),
        }


def manifest_for_paths(paths: Sequence[str | Path]) -> tuple[dict[str, str], ...]:
    records: list[dict[str, str]] = []
    for path_value in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        digest = hashlib.sha256(path_value.read_bytes()).hexdigest()
        records.append({"path": str(path_value), "sha256": digest})
    return tuple(records)


def evaluate_arrays(
    samples: list[np.ndarray],
    sample_rate: int = 16_000,
    config: ProsodyConfig | None = None,
    *,
    seed: int = 0,
    manifest: tuple[dict[str, str], ...] = (),
) -> EvaluationReport:
    """Evaluate extraction reliability on an explicit, caller-owned fixture set."""
    cfg = config or ProsodyConfig()
    success = 0
    rejected = 0
    failed = 0
    latencies: list[float] = []
    sample_list = samples
    for sample in sample_list:
        started = time.perf_counter()
        try:
            extraction = extract_prosodic_features(sample, sample_rate, cfg)
            quality = assess_prosody_quality(extraction, cfg)
            if quality.is_comparable:
                success += 1
            else:
                rejected += 1
        except (ValueError, RuntimeError):
            failed += 1
        latencies.append((time.perf_counter() - started) * 1000.0)
    return EvaluationReport(
            status="COMPLETED" if sample_list else "NOT AVAILABLE",
        seed=seed,
            sample_count=len(sample_list),
        success_count=success,
        quality_rejection_count=rejected,
        failure_count=failed,
        mean_latency_ms=float(np.mean(latencies)) if latencies else None,
        manifest=manifest,
        notes=("Synthetic fixtures are mathematical validation, not human-speech evidence.",)
            if sample_list
        else ("No evaluation samples were supplied.",),
    )


def evaluate_dataset_directory(
    directory: str | Path,
    config: ProsodyConfig | None = None,
    *,
    seed: int = 0,
) -> EvaluationReport:
    """Evaluate WAV files in a directory, or honestly report unavailable data."""
    import soundfile as sf

    paths = sorted(Path(directory).glob("*.wav")) if Path(directory).is_dir() else []
    if not paths:
        return evaluate_arrays([], config=config, seed=seed)
    arrays = []
    for path in paths:
        data, rate = sf.read(path, dtype="float32", always_2d=False)
        if np.asarray(data).ndim != 1 or rate != 16_000:
            continue
        arrays.append(np.asarray(data, dtype=np.float32))
    return evaluate_arrays(
        arrays,
        sample_rate=16_000,
        config=config,
        seed=seed,
        manifest=manifest_for_paths(paths),
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Phase 6 prosody feature extraction")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from voxshield.prosody.config import load_prosody_config

    config = load_prosody_config(args.config)
    report = (
        evaluate_dataset_directory(args.dataset, config, seed=args.seed)
        if args.dataset
        else evaluate_arrays([])
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
