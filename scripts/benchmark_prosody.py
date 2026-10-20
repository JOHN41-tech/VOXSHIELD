#!/usr/bin/env python3
"""Benchmark offline Phase 6 analysis; not a streaming performance claim."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from voxshield.prosody.analysis_service import analyze_prosody
from voxshield.prosody.baseline_service import LocalBaselineStore, enroll_baseline
from voxshield.prosody.config import ProsodyConfig
from voxshield.prosody.feature_extractor import extract_prosodic_features
from voxshield.prosody.quality_checks import assess_prosody_quality


def _stats(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(data)),
        "median_ms": float(np.median(data)),
        "p95_ms": float(np.percentile(data, 95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Phase 6 prosody analysis")
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    config = ProsodyConfig()
    if args.audio:
        audio, sample_rate = sf.read(args.audio, dtype="float32", always_2d=False)
    else:
        sample_rate = 16_000
        t = np.arange(20_000, dtype=np.float32) / sample_rate
        audio = (0.2 * np.sin(2 * np.pi * 130 * t)).astype(np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    validation: list[float] = []
    extraction: list[float] = []
    quality: list[float] = []
    comparison: list[float] = []
    total: list[float] = []
    benchmark_store_path = Path(tempfile.mkdtemp(prefix="voxshield-prosody-"))
    store = LocalBaselineStore(benchmark_store_path)
    baseline_features = [extract_prosodic_features(audio, sample_rate, config).features] * 3
    baseline = enroll_baseline(baseline_features, config)
    store.create(baseline)
    for _ in range(max(1, args.runs)):
        started = time.perf_counter()
        finite = bool(np.all(np.isfinite(audio)))
        validation.append((time.perf_counter() - started) * 1000.0)
        if not finite:
            continue
        started = time.perf_counter()
        result = extract_prosodic_features(audio, sample_rate, config)
        extraction.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        assess_prosody_quality(result, config)
        quality.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        analyzed = analyze_prosody(
            audio,
            sample_rate,
            baseline_id=baseline.baseline_id,
            config=config,
            baseline_store=store,
        )
        comparison.append(analyzed.latency_ms)
        total.append((time.perf_counter() - started) * 1000.0)
    report = {
        "status": "COMPLETED",
        "hardware": platform.platform(),
        "sample_rate_hz": sample_rate,
        "audio_duration_seconds": len(audio) / sample_rate,
        "benchmark_count": max(1, args.runs),
        "offline_only": True,
        "stages": {
            "validation": _stats(validation),
            "pitch_energy_timing_extraction": _stats(extraction),
            "quality": _stats(quality),
            "baseline_comparison_total_service": _stats(comparison),
            "total_analysis_loop": _stats(total),
        },
        "notes": ["Offline benchmark; does not establish Phase 7 streaming performance."],
    }
    shutil.rmtree(benchmark_store_path, ignore_errors=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
