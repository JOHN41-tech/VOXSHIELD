"""Command-line workflow for Phase 6 prosody analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from voxshield.evaluation.prosody.evaluate_features import evaluate_dataset_directory
from voxshield.prosody.analysis_service import analyze_prosody
from voxshield.prosody.baseline_service import LocalBaselineStore, enroll_baseline
from voxshield.prosody.config import ProsodyConfig, load_prosody_config
from voxshield.prosody.feature_extractor import extract_prosodic_features


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("the CLI requires a mono WAV file")
    return arr, int(sample_rate)


def _store(config: ProsodyConfig, path: str | None) -> LocalBaselineStore:
    return LocalBaselineStore(path or config.storage_path)


def _analyze(args: argparse.Namespace) -> int:
    config = load_prosody_config(args.config)
    audio, rate = _read_audio(args.audio)
    result = analyze_prosody(
        audio,
        rate,
        baseline_id=args.baseline_id,
        config=config,
        baseline_store=_store(config, args.store) if args.baseline_id else None,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.status.value != "FAILED" else 1


def _enroll(args: argparse.Namespace) -> int:
    config = load_prosody_config(args.config)
    features = []
    for path in args.audio:
        audio, rate = _read_audio(path)
        extraction = extract_prosodic_features(audio, rate, config)
        features.append(extraction.features)
    baseline = enroll_baseline(features, config, speaker_profile_id=args.speaker_profile_id)
    store = _store(config, args.store)
    store.create(baseline)
    print(json.dumps(baseline.as_dict(), indent=2, sort_keys=True))
    return 0


def _lifecycle(args: argparse.Namespace) -> int:
    config = load_prosody_config(args.config)
    store = _store(config, args.store)
    if args.operation == "get":
        output = store.get(args.baseline_id).as_dict()
    elif args.operation == "deactivate":
        output = store.deactivate(args.baseline_id).as_dict()
    else:
        output = {"deleted": store.delete(args.baseline_id), "baseline_id": args.baseline_id}
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    config = load_prosody_config(args.config)
    report = evaluate_dataset_directory(args.dataset, config, seed=args.seed)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VoxShield Phase 6 prosody analysis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="analyze one WAV without enrolling")
    analyze.add_argument("--audio", type=Path, required=True)
    analyze.add_argument("--baseline-id")
    analyze.add_argument("--store")
    analyze.add_argument("--config", type=Path)
    analyze.set_defaults(func=_analyze)
    enroll = subparsers.add_parser("enroll", help="explicitly enroll a behavioral baseline")
    enroll.add_argument("--audio", type=Path, nargs="+", required=True)
    enroll.add_argument("--speaker-profile-id")
    enroll.add_argument("--store")
    enroll.add_argument("--config", type=Path)
    enroll.set_defaults(func=_enroll)
    lifecycle = subparsers.add_parser("baseline", help="get, deactivate, or delete a baseline")
    lifecycle.add_argument("operation", choices=("get", "deactivate", "delete"))
    lifecycle.add_argument("--baseline-id", required=True)
    lifecycle.add_argument("--store")
    lifecycle.add_argument("--config", type=Path)
    lifecycle.set_defaults(func=_lifecycle)
    evaluate = subparsers.add_parser("evaluate", help="evaluate a WAV dataset directory")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.set_defaults(func=_evaluate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
