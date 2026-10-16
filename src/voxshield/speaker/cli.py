"""CLI commands for consented speaker verification and evaluation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from voxshield.speaker.embeddings.encoder import PREPROCESSING_VERSION, normalized_embedding
from voxshield.speaker.embeddings.pretrained_encoder import SpeechBrainEcapaEncoder
from voxshield.speaker.enrollment import EnrollmentService
from voxshield.speaker.evaluation import (
    VerificationTrial,
    build_trials,
    deterministic_robustness_variants,
    latency_summary,
    load_recordings,
    select_threshold,
    stratified_metrics,
    validate_disjoint_splits,
    verification_metrics,
    write_trial_manifest,
)
from voxshield.speaker.quality import SpeakerQualityPolicy, assess_sample
from voxshield.speaker.schemas import SpeakerProfile
from voxshield.speaker.storage import LocalProfileStore
from voxshield.speaker.verification import ThresholdPolicy, VerificationService, cosine_similarity


def _store_path(args: Any) -> Path:
    return Path(args.store or os.environ.get("VOXSHIELD_SPEAKER_STORE", "data/speaker-profiles"))


def _model(args: Any) -> SpeechBrainEcapaEncoder:
    return SpeechBrainEcapaEncoder(args.model_dir, device=args.device)


def _quality_policy(args: Any) -> SpeakerQualityPolicy:
    return SpeakerQualityPolicy(
        minimum_voiced_seconds=args.minimum_voiced_seconds,
        maximum_silence_ratio=args.maximum_silence_ratio,
        maximum_clipping_ratio=args.maximum_clipping_ratio,
    )


def handle_enroll(args: Any) -> int:
    store = LocalProfileStore(_store_path(args))
    service = EnrollmentService(_model(args), store, quality_policy=_quality_policy(args))
    result = service.enroll(
        speaker_profile_id=args.profile_id,
        samples=[Path(input_path) for input_path in args.input],
        consent_confirmed=args.consent_confirmed,
        consent_reference=args.consent_reference,
        authorized=args.authorized,
    )
    _emit(
        {
            "status": "ENROLLED" if result.profile_created else "NOT_ENROLLED",
            "speaker_profile_id": result.speaker_profile_id,
            "profile_version": result.profile_version or None,
            "accepted_sample_count": result.accepted_sample_count,
            "samples": [
                {
                    "sample_index": row.sample_index,
                    "accepted": row.accepted,
                    "rejection_reasons": list(row.rejection_reasons),
                }
                for row in result.samples
            ],
        },
        args.compact,
    )
    return 0 if result.profile_created else 1


def handle_verify(args: Any) -> int:
    store = LocalProfileStore(_store_path(args))
    policy = ThresholdPolicy(args.threshold, args.threshold_version)
    service = VerificationService(
        _model(args), store, threshold_policy=policy, quality_policy=_quality_policy(args)
    )
    result = service.verify(audio=Path(args.input), speaker_profile_id=args.profile_id)
    _emit(result.as_dict(), args.compact)
    return 0


def handle_profile(args: Any) -> int:
    store = LocalProfileStore(_store_path(args))
    if args.delete:
        deleted = store.delete(args.profile_id)
        _emit(
            {
                "status": "DELETED" if deleted else "NOT_FOUND",
                "speaker_profile_id": args.profile_id,
                "deleted": deleted,
            },
            args.compact,
        )
        return 0 if deleted else 1
    profile: SpeakerProfile = store.get(args.profile_id)
    # Never print the embedding or consent reference through a public CLI surface.
    _emit(
        {
            "speaker_profile_id": profile.speaker_profile_id,
            "profile_version": profile.profile_version,
            "active": profile.active,
            "encoder_id": profile.encoder_id,
            "encoder_version": profile.encoder_version,
            "embedding_dimension": profile.embedding_dimension,
            "enrollment_sample_count": profile.enrollment_sample_count,
            "preprocessing_version": profile.preprocessing_version,
            "created_at": profile.created_at,
            "expires_at": profile.expires_at,
            "quality_summary": profile.quality_summary,
        },
        args.compact,
    )
    return 0


def _emit(value: dict[str, Any], compact: bool) -> None:
    print(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":") if compact else None,
            indent=None if compact else 2,
        )
    )


def _duration_band(seconds: float) -> str:
    if seconds < 3.0:
        return "under_3s"
    if seconds <= 6.0:
        return "3_to_6s"
    return "over_6s"


def handle_evaluate(args: Any) -> int:
    manifest = Path(args.manifest)
    if not manifest.is_file():
        _emit(
            {
                "status": "NOT_RUN",
                "reason": "speaker-labeled evaluation manifest is absent",
                "manifest": str(manifest),
            },
            args.compact,
        )
        return 2
    records = load_recordings(manifest)
    dev_trials = build_trials(records, split="dev", seed=args.seed, max_trials=args.max_trials)
    test_trials = build_trials(records, split="test", seed=args.seed, max_trials=args.max_trials)
    trials = dev_trials + test_trials
    try:
        validate_disjoint_splits(trials)
    except ValueError as exc:
        _emit({"status": "NOT_RUN", "reason": str(exc)}, args.compact)
        return 2
    if not dev_trials or not test_trials:
        _emit(
            {
                "status": "NOT_RUN",
                "reason": "both dev and test trials are required",
                "n_dev_trials": len(dev_trials),
                "n_test_trials": len(test_trials),
            },
            args.compact,
        )
        return 2
    encoder = _model(args)
    clean_cache: dict[str, np.ndarray | None] = {}
    variant_cache: dict[tuple[str, str], np.ndarray | None] = {}
    quality_reports: dict[str, dict[str, Any]] = {}
    preprocessing_by_path: dict[str, float] = {}
    embedding_by_path: dict[str, float] = {}
    similarity_ms: list[float] = []
    failure_counts: dict[str, int] = {}

    def encode_path(path: str) -> np.ndarray | None:
        if path in clean_cache:
            return clean_cache[path]
        started = time.perf_counter()
        assessed = assess_sample(Path(path), quality_policy=_quality_policy(args))
        preprocessing_by_path[path] = (time.perf_counter() - started) * 1000.0
        quality_reports[path] = {
            "acceptable": assessed.quality.acceptable,
            "reasons": assessed.quality.rejection_reasons,
            "duration_seconds": assessed.quality.duration_seconds,
            "voiced_seconds": assessed.quality.voiced_seconds,
            "snr_db": assessed.quality.snr_db,
        }
        if not assessed.quality.acceptable:
            for reason in assessed.quality.rejection_reasons:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
            assessed.samples = np.empty(0, dtype=np.float32)
            clean_cache[path] = None
            return None
        try:
            started = time.perf_counter()
            output = encoder.encode(assessed.samples, assessed.sample_rate)
            if (
                output.encoder_id != encoder.encoder_id
                or output.encoder_version != encoder.encoder_version
                or output.preprocessing_version != PREPROCESSING_VERSION
            ):
                raise ValueError("encoder returned incompatible embedding metadata")
            vector = normalized_embedding(
                np.asarray(output.vector), dimension=encoder.embedding_dimension
            )
            embedding_by_path[path] = (time.perf_counter() - started) * 1000.0
            clean_cache[path] = vector
            return vector
        except Exception:
            failure_counts["EMBEDDING_FAILURE"] = failure_counts.get("EMBEDDING_FAILURE", 0) + 1
            clean_cache[path] = None
            return None
        finally:
            assessed.samples = np.empty(0, dtype=np.float32)

    def encode_variant(path: str, condition: str) -> np.ndarray | None:
        key = (path, condition)
        if key in variant_cache:
            return variant_cache[key]
        assessed = assess_sample(Path(path), quality_policy=_quality_policy(args))
        if not assessed.quality.acceptable:
            assessed.samples = np.empty(0, dtype=np.float32)
            variant_cache[key] = None
            return None
        variants = deterministic_robustness_variants(assessed.samples, assessed.sample_rate)
        try:
            output = encoder.encode(variants[condition], assessed.sample_rate)
            if (
                output.encoder_id != encoder.encoder_id
                or output.encoder_version != encoder.encoder_version
                or output.preprocessing_version != PREPROCESSING_VERSION
            ):
                raise ValueError("encoder returned incompatible embedding metadata")
            vector = normalized_embedding(
                np.asarray(output.vector), dimension=encoder.embedding_dimension
            )
            variant_cache[key] = vector
            return vector
        except Exception:
            variant_cache[key] = None
            return None
        finally:
            assessed.samples = np.empty(0, dtype=np.float32)
            for waveform in variants.values():
                waveform.fill(0.0)

    def score_trials(
        rows: tuple[VerificationTrial, ...], *, condition: str | None = None
    ) -> tuple[list[VerificationTrial], list[bool], list[float]]:
        scored_rows: list[VerificationTrial] = []
        labels: list[bool] = []
        scores: list[float] = []
        for trial in rows:
            left = encode_path(trial.enrollment_path)
            right = (
                encode_path(trial.test_path)
                if condition is None
                else encode_variant(trial.test_path, condition)
            )
            if left is None or right is None:
                continue
            started = time.perf_counter()
            score = cosine_similarity(left, right)
            similarity_ms.append((time.perf_counter() - started) * 1000.0)
            enrollment_quality = quality_reports.get(trial.enrollment_path, {})
            test_quality = quality_reports.get(trial.test_path, {})
            scored_rows.append(
                replace(
                    trial,
                    enrollment_duration_seconds=enrollment_quality.get(
                        "duration_seconds", trial.enrollment_duration_seconds
                    ),
                    test_duration_seconds=test_quality.get(
                        "duration_seconds", trial.test_duration_seconds
                    ),
                    enrollment_quality_status="ACCEPTABLE",
                    test_quality_status="ACCEPTABLE",
                )
            )
            labels.append(trial.same_speaker)
            scores.append(score)
        return scored_rows, labels, scores

    _scored_dev, dev_labels, dev_scores = score_trials(dev_trials)
    scored_test, test_labels, test_scores = score_trials(test_trials)
    if args.trials_output:
        write_trial_manifest(scored_test + _scored_dev, args.trials_output, seed=args.seed)
    try:
        threshold = select_threshold(
            dev_labels,
            dev_scores,
            objective=args.threshold_policy,
            target_far=args.target_far,
            target_frr=args.target_frr,
        )
    except ValueError as exc:
        _emit(
            {"status": "NOT_RUN", "reason": f"development trials are insufficient: {exc}"},
            args.compact,
        )
        return 2

    test_by_duration: dict[str, tuple[list[bool], list[float]]] = {}
    for trial, label, score in zip(scored_test, test_labels, test_scores, strict=True):
        if trial.test_duration_seconds is None:
            continue
        key = _duration_band(trial.test_duration_seconds)
        test_by_duration.setdefault(key, ([], []))
        test_by_duration[key][0].append(label)
        test_by_duration[key][1].append(score)

    robustness: dict[str, Any] = {
        "status": "NOT_RUN",
        "reason": "pass --robustness to run deterministic gain/noise conditions",
    }
    if args.robustness:
        conditions: dict[str, Any] = {}
        for condition in ("clean", "gain_minus_6db", "noise_26db_snr"):
            if condition == "clean":
                _rows, labels, scores = scored_test, test_labels, test_scores
            else:
                _rows, labels, scores = score_trials(test_trials, condition=condition)
            conditions[condition] = {
                "n_trials_scored": len(scores),
                "metrics": verification_metrics(labels, scores, threshold),
            }
        robustness = {
            "status": "MEASURED",
            "method": "deterministic test-side gain/noise transforms; clean enrollment references",
            "conditions": conditions,
        }

    report = {
        "status": "MEASURED",
        "protocol": {
            "seed": args.seed,
            "split": "speaker-disjoint dev/test; within-test same-speaker verification pairs",
            "n_dev_trials_generated": len(dev_trials),
            "n_test_trials_generated": len(test_trials),
            "n_dev_trials_scored": len(dev_scores),
            "n_test_trials_scored": len(test_scores),
            "trials_output": args.trials_output,
        },
        "model": {
            "encoder_id": encoder.encoder_id,
            "encoder_version": encoder.encoder_version,
            "embedding_dimension": encoder.embedding_dimension,
            "device": args.device,
        },
        "score_interpretation": "cosine similarity in [-1, 1], not a probability",
        "calibration_status": "NOT_CALIBRATED",
        "threshold_selection": {
            "split": "dev",
            "objective": args.threshold_policy,
            "threshold_version": f"{args.threshold_policy}-dev-v1",
            "development_manifest": str(manifest.resolve()),
            "encoder_id": encoder.encoder_id,
            "encoder_version": encoder.encoder_version,
            "target_far": args.target_far if args.threshold_policy == "target_far" else None,
            "target_frr": args.target_frr if args.threshold_policy == "target_frr" else None,
            "threshold": threshold,
        },
        "dev": verification_metrics(dev_labels, dev_scores, threshold),
        "test": verification_metrics(test_labels, test_scores, threshold),
        "test_by_enrollment_dataset": stratified_metrics(
            scored_test, test_labels, test_scores, threshold=threshold, attribute="dataset_id"
        ),
        "test_by_test_dataset": stratified_metrics(
            scored_test, test_labels, test_scores, threshold=threshold, attribute="test_dataset_id"
        ),
        "test_by_test_session": stratified_metrics(
            scored_test, test_labels, test_scores, threshold=threshold, attribute="session_id"
        ),
        "test_by_duration_band": {
            key: verification_metrics(labels, scores, threshold)
            for key, (labels, scores) in sorted(test_by_duration.items())
        },
        "quality": {
            "unique_recordings_assessed": len(quality_reports),
            "unique_recordings_acceptable": sum(
                int(item["acceptable"]) for item in quality_reports.values()
            ),
            "rejection_reason_counts": failure_counts,
        },
        "latency_ms": {
            "preprocessing_per_unique_recording": latency_summary(preprocessing_by_path.values()),
            "embedding_extraction_per_unique_recording": latency_summary(
                embedding_by_path.values()
            ),
            "similarity_per_scored_trial": latency_summary(similarity_ms),
            "single_sample_preprocess_plus_embedding": latency_summary(
                preprocessing_by_path[path] + embedding_by_path[path] for path in embedding_by_path
            ),
        },
        "robustness": robustness,
        "limitations": [
            "No calibrated probability is reported.",
            "Metrics are valid only for this licensed manifest and split protocol.",
            "Dataset/session/duration slices are descriptive and may be unavailable or small.",
            "Gain/noise tests do not establish codec, microphone, language, or clone robustness.",
        ],
    }
    _emit(report, args.compact)
    return 0


def add_parser(sub: Any) -> None:
    speaker = sub.add_parser(
        "speaker", help="Consent-based speaker profile enrollment and verification."
    )
    commands = speaker.add_subparsers(dest="speaker_command", required=True)

    def model_options(parser: Any) -> None:
        parser.add_argument(
            "--model-dir",
            required=True,
            help="Local pretrained model directory; no model download is performed.",
        )
        parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
        parser.add_argument(
            "--store", default=None, help="Profile store directory (or VOXSHIELD_SPEAKER_STORE)."
        )
        parser.add_argument("--minimum-voiced-seconds", type=float, default=2.0)
        parser.add_argument("--maximum-silence-ratio", type=float, default=0.60)
        parser.add_argument("--maximum-clipping-ratio", type=float, default=0.01)
        parser.add_argument("--compact", action="store_true")

    enroll = commands.add_parser("enroll", help="Create a consented speaker profile.")
    enroll.add_argument(
        "--input",
        action="append",
        required=True,
        help="Consented sample path; repeat this flag to supply more than one sample.",
    )
    enroll.add_argument("--profile-id", required=True)
    enroll.add_argument("--consent-reference", required=True)
    enroll.add_argument("--authorized", action="store_true", required=True)
    enroll.add_argument("--consent-confirmed", action="store_true", required=True)
    model_options(enroll)
    enroll.set_defaults(func=handle_enroll)

    verify = commands.add_parser(
        "verify", help="Compare a sample to the exact requested enrolled profile."
    )
    verify.add_argument("--input", required=True)
    verify.add_argument("--profile-id", required=True)
    verify.add_argument("--threshold", type=float, default=0.75)
    verify.add_argument("--threshold-version", default="development-uncalibrated-v1")
    model_options(verify)
    verify.set_defaults(func=handle_verify)

    profile = commands.add_parser("profile", help="Inspect or delete one UUID profile.")
    profile.add_argument("--profile-id", required=True)
    profile.add_argument(
        "--delete", action="store_true", help="Permanently remove only this profile file."
    )
    profile.add_argument("--store", default=None)
    profile.add_argument("--compact", action="store_true")
    profile.set_defaults(func=handle_profile)

    evaluate = commands.add_parser("evaluate", help="Evaluate a speaker-labeled JSONL manifest.")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--seed", type=int, default=17)
    evaluate.add_argument("--max-trials", type=int, default=10_000)
    evaluate.add_argument("--model-dir", required=True)
    evaluate.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    evaluate.add_argument("--minimum-voiced-seconds", type=float, default=2.0)
    evaluate.add_argument("--maximum-silence-ratio", type=float, default=0.60)
    evaluate.add_argument("--maximum-clipping-ratio", type=float, default=0.01)
    evaluate.add_argument("--trials-output", default=None)
    evaluate.add_argument(
        "--threshold-policy", choices=("eer", "target_far", "target_frr"), default="eer"
    )
    evaluate.add_argument("--target-far", type=float, default=0.01)
    evaluate.add_argument("--target-frr", type=float, default=0.05)
    evaluate.add_argument("--robustness", action="store_true")
    evaluate.add_argument("--compact", action="store_true")
    evaluate.set_defaults(func=handle_evaluate)
