"""Reproducible verification trials, metrics, threshold selection, and stress slices."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class SpeakerRecording:
    recording_id: str
    path: str
    speaker_id: str
    dataset_id: str
    split: str
    session_id: str | None = None
    duration_seconds: float | None = None
    quality_status: str | None = None


@dataclass(frozen=True, slots=True)
class VerificationTrial:
    trial_id: str
    enrollment_recording_id: str
    enrollment_path: str
    test_recording_id: str
    test_path: str
    enrollment_speaker_id: str
    test_speaker_id: str
    same_speaker: bool
    dataset_id: str
    test_dataset_id: str
    split: str
    enrollment_session_id: str | None = None
    session_id: str | None = None
    enrollment_duration_seconds: float | None = None
    test_duration_seconds: float | None = None
    enrollment_quality_status: str | None = None
    test_quality_status: str | None = None


def build_trials(
    records: Iterable[SpeakerRecording],
    *,
    split: str,
    seed: int = 17,
    max_trials: int = 10_000,
) -> tuple[VerificationTrial, ...]:
    """Build reproducible genuine/impostor trials from distinct source recordings.

    Same-speaker trials prefer different sessions where session metadata supports
    it; otherwise they use any two distinct recordings. Speaker IDs only govern
    trial construction and never enter the embedding encoder.
    """
    if max_trials < 2:
        raise ValueError("max_trials must be at least two to retain both trial classes")
    groups: dict[str, list[SpeakerRecording]] = defaultdict(list)
    for record in records:
        if record.split == split:
            groups[record.speaker_id].append(record)
    rng = np.random.default_rng(seed)
    same: list[VerificationTrial] = []
    per_speaker: dict[str, tuple[SpeakerRecording, SpeakerRecording]] = {}
    for speaker, group in sorted(groups.items()):
        unique = {item.recording_id: item for item in group}
        candidates = list(unique.values())
        pairs = [
            (left, right)
            for left, right in itertools.combinations(candidates, 2)
            if left.path != right.path
        ]
        if not pairs:
            continue
        cross_session = [
            pair
            for pair in pairs
            if pair[0].session_id is not None
            and pair[1].session_id is not None
            and pair[0].session_id != pair[1].session_id
        ]
        choices = cross_session or pairs
        left, right = choices[int(rng.integers(len(choices)))]
        per_speaker[speaker] = (left, right)
        same.append(_trial(left, right, True, split))

    different: list[VerificationTrial] = []
    speakers = sorted(per_speaker)
    for left_speaker, right_speaker in itertools.combinations(speakers, 2):
        left, right = per_speaker[left_speaker][0], per_speaker[right_speaker][1]
        if left.path != right.path:
            different.append(_trial(left, right, False, split))
    if not same or not different:
        return tuple((same + different)[:max_trials])
    rng.shuffle(same)
    rng.shuffle(different)
    same_count = min(len(same), max_trials // 2)
    different_count = min(len(different), max_trials - same_count)
    if same_count + different_count < max_trials:
        same_count = min(len(same), max_trials - different_count)
    result = same[:same_count] + different[:different_count]
    result.sort(key=lambda trial: trial.trial_id)
    return tuple(result[:max_trials])


def _trial(
    enrollment: SpeakerRecording,
    test: SpeakerRecording,
    same: bool,
    split: str,
) -> VerificationTrial:
    digest = hashlib.sha256(
        f"{split}:{enrollment.recording_id}:{test.recording_id}:{int(same)}".encode()
    ).hexdigest()[:20]
    return VerificationTrial(
        trial_id=f"trial-{digest}",
        enrollment_recording_id=enrollment.recording_id,
        enrollment_path=enrollment.path,
        test_recording_id=test.recording_id,
        test_path=test.path,
        enrollment_speaker_id=enrollment.speaker_id,
        test_speaker_id=test.speaker_id,
        same_speaker=same,
        dataset_id=enrollment.dataset_id,
        test_dataset_id=test.dataset_id,
        split=split,
        enrollment_session_id=enrollment.session_id,
        session_id=test.session_id,
        enrollment_duration_seconds=enrollment.duration_seconds,
        test_duration_seconds=test.duration_seconds,
        enrollment_quality_status=enrollment.quality_status,
        test_quality_status=test.quality_status,
    )


def write_trial_manifest(
    trials: Iterable[VerificationTrial], path: str | Path, *, seed: int
) -> None:
    """Write a deterministic JSONL trial manifest and its reproducibility seed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"_manifest_seed": seed}, sort_keys=True) + "\n")
        for trial in trials:
            handle.write(json.dumps(asdict(trial), sort_keys=True) + "\n")


def validate_disjoint_splits(trials: Iterable[VerificationTrial]) -> None:
    """Reject dev/test speaker overlap, label contradictions, and reused paths."""
    speakers: dict[str, set[str]] = defaultdict(set)
    path_speakers: dict[str, str] = {}
    for trial in trials:
        if trial.enrollment_recording_id == trial.test_recording_id:
            raise ValueError("a trial must use distinct enrollment and test recordings")
        if trial.enrollment_path == trial.test_path:
            raise ValueError("enrollment and test paths must be distinct")
        if trial.same_speaker != (trial.enrollment_speaker_id == trial.test_speaker_id):
            raise ValueError("trial label disagrees with speaker metadata")
        speakers[trial.split].update((trial.enrollment_speaker_id, trial.test_speaker_id))
        for path, speaker in (
            (trial.enrollment_path, trial.enrollment_speaker_id),
            (trial.test_path, trial.test_speaker_id),
        ):
            if path in path_speakers and path_speakers[path] != speaker:
                raise ValueError("the same source path is assigned to multiple speakers")
            path_speakers[path] = speaker
    if speakers.get("dev", set()) & speakers.get("test", set()):
        raise ValueError("speaker IDs overlap between dev and test splits")


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = labels == 1
    negatives = labels == 0
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_ranks = np.arange(1, len(scores) + 1, dtype=np.float64)
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        sorted_ranks[index:end] = (index + 1 + end) / 2.0
        index = end
    ranks = np.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _score_distribution(scores: np.ndarray) -> dict[str, float]:
    return {
        "n": int(scores.size),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "p05": float(np.percentile(scores, 5)),
        "p25": float(np.percentile(scores, 25)),
        "median": float(np.median(scores)),
        "p75": float(np.percentile(scores, 75)),
        "p95": float(np.percentile(scores, 95)),
        "max": float(np.max(scores)),
    }


def _operating_points(labels: np.ndarray, scores: np.ndarray) -> list[tuple[float, float, float]]:
    """Return (threshold, FAR, FRR) for each unique threshold in O(n log n)."""
    genuine = labels == 1
    impostor = labels == 0
    order = np.argsort(scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_genuine = genuine[order].astype(np.int64)
    ordered_impostor = impostor[order].astype(np.int64)
    genuine_prefix = np.r_[0, np.cumsum(ordered_genuine)]
    impostor_prefix = np.r_[0, np.cumsum(ordered_impostor)]
    thresholds = np.r_[np.min(scores) - 1e-9, np.unique(scores), np.max(scores) + 1e-9]
    result = []
    for threshold in thresholds:
        below = int(np.searchsorted(ordered_scores, threshold, side="left"))
        frr = float(genuine_prefix[below] / genuine.sum())
        far = float((impostor.sum() - impostor_prefix[below]) / impostor.sum())
        result.append((float(threshold), far, frr))
    return result


def verification_metrics(
    labels: Iterable[bool | int], scores: Iterable[float], threshold: float
) -> dict[str, Any]:
    """Report FAR, FRR, approximate EER, ROC-AUC, score distributions, and counts."""
    y = np.asarray(list(labels), dtype=np.int8)
    s = np.asarray(list(scores), dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or y.size != s.size or y.size == 0:
        return {"status": "NOT_AVAILABLE", "reason": "empty or mismatched trial labels/scores"}
    if not np.isin(y, (0, 1)).all() or not np.isfinite(s).all() or not np.isfinite(threshold):
        return {
            "status": "NOT_AVAILABLE",
            "reason": "invalid labels, threshold, or non-finite scores",
        }
    genuine, impostor = y == 1, y == 0
    if not genuine.any() or not impostor.any():
        return {
            "status": "NOT_AVAILABLE",
            "reason": "both genuine and impostor trials are required",
            "n_trials": int(y.size),
        }
    accepted = s >= threshold
    tp = int(np.count_nonzero(accepted & genuine))
    fn = int(np.count_nonzero(~accepted & genuine))
    fp = int(np.count_nonzero(accepted & impostor))
    tn = int(np.count_nonzero(~accepted & impostor))
    far = fp / int(impostor.sum())
    frr = fn / int(genuine.sum())
    operating_points = _operating_points(y, s)
    eer_gap, eer, eer_threshold, eer_far, eer_frr = min(
        (abs(point_far - point_frr), (point_far + point_frr) / 2, point, point_far, point_frr)
        for point, point_far, point_frr in operating_points
    )
    return {
        "status": "MEASURED",
        "n_trials": int(y.size),
        "n_same_speaker": int(genuine.sum()),
        "n_different_speaker": int(impostor.sum()),
        "far": far,
        "frr": frr,
        "eer": eer,
        "eer_threshold": eer_threshold,
        "eer_far": eer_far,
        "eer_frr": eer_frr,
        "eer_gap": eer_gap,
        "roc_auc": _roc_auc(y, s),
        "threshold": float(threshold),
        "confusion_matrix": {
            "true_match": tp,
            "false_reject": fn,
            "false_accept": fp,
            "true_non_match": tn,
        },
        "same_speaker_scores": _score_distribution(s[genuine]),
        "different_speaker_scores": _score_distribution(s[impostor]),
    }


def select_threshold(
    labels: Iterable[bool | int],
    scores: Iterable[float],
    *,
    objective: str = "eer",
    target_far: float = 0.01,
    target_frr: float = 0.05,
) -> float:
    """Choose a policy threshold from development scores only.

    `eer` balances FAR and FRR, `target_far` prioritizes lower false acceptance
    while maximizing genuine acceptance subject to the target, and `target_frr`
    prioritizes acceptance of genuine speakers while minimizing FAR subject to
    the target. No objective is applied to final-test scores.
    """
    y = np.asarray(list(labels), dtype=np.int8)
    s = np.asarray(list(scores), dtype=np.float64)
    base = verification_metrics(y, s, threshold=0.0)
    if base.get("status") != "MEASURED":
        raise ValueError("development threshold selection requires both trial classes")
    if objective not in {"eer", "target_far", "target_frr"}:
        raise ValueError("objective must be eer, target_far, or target_frr")
    if not 0 <= target_far <= 1 or not 0 <= target_frr <= 1:
        raise ValueError("target FAR/FRR must be within [0, 1]")
    points = _operating_points(y, s)
    if objective == "eer":
        return float(base["eer_threshold"])
    if objective == "target_far":
        eligible = [item for item in points if item[1] <= target_far]
        return min(eligible, key=lambda item: (item[0], item[2]))[0]
    eligible = [item for item in points if item[2] <= target_frr]
    return min(eligible, key=lambda item: (item[1], -item[0]))[0]


def select_eer_threshold(labels: Iterable[bool | int], scores: Iterable[float]) -> float:
    """Compatibility wrapper for the balanced EER development policy."""
    return select_threshold(labels, scores, objective="eer")


def stratified_metrics(
    trials: Sequence[VerificationTrial],
    labels: Sequence[bool | int],
    scores: Sequence[float],
    *,
    threshold: float,
    attribute: str,
) -> dict[str, dict[str, Any]]:
    """Report metrics for dataset/domain or test-session slices when labels exist."""
    if len(trials) != len(labels) or len(trials) != len(scores):
        raise ValueError("trial, label, and score arrays must have equal lengths")
    if attribute not in {"dataset_id", "session_id", "test_dataset_id"}:
        raise ValueError("unsupported evaluation slice attribute")
    groups: dict[str, tuple[list[bool | int], list[float]]] = {}
    for trial, label, score in zip(trials, labels, scores, strict=True):
        value = getattr(trial, attribute)
        if value is None:
            continue
        key = str(value)
        if key not in groups:
            groups[key] = ([], [])
        groups[key][0].append(label)
        groups[key][1].append(score)
    return {
        key: verification_metrics(group_labels, group_scores, threshold)
        for key, (group_labels, group_scores) in sorted(groups.items())
    }


def deterministic_robustness_variants(
    samples: np.ndarray, sample_rate: int
) -> dict[str, np.ndarray]:
    """Deterministically simulate clean, -6 dB gain and 26 dB SNR additive noise."""
    source = np.asarray(samples, dtype=np.float32)
    if source.ndim != 1 or sample_rate <= 0 or not np.isfinite(source).all():
        raise ValueError("robustness input must be finite mono audio with a positive sample rate")
    rng = np.random.default_rng(2025)
    noise = rng.standard_normal(source.size).astype(np.float32)
    noise_scale = max(float(np.sqrt(np.mean(source.astype(np.float64) ** 2))) * 0.05, 1e-6)
    return {
        "clean": source.copy(),
        "gain_minus_6db": np.ascontiguousarray(source * np.float32(10 ** (-6 / 20))),
        "noise_26db_snr": np.ascontiguousarray(source + noise * noise_scale),
    }


def latency_summary(samples: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(list(samples), dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return {"n": int(values.size), "mean_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}
    return {
        "n": int(values.size),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def load_recordings(path: str | Path) -> list[SpeakerRecording]:
    """Read JSONL recording metadata; labels are used only for trial generation."""
    result: list[SpeakerRecording] = []
    manifest_path = Path(path).resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if "_manifest_seed" in item:
                    continue
                recording = SpeakerRecording(**item)
                recording_path = Path(recording.path)
                resolved = (
                    recording_path
                    if recording_path.is_absolute()
                    else manifest_path.parent / recording_path
                )
                result.append(
                    SpeakerRecording(
                        **{
                            **asdict(recording),
                            "path": str(resolved.resolve()),
                        }
                    )
                )
            except Exception as exc:
                raise ValueError(f"invalid speaker manifest row {line_number}") from exc
    return result
