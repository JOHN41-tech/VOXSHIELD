from __future__ import annotations

import numpy as np
import pytest

from voxshield.speaker.evaluation import (
    SpeakerRecording,
    build_trials,
    deterministic_robustness_variants,
    select_eer_threshold,
    select_threshold,
    stratified_metrics,
    validate_disjoint_splits,
    verification_metrics,
)


def test_trial_generation_is_reproducible_and_uses_distinct_recordings() -> None:
    records = [
        SpeakerRecording(
            f"{split}-{speaker}-{i}", f"{split}/{speaker}/{i}.wav", speaker, "fixture", split
        )
        for split, speakers in (("dev", ("d1", "d2")), ("test", ("t1", "t2")))
        for speaker in speakers
        for i in range(2)
    ]
    dev = build_trials(records, split="dev", seed=19)
    assert dev == build_trials(records, split="dev", seed=19)
    assert {trial.same_speaker for trial in dev} == {True, False}
    assert all(trial.enrollment_recording_id != trial.test_recording_id for trial in dev)
    validate_disjoint_splits(dev + build_trials(records, split="test", seed=19))


def test_disjoint_gate_rejects_dev_test_speaker_overlap() -> None:
    records = [
        SpeakerRecording(f"{split}-{i}", f"{split}-{i}.wav", speaker, "fixture", split)
        for split, speaker in (("dev", "shared"), ("test", "shared"))
        for i in range(2)
    ]
    trials = build_trials(records, split="dev") + build_trials(records, split="test")
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_splits(trials)


def test_metrics_and_development_threshold() -> None:
    labels = [True, True, False, False]
    scores = [0.9, 0.8, 0.2, 0.1]
    threshold = select_eer_threshold(labels, scores)
    report = verification_metrics(labels, scores, threshold)
    assert report["status"] == "MEASURED"
    assert report["far"] == 0.0
    assert report["frr"] == 0.0
    assert report["roc_auc"] == pytest.approx(1.0)
    assert report["same_speaker_scores"]["n"] == 2
    assert report["different_speaker_scores"]["median"] == pytest.approx(0.15)
    assert verification_metrics([True, True], [0.8, 0.9], 0.5)["status"] == "NOT_AVAILABLE"


def test_target_far_and_target_frr_select_on_dev_scores() -> None:
    labels = [True, True, False, False]
    scores = [0.9, 0.8, 0.2, 0.1]
    low_far = select_threshold(labels, scores, objective="target_far", target_far=0.0)
    high_genuine_acceptance = select_threshold(
        labels, scores, objective="target_frr", target_frr=0.0
    )
    assert low_far == pytest.approx(0.8)
    assert high_genuine_acceptance == pytest.approx(0.8)


def test_auc_ties_and_dataset_session_slices_are_reported() -> None:
    tied = verification_metrics([True, False], [0.5, 0.5], 0.5)
    assert tied["roc_auc"] == pytest.approx(0.5)
    trials = build_trials(
        [
            SpeakerRecording("a1", "a1.wav", "a", "dataset-a", "test", "s1"),
            SpeakerRecording("a2", "a2.wav", "a", "dataset-a", "test", "s2"),
            SpeakerRecording("b1", "b1.wav", "b", "dataset-a", "test", "s1"),
            SpeakerRecording("b2", "b2.wav", "b", "dataset-a", "test", "s2"),
        ],
        split="test",
        seed=3,
    )
    assert (
        stratified_metrics(
            trials,
            [row.same_speaker for row in trials],
            [0.9 if row.same_speaker else 0.1 for row in trials],
            threshold=0.5,
            attribute="dataset_id",
        )["dataset-a"]["status"]
        == "MEASURED"
    )


def test_deterministic_robustness_variants_do_not_mutate_input(speech_samples: np.ndarray) -> None:
    before = speech_samples.copy()
    first = deterministic_robustness_variants(speech_samples, 16_000)
    second = deterministic_robustness_variants(speech_samples, 16_000)
    assert np.array_equal(speech_samples, before)
    assert first.keys() == second.keys()
    for name in first:
        assert np.array_equal(first[name], second[name])
