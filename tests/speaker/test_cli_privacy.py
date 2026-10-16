from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voxshield.cli import build_parser
from voxshield.speaker.embeddings.encoder import PREPROCESSING_VERSION
from voxshield.speaker.quality import SpeakerQualityPolicy
from voxshield.speaker.schemas import Decision, Embedding, QualityStatus, VerificationResult


def test_speaker_cli_is_registered_and_enrollment_requires_explicit_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "speaker",
            "enroll",
            "--input",
            "voice.wav",
            "--input",
            "voice-second.wav",
            "--profile-id",
            str(uuid.uuid4()),
            "--consent-reference",
            "opaque-ref",
            "--authorized",
            "--consent-confirmed",
            "--model-dir",
            "/local/model",
        ]
    )
    assert args.speaker_command == "enroll"
    assert args.input == ["voice.wav", "voice-second.wav"]
    assert args.authorized is True
    assert args.consent_confirmed is True


def test_verification_result_contains_no_embedding_vector() -> None:
    result = VerificationResult(
        verification_id="opaque-id",
        speaker_profile_id=str(uuid.uuid4()),
        decision=Decision.INCONCLUSIVE,
        similarity_score=None,
        verification_score=None,
        threshold=0.75,
        threshold_version="test-v1",
        quality_status=QualityStatus.INSUFFICIENT,
        quality_reasons=("INSUFFICIENT_SPEECH",),
        quality_metrics={"duration_seconds": 0.0},
        encoder_id="ecapa",
        encoder_version="1",
        profile_version=1,
        latency_ms=1.0,
    ).as_dict()
    assert "embedding" not in result
    assert result["verification_score"] is None
    assert result["calibration_status"] == "NOT_CALIBRATED"


def test_quality_policy_is_configurable_and_validated() -> None:
    policy = SpeakerQualityPolicy(
        minimum_voiced_seconds=1.25,
        maximum_silence_ratio=0.4,
        maximum_clipping_ratio=0.02,
    )
    assert policy.minimum_voiced_seconds == 1.25
    try:
        SpeakerQualityPolicy(maximum_silence_ratio=1.2)
    except ValueError as exc:
        assert "silence ratio" in str(exc)
    else:
        raise AssertionError("invalid quality policy was accepted")


def test_evaluation_cli_runs_a_separate_speaker_disjoint_dev_test_protocol(
    tmp_path: Path, speech_samples: np.ndarray, monkeypatch: Any, capsys: Any
) -> None:
    class FakeEncoder:
        encoder_id = "fixture-encoder"
        encoder_version = "fixture-v1"
        sample_rate = 16_000
        embedding_dimension = 3

        def encode(self, audio: np.ndarray, sample_rate: int) -> Embedding:
            assert sample_rate == self.sample_rate
            return Embedding(
                (1.0, 2.0, 3.0),
                self.encoder_id,
                self.encoder_version,
                PREPROCESSING_VERSION,
            )

    manifest = tmp_path / "recordings.jsonl"
    recordings = []
    for split, speakers in (("dev", ("dev-a", "dev-b")), ("test", ("test-a", "test-b"))):
        for speaker in speakers:
            for index in range(2):
                path = tmp_path / f"{split}-{speaker}-{index}.wav"
                sf.write(path, speech_samples, 16_000)
                recordings.append(
                    {
                        "recording_id": f"{split}-{speaker}-{index}",
                        "path": path.name,
                        "speaker_id": speaker,
                        "dataset_id": f"fixture-{split}",
                        "split": split,
                        "session_id": f"session-{index}",
                    }
                )
    manifest.write_text("".join(json.dumps(row) + "\n" for row in recordings), encoding="utf-8")
    trials_output = tmp_path / "generated-trials.jsonl"
    monkeypatch.setattr("voxshield.speaker.cli._model", lambda args: FakeEncoder())
    args = build_parser().parse_args(
        [
            "speaker",
            "evaluate",
            "--manifest",
            str(manifest),
            "--model-dir",
            "/not-used-by-fixture",
            "--trials-output",
            str(trials_output),
            "--robustness",
        ]
    )
    assert args.func(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "MEASURED"
    assert report["protocol"]["n_dev_trials_scored"] > 0
    assert report["test"]["status"] == "MEASURED"
    assert report["test_by_test_dataset"]["fixture-test"]["status"] == "MEASURED"
    assert report["robustness"]["status"] == "MEASURED"
    assert set(report["robustness"]["conditions"]) == {
        "clean",
        "gain_minus_6db",
        "noise_26db_snr",
    }
    assert "_manifest_seed" in trials_output.read_text(encoding="utf-8")
