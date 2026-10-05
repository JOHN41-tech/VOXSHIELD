"""Tests for artefact persistence and the runner's protocol enforcement.

Three properties are worth more than the rest, and each has a test here:

1. An artefact round-trips to a model that reproduces the original scores
   exactly. A saved model that loads but scores differently is worse than one
   that fails to load, because nothing reports the difference.
2. The scaler travels with the estimator. This is asserted directly, because the
   failure mode it prevents is silent.
3. Test is scored exactly once, at the threshold chosen on dev. The runner is the
   only thing standing between a dev-tuned threshold and a threshold quietly
   re-tuned on test, so it is tested rather than trusted.
"""

from __future__ import annotations

import hashlib
import json
import wave
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

from voxshield.data.manifest import Manifest, ManifestEntry, ManifestHeader
from voxshield.data.schema import SampleRecord
from voxshield.training.artifacts import (
    ArtifactMetadata,
    load_artifact,
    save_artifact,
    scores_match,
)
from voxshield.training.config import (
    FeatureSpec,
    ModelSpec,
    ThresholdPolicy,
    TrainingConfig,
    TrainingError,
    TrainSpec,
)
from voxshield.training.datasets import FeatureMatrix, resolve_rows
from voxshield.training.models import build_baseline
from voxshield.training.runner import RunOutcome, run_baseline, run_from_artifact

SAMPLE_RATE = 16000
N_FRAMES = 24
N_BANDS = 12


def make_matrix(
    split: str,
    n: int,
    *,
    matrices: bool = False,
    seed: int = 7,
) -> FeatureMatrix:
    """Build a synthetic split whose classes are linearly separable."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n).astype(np.int64)
    vectors = rng.normal(size=(n, 10))
    vectors[labels == 1] += 1.4
    frames = None
    if matrices:
        frames = rng.normal(size=(n, N_FRAMES, N_BANDS)).astype(np.float32)
        frames[labels == 1] += 1.4
    return FeatureMatrix(
        split=split,
        vectors=vectors,
        labels=labels,
        sample_ids=tuple(f"{split}-{index}" for index in range(n)),
        metadata={"speaker": tuple(f"spk-{index % 6}" for index in range(n))},
        durations=tuple(2.0 for _ in range(n)),
        matrices=frames,
    )


def make_config(
    output_dir: Path,
    family: str = "mfcc_logreg",
    **train_overrides: Any,
) -> TrainingConfig:
    train = TrainSpec(
        seed=11,
        epochs=2,
        min_train_samples=10,
        min_dev_samples=10,
        **{"verify_leakage": False, **train_overrides},
    )
    return TrainingConfig(
        model_id=family,
        features=FeatureSpec(
            kind="log_mel" if family == "logmel_cnn" else "mfcc",
            n_mels=N_BANDS,
            n_coefficients=6,
            n_frames=N_FRAMES if family == "logmel_cnn" else None,
            target_frames=N_FRAMES if family == "logmel_cnn" else None,
        ),
        model=ModelSpec(family=family, max_iter=20, batch_size=16),
        train=train,
        threshold=ThresholdPolicy(strategy="eer"),
        output_dir=output_dir,
    )


def fit_model(config: TrainingConfig, train: FeatureMatrix, dev: FeatureMatrix) -> tuple[Any, Any]:
    """Fit one baseline and return it with its summary."""
    want = config.model.family == "logmel_cnn"
    model = build_baseline(
        config.model,
        n_frames=N_FRAMES if want else None,
        n_bands=N_BANDS if want else None,
        seed=config.train.seed,
        epochs=config.train.epochs,
        eval_every=config.train.eval_every,
        patience=config.train.patience,
    )
    return model, model.fit(train, dev=dev)


# --------------------------------------------------------------------------
# Artefact round-trips
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["mfcc_logreg", "mfcc_xgboost", "logmel_cnn"])
def test_artifact_round_trip_reproduces_scores(tmp_path: Path, family: str) -> None:
    train = make_matrix("train", 60, matrices=family == "logmel_cnn")
    dev = make_matrix("dev", 20, matrices=family == "logmel_cnn")
    test = make_matrix("test", 20, matrices=family == "logmel_cnn", seed=99)

    config = make_config(tmp_path, family)
    model, summary = fit_model(config, train, dev)

    metadata = ArtifactMetadata(
        model_id=family,
        family=family,
        config_hash=config.config_hash(),
        n_features=int(train.vectors.shape[1]),
        n_frames=N_FRAMES if family == "logmel_cnn" else None,
        n_bands=N_BANDS if family == "logmel_cnn" else None,
        class_counts={"0": 30, "1": 30},
        failed_decodes=({"sample_id": "gone.wav", "reason": "unreadable"},),
        notes=("test note",),
        selection_metric=summary.selection_metric,
        selection_value=summary.selection_value,
        epochs_run=summary.epochs_run,
        best_epoch=summary.best_epoch,
    )
    directory = save_artifact(model, metadata, tmp_path)
    loaded, loaded_metadata = load_artifact(directory)

    assert loaded.fitted
    assert loaded_metadata.family == family
    assert scores_match(model, loaded, test)


def test_tabular_artifact_stores_scaler_with_estimator(tmp_path: Path) -> None:
    """The scaler must ship in the same file as the estimator.

    A saved scaler cannot be separated from the estimator by accident, and this
    asserts the property that makes a tabular artefact loadable at all.
    """
    import joblib

    train = make_matrix("train", 60)
    dev = make_matrix("dev", 20)
    config = make_config(tmp_path, "mfcc_logreg")
    model, _ = fit_model(config, train, dev)

    metadata = ArtifactMetadata(model_id="mfcc_logreg", family="mfcc_logreg", n_features=10)
    directory = save_artifact(model, metadata, tmp_path)

    payload = joblib.load(directory / "model.joblib")
    assert set(payload) == {"scaler", "estimator"}
    assert payload["scaler"] is not None
    assert payload["estimator"] is not None
    # Only one weight file exists, so there is no second file to pair wrongly.
    assert sorted(p.name for p in directory.iterdir()) == ["metadata.json", "model.joblib"]


def test_cnn_artifact_records_frame_geometry(tmp_path: Path) -> None:
    """A CNN cannot be rebuilt without the frame geometry it was fitted on."""
    train = make_matrix("train", 60, matrices=True)
    dev = make_matrix("dev", 20, matrices=True)
    config = make_config(tmp_path, "logmel_cnn")
    model, _ = fit_model(config, train, dev)

    metadata = ArtifactMetadata(
        model_id="logmel_cnn",
        family="logmel_cnn",
        n_frames=N_FRAMES,
        n_bands=N_BANDS,
    )
    directory = save_artifact(model, metadata, tmp_path)
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))

    assert state["n_frames"] == N_FRAMES
    assert state["n_bands"] == N_BANDS
    assert (directory / "weights.pt").is_file()


def test_artifact_records_feature_spec_so_a_consumer_can_reproduce_it(tmp_path: Path) -> None:
    """A saved model is useless if the front end cannot be reconstructed."""
    from voxshield.training.artifacts import build_metadata

    config = make_config(tmp_path, "mfcc_logreg")
    train = make_matrix("train", 60)
    dev = make_matrix("dev", 20)
    model, summary = fit_model(config, train, dev)
    metadata = build_metadata(config, model, summary, n_features=10)

    assert metadata.feature_spec["n_mels"] == N_BANDS
    assert metadata.feature_spec["n_fft"] == config.features.n_fft

    # The round-trip must be exact, not merely close.
    rebuilt = FeatureSpec.from_mapping(metadata.feature_spec)
    assert rebuilt == config.features


def test_artifact_marks_test_evaluation_only_after_one_happens(tmp_path: Path) -> None:
    train = make_matrix("train", 60)
    dev = make_matrix("dev", 20)
    config = make_config(tmp_path, "mfcc_xgboost")
    fit_model(config, train, dev)
    metadata = ArtifactMetadata(model_id="mfcc_xgboost", family="mfcc_xgboost")

    assert metadata.test_evaluated is False, "a fresh artefact must not claim a test result"

    payload = {**metadata.to_dict(), "test_evaluated": True}
    assert ArtifactMetadata.from_dict(payload).test_evaluated is True


def test_refuses_to_save_an_unfitted_model(tmp_path: Path) -> None:
    config = make_config(tmp_path, "mfcc_logreg")
    model = build_baseline(config.model)
    metadata = ArtifactMetadata(model_id="mfcc_logreg", family="mfcc_logreg")

    with pytest.raises(TrainingError, match="unfitted"):
        save_artifact(model, metadata, tmp_path)


def test_loading_a_directory_that_is_not_an_artifact_is_refused(tmp_path: Path) -> None:
    (tmp_path / "somewhere_else").mkdir()
    with pytest.raises(TrainingError, match="not a VoxShield artefact"):
        load_artifact(tmp_path / "somewhere_else")


def test_artifact_rejects_a_future_format_version(tmp_path: Path) -> None:
    metadata = ArtifactMetadata(model_id="mfcc_logreg", family="mfcc_logreg")
    payload = {**metadata.to_dict(), "format_version": 99}
    with pytest.raises(TrainingError, match="format version"):
        ArtifactMetadata.from_dict(payload)


def test_artifact_rejects_an_unknown_metadata_field(tmp_path: Path) -> None:
    metadata = ArtifactMetadata(model_id="mfcc_logreg", family="mfcc_logreg")
    payload = {**metadata.to_dict(), "mystery": 1}
    with pytest.raises(TrainingError, match="unknown field"):
        ArtifactMetadata.from_dict(payload)


def test_wrong_estimator_family_in_the_artefact_is_caught(tmp_path: Path) -> None:
    """A family mismatch must fail loudly, not score with the wrong model."""
    import joblib

    train = make_matrix("train", 60)
    dev = make_matrix("dev", 20)
    config = make_config(tmp_path, "mfcc_logreg")
    model, _ = fit_model(config, train, dev)
    metadata = ArtifactMetadata(model_id="mfcc_xgboost", family="mfcc_xgboost")
    directory = save_artifact(model, metadata, tmp_path)
    # Claim XGBoost while the file holds a logistic regression.
    (directory / "metadata.json").write_text(json.dumps(metadata.to_dict()), encoding="utf-8")

    with pytest.raises(TrainingError, match="declares family"):
        load_artifact(directory)
    assert (directory / "model.joblib").is_file()
    assert joblib is not None


def test_decode_failure_entries_must_be_pairs(tmp_path: Path) -> None:
    from voxshield.training.artifacts import _decode_entries

    with pytest.raises(TrainingError, match="sample_id, reason"):
        _decode_entries(("just-a-string",))

    converted = _decode_entries((("a.wav", "unreadable"),))
    assert converted == ({"sample_id": "a.wav", "reason": "unreadable"},)


# --------------------------------------------------------------------------
# Runner protocol
# --------------------------------------------------------------------------


def empty_manifest() -> Manifest:
    header = ManifestHeader(
        dataset_build_id="empty",
        config_hash="0" * 64,
        content_fingerprint="1" * 64,
        created_at="2026-10-05T00:00:00Z",
        split="all",
        n_rows=0,
        n_active=0,
        voxshield_version="0.0.0",
        schema_version=1,
        record_type="audio_segment",
        datasets={},
    )
    return Manifest(header, ())


def _write_wav(path: Path, seconds: float, spoof: bool, seed: int) -> int:
    rng = np.random.default_rng(seed)
    times = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    body = 0.1 * np.sin(2 * np.pi * (140 if not spoof else 320) * times)
    body += rng.normal(0, 0.02, times.size)
    if spoof:
        # A clearly separable tone, so the fixture tests plumbing rather than
        # model quality. Its scores are not a performance claim.
        body += 0.08 * np.sin(2 * np.pi * 2400 * times)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes((np.clip(body, -1, 1) * 32767).astype("<i2").tobytes())
    return times.size


def audio_manifest(root: Path) -> Manifest:
    """Build a leakage-free manifest over real 16 kHz wav files.

    Every provenance axis is scoped per split: the noise seed includes the sample
    id, so no two rows share a file hash, and parent ids carry the split. A
    fixture that leaked on ``file`` or ``parent`` would make a leakage test pass
    for the wrong reason and would stop exercising the axis it names.
    """
    records: list[SampleRecord] = []
    for split, count in (("train", 40), ("dev", 20), ("test", 20)):
        for index in range(count):
            spoof = index % 2 == 1
            sample_id = f"{split}-{index:03d}"
            relative = f"{split}/{sample_id}.wav"
            seed = zlib.crc32(sample_id.encode())
            samples = _write_wav(root / relative, 0.5, spoof, seed=seed)
            audio = (root / relative).read_bytes()
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    dataset_id="fixture",
                    audio_path=relative,
                    label="spoof" if spoof else "bona_fide",
                    label_index=1 if spoof else 0,
                    split=split,
                    parent_id=f"utt-{split}-{index:03d}",
                    segment_index=0,
                    start_seconds=0.0,
                    duration_seconds=samples / SAMPLE_RATE,
                    sample_rate=SAMPLE_RATE,
                    speech_seconds=samples / SAMPLE_RATE,
                    coverage=1.0,
                    is_padded=False,
                    waveform_samples=samples,
                    speaker_id=f"spk-{split}-{index}",
                    generator_id="ttsA" if spoof else "real",
                    language="en",
                    codec="pcm",
                    session_id=f"sess-{split}",
                    attack_type="vc" if spoof else "",
                    channel="mono",
                    device="synthetic",
                    recorded_at="2026-01-01T00:00:00Z",
                    file_hash=hashlib.sha256(audio).hexdigest(),
                    content_hash=hashlib.sha256(audio + sample_id.encode()).hexdigest(),
                    source_split=split,
                    preprocessing_version="fixture-v1",
                    dataset_build_id="fixture-build",
                    extra={},
                )
            )
    header = ManifestHeader(
        dataset_build_id="fixture-build",
        config_hash="0" * 64,
        content_fingerprint="1" * 64,
        created_at="2026-10-05T00:00:00Z",
        split="all",
        n_rows=len(records),
        n_active=len(records),
        voxshield_version="0.0.0",
        schema_version=1,
        record_type="audio_segment",
        datasets={"fixture": {}},
    )
    return Manifest(header, tuple(ManifestEntry(record) for record in records))


def _leaky_manifest(root: Path) -> Manifest:
    """A manifest whose test split shares one speaker with train.

    Exactly one test row is rewritten, so the only axis in violation is
    ``speaker``. Anything else leaking here would make this test pass for the
    wrong reason.
    """
    base = audio_manifest(root)
    train_speaker = next(e.record.speaker_id for e in base.entries if e.record.split == "train")
    entries = list(base.entries)
    for position, entry in enumerate(entries):
        if entry.record.split == "test":
            entries[position] = ManifestEntry(replace(entry.record, speaker_id=train_speaker))
            break
    return Manifest(replace(base.header, n_rows=len(entries)), tuple(entries))


def test_train_to_test_speaker_leakage_is_refused(tmp_path: Path) -> None:
    """Leakage between train and test must stop the run, not the EER.

    This is the overlap that matters most: the model has seen that speaker's
    voice during fitting, so the test EER flatters the model and every number
    derived from it is wrong. The gate has to see the test identities, which is
    why they are resolved before fitting even though nothing is scored yet.
    """
    manifest = _leaky_manifest(tmp_path)
    config = make_config(tmp_path, "mfcc_logreg", verify_leakage=True)

    result = run_baseline(config, manifest, root=tmp_path)

    assert result.outcome is RunOutcome.FAILED
    assert "leakage gates failed" in result.reason
    assert "speaker" in result.reason
    assert result.artifact_path is None
    assert result.model is None


def test_skipping_test_evaluation_does_not_resolve_the_test_split(tmp_path: Path) -> None:
    """With test evaluation off, the test split is never even resolved.

    Keeping this literal matters: resolving test rows is metadata only and is
    safe, but a future change that starts decoding them would break the promise
    the flag makes.
    """
    seen: list[str] = []

    def spy(manifest: Any, split: str) -> tuple[Any, ...]:
        seen.append(split)
        return resolve_rows(manifest, split)

    manifest = _leaky_manifest(tmp_path)
    config = make_config(tmp_path, "mfcc_logreg", verify_leakage=True)

    with mock.patch("voxshield.training.runner.resolve_rows", side_effect=spy):
        result = run_baseline(config, manifest, root=tmp_path, evaluate_test=False)

    assert result.outcome is RunOutcome.NOT_RUN
    assert "test" not in seen, f"test split was resolved: {seen}"


def test_absent_corpus_is_not_run_not_failed(tmp_path: Path) -> None:
    """The project's real state must be reported as NOT RUN, never as a failure."""
    result = run_baseline(make_config(tmp_path), empty_manifest(), root=tmp_path)

    assert result.outcome is RunOutcome.NOT_RUN
    assert result.measured is False
    assert "no rows" in result.reason
    assert result.test_report is not None
    assert str(result.test_report.status) == "not_run"


def test_dry_run_never_fits_or_scores(tmp_path: Path) -> None:
    manifest = audio_manifest(tmp_path)
    result = run_baseline(make_config(tmp_path), manifest, root=tmp_path, dry_run=True)

    assert result.outcome is RunOutcome.NOT_RUN
    assert result.model is None
    assert result.threshold is None
    assert result.artifact_path is None
    assert "dry run" in result.reason


def test_measured_run_applies_the_dev_threshold_to_test_unchanged(tmp_path: Path) -> None:
    """The core protocol claim: dev chooses, test inherits, test is not tuned."""
    manifest = audio_manifest(tmp_path)
    result = run_baseline(make_config(tmp_path), manifest, root=tmp_path)

    assert result.outcome is RunOutcome.MEASURED
    assert result.threshold is not None
    assert result.test_report is not None
    assert result.test_report.to_dict()["threshold"] == pytest.approx(result.threshold)
    assert result.metadata is not None
    assert result.metadata.test_evaluated is True
    # Only dev is allowed to drive selection.
    assert result.summary is not None
    assert "test" not in " ".join(result.summary.notes).lower()


def test_skipping_test_evaluation_never_reads_the_test_split(tmp_path: Path) -> None:
    manifest = audio_manifest(tmp_path)
    result = run_baseline(make_config(tmp_path), manifest, root=tmp_path, evaluate_test=False)

    assert result.outcome is RunOutcome.NOT_RUN
    assert result.metadata is not None
    assert result.metadata.test_evaluated is False
    assert result.artifact_path is not None
    assert result.test_report is not None
    assert str(result.test_report.status) == "not_run"


def test_underpowered_split_is_refused_rather_than_reported(tmp_path: Path) -> None:
    manifest = audio_manifest(tmp_path)
    config = make_config(tmp_path, "mfcc_logreg")
    object.__setattr__(config.train, "min_train_samples", 10_000)

    result = run_baseline(config, manifest, root=tmp_path)

    assert result.outcome is RunOutcome.NOT_RUN
    assert "below the configured floor" in result.reason


def test_one_classed_split_is_refused_at_fit_time(tmp_path: Path) -> None:
    """A single-class split fits a constant and then reports confidently."""
    seeded = make_matrix("train", 20, seed=3)
    single = FeatureMatrix(
        split="train",
        vectors=seeded.vectors,
        labels=np.zeros_like(seeded.labels),
        sample_ids=seeded.sample_ids,
        metadata=seeded.metadata,
        durations=seeded.durations,
    )

    config = make_config(tmp_path, "mfcc_logreg")
    model = build_baseline(config.model)
    with pytest.raises(TrainingError, match="one-class fit"):
        model.fit(single, dev=make_matrix("dev", 20))


def test_runner_refuses_a_one_classed_split_before_fitting() -> None:
    """The runner's own guard, which fires before any model is built."""
    from voxshield.training.runner import _check_both_classes

    seeded = make_matrix("dev", 20, seed=5)
    single = FeatureMatrix(
        split="dev",
        vectors=seeded.vectors,
        labels=np.zeros_like(seeded.labels),
        sample_ids=seeded.sample_ids,
        metadata=seeded.metadata,
        durations=seeded.durations,
    )
    with pytest.raises(TrainingError, match="both classes are required"):
        _check_both_classes(single, role="dev", model_id="mfcc_logreg")


def test_run_record_round_trips_to_json(tmp_path: Path) -> None:
    manifest = audio_manifest(tmp_path)
    result = run_baseline(make_config(tmp_path), manifest, root=tmp_path)
    assert result.config is not None
    assert result.test_report is not None

    path = result.write(tmp_path / "reports")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["outcome"] == "measured"
    assert payload["model_id"] == "mfcc_logreg"
    assert payload["config_hash"] == result.config.config_hash()
    assert payload["test_report"]["status"] == "measured"
    assert payload["test_report"]["metrics"] == result.test_report.to_dict()["metrics"]


def test_rescoring_from_an_artefact_reproduces_the_test_metrics(tmp_path: Path) -> None:
    manifest = audio_manifest(tmp_path)
    result = run_baseline(make_config(tmp_path), manifest, root=tmp_path)
    assert result.artifact_path is not None
    assert result.test_report is not None

    again = run_from_artifact(result.artifact_path, manifest, root=tmp_path, split="test")

    assert again.outcome is RunOutcome.MEASURED
    assert again.test_report is not None
    original = result.test_report.to_dict()["metrics"]
    repeated = again.test_report.to_dict()["metrics"]
    assert repeated["eer"] == pytest.approx(original["eer"])
    assert repeated["roc_auc"] == pytest.approx(original["roc_auc"])


def test_latency_is_not_reported_unless_it_was_measured(tmp_path: Path) -> None:
    manifest = audio_manifest(tmp_path)
    result = run_baseline(make_config(tmp_path), manifest, root=tmp_path, time_inference=False)

    assert result.test_report is not None
    assert str(result.test_report.latency.status) == "not_run"

    timed = run_baseline(
        make_config(tmp_path / "timed"), manifest, root=tmp_path, time_inference=True
    )
    assert timed.test_report is not None
    assert str(timed.test_report.latency.status) == "measured"
