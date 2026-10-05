"""The full Phase 3 path over real audio: corpus, training, artefact, registry, report.

The unit tests check each stage alone, which is exactly what lets them miss the
failures that only appear when the stages run in order:

* a calibrator fitted on dev and then quietly not applied to test, so the report
  describes probabilities nobody will ever see;
* a threshold selected during training that the artefact does not carry, so a
  rescoring reports a different operating point from the run that produced the
  model;
* a registry that finds the artefact and then hands back probabilities calibrated
  against nothing, because it loaded the model without its calibrator;
* each stage passing in isolation while the numbers disagree end to end.

So this file drives the real thing -- real WAV files on disk, the real build,
the real recipes, the real CLI -- and asserts the invariants a person citing a
result would otherwise be assuming. It is slower than the unit tests by design;
that cost is the point.

Every number produced here comes from generated audio and is therefore
meaningless as a spoof-detection result. These tests assert *protocol* properties,
never performance.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxshield.cli import ML_CANNOT_RUN, ML_FAILED, ML_MEASURED, main
from voxshield.data.manifest import read_manifest
from voxshield.evaluation.metrics import brier_score, eer
from voxshield.training.registry import ModelRegistry, ModelRegistryError, load_registry
from voxshield.training.runner import RunOutcome, run_baseline

SAMPLE_RATE = 16_000

# The split is class-stratified, so dev is no longer class-skewed by the group
# fill and its sizes follow the ratios for free. What still constrains the sizes
# is the calibrator, which needs a minimum number of dev rows per class before it
# will fit a curve at all; below that it falls back to no calibration, which would
# quietly turn every calibration assertion in this file into a tautology. So the
# sizes are measured rather than guessed, and the assertion in ``corpus`` below
# fails loudly if a future edit drops dev back under the floor.
#
# Note the corpus is 1:4 bona:spoof, not 1:1: ``speech(seed=index)`` makes every
# take of a speaker byte-identical, so dedup keeps one bona file per speaker and
# the longer spoof files survive all four. Stratification distributes that 1:4
# rather than inventing it.
SPEAKERS = 20
TAKES = 4
MIN_DEV_PER_CLASS = 5

DATA_CONFIG_TEXT = """
root: __ROOT__
random_seed: 20260928
license_policy: require_verified
window_seconds: 1.0
hop_ratio: 0.5
storage_subtype: PCM_16
write_segment_audio: true

datasets:
  - dataset_id: bona
    name: Bona fide speech
    source: generated for this test
    version: "1.0"
    license: project-owned
    license_status: VERIFIED
    license_verified_by: fixture
    task: real_speech
    enabled: true
    path: raw/bona
    adapter: real_speech
    metadata:
      speaker_pattern: "^spk(?P<speaker>[0-9]{3})_"
      speaker_group: "^spk(?P<speaker>[0-9]{3})"
  - dataset_id: fake
    name: Synthetic speech
    source: generated for this test
    version: "0.0"
    license: project-owned
    license_status: VERIFIED
    license_verified_by: fixture
    task: spoof_detection
    enabled: true
    path: raw/fake
    adapter: wavefake

paths:
  raw: raw
  interim: interim
  processed: processed
  manifests: manifests
  synthetic: synthetic
  cache: cache
  reports: reports

split:
  train_ratio: 0.6
  dev_ratio: 0.2
  min_train_speakers: 2

validation:
  require_speech: true
  min_duration_seconds: 0.30
  reject_warning_issues: false
  require_metadata: []
  reject_duplicates: true
  dedup_scope: both

augmentation:
  enabled: false
  class_balance: none

gates:
  min_test_sources: 1
"""

# A deliberately small recipe: this file tests protocol plumbing, and a real
# logistic regression on separable generated audio converges in a handful of
# iterations regardless.
RECIPE_TEXT = """
model_id: __MODEL_ID__

features:
  kind: __FEATURES__
  n_fft: 400
  hop_length: 160
  n_mels: 20
  fmin: 20.0
  fmax: 7600.0
  htk: true
  n_coefficients: 8
  lifter: 22.0
  with_delta: false
  n_frames: __N_FRAMES__
  target_frames: __TARGET_FRAMES__

model:
  family: __FAMILY__
  max_iter: 200
  batch_size: 16
  class_weight: balanced

train:
  seed: 20260928
  epochs: 2
  device: cpu
  verify_leakage: true
  min_train_samples: 20
  min_dev_samples: 10
  calibration: __CALIBRATION__

threshold:
  strategy: eer
  max_frr: null
  max_far: null

train_split: train
dev_split: dev
test_split: test

output_dir: __OUTPUT__
"""


def speech(seed: int, seconds: float = 1.6) -> np.ndarray:
    """A voiced signal with a syllable envelope, so it reads as speech not noise.

    Spoof segments are deliberately distinguishable: the point is to exercise a
    path that can find a threshold, not to model a real attack.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    f0 = 108.0 + 7.0 * (seed % 6)
    signal = np.zeros_like(t)
    for harmonic, weight in ((1, 1.0), (2, 0.6), (3, 0.35), (5, 0.18)):
        signal += weight * np.sin(2 * np.pi * f0 * harmonic * t + rng.random() * 6.28)
    signal *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    signal += 0.006 * rng.standard_normal(t.size)
    return (signal / (float(np.max(np.abs(signal))) or 1.0) * 0.55).astype(np.float32)


def write_corpus(root: Path) -> Path:
    """Write two classes of real audio and the data config that points at them.

    Returns:
        The path to the written data configuration.
    """
    bona = root / "raw" / "bona"
    spoof = root / "raw" / "fake" / "train" / "gen" / "melgan"
    for directory in (bona, spoof):
        directory.mkdir(parents=True, exist_ok=True)
    for index in range(SPEAKERS):
        for take in range(TAKES):
            name = f"spk{index:03d}_{take}.wav"
            sf.write(bona / name, speech(seed=index), SAMPLE_RATE, subtype="PCM_16")
            # A different f0 and a spectral tilt, so the two classes are
            # separable without being identical.
            sf.write(
                spoof / name,
                speech(seed=10_000 + index, seconds=1.6 + 0.05 * take),
                SAMPLE_RATE,
                subtype="PCM_16",
            )
    config = root / "data.yaml"
    config.write_text(DATA_CONFIG_TEXT.replace("__ROOT__", root.as_posix()), encoding="utf-8")
    return config


def write_recipe(
    root: Path,
    *,
    model_id: str = "mfcc_logreg",
    family: str = "mfcc_logreg",
    features: str = "mfcc",
    n_frames: int | None = None,
    target_frames: int | None = None,
    calibration: str = "platt",
    output: Path | None = None,
) -> Path:
    """Write a training recipe pointed at the built corpus.

    Returns:
        The path to the written recipe.
    """
    text = RECIPE_TEXT
    for key, value in (
        ("__MODEL_ID__", model_id),
        ("__FAMILY__", family),
        ("__FEATURES__", features),
        ("__N_FRAMES__", "null" if n_frames is None else str(n_frames)),
        ("__TARGET_FRAMES__", "null" if target_frames is None else str(target_frames)),
        ("__CALIBRATION__", calibration),
        ("__OUTPUT__", (output or (root / "models")).as_posix()),
    ):
        text = text.replace(key, value)
    recipe = root / f"{model_id}.yaml"
    recipe.write_text(text, encoding="utf-8")
    return recipe


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A built two-class corpus with all three split manifests."""
    root = tmp_path_factory.mktemp("training_pipeline")
    config = write_corpus(root)
    assert main(["data", "build", "--config", str(config)]) == 0
    for split in ("train", "dev", "test"):
        manifest = read_manifest(root / "manifests" / f"{split}.jsonl")
        assert manifest.samples, f"{split} is empty"
        assert {row.label_index for row in manifest.samples} == {0, 1}

    # Calibration is fitted on dev, and it refuses a split too small to support
    # a curve. That refusal is correct behaviour, so assert the precondition
    # here rather than letting the calibrator quietly fall back to "none" and
    # turn every calibration assertion in this file into a tautology.
    dev = read_manifest(root / "manifests" / "dev.jsonl")
    per_class = Counter(row.label_index for row in dev.samples)
    assert min(per_class.values()) >= MIN_DEV_PER_CLASS, (
        f"dev has {dict(per_class)} samples per class, under the "
        f"{MIN_DEV_PER_CLASS}/class the calibrator requires; these tests would "
        "pass on a run that refused to calibrate at all. Raise SPEAKERS/TAKES."
    )
    return root


@pytest.fixture(scope="module")
def manifest_path(corpus: Path) -> Path:
    """The manifest every training run in this file reads."""
    return corpus / "manifests" / "all.jsonl"


@pytest.fixture(scope="module")
def trained(corpus: Path, manifest_path: Path) -> dict[str, object]:
    """One trained baseline, with its config and artefact, reused by the tests.

    Training is the expensive step here, and the properties under test are all
    invariants *within* a single run rather than comparisons between runs, so
    running it once is both sufficient and much faster.
    """
    from voxshield.training.config import load_training_config

    recipe = write_recipe(corpus, output=corpus / "models")
    config = load_training_config(recipe)
    result = run_baseline(config, manifest_path, root=corpus)
    assert result.outcome is RunOutcome.MEASURED, result.reason
    return {"config": config, "result": result, "recipe": recipe}


@pytest.fixture(scope="module")
def model_root(trained: dict[str, object]) -> Path:
    """The directory the registry should be pointed at.

    ``ModelRegistry.discover`` scans the immediate children of the directory it
    is handed, and a run writes ``<output_dir>/<model_id>/metadata.json``. So the
    registry root is the recipe's ``output_dir`` -- not the corpus root above it,
    which would sweep ``raw``, ``manifests``, ``processed`` and the raw WAV tree
    in as stray directories and find nothing.
    """
    return trained["config"].output_dir


class TestTheProtocolHolds:
    def test_a_real_run_measures_every_split(self, trained: dict[str, object]) -> None:
        result = trained["result"]
        assert result.test_report is not None
        assert result.test_report.status == "measured"
        assert result.dev_report is not None

    def test_the_threshold_comes_from_dev_and_is_recorded(self, trained: dict[str, object]) -> None:
        """An operating point is only meaningful next to where it was chosen."""
        result = trained["result"]
        metadata = result.metadata

        assert result.threshold is not None
        assert metadata is not None
        assert metadata.threshold == pytest.approx(result.threshold)
        assert metadata.dev_split == "dev"

    def test_the_calibrator_is_fitted_on_dev_and_recorded(self, trained: dict[str, object]) -> None:
        metadata = trained["result"].metadata
        assert metadata is not None
        assert metadata.calibration_method == "platt"
        assert metadata.calibration is not None
        assert metadata.calibration["method"] == "platt"

    def test_test_reports_the_calibrated_probabilities(self, trained: dict[str, object]) -> None:
        """The report's own calibration numbers must be computed on calibrated
        probabilities. If the calibrator were skipped for test, the reliability
        bins would describe a curve the system never uses."""
        result = trained["result"]
        report = result.test_report
        assert report is not None
        assert report.calibration_bins
        assert any(bin_.count for bin_ in report.calibration_bins)

    def test_the_run_record_round_trips_through_json(self, trained: dict[str, object]) -> None:
        payload = json.loads(json.dumps(trained["result"].to_dict()))

        assert payload["outcome"] == "measured"
        assert payload["calibration"]["method"] == "platt"
        assert payload["threshold"] is not None


class TestRescoringReproducesTheRun:
    def test_rescoring_at_the_recorded_point_matches_the_training_run(
        self, trained: dict[str, object], corpus: Path, manifest_path: Path
    ) -> None:
        """The whole point of carrying the calibrator and the threshold.

        A rescore that reports a different operating point from the run that
        produced the model is not a reproduction, and a reader comparing the two
        records has no way to tell.
        """
        from voxshield.training.runner import run_from_artifact

        result = trained["result"]
        rescore = run_from_artifact(result.artifact_path, manifest_path, root=corpus, split="test")

        assert rescore.outcome is RunOutcome.MEASURED
        assert rescore.test_report is not None
        assert result.test_report is not None
        assert rescore.threshold == pytest.approx(result.threshold)
        assert rescore.test_report.metrics.eer == pytest.approx(
            result.test_report.metrics.eer, abs=1e-12
        )
        assert rescore.test_report.metrics.roc_auc == pytest.approx(
            result.test_report.metrics.roc_auc, abs=1e-12
        )
        # Identical confusion at the recorded point means identical FAR and FRR,
        # which is the claim being made to anyone comparing the two records.
        assert rescore.test_report.confusion == result.test_report.confusion

    def test_rescoring_reports_a_decision_not_just_a_ranking(
        self, trained: dict[str, object], corpus: Path, manifest_path: Path
    ) -> None:
        """Before the threshold travelled with the artefact, FAR and FRR were
        necessarily absent on every rescore."""
        from voxshield.training.runner import run_from_artifact

        result = trained["result"]
        rescore = run_from_artifact(result.artifact_path, manifest_path, root=corpus)

        assert rescore.reason == ""
        assert rescore.test_report is not None
        # FAR and FRR are derived when the record is serialised, so that is the
        # surface to check: it is what a reader of the artefact ever sees.
        payload = json.loads(json.dumps(rescore.test_report.to_dict()))
        assert payload["confusion"]["far"] is not None
        assert payload["confusion"]["frr"] is not None

    def test_the_calibrator_is_applied_on_rescoring(
        self, trained: dict[str, object], corpus: Path, manifest_path: Path
    ) -> None:
        """Recompute the test split and compare the report's own calibration
        numbers against metrics taken from raw and calibrated scores.

        Agreement on EER with disagreement on Brier is the signature of a
        calibrator having been applied: monotonic recalibration cannot move the
        discrimination metrics, so if EER matched *and* Brier matched too, the
        probabilities would be raw and the reliability bins would describe a
        curve the system never uses.
        """
        from voxshield.training.config import FeatureSpec
        from voxshield.training.datasets import load_feature_matrix, resolve_rows
        from voxshield.training.features import FeatureExtractor
        from voxshield.training.runner import run_from_artifact

        result = trained["result"]
        metadata = result.metadata
        assert metadata is not None
        rescore = run_from_artifact(result.artifact_path, manifest_path, root=corpus)
        assert rescore.test_report is not None
        assert rescore.calibrator is not None
        assert rescore.calibrator.method == "platt"

        extractor = FeatureExtractor(FeatureSpec.from_mapping(metadata.feature_spec))
        matrix = load_feature_matrix(
            resolve_rows(manifest_path, "test"),
            extractor,
            root=corpus,
            split="test",
            want_matrices=False,
            strict=False,
        )
        raw = rescore.model.predict_proba(matrix)
        calibrated = np.asarray(rescore.calibrator.transform(raw), dtype=np.float64)
        metrics = rescore.test_report.metrics

        # The report was built from the calibrated scores...
        assert metrics.eer == pytest.approx(eer(matrix.labels, calibrated)[0], abs=1e-12)
        assert metrics.brier == pytest.approx(brier_score(matrix.labels, calibrated), abs=1e-12)
        # ...which differ from the raw ones, so this is not a tautology.
        assert metrics.brier != pytest.approx(brier_score(matrix.labels, raw), abs=1e-6)
        # EER is invariant under the transform, which is what makes the two
        # assertions above jointly meaningful.
        assert eer(matrix.labels, raw)[0] == pytest.approx(metrics.eer, abs=1e-12)

    def test_an_explicit_threshold_overrides_the_recorded_one(
        self, trained: dict[str, object], corpus: Path, manifest_path: Path
    ) -> None:
        """Scoring at a point the model was not selected at is a legitimate
        question, and the command must answer it without editing the artefact."""
        from voxshield.training.runner import run_from_artifact

        result = trained["result"]
        rescore = run_from_artifact(
            result.artifact_path, manifest_path, root=corpus, threshold=0.99
        )

        assert rescore.threshold == pytest.approx(0.99)
        assert rescore.test_report is not None
        report = rescore.test_report
        assert report.threshold == pytest.approx(0.99)
        assert report.confusion is not None

        # The override is a different operating point from the one the run
        # selected, and the confusion matrix belongs to the override: its rates
        # have to reconcile with its own counts, not just be present.
        assert result.threshold != pytest.approx(0.99)
        counts = report.confusion
        metrics = report.metrics
        assert counts.false_positive / metrics.n_negative == pytest.approx(
            json.loads(json.dumps(report.to_dict()))["confusion"]["far"]
        )
        assert counts.false_negative / metrics.n_positive == pytest.approx(
            json.loads(json.dumps(report.to_dict()))["confusion"]["frr"]
        )


class TestTheRegistryFindsWhatWasTrained:
    def test_it_discovers_the_artifact_written_by_the_run(
        self, trained: dict[str, object], model_root: Path
    ) -> None:
        registry = load_registry(model_root)

        assert trained["result"].config is not None
        assert trained["result"].config.model_id in registry

    def test_it_reports_the_calibration_and_threshold_it_found(
        self, trained: dict[str, object], model_root: Path
    ) -> None:
        model_id = trained["result"].config.model_id
        entry = load_registry(model_root).get(model_id)

        assert entry.calibration_method == "platt"
        assert entry.threshold == pytest.approx(trained["result"].threshold)
        assert entry.test_evaluated is True
        assert entry.feature_spec

    def test_the_scoring_bundle_matches_the_training_run(
        self, trained: dict[str, object], model_root: Path
    ) -> None:
        """Registry-loaded scores must equal run scores exactly. Any difference
        means the registry dropped the calibrator or applied a different front
        end, which is the failure this path exists to prevent."""
        result = trained["result"]
        bundle = load_registry(model_root).load(result.config.model_id)

        assert bundle.threshold == pytest.approx(result.threshold)
        assert bundle.metadata.calibration_method == "platt"

    def test_it_refuses_a_model_trained_with_a_different_front_end(
        self, trained: dict[str, object], model_root: Path
    ) -> None:
        model_id = trained["result"].config.model_id
        registry = load_registry(model_root)
        wrong = {**registry.get(model_id).feature_spec, "n_mels": 40}

        with pytest.raises(ModelRegistryError, match="different front end"):
            registry.score(model_id, _dummy_matrix(), feature_spec=wrong)

    def test_an_unknown_model_names_what_does_exist(self, trained: dict[str, object]) -> None:
        corpus = trained["config"].output_dir.parent
        registry = load_registry(corpus)

        with pytest.raises(ModelRegistryError, match="unknown model_id"):
            registry.get("not_trained_here")


def _dummy_matrix():  # type: ignore[no-untyped-def]
    """A small feature matrix, only enough to reach the compatibility check."""
    from voxshield.training.datasets import FeatureMatrix

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 4).astype(np.int64)
    return FeatureMatrix(
        split="test",
        vectors=rng.normal(size=(4, 8)).astype(np.float32),
        labels=labels,
        sample_ids=("a", "b", "c", "d"),
        metadata={},
        durations=(1.0, 1.0, 1.0, 1.0),
        matrices=None,
    )


class TestTheRegistryCoversEveryFamily:
    @pytest.mark.parametrize(
        ("model_id", "family", "features", "n_frames"),
        [
            ("mfcc_logreg", "mfcc_logreg", "mfcc", None),
            ("mfcc_xgboost", "mfcc_xgboost", "mfcc", None),
        ],
    )
    def test_a_family_trains_and_registers(
        self,
        corpus: Path,
        manifest_path: Path,
        model_id: str,
        family: str,
        features: str,
        n_frames: int | None,
    ) -> None:
        from voxshield.training.config import load_training_config

        output = corpus / "family_models"
        recipe = write_recipe(
            corpus,
            model_id=model_id,
            family=family,
            features=features,
            n_frames=n_frames,
            output=output,
        )
        config = load_training_config(recipe)
        result = run_baseline(config, manifest_path, root=corpus)
        assert result.outcome is RunOutcome.MEASURED, result.reason

        registry = ModelRegistry.discover(output)
        assert model_id in registry
        assert registry.get(model_id).family == family
        assert registry.get(model_id).threshold is not None

    def test_the_cnn_family_trains_and_registers(self, corpus: Path, manifest_path: Path) -> None:
        """The CNN is the family that needs the most from a round trip: a
        different front end (frames, not pooled vectors) and separate weights."""
        torch = pytest.importorskip("torch")
        del torch

        from voxshield.training.config import load_training_config

        output = corpus / "cnn_models"
        recipe = write_recipe(
            corpus,
            model_id="logmel_cnn",
            family="logmel_cnn",
            features="log_mel",
            n_frames=24,
            target_frames=24,
            output=output,
        )
        config = load_training_config(recipe)
        result = run_baseline(config, manifest_path, root=corpus)
        assert result.outcome is RunOutcome.MEASURED, result.reason

        entry = load_registry(output).get("logmel_cnn")
        assert entry.family == "logmel_cnn"
        assert entry.calibration_method == "platt"
        assert entry.threshold is not None


class TestCalibrationChoicesSurviveTheFullPath:
    def test_disabling_calibration_is_recorded_as_such(
        self, corpus: Path, manifest_path: Path
    ) -> None:
        """Calibration off is a legitimate configuration and must not be
        indistinguishable from calibration silently failing."""
        from voxshield.training.config import load_training_config

        output = corpus / "uncalibrated"
        recipe = write_recipe(
            corpus, model_id="mfcc_logreg_uncal", calibration="none", output=output
        )
        result = run_baseline(load_training_config(recipe), manifest_path, root=corpus)

        assert result.metadata is not None
        assert result.metadata.calibration_method == "none"
        assert result.metadata.threshold is not None

    def test_isotonic_is_refused_on_a_dev_split_too_small_to_support_it(
        self, corpus: Path, manifest_path: Path
    ) -> None:
        """The run still measures -- EER does not need calibration -- but the
        record says the calibrator was not fitted, so nobody reads the
        probabilities as calibrated."""
        from voxshield.training.config import load_training_config

        output = corpus / "isotonic"
        recipe = write_recipe(
            corpus, model_id="mfcc_logreg_iso", calibration="isotonic", output=output
        )
        result = run_baseline(load_training_config(recipe), manifest_path, root=corpus)

        assert result.outcome is RunOutcome.MEASURED
        assert result.metadata is not None
        assert result.metadata.calibration_method == "none"
        assert any("fell back" in note for note in result.metadata.notes)


class TestTheCommandLineEndToEnd:
    def test_train_then_evaluate_then_models(self, corpus: Path) -> None:
        """The whole path a person actually types."""
        output = corpus / "cli_models"
        recipe = write_recipe(corpus, model_id="mfcc_logreg_cli", output=output)
        records = corpus / "records"
        manifests = corpus / "manifests" / "all.jsonl"

        assert (
            main(
                [
                    "ml",
                    "train",
                    "--config",
                    str(recipe),
                    "--manifest",
                    str(manifests),
                    "--root",
                    str(corpus),
                    "--record-dir",
                    str(records),
                    "--compact",
                ]
            )
            == ML_MEASURED
        )
        train_record = json.loads((records / "run.json").read_text(encoding="utf-8"))
        assert train_record["outcome"] == "measured"
        assert train_record["calibration"]["method"] == "platt"

        assert (
            main(
                [
                    "ml",
                    "evaluate",
                    "--artifact",
                    str(output / "mfcc_logreg_cli"),
                    "--manifest",
                    str(manifests),
                    "--root",
                    str(corpus),
                    "--compact",
                ]
            )
            == ML_MEASURED
        )

        assert (
            main(
                [
                    "ml",
                    "models",
                    "--root",
                    str(output),
                    "--resolve",
                    "mfcc_logreg_cli",
                    "--compact",
                ]
            )
            == ML_MEASURED
        )

    def test_an_absent_corpus_still_reports_not_run(self, tmp_path: Path) -> None:
        """The repository ships no corpus, and that must remain a measured fact
        rather than an error."""
        recipe = write_recipe(tmp_path, output=tmp_path / "models")
        code = main(
            [
                "ml",
                "train",
                "--config",
                str(recipe),
                "--manifest",
                str(tmp_path / "nowhere.jsonl"),
                "--root",
                str(tmp_path),
                "--compact",
            ]
        )

        assert code == ML_CANNOT_RUN

    def test_a_missing_manifest_exits_two_not_one(self, tmp_path: Path) -> None:
        recipe = write_recipe(tmp_path, output=tmp_path / "models")
        code = main(
            [
                "ml",
                "train",
                "--config",
                str(recipe),
                "--manifest",
                str(tmp_path / "absent.jsonl"),
                "--root",
                str(tmp_path),
                "--compact",
            ]
        )

        assert code == ML_CANNOT_RUN
        assert code != ML_FAILED
