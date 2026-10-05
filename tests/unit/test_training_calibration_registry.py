"""Calibration and the model registry, checked against their failure modes.

Both modules exist to stop something going wrong quietly, which means the tests
that matter are the refusal tests: a calibrator that overfits a small dev split,
an isotonic fit with too few knots to mean anything, a registry that hands back a
model trained with a different front end. A happy-path test for each would pass
whether or not the guards were there.

So the emphasis is inverted from the rest of the suite. The properties asserted
first are the ones where a missing check produces a *plausible wrong number*
rather than an exception -- which is why the EER-invariance and
feature-spec-mismatch tests are load-bearing rather than incidental.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from voxshield.evaluation.metrics import (
    brier_score,
    eer,
    expected_calibration_error,
)
from voxshield.training.artifacts import (
    load_artifact,
    load_scoring_bundle,
    save_artifact,
)
from voxshield.training.calibration import (
    MIN_CALIBRATION_SAMPLES_PER_CLASS,
    MIN_ISOTONIC_SAMPLES,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    build_calibrator,
    calibrator_from_dict,
    fit_calibrator,
)
from voxshield.training.config import TrainingError
from voxshield.training.registry import (
    ModelRegistry,
    ModelRegistryEntry,
    ModelRegistryError,
    load_registry,
)


def miscalibrated_scores(seed: int = 3, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores from a separable model whose probabilities are wrong.

    The scores are squashed toward the middle, which is what an uncalibrated
    model does on a small, imbalanced split: perfectly good at ranking, and
    confidently wrong about how sure it is.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = 1.0 / (1.0 + np.exp(-2.0 * (y - 0.5) * 3.0 - 1.2))
    return y, s


class TestPlattCalibration:
    def test_it_improves_a_miscalibrated_score(self) -> None:
        y, s = miscalibrated_scores()
        calibrated = PlattCalibrator().fit(y, s).transform(s)

        assert brier_score(y, calibrated) < brier_score(y, s)
        assert expected_calibration_error(y, calibrated, 10) < expected_calibration_error(y, s, 10)

    def test_it_cannot_change_the_discrimination_metrics(self) -> None:
        """Monotone recalibration preserves EER. A report claiming otherwise is wrong.

        This is the property that makes calibration safe to add late: it changes
        what a probability *means* without changing what the model *can do*.
        """
        y, s = miscalibrated_scores()
        calibrated = PlattCalibrator().fit(y, s).transform(s)

        assert eer(y, calibrated)[0] == pytest.approx(eer(y, s)[0], abs=1e-12)

    def test_it_preserves_score_ordering(self) -> None:
        _, s = miscalibrated_scores()
        order = np.argsort(s)
        calibrated = PlattCalibrator().fit(*miscalibrated_scores()).transform(s)

        assert list(np.argsort(calibrated)) == list(order)

    def test_it_stays_finite_for_saturated_scores(self) -> None:
        """A raw score of exactly 0 or 1 has no logit. It must not become NaN."""
        y, s = miscalibrated_scores()
        calibrator = PlattCalibrator().fit(y, s)
        out = calibrator.transform(np.array([0.0, 1.0, 1e-18, 1 - 1e-18]))

        assert np.all(np.isfinite(out))
        assert np.all((out >= 0.0) & (out <= 1.0))

    def test_transforming_before_fitting_is_refused(self) -> None:
        with pytest.raises(TrainingError, match="before fit"):
            PlattCalibrator().transform(np.array([0.5]))

    def test_a_single_classed_split_is_refused(self) -> None:
        with pytest.raises(TrainingError, match="both classes"):
            PlattCalibrator().fit(np.zeros(40, dtype=np.int64), np.linspace(0, 1, 40))

    def test_a_split_too_small_to_check_is_refused(self) -> None:
        """Two samples per class cannot support a curve anyone should trust."""
        y = np.array([0, 0, 1, 1], dtype=np.int64)
        with pytest.raises(TrainingError, match="per class"):
            PlattCalibrator().fit(y, np.array([0.1, 0.2, 0.8, 0.9]))

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(TrainingError, match="differ in length"):
            PlattCalibrator().fit(np.array([0, 1, 0, 1] * 3), np.linspace(0, 1, 5))


class TestIsotonicCalibration:
    def test_it_needs_enough_samples_to_not_be_memorisation(self) -> None:
        """The refusal is the feature. Below the floor, isotonic fits a step
        function through the dev points and is confidently wrong in the gaps."""
        y, s = miscalibrated_scores(n=MIN_ISOTONIC_SAMPLES - 1)
        with pytest.raises(TrainingError, match="memorisation"):
            IsotonicCalibrator().fit(y, s)

    def test_it_fits_when_the_dev_split_supports_it(self) -> None:
        y, s = miscalibrated_scores(n=MIN_ISOTONIC_SAMPLES + 50)
        calibrated = IsotonicCalibrator().fit(y, s).transform(s)

        assert expected_calibration_error(y, calibrated, 10) < expected_calibration_error(y, s, 10)

    def test_it_also_preserves_eer(self) -> None:
        y, s = miscalibrated_scores(n=MIN_ISOTONIC_SAMPLES + 50)
        calibrated = IsotonicCalibrator().fit(y, s).transform(s)

        assert eer(y, calibrated)[0] == pytest.approx(eer(y, s)[0], abs=1e-12)


class TestFallback:
    def test_a_calibrator_that_cannot_be_fitted_degrades_rather_than_failing(self) -> None:
        """A trained model is still worth reporting when its dev split is too small
        to calibrate. EER and AUC do not depend on calibration, so the run
        continues -- but it says so, rather than passing off raw scores as
        calibrated ones."""
        y, s = miscalibrated_scores(n=MIN_CALIBRATION_SAMPLES_PER_CLASS * 2)
        calibrator, note = fit_calibrator("platt", y, s)

        assert isinstance(calibrator, IdentityCalibrator)
        assert "fell back to none" in note

    def test_disabling_calibration_is_recorded(self) -> None:
        y, s = miscalibrated_scores()
        calibrator, note = fit_calibrator("none", y, s)

        assert isinstance(calibrator, IdentityCalibrator)
        assert "disabled" in note
        assert calibrator.method == "none"

    def test_an_unknown_method_is_a_configuration_error(self) -> None:
        with pytest.raises(TrainingError, match="calibration method"):
            build_calibrator("magic")


class TestRoundTrip:
    def test_platt_parameters_survive_serialisation(self) -> None:
        y, s = miscalibrated_scores()
        original = PlattCalibrator().fit(y, s)
        restored = calibrator_from_dict(original.to_dict())

        assert np.allclose(restored.transform(s), original.transform(s), atol=1e-12)

    def test_isotonic_parameters_survive_serialisation(self) -> None:
        y, s = miscalibrated_scores(n=MIN_ISOTONIC_SAMPLES + 50)
        original = IsotonicCalibrator().fit(y, s)
        restored = calibrator_from_dict(original.to_dict())

        assert np.allclose(restored.transform(s), original.transform(s), atol=1e-12)

    def test_an_artefact_without_a_calibrator_still_loads(self) -> None:
        """Artifacts predating calibration must not become unreadable.

        Falling back to identity keeps old models usable and, critically, keeps
        them honest: their probabilities are uncalibrated, and a silent
        fabrication would be worse than a known absence.
        """
        assert isinstance(calibrator_from_dict(None), IdentityCalibrator)
        assert isinstance(calibrator_from_dict({}), IdentityCalibrator)

    def test_a_corrupt_calibrator_is_refused_not_ignored(self) -> None:
        with pytest.raises(TrainingError, match="unusable"):
            calibrator_from_dict({"method": "platt", "coefficient": "not-a-number"})


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def write_artifact(
    root: Path,
    model_id: str,
    *,
    family: str = "mfcc_logreg",
    n_mels: int = 26,
    calibration: dict[str, Any] | None = None,
    threshold: float | None = 0.42,
    test_evaluated: bool = True,
) -> Path:
    """Write a small valid artefact so registry behaviour can be tested directly."""
    from tests.unit.test_training_artifacts_runner import fit_model, make_config, make_matrix

    config = make_config(root, family=family)
    config = replace(
        config,
        model_id=model_id,
        features=replace(config.features, n_mels=n_mels),
    )
    train = make_matrix("train", 60, seed=1)
    dev = make_matrix("dev", 60, seed=2)
    model, summary = fit_model(config, train, dev)

    from voxshield.training.artifacts import build_metadata

    metadata = replace(
        build_metadata(
            config,
            model,
            summary,
            n_features=train.vectors.shape[1],
            class_counts=train.counts,
            rows_considered=len(train),
        ),
        model_id=model_id,
        calibration=calibration,
        calibration_method="platt" if calibration else "none",
        threshold=threshold,
        threshold_policy="eer" if threshold is not None else None,
        test_evaluated=test_evaluated,
    )
    return save_artifact(model, metadata, root)


class TestRegistryDiscovery:
    def test_it_finds_every_artifact_under_a_root(self, tmp_path: Path) -> None:
        write_artifact(tmp_path, "mfcc_logreg")
        write_artifact(tmp_path, "mfcc_xgboost", family="mfcc_xgboost")
        registry = load_registry(tmp_path)

        assert len(registry) == 2
        assert {entry.model_id for entry in registry} == {"mfcc_logreg", "mfcc_xgboost"}
        assert registry.families() == ("mfcc_logreg", "mfcc_xgboost")

    def test_an_empty_root_is_measured_not_an_error(self, tmp_path: Path) -> None:
        """A machine that has trained nothing is a normal machine.

        Raising here would make the registry unusable on every checkout but one,
        which is precisely the mistake the dataset registry already had to be
        written around.
        """
        registry = load_registry(tmp_path)

        assert len(registry) == 0
        assert registry.report()["count"] == 0

    def test_a_missing_root_is_empty_rather_than_an_error(self, tmp_path: Path) -> None:
        assert len(load_registry(tmp_path / "nowhere")) == 0

    def test_a_stray_directory_is_not_reported_as_a_broken_model(self, tmp_path: Path) -> None:
        """Noise in the output directory must not look like a corrupt artefact."""
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "run.txt").write_text("noise", encoding="utf-8")

        registry = load_registry(tmp_path)

        assert len(registry) == 0
        assert registry.skipped() == ()

    def test_a_corrupt_artifact_is_skipped_with_a_reason_not_raised(self, tmp_path: Path) -> None:
        write_artifact(tmp_path, "mfcc_logreg_b")
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / "metadata.json").write_text("{not json", encoding="utf-8")

        registry = load_registry(tmp_path)

        assert {entry.model_id for entry in registry} == {"mfcc_logreg_b"}
        assert [reason.model_id for reason in registry.skipped()] == ["broken"]
        assert registry.skipped()[0].reason == "unreadable"


class TestRegistryResolution:
    def test_an_unknown_identifier_names_the_alternatives(self, tmp_path: Path) -> None:
        """Raising beats defaulting. Silently substituting a different model is
        the failure this registry exists to prevent, so the error must be loud
        and must say what does exist."""
        write_artifact(tmp_path, "mfcc_logreg")
        registry = load_registry(tmp_path)

        with pytest.raises(ModelRegistryError, match="mfcc_logreg"):
            registry.get("something_else")

    def test_it_resolves_a_known_identifier(self, tmp_path: Path) -> None:
        write_artifact(tmp_path, "mfcc_logreg")
        entry = load_registry(tmp_path).get("mfcc_logreg")

        assert entry.family == "mfcc_logreg"
        assert entry.threshold == pytest.approx(0.42)

    def test_two_directories_claiming_one_identifier_are_refused(self, tmp_path: Path) -> None:
        """Ambiguity must not be resolved by directory order."""
        write_artifact(tmp_path, "mfcc_logreg")
        other = tmp_path / "elsewhere"
        other.mkdir()
        write_artifact(other, "mfcc_logreg")

        registry = ModelRegistry.discover(tmp_path)
        entry = ModelRegistryEntry(model_id="mfcc_logreg", family="mfcc_logreg", directory=other)

        with pytest.raises(ModelRegistryError, match="duplicate"):
            registry.add(entry)


class TestRegistryCompatibility:
    def test_a_front_end_mismatch_is_refused(self, tmp_path: Path) -> None:
        """The failure this check exists for.

        Loading a model trained with 26 mel bands and featurising with 40
        produces scores that look like probabilities and mean nothing. Nothing
        downstream would ever raise, so it has to be refused here.
        """
        write_artifact(tmp_path, "mfcc_logreg", n_mels=26)
        registry = load_registry(tmp_path)

        with pytest.raises(ModelRegistryError, match="different front end"):
            registry.compatible_with("mfcc_logreg", {"n_mels": 40})

    def test_the_error_names_the_differing_settings(self, tmp_path: Path) -> None:
        write_artifact(tmp_path, "mfcc_logreg", n_mels=26)
        registry = load_registry(tmp_path)

        with pytest.raises(ModelRegistryError, match="n_mels: trained 26 vs requested 40"):
            registry.compatible_with("mfcc_logreg", {"n_mels": 40})

    def test_a_matching_front_end_is_accepted(self, tmp_path: Path) -> None:
        write_artifact(tmp_path, "mfcc_logreg", n_mels=26)
        registry = load_registry(tmp_path)

        entry = registry.compatible_with("mfcc_logreg", registry.get("mfcc_logreg").feature_spec)

        assert entry.model_id == "mfcc_logreg"

    def test_no_requested_spec_means_no_check(self, tmp_path: Path) -> None:
        write_artifact(tmp_path, "mfcc_logreg")
        registry = load_registry(tmp_path)

        assert registry.compatible_with("mfcc_logreg", None).model_id == "mfcc_logreg"


class TestRegistryScoring:
    def test_a_registered_model_scores_with_its_calibrator(self, tmp_path: Path) -> None:
        """Loading must not silently drop the calibrator.

        A model and its calibrator are only correct together: without the
        calibrator the probabilities are the wrong ones, and they still look
        entirely plausible.
        """
        write_artifact(
            tmp_path,
            "mfcc_logreg",
            calibration=PlattCalibrator().fit(*miscalibrated_scores()).to_dict(),
        )
        registry = load_registry(tmp_path)
        bundle = registry.load("mfcc_logreg")

        assert bundle.calibrator.method == "platt"
        assert bundle.threshold == pytest.approx(0.42)

        from tests.unit.test_training_artifacts_runner import make_matrix

        matrix = make_matrix("test", 10, seed=9)
        raw = bundle.model.predict_proba(matrix)
        assert not np.allclose(bundle.score(matrix), raw)
        assert bundle.decision(matrix).dtype == bool

    def test_decisions_without_a_recorded_threshold_are_all_negative(self, tmp_path: Path) -> None:
        """No threshold means no decision. Guessing one would be inventing a
        claim the artefact cannot support."""
        write_artifact(tmp_path, "mfcc_logreg_b", threshold=None)
        bundle = load_registry(tmp_path).load("mfcc_logreg_b")

        from tests.unit.test_training_artifacts_runner import make_matrix

        matrix = make_matrix("test", 8, seed=4)
        assert not bundle.decision(matrix).any()
        assert bundle.score(matrix).shape == (8,)

    def test_the_bundle_is_reachable_from_a_path_alone(self, tmp_path: Path) -> None:
        directory = write_artifact(tmp_path, "mfcc_logreg")
        bundle = load_scoring_bundle(directory)

        assert bundle.metadata.model_id == "mfcc_logreg"
        assert bundle.threshold == pytest.approx(0.42)

    def test_a_model_that_cannot_be_loaded_is_reported_as_a_registry_error(
        self, tmp_path: Path
    ) -> None:
        directory = write_artifact(tmp_path, "mfcc_logreg")
        registry = load_registry(tmp_path)
        (directory / "model.joblib").unlink()

        with pytest.raises(ModelRegistryError, match="could not be loaded"):
            registry.load("mfcc_logreg")


class TestRegistryRegistration:
    def test_an_artifact_from_outside_the_root_can_be_registered(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "shared"
        elsewhere.mkdir()
        write_artifact(elsewhere, "mfcc_logreg")
        empty_root = tmp_path / "empty"
        empty_root.mkdir()

        registry = load_registry(empty_root)
        entry = registry.register(elsewhere / "mfcc_logreg")

        assert entry.model_id == "mfcc_logreg"
        assert len(registry) == 1

    def test_registering_a_duplicate_needs_an_explicit_override(self, tmp_path: Path) -> None:
        elsewhere = tmp_path / "shared"
        elsewhere.mkdir()
        write_artifact(elsewhere, "mfcc_logreg")
        write_artifact(tmp_path, "mfcc_logreg")
        registry = load_registry(tmp_path)

        with pytest.raises(ModelRegistryError, match="already registered"):
            registry.register(elsewhere / "mfcc_logreg")

        assert registry.register(elsewhere / "mfcc_logreg", replace=True).model_id == "mfcc_logreg"

    def test_registering_something_that_is_not_an_artifact_is_refused(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(ModelRegistryError, match="cannot register"):
            load_registry(empty).register(empty)

    def test_the_report_is_json_ready(self, tmp_path: Path) -> None:
        import json

        write_artifact(tmp_path, "mfcc_logreg")
        broken = tmp_path / "broken"
        broken.mkdir()
        (broken / "metadata.json").write_text("{not json", encoding="utf-8")

        report = load_registry(tmp_path).report()

        assert json.loads(json.dumps(report))["count"] == 1
        assert report["skipped"][0]["model_id"] == "broken"


class TestFingerprintLookup:
    def test_a_fingerprint_finds_the_model_that_produced_it(self, tmp_path: Path) -> None:
        from tests.unit.test_training_artifacts_runner import N_BANDS, make_config

        config = make_config(tmp_path)
        write_artifact(tmp_path, config.model_id, n_mels=N_BANDS)

        found = load_registry(tmp_path).find_by_fingerprint(config.content_fingerprint())

        assert found is not None
        assert found.model_id == config.model_id

    def test_an_unrecorded_fingerprint_never_matches(self, tmp_path: Path) -> None:
        """A legacy artefact must not be treated as current just because its
        fingerprint happens to be empty."""
        entry = ModelRegistryEntry(model_id="x", family="mfcc_logreg", directory=tmp_path)

        assert not entry.matches_fingerprint("")
        assert not entry.matches_fingerprint("anything")
        assert load_registry(tmp_path).find_by_fingerprint("nothing") is None


class TestArtifactCarriesTheOperatingPoint:
    def test_metadata_survives_a_round_trip(self, tmp_path: Path) -> None:
        directory = write_artifact(
            tmp_path,
            "mfcc_logreg",
            calibration=PlattCalibrator().fit(*miscalibrated_scores()).to_dict(),
            threshold=0.37,
        )
        _model, metadata = load_artifact(directory)

        assert metadata.threshold == pytest.approx(0.37)
        assert metadata.threshold_policy == "eer"
        assert metadata.calibration_method == "platt"
        assert metadata.calibration is not None
