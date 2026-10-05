"""Unit tests for the three Phase 3 baselines.

The tests here are deliberately about the failure modes that produce
*plausible-looking wrong numbers* rather than about whether a model can fit at
all:

* scoring with an unfitted model, or with a fitted estimator and an absent
  scaler, which silently changes what the features mean;
* a one-class training split, which fits a model that always predicts that
  class and reports a confident EER;
* a swapped probability column, which inverts every metric in the report while
  still producing numbers;
* an early-stopping path that reports selection it did not perform;
* train/test contact of any kind.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxshield.training.config import ModelSpec, TrainingError
from voxshield.training.datasets import FeatureMatrix
from voxshield.training.models import (
    BaselineModel,
    FitSummary,
    LogisticBaseline,
    LogMelCNN,
    XGBoostBaseline,
    build_baseline,
)

N_FRAMES = 24
N_BANDS = 12
N_FEATURES = 10


def make_matrix(
    split: str,
    n: int,
    *,
    seed: int = 0,
    labels: np.ndarray | None = None,
    matrices: bool = True,
    shift: float = 1.5,
) -> FeatureMatrix:
    """Build a separable synthetic feature matrix.

    Args:
        split: Split name.
        n: Row count.
        seed: RNG seed, so a test can vary the data while holding the shape.
        labels: Explicit labels, for one-class cases.
        matrices: Whether to attach frame matrices.
        shift: How far to separate the spoof class from bona fide.

    Returns:
        A synthetic :class:`FeatureMatrix`.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n) if labels is None else np.asarray(labels)
    vectors = rng.normal(size=(n, N_FEATURES))
    vectors[y == 1] += shift
    mats = None
    if matrices:
        mats = rng.normal(size=(n, N_FRAMES, N_BANDS)).astype(np.float32)
        mats[y == 1] += shift
    return FeatureMatrix(
        split=split,
        vectors=vectors,
        labels=y.astype(np.int64),
        sample_ids=tuple(f"{split}-{index}" for index in range(n)),
        metadata={"speaker": tuple("spk0"), "dataset": ("synthetic",) * n},
        durations=tuple(2.0 for _ in range(n)),
        matrices=mats,
    )


def train_dev_test(seed: int = 1) -> tuple[FeatureMatrix, FeatureMatrix, FeatureMatrix]:
    """Build the three splits.

    Args:
        seed: RNG seed.

    Returns:
        Train, dev, and test matrices.
    """
    return (
        make_matrix("train", 96, seed=seed),
        make_matrix("dev", 32, seed=seed + 1),
        make_matrix("test", 32, seed=seed + 2),
    )


class TestConstruction:
    @pytest.mark.parametrize("family", ["mfcc_logreg", "mfcc_xgboost"])
    def test_scikit_families_need_no_frame_shape(self, family: str) -> None:
        # The two tabular models must build without the caller knowing anything
        # about the front end's frame geometry.
        model = build_baseline(ModelSpec(family=family))
        assert isinstance(model, BaselineModel)
        assert model.requires_matrices is False
        assert model.fitted is False

    def test_cnn_requires_frame_shape(self) -> None:
        with pytest.raises(TrainingError, match=r"n_frames and n_bands"):
            build_baseline(ModelSpec(family="logmel_cnn"))

    def test_cnn_declares_matrix_requirement(self) -> None:
        model = build_baseline(ModelSpec(family="logmel_cnn"), n_frames=N_FRAMES, n_bands=N_BANDS)
        assert model.requires_matrices is True

    def test_importing_the_module_does_not_import_torch(self) -> None:
        # The tabular baselines must remain usable without PyTorch, so this
        # module may not drag torch in at import time.
        import subprocess
        import sys

        code = (
            "import sys, voxshield.training.models as m;"
            "m.build_baseline(m.ModelSpec(family='mfcc_logreg'));"
            "print('torch' in sys.modules)"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"


class TestGuards:
    @pytest.mark.parametrize("family", ["mfcc_logreg", "mfcc_xgboost"])
    def test_scoring_before_fit_is_refused(self, family: str) -> None:
        model = build_baseline(ModelSpec(family=family))
        with pytest.raises(TrainingError, match=r"before it was fitted"):
            model.predict_proba(make_matrix("test", 8))

    @pytest.mark.parametrize("family", ["mfcc_logreg", "mfcc_xgboost"])
    @pytest.mark.parametrize("only", [0, 1])
    def test_one_class_train_split_is_refused(self, family: str, only: int) -> None:
        train = make_matrix("train", 32, labels=np.full(32, only))
        model = build_baseline(ModelSpec(family=family))
        with pytest.raises(TrainingError, match=r"have no training rows"):
            model.fit(train)

    @pytest.mark.parametrize("family", ["mfcc_logreg", "mfcc_xgboost"])
    def test_non_finite_vectors_are_refused(self, family: str) -> None:
        train = make_matrix("train", 32)
        train.vectors[2, 1] = np.nan
        model = build_baseline(ModelSpec(family=family))
        with pytest.raises(TrainingError, match=r"non-finite"):
            model.fit(train)

    @pytest.mark.parametrize("family", ["mfcc_logreg", "mfcc_xgboost"])
    def test_non_finite_vectors_are_refused_at_score_time(self, family: str) -> None:
        train, _, test = train_dev_test()
        model = build_baseline(ModelSpec(family=family))
        model.fit(train)
        test.vectors[0, 0] = np.inf
        with pytest.raises(TrainingError, match=r"non-finite"):
            model.predict_proba(test)

    def test_cnn_refuses_pooled_vectors(self) -> None:
        train = make_matrix("train", 32, matrices=False)
        model = build_baseline(ModelSpec(family="logmel_cnn"), n_frames=N_FRAMES, n_bands=N_BANDS)
        with pytest.raises(TrainingError, match=r"frame matrices"):
            model.fit(train)

    @pytest.mark.parametrize("shape", [(0, 12), (24, 0), (24, -1)])
    def test_cnn_rejects_degenerate_frame_shape(self, shape: tuple[int, int]) -> None:
        with pytest.raises(TrainingError, match=r"positive frame shape"):
            LogMelCNN(ModelSpec(family="logmel_cnn"), n_frames=shape[0], n_bands=shape[1])

    def test_cnn_refuses_non_finite_matrices(self) -> None:
        train = make_matrix("train", 32)
        assert train.matrices is not None
        train.matrices[1, 1, 1] = np.nan
        model = build_baseline(ModelSpec(family="logmel_cnn"), n_frames=N_FRAMES, n_bands=N_BANDS)
        with pytest.raises(TrainingError, match=r"non-finite"):
            model.fit(train)


class TestProbabilityContract:
    @pytest.mark.parametrize("family", ["mfcc_logreg", "mfcc_xgboost", "logmel_cnn"])
    def test_returns_one_column_spoof_positive(self, family: str) -> None:
        # Spoof is label 1, so column 1 is the spoof column for both libraries.
        # A silent swap here would invert every downstream metric.
        train, dev, test = train_dev_test()
        model = build_baseline(
            ModelSpec(family=family, max_iter=40, batch_size=16),
            n_frames=N_FRAMES,
            n_bands=N_BANDS,
            epochs=2,
        )
        model.fit(train, dev=dev)
        scores = model.predict_proba(test)
        assert scores.shape == (len(test),), scores.shape
        assert np.isfinite(scores).all()
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    @pytest.mark.parametrize("family", ["mfcc_logreg", "mfcc_xgboost", "logmel_cnn"])
    def test_separable_data_scores_spoof_higher(self, family: str) -> None:
        train, dev, test = train_dev_test()
        model = build_baseline(
            ModelSpec(family=family, max_iter=40, batch_size=16),
            n_frames=N_FRAMES,
            n_bands=N_BANDS,
            epochs=2,
        )
        model.fit(train, dev=dev)
        scores = model.predict_proba(test)
        spoof_mean = float(scores[test.labels == 1].mean())
        bona_mean = float(scores[test.labels == 0].mean())
        assert spoof_mean > bona_mean

    @pytest.mark.parametrize("family", ["mfcc_logreg", "mfcc_xgboost", "logmel_cnn"])
    def test_scoring_is_deterministic(self, family: str) -> None:
        train, dev, test = train_dev_test()
        model = build_baseline(
            ModelSpec(family=family, max_iter=40, batch_size=16),
            n_frames=N_FRAMES,
            n_bands=N_BANDS,
            epochs=2,
        )
        model.fit(train, dev=dev)
        first = model.predict_proba(test)
        second = model.predict_proba(test)
        np.testing.assert_allclose(first, second)


class TestNoTestContact:
    def test_fit_never_accepts_a_test_split(self) -> None:
        # fit() accepts train and dev only. If a test split were ever consulted
        # during fitting, this would stop being checkable, so the signature is
        # asserted directly.
        import inspect

        for cls in (LogisticBaseline, XGBoostBaseline, LogMelCNN):
            params = set(inspect.signature(cls.fit).parameters)
            assert params == {"self", "train", "dev"}, (cls, params)


class TestSelectionReporting:
    def test_linear_model_reports_no_selection(self) -> None:
        train, dev, _test = train_dev_test()
        model = build_baseline(ModelSpec(family="mfcc_logreg", max_iter=200))
        summary = model.fit(train, dev=dev)
        assert isinstance(summary, FitSummary)
        assert summary.selection_metric == "none"
        assert summary.best_epoch is None
        assert summary.epochs_run == 1
        assert summary.n_train == len(train)

    def test_boosted_model_selects_a_round_on_dev(self) -> None:
        train, dev, _ = train_dev_test()
        model = build_baseline(ModelSpec(family="mfcc_xgboost", max_iter=60))
        summary = model.fit(train, dev=dev)
        assert summary.selection_metric == "dev_logloss"
        assert summary.best_epoch is not None
        assert summary.selection_value is not None
        assert summary.epochs_run == summary.best_epoch + 1
        assert any("early stopping" in note for note in summary.notes)

    def test_boosted_model_without_dev_reports_no_selection(self) -> None:
        # Claiming selection that did not happen is the specific dishonesty this
        # guards against: the artefact would read "dev-selected" for a model that
        # used every round.
        train, _, _ = train_dev_test()
        model = build_baseline(ModelSpec(family="mfcc_xgboost", max_iter=30))
        summary = model.fit(train)
        assert summary.selection_metric == "none"
        assert summary.selection_value is None
        assert summary.best_epoch is None
        assert summary.epochs_run == 30
        assert any("untuned" in note for note in summary.notes)

    def test_disabled_early_stopping_still_trains(self) -> None:
        # xgboost raises if early_stopping_rounds is set without an eval set, so
        # a no-dev run must not set it.
        train, _, _ = train_dev_test()
        model = build_baseline(
            ModelSpec(family="mfcc_xgboost", max_iter=20), early_stopping_rounds=25
        )
        summary = model.fit(train)
        assert summary.epochs_run == 20
        assert summary.selection_metric == "none"

    def test_disabled_early_stopping_with_dev_says_so(self) -> None:
        train, dev, _ = train_dev_test()
        model = build_baseline(
            ModelSpec(family="mfcc_xgboost", max_iter=20), early_stopping_rounds=0
        )
        summary = model.fit(train, dev=dev)
        assert summary.selection_metric == "none"
        assert any("disabled" in note for note in summary.notes)

    def test_cnn_selects_an_epoch_on_dev(self) -> None:
        train, dev, _ = train_dev_test()
        model = build_baseline(
            ModelSpec(family="logmel_cnn", batch_size=16),
            n_frames=N_FRAMES,
            n_bands=N_BANDS,
            epochs=3,
        )
        summary = model.fit(train, dev=dev)
        assert summary.selection_metric == "dev_logloss"
        assert summary.best_epoch is not None
        assert 1 <= summary.best_epoch <= 3
        assert summary.epochs_run == 3

    def test_cnn_without_dev_reports_no_selection(self) -> None:
        train, _, _ = train_dev_test()
        model = build_baseline(
            ModelSpec(family="logmel_cnn", batch_size=16),
            n_frames=N_FRAMES,
            n_bands=N_BANDS,
            epochs=2,
        )
        summary = model.fit(train)
        assert summary.selection_metric == "none"
        assert summary.selection_value is None
        assert any("untuned" in note for note in summary.notes)

    def test_cnn_stops_early_on_patience(self) -> None:
        # Dev loss that cannot improve still has to terminate the loop, rather
        # than running the full epoch budget.
        train, dev, _ = train_dev_test(seed=5)
        model = build_baseline(
            ModelSpec(family="logmel_cnn", batch_size=16),
            n_frames=N_FRAMES,
            n_bands=N_BANDS,
            epochs=50,
            eval_every=1,
            patience=1,
        )
        summary = model.fit(train, dev=dev)
        assert summary.epochs_run < 50
        assert any("early stopped" in note for note in summary.notes)

    def test_deep_trees_are_flagged(self) -> None:
        train, dev, _ = train_dev_test()
        model = build_baseline(ModelSpec(family="mfcc_xgboost", max_depth=12, max_iter=10))
        summary = model.fit(train, dev=dev)
        assert any("memorise" in note for note in summary.notes)


class TestScalerContract:
    @pytest.mark.parametrize("family", ["mfcc_logreg", "mfcc_xgboost"])
    def test_scaler_is_fitted_on_train_only(self, family: str) -> None:
        # Fitting the scaler on train+dev leaks the dev distribution into the
        # model before any threshold is chosen.
        train, dev, _ = train_dev_test()
        model = build_baseline(ModelSpec(family=family, max_iter=20))
        model.fit(train, dev=dev)
        scaler = model._scaler
        assert scaler is not None
        expected = np.asarray(train.vectors).mean(axis=0)
        np.testing.assert_allclose(np.asarray(scaler.mean_), expected, rtol=1e-9)

    @pytest.mark.parametrize("family", ["mfcc_logreg", "mfcc_xgboost"])
    def test_transform_before_fit_is_refused(self, family: str) -> None:
        model = build_baseline(ModelSpec(family=family))
        with pytest.raises(TrainingError, match=r"no fitted scaler"):
            model._transform(np.zeros((2, N_FEATURES)))

    def test_class_weight_reaches_the_estimator(self) -> None:
        train, _, _ = train_dev_test()
        model = build_baseline(ModelSpec(family="mfcc_logreg", class_weight="balanced"))
        model.fit(train)
        assert model._estimator.class_weight == "balanced"


class TestDescribe:
    @pytest.mark.parametrize("family", ["mfcc_logreg", "mfcc_xgboost"])
    def test_describe_reports_feature_count_after_fit(self, family: str) -> None:
        train, dev, _ = train_dev_test()
        model = build_baseline(ModelSpec(family=family, max_iter=20))
        assert "n_features" not in model.describe()
        model.fit(train, dev=dev)
        described = model.describe()
        assert described["family"] == family
        assert described["fitted"] is True
        assert described["n_features"] == N_FEATURES

    def test_cnn_describe_reports_parameter_count(self) -> None:
        train, dev, _ = train_dev_test()
        model = build_baseline(
            ModelSpec(family="logmel_cnn", hidden_channels=(8, 16), batch_size=16),
            n_frames=N_FRAMES,
            n_bands=N_BANDS,
            epochs=1,
        )
        model.fit(train, dev=dev)
        described = model.describe()
        assert described["n_frames"] == N_FRAMES
        assert described["n_bands"] == N_BANDS
        assert 0 < described["n_parameters"] < 1_000_000


class TestFitSummary:
    def test_to_dict_is_json_ready(self) -> None:
        summary = FitSummary(
            n_train=10, n_dev=5, epochs_run=3, best_epoch=2, selection_metric="dev_logloss"
        )
        payload = summary.to_dict()
        assert payload["n_train"] == 10
        assert payload["best_epoch"] == 2
        assert payload["selection_value"] is None
        assert isinstance(payload["notes"], list)

    def test_repr_shows_the_selection_metric(self) -> None:
        text = repr(FitSummary(n_train=4, n_dev=2, selection_metric="dev_logloss"))
        assert "dev_logloss" in text
        assert "n/a" in text
