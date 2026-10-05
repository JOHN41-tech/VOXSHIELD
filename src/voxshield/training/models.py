"""The three Phase 3 baselines behind one interface.

Two linear models and one neural model, selected by
:func:`build_baseline` from :class:`~voxshield.training.config.ModelSpec`. They
share a contract that is deliberately narrow:

* :meth:`BaselineModel.fit` takes a :class:`~voxshield.training.datasets.FeatureMatrix`
  and returns a :class:`FitSummary`.
* :meth:`BaselineModel.predict_proba` takes a feature matrix and returns
  **one** column of synthetic probability, spoof positive, shape ``(n,)``.

Returning a single column rather than the two-column ``predict_proba`` these
libraries natively produce is the point of the wrapper. Spoof is positive in
VoxShield's labels but positive is the *first* column for neither library, and a
silent column swap inverts every metric in the report while still producing
numbers that look entirely reasonable. One column, shaped by one code path,
removes the possibility.

**Dev is used for model selection; test is never touched.** The boosted model
gets ``eval_set=(dev)`` for early stopping, and the CNN selects its best
checkpoint by dev loss. Both are legitimate uses of a development split. The
threshold is also chosen on dev. None of the three is allowed to see test.

**Torch is imported lazily.** :class:`LogMelCNN` must be constructible in an
environment without PyTorch only in the sense that asking for it raises a clear
error; importing this module never requires Torch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from voxshield.training.config import ModelSpec, TrainingError
from voxshield.training.datasets import FeatureMatrix

__all__ = [
    "BaselineModel",
    "FitSummary",
    "LogMelCNN",
    "LogisticBaseline",
    "XGBoostBaseline",
    "build_baseline",
]

_MISSING_TORCH = (
    "the logmel_cnn baseline needs PyTorch; install torch, or choose "
    "mfcc_logreg / mfcc_xgboost, which need only scikit-learn and XGBoost"
)


class FitSummary:
    """What one fit did, for the run log and the artefact metadata.

    Attributes:
        n_train: Rows fitted on.
        n_dev: Rows used for selection, or ``0`` when none were supplied.
        epochs_run: Epochs executed, or boosting rounds, or ``1`` for the linear
            models, which are single-shot.
        best_epoch: Selected epoch, when the model selects one.
        selection_metric: Name of the dev metric that drove selection.
        selection_value: Its value at the selected point.
        notes: Anything a reader of the artefact would otherwise have to guess.
    """

    __slots__ = (
        "best_epoch",
        "epochs_run",
        "n_dev",
        "n_train",
        "notes",
        "selection_metric",
        "selection_value",
    )

    def __init__(
        self,
        *,
        n_train: int,
        n_dev: int = 0,
        epochs_run: int = 1,
        best_epoch: int | None = None,
        selection_metric: str = "none",
        selection_value: float | None = None,
        notes: tuple[str, ...] = (),
    ) -> None:
        self.n_train = int(n_train)
        self.n_dev = int(n_dev)
        self.epochs_run = int(epochs_run)
        self.best_epoch = best_epoch
        self.selection_metric = selection_metric
        self.selection_value = selection_value
        self.notes = notes

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "n_train": self.n_train,
            "n_dev": self.n_dev,
            "epochs_run": self.epochs_run,
            "best_epoch": self.best_epoch,
            "selection_metric": self.selection_metric,
            "selection_value": self.selection_value,
            "notes": list(self.notes),
        }

    def __repr__(self) -> str:
        value = "n/a" if self.selection_value is None else f"{self.selection_value:.4f}"
        return (
            f"FitSummary(n_train={self.n_train}, n_dev={self.n_dev}, "
            f"epochs_run={self.epochs_run}, {self.selection_metric}={value})"
        )


class BaselineModel(ABC):
    """Common surface for the three baselines.

    Attributes:
        spec: The architecture settings this model was built from.
        requires_matrices: Whether the model consumes frame matrices rather than
            pooled vectors.
    """

    requires_matrices: bool = False

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._fitted = False

    @property
    def fitted(self) -> bool:
        """Whether :meth:`fit` has completed."""
        return self._fitted

    @abstractmethod
    def fit(self, train: FeatureMatrix, *, dev: FeatureMatrix | None = None) -> FitSummary:
        """Fit on ``train``, selecting on ``dev``.

        Args:
            train: Training features and labels.
            dev: Development features for checkpoint or round selection.

        Returns:
            A :class:`FitSummary`.

        Raises:
            TrainingError: If the data is unusable or a class is missing.
        """

    @abstractmethod
    def predict_proba(self, data: FeatureMatrix) -> np.ndarray:
        """Score rows as synthetic probability, spoof positive.

        Args:
            data: Features to score.

        Returns:
            Float64 array of shape ``(n,)`` with values in ``[0, 1]``.

        Raises:
            TrainingError: If called before :meth:`fit`.
        """

    def _require_fitted(self) -> None:
        """Guard against scoring with an unfitted model."""
        if not self._fitted:
            msg = f"{type(self).__name__} was asked to score before it was fitted"
            raise TrainingError(msg)

    def _check_labels(self, train: FeatureMatrix) -> dict[int, int]:
        """Confirm both classes are present, and return the counts."""
        counts = train.counts
        missing = [index for index, count in counts.items() if count == 0]
        if missing:
            names = ", ".join(str(index) for index in sorted(missing))
            msg = (
                f"{type(self).__name__} cannot be fitted: encoded label(s) {names} "
                "have no training rows; a one-class fit produces a model that "
                "always predicts the class it saw"
            )
            raise TrainingError(msg)
        return counts

    def describe(self) -> dict[str, Any]:
        """JSON-ready model description, for artefact metadata."""
        return {"class": type(self).__name__, "family": self.spec.family, "fitted": self._fitted}


class _SklearnBaseline(BaselineModel):
    """Shared machinery for the two scikit-learn baselines.

    Both standardise the features with a :class:`~sklearn.preprocessing.StandardScaler`
    fitted on train only, then fit an estimator on the result. The scaler is
    held as an attribute rather than hidden inside a
    :class:`~sklearn.pipeline.Pipeline` for two reasons. Pipeline's ``fit`` will
    not route an estimator-specific argument such as xgboost's ``eval_set``, so
    dev-based early stopping would have to bypass the pipeline anyway; and
    holding the pair explicitly means the artefact writer has two named things
    to serialise instead of one opaque object whose contents nobody has checked.

    Estimator and scaler are kept together on one object so that
    :meth:`predict_proba` cannot be reached with a fitted estimator and an
    absent scaler, which is the failure that silently scores unstandardised
    input.
    """

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self._scaler: Any = None
        self._estimator: Any = None

    def _build_estimator(self, *, enable_early_stopping: bool = False) -> Any:
        """Construct the underlying estimator.

        Args:
            enable_early_stopping: Whether the caller will supply a dev set. The
                boosted model needs this because xgboost refuses to construct a
                training run with ``early_stopping_rounds`` set and no eval set.

        Returns:
            An unfitted estimator.

        Raises:
            NotImplementedError: If a subclass does not override this.
        """
        raise NotImplementedError

    def _fit_scaler(self, vectors: np.ndarray) -> np.ndarray:
        """Fit the standardiser on training vectors and return the transform.

        The scaler is fitted on train only. Fitting it on train+dev, or on
        anything else, leaks the development distribution into the model.

        Args:
            vectors: Training vectors.

        Returns:
            Standardised training vectors.

        Raises:
            TrainingError: If PyTorch-free scikit-learn is unavailable.
        """
        try:
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:  # pragma: no cover - environment problem
            msg = "the MFCC baselines need scikit-learn; install scikit-learn"
            raise TrainingError(msg) from exc
        self._scaler = StandardScaler()
        return self._scaler.fit_transform(vectors)

    def _transform(self, vectors: np.ndarray) -> np.ndarray:
        """Apply the fitted standardiser.

        Args:
            vectors: Vectors to standardise.

        Returns:
            Standardised vectors.

        Raises:
            TrainingError: If called before the scaler is fitted.
        """
        if self._scaler is None:
            msg = f"{type(self).__name__} has no fitted scaler"
            raise TrainingError(msg)
        return self._scaler.transform(vectors)

    def fit(self, train: FeatureMatrix, *, dev: FeatureMatrix | None = None) -> FitSummary:
        """Fit scaler and estimator on the training split.

        Args:
            train: Training features. Frame matrices are not used.
            dev: Unused by the linear models. They select a threshold, which is
                the only degree of freedom they have, and a single-shot fit has
                nothing to select.

        Returns:
            A :class:`FitSummary` reporting a single-shot fit.

        Raises:
            TrainingError: If a class is missing or the features are not finite.
        """
        self._check_labels(train)
        _require_finite(train.vectors, split=train.split)
        scaled = self._fit_scaler(train.vectors)
        self._estimator = self._build_estimator()
        self._estimator.fit(scaled, train.labels)
        self._fitted = True
        return FitSummary(
            n_train=len(train),
            n_dev=0 if dev is None else len(dev),
            epochs_run=1,
            selection_metric="none",
            notes=(
                "single-shot fit; the scaler saw train only, and the operating "
                "point is the only selection performed, on dev",
            ),
        )

    def predict_proba(self, data: FeatureMatrix) -> np.ndarray:
        """Score rows as synthetic probability.

        Args:
            data: Features to score.

        Returns:
            Float64 array of shape ``(n,)``, the spoof column.

        Raises:
            TrainingError: If unfitted or the features are not finite.
        """
        self._require_fitted()
        _require_finite(data.vectors, split=data.split)
        # Column 1 is the spoof class: labels encode spoof as 1, and the
        # estimator was fitted on those labels, so the positive column is spoof.
        scaled = self._transform(data.vectors)
        return np.asarray(self._estimator.predict_proba(scaled)[:, 1], dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        """JSON-ready model description."""
        base = super().describe()
        if self._scaler is not None:
            base["n_features"] = int(np.asarray(self._scaler.mean_).shape[0])
        return base


class LogisticBaseline(_SklearnBaseline):
    """MFCC + logistic regression: the floor every later model must beat.

    Deliberately the simplest thing that can work. If a neural model cannot beat
    a standardised MFCC mean/std vector under a linear model, the neural model is
    not earning its complexity, and that is the single most useful result this
    phase can produce.
    """

    def _build_estimator(self, *, enable_early_stopping: bool = False) -> Any:
        """Construct the logistic regression estimator.

        Args:
            enable_early_stopping: Ignored. A single-shot linear fit has no
                rounds to stop early.

        Returns:
            An unfitted :class:`~sklearn.linear_model.LogisticRegression`.
        """
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            max_iter=self.spec.max_iter,
            C=self.spec.reg_lambda,
            class_weight=self.spec.sklearn_class_weight,
            solver="lbfgs",
        )


class XGBoostBaseline(_SklearnBaseline):
    """MFCC + gradient-boosted trees: the strong tabular baseline.

    Depth is capped at a low value and rows and columns are subsampled. Deep,
    un-subsampled trees will fit a training corpus almost exactly, and a model
    that has memorised its training set reports an EER on a resampled test split
    that it will not come close to on an unseen generator.

    Args:
        spec: Architecture settings.
        early_stopping_rounds: Rounds without dev improvement before boosting
            stops. ``0`` disables early stopping even when dev is supplied, which
            is a legitimate way to run but must not be reported as selection.
    """

    def __init__(self, spec: ModelSpec, *, early_stopping_rounds: int = 25) -> None:
        super().__init__(spec)
        self.early_stopping_rounds = max(0, int(early_stopping_rounds))

    def fit(self, train: FeatureMatrix, *, dev: FeatureMatrix | None = None) -> FitSummary:
        """Fit the boosted ensemble, using dev for early stopping.

        Args:
            train: Training features.
            dev: Development features. When given and early stopping is enabled,
                boosting stops on dev log loss and the best iteration is
                restored.

        Returns:
            A :class:`FitSummary` naming the selection metric, or reporting that
            no selection occurred.

        Raises:
            TrainingError: If a class is missing or the features are not finite.
        """
        self._check_labels(train)
        _require_finite(train.vectors, split=train.split)
        scaled_train = self._fit_scaler(train.vectors)

        fit_kwargs: dict[str, Any] = {}
        stop_on_dev = dev is not None and len(dev) > 0 and self.early_stopping_rounds > 0
        if stop_on_dev and dev is not None:
            _require_finite(dev.vectors, split=dev.split)
            # The dev set goes through the train-fitted scaler. Early stopping
            # on unscaled dev would score a different data distribution than the
            # one the model was fitted on.
            fit_kwargs["eval_set"] = [(self._transform(dev.vectors), dev.labels)]
            fit_kwargs["verbose"] = False

        self._estimator = self._build_estimator(enable_early_stopping=stop_on_dev)
        self._estimator.fit(scaled_train, train.labels, **fit_kwargs)
        self._fitted = True

        estimator = self._estimator
        # xgboost sets best_iteration only when an eval_set was supplied and
        # early stopping was enabled. Its absence means no selection happened,
        # which is different from a selection that chose round zero.
        best = getattr(estimator, "best_iteration", None)
        selected = isinstance(best, int) and best >= 0
        notes: list[str] = []
        if selected and best is not None:
            notes.append(
                f"early stopping over {self.early_stopping_rounds}-round patience; "
                f"best round {best} restored"
            )
        elif dev is None or len(dev) == 0:
            notes.append(
                "no dev split supplied, so all boosting rounds were used; the "
                "round count is therefore untuned"
            )
        else:
            notes.append(
                f"early stopping disabled (early_stopping_rounds="
                f"{self.early_stopping_rounds}), so all {self.spec.max_iter} "
                "rounds were used despite a dev split being available"
            )
        if self.spec.max_depth > 6:
            notes.append(
                f"max_depth={self.spec.max_depth} is deep enough to memorise "
                "small corpora; treat the result with suspicion"
            )
        score = getattr(estimator, "best_score", None)
        return FitSummary(
            n_train=len(train),
            n_dev=0 if dev is None else len(dev),
            epochs_run=(best + 1) if selected and best is not None else self.spec.max_iter,
            best_epoch=best if selected and best is not None else None,
            selection_metric="dev_logloss" if selected else "none",
            selection_value=float(score) if selected and score is not None else None,
            notes=tuple(notes),
        )

    def _build_estimator(self, *, enable_early_stopping: bool = False) -> Any:
        """Construct the boosted-tree estimator.

        Args:
            enable_early_stopping: Set ``early_stopping_rounds`` on the
                estimator. Only safe to do when a dev set will be passed, because
                xgboost raises "Must have at least 1 validation dataset for early
                stopping" otherwise, so a no-dev run would fail rather than
                quietly train every round.

        Returns:
            An unfitted :class:`~xgboost.XGBClassifier`.

        Raises:
            TrainingError: If XGBoost is unavailable.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - environment problem
            msg = "the mfcc_xgboost baseline needs XGBoost; install xgboost, or choose mfcc_logreg"
            raise TrainingError(msg) from exc

        kwargs: dict[str, Any] = {
            "n_estimators": self.spec.max_iter,
            "max_depth": self.spec.max_depth,
            "learning_rate": self.spec.learning_rate,
            "subsample": self.spec.subsample,
            "colsample_bytree": self.spec.colsample,
            "reg_lambda": self.spec.reg_lambda,
            "tree_method": "hist",
            "random_state": 0,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        if enable_early_stopping and self.early_stopping_rounds > 0:
            # A constructor argument in xgboost >= 2.0, not a fit() argument.
            kwargs["early_stopping_rounds"] = self.early_stopping_rounds
        return XGBClassifier(**kwargs)


class LogMelCNN(BaselineModel):
    """A small 2-D CNN over log-mel frames.

    Two convolutional blocks with batch norm, then global average pooling and a
    linear head. Global average pooling rather than flatten is the important
    choice: it makes the model indifferent to where a cue sits in time, so a
    three-second pad does not shift the answer, and it keeps the parameter count
    low enough to train on CPU in a smoke test.

    Args:
        spec: Architecture settings.
        n_frames: Frames per clip, from the feature configuration.
        n_bands: Mel bands per frame.
        seed: Random seed, applied to every torch RNG this model touches.
        epochs: Training epochs. This lives on
            :class:`~voxshield.training.config.TrainSpec`, not on ``spec``, and
            is passed in explicitly: reading ``spec.epochs`` looks correct and
            silently yields nothing, which would train on a hard-coded default
            while the artefact recorded a different epoch count.
        eval_every: Score dev every N epochs.
        patience: Stop after N evaluations without improvement. ``0`` disables.

    Raises:
        TrainingError: If PyTorch is unavailable, or the frame shape is unusable.
    """

    requires_matrices = True

    def __init__(
        self,
        spec: ModelSpec,
        *,
        n_frames: int,
        n_bands: int,
        seed: int = 0,
        epochs: int = 12,
        eval_every: int = 1,
        patience: int = 0,
    ) -> None:
        super().__init__(spec)
        if n_frames < 1 or n_bands < 1:
            msg = f"CNN needs a positive frame shape, got ({n_frames}, {n_bands})"
            raise TrainingError(msg)
        self.n_frames = int(n_frames)
        self.n_bands = int(n_bands)
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.eval_every = max(1, int(eval_every))
        self.patience = int(patience)
        self._network: Any = None

    @staticmethod
    def _torch() -> Any:
        """Import torch on demand.

        Returns:
            The ``torch`` module.

        Raises:
            TrainingError: If PyTorch is not installed.
        """
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment problem
            raise TrainingError(_MISSING_TORCH) from exc
        return torch

    def _build_network(self) -> Any:
        """Construct the torch module.

        Returns:
            A :class:`~voxshield.training._cnn.LogMelNet`.

        Raises:
            TrainingError: If PyTorch is unavailable.
        """
        try:
            from voxshield.training._cnn import LogMelNet, seed_everything
        except ImportError as exc:
            raise TrainingError(_MISSING_TORCH) from exc
        seed_everything(self.seed)
        return LogMelNet(self.n_bands, self.spec.hidden_channels or (32,), self.spec.dropout)

    def fit(self, train: FeatureMatrix, *, dev: FeatureMatrix | None = None) -> FitSummary:
        """Train with dev-based checkpoint selection.

        Args:
            train: Training data. Frame matrices are required.
            dev: Development data, used to pick the best epoch and to score.

        Returns:
            A :class:`FitSummary` naming the selected epoch and its dev loss.

        Raises:
            TrainingError: If frame matrices are absent, a class is missing, or
                the features are not finite.
        """
        self._check_labels(train)
        if train.matrices is None:
            msg = (
                "logmel_cnn needs frame matrices; call load_feature_matrix with want_matrices=True"
            )
            raise TrainingError(msg)
        _require_finite(train.matrices, split=train.split)

        torch = self._torch()
        self._network = self._build_network()
        optimiser = torch.optim.AdamW(self._network.parameters(), lr=self.spec.learning_rate)
        # Inverse-frequency weights, so a corpus with 20x more bona fide windows
        # does not train the model to call everything bona fide.
        counts = train.counts
        total = float(sum(counts.values()))
        weights = torch.tensor(
            [total / (2.0 * max(counts[i], 1)) for i in range(2)], dtype=torch.float32
        )
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        x_train = torch.from_numpy(np.asarray(train.matrices, dtype=np.float32))
        y_train = torch.from_numpy(np.asarray(train.labels, dtype=np.int64))
        x_dev = None
        if dev is not None and dev.matrices is not None and len(dev) > 0:
            _require_finite(dev.matrices, split=dev.split)
            x_dev = torch.from_numpy(np.asarray(dev.matrices, dtype=np.float32))
            y_dev = torch.from_numpy(np.asarray(dev.labels, dtype=np.int64))

        generator = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train),
            batch_size=self.spec.batch_size,
            shuffle=True,
            generator=generator,
        )

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        best_epoch = 0
        stale = 0
        history: list[float] = []
        evaluations = 0
        ran = 0
        for epoch in range(1, self.epochs + 1):
            ran = epoch
            self._network.train()
            for batch_x, batch_y in loader:
                optimiser.zero_grad()
                loss = criterion(self._network(batch_x), batch_y)
                loss.backward()
                optimiser.step()

            if x_dev is None or epoch % self.eval_every != 0:
                continue

            evaluations += 1
            dev_loss = self._dev_loss(criterion, x_dev, y_dev)
            history.append(dev_loss)
            if dev_loss < best_loss - 1e-6:
                best_loss = dev_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in self._network.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if self.patience and stale >= self.patience:
                    break

        if best_state is not None:
            self._network.load_state_dict(best_state)
        self._fitted = True

        notes = [
            "global average pooling over time, so a padded clip does not shift the score",
            "class-weighted cross entropy",
        ]
        if x_dev is None:
            notes.append(
                "no dev matrices supplied, so the final epoch was used rather than "
                "a selected one; the epoch count is untuned"
            )
        if evaluations < ran and x_dev is not None:
            notes.append(
                f"dev was scored every {self.eval_every} epoch(s), so "
                f"{ran - evaluations} epoch(s) went unevaluated"
            )
        if self.patience and stale >= self.patience:
            notes.append(f"early stopped after {self.patience} evaluation(s) without improvement")
        return FitSummary(
            n_train=len(train),
            n_dev=0 if dev is None else len(dev),
            epochs_run=ran,
            best_epoch=best_epoch if best_state is not None else None,
            selection_metric="dev_logloss" if x_dev is not None else "none",
            selection_value=best_loss if best_state is not None else None,
            notes=tuple(notes),
        )

    def _dev_loss(self, criterion: Any, x_dev: Any, y_dev: Any) -> float:
        """Mean weighted cross-entropy on the development split.

        Args:
            criterion: The training loss, which is class-weighted.
            x_dev: Development matrices.
            y_dev: Development labels.

        Returns:
            The row-weighted mean loss, comparable across epochs.
        """
        torch = self._torch()
        self._network.eval()
        batch = max(1, self.spec.batch_size)
        total = 0.0
        seen = 0
        with torch.no_grad():
            for start in range(0, int(x_dev.shape[0]), batch):
                stop = min(start + batch, int(x_dev.shape[0]))
                logits = self._network(x_dev[start:stop])
                total += float(criterion(logits, y_dev[start:stop])) * (stop - start)
                seen += stop - start
        return total / max(1, seen)

    def predict_proba(self, data: FeatureMatrix) -> np.ndarray:
        """Score rows as synthetic probability.

        Args:
            data: Features with frame matrices.

        Returns:
            Float64 array of shape ``(n,)``, softmax column 1 (spoof).

        Raises:
            TrainingError: If unfitted, matrices are absent, or the features are
                not finite.
        """
        self._require_fitted()
        if data.matrices is None:
            msg = (
                "logmel_cnn scoring needs frame matrices; call "
                "load_feature_matrix with want_matrices=True"
            )
            raise TrainingError(msg)
        _require_finite(data.matrices, split=data.split)

        torch = self._torch()
        self._network.eval()
        x = torch.from_numpy(np.asarray(data.matrices, dtype=np.float32))
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x.shape[0], max(1, self.spec.batch_size)):
                logits = self._network(x[start : start + self.spec.batch_size])
                outputs.append(torch.softmax(logits, dim=1)[:, 1].numpy())
        return np.concatenate(outputs).astype(np.float64)

    def describe(self) -> dict[str, Any]:
        """JSON-ready model description, including the parameter count."""
        base = super().describe()
        base.update({"n_frames": self.n_frames, "n_bands": self.n_bands})
        if self._network is not None:
            base["n_parameters"] = int(sum(p.numel() for p in self._network.parameters()))
        return base


def _require_finite(array: np.ndarray, *, split: str) -> None:
    """Reject non-finite features before they reach a model.

    A NaN reaching a gradient-descent step does not stay put: it propagates
    through every weight and destroys the fit. Failing here, naming the split,
    is much cheaper than discovering it as a model that never converges.

    Args:
        array: Feature array to check.
        split: Split name, for the error message.

    Raises:
        TrainingError: If any value is NaN or infinite.
    """
    if not np.isfinite(array).all():
        bad = int(np.count_nonzero(~np.isfinite(array)))
        msg = (
            f"split {split!r} has {bad} non-finite feature value(s); refusing to "
            "fit, because a NaN in a gradient step does not stay local"
        )
        raise TrainingError(msg)


def build_baseline(
    spec: ModelSpec,
    *,
    n_frames: int | None = None,
    n_bands: int | None = None,
    seed: int = 0,
    epochs: int = 12,
    eval_every: int = 1,
    patience: int = 0,
    early_stopping_rounds: int = 25,
) -> BaselineModel:
    """Construct the baseline for a family.

    Args:
        spec: Architecture settings, whose ``family`` selects the implementation.
        n_frames: Frames per clip, required by the CNN.
        n_bands: Mel bands per frame, required by the CNN.
        seed: Random seed for the CNN.
        epochs: Epochs for the CNN. From
            :attr:`~voxshield.training.config.TrainSpec.epochs`.
        eval_every: Dev evaluation interval for the CNN.
        patience: Early-stopping patience for the CNN, in evaluations.
        early_stopping_rounds: Early-stopping patience for the boosted trees,
            in boosting rounds. The two are the same idea at different granularities
            and are kept separate because a round is not an epoch.

    Returns:
        An unfitted :class:`BaselineModel`.

    Raises:
        TrainingError: If the family is unknown, or the CNN is requested without
            a frame shape.
    """
    if spec.family == "mfcc_logreg":
        return LogisticBaseline(spec)
    if spec.family == "mfcc_xgboost":
        return XGBoostBaseline(spec, early_stopping_rounds=early_stopping_rounds)
    if spec.family == "logmel_cnn":
        if n_frames is None or n_bands is None:
            msg = (
                "logmel_cnn requires n_frames and n_bands from the feature "
                "configuration; pass them from the front end's describe()"
            )
            raise TrainingError(msg)
        return LogMelCNN(
            spec,
            n_frames=n_frames,
            n_bands=n_bands,
            seed=seed,
            epochs=epochs,
            eval_every=eval_every,
            patience=patience,
        )
    msg = f"unknown model family {spec.family!r}; known: mfcc_logreg, mfcc_xgboost, logmel_cnn"
    raise TrainingError(msg)
