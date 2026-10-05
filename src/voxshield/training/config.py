"""Configuration for the Phase 3 baselines.

A training configuration is the complete recipe for one model: which features,
which architecture, which hyper-parameters, which threshold rule, and where the
result goes. It is hashed, and the hash travels with every artefact, for one
reason: an EER without the configuration that produced it is not reproducible
and not comparable to any other EER in the report. Two experiments that differ
only in ``C`` are different experiments.

The configuration also carries the constraints that keep a baseline honest,
which is why several fields exist even though nothing reads them during fitting:

* ``dev_split`` and ``test_split`` name the splits, and
  :meth:`TrainingConfig.validate_protocol` refuses a configuration that would
  train or tune on test. The refusal is in code rather than in documentation
  because a documented rule is not enforced and an unenforced rule is the one
  that gets broken by the third person who reads this file.
* ``max_frr``/``max_far`` express the operational cost of each error as an
  explicit ceiling, so the operating point is a stated decision rather than a
  number that fell out of a sweep.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from voxshield.evaluation.metrics import ThresholdPolicy

#: Probability calibrations a run may request. Declared here rather than in
#: ``calibration`` so that validating a recipe needs no import of the module
#: that imports this one.
CALIBRATION_METHODS: Final = ("platt", "isotonic", "none")

__all__ = [
    "DEFAULT_ML_CONFIG_DIR",
    "MODEL_FAMILIES",
    "FeatureSpec",
    "ModelSpec",
    "TrainSpec",
    "TrainingConfig",
    "TrainingError",
    "load_training_config",
    "parse_training_config",
]

#: Where the shipped baseline configurations live, relative to the repository root.
DEFAULT_ML_CONFIG_DIR: Final = Path("configs/ml")

#: Model families this phase implements. A configuration naming anything else is
#: rejected at load time rather than at fit time: discovering an unimplemented
#: family after a 40-minute training run is a poor way to learn it.
MODEL_FAMILIES: Final[frozenset[str]] = frozenset({"mfcc_logreg", "mfcc_xgboost", "logmel_cnn"})


class TrainingError(Exception):
    """A training configuration or run could not proceed as asked."""


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Front-end settings, aligned with :class:`voxshield.config.FeatureConfig`.

    The defaults are the same numbers the production feature extractor already
    uses, and that is the whole point. A baseline trained on 40 mel bands
    measures a different system from the one the API serves, and the resulting
    EER describes a model nobody will ever deploy. The front end is specified
    once, in :mod:`voxshield.audio.features`, and copied here as defaults.

    Attributes:
        kind: ``"mfcc"`` or ``"log_mel"``.
        n_fft: FFT size in samples. Also sets the frame length, since the window
            is applied per frame.
        hop_length: Advance between frames, in samples.
        n_mels: Mel bands before the DCT.
        fmin: Lowest frequency of the mel filterbank, in Hz.
        fmax: Highest frequency of the mel filterbank, in Hz.
        htk: Use the HTK mel scale. Matches the production front end; changing
            it changes the features, not just their scale.
        n_coefficients: MFCC coefficients to keep. Ignored for ``"log_mel"``.
        lifter: Cepstral mean-normalisation lifter coefficient. ``0`` disables
            it. Mean normalisation is what keeps the model from learning channel
            or microphone identity instead of synthesis, so it defaults on.
        with_delta: Append first-order delta coefficients.
        n_frames: Fixed frame count per clip. ``None`` keeps every clip's natural
            length, which no fixed-shape classifier can consume.
        target_frames: Length used when averaging a clip down to one vector.
    """

    kind: str = "mfcc"
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    fmin: float = 20.0
    fmax: float = 7600.0
    htk: bool = True
    n_coefficients: int = 20
    lifter: float = 22.0
    with_delta: bool = False
    n_frames: int | None = 300
    target_frames: int | None = 300

    def __post_init__(self) -> None:
        """Reject settings that cannot produce usable features."""
        if self.kind not in {"mfcc", "log_mel"}:
            msg = f"feature kind must be 'mfcc' or 'log_mel', got {self.kind!r}"
            raise TrainingError(msg)
        for name in ("n_fft", "hop_length", "n_mels", "n_coefficients", "target_frames"):
            # target_frames and n_frames both accept None, meaning "keep each
            # clip's natural length". YAML spells that as `null`, and comparing
            # None to 1 is a TypeError rather than a validation message.
            value = getattr(self, name)
            if value is not None and value < 1:
                msg = f"{name} must be positive, got {value}"
                raise TrainingError(msg)
        if not 0.0 <= self.fmin < self.fmax:
            msg = f"require 0 <= fmin < fmax, got fmin={self.fmin} fmax={self.fmax}"
            raise TrainingError(msg)
        if self.kind == "mfcc" and self.n_coefficients > self.n_mels:
            msg = (
                f"n_coefficients ({self.n_coefficients}) cannot exceed n_mels "
                f"({self.n_mels}); the DCT cannot produce more coefficients than "
                "there are mel bands"
            )
            raise TrainingError(msg)
        if self.lifter < 0.0:
            msg = f"lifter must be non-negative, got {self.lifter}"
            raise TrainingError(msg)
        if self.n_frames is not None and self.n_frames < 1:
            msg = f"n_frames must be positive or None, got {self.n_frames}"
            raise TrainingError(msg)

    @property
    def output_dim(self) -> int:
        """Width of one feature vector."""
        base = self.n_coefficients if self.kind == "mfcc" else self.n_mels
        return base * 2 if self.with_delta else base

    @property
    def per_frame_dim(self) -> int:
        """Width of one frame, before any fixed-length reduction.

        Equal to :attr:`output_dim`; distinct because the CNN consumes per-frame
        matrices while the linear models consume pooled vectors, and the two are
        easy to conflate when the numbers happen to match.
        """
        return self.output_dim

    @property
    def pooled_dim(self) -> int:
        """Width of one pooled vector, mean and standard deviation concatenated."""
        return 2 * self.output_dim

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "kind": self.kind,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "htk": self.htk,
            "n_coefficients": self.n_coefficients,
            "lifter": self.lifter,
            "with_delta": self.with_delta,
            "n_frames": self.n_frames,
            "target_frames": self.target_frames,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FeatureSpec:
        """Build from a parsed YAML mapping, rejecting unknown keys.

        Args:
            payload: The ``features`` block.

        Returns:
            A validated :class:`FeatureSpec`.

        Raises:
            TrainingError: On an unknown key or a bad value. Unknown keys are
                errors rather than ignored, because a misspelled ``n_mel`` that
                silently falls back to 80 is a configuration that looks applied
                and is not.
        """
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            msg = f"unknown feature option(s): {', '.join(unknown)}; known: {sorted(known)}"
            raise TrainingError(msg)
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Architecture and optimiser settings.

    Hyper-parameters default to conservative values rather than tuned ones,
    because there is nothing to tune against. The one that matters most is
    ``max_depth`` on the boosted model: deep trees memorise a corpus, and a
    memorising model scores well on a resampled test set and terribly on a new
    generator.

    Attributes:
        family: One of :data:`MODEL_FAMILIES`.
        max_iter: Boosting rounds, or epochs for the CNN. Shared deliberately so
            a single ``epochs``-style knob reads the same across families.
        learning_rate: Step size for boosting and the CNN.
        max_depth: Tree depth. Ignored by the linear models.
        subsample: Row fraction per boosting round.
        colsample: Column fraction per boosting round.
        reg_lambda: L2 penalty.
        hidden_channels: CNN widths.
        dropout: CNN dropout probability.
        batch_size: CNN batch size.
        class_weight: Reweight the minority class. On for the linear models,
            because a corpus's spoof/bona fide ratio is a property of how it was
            assembled rather than of the problem.
        max_scaler_samples: Cap on rows used to fit the scaler, so a large corpus
            does not make preprocessing the slowest step in the run.
    """

    family: str = "mfcc_logreg"
    max_iter: int = 300
    learning_rate: float = 0.1
    max_depth: int = 4
    subsample: float = 0.9
    colsample: float = 0.9
    reg_lambda: float = 1.0
    hidden_channels: tuple[int, ...] = (32, 64)
    dropout: float = 0.2
    batch_size: int = 32
    class_weight: str = "balanced"
    max_scaler_samples: int = 50_000

    def __post_init__(self) -> None:
        """Reject an unsupported family or a nonsensical hyper-parameter."""
        if self.family not in MODEL_FAMILIES:
            msg = f"unknown model family {self.family!r}; known: {sorted(MODEL_FAMILIES)}"
            raise TrainingError(msg)
        if self.max_iter < 1:
            msg = f"max_iter must be positive, got {self.max_iter}"
            raise TrainingError(msg)
        if self.learning_rate <= 0.0:
            msg = f"learning_rate must be positive, got {self.learning_rate}"
            raise TrainingError(msg)
        if self.max_depth < 1:
            msg = f"max_depth must be positive, got {self.max_depth}"
            raise TrainingError(msg)
        if not 0.0 < self.subsample <= 1.0:
            msg = f"subsample must lie in (0, 1], got {self.subsample}"
            raise TrainingError(msg)
        if not 0.0 < self.colsample <= 1.0:
            msg = f"colsample must lie in (0, 1], got {self.colsample}"
            raise TrainingError(msg)
        if self.reg_lambda < 0.0:
            msg = f"reg_lambda must be non-negative, got {self.reg_lambda}"
            raise TrainingError(msg)
        if not 0.0 <= self.dropout < 1.0:
            msg = f"dropout must lie in [0, 1), got {self.dropout}"
            raise TrainingError(msg)
        if self.batch_size < 1:
            msg = f"batch_size must be positive, got {self.batch_size}"
            raise TrainingError(msg)
        if any(width < 1 for width in self.hidden_channels):
            msg = f"hidden_channels must all be positive, got {self.hidden_channels}"
            raise TrainingError(msg)
        if self.class_weight not in {"balanced", "none"}:
            msg = f"class_weight must be 'balanced' or 'none', got {self.class_weight!r}"
            raise TrainingError(msg)

    @property
    def sklearn_class_weight(self) -> str | None:
        """The value to hand scikit-learn, or ``None`` for no reweighting."""
        return None if self.class_weight == "none" else "balanced"

    @property
    def uses_torch(self) -> bool:
        """Whether this family requires PyTorch."""
        return self.family == "logmel_cnn"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping, with the channel tuple as a list."""
        return {
            "family": self.family,
            "max_iter": self.max_iter,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample": self.colsample,
            "reg_lambda": self.reg_lambda,
            "hidden_channels": list(self.hidden_channels),
            "dropout": self.dropout,
            "batch_size": self.batch_size,
            "class_weight": self.class_weight,
            "max_scaler_samples": self.max_scaler_samples,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ModelSpec:
        """Build from a parsed YAML mapping, rejecting unknown keys.

        Args:
            payload: The ``model`` block.

        Returns:
            A validated :class:`ModelSpec`.

        Raises:
            TrainingError: On an unknown key or a bad value.
        """
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            msg = f"unknown model option(s): {', '.join(unknown)}; known: {sorted(known)}"
            raise TrainingError(msg)
        values = dict(payload)
        if "hidden_channels" in values and values["hidden_channels"] is not None:
            values["hidden_channels"] = tuple(values["hidden_channels"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class TrainSpec:
    """Reproducibility and runtime settings.

    Attributes:
        seed: Seed for every stochastic component. Recorded in the artefact so
            that a rerun can be checked for bitwise agreement rather than merely
            similar accuracy.
        epochs: Epoch count for the CNN. Ignored by the other families, which
            use :attr:`~ModelSpec.max_iter`.
        device: ``"cpu"`` or ``"cuda"``. Forced to CPU when CUDA is unavailable
            rather than failing the run, because a baseline that only trains on
            one machine is not reproducible.
        eval_every: Score the dev split every N epochs, for the CNN's
            checkpoint selection.
        patience: Early-stopping patience in evaluation rounds. ``0`` disables.
        verify_leakage: Refuse to train when a manifest gate fails.
        min_train_samples: Floor on training rows, below which a run is refused
            as underpowered rather than reported as a poor result.
        min_dev_samples: Floor on dev rows for threshold selection.
        calibration: How to map raw scores to probabilities, fitted on dev.
            ``"platt"`` is the default and the safe choice; ``"isotonic"`` is
            refused on dev splits too small to support it; ``"none"``
            disables calibration and is recorded in the artefact.
    """

    seed: int = 20260928
    epochs: int = 12
    device: str = "cpu"
    eval_every: int = 1
    patience: int = 0
    verify_leakage: bool = True
    min_train_samples: int = 100
    min_dev_samples: int = 50
    calibration: str = "platt"

    def __post_init__(self) -> None:
        """Reject settings that would make a run non-reproducible or unsafe."""
        if self.epochs < 1:
            msg = f"epochs must be positive, got {self.epochs}"
            raise TrainingError(msg)
        if self.eval_every < 1:
            msg = f"eval_every must be positive, got {self.eval_every}"
            raise TrainingError(msg)
        if self.patience < 0:
            msg = f"patience must be non-negative, got {self.patience}"
            raise TrainingError(msg)
        if self.calibration not in CALIBRATION_METHODS:
            msg = f"calibration must be one of {CALIBRATION_METHODS}, got {self.calibration!r}"
            raise TrainingError(msg)
        if self.min_train_samples < 1:
            msg = f"min_train_samples must be positive, got {self.min_train_samples}"
            raise TrainingError(msg)
        if self.min_dev_samples < 2:
            msg = (
                f"min_dev_samples must be at least 2 to bound both error rates, "
                f"got {self.min_dev_samples}"
            )
            raise TrainingError(msg)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping."""
        return {
            "seed": self.seed,
            "epochs": self.epochs,
            "device": self.device,
            "eval_every": self.eval_every,
            "patience": self.patience,
            "verify_leakage": self.verify_leakage,
            "min_train_samples": self.min_train_samples,
            "min_dev_samples": self.min_dev_samples,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> TrainSpec:
        """Build from a parsed YAML mapping, rejecting unknown keys.

        Args:
            payload: The ``train`` block.

        Returns:
            A validated :class:`TrainSpec`.

        Raises:
            TrainingError: On an unknown key or a bad value.
        """
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            msg = f"unknown train option(s): {', '.join(unknown)}; known: {sorted(known)}"
            raise TrainingError(msg)
        return cls(**dict(payload))


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """The complete recipe for one baseline.

    Attributes:
        model_id: Identifier used for the artefact and every report citing it.
        features: Front-end settings.
        model: Architecture settings.
        train: Reproducibility and runtime settings.
        threshold: Operating-point selection rule, applied to dev only.
        train_split: Split fitted on. Must not be ``"test"``.
        dev_split: Split the operating point is chosen on. Must not be
            ``"test"``.
        test_split: Split reported. Never fitted or tuned on.
        output_dir: Directory for checkpoints and reports.
        notes: Caveats recorded in every artefact written under this
            configuration.
    """

    model_id: str = "mfcc_logreg"
    features: FeatureSpec = field(default_factory=FeatureSpec)
    model: ModelSpec = field(default_factory=ModelSpec)
    train: TrainSpec = field(default_factory=TrainSpec)
    threshold: ThresholdPolicy = field(default_factory=ThresholdPolicy)
    train_split: str = "train"
    dev_split: str = "dev"
    test_split: str = "test"
    output_dir: Path = Path("data/models")
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalise nested blocks, then validate identifiers and splits.

        Nested mappings are coerced here as well as in
        :func:`parse_training_config`, so that a direct
        ``TrainingConfig(model={"family": "logmel_cnn"})`` behaves the same as
        the YAML route. Without this, passing a mapping crashed with an opaque
        ``AttributeError: 'dict' object has no attribute 'family'`` from deep
        inside the validator, instead of the ``TrainingError`` that names the
        actual problem.
        """
        if isinstance(self.features, Mapping):
            object.__setattr__(self, "features", FeatureSpec.from_mapping(self.features))
        if isinstance(self.model, Mapping):
            object.__setattr__(self, "model", ModelSpec.from_mapping(self.model))
        if isinstance(self.train, Mapping):
            object.__setattr__(self, "train", TrainSpec.from_mapping(self.train))
        if isinstance(self.threshold, Mapping):
            object.__setattr__(self, "threshold", _parse_threshold(self.threshold))

        if not self.model_id or any(sep in self.model_id for sep in "/\\"):
            msg = (
                f"model_id must be non-empty and free of path separators, "
                f"got {self.model_id!r}; it becomes an artefact filename"
            )
            raise TrainingError(msg)
        if self.model.family != self.model_id and not self.model_id.startswith(self.model.family):
            msg = (
                f"model_id {self.model_id!r} does not describe family "
                f"{self.model.family!r}; an artefact whose name contradicts its "
                "own architecture is how the wrong model gets loaded later"
            )
            raise TrainingError(msg)
        self.validate_protocol()

    def validate_protocol(self) -> None:
        """Refuse a configuration that would train or tune on test.

        Raises:
            TrainingError: If any split violates the protocol. Called from
                :meth:`__post_init__`, so a bad configuration cannot be
                constructed and then accidentally used.
        """
        if self.train_split == self.test_split:
            msg = (
                f"train_split and test_split are both {self.test_split!r}; "
                "fitting on test makes every reported figure meaningless"
            )
            raise TrainingError(msg)
        if self.dev_split == self.test_split:
            msg = (
                f"dev_split and test_split are both {self.test_split!r}; "
                "choosing a threshold on test is not an evaluation"
            )
            raise TrainingError(msg)
        if self.train_split == self.dev_split:
            msg = (
                f"train_split and dev_split are both {self.train_split!r}; the "
                "operating point would be selected on data the model was fitted on"
            )
            raise TrainingError(msg)

    @property
    def threshold_policy(self) -> ThresholdPolicy:
        """The operating-point rule, for passing to the evaluation layer."""
        return self.threshold

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping, with the output directory as a string."""
        return {
            "model_id": self.model_id,
            "features": self.features.to_dict(),
            "model": self.model.to_dict(),
            "train": self.train.to_dict(),
            "threshold": self.threshold.to_dict(),
            "train_split": self.train_split,
            "dev_split": self.dev_split,
            "test_split": self.test_split,
            "output_dir": str(self.output_dir),
            "notes": list(self.notes),
        }

    def config_hash(self) -> str:
        """Stable hash of :meth:`to_dict`, recorded in every artefact.

        Deliberately includes :attr:`output_dir` and the seed, so two runs that
        differ only in where they wrote cannot be mistaken for the same run.
        A result is only comparable to another when this matches.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def content_fingerprint(self) -> str:
        """Hash of the settings that can change the fitted model.

        Excludes :attr:`output_dir`, mirroring
        :meth:`voxshield.data.config.DataConfig.content_fingerprint`: where a run
        wrote its artefact does not change what it learned, so two runs at
        different paths with identical settings share one fingerprint.
        """
        payload = self.to_dict()
        payload.pop("output_dir", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def parse_training_config(payload: Mapping[str, Any]) -> TrainingConfig:
    """Build and validate a configuration from a parsed mapping.

    Args:
        payload: Top-level mapping, with optional ``features``, ``model``,
            ``train``, and ``threshold`` blocks.

    Returns:
        A validated :class:`TrainingConfig`.

    Raises:
        TrainingError: On an unknown top-level key, an unknown option inside a
            block, or an invalid value. Unknown keys are errors rather than
            warnings so that a typo in a tracked YAML file fails the load
            instead of quietly reverting to a default.
    """
    known = {
        "model_id",
        "features",
        "model",
        "train",
        "threshold",
        "train_split",
        "dev_split",
        "test_split",
        "output_dir",
        "notes",
    }
    unknown = sorted(set(payload) - known)
    if unknown:
        msg = f"unknown configuration key(s): {', '.join(unknown)}; known: {sorted(known)}"
        raise TrainingError(msg)

    values = dict(payload)
    feature = FeatureSpec.from_mapping(values.pop("features", {}) or {})
    model = ModelSpec.from_mapping(values.pop("model", {}) or {})
    train = TrainSpec.from_mapping(values.pop("train", {}) or {})
    raw_threshold = values.pop("threshold", None)
    threshold = _parse_threshold(raw_threshold)
    values["features"] = feature
    values["model"] = model
    values["train"] = train
    values["threshold"] = threshold
    if "output_dir" in values and values["output_dir"] is not None:
        values["output_dir"] = Path(str(values["output_dir"]))
    if "notes" in values and values["notes"] is not None:
        values["notes"] = tuple(str(item) for item in values["notes"])
    return TrainingConfig(**values)


def _parse_threshold(payload: Mapping[str, Any] | None) -> ThresholdPolicy:
    """Build a :class:`ThresholdPolicy` from a mapping, rejecting unknown keys."""
    if payload is None:
        return ThresholdPolicy()
    known = {"strategy", "max_frr", "max_far", "fixed_threshold"}
    unknown = sorted(set(payload) - known)
    if unknown:
        msg = f"unknown threshold option(s): {', '.join(unknown)}; known: {sorted(known)}"
        raise TrainingError(msg)
    try:
        return ThresholdPolicy(**dict(payload))
    except ValueError as exc:
        raise TrainingError(str(exc)) from exc


def load_training_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> TrainingConfig:
    """Load a configuration from YAML and validate it.

    Args:
        path: A YAML document. ``None`` yields the built-in defaults, which is
            the configuration a caller gets when nothing is specified -- and is
            why the defaults are conservative rather than tuned.
        overrides: In-memory replacements applied after the document, for
            ``--seed``-style command-line changes.

    Returns:
        A validated :class:`TrainingConfig`.

    Raises:
        TrainingError: If the file is missing, is not a mapping, or fails
            validation.
    """
    payload: dict[str, Any] = {}
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            msg = f"training configuration not found: {candidate}"
            raise TrainingError(msg)
        try:
            import yaml

            loaded = yaml.safe_load(candidate.read_text(encoding="utf-8"))
        except ImportError as exc:  # pragma: no cover - environment problem
            msg = "PyYAML is required to read a training configuration"
            raise TrainingError(msg) from exc
        except Exception as exc:
            msg = f"{candidate} is not readable YAML: {exc}"
            raise TrainingError(msg) from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            msg = f"{candidate} must contain a mapping at the top level"
            raise TrainingError(msg)
        payload.update(dict(loaded))

    if overrides:
        payload.update(dict(overrides))
    return parse_training_config(payload)
