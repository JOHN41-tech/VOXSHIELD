"""Checkpoint and provenance serialisation for the Phase 3 baselines.

An artefact is a directory, not a file, and it exists so that a score can be
traced back to the exact things that produced it:

.. code-block:: text

    <output_dir>/<model_id>/
        metadata.json     provenance: config hash, dataset build, counts, fit facts
        model.joblib      the two tabular baselines (scaler + estimator)
        weights.pt        the CNN state dict
        state.json        the CNN architecture, needed to rebuild it

Two decisions are worth stating because they are the ones that go wrong:

**The scaler travels with the estimator.** A saved logistic regression without
its :class:`~sklearn.preprocessing.StandardScaler` still loads, still scores, and
produces numbers that are simply wrong, because the coefficients were fitted
against standardised inputs. There is no error to notice.

**Metadata records what did not happen.** ``selection_metric`` is ``"none"``
when no selection occurred, and ``test_evaluated`` is ``False`` unless a test
evaluation was actually run. A field that is always populated is a field nobody
reads.

Artefacts round-trip: :func:`save_artifact` followed by :func:`load_artifact`
returns a fitted model that reproduces the original scores. That property is
tested, because an artefact that loads but scores differently is worse than one
that fails to load.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from voxshield.evaluation.metrics import ThresholdPolicy
from voxshield.training.calibration import Calibrator, calibrator_from_dict
from voxshield.training.config import ModelSpec, TrainingConfig, TrainingError
from voxshield.training.datasets import FeatureMatrix
from voxshield.training.models import BaselineModel, FitSummary

__all__ = [
    "ArtifactMetadata",
    "ScoringBundle",
    "artifact_dir",
    "load_artifact",
    "load_scoring_bundle",
    "save_artifact",
]

METADATA_NAME = "metadata.json"
JOBLIB_NAME = "model.joblib"
WEIGHTS_NAME = "weights.pt"
STATE_NAME = "state.json"

#: Bumped when the on-disk layout changes in a way that older readers cannot
#: understand. A mismatch is refused rather than best-effort parsed.
ARTIFACT_FORMAT_VERSION = 1


@dataclass(slots=True)
class ArtifactMetadata:
    """Provenance for one trained baseline.

    Attributes:
        model_id: Which configuration produced this model.
        family: Model family, duplicated for reader convenience.
        format_version: On-disk layout version.
        config_hash: Hash of the complete configuration.
        content_fingerprint: Hash of the settings that can change the fitted
            model, excluding ``output_dir``.
        feature_spec: The front-end settings as a mapping. Recorded because a
            saved model is only useful if whoever loads it can reproduce the
            features the model was fitted on, and a reader cannot guess that
            ``n_mels`` was 80.
        train_split: Split name used for fitting.
        dev_split: Split name used for selection, or ``None``.
        n_train: Rows fitted on.
        n_dev: Rows used for selection.
        n_features: Pooled vector width, or ``None`` for the CNN.
        n_frames: Frames per clip for the CNN, or ``None``.
        n_bands: Mel bands per frame for the CNN, or ``None``.
        class_counts: Encoded label to row count for the training split.
        rows_considered: Manifest rows considered for the training split.
        failed_decodes: Sample ID and reason for every row that could not be
            decoded. Recorded rather than discarded: a corpus with a 4% decode
            failure rate looks smaller than it is.
        selection_metric: Metric that drove selection, or ``"none"``.
        selection_value: Its value, or ``None``.
        calibration_method: Which calibrator was fitted on dev, or ``"none"``.
        calibration: Fitted calibrator parameters, or ``None``.
        threshold: The dev-selected operating point, or ``None``. Carried
            because a probability is only meaningful next to the threshold that
            turns it into a decision; a rescoring that drops it silently reports
            a different operating point than the run that produced the model.
        threshold_policy: The rule that chose :attr:`threshold`.
        epochs_run: Epochs or boosting rounds actually executed.
        best_epoch: Selected epoch or round, or ``None``.
        notes: Fit notes and configuration notes.
        test_evaluated: Whether a test split was ever scored. ``False`` for every
            artefact written by a training run.
        libraries: Versions of the libraries that produced the weights.
        python: Interpreter version.
        platform: Machine string.
        device: Device the fit ran on.
    """

    model_id: str
    family: str
    format_version: int = ARTIFACT_FORMAT_VERSION
    config_hash: str = ""
    content_fingerprint: str = ""
    feature_spec: dict[str, Any] = field(default_factory=dict)
    train_split: str = "train"
    dev_split: str | None = "dev"
    n_train: int = 0
    n_dev: int = 0
    n_features: int | None = None
    n_frames: int | None = None
    n_bands: int | None = None
    class_counts: dict[str, int] = field(default_factory=dict)
    rows_considered: int = 0
    failed_decodes: tuple[dict[str, str], ...] = ()
    selection_metric: str = "none"
    selection_value: float | None = None
    calibration_method: str = "none"
    calibration: dict[str, Any] | None = None
    threshold: float | None = None
    threshold_policy: str | None = None
    epochs_run: int = 0
    best_epoch: int | None = None
    notes: tuple[str, ...] = ()
    test_evaluated: bool = False
    libraries: dict[str, str] = field(default_factory=dict)
    python: str = ""
    platform: str = ""
    device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready mapping.

        Returns:
            A mapping with tuples converted to lists, safe for
            :func:`json.dump`.
        """
        payload = asdict(self)
        payload["failed_decodes"] = [dict(entry) for entry in self.failed_decodes]
        payload["notes"] = list(self.notes)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ArtifactMetadata:
        """Rebuild metadata from a mapping.

        Args:
            payload: A mapping produced by :meth:`to_dict`.

        Returns:
            The reconstructed metadata.

        Raises:
            TrainingError: If required fields are missing or the format version
                is not readable.
        """
        version = payload.get("format_version")
        if version != ARTIFACT_FORMAT_VERSION:
            msg = (
                f"artefact format version {version!r} cannot be read by this "
                f"build, which writes version {ARTIFACT_FORMAT_VERSION}"
            )
            raise TrainingError(msg)
        known = set(cls.__slots__)
        extra = sorted(set(payload) - known)
        if extra:
            msg = f"artefact metadata has unknown field(s): {', '.join(extra)}"
            raise TrainingError(msg)
        data = dict(payload)
        data["failed_decodes"] = tuple(dict(entry) for entry in payload.get("failed_decodes", ()))
        data["notes"] = tuple(payload.get("notes", ()))
        return cls(**data)


def artifact_dir(output_dir: Path | str, model_id: str) -> Path:
    """Resolve the artefact directory for a model identifier.

    Args:
        output_dir: Root output directory.
        model_id: Model identifier, used as the directory name.

    Returns:
        The artefact directory, which need not exist yet.
    """
    return Path(output_dir) / model_id


def _library_versions() -> dict[str, str]:
    """Collect versions of every library that can influence the weights.

    Returns:
        A mapping of package name to version. Absent packages are omitted rather
        than recorded as ``None``, so a missing entry is unambiguous.
    """
    versions: dict[str, str] = {}
    for name in ("numpy", "scipy", "sklearn", "xgboost", "torch"):
        try:
            module = __import__(name)
        except ImportError:
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _decode_entries(entries: tuple[tuple[str, str], ...]) -> tuple[dict[str, str], ...]:
    """Convert sample/reason pairs into self-describing JSON entries.

    The pair form matches :attr:`FeatureMatrix.failed`, which is where these come
    from. The dictionary form is what lands on disk, so that reading an artefact
    does not require remembering which position is the sample and which is the
    reason.

    Args:
        entries: ``(sample_id, reason)`` pairs.

    Returns:
        One ``{"sample_id": ..., "reason": ...}`` mapping per pair.

    Raises:
        TrainingError: If an entry is not a two-element pair. Unpacking it
            directly would raise an opaque "too many values to unpack" from
            inside a generator, which reads like a bug in this function.
    """
    converted: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            msg = (
                "failed_decodes entries must be (sample_id, reason) pairs; got "
                f"{entry!r}. Pass the tuples from FeatureMatrix.failed unchanged."
            )
            raise TrainingError(msg)
        converted.append({"sample_id": str(entry[0]), "reason": str(entry[1])})
    return tuple(converted)


def build_metadata(
    config: TrainingConfig,
    model: BaselineModel,
    summary: FitSummary,
    *,
    n_features: int | None = None,
    n_frames: int | None = None,
    n_bands: int | None = None,
    class_counts: dict[int, int] | None = None,
    rows_considered: int = 0,
    failed_decodes: tuple[tuple[str, str], ...] = (),
    device: str = "cpu",
    calibrator: Calibrator | None = None,
    threshold: float | None = None,
    threshold_policy: ThresholdPolicy | None = None,
    notes: Sequence[str] = (),
) -> ArtifactMetadata:
    """Assemble provenance for a fitted model.

    Args:
        config: The configuration that produced the model.
        model: The fitted model.
        summary: What the fit did.
        n_features: Pooled vector width.
        n_frames: Frames per clip, for the CNN.
        n_bands: Mel bands per frame, for the CNN.
        class_counts: Encoded label counts for the training split.
        rows_considered: Manifest rows considered for the training split.
        failed_decodes: Sample ID and reason per undecodable row.
        device: Device the fit ran on.
        calibrator: Calibrator fitted on dev, stored so rescoring reproduces the
            same probabilities rather than merely the same ranking.
        threshold: Dev-selected operating point.
        threshold_policy: Rule that chose the threshold.
        notes: Provenance the fit summary does not carry. The calibration
            fallback note lives here: without it an artefact recording
            ``calibration_method="none"`` cannot be told apart from one where the
            recipe asked for calibration and the dev split was too small to
            support it, which is the silent-degradation case this whole feature
            exists to prevent.

    Returns:
        Populated metadata.
    """
    return ArtifactMetadata(
        model_id=config.model_id,
        family=config.model.family,
        config_hash=config.config_hash(),
        content_fingerprint=config.content_fingerprint(),
        feature_spec=config.features.to_dict(),
        train_split=config.train_split,
        dev_split=config.dev_split,
        n_train=summary.n_train,
        n_dev=summary.n_dev,
        n_features=n_features,
        n_frames=n_frames,
        n_bands=n_bands,
        class_counts={str(key): int(value) for key, value in (class_counts or {}).items()},
        rows_considered=int(rows_considered),
        failed_decodes=_decode_entries(failed_decodes),
        selection_metric=summary.selection_metric,
        selection_value=summary.selection_value,
        calibration_method=calibrator.method if calibrator is not None else "none",
        calibration=calibrator.to_dict() if calibrator is not None else None,
        threshold=None if threshold is None else float(threshold),
        threshold_policy=None if threshold_policy is None else str(threshold_policy),
        epochs_run=summary.epochs_run,
        best_epoch=summary.best_epoch,
        notes=tuple(note for note in (*config.notes, *summary.notes, *notes) if note),
        test_evaluated=False,
        libraries=_library_versions(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        device=device,
    )


def save_artifact(
    model: BaselineModel,
    metadata: ArtifactMetadata,
    output_dir: Path | str,
) -> Path:
    """Write a fitted model and its provenance to disk.

    Args:
        model: The fitted model.
        metadata: Provenance to record beside it.
        output_dir: Root output directory.

    Returns:
        The artefact directory.

    Raises:
        TrainingError: If the model is unfitted, or the write fails.
    """
    if not model.fitted:
        msg = f"refusing to save an unfitted {type(model).__name__}"
        raise TrainingError(msg)

    directory = artifact_dir(output_dir, metadata.model_id)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / METADATA_NAME).write_text(
            json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        msg = f"could not write the artefact directory {directory}: {exc}"
        raise TrainingError(msg) from exc

    family = metadata.family
    try:
        if family == "logmel_cnn":
            _save_cnn(model, metadata, directory)
        else:
            _save_sklearn(model, directory)
    except TrainingError:
        raise
    except Exception as exc:
        msg = f"could not serialise the {family} model: {exc}"
        raise TrainingError(msg) from exc
    return directory


def _save_sklearn(model: BaselineModel, directory: Path) -> None:
    """Write the scaler and estimator together.

    joblib is used because it is scikit-learn's own persistence format, so a
    saved model loads with the version of scikit-learn that wrote it.

    Args:
        model: A fitted tabular baseline.
        directory: Destination directory.

    Raises:
        TrainingError: If the model is not a tabular baseline.
    """
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - environment problem
        msg = "saving a tabular baseline needs joblib, which ships with scikit-learn"
        raise TrainingError(msg) from exc

    scaler = getattr(model, "_scaler", None)
    estimator = getattr(model, "_estimator", None)
    if scaler is None or estimator is None:
        msg = f"{type(model).__name__} has no fitted scaler/estimator pair to save"
        raise TrainingError(msg)
    # Saved as one object so that they cannot be separated on disk. A file pair
    # invites loading the estimator against the wrong scaler.
    joblib.dump({"scaler": scaler, "estimator": estimator}, directory / JOBLIB_NAME)


def _save_cnn(model: BaselineModel, metadata: ArtifactMetadata, directory: Path) -> None:
    """Write the CNN state dict and the architecture needed to rebuild it.

    Args:
        model: A fitted CNN.
        metadata: Provenance, which carries the frame geometry.
        directory: Destination directory.

    Raises:
        TrainingError: If the network is missing or torch is unavailable.
    """
    import torch

    network = getattr(model, "_network", None)
    if network is None:
        msg = f"{type(model).__name__} has no fitted network to save"
        raise TrainingError(msg)
    torch.save(network.state_dict(), directory / WEIGHTS_NAME)
    state = {
        "n_frames": metadata.n_frames,
        "n_bands": metadata.n_bands,
        "hidden_channels": list(model.spec.hidden_channels),
        "dropout": model.spec.dropout,
    }
    (directory / STATE_NAME).write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


@dataclass(frozen=True, slots=True)
class ScoringBundle:
    """Everything needed to turn raw model scores into a calibrated decision.

    Grouped because these three are only correct together. Loading the model
    without its calibrator yields plausible probabilities that are not the ones
    the model was selected with, and loading without the threshold yields no
    decision at all. Bundling them makes the incomplete version awkward to write.

    Attributes:
        model: The fitted model.
        calibrator: The dev-fitted calibrator, or an identity calibrator for an
            artefact written before calibration existed.
        threshold: The dev-selected operating point, or ``None`` when the
            artefact never recorded one.
        metadata: Full provenance.
    """

    model: BaselineModel
    calibrator: Calibrator
    threshold: float | None
    metadata: ArtifactMetadata

    def score(self, frames: FeatureMatrix) -> np.ndarray:
        """Score, calibrate, and threshold in one call.

        Args:
            frames: Raw model inputs.

        Returns:
            Calibrated probabilities, before thresholding. Thresholding is left
            to the caller so the rates remain computable.
        """
        return np.asarray(self.calibrator.transform(self.model.predict_proba(frames)))

    def decision(self, frames: FeatureMatrix) -> np.ndarray:
        """Decide using the recorded operating point.

        Args:
            frames: Raw model inputs.

        Returns:
            Boolean spoof decisions. When no threshold was recorded, every
            decision is ``False`` rather than an arbitrary guess.
        """
        scores = self.score(frames)
        if self.threshold is None:
            return np.zeros(scores.shape, dtype=bool)
        return scores >= self.threshold


def load_scoring_bundle(path: Path | str) -> ScoringBundle:
    """Load a model together with the calibration that makes its scores mean something.

    Args:
        path: An artefact directory.

    Returns:
        The bundle, ready to score.

    Raises:
        TrainingError: If the artefact cannot be loaded.
    """
    model, metadata = load_artifact(path)
    return ScoringBundle(
        model=model,
        calibrator=calibrator_from_dict(metadata.calibration),
        threshold=metadata.threshold,
        metadata=metadata,
    )


def load_artifact(path: Path | str) -> tuple[BaselineModel, ArtifactMetadata]:
    """Load a fitted model and its provenance.

    Args:
        path: An artefact directory.

    Returns:
        The fitted model and its metadata.

    Raises:
        TrainingError: If the directory is incomplete, unreadable, or holds no
            recognisable model.
    """
    directory = Path(path)
    metadata_path = directory / METADATA_NAME
    if not metadata_path.is_file():
        msg = f"{directory} has no {METADATA_NAME}; it is not a VoxShield artefact"
        raise TrainingError(msg)
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"{metadata_path} is unreadable: {exc}"
        raise TrainingError(msg) from exc

    metadata = ArtifactMetadata.from_dict(payload)
    try:
        model = _rebuild(metadata, directory)
    except TrainingError:
        raise
    except Exception as exc:
        msg = f"could not reconstruct the model in {directory}: {exc}"
        raise TrainingError(msg) from exc
    return model, metadata


def _rebuild(metadata: ArtifactMetadata, directory: Path) -> BaselineModel:
    """Reconstruct a fitted model from an artefact directory.

    Args:
        metadata: The artefact's provenance.
        directory: The artefact directory.

    Returns:
        A fitted model.

    Raises:
        TrainingError: If a required file is missing or the family is unknown.
    """
    spec = _spec_from_metadata(metadata)
    family = metadata.family

    if family == "logmel_cnn":
        return _rebuild_cnn(spec, metadata, directory)
    if family in {"mfcc_logreg", "mfcc_xgboost"}:
        return _rebuild_sklearn(spec, family, directory)

    msg = f"artefact names unknown model family {family!r}"
    raise TrainingError(msg)


def _spec_from_metadata(metadata: ArtifactMetadata) -> ModelSpec:
    """Rebuild the architecture settings recorded for a model.

    Only the CNN's architecture has to be reconstructed exactly, because its
    layer shapes are baked into the state dict. For the tabular models the spec
    is not needed to *load* weights, so a default is sufficient.

    Args:
        metadata: The artefact's provenance.

    Returns:
        A :class:`~voxshield.training.config.ModelSpec`.
    """
    return ModelSpec(
        family=metadata.family,
        hidden_channels=(32, 64),
    )


def _rebuild_sklearn(spec: ModelSpec, family: str, directory: Path) -> BaselineModel:
    """Restore a tabular baseline from its scaler/estimator pair.

    Args:
        spec: Architecture settings.
        family: Either ``mfcc_logreg`` or ``mfcc_xgboost``.
        directory: The artefact directory.

    Returns:
        A fitted baseline.

    Raises:
        TrainingError: If joblib or the artefact file is unavailable, or if the
            estimator family does not match the artefact.
    """
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - environment problem
        msg = "loading a tabular baseline needs joblib, which ships with scikit-learn"
        raise TrainingError(msg) from exc

    path = directory / JOBLIB_NAME
    if not path.is_file():
        msg = f"{directory} has no {JOBLIB_NAME}; the {family} weights are missing"
        raise TrainingError(msg)
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "scaler" not in payload or "estimator" not in payload:
        msg = f"{path} does not hold a scaler/estimator pair"
        raise TrainingError(msg)

    from voxshield.training.models import LogisticBaseline, XGBoostBaseline

    if family == "mfcc_logreg":
        model: LogisticBaseline | XGBoostBaseline = LogisticBaseline(spec)
    else:
        model = XGBoostBaseline(spec)
    estimator = payload["estimator"]
    expected = "LogisticRegression" if family == "mfcc_logreg" else "XGBClassifier"
    if type(estimator).__name__ != expected:
        msg = (
            f"{path} holds a {type(estimator).__name__}, but the artefact "
            f"declares family {family!r} and expects {expected}"
        )
        raise TrainingError(msg)
    model._scaler = payload["scaler"]
    model._estimator = estimator
    model._fitted = True
    return model


def _rebuild_cnn(spec: ModelSpec, metadata: ArtifactMetadata, directory: Path) -> BaselineModel:
    """Restore the CNN from its state dict and recorded architecture.

    Args:
        spec: Architecture settings.
        metadata: The artefact's provenance.
        directory: The artefact directory.

    Returns:
        A fitted CNN.

    Raises:
        TrainingError: If torch, the state dict, or the architecture is missing.
    """
    state_path = directory / STATE_NAME
    weights_path = directory / WEIGHTS_NAME
    if not state_path.is_file() or not weights_path.is_file():
        msg = (
            f"{directory} is missing {STATE_NAME} or {WEIGHTS_NAME}; the CNN weights are incomplete"
        )
        raise TrainingError(msg)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    n_frames = state.get("n_frames") or metadata.n_frames
    n_bands = state.get("n_bands") or metadata.n_bands
    if not n_frames or not n_bands:
        msg = f"{state_path} does not record the frame geometry, so it cannot be rebuilt"
        raise TrainingError(msg)

    import torch

    from voxshield.training.models import LogMelCNN

    spec = ModelSpec(
        family="logmel_cnn",
        hidden_channels=tuple(state.get("hidden_channels", (32, 64))),
        dropout=float(state.get("dropout", 0.2)),
    )
    model = LogMelCNN(spec, n_frames=int(n_frames), n_bands=int(n_bands))
    network = model._build_network()
    network.load_state_dict(torch.load(weights_path, weights_only=True))
    network.eval()
    model._network = network
    model._fitted = True
    return model


def scores_match(
    left: BaselineModel, right: BaselineModel, data: Any, *, tolerance: float = 1e-9
) -> bool:
    """Compare two models' scores on the same data.

    Used by the round-trip test to assert that loading an artefact reproduces the
    original model's output, which is the property that makes an artefact useful
    at all.

    Args:
        left: First model.
        right: Second model.
        data: A :class:`~voxshield.training.datasets.FeatureMatrix`.
        tolerance: Maximum absolute difference per row.

    Returns:
        Whether the scores agree within ``tolerance``.
    """
    first = np.asarray(left.predict_proba(data), dtype=np.float64)
    second = np.asarray(right.predict_proba(data), dtype=np.float64)
    if first.shape != second.shape:
        return False
    return bool(np.all(np.abs(first - second) <= tolerance))
