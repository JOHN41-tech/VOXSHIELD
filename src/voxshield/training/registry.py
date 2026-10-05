"""The model registry: which trained baselines exist on this machine, and which to use.

Training writes artefacts. Nothing, until now, could find them again. That gap is
where the quiet failures live:

* Two people ask "what does ``mfcc_logreg`` do?" and get two different
  directories, because the output root is a command-line default rather than
  recorded anywhere.
* A caller loads the newest artefact whose ``model_id`` looks right, and that
  happens to be the one trained with ``n_mels=40`` against the 80 the caller
  featurises with. Nothing raises. Every score is wrong by an amount that still
  looks like a probability.
* A registry that raised on the first missing directory would be unusable
  everywhere except on the machine that happened to train the models, which is
  the same mistake ``data.registry`` had to be written around.

So this mirrors ``data.registry`` and separates three concerns that are easy to
conflate:

* **Resolution** -- ``model_id`` resolves to an entry. An unknown identifier is a
  configuration error and raises, because there is no sensible fallback model.
* **Availability** -- an entry whose directory is absent, unreadable, or
  unloadable is *skipped with a reason*. Never an error, never silently omitted.
* **Compatibility** -- the front end is checked before anything is loaded. A
  feature-spec mismatch is refused loudly, because it is the failure that
  produces plausible numbers and is nearly impossible to diagnose downstream.

Every skip produces a :class:`SkipReason` rather than a bare string, so a report
can group them and a reader can tell "not on this machine" from "metadata
corrupt" -- two problems with two completely different fixes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from voxshield.training.artifacts import (
    ArtifactMetadata,
    ScoringBundle,
    load_artifact,
    load_scoring_bundle,
)
from voxshield.training.config import TrainingError
from voxshield.training.datasets import FeatureMatrix

__all__ = [
    "ModelRegistry",
    "ModelRegistryEntry",
    "ModelRegistryError",
    "SkipReason",
    "load_registry",
]


class ModelRegistryError(TrainingError):
    """The model registry is inconsistent -- duplicate, unknown, or incompatible entry."""


@dataclass(frozen=True, slots=True)
class ModelRegistryEntry:
    """One trained baseline the registry knows about.

    Attributes:
        model_id: Identifier, matching the artefact directory name.
        family: Model family.
        directory: Where the artefact lives.
        config_hash: Hash of the configuration that produced it.
        content_fingerprint: Hash of only the settings that change the weights,
            so two runs differing merely in output directory are recognised as
            the same model.
        calibration_method: Which calibrator was fitted on dev.
        threshold: The dev-selected operating point, or ``None``.
        n_train: Rows fitted.
        n_dev: Rows used for selection.
        test_evaluated: Whether the producing run scored a test split.
        feature_spec: Front-end settings, needed for the compatibility check.
    """

    model_id: str
    family: str
    directory: Path
    config_hash: str = ""
    content_fingerprint: str = ""
    calibration_method: str = "none"
    threshold: float | None = None
    n_train: int = 0
    n_dev: int = 0
    test_evaluated: bool = False
    feature_spec: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metadata(cls, metadata: ArtifactMetadata, directory: Path | str) -> ModelRegistryEntry:
        """Build an entry from an artefact's metadata.

        Args:
            metadata: Provenance loaded from the artefact.
            directory: Directory holding the artefact.

        Returns:
            The entry. The ``model_id`` comes from the directory name when the
            metadata carries one, so a renamed directory cannot masquerade as a
            different model.
        """
        return cls(
            model_id=metadata.model_id,
            family=metadata.family,
            directory=Path(directory),
            config_hash=metadata.config_hash,
            content_fingerprint=metadata.content_fingerprint,
            calibration_method=metadata.calibration_method,
            threshold=metadata.threshold,
            n_train=metadata.n_train,
            n_dev=metadata.n_dev,
            test_evaluated=metadata.test_evaluated,
            feature_spec=dict(metadata.feature_spec),
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready description.

        Returns:
            A mapping safe for ``json.dump``.
        """
        return {
            "model_id": self.model_id,
            "family": self.family,
            "directory": str(self.directory),
            "config_hash": self.config_hash,
            "content_fingerprint": self.content_fingerprint,
            "calibration_method": self.calibration_method,
            "threshold": self.threshold,
            "n_train": self.n_train,
            "n_dev": self.n_dev,
            "test_evaluated": self.test_evaluated,
            "feature_spec": dict(self.feature_spec),
        }

    def matches_fingerprint(self, fingerprint: str) -> bool:
        """Whether this entry is the model a fingerprint identifies.

        Args:
            fingerprint: A content fingerprint.

        Returns:
            ``True`` when the fingerprints agree. An entry that recorded no
            fingerprint never matches, so a legacy artefact is not silently
            treated as current.
        """
        return bool(self.content_fingerprint) and self.content_fingerprint == fingerprint


@dataclass(frozen=True, slots=True)
class SkipReason:
    """Why an entry could not be used.

    Attributes:
        model_id: The identifier it was filed under.
        reason: Machine-readable cause.
        detail: Human-readable explanation.
    """

    model_id: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        """JSON-ready description.

        Returns:
            A mapping safe for ``json.dump``.
        """
        return {"model_id": self.model_id, "reason": self.reason, "detail": self.detail}


def _entry_from_directory(directory: Path) -> tuple[ModelRegistryEntry | None, SkipReason | None]:
    """Read one directory into an entry, or explain why it cannot be.

    Args:
        directory: Candidate artefact directory.

    Returns:
        Either an entry, or a skip reason. Exactly one is ``None``.
    """
    model_id = directory.name
    try:
        _model, metadata = load_artifact(directory)
    except TrainingError as exc:
        return None, SkipReason(model_id, "unreadable", str(exc))
    except Exception as exc:  # a corrupt joblib file is not a TrainingError
        return None, SkipReason(model_id, "unreadable", f"could not load: {exc}")
    return ModelRegistryEntry.from_metadata(metadata, directory), None


class ModelRegistry:
    """The set of trained baselines visible under a root directory.

    Construction never fails for the ordinary reason that a model is missing.
    A machine that trained two of the six baselines is a normal machine, and the
    report needs to say which four are absent rather than refuse to start.

    Args:
        root: Directory scanned for artefacts.
        entries: Entries that loaded successfully.
        skipped: Entries that did not, with reasons.

    Attributes:
        root: The scanned directory.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        entries: Iterable[ModelRegistryEntry] = (),
        skipped: Iterable[SkipReason] = (),
    ) -> None:
        self.root = Path(root)
        self._entries: dict[str, ModelRegistryEntry] = {}
        self._skipped: dict[str, SkipReason] = {}
        for entry in entries:
            self.add(entry)
        for reason in skipped:
            self._skipped[reason.model_id] = reason

    def add(self, entry: ModelRegistryEntry, *, replace: bool = False) -> None:
        """File an entry.

        Args:
            entry: The entry to file.
            replace: Overwrite an entry already filed for the same identifier
                from a different directory.

        Raises:
            ModelRegistryError: A different directory already claims this
                identifier and ``replace`` is false. Two artefacts under one name
                is ambiguity, and picking one silently would make the registry's
                answer depend on directory order.
        """
        existing = self._entries.get(entry.model_id)
        if existing is not None and existing.directory != entry.directory and not replace:
            msg = (
                f"duplicate model_id {entry.model_id!r}: {existing.directory} and "
                f"{entry.directory} both claim it"
            )
            raise ModelRegistryError(msg)
        self._entries[entry.model_id] = entry

    @classmethod
    def discover(cls, root: Path | str) -> ModelRegistry:
        """Scan a directory for artefacts.

        Args:
            root: Directory whose immediate subdirectories may be artefacts.

        Returns:
            A registry holding every entry found, plus a skip reason for every
            subdirectory that looked like an artefact but was not one.
        """
        base = Path(root)
        entries: list[ModelRegistryEntry] = []
        skipped: list[SkipReason] = []
        if not base.is_dir():
            return cls(base, entries=entries, skipped=skipped)
        for directory in sorted(p for p in base.iterdir() if p.is_dir()):
            entry, reason = _entry_from_directory(directory)
            if entry is not None:
                entries.append(entry)
            elif reason is not None:
                # Only report directories that were plausibly meant to be
                # artefacts; a stray output folder should not pollute the report.
                if (directory / "metadata.json").is_file():
                    skipped.append(reason)
        return cls(base, entries=entries, skipped=skipped)

    def entries(self) -> tuple[ModelRegistryEntry, ...]:
        """Every known entry.

        Returns:
            Entries sorted by identifier, so output does not depend on
            filesystem order.
        """
        return tuple(self._entries[key] for key in sorted(self._entries))

    def skipped(self) -> tuple[SkipReason, ...]:
        """Every entry that could not be used.

        Returns:
            Reasons sorted by identifier.
        """
        return tuple(self._skipped[key] for key in sorted(self._skipped))

    def families(self) -> tuple[str, ...]:
        """Every family represented.

        Returns:
            Distinct family names, sorted.
        """
        return tuple(sorted({entry.family for entry in self._entries.values()}))

    def __len__(self) -> int:
        """The number of usable entries.

        Returns:
            Entry count, excluding skips.
        """
        return len(self._entries)

    def __contains__(self, model_id: object) -> bool:
        """Whether an identifier is known.

        Args:
            model_id: Identifier to test.

        Returns:
            ``True`` when the identifier resolves to an entry.
        """
        return model_id in self._entries

    def __iter__(self) -> Iterator[ModelRegistryEntry]:
        """Iterate entries in identifier order.

        Yields:
            Each entry.
        """
        return iter(self.entries())

    def get(self, model_id: str) -> ModelRegistryEntry:
        """Resolve an identifier to its entry.

        Args:
            model_id: Identifier to resolve.

        Returns:
            The entry.

        Raises:
            ModelRegistryError: The identifier is unknown. Raised rather than
                defaulted, because silently substituting a different model is
                the failure this registry exists to prevent. The message names
                the alternatives that do exist.
        """
        try:
            return self._entries[model_id]
        except KeyError:
            known = ", ".join(sorted(self._entries)) or "none"
            msg = f"unknown model_id {model_id!r}; the registry holds: {known}"
            raise ModelRegistryError(msg) from None

    def find_by_fingerprint(self, fingerprint: str) -> ModelRegistryEntry | None:
        """Find the entry a content fingerprint identifies.

        Args:
            fingerprint: A content fingerprint from a configuration.

        Returns:
            The matching entry, or ``None`` when nothing matches.
        """
        for entry in self.entries():
            if entry.matches_fingerprint(fingerprint):
                return entry
        return None

    def compatible_with(
        self, model_id: str, feature_spec: dict[str, Any] | None
    ) -> ModelRegistryEntry:
        """Resolve an entry and check its front end against the caller's.

        Args:
            model_id: Identifier to resolve.
            feature_spec: The feature settings the caller will featurise with.

        Returns:
            The entry.

        Raises:
            ModelRegistryError: The identifier is unknown, or the entry was
                trained with different feature settings. The mismatch is refused
                because featurising a 40-band model with 80 bands produces a
                score that looks entirely reasonable and is meaningless.
        """
        entry = self.get(model_id)
        if feature_spec is None or not entry.feature_spec:
            return entry
        if dict(feature_spec) != entry.feature_spec:
            differing = sorted(
                key
                for key in set(feature_spec) | set(entry.feature_spec)
                if feature_spec.get(key) != entry.feature_spec.get(key)
            )
            summary = ", ".join(
                f"{key}: trained {entry.feature_spec.get(key)!r} vs requested "
                f"{feature_spec.get(key)!r}"
                for key in differing
            )
            msg = (
                f"model {model_id!r} was trained with a different front end ({summary}). "
                "Refusing to load: the scores would be meaningless."
            )
            raise ModelRegistryError(msg)
        return entry

    def load(self, model_id: str) -> ScoringBundle:
        """Load an entry ready to score.

        Args:
            model_id: Identifier to resolve.

        Returns:
            A bundle carrying the model, its calibrator, and its threshold.

        Raises:
            ModelRegistryError: The identifier is unknown or the artefact cannot
                be loaded.
        """
        entry = self.get(model_id)
        try:
            return load_scoring_bundle(entry.directory)
        except TrainingError as exc:
            msg = f"model {model_id!r} at {entry.directory} could not be loaded: {exc}"
            raise ModelRegistryError(msg) from exc

    def score(
        self,
        model_id: str,
        frames: FeatureMatrix,
        *,
        feature_spec: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Score with a registered model.

        Args:
            model_id: Identifier to resolve.
            frames: Raw model inputs.
            feature_spec: Front-end settings the caller featurised with, checked
                against the artefact before anything is loaded.

        Returns:
            Calibrated probabilities.

        Raises:
            ModelRegistryError: The identifier is unknown, the front end does
                not match, or the artefact cannot be loaded.
        """
        self.compatible_with(model_id, feature_spec)
        return self.load(model_id).score(frames)

    def register(self, directory: Path | str, *, replace: bool = False) -> ModelRegistryEntry:
        """Add an artefact from outside the scanned root.

        Useful when artefacts live somewhere other than the default output
        directory, which is common once a run moves to shared storage.

        Args:
            directory: Artefact directory.
            replace: Overwrite an existing entry for the same identifier.

        Returns:
            The registered entry.

        Raises:
            ModelRegistryError: The artefact cannot be read, or the identifier is
                already registered and ``replace`` is false.
        """
        path = Path(directory)
        entry, reason = _entry_from_directory(path)
        if entry is None:
            detail = reason.detail if reason is not None else "unknown"
            msg = f"cannot register {path}: {detail}"
            raise ModelRegistryError(msg)
        if entry.model_id in self._entries and not replace:
            msg = (
                f"model_id {entry.model_id!r} is already registered at "
                f"{self._entries[entry.model_id].directory}; pass replace=True to override"
            )
            raise ModelRegistryError(msg)
        self.add(entry, replace=replace)
        return entry

    def report(self) -> dict[str, Any]:
        """JSON-ready summary.

        Returns:
            A mapping describing the root, the usable entries, and the skips
            with their reasons. Safe to write to a run record.
        """
        return {
            "root": str(self.root),
            "count": len(self._entries),
            "families": list(self.families()),
            "entries": [entry.to_dict() for entry in self.entries()],
            "skipped": [reason.to_dict() for reason in self.skipped()],
        }


def load_registry(root: Path | str) -> ModelRegistry:
    """Scan a root directory for trained baselines.

    Args:
        root: Directory whose immediate subdirectories may be artefacts.

    Returns:
        The populated registry.
    """
    return ModelRegistry.discover(root)


def load_artifact_metadata(directory: Path | str) -> ArtifactMetadata:
    """Read one artefact's provenance without loading its weights.

    Args:
        directory: Artefact directory.

    Returns:
        The metadata.

    Raises:
        ModelRegistryError: The artefact cannot be read.
    """
    try:
        _model, metadata = load_artifact(directory)
    except TrainingError as exc:
        msg = f"{directory} could not be read: {exc}"
        raise ModelRegistryError(msg) from exc
    return metadata
