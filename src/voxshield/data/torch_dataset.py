"""The training-facing loader: manifest rows in, model batches out.

A manifest is the only thing a trainer is allowed to read. This module is the
single place where a manifest row becomes a tensor, so the invariants are
enforced once rather than in every training script:

* audio is read lazily, per item, so a corpus larger than memory still loads;
* augmentation is applied to ``train`` and to nothing else. The decision is made
  from the row's split rather than from a flag the caller passes, so a training
  script that accidentally reuses its train dataset for evaluation gets clean
  audio back instead of an augmented test set;
* augmentation seeds derive from ``(base seed, sample id, epoch)``, so a run is
  reproducible and successive epochs are not identical;
* class balancing is done by a sampler, never by duplicating rows. A duplicated
  manifest row would be counted twice in the dataset statistics and would defeat
  the duplicate check that guards the corpus.

Torch is an optional dependency, so this module refuses to import without it
and says which extra installs it. The guard lives on the module rather than on
:func:`build_dataloader`: a caller that got this far wants tensors, and failing
at the first ``__getitem__`` with a missing attribute would be a worse error
than failing at the import that introduced the dependency. Importing
``voxshield.data`` does not import this module, so ingestion, manifest writing,
and the CLI's non-ML commands never pay for Torch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voxshield.data.augment import AugmentationConfig, augment, class_weights
from voxshield.data.config import DataConfig
from voxshield.data.manifest import Manifest, read_manifest
from voxshield.data.paths import DataPaths
from voxshield.data.schema import SampleRecord
from voxshield.errors import AudioDecodeError

_MISSING_TORCH = (
    "PyTorch is required for the dataset loader. Install it with "
    "'pip install voxshield[ml]'. Ingestion, manifest writing, and the CLI's "
    "non-ML commands do not need it, and none of them import this module."
)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise ImportError(_MISSING_TORCH) from exc

SPLITS = ("train", "dev", "test")


def resolve_split(value: str) -> str:
    """Validate a split name.

    Args:
        value: Candidate split.

    Returns:
        The split, unchanged.

    Raises:
        ValueError: The name is not a known split.
    """
    if value not in SPLITS:
        msg = f"split must be one of {SPLITS}, got {value!r}"
        raise ValueError(msg)
    return value


def read_segment(path: str | Path, *, expected_samples: int | None = None) -> np.ndarray:
    """Read one stored segment as mono float32.

    Segments are written canonically by Phase 1, so this is a plain read of a
    known-good file rather than a decode of untrusted input.

    Args:
        path: Segment location.
        expected_samples: Length from the manifest row, checked when given. A
            manifest that disagrees with the file on disk is a corrupt build, and
            a batch silently padded to the wrong width trains on audio the
            statistics never described.

    Returns:
        Float32 mono samples.

    Raises:
        AudioDecodeError: The file is missing, unreadable, or the wrong length.
    """
    location = Path(path)
    try:
        data, rate = sf.read(location, dtype="float32", always_2d=True)
    except Exception as exc:
        msg = f"could not read segment {location}: {exc}"
        raise AudioDecodeError(msg) from exc
    if rate <= 0:  # pragma: no cover - libsndfile rejects this first
        msg = f"segment {location} reports a sample rate of {rate}"
        raise AudioDecodeError(msg)
    mono = data.mean(axis=1, dtype=np.float32)
    if expected_samples is not None and mono.size != expected_samples:
        msg = (
            f"segment {location} holds {mono.size} samples but its manifest row "
            f"records {expected_samples}"
        )
        raise AudioDecodeError(msg)
    return np.ascontiguousarray(mono, dtype=np.float32)


class VoxShieldDataset(Dataset):
    """Map-style dataset over one manifest split.

    Args:
        manifest: Loaded manifest, or a path to one. Defaults to
            ``<paths.manifests>/<split>.jsonl``.
        split: Split to read.
        paths: Data paths, used to resolve the manifest's relative paths.
        config: Build configuration, for the augmentation and balance settings.
        augment: Overrides the config's augmentation switch. ``False`` disables
            augmentation whatever the config says; ``True`` enables it for
            ``train`` only.
        epoch: Starting epoch, for reproducible-but-varying augmentation seeds.
        min_coverage: Drop windows whose speech coverage is below this. Coverage
            is recorded in the manifest precisely so a trainer can filter quiet
            windows without re-deriving VAD.
        include_padded: Keep zero-padded final windows. A padded window is real
            audio plus silence; dropping it removes the tail of every recording
            whose length is not a whole number of windows.
        root: Data root override, for a manifest read from outside the tree it
            was built in.
    """

    def __init__(
        self,
        manifest: Manifest | str | Path | None = None,
        *,
        split: str = "train",
        paths: DataPaths | None = None,
        config: DataConfig | None = None,
        augment: bool | None = None,
        epoch: int = 0,
        min_coverage: float = 0.0,
        include_padded: bool = True,
        root: str | Path | None = None,
    ) -> None:
        self.paths = paths if paths is not None else DataPaths.resolve("data")
        resolve_split(split)
        if manifest is None:
            manifest = self.paths.manifests / f"{split}.jsonl"
        self.manifest = read_manifest(manifest) if isinstance(manifest, str | Path) else manifest
        self.split = split
        self.root = Path(root) if root is not None else self.paths.root
        self.epoch = int(epoch)
        self.min_coverage = float(min_coverage)
        self.include_padded = bool(include_padded)

        self.rows: tuple[SampleRecord, ...] = tuple(
            row
            for row in self.manifest.by_split(split)
            if row.coverage >= self.min_coverage and (self.include_padded or not row.is_padded)
        )

        settings = config.augmentation if config is not None else None
        if augment is not None and settings is not None:
            settings = replace(settings, enabled=bool(augment))
        # The dataset, not the caller, decides what may be augmented. A config
        # that enables augmentation cannot make a dev or test window noisy.
        self.augmentation: AugmentationConfig | None = (
            settings if (settings is not None and split == "train") else None
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(split={self.split!r}, rows={len(self.rows)}, "
            f"augmentation={'on' if self.augmentation else 'off'})"
        )

    def set_epoch(self, epoch: int) -> VoxShieldDataset:
        """Select the augmentation epoch and return self.

        Call once per epoch, before iterating. Successive epochs see different
        augmentation of the same stored audio, and re-running a given epoch
        reproduces it exactly.

        Args:
            epoch: Epoch index.

        Returns:
            Self, so the call can be chained onto construction.
        """
        self.epoch = int(epoch)
        return self

    def locate(self, path: str) -> Path:
        """Resolve a manifest-relative path against the data root.

        Args:
            path: Relative path as stored in the manifest.

        Returns:
            An absolute path. Absolute paths are returned unchanged, so a
            relocated corpus is still readable.
        """
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one item.

        Args:
            index: Row index.

        Returns:
            A dict with ``waveform`` (float tensor, shape ``[1, T]``),
            ``label`` (long tensor), ``sample_id``, ``length``, ``is_padded``,
            ``coverage``, ``augmented``, and ``augment_seed``.

        Raises:
            AudioDecodeError: The segment is missing, unreadable, or the wrong
                length. Raised rather than skipped, so a truncated corpus fails
                at the first bad batch instead of quietly training on less data
                than the statistics describe.
        """
        row = self.rows[index]
        waveform = read_segment(self.locate(row.audio_path), expected_samples=row.waveform_samples)
        applied: tuple[str, ...] = ()
        seed = 0
        if self.augmentation is not None:
            result = augment(
                waveform,
                row.sample_rate,
                self.augmentation,
                split=self.split,
                sample_id=row.sample_id,
                epoch=self.epoch,
            )
            waveform = result.waveform
            applied = result.applied
            seed = result.seed
        return {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "label": torch.tensor(row.label_index, dtype=torch.long),
            "sample_id": row.sample_id,
            "length": int(waveform.size),
            "is_padded": bool(row.is_padded),
            "coverage": float(row.coverage),
            "augmented": applied,
            "augment_seed": seed,
        }

    def label_counts(self) -> dict[int, int]:
        """Count rows per encoded label.

        Returns:
            Label index to row count, including zeros, so a caller can see a
            missing class rather than inferring one from a short dict.
        """
        counts: dict[int, int] = {0: 0, 1: 0}
        for row in self.rows:
            counts[row.label_index] = counts.get(row.label_index, 0) + 1
        return counts

    def sample_weights(self) -> np.ndarray | None:
        """Per-sample weights for balanced sampling.

        Returns:
            Weights summing to the row count, or ``None`` when balancing is off
            or a class has no members.
        """
        if self.augmentation is None:
            return None
        return class_weights([row.label_index for row in self.rows], self.augmentation)


def collate_samples(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Group a batch of items, stacking tensors and keeping metadata as objects.

    Torch's default collate is wrong for this dataset, for one reason: it
    recurses into every value, and ``augmented`` is a variable-length tuple of
    transform names. Two windows that had three transforms applied and two that
    had one give it a ragged list, and it raises ``RuntimeError: each element in
    list of batch should be of equal size`` rather than returning the names. That
    failure is about metadata, not audio, and losing a whole training run to it
    would be absurd.

    Tensors are stacked; everything else stays a plain list, indexed per sample.
    A training script wants ``batch["sample_id"][i]`` to be a string, not a
    tensor of strings.

    Args:
        batch: Items from :meth:`VoxShieldDataset.__getitem__`.

    Returns:
        The batched dict, or an empty dict for an empty batch.
    """
    items = list(batch)
    if not items:
        return {}
    batched: dict[str, Any] = {}
    for key in items[0]:
        values = [item[key] for item in items]
        if all(isinstance(value, torch.Tensor) for value in values):
            batched[key] = torch.stack(values)
        else:
            batched[key] = values
    return batched


def _balanced_sampler(
    dataset: VoxShieldDataset,
    class_balance: str | None,
    generator: torch.Generator,
) -> Sampler[int] | None:
    """Build a weighted sampler, or ``None`` when balancing is off.

    Args:
        dataset: The dataset whose rows are being balanced.
        class_balance: Override for the configured setting. ``None`` keeps the
            dataset's own setting, which is ``None`` for anything but ``train``.
        generator: Seeded generator, so the draw is reproducible.

    Returns:
        A sampler, or ``None``.
    """
    settings = dataset.augmentation
    if settings is None:
        if class_balance is None:
            return None
        # A caller balancing a split that carries no augmentation config: the
        # balance setting still has to be expressed as one to be read back.
        settings = AugmentationConfig(enabled=False, class_balance=class_balance)
    elif class_balance is not None:
        settings = replace(settings, class_balance=class_balance)
    labels = [row.label_index for row in dataset.rows]
    if not labels:
        return None
    weights = class_weights(labels, settings)
    if weights is None:
        return None
    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(labels),
        replacement=True,
        generator=generator,
    )


def build_dataloader(
    dataset: VoxShieldDataset,
    *,
    batch_size: int = 32,
    shuffle: bool | None = None,
    num_workers: int = 0,
    class_balance: str | None = None,
    seed: int = 0,
    drop_last: bool = False,
) -> DataLoader[Any]:
    """Build a DataLoader over one split.

    Args:
        dataset: The dataset to iterate.
        batch_size: Samples per batch.
        shuffle: Shuffle order. Defaults to ``True`` for ``train`` and ``False``
            otherwise, because a deterministic evaluation order is what makes two
            evaluation runs comparable.
        num_workers: Worker processes. ``0`` reads in the calling process, which
            is what a reproducible run wants; a nonzero value trades exact
            reproducibility for throughput.
        class_balance: ``"none"``, ``"weighted_sampler"``, or ``None`` to use the
            configured setting.
        seed: Seed for the shuffle and sampling generator.
        drop_last: Drop an incomplete final batch. A ragged final batch is not
            a special case here: the build already zero-fills each window to the
            configured width, so every tensor in the batch is the same shape and
            ``collate_samples`` can stack them. A model that must not see the
            zero fill reads ``coverage`` and ``is_padded`` per sample.

    Returns:
        A configured DataLoader.

    Raises:
        ValueError: ``batch_size`` is not positive, or the dataset holds no
            rows. Both are refused here rather than left to fail deeper in
            Torch, where an empty split surfaces as an opaque sampler error
            that names neither the split nor the reason it is empty.
    """
    if batch_size < 1:
        msg = f"batch_size must be at least 1, got {batch_size}"
        raise ValueError(msg)
    if len(dataset) == 0:
        # An empty split is nearly always the coverage floor or the padded-window
        # filter rather than a missing corpus, because the manifest writer
        # refuses to publish an empty split at all. Name the knobs.
        msg = (
            f"split {dataset.split!r} has no rows to load (min_coverage="
            f"{dataset.min_coverage}, include_padded={dataset.include_padded}); "
            "no build publishes an empty split, so one of those filters removed "
            "every row"
        )
        raise ValueError(msg)
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = _balanced_sampler(dataset, class_balance, generator)
    # A sampler already defines the draw, and DataLoader rejects shuffle=True
    # alongside a sampler.
    order: bool = sampler is None and (
        dataset.split == "train" if shuffle is None else bool(shuffle)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=order,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
        collate_fn=collate_samples,
    )


def load_split(
    split: str,
    *,
    paths: DataPaths | None = None,
    config: DataConfig | None = None,
    manifest: str | Path | None = None,
    **kwargs: Any,
) -> VoxShieldDataset:
    """Open one split for training or evaluation.

    Args:
        split: Split name.
        paths: Data paths.
        config: Build configuration.
        manifest: Explicit manifest path, overriding ``paths.manifests``.
        **kwargs: Passed to :class:`VoxShieldDataset`.

    Returns:
        The dataset.
    """
    resolved = paths if paths is not None else DataPaths.resolve("data")
    target = resolved.manifests / f"{split}.jsonl" if manifest is None else manifest
    return VoxShieldDataset(target, split=split, paths=resolved, config=config, **kwargs)


def summarise_batches(
    dataset: VoxShieldDataset,
    loader: DataLoader[Any],
    *,
    max_batches: int = 8,
) -> dict[str, Any]:
    """Describe what a loader yields, for ``voxshield data test-loader``.

    Reads a bounded number of batches so the check stays cheap on a large corpus.

    Args:
        dataset: The dataset being loaded.
        loader: The loader to sample.
        max_batches: Stop after this many batches.

    Returns:
        A summary of the observed batches, or the reason the check could not run.
    """
    if len(dataset) == 0:
        return {"ok": False, "reason": f"split {dataset.split!r} contains no rows", "rows": 0}
    batches = 0
    samples = 0
    width = 0
    labels: set[int] = set()
    seen: set[str] = set()
    for batch in loader:
        waveform = batch["waveform"]
        batches += 1
        samples += int(waveform.shape[0])
        width = int(waveform.shape[-1])
        labels.update(int(label) for label in batch["label"].tolist())
        seen.update(str(sample_id) for sample_id in batch["sample_id"])
        if batches >= max_batches:
            break
    return {
        "ok": True,
        "split": dataset.split,
        "rows": len(dataset),
        "batches_read": batches,
        "samples_read": samples,
        "unique_samples_seen": len(seen),
        "batch_width": width,
        "labels_present": sorted(labels),
        "augmentation": "on" if dataset.augmentation else "off",
    }
