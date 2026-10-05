"""Manifest-backed feature loading for the Phase 3 baselines.

This module is the only place that turns a manifest into arrays. That is a
deliberate boundary: features are fitted and scored from here, so any second
path to the same numbers would be a second definition of what the model saw.

Three decisions are worth stating up front.

**Torch is not a dependency of this module.** Two of the three baselines are
scikit-learn models that must be trainable in an environment without PyTorch.
:func:`voxshield.data.torch_dataset.read_segment` would have been the obvious
reuse, but that module imports ``torch`` at module scope and raises
``ImportError`` without it, so importing it would make Torch mandatory for the
logistic-regression baseline. The loader below is a near-duplicate of it,
deliberately, and the duplication is documented at the function.

**Failures are counted, not swallowed.** A corpus with a handful of corrupt
segments should not abort a long run, and it must not silently become a smaller
dataset than the manifest describes. Decodes that fail are collected in
:meth:`FeatureMatrix.failed` and carried into the artefact metadata, so the
reported row count can always be reconciled against the manifest row count.

**Dropped rows are never silently dropped.** ``strict=True`` raises on the first
bad segment; ``strict=False`` collects. Either way the count is visible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voxshield.data.config import GateConfig
from voxshield.data.gates import GateReport, evaluate_leakage_gates
from voxshield.data.manifest import Manifest, read_manifest
from voxshield.data.schema import SampleRecord
from voxshield.errors import AudioDecodeError
from voxshield.training.config import TrainingError
from voxshield.training.features import FeatureExtractor

__all__ = [
    "SUBGROUP_AXES",
    "FeatureMatrix",
    "class_counts",
    "load_feature_matrix",
    "read_segment",
    "require_clean_gates",
    "resolve_rows",
]

#: Identity axes carried through to subgroup reporting, in report order.
#: These are the attributes a reviewer asks about first, because a strong
#: aggregate number that comes from one speaker or one codec is not a result.
SUBGROUP_AXES: tuple[str, ...] = (
    "generator_id",
    "speaker_id",
    "language",
    "codec",
    "channel",
    "device",
    "attack_type",
    "dataset_id",
)


def read_segment(path: str | Path, *, expected_samples: int | None = None) -> np.ndarray:
    """Read one standardised segment as mono float32.

    Mirrors :func:`voxshield.data.torch_dataset.read_segment` rather than
    importing it, because that module requires ``torch`` at import time and two
    of the three baselines must run without it. The behaviour is kept identical
    so a model trained here scores the same audio the Phase 2 loader would hand
    it.

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


@dataclass(frozen=True, slots=True)
class FeatureMatrix:
    """Materialised features for one split, aligned row-for-row.

    Attributes:
        split: Split name, for provenance.
        vectors: Pooled features of shape ``(n_samples, vector_dim)``. Always
            populated, even for the CNN, because it is what the metrics and the
            manifest summary need.
        labels: Encoded labels of shape ``(n_samples,)``, spoof positive.
        matrices: Fixed-length frame matrices of shape
            ``(n_samples, n_frames, frame_dim)``, or ``None`` when not
            requested. Only the CNN needs them, and materialising them for a
            large corpus costs roughly ``n * 300 * 80 * 4`` bytes, so they are
            opt-in.
        sample_ids: Manifest sample ids, row-aligned with ``vectors``.
        metadata: Subgroup axis to row-aligned value tuples.
        durations: Audio seconds per row, for the latency report's RTF.
        failed: ``(sample_id, reason)`` pairs for segments that could not be
            read. Empty when ``strict`` decoding was used.
        rows_considered: Rows in the manifest before any decode failure. Comparing
            this with ``len(vectors)`` is how a reader tells whether a reported
            result came from the whole split.
    """

    split: str
    vectors: np.ndarray
    labels: np.ndarray
    sample_ids: tuple[str, ...]
    metadata: dict[str, tuple[str, ...]]
    durations: tuple[float, ...]
    matrices: np.ndarray | None = None
    failed: tuple[tuple[str, str], ...] = ()
    rows_considered: int = 0

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @property
    def counts(self) -> dict[int, int]:
        """Row count per encoded label, zeros included."""
        return class_counts(self.labels)

    def subgroup_axis(self, name: str) -> tuple[str, ...]:
        """Row-aligned values for one subgroup axis.

        Args:
            name: An attribute of :data:`SUBGROUP_AXES`, or any other
                :class:`~voxshield.data.schema.SampleRecord` field.

        Returns:
            One value per row. An unknown axis yields ``UNKNOWN`` for every row
            rather than raising, so a report naming an axis a manifest lacks
            renders as one uniform group instead of failing the whole report.
        """
        return self.metadata.get(name, tuple("UNKNOWN" for _ in self.labels))

    def describe(self) -> dict[str, Any]:
        """JSON-ready summary, safe to embed in an artefact."""
        return {
            "split": self.split,
            "n_samples": len(self),
            "n_rows_considered": self.rows_considered,
            "vector_dim": int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0,
            "has_frame_matrices": self.matrices is not None,
            "class_counts": {str(k): v for k, v in sorted(self.counts.items())},
            "audio_seconds": float(sum(self.durations)),
            "n_failed_decodes": len(self.failed),
            "failed_sample_ids": [sample_id for sample_id, _ in self.failed],
        }


def class_counts(labels: np.ndarray) -> dict[int, int]:
    """Count encoded labels, including classes with zero members.

    Args:
        labels: 1-D array of encoded labels.

    Returns:
        Label index to count. Keys ``0`` and ``1`` are always present, so a
        missing class is visible rather than inferred from a short dict -- the
        difference between "the model failed" and "the corpus had one class".
    """
    counts = {0: 0, 1: 0}
    for value in np.asarray(labels).reshape(-1).tolist():
        index = int(value)
        counts[index] = counts.get(index, 0) + 1
    return counts


def resolve_rows(
    manifest: Manifest | str | Path,
    split: str,
    *,
    min_coverage: float = 0.0,
    include_padded: bool = True,
) -> tuple[SampleRecord, ...]:
    """Select the rows of one split, applying the coverage and padding filters.

    Args:
        manifest: Loaded manifest or a path to one.
        split: Split name.
        min_coverage: Drop windows whose speech coverage is below this. Coverage
            is recorded in the manifest precisely so a trainer can filter quiet
            windows without re-deriving VAD.
        include_padded: Keep zero-padded final windows. Padded windows are real
            audio plus silence, and dropping them removes the tail of every
            recording whose length is not a whole number of windows -- which
            biases the corpus toward recordings of a particular length.

    Returns:
        The selected rows, in manifest order.
    """
    loaded = read_manifest(manifest) if isinstance(manifest, str | Path) else manifest
    return tuple(
        row
        for row in loaded.by_split(split)
        if row.coverage >= min_coverage and (include_padded or not row.is_padded)
    )


def load_feature_matrix(
    rows: Sequence[SampleRecord],
    extractor: FeatureExtractor,
    *,
    root: str | Path,
    split: str = "train",
    want_matrices: bool = False,
    strict: bool = False,
    max_items: int | None = None,
) -> FeatureMatrix:
    """Decode audio and materialise features for a set of manifest rows.

    Args:
        rows: Manifest rows to load.
        extractor: Front end to apply.
        root: Data root that relative ``audio_path`` values resolve against.
        split: Split name, recorded on the result.
        want_matrices: Also materialise fixed-length frame matrices for the CNN.
        strict: Raise on the first decode failure instead of collecting it.
        max_items: Cap on rows read. For smoke runs; recorded on the result so a
            capped result is never mistaken for a full-split one.

    Returns:
        A :class:`FeatureMatrix` whose arrays are row-aligned.

    Raises:
        TrainingError: If ``rows`` is empty, or ``strict`` is set and a decode
            fails.
    """
    selected = tuple(rows[:max_items] if max_items is not None else rows)
    if not selected:
        msg = f"split {split!r} selected no rows; nothing to featurise"
        raise TrainingError(msg)

    base = Path(root)
    vectors: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    durations: list[float] = []
    failed: list[tuple[str, str]] = []
    columns: dict[str, list[str]] = {axis: [] for axis in SUBGROUP_AXES}

    for row in selected:
        location = row.audio_path
        candidate = Path(location)
        if not candidate.is_absolute():
            candidate = base / candidate
        try:
            samples = read_segment(candidate, expected_samples=row.waveform_samples)
        except AudioDecodeError as exc:
            if strict:
                raise TrainingError(
                    f"split {split!r}: refusing to continue past a bad segment "
                    f"({row.sample_id}): {exc}"
                ) from exc
            failed.append((row.sample_id, str(exc)))
            continue

        vectors.append(extractor.vector(samples))
        if want_matrices:
            matrices.append(extractor.matrix(samples))
        labels.append(int(row.label_index))
        sample_ids.append(row.sample_id)
        durations.append(float(row.duration_seconds))
        for axis in SUBGROUP_AXES:
            columns[axis].append(str(getattr(row, axis, "UNKNOWN") or "UNKNOWN"))

    if not vectors:
        msg = (
            f"split {split!r} had {len(selected)} row(s) and every decode failed; "
            f"first failure: {failed[0][1] if failed else 'none recorded'}"
        )
        raise TrainingError(msg)

    # Every collection above is appended inside the same success path, so they
    # are row-aligned by construction. This asserts that rather than assuming it:
    # adding a sixth collection later and forgetting to append it here would
    # otherwise produce a report whose subgroup values belong to the wrong rows,
    # which is the kind of error no downstream metric would ever catch.
    _verify_alignment(
        split=split,
        lengths={
            "vectors": len(vectors),
            "labels": len(labels),
            "sample_ids": len(sample_ids),
            "durations": len(durations),
            **dict.fromkeys(columns, len(next(iter(columns.values()), ()))),
            **({"matrices": len(matrices)} if want_matrices else {}),
        },
    )

    metadata = {axis: tuple(values) for axis, values in columns.items()}

    return FeatureMatrix(
        split=split,
        vectors=np.vstack(vectors).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        sample_ids=tuple(sample_ids),
        metadata=metadata,
        durations=tuple(durations),
        matrices=np.stack(matrices).astype(np.float32) if matrices else None,
        failed=tuple(failed),
        rows_considered=len(selected),
    )


def _verify_alignment(*, split: str, lengths: dict[str, int]) -> None:
    """Assert that every row-aligned collection has the same length.

    Args:
        split: Split name, for the error message.
        lengths: Collection name to row count.

    Raises:
        TrainingError: If the counts differ. This is a programming-error guard,
            not a data-quality check: a mismatch means the loader itself is
            inconsistent, and continuing would emit a report whose labels,
            predictions, and subgroup columns describe different rows.
    """
    distinct = set(lengths.values())
    if len(distinct) <= 1:
        return
    detail = ", ".join(f"{name}={count}" for name, count in sorted(lengths.items()))
    msg = (
        f"split {split!r}: row-aligned collections disagree in length ({detail}); "
        "the loader is inconsistent and the result would not be trustworthy"
    )
    raise TrainingError(msg)


def require_clean_gates(
    records: Sequence[SampleRecord],
    *,
    gate_config: GateConfig | None = None,
    fail_on_unavailable: bool = True,
) -> GateReport:
    """Evaluate leakage gates and refuse to proceed if any mandatory axis failed.

    The refusal is the point. A baseline trained on a corpus whose speaker or
    generator axis leaked between train and test still produces a confident
    EER, and that EER will be quoted later as if it meant something. Failing here
    means the number is never produced.

    Args:
        records: Records covering every split that will be used.
        gate_config: Gate policy. Defaults to :class:`GateConfig`, which makes
            speaker, file, and parent disjointness mandatory and leaves
            generator and session checks optional-but-reported.
        fail_on_unavailable: Treat a mandatory axis with no known identity values
            as a failure. Defaults to ``True`` here, overriding the ingestion
            default, because "unknown" cannot demonstrate disjointness and must
            not be recorded as though it did. This is the one place the
            training layer is deliberately stricter than the build layer: the
            build can degrade a report, a training run cannot produce a
            misleading one.

    Returns:
        The :class:`~voxshield.data.gates.GateReport`, for recording in the
        artefact even when it passes.

    Raises:
        TrainingError: With every blocking failure listed, not just the first.
            One at a time would mean rediscovering the same problem once per
            leaking axis.
    """
    if not records:
        msg = "cannot evaluate leakage gates on an empty record set"
        raise TrainingError(msg)

    cfg = gate_config or GateConfig()
    policy = replace(cfg, fail_on_unavailable=bool(fail_on_unavailable))
    report = evaluate_leakage_gates(records, config=policy)
    failures = report.failures
    if failures:
        listed = "\n  - ".join(failures)
        msg = (
            "leakage gates failed, refusing to train; every reported figure "
            f"would be meaningless:\n  - {listed}"
        )
        raise TrainingError(msg)
    return report
