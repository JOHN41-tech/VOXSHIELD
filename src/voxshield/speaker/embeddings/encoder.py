"""Replaceable speaker embedding encoder contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from voxshield.speaker.exceptions import EmbeddingExtractionError
from voxshield.speaker.schemas import Embedding

PREPROCESSING_VERSION = "voxshield-audio-preprocess-v1"


@runtime_checkable
class SpeakerEncoder(Protocol):
    """Encoder implementations accept canonical mono float32 audio."""

    encoder_id: str
    encoder_version: str
    sample_rate: int
    embedding_dimension: int

    def encode(self, audio: np.ndarray, sample_rate: int) -> Embedding:
        """Extract one embedding from canonical speech audio."""


def normalized_embedding(vector: np.ndarray, *, dimension: int | None = None) -> np.ndarray:
    """Validate and L2-normalize a finite, non-zero one-dimensional vector."""
    result = np.asarray(vector, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise EmbeddingExtractionError("encoder returned an empty or non-vector embedding")
    if dimension is not None and result.size != dimension:
        raise EmbeddingExtractionError(
            f"encoder returned dimension {result.size}; expected {dimension}"
        )
    if not np.isfinite(result).all():
        raise EmbeddingExtractionError("encoder returned a non-finite embedding")
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise EmbeddingExtractionError("encoder returned a zero-norm embedding")
    return np.ascontiguousarray(result / norm, dtype=np.float64)
