"""Detector interface for synthetic-speech detection.

The contract is defined by a :class:`~typing.Protocol` so that a model can be
injected without VoxShield depending on torch, and so that tests can drive the
full API path with a deterministic stub.

MVP status: **no model is trained yet.** The preprocessing contract is
implemented and tested, but :class:`UnavailableDetector` is what ships, and the
API returns ``MODEL_UNAVAILABLE`` rather than a fabricated probability. That is
a deliberate choice. A plausible-looking score from an untrained or
unvalidated model is worse than no score at all, because it can trigger a
step-up verification against a real customer and consume analyst attention.
Evaluation infrastructure must land first -- see
``docs/evaluation-protocol.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from voxshield.errors import ModelContractError, ModelUnavailableError

__all__ = ["SegmentScore", "SynthSpeechDetector", "UnavailableDetector"]


@dataclass(frozen=True, slots=True)
class SegmentScore:
    """Detector output for one analysis window.

    Attributes:
        index: Zero-based segment index.
        synthetic_probability: Uncalibrated probability in ``[0, 1]`` that the
            window is synthetic. Calibration happens downstream, not here.
        model_version: Version string of the weights that produced this.
    """

    index: int
    synthetic_probability: float
    model_version: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.synthetic_probability <= 1.0:
            msg = (
                "synthetic_probability must be in [0, 1], got "
                f"{self.synthetic_probability}"
            )
            raise ModelContractError(msg)


@runtime_checkable
class SynthSpeechDetector(Protocol):
    """What the pipeline needs from a synthetic-speech detector."""

    @property
    def model_version(self) -> str:
        """Version of the loaded weights, recorded on every decision."""

    def score_segments(self, features: list[np.ndarray]) -> list[SegmentScore]:
        """Score a list of per-window feature arrays.

        Args:
            features: One log-mel array per window, as produced by
                :meth:`voxshield.audio.pipeline.PreparedAudio.segment_features`.

        Returns:
            One :class:`SegmentScore` per input, in the same order.
        """


class UnavailableDetector:
    """The detector that ships in MVP Phase 0.

    Raises :class:`ModelUnavailableError` on every scoring attempt so that a
    missing model is loud and typed rather than silently degrading to a
    constant or a random score.
    """

    @property
    def model_version(self) -> str:
        """Sentinel version recorded on abstentions."""
        return "unavailable"

    def score_segments(self, features: list[np.ndarray]) -> list[SegmentScore]:
        """Always raise. A model must be loaded before scoring."""
        msg = (
            "no synthetic-speech detector is loaded; Phase 0 ships the "
            "preprocessing contract only. See docs/evaluation-protocol.md."
        )
        raise ModelUnavailableError(msg)
