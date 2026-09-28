"""Model interfaces.

No trained weights ship in Phase 0. See ``interface.py`` for why the default
detector raises rather than returning a placeholder score.
"""

from voxshield.models.interface import (
    SegmentScore,
    SynthSpeechDetector,
    UnavailableDetector,
)

__all__ = ["SegmentScore", "SynthSpeechDetector", "UnavailableDetector"]
