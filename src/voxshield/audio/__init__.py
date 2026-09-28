"""Audio intake, preprocessing, feature extraction, and analysis.

The public surface is organised around three ways to handle audio, all of which
resolve to the same canonical representation:

* :func:`load_audio` -- one entry point for encoded bytes, files, streams, and
  in-memory arrays.
* :func:`prepare` -- the offline batch path, producing windowed features plus
  speech regions and a quality report via :func:`process_audio`.
* :class:`StreamingProcessor` -- the chunked path, reusing the same window
  geometry and feature function.

``ProcessResult`` is metadata-only on purpose: it carries counts, issue codes,
and timings, never samples or the feature matrix. See ``docs/privacy-design.md``.
"""

from typing import TYPE_CHECKING, Any

from voxshield.audio.decode import (
    DecodedAudio,
    assert_usable_decoded,
    decode_audio,
    decode_audio_bytes,
)
from voxshield.audio.features import compute_log_mel, mel_filterbank
from voxshield.audio.loader import AudioSource, load_audio
from voxshield.audio.pipeline import PreparedAudio, prepare, prepare_from_bytes
from voxshield.audio.preprocess import PreprocessedAudio, preprocess, preprocess_decoded
from voxshield.audio.quality import QualityReport, assess_quality
from voxshield.audio.segmentation import Segment, aggregate_segment_scores, segment_speech
from voxshield.audio.streaming import StreamingProcessor, StreamWindow
from voxshield.audio.vad import (
    EnergyVadDetector,
    SpeechMask,
    SpeechRegion,
    VadDetector,
    detect_speech,
    find_speech_regions,
)

__all__ = [
    "AudioSource",
    "DecodedAudio",
    "EnergyVadDetector",
    "PreparedAudio",
    "PreprocessedAudio",
    "ProcessResult",
    "QualityReport",
    "Segment",
    "SpeechMask",
    "SpeechRegion",
    "StreamWindow",
    "StreamingProcessor",
    "VadDetector",
    "aggregate_segment_scores",
    "assert_usable_decoded",
    "assess_quality",
    "compute_log_mel",
    "decode_audio",
    "decode_audio_bytes",
    "detect_speech",
    "find_speech_regions",
    "load_audio",
    "mel_filterbank",
    "prepare",
    "prepare_from_bytes",
    "preprocess",
    "preprocess_decoded",
    "process_audio",
    "segment_speech",
]

# ``process`` is deliberately not imported alongside the rest. This package is the
# parent of a runnable module, so importing it eagerly makes
# ``python -m voxshield.audio.process`` execute the module twice -- once as an
# import side effect, then again as ``__main__`` -- and CPython emits a
# RuntimeWarning saying exactly that. This import is the cause, so the fix has to
# sit on this side of it.
#
# The names stay exported. PEP 562 resolves them on first access, and the
# TYPE_CHECKING branch keeps full type information for readers and type checkers,
# so nothing downstream degrades to ``Any``.
if TYPE_CHECKING:
    from voxshield.audio.process import ProcessResult, process_audio

_LAZY_PROCESS_EXPORTS = frozenset({"ProcessResult", "process_audio"})


def __getattr__(name: str) -> Any:
    """Resolve the :mod:`voxshield.audio.process` exports on first access."""
    if name not in _LAZY_PROCESS_EXPORTS:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    from importlib import import_module

    return getattr(import_module("voxshield.audio.process"), name)
