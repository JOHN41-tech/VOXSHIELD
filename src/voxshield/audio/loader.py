"""The single entry point for getting audio into VoxShield.

Before this module, a caller wanting samples had to know whether it had bytes or
a path, and reach for the right of two functions. That is a small amount of
friction until someone forgets: an array path that skips the size check, a
``memoryview`` that no branch recognises, a numpy array that reaches a model
having bypassed validation entirely. :func:`load_audio` removes the choice.

Every supported source is brought to the same place -- a validated
:class:`~voxshield.audio.decode.DecodedAudio` carrying float32 samples, real
channel count, and a real sample rate -- under the same limits, with the same
error types. Callers up the stack therefore have exactly one code path to trust,
and the trust boundary stays in one module.

Two source kinds exist, and the difference is preserved rather than hidden:

* **Encoded** sources (bytes, paths, streams) are parsed by libsndfile, so the
  container's own format detection decides what they are. Nothing about the
  filename or any declared content type is believed, which is the property
  :mod:`voxshield.audio.decode` exists to guarantee.
* **Array** sources arrive already decoded, so there is no container to check.
  The limits that still apply are the ones that bound memory and work: sample
  rate, channel count, duration, finiteness, and whether the signal is audible
  at all. The size check is necessarily absent -- an array is already in memory,
  so bounding it is the caller's decision, not ours to make after the fact.
  :attr:`DecodedAudio.source_kind` records which path was taken, so an audit
  reader can tell the two apart.

Nothing here writes to disk, and no returned object can serialise a sample: the
metadata helpers in :mod:`voxshield.audio.decode` are the only sanctioned
description of a loaded clip.
"""

from __future__ import annotations

import io
import os

import numpy as np

from voxshield.audio.decode import (
    DecodedAudio,
    assert_usable_decoded,
    decode_audio,
)
from voxshield.config import AudioConfig
from voxshield.errors import (
    AudioDecodeError,
    AudioIntakeError,
    AudioTooLargeError,
    InvalidAudioSignalError,
)

__all__ = ["AudioSource", "load_audio"]

#: Every source shape :func:`load_audio` accepts. Encoded audio first, then
#: already-decoded samples, so the ordering reads as "cheap checks to expensive".
AudioSource = (
    bytes | bytearray | memoryview | str | os.PathLike[str] | io.BufferedIOBase | np.ndarray
)

_ARRAY_FORMATS = ("RAW",)
_ARRAY_SUBTYPES = ("PCM_FLOAT32", "PCM_FLOAT64", "PCM_16", "PCM_32")


def _load_array(
    samples: np.ndarray,
    sample_rate: int,
    config: AudioConfig,
    *,
    downmix: bool,
) -> DecodedAudio:
    """Validate and wrap already-decoded samples.

    Applies every limit that still means something once the samples are in
    memory, and skips only the one that cannot: the byte-size check, because by
    the time an array exists the memory has already been committed.
    """
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        msg = f"sample_rate must be a positive integer, got {sample_rate!r}"
        raise AudioIntakeError(msg)
    rate = int(sample_rate)

    if rate > config.max_sample_rate:
        msg = f"sample rate {rate} Hz exceeds the limit of {config.max_sample_rate} Hz"
        raise AudioTooLargeError(msg)

    array = np.asarray(samples)
    if array.dtype == object or not np.issubdtype(array.dtype, np.number):
        msg = f"audio array must have a numeric dtype, got {array.dtype!r}"
        raise AudioIntakeError(msg)
    if np.issubdtype(array.dtype, np.complexfloating):
        msg = f"complex audio is not supported, got {array.dtype!r}"
        raise AudioIntakeError(msg)

    if array.ndim == 1:
        data = array.reshape(-1, 1)
    elif array.ndim == 2:
        data = array
    else:
        msg = f"audio array must be 1-D or 2-D, got {array.ndim}-D with shape {array.shape}"
        raise AudioIntakeError(msg)

    frames = int(data.shape[0])
    channels = int(data.shape[1])

    if frames <= 0:
        msg = "audio array is empty"
        raise AudioDecodeError(msg)
    if channels <= 0:
        msg = "audio array declares zero channels"
        raise AudioDecodeError(msg)
    if channels > config.max_channels:
        msg = f"audio has {channels} channels, exceeding the limit of {config.max_channels}"
        raise AudioTooLargeError(msg)

    duration = frames / float(rate)
    if duration > config.max_duration_seconds:
        msg = f"audio is {duration:.1f}s, exceeding the {config.max_duration_seconds:.0f}s limit"
        raise AudioTooLargeError(msg)

    samples32 = np.ascontiguousarray(data, dtype=np.float32)
    if not np.isfinite(samples32).all():
        msg = "audio contains non-finite samples (NaN or Inf)"
        raise InvalidAudioSignalError(msg)

    if downmix and channels > 1:
        samples32 = samples32.mean(axis=1, dtype=np.float32)

    decoded = DecodedAudio(
        samples=np.ascontiguousarray(samples32, dtype=np.float32),
        sample_rate=rate,
        channels=channels,
        frames=frames,
        duration_seconds=duration,
        source_format=_ARRAY_FORMATS[0],
        source_subtype=_ARRAY_SUBTYPES[0],
        source_kind="array",
    )
    assert_usable_decoded(decoded)
    return decoded


def load_audio(
    source: AudioSource,
    sample_rate: int | None = None,
    config: AudioConfig | None = None,
    *,
    downmix: bool = True,
) -> DecodedAudio:
    """Load audio from any supported source, with one set of limits.

    Args:
        source: Encoded bytes, a path, a binary stream, or an already-decoded
            array. Paths and streams need no size hint; the file itself is
            stat'ed or read and bounded like any upload.
        sample_rate: Required when ``source`` is an array, ignored otherwise --
            the container states its own rate, and believing a caller's
            declaration over it would let a mismatched rate silently resample
            every sample.
        config: Limits to enforce. Defaults to :class:`AudioConfig`.
        downmix: Average channels to mono. Applied on both paths.

    Returns:
        A validated :class:`~voxshield.audio.decode.DecodedAudio`. Hold the
        samples only as long as the analysis needs them; this object is the
        audio itself and must not be logged or persisted.

    Raises:
        AudioTooLargeError: A size, rate, channel, or duration budget is
            exceeded.
        UnsupportedAudioFormatError: An encoded source is not on the allow-list.
        AudioDecodeError: The source is not decodable audio.
        InvalidAudioSignalError: The audio decodes but is unusable -- silent,
            non-finite, or below the audible floor.
        AudioIntakeError: The source is of an unsupported type, or an array
            arrived without a usable sample rate.
    """
    cfg = config or AudioConfig()

    if isinstance(source, np.ndarray):
        if sample_rate is None:
            msg = "sample_rate is required when loading from an array"
            raise AudioIntakeError(msg)
        return _load_array(source, sample_rate, cfg, downmix=downmix)

    # For encoded sources the rate comes from the container and any argument is
    # ignored. That is deliberate: trusting a caller's declared rate over the
    # container's would let a mismatch resample every sample by a silent factor
    # and quietly change the result. The parameter is therefore documented as
    # required for arrays and ignored otherwise, rather than being an error --
    # refusing it would break callers that pass a rate they happened to have.
    return decode_audio(source, cfg, downmix=downmix)
