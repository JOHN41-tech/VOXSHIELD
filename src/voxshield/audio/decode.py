"""Safe in-memory audio decoding.

This module is the trust boundary. Everything above it is attacker-controlled;
everything below it is trusted internal state. The ordering of checks in
:func:`decode_audio` is the security design, and it is deliberate:

1. **Size first.** Reject on byte count before parsing anything.
2. **Header before samples.** Parse container metadata via ``soundfile.info``
   (which does not decode) so the declared frame count can be checked *before*
   any large buffer is allocated. A WAV header claiming hours of audio is
   rejected on its header alone.
3. **Allow-list the container.** The format decision comes from libsndfile's own
   detection, never from the filename, the declared content type, or any client
   string. A request named ``evidence.wav`` that is actually an MP3 is detected
   as MP3 and rejected because MP3 is not allow-listed.
4. **Then decode**, in memory, into bounded float32.
5. **Then validate** that the signal is actually usable.

No audio is ever written to disk by this module. The decoded array is returned
to the caller, who is responsible for dropping the reference once inference
completes (see ``pipeline.py``, which scopes it to a ``del``).
"""

from __future__ import annotations

import io
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from voxshield.config import AudioConfig
from voxshield.errors import (
    AudioDecodeError,
    AudioIntakeError,
    AudioTooLargeError,
    InvalidAudioSignalError,
    UnsupportedAudioFormatError,
)

__all__ = [
    "DecodedAudio",
    "assert_usable_decoded",
    "decode_audio",
    "decode_audio_bytes",
]

# Formats libsndfile reports that are structurally unable to smuggle an
# unbounded decode. Anything else is refused without further inspection.
_HARD_REJECT_SUBTYPES = frozenset({
    "ULAW",
    "ALAW",
    "IMA_ADPCM",
    "MS_ADPCM",
    "GSM610",
    "DWVW",
    "DWVN",
    "VOICEWORKS",
    "MPC2K",
})


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    """Validated, decoded audio. Never persisted.

    Attributes:
        samples: ``float32`` array. A mono source stays ``(n_frames, 1)``,
            keeping the channel axis that ``always_2d=True`` produced rather
            than being flattened, so the shape does not change with ``downmix``
            for mono input. A multichannel source folded to mono becomes
            ``(n_frames,)``; with ``downmix=False`` it stays
            ``(n_frames, n_channels)``.
        sample_rate: Sample rate of ``samples`` in Hz, as reported by the
            container.
        channels: Channel count of the source.
        frames: Frame count of the source.
        duration_seconds: ``frames / sample_rate``.
        source_format: Container format as detected by libsndfile, e.g. ``WAV``.
        source_subtype: Sample encoding, e.g. ``PCM_16``.
        source_kind: How the audio reached the loader -- ``encoded`` for a
            container parsed by libsndfile, ``array`` for samples handed over
            already decoded. Recorded so an audit reader can tell a raw PCM
            capture from a re-encoded upload, which have different provenance.
    """

    samples: np.ndarray
    sample_rate: int
    channels: int
    frames: int
    duration_seconds: float
    source_format: str
    source_subtype: str
    source_kind: str = "encoded"

    def __post_init__(self) -> None:
        # Defensive: prevent an accidentally-escaping non-finite sample from
        # reaching a model and producing a meaningless-but-confident score.
        if self.samples.size and not np.isfinite(self.samples).all():
            msg = "decoded audio contains non-finite samples (NaN or Inf)"
            raise InvalidAudioSignalError(msg)


def _resolve_source(
    source: bytes | bytearray | memoryview | io.BufferedIOBase | str | os.PathLike[str],
    config: AudioConfig,
) -> tuple[io.BytesIO, int]:
    """Return a seekable binary stream and the byte length, enforcing size first."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        n_bytes = len(source)
        if n_bytes == 0:
            msg = "uploaded audio is empty"
            raise AudioDecodeError(msg)
        if n_bytes > config.max_upload_bytes:
            msg = (
                f"upload is {n_bytes} bytes, exceeding the "
                f"{config.max_upload_bytes}-byte limit"
            )
            raise AudioTooLargeError(msg)
        return io.BytesIO(bytes(source)), n_bytes

    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        try:
            n_bytes = path.stat().st_size
        except OSError as exc:
            msg = f"could not stat audio path: {exc}"
            raise AudioIntakeError(msg) from exc
        if n_bytes == 0:
            msg = f"audio file at {path} is empty"
            raise AudioDecodeError(msg)
        if n_bytes > config.max_upload_bytes:
            msg = (
                f"audio file is {n_bytes} bytes, exceeding the "
                f"{config.max_upload_bytes}-byte limit"
            )
            raise AudioTooLargeError(msg)
        return io.BytesIO(path.read_bytes()), n_bytes

    if isinstance(source, io.BufferedIOBase) or hasattr(source, "read"):
        raw = source.read()
        if not isinstance(raw, (bytes, bytearray)):
            msg = "audio stream did not yield bytes"
            raise AudioIntakeError(msg)
        return decode_stream_bytes(bytes(raw), config)

    msg = (
        "unsupported audio source type "
        f"{type(source).__name__!r}; expected bytes, path, or binary stream"
    )
    raise AudioIntakeError(msg)


def decode_stream_bytes(raw: bytes, config: AudioConfig) -> tuple[io.BytesIO, int]:
    """Validate byte length and wrap in a stream."""
    n_bytes = len(raw)
    if n_bytes == 0:
        msg = "uploaded audio is empty"
        raise AudioDecodeError(msg)
    if n_bytes > config.max_upload_bytes:
        msg = (
            f"upload is {n_bytes} bytes, exceeding the "
            f"{config.max_upload_bytes}-byte limit"
        )
        raise AudioTooLargeError(msg)
    return io.BytesIO(raw), n_bytes


def _validate_header(
    info: sf.SoundFile, config: AudioConfig
) -> tuple[int, int, float]:
    """Validate container metadata. Returns ``(channels, frames, duration)``.

    This runs against the *declared* header, before any sample data is read.
    """
    fmt = (info.format or "").upper()
    subtype = (info.subtype or "").upper()

    if fmt not in config.allowed_formats:
        msg = (
            f"format {fmt or 'unknown'!r} is not supported; "
            f"allowed formats are {sorted(config.allowed_formats)}"
        )
        raise UnsupportedAudioFormatError(msg)

    if subtype in _HARD_REJECT_SUBTYPES:
        msg = f"subtype {subtype!r} is not supported for analysis"
        raise UnsupportedAudioFormatError(msg)

    if subtype not in config.allowed_subtypes:
        msg = (
            f"subtype {subtype or 'unknown'!r} is not supported; "
            f"allowed subtypes are {sorted(config.allowed_subtypes)}"
        )
        raise UnsupportedAudioFormatError(msg)

    channels = int(info.channels)
    if channels <= 0:
        msg = "audio declares zero channels"
        raise AudioDecodeError(msg)
    if channels > config.max_channels:
        msg = f"audio has {channels} channels, exceeding the limit of {config.max_channels}"
        raise AudioTooLargeError(msg)

    sample_rate = int(info.samplerate)
    if sample_rate <= 0:
        msg = "audio declares a sample rate of zero"
        raise AudioDecodeError(msg)
    if sample_rate > config.max_sample_rate:
        msg = (
            f"sample rate {sample_rate} Hz exceeds the limit of "
            f"{config.max_sample_rate} Hz"
        )
        raise AudioTooLargeError(msg)

    frames = int(info.frames)
    if frames <= 0:
        msg = "audio declares zero frames"
        raise AudioDecodeError(msg)

    duration = frames / float(sample_rate)
    if duration > config.max_duration_seconds:
        msg = (
            f"audio declares {duration:.1f}s of audio, exceeding the "
            f"{config.max_duration_seconds:.0f}s limit"
        )
        raise AudioTooLargeError(msg)

    return channels, frames, duration


def decode_audio(
    source: bytes | bytearray | memoryview | io.BufferedIOBase | str | os.PathLike[str],
    config: AudioConfig | None = None,
    *,
    downmix: bool = True,
) -> DecodedAudio:
    """Decode audio from bytes, a path, or a binary stream, safely.

    Args:
        source: Raw bytes, a filesystem path, or a readable binary stream.
        config: Limits to enforce. Defaults to :func:`load_audio_config`.
        downmix: If true, average channels into a single mono channel.

    Returns:
        A :class:`DecodedAudio` holding in-memory samples.

    Raises:
        AudioTooLargeError: Upload exceeds a size, duration, channel, or rate
            budget. Raised before large allocations.
        UnsupportedAudioFormatError: Container or sample encoding is not on the
            allow-list.
        AudioDecodeError: Bytes are not decodable audio.
        InvalidAudioSignalError: Audio decodes but contains non-finite or
            degenerate samples.
    """
    cfg = config or AudioConfig()
    stream, _ = _resolve_source(source, cfg)
    stream.seek(0)

    # --- Step 2: header only, no sample decoding --------------------------
    try:
        info = sf.info(stream)
    except Exception as exc:  # libsndfile raises RuntimeError/LibsndfileError
        # The underlying message names internal types and object addresses. It is
        # chained onto the exception for the log, but kept out of the client
        # response, which only ever carries this sentence.
        msg = "audio header could not be parsed"
        raise AudioDecodeError(msg) from exc
    finally:
        stream.seek(0)

    # _validate_header performs the limit checks; the caller derives frame count
    # and duration from the decoded array, which is the authoritative source
    # once resampling and mono-folddown have run.
    channels, _, _ = _validate_header(info, cfg)

    # --- Step 4: decode, bounded and in memory ----------------------------
    try:
        data, sample_rate = sf.read(
            stream,
            dtype="float32",
            always_2d=True,
        )
    except Exception as exc:
        msg = "audio samples could not be decoded"
        raise AudioDecodeError(msg) from exc

    if data.size == 0:
        msg = "audio decoded to zero samples"
        raise AudioDecodeError(msg)

    if not np.isfinite(data).all():
        msg = "decoded audio contains non-finite samples (NaN or Inf)"
        raise InvalidAudioSignalError(msg)

    # A truncated or dishonest header can make libsndfile return fewer frames
    # than declared. Trust the decoded length from here on.
    actual_frames = int(data.shape[0])
    actual_duration = actual_frames / float(sample_rate)
    if actual_duration > cfg.max_duration_seconds:
        msg = (
            f"decoded audio is {actual_duration:.1f}s, exceeding the "
            f"{cfg.max_duration_seconds:.0f}s limit"
        )
        raise AudioTooLargeError(msg)

    if downmix and channels > 1:
        data = data.mean(axis=1, dtype=np.float32)

    decoded = DecodedAudio(
        samples=np.ascontiguousarray(data, dtype=np.float32),
        sample_rate=sample_rate,
        channels=channels,
        frames=actual_frames,
        duration_seconds=actual_duration,
        source_format=(info.format or "UNKNOWN").upper(),
        source_subtype=(info.subtype or "UNKNOWN").upper(),
    )
    assert_usable_decoded(decoded)
    return decoded


def assert_usable_decoded(decoded: DecodedAudio) -> None:
    """Reject degenerate signals that would poison downstream inference.

    Public because the unified loader applies the same bar to already-decoded
    arrays: a caller who bypasses the container parser must not thereby bypass
    the check that the signal is analysable.
    """
    samples = decoded.samples
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak == 0.0:
        msg = "audio is entirely silent; nothing to analyse"
        raise InvalidAudioSignalError(msg)
    if not np.isfinite(peak):
        msg = "audio contains non-finite amplitude"
        raise InvalidAudioSignalError(msg)
    # A digital-black signal that is not exactly zero is still unusable.
    if peak < 1e-8:
        msg = f"audio peak amplitude {peak:.3e} is below the usable floor"
        raise InvalidAudioSignalError(msg)


def decode_audio_bytes(
    payload: bytes | bytearray,
    config: AudioConfig | None = None,
    *,
    downmix: bool = True,
) -> DecodedAudio:
    """:func:`decode_audio` for the common case of an in-memory upload."""
    return decode_audio(payload, config, downmix=downmix)


def summarise_metadata(decoded: DecodedAudio) -> dict[str, object]:
    """Metadata-only description of a decoded clip.

    This is the *only* shape permitted to reach logs, the audit store, and the
    dashboard. It contains no audio samples, no transcript, and no
    identifier-bearing field. See ``docs/privacy-design.md``.
    """
    samples: np.ndarray = decoded.samples
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return {
        "duration_seconds": round(decoded.duration_seconds, 4),
        "source_sample_rate_hz": decoded.sample_rate,
        "channels": decoded.channels,
        "frames": decoded.frames,
        "source_format": decoded.source_format,
        "source_subtype": decoded.source_subtype,
        "source_kind": decoded.source_kind,
        "peak_amplitude": round(peak, 6),
    }


def batch_max_duration(config: AudioConfig, n_files: int) -> float:
    """Total audio seconds permitted for a batch of ``n_files`` clips."""
    return float(n_files) * config.max_duration_seconds


def allowed_formats(config: AudioConfig) -> Sequence[str]:
    """Human-readable allow-list, for error messages and documentation."""
    return sorted(config.allowed_formats)
