"""Training-only augmentation, reproducible from parameters alone.

Three rules govern this module, and each one exists because of a specific way a
spoof-detection result goes quietly wrong:

* **Evaluation audio is never augmented.** A test set that has been augmented
  measures the augmentation rather than the detector, and reports a headline
  number that will not reproduce on any other corpus. The guard here is a
  parameter, not a convention: :func:`augment` refuses a non-``train`` split
  outright rather than trusting the caller to have checked.
* **Nothing is downloaded and nothing is shared.** The noise corpus is a local
  directory; if it is missing, the fallback is synthetic and says so. A training
  run whose audio depends on a network call is a run that cannot be repeated.
* **What was asked for is what gets reported.** Every transform returns the name
  it applied. A scheme this module cannot implement is named in
  :attr:`AugmentedAudio.unavailable` rather than skipped quietly, because a
  silently skipped codec looks exactly like a codec that ran.

Augmentation happens at *load* time, not build time. The stored corpus stays
canonical, so the same manifest yields both the clean and the augmented view and
the two never disagree about which samples exist.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from voxshield.data.config import BALANCE_NONE, BALANCE_WEIGHTED_SAMPLER, AugmentationConfig
from voxshield.data.errors import AugmentationError

__all__ = [
    "G711_ALAW",
    "G711_MULAW",
    "SUPPORTED_CODEC_SCHEMES",
    "TRAIN_SPLIT",
    "AugmentedAudio",
    "augment",
    "available_codec_schemes",
    "class_weights",
    "derive_seed",
    "g711_alaw_decode",
    "g711_alaw_encode",
    "g711_mulaw_decode",
    "g711_mulaw_encode",
]

TRAIN_SPLIT = "train"
G711_MULAW = "g711_mulaw"
G711_ALAW = "g711_alaw"

#: Every codec this module can actually implement. A scheme outside this set is
#: reported as unavailable rather than skipped, so the gap is visible in the run.
SUPPORTED_CODEC_SCHEMES: frozenset[str] = frozenset({G711_MULAW, G711_ALAW})

#: A waveform is float32 in [-1, 1]; G.711 is defined on int16. This is the
#: conversion factor between them, and keeping it in one place keeps the codec
#: round-trips from disagreeing about the full-scale point.
_INT16_SCALE = 32768.0

# G.711 works on a reduced-precision integer and recovers the rest by segment,
# so both codecs are written to the reference formulation rather than derived from
# a closed-form formula. Deriving one is where an off-by-one in the segment table
# hides: the round trip still runs, and it is still audibly lossy, and nothing
# says the curve is wrong.
_ULAW_BIAS = 0x84
_ULAW_CLIP = 8159  # 14-bit full scale, after the 16-bit >> 2
_ULAW_SEG_END = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int64)
_ULAW_SIGN = 0x80
_ULAW_QUANT_MASK = 0x0F
_ULAW_SEG_MASK = 0x70
_ULAW_SEG_SHIFT = 4

_ALAW_ORDINARY = 0x55
_ALAW_ORD = 0xD5
_ALAW_SEG_END = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF], dtype=np.int64)
#: First 13-bit value in each segment. The encoder searches against
#: ``_ALAW_SEG_END``; the decoder walks back from these starts, so the two stay
#: consistent by construction rather than by arithmetic coincidence.
_ALAW_SEG_START = np.array([0x00, 0x20, 0x40, 0x80, 0x100, 0x200, 0x400, 0x800], dtype=np.int64)
_ALAW_SIGN = 0x80
_ALAW_QUANT_MASK = 0x0F
_ALAW_SEG_MASK = 0x70
_ALAW_SEG_SHIFT = 4


def _to_int16(samples: np.ndarray) -> np.ndarray:
    """Scale float samples in ``[-1, 1]`` to clipped int16."""
    scaled = np.asarray(samples, dtype=np.float64) * _INT16_SCALE
    return np.clip(np.round(scaled), -32768.0, 32767.0).astype(np.int64)


@dataclass(frozen=True, slots=True)
class AugmentedAudio:
    """One augmented window, and an account of how it was produced.

    Attributes:
        waveform: The augmented samples, float32 in ``[-1, 1]``, same length as
            the input. Augmentation never changes a window's length, so a
            manifest row and its audio stay aligned.
        applied: Transforms actually applied, in the order they ran.
        seed: The seed that produced this result. Recorded so a surprising
            sample can be reproduced on its own rather than by replaying the run.
        unavailable: Requested codecs this module cannot implement. Empty when
            the configuration was fully honoured.
        notes: Human-readable remarks worth surfacing, such as a missing noise
            corpus falling back to synthetic noise.
    """

    waveform: np.ndarray
    applied: tuple[str, ...] = ()
    seed: int = 0
    unavailable: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.waveform.dtype != np.float32:
            msg = f"augmented waveform must be float32, got {self.waveform.dtype}"
            raise AugmentationError(msg)


def derive_seed(base_seed: int, sample_id: str, epoch: int = 0) -> int:
    """Derive a per-sample, per-epoch seed from the configured base seed.

    The seed is a hash of the base seed, the sample id and the epoch, so it does
    not depend on iteration order. That matters: deriving a seed from a counter
    would make the augmentation of a sample depend on how many samples the
    loader happened to visit before it, and the same corpus would augment
    differently under a different batch size or sampler.

    The hash is BLAKE2b over a length-delimited encoding, not the built-in
    :func:`hash`. Python salts ``hash`` of a ``str``/``tuple`` per process, so a
    built-in-hash seed is different in every worker and every rerun: the same
    checkpoint would resume onto different augmentation and every reported
    reproducibility claim would be false. A cryptographic digest is also
    seedable-free, which is the entire requirement here -- it is not doing
    security work, it is doing process-stability work.

    Args:
        base_seed: The configured base seed.
        sample_id: Stable sample identifier, usually the manifest ``sample_id``.
        epoch: Epoch index, so a second pass over the data differs from the first.

    Returns:
        A non-negative seed for :func:`numpy.random.default_rng`.
    """
    payload = b"".join(
        part.encode("utf-8") + b"\x00"
        for part in (str(int(base_seed)), str(sample_id), str(int(epoch)))
    )
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def available_codec_schemes(schemes: Sequence[str]) -> tuple[str, ...]:
    """Return the subset of ``schemes`` this module can implement.

    Args:
        schemes: The requested codec scheme names.

    Returns:
        The requested names that :data:`SUPPORTED_CODEC_SCHEMES` covers, in
        request order.
    """
    return tuple(name for name in schemes if name in SUPPORTED_CODEC_SCHEMES)


def g711_mulaw_encode(samples: np.ndarray) -> np.ndarray:
    """Encode float samples in ``[-1, 1]`` to 8-bit G.711 mu-law bytes.

    Args:
        samples: Float samples in ``[-1, 1]``.

    Returns:
        ``uint8`` array of mu-law codes.
    """
    pcm16 = _to_int16(samples)
    pcm = np.abs(pcm16) >> 2
    pcm = np.clip(pcm, 0, _ULAW_CLIP) + (_ULAW_BIAS >> 2)
    # ``_ULAW_SEG_END`` holds inclusive upper bounds, so a value sitting exactly on
    # a boundary belongs to that segment, not to the next one. ``side="left"`` is
    # what encodes that; the other choice quietly shifts every boundary sample one
    # segment too narrow, which leaves the transfer curve still monotone and the
    # round trip still lossy -- just lossier than G.711 at the wrong places.
    segment = np.searchsorted(_ULAW_SEG_END, pcm, side="left")
    clipped = segment >= _ULAW_SEG_END.size
    segment = np.minimum(segment, _ULAW_SEG_END.size - 1)
    mantissa = (pcm >> (segment + 1)) & _ULAW_QUANT_MASK
    code = np.where(clipped, 0x7F, (segment << _ULAW_SEG_SHIFT) | mantissa)
    # The polarity is carried by the mask, not by the sign of the sample. A plain
    # complement would set the sign bit unconditionally and collapse the negative
    # half of the waveform onto the positive one, which still sounds like audio
    # and still round-trips without error -- on the wrong signal.
    mask = np.where(pcm16 < 0, 0x7F, 0xFF)
    return ((code ^ mask) & 0xFF).astype(np.uint8)


def g711_mulaw_decode(codes: np.ndarray) -> np.ndarray:
    """Decode 8-bit G.711 mu-law bytes to float samples in ``[-1, 1]``.

    Args:
        codes: ``uint8`` array of mu-law codes.

    Returns:
        Float64 array of decoded samples in ``[-1, 1]``.
    """
    value = ~np.asarray(codes, dtype=np.uint8).astype(np.int64) & 0xFF
    sign = (value & _ULAW_SIGN) != 0
    segment = (value & _ULAW_SEG_MASK) >> _ULAW_SEG_SHIFT
    mantissa = value & _ULAW_QUANT_MASK
    # The bias is added before the shift is undone, which returns the zero point
    # to zero instead of leaving it at a third of full scale.
    magnitude = (((mantissa << 3) + _ULAW_BIAS) << segment) - _ULAW_BIAS
    decoded = np.where(sign, -magnitude, magnitude)
    return decoded / _INT16_SCALE


def g711_alaw_encode(samples: np.ndarray) -> np.ndarray:
    """Encode float samples in ``[-1, 1]`` to 8-bit G.711 A-law bytes.

    Args:
        samples: Float samples in ``[-1, 1]``.

    Returns:
        ``uint8`` array of A-law codes.
    """
    pcm = _to_int16(samples) >> 3
    positive = pcm >= 0
    # A negative sample is stored as the one's complement of its magnitude below
    # 1, so that -1 and 0 cannot collide on the same code.
    magnitude = np.where(positive, pcm, -pcm - 1)
    segment = np.searchsorted(_ALAW_SEG_END, magnitude, side="left")
    clipped = segment >= _ALAW_SEG_END.size
    segment = np.minimum(segment, _ALAW_SEG_END.size - 1)
    # The two lowest segments are wider than the shift below would give them.
    shifted = np.where(segment < 2, magnitude >> 1, magnitude >> segment)
    mantissa = shifted & _ALAW_QUANT_MASK
    code = np.where(clipped, 0x7F, (segment << _ALAW_SEG_SHIFT) | mantissa)
    mask = np.where(positive, _ALAW_ORDINARY, _ALAW_ORD)
    return ((code ^ mask) & 0xFF).astype(np.uint8)


def g711_alaw_decode(codes: np.ndarray) -> np.ndarray:
    """Decode 8-bit G.711 A-law bytes to float samples in ``[-1, 1]``.

    Args:
        codes: ``uint8`` array of A-law codes.

    Returns:
        Float64 array of decoded samples in ``[-1, 1]``.
    """
    value = np.asarray(codes, dtype=np.uint8).astype(np.int64) ^ _ALAW_ORDINARY
    sign = (value & _ALAW_SIGN) != 0
    segment = (value & _ALAW_SEG_MASK) >> _ALAW_SEG_SHIFT
    mantissa = value & _ALAW_QUANT_MASK
    # Reconstruct in the 13-bit domain the encoder worked in, then scale up. The
    # segment's first value is a fixed offset, the mantissa counts quanta of
    # 2**segment, and a half-quantum is added so the reconstruction sits between
    # the two values it could have come from instead of on the lower one.
    start = _ALAW_SEG_START[segment]
    step = np.maximum(segment, 1)
    half = np.where(segment == 0, 1, 1 << (segment - 1))
    reconstructed = start + (mantissa << step) + half
    magnitude = reconstructed << 3
    decoded = np.where(sign, -magnitude, magnitude)
    return decoded / _INT16_SCALE


def _codec_round_trip(samples: np.ndarray, scheme: str) -> np.ndarray:
    """Simulate a codec by encoding to 8 bits and decoding back."""
    if scheme == G711_MULAW:
        return g711_mulaw_decode(g711_mulaw_encode(samples))
    if scheme == G711_ALAW:
        return g711_alaw_decode(g711_alaw_encode(samples))
    msg = f"unsupported codec scheme {scheme!r}"
    raise AugmentationError(msg)


def _apply_gain(
    samples: np.ndarray,
    rng: np.random.Generator,
    config: AugmentationConfig,
) -> np.ndarray:
    gain_db = float(rng.uniform(config.gain_db_min, config.gain_db_max))
    return samples * (10.0 ** (gain_db / 20.0))


def _synthetic_noise(
    length: int,
    rng: np.random.Generator,
    sample_rate: int,
) -> np.ndarray:
    """Pink-ish noise, generated rather than downloaded.

    White noise is spectrally flat, which sounds unlike any real environment and
    produces an unrealistically easy denoising task. A one-pole filtered white
    sequence tilts the spectrum towards the range where real room and street
    noise actually lives.
    """
    white = rng.standard_normal(length)
    spectrum = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(length, d=1.0 / sample_rate)
    tilt = 1.0 / np.sqrt(np.maximum(frequencies, 1.0))
    coloured = np.fft.irfft(spectrum * tilt, n=length)
    peak = float(np.max(np.abs(coloured)))
    if peak > 0.0:
        coloured = coloured / peak
    return coloured


def _load_noise_corpus(directory: Path) -> tuple[np.ndarray, ...]:
    """Load every readable audio file in ``directory`` as mono float32.

    Returns an empty tuple when the directory is missing or holds nothing
    readable, which the caller reports as a fallback rather than an error: a
    missing optional noise corpus should not stop a build, but it must not pass
    unnoticed either.
    """
    if not directory.is_dir():
        return ()
    clips: list[np.ndarray] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".wav", ".flac", ".ogg"}:
            continue
        try:
            samples, _ = sf.read(path, dtype="float32", always_2d=False)
        except (RuntimeError, OSError, ValueError):
            continue
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if samples.size:
            clips.append(np.asarray(samples, dtype=np.float32))
    return tuple(clips)


def _apply_noise(
    samples: np.ndarray,
    rng: np.random.Generator,
    config: AugmentationConfig,
    sample_rate: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Add noise at a chosen SNR."""
    notes: list[str] = []
    noise: np.ndarray | None = None
    if config.noise_corpus_dir:
        clips = _load_noise_corpus(Path(config.noise_corpus_dir))
        if clips:
            noise = clips[int(rng.integers(len(clips)))]
        else:
            notes.append(
                f"noise_corpus_dir {config.noise_corpus_dir!r} held no readable audio; "
                "using synthetic noise"
            )
    if noise is None:
        noise = _synthetic_noise(samples.size, rng, sample_rate)

    if noise.size < samples.size:
        repeats = math.ceil(samples.size / max(noise.size, 1))
        noise = np.tile(noise, repeats)
    start = 0 if noise.size == samples.size else int(rng.integers(noise.size - samples.size + 1))
    window = noise[start : start + samples.size]

    signal_power = float(np.mean(np.square(samples)))
    noise_power = float(np.mean(np.square(window)))
    if noise_power <= 0.0 or signal_power <= 0.0:
        return samples, tuple(notes)
    snr_db = float(rng.uniform(config.noise_snr_db_min, config.noise_snr_db_max))
    scale = math.sqrt(signal_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    return samples + window * scale, tuple(notes)


def _apply_channel(samples: np.ndarray, rng: np.random.Generator, sample_rate: int) -> np.ndarray:
    """Band-limit the signal the way a handset microphone would.

    A telephone channel is roughly 300 Hz to 3.4 kHz. Filtering there removes
    spectral detail a detector may have learned to rely on, which is a different
    failure mode from adding noise.
    """
    upper = min(3400.0, sample_rate / 2.0 - 1.0)
    lower = min(300.0, upper / 2.0)
    if upper <= lower:
        return samples
    sos = signal.butter(4, [lower, upper], btype="bandpass", fs=sample_rate, output="sos")
    filtered = signal.sosfilt(sos, samples)
    return np.asarray(filtered, dtype=np.float32)


def _apply_reverb(
    samples: np.ndarray,
    rng: np.random.Generator,
    sample_rate: int,
) -> np.ndarray:
    """Convolve with a synthetic exponentially-decaying impulse response."""
    length = int(0.08 * sample_rate * float(rng.uniform(0.5, 1.5)))
    length = max(length, 8)
    decay = float(rng.uniform(0.25, 0.7))
    time = np.arange(length) / sample_rate
    impulse = rng.standard_normal(length) * np.exp(-decay * time * sample_rate * 0.1)
    direct_delay = max(int(0.005 * sample_rate), 1)
    impulse[direct_delay] += 1.0
    peak = float(np.max(np.abs(impulse)))
    if peak > 0.0:
        impulse = impulse / peak
    wet = signal.fftconvolve(samples, impulse, mode="full")[: samples.size]
    mix = float(rng.uniform(0.15, 0.4))
    return np.asarray((1.0 - mix) * samples + mix * wet, dtype=np.float32)


def _peak_limit(samples: np.ndarray) -> np.ndarray:
    """Scale back if a chain of transforms pushed the signal past full scale.

    Clipping would add broadband distortion that no configured transform asked
    for, so the gain is reduced to preserve it instead.
    """
    peak = float(np.max(np.abs(samples)))
    if peak > 1.0:
        return np.asarray(samples / peak, dtype=np.float32)
    return samples


def augment(
    samples: np.ndarray,
    sample_rate: int,
    config: AugmentationConfig | None = None,
    *,
    split: str,
    sample_id: str = "",
    epoch: int = 0,
) -> AugmentedAudio:
    """Augment one window according to ``config``.

    Args:
        samples: Float32 samples in ``[-1, 1]``.
        sample_rate: Sample rate of ``samples``.
        config: Augmentation configuration. ``None`` or ``enabled=False``
            returns the input untouched.
        split: The split this window belongs to. Only ``train`` may be
            augmented.
        sample_id: Stable sample identifier, used to derive the seed.
        epoch: Epoch index, used to derive the seed.

    Returns:
        An :class:`AugmentedAudio` carrying the waveform and an account of what
        was applied.

    Raises:
        AugmentationError: ``split`` is not ``train`` while augmentation is
            enabled, or the waveform is empty or not float.
    """
    waveform = np.asarray(samples, dtype=np.float32)
    if waveform.ndim != 1 or waveform.size == 0:
        msg = f"augment expects a non-empty 1-D waveform, got shape {waveform.shape}"
        raise AugmentationError(msg)
    if config is None or not config.enabled:
        base_seed = config.seed if config is not None else 0
        return AugmentedAudio(
            waveform=waveform,
            seed=derive_seed(base_seed, sample_id, epoch),
        )

    if split != TRAIN_SPLIT:
        msg = (
            f"refusing to augment {split!r} audio: augmentation is defined only for "
            f"{TRAIN_SPLIT!r}, because an augmented evaluation split measures the "
            "augmentation rather than the detector"
        )
        raise AugmentationError(msg)

    seed = derive_seed(config.seed, sample_id, epoch)
    rng = np.random.default_rng(seed)

    applied: list[str] = []
    notes: list[str] = []
    unavailable = tuple(
        name for name in config.codec_schemes if name not in SUPPORTED_CODEC_SCHEMES
    )
    for name in unavailable:
        notes.append(
            f"codec scheme {name!r} is not implemented by voxshield.data.augment; "
            "it was not applied"
        )

    result = waveform
    if rng.random() < config.noise_probability:
        result, noise_notes = _apply_noise(result, rng, config, sample_rate)
        applied.append("noise")
        notes.extend(noise_notes)
    if rng.random() < config.codec_probability:
        usable = available_codec_schemes(config.codec_schemes)
        if usable:
            scheme = usable[int(rng.integers(len(usable)))]
            result = _peak_limit(np.asarray(_codec_round_trip(result, scheme), dtype=np.float32))
            applied.append(f"codec:{scheme}")
        else:
            notes.append("codec_probability was set but no configured scheme is implemented")
    if rng.random() < config.channel_probability:
        result = _apply_channel(result, rng, sample_rate)
        applied.append("channel")
    if rng.random() < config.reverb_probability:
        result = _apply_reverb(result, rng, sample_rate)
        applied.append("reverb")
    if config.gain_db_min != 0.0 or config.gain_db_max != 0.0:
        result = _peak_limit(_apply_gain(result, rng, config))
        applied.append("gain")

    return AugmentedAudio(
        waveform=_peak_limit(np.asarray(result, dtype=np.float32)),
        applied=tuple(applied),
        seed=seed,
        unavailable=unavailable,
        notes=tuple(notes),
    )


def class_weights(
    labels: Sequence[int],
    config: AugmentationConfig | None = None,
) -> np.ndarray | None:
    """Per-sample weights for balanced sampling, or ``None`` when unneeded.

    Weights are returned rather than a resampled index list on purpose. A
    duplicated row would be counted twice in the dataset statistics and would
    trip the duplicate check that guards the corpus; weighting the draw achieves
    the same balance without inventing rows that are not in the manifest.

    Args:
        labels: Integer label for each sample.
        config: Augmentation configuration, read for ``class_balance``.

    Returns:
        One weight per sample summing to the sample count, or ``None`` when
        balancing is off or a class has no members.
    """
    if config is None or config.class_balance == BALANCE_NONE:
        return None
    if config.class_balance != BALANCE_WEIGHTED_SAMPLER:
        msg = f"unsupported class_balance {config.class_balance!r}"
        raise AugmentationError(msg)
    if not labels:
        return None
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=int(max(labels)) + 1)
    present = counts[counts > 0]
    if present.size == 0:
        return None
    inverse = np.zeros_like(counts, dtype=np.float64)
    for index in np.flatnonzero(counts > 0):
        inverse[index] = 1.0 / float(counts[index])
    return (inverse[np.asarray(labels, dtype=np.int64)] / present.size) * len(labels)
