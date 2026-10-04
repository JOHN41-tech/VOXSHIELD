"""Canonical audio preprocessing.

Turns validated :class:`~voxshield.audio.decode.DecodedAudio` into the single
internal representation the rest of VoxShield assumes: **16 kHz mono float32
PCM, DC-free, loudness-normalised, amplitude-clipped**.

The transform order is fixed and is part of the contract, because reordering
these steps changes the output:

1. ``sanitize``      -- replace non-finite values; a NaN that survives here
                        propagates through every downstream FFT as NaN.
2. ``to_mono``       -- downmix before resampling, so we do the resample work
                        once instead of once per channel.
3. ``remove_dc``     -- resampling a DC offset smears it across low frequencies
                        and can look like broadband noise to the detector.
4. ``resample``      -- polyphase rational resampling to the target rate.
5. ``normalize``     -- RMS loudness targeting, then a hard peak ceiling.

Every stage is a pure function of its input, so a given input file and code
version always produce byte-identical features. That reproducibility is what
makes an evaluation result meaningful six months later.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy import signal as scipy_signal

from voxshield.audio.decode import DecodedAudio
from voxshield.config import AudioConfig
from voxshield.errors import InvalidAudioSignalError

__all__ = [
    "PreprocessedAudio",
    "dbfs",
    "normalize_loudness",
    "preprocess",
    "preprocess_decoded",
    "remove_dc_offset",
    "resample_to",
    "rms",
    "sanitize",
    "to_mono",
]

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class PreprocessedAudio:
    """Canonical internal audio representation.

    Attributes:
        samples: ``float32``, shape ``(n,)``. 16 kHz mono.
        sample_rate: Always ``config.target_sample_rate``.
        gain_db_applied: Loudness gain that was applied. Recorded for audit so
            a downstream reviewer can tell a quiet genuine recording from a
            quiet synthetic one.
        peak_before_normalize: Peak amplitude prior to gain, useful for quality
            diagnostics.
        clipped: Whether the peak ceiling reduced the signal.
    """

    samples: np.ndarray
    sample_rate: int
    gain_db_applied: float
    peak_before_normalize: float
    clipped: bool

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds at the canonical rate."""
        return len(self.samples) / float(self.sample_rate)

    def __post_init__(self) -> None:
        if self.samples.size and not np.isfinite(self.samples).all():
            msg = "preprocessed audio contains non-finite samples"
            raise InvalidAudioSignalError(msg)


def dbfs(amplitude: float | np.ndarray) -> float | np.ndarray:
    """Convert linear amplitude to dBFS."""
    return 20.0 * np.log10(np.maximum(np.abs(amplitude), _EPS))


def rms(x: np.ndarray) -> float:
    """Root-mean-square amplitude of a 1-D signal."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def sanitize(x: np.ndarray) -> np.ndarray:
    """Return a copy with NaN/Inf replaced by zero.

    Input is never mutated. This is the first preprocessing stage precisely so
    that a single bad sample cannot turn an entire spectrogram into NaN and
    yield a confident, meaningless score.
    """
    out = np.array(x, dtype=np.float32, copy=True)
    if not np.isfinite(out).all():
        out[~np.isfinite(out)] = 0.0
    return out


def to_mono(x: np.ndarray) -> np.ndarray:
    """Downmix to mono.

    Accepts ``(n,)`` or ``(n, channels)``. Channel averaging is the standard
    choice for speech and preserves energy better than selecting one channel,
    which risks analysing a muted or noise-dominated track.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return np.ascontiguousarray(arr[:, 0], dtype=np.float32)
        return np.ascontiguousarray(arr.mean(axis=1), dtype=np.float32)
    msg = f"expected 1-D or 2-D audio, got shape {arr.shape}"
    raise InvalidAudioSignalError(msg)


def remove_dc_offset(x: np.ndarray) -> np.ndarray:
    """Subtract the mean sample value.

    Microphone and codec offsets are typically a few LSBs. Left in place they
    concentrate energy in bin 0, which both wastes a mel band and can leak into
    the lowest formant region the detector attends to.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return arr.copy()
    return np.ascontiguousarray(arr - np.float32(arr.mean()), dtype=np.float32)


def resample_to(x: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Resample a 1-D signal to ``target_rate`` via polyphase filtering.

    ``scipy.signal.resample_poly`` is used with an exact rational ratio rather
    than FFT-based resampling because it is faster, has no wrap-around artefacts,
    and its output depends only on the ratio -- so a 44.1 kHz input and a 48 kHz
    input at the same duration both land on a comparable 16 kHz grid.
    """
    if orig_rate == target_rate:
        return np.ascontiguousarray(x, dtype=np.float32)
    if orig_rate <= 0 or target_rate <= 0:
        msg = f"invalid sample rates: {orig_rate} -> {target_rate}"
        raise InvalidAudioSignalError(msg)

    ratio = Fraction(target_rate, orig_rate)
    return np.ascontiguousarray(
        scipy_signal.resample_poly(x, ratio.numerator, ratio.denominator),
        dtype=np.float32,
    )


def normalize_loudness(
    x: np.ndarray,
    config: AudioConfig,
) -> tuple[np.ndarray, float, float, bool]:
    """Apply the configured normalization policy.

    The policy comes from ``config.normalization_settings``, which resolves the
    legacy top-level targets and a fully specified :class:`NormalizationConfig`
    to one object, so this function has a single input to reason about. Three
    guards matter:

    * **Silence is not amplified.** Digital silence would otherwise be scaled
      into audible noise, and the gain would be reported as meaningful. The
      floor is the configured ``silence_rms`` rather than a literal so a caller
      can widen it deliberately.
    * **Gain is capped** at ``max_gain_db`` so a near-silent recording is not
      lifted into the noise floor and then confidently classified.
    * **The ceiling is a scale, not a clip.** Hard clipping would introduce the
      very inter-sample distortion :mod:`voxshield.audio.quality` is looking for.

    Strategies:

    * ``rms``  -- scale toward ``target_rms_dbfs`` (the historical behaviour).
    * ``peak`` -- scale toward ``target_peak`` instead, for material whose RMS
      is unrepresentative of its loudness, such as speech with long pauses.
    * ``none`` -- pass through, only bounded.

    Returns:
        ``(samples, gain_db_applied, peak_before, ceiling_applied)``.

        ``ceiling_applied`` is True when normalisation pushed the signal past
        ``peak_ceiling`` and the result had to be attenuated to fit. It is not
        the input-clipping measurement; that is
        :data:`voxshield.audio.quality.ISSUE_CLIPPED`.
    """
    settings = config.normalization_settings
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return arr.copy(), 0.0, 0.0, False

    peak_before = float(np.max(np.abs(arr)))

    if not settings.enabled or settings.strategy == "none":
        bounded = np.ascontiguousarray(np.clip(arr, -1.0, 1.0), dtype=np.float32)
        return bounded, 0.0, peak_before, False

    if settings.strategy == "peak":
        if peak_before <= settings.silence_rms:
            return arr.copy(), 0.0, peak_before, False
        gain_db = float(
            np.clip(
                20.0 * np.log10(settings.target_peak / peak_before),
                -settings.max_gain_db,
                settings.max_gain_db,
            )
        )
    else:
        current_rms = rms(arr)
        # Below the silence floor there is no speech to normalise; treat the
        # signal as digital silence and leave it alone.
        if current_rms <= settings.silence_rms:
            return arr.copy(), 0.0, peak_before, False
        gain_db = float(
            np.clip(
                settings.target_rms_dbfs - float(20.0 * np.log10(current_rms)),
                -settings.max_gain_db,
                settings.max_gain_db,
            )
        )

    out = arr * np.float32(10.0 ** (gain_db / 20.0))

    # ``ceiling_applied`` reports that the peak ceiling forced attenuation,
    # which is not the same event as the input clipping that
    # :mod:`voxshield.audio.quality` reports separately.
    ceiling_applied = False
    peak_after = float(np.max(np.abs(out)))
    if peak_after > settings.peak_ceiling:
        # Scale down to the ceiling rather than hard-clipping.
        out = out * np.float32(settings.peak_ceiling / peak_after)
        ceiling_applied = True

    out = np.ascontiguousarray(np.clip(out, -1.0, 1.0), dtype=np.float32)
    return out, gain_db, peak_before, ceiling_applied


def preprocess(
    samples: np.ndarray,
    sample_rate: int,
    config: AudioConfig | None = None,
) -> PreprocessedAudio:
    """Run the full preprocessing chain on a raw 1-D or 2-D signal.

    This is the entry point used by tests and by any caller that already holds
    a numpy array rather than encoded bytes.
    """
    cfg = config or AudioConfig()
    arr = sanitize(samples)
    arr = to_mono(arr)
    arr = remove_dc_offset(arr)
    arr = resample_to(arr, sample_rate, cfg.target_sample_rate)
    normalised, gain_db, peak_before, clipped = normalize_loudness(arr, cfg)
    return PreprocessedAudio(
        samples=normalised,
        sample_rate=cfg.target_sample_rate,
        gain_db_applied=gain_db,
        peak_before_normalize=peak_before,
        clipped=clipped,
    )


def preprocess_decoded(
    decoded: DecodedAudio,
    config: AudioConfig | None = None,
) -> PreprocessedAudio:
    """Run the full preprocessing chain on already-validated decoded audio."""
    return preprocess(decoded.samples, decoded.sample_rate, config)
