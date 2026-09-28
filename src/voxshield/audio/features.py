"""Log-mel spectrogram features.

The CNN baseline from ``docs/architecture.md`` consumes a log-mel spectrogram.
This module computes it in numpy so that the feature path has no torch
dependency: features must be computable, and unit-testable, in an environment
where no model is loaded.

:func:`compute_log_mel` is the single source of truth for VoxShield features.
Training and inference both call it. A future torch implementation is not
adopted as the reference, because two implementations that differ in binning
detail produce a model trained on one representation and served another -- a
failure that is invisible until the model silently degrades in production.
``tests/unit/test_features.py`` pins the filterbank's structural invariants
instead.

Parameters default to ``n_fft=400``, ``hop_length=160``, ``n_mels=80``, HTK mel
scale, 20 Hz to 7600 Hz -- the conventional anti-spoofing configuration at
16 kHz.

Two deliberate choices, both documented in ``docs/architecture.md``:

* **Minimum one-bin band width.** HTK mel bands are near-linear in Hz at low
  frequency, so the lowest edges collapse onto a single 40 Hz FFT bin. The
  textbook triangular ramp then evaluates that bin to ``0.0`` and the band is
  silently dead. See :func:`_enforce_min_band_width`.
* **No ``top_db`` relative clipping.** Clipping each clip relative to its own
  maximum would make one frame's value depend on whether louder frames happen to
  be in the same clip. A fixed floor keeps a frame's value a function of that
  frame alone, which is what reproducibility and per-window attribution need.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from voxshield.config import FeatureConfig

__all__ = [
    "compute_log_mel",
    "hz_to_mel",
    "log_mel_spectrogram",
    "mel_filterbank",
    "mel_to_hz",
    "stft_magnitude",
]

_EPS = 1e-12


def hz_to_mel(freq_hz: np.ndarray | float) -> np.ndarray | float:
    """Convert Hz to mels using the HTK formula."""
    return 2595.0 * np.log10(1.0 + np.asarray(freq_hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    """Convert mels to Hz using the HTK formula."""
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Build a peak-normalised triangular mel filterbank.

    Args:
        sample_rate: Sample rate in Hz.
        n_fft: FFT size; the number of frequency bins is ``n_fft // 2 + 1``.
        n_mels: Number of mel bands.
        f_min: Lowest frequency of the lowest band, in Hz.
        f_max: Highest frequency of the highest band, in Hz.

    Returns:
        Float32 array of shape ``(n_freqs, n_mels)`` mapping a power spectrum
        to mel-band energies.
    """
    n_freqs = n_fft // 2 + 1
    if f_max <= f_min:
        msg = f"f_max ({f_max}) must exceed f_min ({f_min})"
        raise ValueError(msg)
    if f_max > sample_rate / 2.0:
        f_max = sample_rate / 2.0

    mel_min = float(hz_to_mel(f_min))
    mel_max = float(hz_to_mel(f_max))
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = np.asarray(mel_to_hz(mel_points), dtype=np.float64)

    bin_index = np.floor((n_fft + 1) * hz_points / sample_rate).astype(np.int64)
    bin_index = _enforce_min_band_width(np.clip(bin_index, 0, n_freqs - 1), n_freqs)

    fb: np.ndarray = np.zeros((n_freqs, n_mels), dtype=np.float64)
    for i in range(n_mels):
        left = int(bin_index[i])
        centre = int(bin_index[i + 1])
        right = int(bin_index[i + 2])
        if centre > left:
            fb[left:centre, i] = (np.arange(left, centre) - left) / (centre - left)
        if right > centre:
            fb[centre:right, i] = (right - np.arange(centre, right)) / (right - centre)
        else:
            # Only reachable if _enforce_min_band_width could not widen the
            # band because it was pinned at the top of the spectrum. Give it a
            # unit gain at the centre rather than leaving it numerically dead.
            fb[min(centre, n_freqs - 1), i] = 1.0

    return fb.astype(np.float32)


def _enforce_min_band_width(bin_index: np.ndarray, n_freqs: int) -> np.ndarray:
    """Make mel bin edges strictly increasing by at least one bin.

    Without this step the filterbank is quietly malformed. HTK mel bands are
    near-linear in Hz at low frequency -- roughly 3.7 Hz wide near 20 Hz --
    while the FFT resolution at ``n_fft=400`` and 16 kHz is 40 Hz per bin, so
    the lowest mel edges land on the same FFT bin. A band whose edges collapse
    to ``(l, l+1, l+1)`` evaluates under the ramp
    ``(arange(l, c) - l) / (c - l)`` to a single bin holding ``0.0``: the band
    is numerically dead while the filterbank still reports it as present. At the
    default 80-band configuration that silently kills bands 0, 2, 5, 8, and 12
    -- 6% of a CNN's input channels carrying no information.

    Enforcing a one-bin minimum span makes every band a real triangle. The
    forward pass moves later edges up by at most one bin per collapsed edge,
    which at the default configuration moves the top edge from 190 to 197,
    still inside the 201 available bins, so no band is lost at the top.
    """
    out: np.ndarray = bin_index.astype(np.int64, copy=True)
    for i in range(1, out.size):
        if out[i] < out[i - 1] + 1:
            out[i] = out[i - 1] + 1
    # If the widening overflowed the spectrum, walk it back down while keeping
    # every band at least one bin wide where the spectrum allows.
    overflow = int(out[-1]) - (n_freqs - 1)
    if overflow > 0:
        for i in range(out.size - 1, -1, -1):
            out[i] = out[i] - overflow
            if i < out.size - 1 and out[i] > out[i + 1] - 1:
                out[i] = out[i + 1] - 1
            else:
                break
        out = np.clip(out, 0, n_freqs - 1)
    return out


def stft_magnitude(
    x: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
    *,
    center: bool = True,
) -> np.ndarray:
    """Magnitude STFT of a 1-D signal.

    Args:
        x: 1-D signal.
        n_fft: FFT size.
        hop_length: Samples between successive frames.
        win_length: Window length; padded symmetrically to ``n_fft`` if smaller.
        center: Pad the signal by ``n_fft // 2`` on both sides so frame ``t`` is
            centred on sample ``t * hop_length``.

    Returns:
        Float32 array of shape ``(n_frames, n_fft // 2 + 1)``.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 1:
        msg = f"stft expects a 1-D signal, got shape {arr.shape}"
        raise ValueError(msg)
    if arr.size == 0:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float32)

    if center:
        pad = n_fft // 2
        # Written as two branches rather than a `mode` variable: np.pad's mode
        # parameter is a Literal, so a str-typed variable does not select an
        # overload.
        if arr.size > pad:
            arr = np.pad(arr, (pad, pad), mode="reflect")
        else:
            arr = np.pad(arr, (pad, pad), mode="constant")

    window: np.ndarray = np.hanning(win_length).astype(np.float32)
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        window = np.pad(window, (left, n_fft - win_length - left))
    elif win_length > n_fft:
        window = window[:n_fft]

    if arr.size < n_fft:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float32)

    n_frames = 1 + (arr.size - n_fft) // hop_length
    frames = sliding_window_view(arr, n_fft)[::hop_length][:n_frames]
    spectrum = np.fft.rfft(frames * window, axis=-1)
    return np.abs(spectrum).astype(np.float32)


def log_mel_spectrogram(
    magnitude: np.ndarray,
    filterbank: np.ndarray,
    *,
    log_eps: float = 1e-6,
) -> np.ndarray:
    """Project a magnitude spectrogram to log-mel bands.

    Power (not magnitude) is used inside the filterbank, matching standard
    anti-spoofing feature extraction.
    """
    if magnitude.size == 0:
        return np.zeros((0, filterbank.shape[1]), dtype=np.float32)
    power = np.square(magnitude, dtype=np.float64)
    mel = power @ filterbank.astype(np.float64)
    return np.log(mel + log_eps).astype(np.float32)


def _cmvn(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-utterance cepstral mean and variance normalisation.

    Removes the recording-conditions fingerprint (channel, gain, room) that
    would otherwise let a detector key on microphone quality rather than on
    whether the speech is synthetic.
    """
    if features.size == 0:
        return features
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return ((features - mean) / (std + eps)).astype(np.float32)


def compute_log_mel(
    samples: np.ndarray,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Compute the log-mel spectrogram of a preprocessed mono signal.

    Args:
        samples: 1-D audio at ``config.sample_rate``.
        config: Feature parameters.

    Returns:
        Float32 array of shape ``(n_frames, n_mels)``.
    """
    cfg = config or FeatureConfig()
    arr = np.asarray(samples, dtype=np.float32)

    if arr.size == 0:
        return np.zeros((0, cfg.n_mels), dtype=np.float32)

    if not np.isfinite(arr).all():
        # A NaN sample survives every downstream FFT as NaN, and CMVN then spreads
        # it across every band and every frame -- turning one bad sample from a
        # codec into a confident, entirely meaningless spectrogram. Replace
        # rather than raise: a handful of bad samples should not discard an
        # otherwise analysable recording, and decode/preprocess already reject
        # non-finite audio upstream, so this is defence in depth.
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if cfg.preemphasis > 0.0:
        # Pre-emphasis flattens the -6 dB/octave source tilt so the mel bands
        # are not dominated by low frequencies.
        emphasised: np.ndarray = np.append(
            arr[:1] * (1.0 - cfg.preemphasis),
            arr[1:] - cfg.preemphasis * arr[:-1],
        ).astype(np.float32)
        arr = emphasised

    magnitude = stft_magnitude(
        arr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        center=cfg.center,
    )
    if magnitude.size == 0:
        return np.zeros((0, cfg.n_mels), dtype=np.float32)

    fb = mel_filterbank(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )
    features = log_mel_spectrogram(magnitude, fb, log_eps=cfg.log_eps)

    if cfg.per_utterance_cmvn:
        features = _cmvn(features)

    return features
