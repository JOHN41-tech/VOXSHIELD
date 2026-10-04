"""Frame-based voice activity detection.

The MVP VAD is deliberately **not** a neural detector. It is a deterministic
energy + zero-crossing-rate + spectral-flatness frame classifier, chosen so that
Phase 0 has zero model dependencies, fully reproducible output, and inspectable
failure modes. Silero-VAD is a Phase 1 upgrade (see
``docs/decision-log/ADR-002-vad-baseline.md``).

Two properties matter more than raw accuracy here:

* **Gain invariance.** Callers submit audio captured on unknown devices, and
  loudness normalisation happens upstream. A fixed absolute dBFS threshold would
  therefore be tuned to the loudest recording in the test set and would fail on
  the quietest. The threshold here is derived from the signal's own frame-energy
  distribution, so it tracks the recording instead of assuming a level.
* **Failure in the safe direction.** When uncertain the VAD reports *no* speech,
  which the pipeline surfaces as ``INSUFFICIENT_SPEECH`` and abstains. It never
  fabricates speech regions to justify a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from voxshield.audio.preprocess import sanitize
from voxshield.config import AudioConfig, VadConfig

__all__ = [
    "EnergyVadDetector",
    "SpeechMask",
    "SpeechRegion",
    "VadDetector",
    "compute_frame_features",
    "detect_speech",
    "find_speech_regions",
    "frame_signal",
    "region_waveforms",
]

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class SpeechMask:
    """Frame-level speech decisions plus derived timings.

    Attributes:
        is_speech: Boolean per frame.
        speech_ratio: Fraction of frames classified as speech.
        speech_seconds: Total speech duration after morphological cleanup.
        n_frames: Number of frames analysed.
        frame_seconds: Duration of one frame.
        hop_seconds: Duration of one hop.
        threshold_dbfs: The adaptive threshold actually used, for audit.
    """

    is_speech: np.ndarray
    speech_ratio: float
    speech_seconds: float
    n_frames: int
    frame_seconds: float
    hop_seconds: float
    threshold_dbfs: float

    @property
    def has_speech(self) -> bool:
        """Whether enough speech was found to justify a verdict."""
        return self.speech_seconds > 0.0

    def regions(self) -> list[tuple[float, float]]:
        """Speech regions as ``(start_seconds, end_seconds)`` intervals.

        Boundaries advance by the hop, not the frame length, since frame ``i``
        covers audio starting at ``i * hop_seconds``.
        """
        return [
            (start * self.hop_seconds, end * self.hop_seconds)
            for start, end in _runs(self.is_speech)
        ]

    def sample_mask(self, sample_rate: int, n_samples: int) -> np.ndarray:
        """Expand the frame mask to a per-sample boolean mask.

        Args:
            sample_rate: Canonical sample rate.
            n_samples: Length of the signal the mask should cover.

        Returns:
            Boolean array of length ``n_samples``.
        """
        if self.n_frames == 0:
            return np.zeros(n_samples, dtype=bool)
        samples_per_hop = max(1, round(self.hop_seconds * sample_rate))
        out: np.ndarray = np.zeros(n_samples, dtype=bool)
        for idx, flag in enumerate(self.is_speech):
            if not flag:
                continue
            start = idx * samples_per_hop
            stop = min(start + samples_per_hop, n_samples)
            out[start:stop] = True
        return out


def frame_signal(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Split ``x`` into overlapping frames of shape ``(n_frames, frame_length)``.

    Uses a strided view, so framing does not copy the signal.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.size < frame_length:
        return np.empty((0, frame_length), dtype=np.float32)
    n_frames = 1 + (arr.size - frame_length) // hop_length
    return sliding_window_view(arr, frame_length)[::hop_length][:n_frames]


def compute_frame_features(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame energy (dBFS), zero-crossing rate, and spectral flatness.

    Spectral flatness is the geometric mean of the magnitude spectrum divided
    by its arithmetic mean. Broadband noise and hiss sit near 1.0; voiced
    speech, which is strongly harmonic and peaky, sits far lower. It is a
    cheap, effective way to stop a VAD from locking onto a fan or line hiss.
    """
    if frames.size == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    window: np.ndarray = np.hanning(frames.shape[1]).astype(np.float32)
    windowed = frames * window

    energy = np.sqrt(np.mean(np.square(windowed, dtype=np.float64), axis=1))
    energy_db = (20.0 * np.log10(np.maximum(energy, _EPS))).astype(np.float32)

    # Zero crossings within each frame, normalised by frame length.
    signs = np.signbit(windowed)
    zcr = (np.diff(signs, axis=1) != 0).sum(axis=1) / float(frames.shape[1])
    zcr = zcr.astype(np.float32)

    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) + _EPS
    log_mean = np.log(spectrum).mean(axis=1)
    arith_mean = spectrum.mean(axis=1)
    flatness = np.exp(log_mean) / np.maximum(arith_mean, _EPS)
    flatness = np.clip(flatness, 0.0, 1.0).astype(np.float32)

    return energy_db, zcr, flatness


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``True`` runs as half-open ``(start, end)`` frame ranges."""
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(changes[0::2], changes[1::2], strict=True)]


def _drop_short_true_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Clear ``True`` runs shorter than ``min_len`` frames."""
    out = mask.copy()
    for start, end in _runs(out):
        if end - start < min_len:
            out[start:end] = False
    return out


def _fill_short_false_runs(mask: np.ndarray, max_len: int) -> np.ndarray:
    """Set ``True`` in interior ``False`` runs no longer than ``max_len`` frames."""
    out = mask.copy()
    n = out.size
    for start, end in _runs(~out):
        # A False run touching an edge is real leading/trailing silence, not an
        # internal pause, so it is never filled.
        if start == 0 or end == n:
            continue
        if end - start <= max_len:
            out[start:end] = True
    return out


def detect_speech(
    samples: np.ndarray,
    sample_rate: int,
    config: AudioConfig | None = None,
) -> SpeechMask:
    """Classify frames as speech or non-speech.

    Args:
        samples: 1-D mono audio at any sample rate.
        sample_rate: Sample rate of ``samples``.
        config: Pipeline configuration; the ``config.vad`` section is used.

    Returns:
        A :class:`SpeechMask`. Callers must treat ``speech_seconds`` as the
        authoritative measure of how much analysable speech was found.
    """
    cfg = config or AudioConfig()
    vad: VadConfig = cfg.vad

    # A single non-finite sample would poison np.percentile and np.max for the
    # whole file: the seed arm collapses, the threshold falls back to the
    # absolute floor, and the fallback path can then mark near-silence as
    # speech. Sanitising here keeps this function safe to call on its own,
    # rather than only after preprocessing has already run.
    arr = sanitize(np.asarray(samples, dtype=np.float32))
    frame_length = max(1, round(sample_rate * vad.frame_ms / 1000.0))
    hop_length = max(1, round(sample_rate * vad.hop_ms / 1000.0))
    frame_seconds = frame_length / float(sample_rate)
    hop_seconds = hop_length / float(sample_rate)

    frames = frame_signal(arr, frame_length, hop_length)
    if frames.shape[0] == 0:
        return SpeechMask(
            is_speech=np.zeros(0, dtype=bool),
            speech_ratio=0.0,
            speech_seconds=0.0,
            n_frames=0,
            frame_seconds=frame_seconds,
            hop_seconds=hop_seconds,
            threshold_dbfs=vad.absolute_floor_dbfs,
        )

    energy_db, zcr, flatness = compute_frame_features(frames)

    spectral_ok = (zcr <= vad.max_zero_crossing_rate) & (flatness <= vad.max_spectral_flatness)

    # --- Seed arm: find frames that are confidently speech -----------------
    seed_level = float(np.percentile(energy_db, vad.seed_percentile))
    seed_threshold = seed_level - vad.seed_margin_db
    seed_mask = (energy_db >= seed_threshold) & spectral_ok

    if np.any(seed_mask):
        # Reference is the quiet end of the confident-speech frames, not its
        # median, so the spread arm is anchored to the weakest speech we are
        # already sure about.
        reference = float(np.percentile(energy_db[seed_mask], vad.seed_reference_percentile))
    else:
        # No frame is confident speech. Fall back to the loudest frame in the
        # clip so the threshold still adapts, and let the spectral gates and the
        # floor decide. This is the abstention-leaning path.
        reference = float(np.max(energy_db))

    # --- Spread arm: admit quieter speech, floored at the absolute floor ----
    threshold_dbfs = max(vad.absolute_floor_dbfs, reference - vad.dynamic_range_db)

    is_speech = (energy_db >= threshold_dbfs) & spectral_ok

    min_speech_frames = max(1, round(vad.min_speech_duration_s / hop_seconds))
    min_silence_frames = max(0, round(vad.min_silence_duration_s / hop_seconds))

    is_speech = _drop_short_true_runs(is_speech, min_speech_frames)
    is_speech = _fill_short_false_runs(is_speech, min_silence_frames)

    n_speech = int(np.count_nonzero(is_speech))
    speech_ratio = n_speech / float(is_speech.size)
    # Each speech frame represents hop_samples of audio.
    speech_seconds = n_speech * hop_seconds

    return SpeechMask(
        is_speech=is_speech,
        speech_ratio=speech_ratio,
        speech_seconds=speech_seconds,
        n_frames=int(is_speech.size),
        frame_seconds=frame_seconds,
        hop_seconds=hop_seconds,
        threshold_dbfs=float(threshold_dbfs),
    )


@dataclass(frozen=True, slots=True)
class SpeechRegion:
    """A contiguous span of speech, in seconds, ready to be windowed.

    A frame-level run is not yet a usable region. Onset energy climbs over tens
    of milliseconds and a final syllable decays out of the threshold, so a raw
    run systematically clips the start and end of the utterance it found.
    Regions produced by :func:`find_speech_regions` are padded and merged before
    they are returned, which is the whole point of this type existing separately
    from the raw ``(start, end)`` tuples.

    Attributes:
        start_s: Inclusive start, in seconds from the beginning of the signal.
        end_s: Exclusive end, in seconds.
    """

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """Length of the region in seconds. Never negative."""
        return max(0.0, self.end_s - self.start_s)

    def to_samples(self, sample_rate: int) -> tuple[int, int]:
        """Sample bounds ``[start, end)``, clamped to a non-negative range.

        Flooring the start and ceiling the end guarantees the window contains
        real audio, which matters at 0.0 where rounding down would produce
        ``slice(0, -1)`` and silently return the whole signal minus one sample.
        """
        start = max(0, round(self.start_s * sample_rate))
        end = max(start, round(self.end_s * sample_rate))
        return start, end

    def as_metadata(self) -> dict[str, float]:
        """Rounded timings for logs and audit records."""
        return {
            "region_start_seconds": round(self.start_s, 4),
            "region_end_seconds": round(self.end_s, 4),
            "region_duration_seconds": round(self.duration_s, 4),
        }


def find_speech_regions(
    mask: SpeechMask,
    sample_rate: int,
    n_samples: int,
    config: AudioConfig | None = None,
) -> list[SpeechRegion]:
    """Turn frame-level speech decisions into analysis-ready regions.

    The steps are ordered deliberately:

    1. **Pad** each run by ``config.vad.region_padding_s`` on both sides. The
       padding is allowed to run past 0 or past the end of the signal here, so a
       region that starts at the very first frame is still extended backwards
       rather than left clipped.
    2. **Merge** regions separated by no more than ``config.vad.region_merge_gap_s``
       so one utterance split by a pause is not scored as two.
    3. **Drop** regions shorter than ``config.vad.min_region_seconds``. A window
       built from a region this short carries no more evidence than the frame
       mask already did, and admitting it would hand the model a segment to
       confidently misread.
    4. **Clamp** to the signal and discard anything left empty.

    Args:
        mask: Frame decisions from :func:`detect_speech`.
        sample_rate: Sample rate of the signal the mask describes.
        n_samples: Length of that signal in samples.
        config: Pipeline configuration; the ``config.vad`` section is used.

    Returns:
        Regions in ascending time order. Empty when the mask holds no speech.
    """
    cfg = config or AudioConfig()
    vad: VadConfig = cfg.vad

    padded = [
        (start - vad.region_padding_s, end + vad.region_padding_s) for start, end in mask.regions()
    ]
    if not padded:
        return []

    merged: list[list[float]] = []
    for start, end in sorted(padded):
        if merged and start - merged[-1][1] <= vad.region_merge_gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    duration_s = n_samples / float(sample_rate) if sample_rate > 0 else 0.0
    out: list[SpeechRegion] = []
    for start, end in merged:
        clamped_start = min(max(0.0, start), duration_s)
        clamped_end = min(max(0.0, end), duration_s)
        region = SpeechRegion(start_s=clamped_start, end_s=clamped_end)
        if region.duration_s < vad.min_region_seconds:
            continue
        if region.duration_s <= 0.0:
            continue
        out.append(region)
    return out


def region_waveforms(
    samples: np.ndarray,
    regions: list[SpeechRegion],
    sample_rate: int,
) -> list[np.ndarray]:
    """Slice ``samples`` into one waveform per region.

    Each returned array is a view where possible, not a copy, so extracting
    regions from a long recording does not double peak memory. Bounds are
    clamped to the signal, so a region that falls entirely outside it yields an
    empty array rather than an exception or a wrapped slice.
    """
    arr = np.asarray(samples)
    total = arr.shape[0] if arr.ndim else 0
    out: list[np.ndarray] = []
    for region in regions:
        start, end = region.to_samples(sample_rate)
        start = max(0, min(start, total))
        end = max(start, min(end, total))
        out.append(arr[start:end])
    return out


@runtime_checkable
class VadDetector(Protocol):
    """The seam a detector must satisfy to be swappable.

    Declared so that a neural detector (Silero, per
    ``docs/decision-log/ADR-002-vad-baseline.md``) can replace the energy
    baseline without touching the segmentation, pipeline, or API layers. The
    contract is narrow on purpose: a frame-level mask and nothing else. Anything
    a caller needs beyond that is a place where the interface is wrong.
    """

    def detect(self, samples: np.ndarray, sample_rate: int) -> SpeechMask:
        """Classify frames of ``samples`` as speech or non-speech."""
        ...


@dataclass(frozen=True, slots=True)
class EnergyVadDetector:
    """The deterministic energy baseline, packaged behind :class:`VadDetector`.

    Holds no state beyond its configuration, so the same instance is safe to
    share and two calls with the same input return equal results.
    """

    config: AudioConfig | None = None

    @property
    def settings(self) -> AudioConfig:
        """Configuration to use, defaulting to the code defaults."""
        return self.config if self.config is not None else AudioConfig()

    @property
    def name(self) -> str:
        """Identifier recorded in audit records alongside the decisions."""
        return "energy"

    def detect(self, samples: np.ndarray, sample_rate: int) -> SpeechMask:
        """Run the energy baseline over ``samples``."""
        return detect_speech(samples, sample_rate, self.settings)

    def regions(
        self,
        samples: np.ndarray,
        sample_rate: int,
    ) -> list[SpeechRegion]:
        """Detect speech and return post-processed regions in one call.

        Convenience for the common path, so a caller does not have to remember
        that regions are meaningless without the frame mask they came from.
        """
        arr = np.asarray(samples)
        mask = self.detect(samples, sample_rate)
        n_samples = int(arr.shape[0]) if arr.ndim else 0
        return find_speech_regions(mask, sample_rate, n_samples, self.settings)
