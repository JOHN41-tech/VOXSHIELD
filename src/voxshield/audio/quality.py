"""Structured audio quality assessment.

Preprocessing answers "what shape is this audio?" -- rate, channels, layout.
This module answers the separate question "how much should a human trust a
verdict about it?", and it answers it in numbers rather than prose:

* **How much of this is actually speech-shaped audio** (``silence_ratio``,
  ``snr_db``). A call that is 80% line hiss produces a confident score from a
  window that carries almost no speech evidence, and the number that matters to
  an analyst is the one that says so.
* **How was it captured** (``peak_amplitude``, ``rms_dbfs``,
  ``clipping_ratio``, ``dc_offset``, ``crest_factor_db``). Clipping and a DC
  offset are recording-chain artefacts. Both distort the spectral cues an
  anti-spoofing model reads, so a clipped clip is evidence about the microphone,
  not about the voice.

Two properties are deliberate. First, the metrics are computed from whatever the
caller passes, *including* a signal with NaN or Inf, because a quality report
that refuses to describe a broken signal is useless for diagnosing it -- the
canonical stage is where non-finite values are scrubbed. Second, the report is
**data, not a verdict**: :attr:`QualityReport.issues` is a tuple of stable
codes, and only the caller decides whether a code abstains, warns, or is
ignored. :attr:`QualityReport.is_scorable` exposes the one distinction that is
not a judgement call, namely whether the signal can be analysed at all.

The report holds scalars only. It is safe to log, attach to an audit record, and
serialise, and it is the only part of a processing run that is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from voxshield.audio.vad import compute_frame_features, frame_signal
from voxshield.config import MIN_MEASURABLE_SNR_DB, AudioConfig, QualityConfig

__all__ = [
    "ISSUE_CLIPPED",
    "ISSUE_DC_OFFSET",
    "ISSUE_HIGH_CREST_FACTOR",
    "ISSUE_LOW_LEVEL",
    "ISSUE_LOW_SNR",
    "ISSUE_NON_FINITE",
    "ISSUE_SILENT",
    "QualityReport",
    "assess_quality",
]

# Stable issue codes. These are contractual: an operator's dashboard and any
# alert built on them match on the string, so they are named once here and never
# assembled at a call site.
ISSUE_NON_FINITE = "NON_FINITE_SAMPLES"
ISSUE_SILENT = "SILENT"
ISSUE_CLIPPED = "CLIPPED"
ISSUE_LOW_LEVEL = "LOW_LEVEL"
ISSUE_LOW_SNR = "LOW_SNR"
ISSUE_DC_OFFSET = "DC_OFFSET"
ISSUE_HIGH_CREST_FACTOR = "HIGH_CREST_FACTOR"

# Reported instead of NaN or +/-inf, so a degenerate signal still produces a
# JSON-serialisable report. Chosen to sit outside every plausible real level.
_FLOOR = -120.0

_EPS = 1e-12

# Below this many frames, percentiles are dominated by which single frame landed
# nearest a rank boundary, so an SNR figure computed from them is noise.
_MIN_FRAMES_FOR_SNR = 3


def _to_dbfs(amplitude: float) -> float:
    """Linear amplitude to dBFS, floored so the result is always finite."""
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        return _FLOOR
    return max(_FLOOR, float(20.0 * np.log10(amplitude)))


def _estimate_snr_db(energy_db: np.ndarray, silence_dbfs: float) -> float | None:
    """Estimate speech-to-noise ratio, or ``None`` when it cannot be measured.

    The estimator is the 95th percentile frame level minus the 10th percentile
    frame level, over *non-silent* frames only. Both restrictions are load
    bearing:

    * **Digital silence is excluded.** A clean recording with pauses in it has
      frames at the -120 dBFS floor. Treating those as the noise floor reports
      hundreds of dB of SNR for a signal containing no noise at all -- an
      impressive number that is exactly backwards.
    * **A degenerate spread is reported as unknown, not as 0 dB.** If every
      non-silent frame sits at the same level there is no floor to measure
      against, and "no measurable SNR" is the true statement. A pure tone is
      clean; broadband hiss is not; the two are indistinguishable from the
      signal alone, and the honest answer is that the ratio is undefined rather
      than a number that reads as a verdict.

    Percentiles rather than a mean because a handful of loud transients would
    otherwise set the "speech" level for the whole clip.
    """
    if energy_db.size < _MIN_FRAMES_FOR_SNR:
        return None
    non_silent = energy_db[energy_db > silence_dbfs]
    if non_silent.size < _MIN_FRAMES_FOR_SNR:
        return None
    speech_level = float(np.percentile(non_silent, 95))
    noise_level = float(np.percentile(non_silent, 10))
    spread = speech_level - noise_level
    if spread < MIN_MEASURABLE_SNR_DB:
        return None
    return spread


@dataclass(frozen=True, slots=True)
class QualityReport:
    """Metadata-only quality metrics for one clip.

    Every field is a scalar. There is no attribute that can hold a sample, so
    this object cannot leak audio the way a raw array or a bytes payload can --
    the privacy property is structural, not a convention.

    Attributes:
        sample_rate_hz: Sample rate the metrics were computed at.
        sample_count: Number of samples assessed.
        duration_seconds: ``sample_count / sample_rate_hz``.
        peak_amplitude: Largest absolute sample value.
        rms_amplitude: Root-mean-square level.
        rms_dbfs: ``rms_amplitude`` in dBFS.
        crest_factor_db: ``peak_amplitude / rms_amplitude`` in dB. A high value
            means a signal dominated by transients.
        dc_offset: Mean sample value, i.e. the residual DC component.
        silence_ratio: Fraction of frames at or below
            ``config.quality.silence_dbfs``.
        clipping_ratio: Fraction of samples at or above
            ``config.quality.clip_threshold``.
        non_finite_ratio: Fraction of samples that are NaN or infinite.
        snr_db: Estimated speech-to-noise ratio in dB, or ``None`` when the
            signal has no measurable noise floor. Percentiles rather than a mean,
            so that a few loud transients do not invent a high SNR. A stationary
            signal -- a pure tone, a steady hum -- has no floor to measure
            against and reports ``None``, which is a different statement from
            "0 dB": it means the ratio is undefined, not that the clip is noise.
            Exported as JSON ``null``, never as a misleading finite number.
        speech_frame_ratio: Fraction of frames above
            ``config.quality.silence_dbfs``. The complement of the useful part of
            ``silence_ratio``, provided because "how much of this is worth
            scoring" is the question a caller actually asks.
        issues: Stable issue codes, empty when nothing was flagged.
        thresholds: The thresholds that produced ``issues``, so a reader can tell
            a wrong assessment from an unlucky configuration.
    """

    sample_rate_hz: int
    sample_count: int
    duration_seconds: float
    peak_amplitude: float
    rms_amplitude: float
    rms_dbfs: float
    crest_factor_db: float
    dc_offset: float
    silence_ratio: float
    clipping_ratio: float
    non_finite_ratio: float
    snr_db: float | None
    speech_frame_ratio: float
    issues: tuple[str, ...]
    thresholds: QualityConfig

    @property
    def is_silent(self) -> bool:
        """Whether the clip carries no usable signal."""
        return ISSUE_SILENT in self.issues

    @property
    def has_clipping(self) -> bool:
        """Whether any sample reached the clipping threshold."""
        return ISSUE_CLIPPED in self.issues

    @property
    def is_scorable(self) -> bool:
        """Whether the signal can be analysed at all.

        Only a signal that is silent or non-finite fails this. Everything else
        -- clipping, low SNR, a DC offset -- is a reason to *distrust* a score,
        not a reason to refuse to produce one, and collapsing those two would
        make the detector abstain on recoverable audio.
        """
        return not self.blocking_issues

    @property
    def blocking_issues(self) -> tuple[str, ...]:
        """Issues that make the signal unusable rather than merely suspect."""
        blocking = {ISSUE_SILENT, ISSUE_NON_FINITE}
        return tuple(code for code in self.issues if code in blocking)

    @property
    def warnings(self) -> tuple[str, ...]:
        """Issues that describe a low-confidence clip without blocking it."""
        blocking = set(self.blocking_issues)
        return tuple(code for code in self.issues if code not in blocking)

    def as_metadata(self) -> dict[str, Any]:
        """Rounded, prefixed metrics for logs, audit records, and CLI output.

        The ``quality_`` prefix keeps these keys from colliding with the
        top-level pipeline metadata, and the rounding keeps a report small enough
        to read in a log line. Values are rounded, not truncated: a quality
        figure that is nearly the same is not the same.
        """
        return {
            "quality_duration_seconds": round(self.duration_seconds, 4),
            "quality_peak_amplitude": round(self.peak_amplitude, 6),
            "quality_rms_dbfs": round(self.rms_dbfs, 2),
            "quality_crest_factor_db": round(self.crest_factor_db, 2),
            "quality_dc_offset": round(self.dc_offset, 8),
            "quality_silence_ratio": round(self.silence_ratio, 4),
            "quality_speech_frame_ratio": round(self.speech_frame_ratio, 4),
            "quality_clipping_ratio": round(self.clipping_ratio, 6),
            "quality_non_finite_ratio": round(self.non_finite_ratio, 6),
            "quality_snr_db": None if self.snr_db is None else round(self.snr_db, 2),
            "quality_issues": list(self.issues),
            "quality_scorable": self.is_scorable,
        }


def _frame_energy_db(samples: np.ndarray, sample_rate: int, frame_ms: float) -> np.ndarray:
    """Per-frame energy in dBFS, on the same scale the VAD thresholds use.

    Reuses the VAD's framing and feature computation rather than reimplementing
    them: a quality report whose "quiet" boundary disagrees with the VAD's would
    let a clip be simultaneously "mostly silence" and "full of speech", and the
    two numbers would be impossible to reconcile.
    """
    frame_length = max(1, round(sample_rate * frame_ms / 1000.0))
    hop_length = max(1, round(sample_rate * frame_ms / 2000.0))
    frames = frame_signal(samples, frame_length, hop_length)
    if frames.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    energy_db, _, _ = compute_frame_features(frames)
    return energy_db


def assess_quality(
    samples: np.ndarray,
    sample_rate: int,
    config: AudioConfig | None = None,
    *,
    quality: QualityConfig | None = None,
) -> QualityReport:
    """Measure a clip and flag what would make a verdict untrustworthy.

    Args:
        samples: 1-D or 2-D audio. Non-finite values are counted, not rejected,
            so that a broken signal can be described.
        sample_rate: Sample rate of ``samples`` in Hz.
        config: Pipeline configuration; the ``config.quality`` section is used
            unless ``quality`` overrides it.
        quality: Explicit thresholds, for callers assessing a signal outside the
            pipeline.

    Returns:
        A :class:`QualityReport`. Callers decide what the issues mean; see
        :attr:`QualityReport.is_scorable` for the one distinction that is not a
        judgement call.
    """
    cfg = config or AudioConfig()
    thresholds = quality or cfg.quality

    arr = np.asarray(samples)
    flat = np.ravel(arr).astype(np.float32, copy=False)

    sample_count = int(flat.size)
    duration = sample_count / float(sample_rate) if sample_rate > 0 else 0.0

    if sample_count == 0:
        return QualityReport(
            sample_rate_hz=int(sample_rate),
            sample_count=0,
            duration_seconds=0.0,
            peak_amplitude=0.0,
            rms_amplitude=0.0,
            rms_dbfs=_FLOOR,
            crest_factor_db=0.0,
            dc_offset=0.0,
            silence_ratio=1.0,
            clipping_ratio=0.0,
            non_finite_ratio=0.0,
            snr_db=None,
            speech_frame_ratio=0.0,
            issues=(ISSUE_SILENT,),
            thresholds=thresholds,
        )

    finite_mask = np.isfinite(flat)
    non_finite_ratio = float(1.0 - np.count_nonzero(finite_mask) / sample_count)
    # Metrics are computed over finite samples only. Including NaN in a mean
    # would poison every other figure in the report, and one non-finite sample
    # is not evidence that the other 15,999 are unusable.
    finite = np.where(finite_mask, flat, 0.0).astype(np.float32, copy=False)

    peak = float(np.max(np.abs(finite))) if sample_count else 0.0
    rms_amplitude = float(np.sqrt(np.mean(np.square(finite, dtype=np.float64))))
    dc_offset = float(np.mean(finite, dtype=np.float64))
    clipping_ratio = float(
        np.count_nonzero(np.abs(finite) >= thresholds.clip_threshold) / sample_count
    )

    rms_dbfs = _to_dbfs(rms_amplitude)
    if rms_amplitude > _EPS and peak > _EPS:
        crest_factor_db = max(0.0, _to_dbfs(peak) - _to_dbfs(rms_amplitude))
    else:
        crest_factor_db = 0.0

    energy_db = _frame_energy_db(finite, sample_rate, thresholds.frame_ms)
    if energy_db.size:
        quiet = energy_db <= thresholds.silence_dbfs
        silence_ratio = float(np.count_nonzero(quiet) / energy_db.size)
        speech_frame_ratio = 1.0 - silence_ratio
    else:
        # Shorter than one analysis frame: fall back to whole-signal level so a
        # short clip is still described rather than reported as all-silence.
        silence_ratio = 1.0 if rms_dbfs <= thresholds.silence_dbfs else 0.0
        speech_frame_ratio = 1.0 - silence_ratio

    snr_db = _estimate_snr_db(energy_db, thresholds.silence_dbfs)

    issues: list[str] = []
    if non_finite_ratio > 0.0:
        issues.append(ISSUE_NON_FINITE)
    if peak <= 0.0 or silence_ratio >= thresholds.max_silence_ratio:
        issues.append(ISSUE_SILENT)
    if clipping_ratio > 0.0:
        issues.append(ISSUE_CLIPPED)
    if rms_dbfs < thresholds.low_level_dbfs:
        issues.append(ISSUE_LOW_LEVEL)
    if snr_db is not None and snr_db < thresholds.min_snr_db:
        issues.append(ISSUE_LOW_SNR)
    if abs(dc_offset) > thresholds.dc_offset_limit:
        issues.append(ISSUE_DC_OFFSET)
    if crest_factor_db > thresholds.max_crest_factor_db:
        issues.append(ISSUE_HIGH_CREST_FACTOR)
    if not thresholds.enabled:
        issues = []

    return QualityReport(
        sample_rate_hz=int(sample_rate),
        sample_count=sample_count,
        duration_seconds=duration,
        peak_amplitude=peak,
        rms_amplitude=rms_amplitude,
        rms_dbfs=rms_dbfs,
        crest_factor_db=crest_factor_db,
        dc_offset=dc_offset,
        silence_ratio=silence_ratio,
        clipping_ratio=clipping_ratio,
        non_finite_ratio=non_finite_ratio,
        snr_db=snr_db,
        speech_frame_ratio=speech_frame_ratio,
        issues=tuple(issues),
        thresholds=thresholds,
    )
