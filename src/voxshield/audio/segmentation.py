"""Speech segmentation into fixed-length analysis windows.

The detector scores fixed-width windows, not whole calls, so segmentation
determines what the model actually sees. Three decisions are locked in here:

* **Windows overlap.** The default hop is half the window, i.e. 50% overlap, so
  a 3-second clip yields several scorable windows instead of one, which matters
  when the only thing Phase 0 has is a small evaluation set and we need every
  usable observation. The hop is configurable via ``config.segment_hop_seconds``
  for callers with latency or cost budgets that make that trade differently.
* **A window is kept only if it contains at least
  ``config.min_segment_seconds`` of actual speech.** Padding a quiet window up
  to full width would feed the model mostly silence, and silence-dominated
  inputs are exactly where anti-spoofing models become confidently wrong.
* **A short window is possible only for a clip shorter than one window.** Every
  window the ``range`` loop produces ends at or before the signal end, and a
  final window is anchored at exactly ``n_samples - window``, so the tail of a
  long clip is always covered by a full-width window rather than a fragment.
  That is why the tail anchor exists: without it, everything after the last full
  step would be discarded. The consequence is that
  ``config.short_segment_policy`` only ever has a decision to make when the
  entire clip is under ``segment_seconds`` -- a 3-second recording analysed at
  the default 4-second window. ``drop`` discards it, ``pad`` zero-fills it to
  full width, and ``keep`` scores it at its natural width. ``drop`` is the
  default because it is the only policy that never shows the model a window
  containing audio it did not receive.

The consequence of overlapping windows is that per-window scores are correlated
with their neighbours. Aggregation must therefore not treat them as independent
samples; :func:`aggregate_segment_scores` documents how it handles that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voxshield.audio.vad import SpeechMask
from voxshield.config import AudioConfig

__all__ = ["Segment", "aggregate_segment_scores", "segment_speech"]


@dataclass(frozen=True, slots=True)
class Segment:
    """One analysis window.

    Attributes:
        start_sample: Inclusive start index at the canonical sample rate.
        end_sample: Exclusive end index at the canonical sample rate. For a
            padded segment this is past the end of the signal, which is the
            signal to the consumer that the tail must be zero-filled.
        speech_seconds: Speech content inside the window.
        coverage: ``speech_seconds`` divided by window duration, in ``[0, 1]``.
            The denominator is the *reported* window length, so a padded segment
            reports the low coverage it actually has instead of the flattering
            coverage of the audio that happened to exist.
        is_padded: Whether the window was extended past the signal to reach full
            width under the ``pad`` policy.
        padded_samples: How many zero samples the consumer must append.
    """

    start_sample: int
    end_sample: int
    speech_seconds: float
    coverage: float
    is_padded: bool = False
    padded_samples: int = 0

    @property
    def length_samples(self) -> int:
        return self.end_sample - self.start_sample

    @property
    def duration_seconds(self) -> float:
        """Window length in samples; callers divide by the sample rate."""
        return float(self.length_samples)


def segment_speech(
    n_samples: int,
    speech_mask: SpeechMask,
    sample_rate: int,
    config: AudioConfig | None = None,
) -> list[Segment]:
    """Build overlapping analysis windows over the speech regions.

    Args:
        n_samples: Length of the preprocessed signal.
        speech_mask: Output of :func:`voxshield.audio.vad.detect_speech`.
        sample_rate: Canonical sample rate of the signal.
        config: Pipeline configuration.

    Returns:
        Windows in chronological order. May be empty if the signal contains
        less than ``min_segment_seconds`` of contiguous speech, which the
        pipeline reports as ``INSUFFICIENT_SPEECH``.
    """
    cfg = config or AudioConfig()

    window = round(cfg.segment_seconds * sample_rate)
    hop = max(1, round(cfg.segment_hop * sample_rate))
    policy = cfg.short_segment_policy

    if n_samples <= 0 or window <= 0:
        return []

    mask = speech_mask.sample_mask(sample_rate, n_samples)
    if not mask.any():
        return []

    # Cumulative speech-sample count turns "speech inside this window" into an
    # O(1) lookup per window instead of a scan.
    cumulative = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))

    segments: list[Segment] = []
    starts = list(range(0, max(1, n_samples - window + 1), hop))
    # Anchor a final window at the signal end. Without this, a clip longer than
    # one window but shorter than two -- a very common upload length -- silently
    # discards everything past the first window.
    tail_start = n_samples - window
    if tail_start > 0 and (not starts or tail_start > starts[-1]):
        starts.append(tail_start)

    for start in starts:
        stop = min(start + window, n_samples)
        length = stop - start
        if length <= 0:
            continue
        speech_samples = int(cumulative[stop] - cumulative[start])
        speech_seconds = speech_samples / float(sample_rate)
        if speech_seconds < cfg.min_segment_seconds:
            # No policy rescues this: the window exists but carries less speech
            # than the detector was asked to find evidence of.
            continue

        is_padded = length < window
        if is_padded:
            if policy == "drop":
                continue
            if policy == "pad":
                stop = start + window
            # policy == "keep": report the natural, short width.

        reported_length = stop - start
        segments.append(
            Segment(
                start_sample=start,
                end_sample=stop,
                speech_seconds=speech_seconds,
                coverage=min(1.0, speech_seconds / (reported_length / float(sample_rate))),
                is_padded=is_padded and policy == "pad",
                padded_samples=window - length if is_padded and policy == "pad" else 0,
            )
        )

    return _merge_overlaps(segments)


def _merge_overlaps(segments: list[Segment]) -> list[Segment]:
    """Drop a window fully contained in its predecessor.

    Guards against duplicate windows when a signal is only a little longer than
    one window and the final ``range`` step lands on an already-covered span.
    """
    if not segments:
        return []
    out = [segments[0]]
    for seg in segments[1:]:
        if seg.start_sample < out[-1].end_sample and seg.end_sample <= out[-1].end_sample:
            continue
        out.append(seg)
    return out


def aggregate_segment_scores(scores: np.ndarray) -> float:
    """Aggregate window scores into one audio-level score.

    The mean is used rather than the max. Max is the more alarming choice and
    would be the wrong one: a single spurious high window would then dominate a
    call-level verdict, which is precisely the failure mode that produces false
    accusations against real customers.

    Args:
        scores: Per-window synthetic-speech probabilities.

    Returns:
        Mean probability in ``[0, 1]``, or ``0.0`` for an empty input.
    """
    arr = np.asarray(scores, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    if not np.isfinite(arr).all():
        msg = "segment scores contain non-finite values"
        raise ValueError(msg)
    return float(np.clip(arr.mean(), 0.0, 1.0))
