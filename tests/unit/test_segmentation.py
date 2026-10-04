"""Unit tests for windowing and score aggregation.

The segmenter is where a small bug turns into a systematic error: a window that
is missed means a caller hears a verdict about audio it never analysed, and a
mean-versus-max mistake in aggregation turns one bad window into a confident
accusation.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from voxshield.audio.segmentation import (
    Segment,
    aggregate_segment_scores,
    segment_speech,
)
from voxshield.audio.vad import detect_speech
from voxshield.config import AudioConfig

_SR = 16_000


def _mask_for(signal: np.ndarray):
    return detect_speech(signal, _SR)


class TestSegmentSpeech:
    def test_long_signal_yields_overlapping_windows(self) -> None:
        cfg = AudioConfig()
        n = int(10.0 * _SR)
        signal = np.zeros(n, dtype=np.float32)
        signal[:] = 0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / _SR).astype(np.float32)
        segments = segment_speech(n, _mask_for(signal), _SR, cfg)
        assert len(segments) > 1
        for seg in segments:
            assert seg.start_sample >= 0
            assert seg.end_sample <= n
            assert seg.length_samples > 0
            assert seg.coverage == pytest.approx(min(1.0, seg.coverage))
            assert 0.0 <= seg.coverage <= 1.0

    def test_windows_are_chronological_and_non_nested(self) -> None:
        cfg = AudioConfig()
        n = int(10.0 * _SR)
        signal = (0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / _SR)).astype(np.float32)
        segments = segment_speech(n, _mask_for(signal), _SR, cfg)
        for prev, nxt in pairwise(segments):
            assert nxt.start_sample > prev.start_sample
            assert nxt.end_sample > prev.end_sample, (
                "a window was fully contained in its predecessor"
            )

    def test_short_but_sufficient_signal_is_not_discarded(self) -> None:
        # A clip longer than one window but shorter than two is a very common
        # upload length. If the tail window is dropped, the caller's verdict is
        # about audio nobody looked at.
        cfg = AudioConfig()
        n = int(5.0 * _SR)
        signal = (0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / _SR)).astype(np.float32)
        segments = segment_speech(n, _mask_for(signal), _SR, cfg)
        assert segments
        assert segments[0].start_sample == 0
        assert segments[-1].end_sample == n

    def test_all_speech_is_covered(self) -> None:
        cfg = AudioConfig()
        n = int(8.0 * _SR)
        signal = (0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / _SR)).astype(np.float32)
        segments = segment_speech(n, _mask_for(signal), _SR, cfg)
        assert segments[0].start_sample == 0
        assert segments[-1].end_sample == n

    def test_no_speech_yields_no_segments(self) -> None:
        n = 8 * _SR
        silence = np.zeros(n, dtype=np.float32)
        assert segment_speech(n, _mask_for(silence), _SR) == []

    def test_shorter_than_minimum_yields_nothing(self) -> None:
        # 1.5 s of speech is below min_segment_seconds, so there is no window
        # worth scoring and the pipeline must abstain rather than pad one.
        n = int(1.5 * _SR)
        signal = (0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / _SR)).astype(np.float32)
        assert segment_speech(n, _mask_for(signal), _SR) == []

    @pytest.mark.parametrize("n_samples", [0, -1])
    def test_degenerate_lengths(self, n_samples: int) -> None:
        assert segment_speech(n_samples, _mask_for(np.zeros(_SR, dtype=np.float32)), _SR) == []

    def test_coverage_reflects_speech_fraction(self) -> None:
        # Half the clip is speech: a window spanning the whole clip must report
        # roughly half coverage, which is what lets a caller discount a window.
        cfg = AudioConfig()
        n = int(4.0 * _SR)
        signal = np.zeros(n, dtype=np.float32)
        signal[: 2 * _SR] = (0.2 * np.sin(2 * np.pi * 220 * np.arange(2 * _SR) / _SR)).astype(
            np.float32
        )
        segments = segment_speech(n, _mask_for(signal), _SR, cfg)
        assert segments
        assert any(0.2 < seg.coverage < 0.9 for seg in segments)

    def test_speech_seconds_never_exceeds_window(self) -> None:
        cfg = AudioConfig()
        n = int(6.0 * _SR)
        signal = (0.2 * np.sin(2 * np.pi * 220 * np.arange(n) / _SR)).astype(np.float32)
        for seg in segment_speech(n, _mask_for(signal), _SR, cfg):
            assert seg.speech_seconds <= seg.duration_seconds / _SR + 1e-9


class TestAggregateSegmentScores:
    def test_uses_the_mean_not_the_max(self) -> None:
        # The single most important property in this module: one spurious high
        # window must not become the call-level verdict.
        scores = np.array([0.02, 0.99, 0.03])
        assert aggregate_segment_scores(scores) == pytest.approx(0.3467, abs=1e-3)
        assert aggregate_segment_scores(scores) < 0.5

    def test_uniform_scores_pass_through(self) -> None:
        assert aggregate_segment_scores(np.full(4, 0.7)) == pytest.approx(0.7)

    def test_empty_is_zero_not_an_error(self) -> None:
        assert aggregate_segment_scores(np.array([])) == 0.0

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            aggregate_segment_scores(np.array([0.1, np.nan]))

    def test_clips_out_of_range_input(self) -> None:
        # A misbehaving detector returning >1 must not produce a >1 probability.
        assert aggregate_segment_scores(np.array([-0.5, 1.5])) == pytest.approx(0.5)

    def test_accepts_a_flat_matrix(self) -> None:
        assert aggregate_segment_scores(np.array([[0.2, 0.4], [0.6, 0.8]])) == pytest.approx(0.5)


class TestSegmentShape:
    def test_properties(self) -> None:
        seg = Segment(start_sample=1_000, end_sample=9_000, speech_seconds=2.0, coverage=0.25)
        assert seg.length_samples == 8_000
        assert seg.duration_seconds == 8_000.0
