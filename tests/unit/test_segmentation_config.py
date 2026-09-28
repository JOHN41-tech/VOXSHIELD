"""Configurable segmentation: hop spacing and the short-window policy."""

from __future__ import annotations

import numpy as np
import pytest

from voxshield.audio.segmentation import segment_speech
from voxshield.audio.vad import detect_speech
from voxshield.config import AudioConfig

_SR = 16_000


def _speech(seconds: float) -> np.ndarray:
    t = np.arange(int(_SR * seconds)) / _SR
    return (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _windows(seconds: float, config: AudioConfig | None = None) -> list[tuple[int, int]]:
    signal = _speech(seconds)
    mask = detect_speech(signal, _SR)
    return [
        (seg.start_sample, seg.end_sample)
        for seg in segment_speech(signal.size, mask, _SR, config)
    ]


class TestHop:
    def test_default_hop_is_half_the_window(self) -> None:
        # 50% overlap is the Phase 0 behaviour and must not drift.
        assert AudioConfig().segment_hop == pytest.approx(2.0)
        assert _windows(5.5) == [(0, 64_000), (24_000, 88_000)]

    def test_a_wider_hop_yields_fewer_windows(self) -> None:
        # Long enough that the tail anchor is not the dominant window.
        default = _windows(12.0)
        wider = _windows(12.0, AudioConfig(segment_hop_seconds=3.0))

        assert len(wider) < len(default)

    def test_a_narrower_hop_yields_more_windows(self) -> None:
        assert len(_windows(12.0, AudioConfig(segment_hop_seconds=0.5))) > len(
            _windows(12.0)
        )

    def test_hop_is_independent_of_window_size(self) -> None:
        config = AudioConfig(segment_seconds=2.0, max_segment_seconds=2.0)

        assert config.segment_hop == pytest.approx(1.0)

    def test_hop_larger_than_the_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="segment_hop_seconds"):
            AudioConfig(segment_hop_seconds=8.0)

    def test_non_positive_hop_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="segment_hop_seconds"):
            AudioConfig(segment_hop_seconds=0.0)

    def test_hop_equal_to_the_window_gives_non_overlapping_windows(self) -> None:
        config = AudioConfig(segment_hop_seconds=4.0)

        assert config.segment_hop == pytest.approx(4.0)
        assert _windows(8.0, config) == [(0, 64_000), (64_000, 128_000)]

    def test_a_tail_anchor_still_covers_the_end_of_the_signal(self) -> None:
        """With hop == window the last window is anchored, overlapping its neighbour.

        Without the anchor, the final second of a 10-second clip would sit in no
        window at all and would never be scored -- the exact silent gap this
        module's docstring warns about for short uploads.
        """
        windows = _windows(10.0, AudioConfig(segment_hop_seconds=4.0))

        assert windows == [(0, 64_000), (64_000, 128_000), (96_000, 160_000)]
        assert windows[-1][1] == 10 * _SR


class TestShortSegmentPolicy:
    """A clip shorter than one window is the only case the policy decides."""

    def test_drop_discards_the_window(self) -> None:
        assert _windows(3.0, AudioConfig(short_segment_policy="drop")) == []

    def test_pad_extends_to_full_width_and_declares_the_padding(self) -> None:
        segments = segment_speech(
            int(3.0 * _SR),
            detect_speech(_speech(3.0), _SR),
            _SR,
            AudioConfig(short_segment_policy="pad"),
        )

        assert len(segments) == 1
        segment = segments[0]
        assert (segment.start_sample, segment.end_sample) == (0, 64_000)
        assert segment.is_padded is True
        assert segment.padded_samples == 64_000 - 48_000
        # Coverage is measured against the window the model will actually see.
        assert segment.coverage == pytest.approx(3.0 / 4.0, rel=0.05)

    def test_keep_scores_the_window_at_its_natural_width(self) -> None:
        segments = segment_speech(
            int(3.0 * _SR),
            detect_speech(_speech(3.0), _SR),
            _SR,
            AudioConfig(short_segment_policy="keep"),
        )

        assert len(segments) == 1
        segment = segments[0]
        assert (segment.start_sample, segment.end_sample) == (0, 48_000)
        assert segment.is_padded is False
        assert segment.padded_samples == 0
        assert segment.coverage == pytest.approx(1.0, rel=0.05)

    def test_no_policy_rescues_a_window_with_too_little_speech(self) -> None:
        # 1.5s of speech is below min_segment_seconds, so there is nothing to
        # score however the tail is handled.
        for policy in ("drop", "pad", "keep"):
            config = AudioConfig(short_segment_policy=policy)
            assert _windows(1.5, config) == [], policy

    def test_full_length_windows_are_untouched_by_the_policy(self) -> None:
        baseline = _windows(5.5, AudioConfig(short_segment_policy="drop"))
        for policy in ("pad", "keep"):
            assert _windows(5.5, AudioConfig(short_segment_policy=policy)) == baseline

    def test_an_unknown_policy_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="short_segment_policy"):
            AudioConfig(short_segment_policy="truncate")

    def test_segments_with_padding_are_flagged_not_silently_resized(self) -> None:
        """A consumer must be able to tell padded windows from real ones.

        Feeding a model a window that was never recorded risks a confident score
        on audio that does not exist, so the flag is part of the contract rather
        than an implementation detail.
        """
        signal = _speech(3.0)
        padded = segment_speech(
            signal.size,
            detect_speech(signal, _SR),
            _SR,
            AudioConfig(short_segment_policy="pad"),
        )
        kept = segment_speech(
            signal.size,
            detect_speech(signal, _SR),
            _SR,
            AudioConfig(short_segment_policy="keep"),
        )

        assert any(seg.is_padded for seg in padded)
        assert not any(seg.is_padded for seg in kept)
        # Only the padded variant reaches past the end of the signal.
        assert max(seg.end_sample for seg in padded) > signal.size
        assert max(seg.end_sample for seg in kept) <= signal.size
