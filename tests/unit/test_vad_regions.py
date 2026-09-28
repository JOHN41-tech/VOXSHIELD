"""Speech regions: padding, merging, dropping, clamping, and extraction."""

from __future__ import annotations

import numpy as np
import pytest

from voxshield.audio.vad import (
    EnergyVadDetector,
    SpeechRegion,
    VadDetector,
    detect_speech,
    find_speech_regions,
    region_waveforms,
)
from voxshield.config import AudioConfig, VadConfig

_SR = 16_000


def _tone(seconds: float, freq: float = 220.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(_SR * seconds)) / _SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _embedded(segments: list[tuple[float, float]], total: float = 8.0) -> np.ndarray:
    """Silence with tones placed at the given ``(start, duration)`` offsets."""
    signal = np.zeros(int(_SR * total), dtype=np.float32)
    for start, length in segments:
        signal[int(_SR * start) : int(_SR * (start + length))] = _tone(length)
    return signal


def _regions(signal: np.ndarray, config: AudioConfig | None = None) -> list[SpeechRegion]:
    mask = detect_speech(signal, _SR, config)
    return find_speech_regions(mask, _SR, signal.size, config)


class TestRegionBounds:
    def test_duration_is_never_negative(self) -> None:
        assert SpeechRegion(start_s=2.0, end_s=1.0).duration_s == 0.0

    def test_to_samples_floors_start_and_ceils_end(self) -> None:
        start, end = SpeechRegion(start_s=1.0005, end_s=2.0005).to_samples(_SR)
        assert start == 16_008
        assert end == 32_008

    def test_to_samples_clamps_start_at_zero(self) -> None:
        # A region starting before 0 must not produce slice(0, -1), which would
        # return the entire signal except one sample.
        start, end = SpeechRegion(start_s=-0.5, end_s=0.25).to_samples(_SR)
        assert start == 0
        assert end == 4_000

    def test_metadata_is_rounded_and_sample_free(self) -> None:
        metadata = SpeechRegion(start_s=1.23456, end_s=3.2).as_metadata()
        assert metadata == {
            "region_start_seconds": 1.2346,
            "region_end_seconds": 3.2,
            "region_duration_seconds": 1.9654,
        }
        assert all(isinstance(v, float) for v in metadata.values())


class TestPadding:
    def test_regions_are_padded_on_both_sides(self) -> None:
        signal = _embedded([(2.0, 3.0)])
        raw = detect_speech(signal, _SR).regions()
        padded = _regions(signal, AudioConfig(vad=VadConfig(region_padding_s=0.25)))

        assert len(raw) == len(padded) == 1
        assert padded[0].start_s == pytest.approx(raw[0][0] - 0.25, abs=0.02)
        assert padded[0].end_s == pytest.approx(raw[0][1] + 0.25, abs=0.02)

    def test_zero_padding_leaves_the_raw_frame_bounds(self) -> None:
        signal = _embedded([(2.0, 3.0)])
        raw = detect_speech(signal, _SR).regions()
        regions = _regions(signal, AudioConfig(vad=VadConfig(region_padding_s=0.0)))

        assert regions[0].start_s == pytest.approx(raw[0][0], abs=1e-6)
        assert regions[0].end_s == pytest.approx(raw[0][1], abs=1e-6)

    def test_padding_recovers_an_onset_clipped_by_the_frame_grid(self) -> None:
        """The first 10 ms of an utterance fall below the speech threshold.

        Without padding, every window built from this region would start after
        the consonant that identifies the speaker, which is the part of the
        signal an anti-spoofing model relies on most.
        """
        signal = _embedded([(2.0, 3.0)])
        regions = _regions(signal)

        raw_start = detect_speech(signal, _SR).regions()[0][0]
        assert regions[0].start_s < raw_start

    def test_padding_cannot_push_a_region_outside_the_signal(self) -> None:
        signal = _embedded([(0.0, 3.0)], total=3.0)
        regions = _regions(signal, AudioConfig(vad=VadConfig(region_padding_s=0.5)))

        assert regions[0].start_s == 0.0
        assert regions[0].end_s == pytest.approx(3.0)


class TestMerging:
    def test_a_short_gap_does_not_split_one_utterance(self) -> None:
        # Two bursts 100 ms apart are one utterance separated by a stop
        # consonant, and scoring them as two windows would halve the evidence
        # available per window.
        signal = _embedded([(1.0, 2.0), (3.1, 2.0)])
        regions = _regions(
            signal,
            AudioConfig(vad=VadConfig(region_merge_gap_s=0.5, region_padding_s=0.0)),
        )

        assert len(regions) == 1
        assert regions[0].start_s == pytest.approx(1.0, abs=0.1)
        assert regions[0].end_s == pytest.approx(5.1, abs=0.1)

    def test_distant_utterances_stay_separate(self) -> None:
        signal = _embedded([(0.5, 1.5), (6.0, 1.5)])
        regions = _regions(
            signal,
            AudioConfig(vad=VadConfig(region_merge_gap_s=0.2, region_padding_s=0.0)),
        )

        assert len(regions) == 2

    def test_merging_is_order_independent(self) -> None:
        signal = _embedded([(1.0, 1.0), (1.6, 1.0), (2.2, 1.0)])
        config = AudioConfig(vad=VadConfig(region_merge_gap_s=0.5, region_padding_s=0.0))

        regions = _regions(signal, config)

        assert len(regions) == 1


class TestDropping:
    def test_regions_below_the_minimum_are_dropped(self) -> None:
        signal = _embedded([(1.0, 0.3)])
        regions = _regions(
            signal,
            AudioConfig(
                vad=VadConfig(
                    min_speech_duration_s=0.05,
                    region_padding_s=0.0,
                    min_region_seconds=1.0,
                )
            ),
        )

        assert regions == []

    def test_dropping_does_not_lose_a_neighbouring_valid_region(self) -> None:
        signal = _embedded([(1.0, 0.3), (4.0, 3.0)])
        regions = _regions(
            signal,
            AudioConfig(
                vad=VadConfig(
                    min_speech_duration_s=0.05,
                    region_padding_s=0.0,
                    min_region_seconds=1.0,
                    region_merge_gap_s=0.1,
                )
            ),
        )

        assert len(regions) == 1
        assert regions[0].start_s == pytest.approx(4.0, abs=0.1)

    def test_no_speech_yields_no_regions(self) -> None:
        assert _regions(np.zeros(4 * _SR, dtype=np.float32)) == []

    def test_empty_signal_yields_no_regions(self) -> None:
        assert _regions(np.zeros(0, dtype=np.float32)) == []


class TestExtraction:
    def test_waveforms_match_their_region_length(self) -> None:
        signal = _embedded([(2.0, 3.0)])
        regions = _regions(signal)
        waveforms = region_waveforms(signal, regions, _SR)

        assert len(waveforms) == 1
        expected = round(regions[0].duration_s * _SR)
        assert abs(waveforms[0].size - expected) <= 1

    def test_extraction_is_a_view_not_a_copy(self) -> None:
        signal = _embedded([(2.0, 3.0)])
        waveforms = region_waveforms(signal, _regions(signal), _SR)

        assert waveforms[0].base is not None

    def test_a_region_outside_the_signal_yields_an_empty_array(self) -> None:
        signal = _tone(1.0)
        waveforms = region_waveforms(signal, [SpeechRegion(10.0, 12.0)], _SR)

        assert waveforms[0].size == 0

    def test_extraction_of_no_regions_yields_nothing(self) -> None:
        assert region_waveforms(_tone(1.0), [], _SR) == []


class TestEnergyVadDetector:
    def test_satisfies_the_detector_protocol(self) -> None:
        assert isinstance(EnergyVadDetector(), VadDetector)

    def test_detect_matches_the_free_function(self) -> None:
        signal = _embedded([(2.0, 3.0)])
        detector = EnergyVadDetector()

        assert detector.detect(signal, _SR).speech_seconds == pytest.approx(
            detect_speech(signal, _SR).speech_seconds
        )

    def test_is_deterministic_across_instances(self) -> None:
        signal = _embedded([(1.0, 2.0), (5.0, 2.0)])
        first = EnergyVadDetector().regions(signal, _SR)
        second = EnergyVadDetector().regions(signal, _SR)

        assert [(r.start_s, r.end_s) for r in first] == [
            (r.start_s, r.end_s) for r in second
        ]

    def test_name_is_reported_for_audit(self) -> None:
        assert EnergyVadDetector().name == "energy"

    def test_an_explicit_config_is_honoured(self) -> None:
        signal = _embedded([(2.0, 3.0)])
        # Reject every frame on the spectral gate. An absolute floor alone would
        # not abstain here: the tone sits near -11 dBFS, well above any plausible
        # floor, so the decision has to come from a gate that can say "no".
        deaf = AudioConfig(vad=VadConfig(max_zero_crossing_rate=0.0))

        assert EnergyVadDetector(deaf).regions(signal, _SR) == []
        assert EnergyVadDetector().regions(signal, _SR) != []
