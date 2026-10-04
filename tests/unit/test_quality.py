"""Quality assessment: the metrics, the issue codes, and the abstention rule."""

from __future__ import annotations

import json

import numpy as np
import pytest

from voxshield.audio.quality import (
    ISSUE_CLIPPED,
    ISSUE_DC_OFFSET,
    ISSUE_HIGH_CREST_FACTOR,
    ISSUE_LOW_LEVEL,
    ISSUE_LOW_SNR,
    ISSUE_NON_FINITE,
    ISSUE_SILENT,
    assess_quality,
)
from voxshield.config import AudioConfig, QualityConfig

SAMPLE_RATE = 16_000


def _tone(seconds: float, freq: float = 220.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _gated_tone(seconds: float = 3.0, amplitude: float = 0.3) -> np.ndarray:
    """Tone present only in bursts, with true digital silence between them."""
    tone = _tone(seconds, amplitude=amplitude)
    t = np.arange(tone.size) / SAMPLE_RATE
    gate = (np.abs(np.sin(np.pi * t / 1.5)) > 0.4).astype(np.float32)
    return (tone * gate).astype(np.float32)


def _silence(seconds: float = 2.0) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def test_clean_tone_reports_no_issues() -> None:
    report = assess_quality(_tone(2.0), SAMPLE_RATE)

    assert report.issues == ()
    assert report.is_scorable
    assert report.blocking_issues == ()


def test_metrics_describe_a_steady_tone() -> None:
    report = assess_quality(_tone(2.0, amplitude=0.5), SAMPLE_RATE)

    # A 0.5-amplitude sine has an RMS of 0.5/sqrt(2), i.e. -9.03 dBFS.
    assert report.peak_amplitude == pytest.approx(0.5, rel=0.01)
    assert report.rms_dbfs == pytest.approx(-9.03, abs=0.1)
    assert report.crest_factor_db == pytest.approx(3.01, abs=0.1)
    assert report.duration_seconds == pytest.approx(2.0)
    assert report.sample_count == 32_000
    assert report.snr_db is None


def test_silence_is_blocking_and_not_scorable() -> None:
    report = assess_quality(_silence(), SAMPLE_RATE)

    assert ISSUE_SILENT in report.issues
    assert report.is_silent
    assert not report.is_scorable
    assert ISSUE_SILENT in report.blocking_issues
    assert report.silence_ratio == 1.0
    assert report.speech_frame_ratio == 0.0


def test_empty_audio_is_reported_as_silent_rather_than_raising() -> None:
    report = assess_quality(np.zeros(0, dtype=np.float32), SAMPLE_RATE)

    assert report.issues == (ISSUE_SILENT,)
    assert report.sample_count == 0
    assert report.duration_seconds == 0.0
    assert not report.is_scorable


def test_clipping_warns_without_blocking() -> None:
    clipped = np.clip(_tone(2.0) * 20.0, -1.0, 1.0).astype(np.float32)
    report = assess_quality(clipped, SAMPLE_RATE)

    assert ISSUE_CLIPPED in report.issues
    assert report.has_clipping
    assert report.clipping_ratio > 0.0
    # Clipping is evidence about the recording chain, not about the voice: the
    # clip is still analysable, so it must not be refused.
    assert report.is_scorable
    assert report.blocking_issues == ()
    assert ISSUE_CLIPPED in report.warnings


def test_very_quiet_audio_is_flagged() -> None:
    report = assess_quality(_tone(2.0, amplitude=1e-4), SAMPLE_RATE)

    # -83 dBFS is below the -60 dBFS silence gate, so this is both "too quiet to
    # trust" and "effectively silent". Reporting it as unusable is the honest
    # answer: a signal this far down would be amplified into its own noise floor
    # by any gain-based scorer.
    assert ISSUE_LOW_LEVEL in report.issues
    assert ISSUE_SILENT in report.issues
    assert not report.is_scorable


def test_dc_offset_is_flagged() -> None:
    biased = (_tone(2.0) + 0.5).astype(np.float32)
    report = assess_quality(biased, SAMPLE_RATE)

    assert ISSUE_DC_OFFSET in report.issues
    assert report.dc_offset == pytest.approx(0.5, rel=0.02)


def test_impulsive_signal_is_flagged_by_crest_factor() -> None:
    impulsive = _silence(2.0)
    impulsive[8_000] = 0.99
    report = assess_quality(impulsive, SAMPLE_RATE)

    assert ISSUE_HIGH_CREST_FACTOR in report.issues


def test_measurable_noise_floor_yields_snr() -> None:
    rng = np.random.default_rng(0)
    noisy = (_gated_tone() + 0.01 * rng.standard_normal(48_000)).astype(np.float32)
    report = assess_quality(noisy, SAMPLE_RATE)

    assert report.snr_db is not None
    assert 20.0 < report.snr_db < 40.0
    assert ISSUE_LOW_SNR not in report.issues


def test_low_snr_is_flagged_when_a_floor_is_measurable() -> None:
    # Gated speech with a steady noise bed: the gaps expose the floor, so the
    # ratio is measurable, and it lands between the 3 dB measurability limit and
    # the 10 dB policy. A stationary tone plus stationary noise would not work
    # here -- every frame would sit at the same level and there would be no
    # floor to measure against.
    rng = np.random.default_rng(1)
    speech = _gated_tone(amplitude=0.3)
    drowned = (speech + 0.15 * rng.standard_normal(speech.size)).astype(np.float32)
    report = assess_quality(drowned, SAMPLE_RATE)

    assert report.snr_db is not None
    assert 3.0 <= report.snr_db < 10.0
    assert ISSUE_LOW_SNR in report.issues
    # A measurable floor is a warning about the clip, not a refusal to score it.
    assert report.is_scorable


@pytest.mark.parametrize(
    "samples",
    [
        pytest.param(_tone(2.0), id="stationary_tone"),
        pytest.param(_gated_tone(), id="gated_tone_with_digital_silence"),
    ],
)
def test_stationary_signals_report_snr_as_undefined(samples: np.ndarray) -> None:
    """A clean tone has no noise floor, so the ratio is undefined, not zero.

    Reporting 0 dB here -- or the 200-odd dB that digital silence would produce
    -- would turn an absence of information into a confident-looking number.
    """
    report = assess_quality(samples, SAMPLE_RATE)

    assert report.snr_db is None
    assert ISSUE_LOW_SNR not in report.issues


def test_non_finite_samples_are_counted_and_block() -> None:
    damaged = _tone(2.0).copy()
    damaged[10] = np.nan
    damaged[11] = np.inf
    report = assess_quality(damaged, SAMPLE_RATE)

    assert ISSUE_NON_FINITE in report.issues
    assert report.non_finite_ratio == pytest.approx(2 / 32_000)
    assert not report.is_scorable
    # Remaining metrics stay meaningful: one bad sample must not poison the rest.
    assert np.isfinite(report.rms_dbfs)
    assert np.isfinite(report.peak_amplitude)


def test_all_nan_signal_does_not_produce_nan_metrics() -> None:
    report = assess_quality(np.full(1_000, np.nan, dtype=np.float32), SAMPLE_RATE)

    assert report.non_finite_ratio == 1.0
    assert ISSUE_NON_FINITE in report.issues
    assert ISSUE_SILENT in report.issues
    assert all(
        np.isfinite(value)
        for value in (
            report.peak_amplitude,
            report.rms_amplitude,
            report.rms_dbfs,
            report.dc_offset,
            report.crest_factor_db,
            report.silence_ratio,
        )
    )


def test_stereo_input_is_flattened_before_measurement() -> None:
    left = _tone(2.0)
    stereo = np.stack([left, left], axis=1)
    report = assess_quality(stereo, SAMPLE_RATE)

    assert report.sample_count == 64_000
    assert report.peak_amplitude == pytest.approx(0.3, rel=0.01)


def test_dc_offset_on_one_channel_only_is_flagged() -> None:
    """A per-channel offset is caught even though the joint mean would hide it."""
    stereo = np.stack([_tone(2.0) + 0.5, _tone(2.0)], axis=1).astype(np.float32)
    report = assess_quality(stereo, SAMPLE_RATE)

    assert ISSUE_DC_OFFSET in report.issues
    assert report.dc_offset == pytest.approx(0.25, rel=0.01)


def test_opposing_channel_offsets_cancel_and_are_not_flagged() -> None:
    """A joint mean of zero is a real zero: the offset is genuinely absent."""
    stereo = np.stack([_tone(2.0) + 0.5, _tone(2.0) - 0.5], axis=1).astype(np.float32)
    report = assess_quality(stereo, SAMPLE_RATE)

    assert report.dc_offset == pytest.approx(0.0, abs=1e-3)
    assert ISSUE_DC_OFFSET not in report.issues


def test_quality_can_be_disabled_but_metrics_are_still_reported() -> None:
    config = AudioConfig(quality=QualityConfig(enabled=False))
    report = assess_quality(np.clip(_tone(2.0) * 20.0, -1.0, 1.0), SAMPLE_RATE, config)

    assert report.issues == ()
    assert report.clipping_ratio > 0.0


def test_explicit_thresholds_override_the_config_section() -> None:
    signal = _tone(2.0, amplitude=0.5)
    strict = QualityConfig(clip_threshold=0.1)

    assert assess_quality(signal, SAMPLE_RATE, quality=strict).has_clipping
    assert not assess_quality(signal, SAMPLE_RATE).has_clipping


def test_metadata_is_json_serialisable_and_rounded() -> None:
    metadata = assess_quality(_tone(2.0), SAMPLE_RATE).as_metadata()

    assert json.loads(json.dumps(metadata)) == metadata
    assert set(metadata) >= {
        "quality_duration_seconds",
        "quality_rms_dbfs",
        "quality_peak_amplitude",
        "quality_silence_ratio",
        "quality_snr_db",
        "quality_issues",
        "quality_scorable",
    }
    assert metadata["quality_rms_dbfs"] == round(metadata["quality_rms_dbfs"], 2)


def test_metadata_never_contains_a_sample_series() -> None:
    metadata = assess_quality(_tone(2.0), SAMPLE_RATE).as_metadata()

    for key, value in metadata.items():
        if isinstance(value, list):
            assert all(isinstance(item, str) for item in value), key
        elif isinstance(value, (int, float)):
            assert not isinstance(value, (list, tuple, np.ndarray)), key


def test_gain_scaling_shifts_levels_but_not_ratios() -> None:
    """Doubling the input moves the dB figures by 6 dB and changes nothing else.

    This is the property that makes the report usable as a metric: a caller who
    normalises upstream and a caller who does not must get the same
    signal-to-noise and silence assessment.
    """
    base = _gated_tone(amplitude=0.3)
    louder = (base * 2.0).astype(np.float32)

    quiet_report = assess_quality(base, SAMPLE_RATE)
    loud_report = assess_quality(louder, SAMPLE_RATE)

    assert loud_report.rms_dbfs == pytest.approx(quiet_report.rms_dbfs + 6.02, abs=0.1)
    assert loud_report.peak_amplitude == pytest.approx(quiet_report.peak_amplitude * 2.0, rel=0.01)
    assert loud_report.silence_ratio == quiet_report.silence_ratio
    assert loud_report.snr_db == quiet_report.snr_db
    assert loud_report.crest_factor_db == pytest.approx(quiet_report.crest_factor_db, abs=0.01)


def test_shorter_than_one_analysis_frame_still_reports_a_level() -> None:
    # One full period of 220 Hz, so the mean is a real DC estimate rather than
    # an artefact of sampling a fraction of a cycle.
    tiny = _tone(220.0 / 220.0, amplitude=0.25)
    report = assess_quality(tiny, SAMPLE_RATE)

    assert report.issues == ()
    assert report.silence_ratio == 0.0
    assert report.speech_frame_ratio == 1.0
    assert report.snr_db is None
