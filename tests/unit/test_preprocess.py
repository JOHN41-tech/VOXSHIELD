"""Unit tests for the canonical preprocessing contract.

The invariant under test throughout is that preprocessing is a *pure, bounded*
transform: identical input yields identical output, no input can produce
non-finite samples, and no input can escape the amplitude limits.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxshield.audio.preprocess import (
    dbfs,
    normalize_loudness,
    preprocess,
    remove_dc_offset,
    resample_to,
    rms,
    sanitize,
    to_mono,
)
from voxshield.config import AudioConfig
from voxshield.errors import InvalidAudioSignalError

_SR = 16_000


class TestSanitize:
    def test_replaces_non_finite(self) -> None:
        x = np.array([0.1, np.nan, np.inf, -np.inf, 0.2], dtype=np.float32)
        out = sanitize(x)
        assert np.isfinite(out).all()
        assert out[0] == pytest.approx(0.1)
        assert out[1:4].tolist() == [0.0, 0.0, 0.0]

    def test_does_not_mutate_input(self) -> None:
        x = np.array([0.1, np.nan], dtype=np.float32)
        sanitize(x)
        assert np.isnan(x[1]), "sanitize must not write through to its input"

    def test_promotes_to_float32(self) -> None:
        out = sanitize(np.array([1, 2, 3], dtype=np.int16))
        assert out.dtype == np.float32

    def test_copies_even_when_clean(self) -> None:
        # A clean input must still be copied, or a caller could mutate the
        # canonical buffer through the value they passed in.
        x = np.zeros(4, dtype=np.float32)
        out = sanitize(x)
        out[0] = 1.0
        assert x[0] == 0.0


class TestToMono:
    def test_passes_1d_through(self) -> None:
        x = np.linspace(-0.5, 0.5, 128, dtype=np.float32)
        assert to_mono(x).shape == (128,)

    def test_averages_channels(self) -> None:
        stereo = np.zeros((4, 2), dtype=np.float32)
        stereo[:, 0] = 1.0
        stereo[:, 1] = 0.0
        assert to_mono(stereo).tolist() == pytest.approx([0.5] * 4)

    def test_unwraps_single_channel_column(self) -> None:
        assert to_mono(np.ones((5, 1), dtype=np.float32)).shape == (5,)

    def test_rejects_3d(self) -> None:
        with pytest.raises(InvalidAudioSignalError, match="1-D or 2-D"):
            to_mono(np.zeros((2, 2, 2), dtype=np.float32))


class TestRemoveDcOffset:
    def test_centres_the_signal(self) -> None:
        x = np.full(1000, 0.25, dtype=np.float32)
        out = remove_dc_offset(x)
        assert float(np.mean(out)) == pytest.approx(0.0, abs=1e-6)

    def test_preserves_shape_and_leaves_a_copy(self) -> None:
        x = np.full(8, 0.1, dtype=np.float32)
        out = remove_dc_offset(x)
        out[0] = 99.0
        assert x[0] == pytest.approx(0.1)

    def test_handles_empty(self) -> None:
        assert remove_dc_offset(np.zeros(0, dtype=np.float32)).size == 0


class TestResample:
    @pytest.mark.parametrize("orig", [8_000, 22_050, 44_100, 48_000])
    def test_produces_expected_duration(self, orig: int) -> None:
        x = np.zeros(orig, dtype=np.float32)
        out = resample_to(x, orig, _SR)
        assert out.dtype == np.float32
        # One second in, one second out, within a sample or two of the grid.
        assert abs(out.size - _SR) <= 2

    def test_same_rate_is_identity(self) -> None:
        x = np.linspace(-1, 1, 100, dtype=np.float32)
        assert resample_to(x, _SR, _SR) is not None
        assert np.array_equal(resample_to(x, _SR, _SR), x)

    @pytest.mark.parametrize(("orig", "target"), [(0, _SR), (_SR, 0), (-1, _SR)])
    def test_rejects_invalid_rates(self, orig: int, target: int) -> None:
        with pytest.raises(InvalidAudioSignalError, match="invalid sample rates"):
            resample_to(np.zeros(4, dtype=np.float32), orig, target)

    def test_preserves_a_tone(self) -> None:
        # A 400 Hz tone must still be a 400 Hz tone after resampling; a wrong
        # ratio would shift it and quietly change what the detector hears.
        t = np.arange(44_100) / 44_100
        tone = (0.5 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)
        out = resample_to(tone, 44_100, _SR)
        spectrum = np.abs(np.fft.rfft(out * np.hanning(out.size)))
        peak_hz = float(np.argmax(spectrum)) * _SR / out.size
        assert peak_hz == pytest.approx(400, abs=5)


class TestNormalizeLoudness:
    def test_targets_the_configured_level(self) -> None:
        cfg = AudioConfig()
        x = (np.sin(np.linspace(0, 100, 4_000)) * 0.05).astype(np.float32)
        out, gain_db, peak_before, clipped = normalize_loudness(x, cfg)
        assert float(dbfs(rms(out))) == pytest.approx(cfg.target_rms_dbfs, abs=0.5)
        assert gain_db > 0
        assert peak_before == pytest.approx(0.05, abs=1e-3)
        assert clipped is False

    def test_is_gain_invariant(self) -> None:
        # The whole point of normalising: a quiet recording and a loud one must
        # produce the same internal signal, so the detector cannot key on gain.
        cfg = AudioConfig()
        t = np.linspace(0, 200, 32_000)
        base = np.sin(2 * np.pi * 220 * t).astype(np.float32)
        quiet, *_ = normalize_loudness(base * 0.01, cfg)
        loud, *_ = normalize_loudness(base * 0.9, cfg)
        assert np.allclose(quiet, loud, atol=1e-3)

    def test_does_not_amplify_silence(self) -> None:
        # Amplifying digital silence would manufacture a signal and report a
        # gain that means nothing.
        cfg = AudioConfig()
        out, gain_db, peak_before, clipped = normalize_loudness(
            np.zeros(1_000, dtype=np.float32), cfg
        )
        assert gain_db == 0.0
        assert peak_before == 0.0
        assert clipped is False
        assert not out.any()

    def test_gain_is_capped(self) -> None:
        cfg = AudioConfig()
        # Well below the -70 dBFS silence gate but still non-zero.
        x = (np.sin(np.linspace(0, 100, 4_000)) * 5e-5).astype(np.float32)
        _, gain_db, _, _ = normalize_loudness(x, cfg)
        assert gain_db <= cfg.max_gain_db + 1e-6

    def test_peak_ceiling_is_enforced(self) -> None:
        cfg = AudioConfig()
        # A high-crest signal: almost silent on average, so RMS normalisation
        # gains it up, yet its peaks are already at full scale. Without the
        # ceiling the output would exceed 1.0 and wrap into a false signal.
        x = np.zeros(4_000, dtype=np.float32)
        x[::500] = 1.0
        assert rms(x) < 10 ** (cfg.target_rms_dbfs / 20.0)
        out, _, _, clipped = normalize_loudness(x, cfg)
        assert clipped is True
        assert float(np.max(np.abs(out))) <= cfg.peak_ceiling + 1e-6
        assert float(np.max(np.abs(out))) == pytest.approx(cfg.peak_ceiling, abs=1e-3)

    def test_loud_input_is_attenuated_not_clipped(self) -> None:
        # The mirror image of the ceiling test: normalising downwards must not
        # be reported as clipping, or the flag stops meaning anything.
        cfg = AudioConfig()
        x = np.sign(np.sin(np.linspace(0, 400, 8_000))).astype(np.float32) * 0.5
        out, gain_db, _, clipped = normalize_loudness(x, cfg)
        assert gain_db < 0
        assert clipped is False
        assert float(np.max(np.abs(out))) < cfg.peak_ceiling

    def test_output_always_within_unit_range(self) -> None:
        cfg = AudioConfig()
        rng = np.random.default_rng(11)
        for scale in (1e-4, 0.5, 0.99, 5.0):
            out, *_ = normalize_loudness(
                (rng.standard_normal(4_000) * scale).astype(np.float32), cfg
            )
            assert np.abs(out).max() <= 1.0 + 1e-6

    def test_empty_input(self) -> None:
        out, gain_db, peak_before, clipped = normalize_loudness(
            np.zeros(0, dtype=np.float32), AudioConfig()
        )
        assert out.size == 0
        assert (gain_db, peak_before, clipped) == (0.0, 0.0, False)


class TestPreprocessChain:
    def test_reaches_the_canonical_rate(self) -> None:
        out = preprocess(np.zeros(44_100, dtype=np.float32), 44_100)
        assert out.sample_rate == _SR
        assert out.samples.dtype == np.float32
        assert out.samples.ndim == 1

    def test_output_is_always_finite(self) -> None:
        dirty = np.full(8_000, np.nan, dtype=np.float32)
        dirty[0] = np.inf
        dirty[1] = 0.4
        out = preprocess(dirty, _SR)
        assert np.isfinite(out.samples).all()

    def test_stereo_is_downmixed(self) -> None:
        stereo = np.zeros((8_000, 2), dtype=np.float32)
        stereo[:, 0] = 0.3
        stereo[:, 1] = 0.1
        out = preprocess(stereo, _SR)
        assert out.samples.ndim == 1
        assert out.samples.size > 0

    def test_dc_offset_does_not_survive(self) -> None:
        t = np.linspace(0, 1, 16_000)
        signal = (0.2 * np.sin(2 * np.pi * 300 * t) + 0.4).astype(np.float32)
        out = preprocess(signal, _SR)
        assert float(np.mean(out.samples)) == pytest.approx(0.0, abs=1e-3)

    def test_duration_property(self) -> None:
        out = preprocess(np.zeros(2 * _SR, dtype=np.float32), _SR)
        assert out.duration_seconds == pytest.approx(2.0)

    def test_is_deterministic(self) -> None:
        # Reproducibility is what makes a stored evaluation meaningful later.
        rng = np.random.default_rng(3)
        x = rng.standard_normal(16_000).astype(np.float32) * 0.1
        a = preprocess(x, _SR).samples
        b = preprocess(x, _SR).samples
        assert np.array_equal(a, b)

    def test_input_is_not_mutated(self) -> None:
        x = np.ones(4_000, dtype=np.float32)
        original = x.copy()
        preprocess(x, _SR)
        assert np.array_equal(x, original)

    def test_records_gain_for_audit(self) -> None:
        quiet = (np.sin(np.linspace(0, 100, 16_000)) * 0.02).astype(np.float32)
        out = preprocess(quiet, _SR)
        # A quiet recording must be visibly louder afterwards, and that fact has
        # to be reported rather than hidden.
        assert out.gain_db_applied > 0
        assert out.peak_before_normalize == pytest.approx(0.02, abs=1e-3)
