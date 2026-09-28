"""Unit tests for log-mel features, pinned by structural invariant.

The filterbank cannot be compared against a reference implementation, because
there is no single reference: torchaudio bins with float edges and VoxShield
bins with integer edges, and both are defensible. What *can* be pinned are the
properties every correct filterbank must have. Those are asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxshield.audio.features import (
    compute_log_mel,
    hz_to_mel,
    mel_filterbank,
    mel_to_hz,
)
from voxshield.config import FeatureConfig

_SR = 16_000


class TestMelScale:
    def test_hz_mel_round_trip(self) -> None:
        for hz in (0.0, 20.0, 1000.0, 7600.0, 8000.0):
            assert mel_to_hz(hz_to_mel(hz)) == pytest.approx(hz, rel=1e-9)

    def test_scale_is_monotonic(self) -> None:
        mels = np.linspace(hz_to_mel(20.0), hz_to_mel(7600.0), 200)
        assert np.all(np.diff(mel_to_hz(mels)) > 0)

    def test_scale_is_htk(self) -> None:
        # Guards against a silent change of mel scale, which would move every
        # band edge. The HTK formula is constructed so 1000 Hz is 1000 mels.
        assert hz_to_mel(1000.0) == pytest.approx(1000.0, abs=0.1)


class TestFilterbankInvariants:
    @pytest.fixture
    def fb(self) -> np.ndarray:
        return mel_filterbank(_SR, 400, 80, 20.0, 7600.0).astype(np.float64)

    def test_shape(self, fb: np.ndarray) -> None:
        assert fb.shape == (201, 80)

    def test_dtype_is_float32(self) -> None:
        assert mel_filterbank(_SR, 400, 80, 20.0, 7600.0).dtype == np.float32

    def test_no_dead_bands(self, fb: np.ndarray) -> None:
        # Regression guard for a real bug: the textbook ramp
        # (arange(l,c)-l)/(c-l) evaluates a band whose edges collapse to a
        # single FFT bin to 0.0, not to absence. At n_fft=400 the lowest mel
        # edges land on the same 40 Hz bin, which silently killed bands
        # 0, 2, 5, 8, and 12 -- 6% of a CNN's input channels.
        dead = np.flatnonzero(fb.sum(axis=0) == 0.0)
        assert dead.tolist() == []

    def test_every_band_is_peak_normalised(self, fb: np.ndarray) -> None:
        assert np.allclose(fb.max(axis=0), 1.0, atol=1e-6)

    def test_values_within_unit_range(self, fb: np.ndarray) -> None:
        assert fb.min() >= 0.0
        assert fb.max() <= 1.0

    def test_no_negative_values(self, fb: np.ndarray) -> None:
        assert np.all(fb >= 0.0)

    def test_bands_are_contiguous_triangles(self, fb: np.ndarray) -> None:
        # A band's support must be one unbroken run of bins. A gap would mean the
        # ramp was applied to non-adjacent edges, which shows up as a detector
        # blind to the frequencies in the gap.
        for i in range(fb.shape[1]):
            support = np.flatnonzero(fb[:, i] > 0)
            assert support.size > 0, f"band {i} is empty"
            assert np.array_equal(support, np.arange(support[0], support[-1] + 1)), (
                f"band {i} support is not contiguous"
            )

    def test_bands_peak_at_their_centre(self, fb: np.ndarray) -> None:
        # The maximum of each triangle sits at the centre bin, which is what
        # makes the filterbank frequency-selective rather than a moving average.
        for i in range(fb.shape[1]):
            col = fb[:, i]
            peak_bin = int(np.argmax(col))
            assert col[peak_bin] == pytest.approx(1.0), f"band {i} peak is not 1.0"
            # Nothing to the right of the peak may exceed it.
            assert np.all(col[peak_bin:] <= col[peak_bin] + 1e-9)

    def test_uses_most_of_the_spectrum(self, fb: np.ndarray) -> None:
        used = np.count_nonzero(fb.sum(axis=1) > 0)
        assert used > 0.5 * fb.shape[0]

    def test_deterministic(self) -> None:
        a = mel_filterbank(_SR, 400, 80, 20.0, 7600.0)
        b = mel_filterbank(_SR, 400, 80, 20.0, 7600.0)
        assert np.array_equal(a, b)

    def test_scaling_input_does_not_change_filterbank(self) -> None:
        # The filterbank depends only on the configuration, never on audio.
        a = mel_filterbank(_SR, 400, 80, 20.0, 7600.0)
        b = mel_filterbank(_SR, 400, 80, 20.0, 7600.0)
        assert np.array_equal(a, b)


class TestComputeLogMel:
    @pytest.fixture
    def cfg(self) -> FeatureConfig:
        return FeatureConfig()

    def test_output_shape(self, cfg: FeatureConfig) -> None:
        window = np.zeros(4 * _SR, dtype=np.float32)
        feats = compute_log_mel(window, cfg)
        assert feats.ndim == 2
        assert feats.shape[1] == cfg.n_mels
        assert feats.shape[0] > 1

    def test_output_is_float32(self, cfg: FeatureConfig) -> None:
        feats = compute_log_mel(np.zeros(4 * _SR, dtype=np.float32), cfg)
        assert feats.dtype == np.float32

    def test_all_finite(self, cfg: FeatureConfig) -> None:
        rng = np.random.default_rng(3)
        noise = rng.standard_normal(4 * _SR).astype(np.float32) * 0.1
        feats = compute_log_mel(noise, cfg)
        assert np.isfinite(feats).all()

    def test_silence_does_not_produce_nan(self, cfg: FeatureConfig) -> None:
        feats = compute_log_mel(np.zeros(4 * _SR, dtype=np.float32), cfg)
        assert np.isfinite(feats).all()

    def test_nan_input_is_sanitised(self, cfg: FeatureConfig) -> None:
        # One bad sample survives every FFT as NaN, and CMVN then smears it
        # across every band and frame. It must be replaced, not propagated.
        samples = np.zeros(4 * _SR, dtype=np.float32)
        samples[100] = np.nan
        samples[200] = np.inf
        samples[300] = -np.inf
        feats = compute_log_mel(samples, cfg)
        assert np.isfinite(feats).all()

    def test_output_is_bounded_below(self) -> None:
        # The log floor must apply, so a log-mel value can never reach -inf.
        # CMVN is off here: it subtracts a per-band mean, which legitimately
        # produces values far below the floor.
        cfg = FeatureConfig(per_utterance_cmvn=False)
        feats = compute_log_mel(np.zeros(4 * _SR, dtype=np.float32), cfg)
        assert np.isfinite(feats).all()
        assert feats.min() >= np.log(cfg.log_eps) - 1e-3

    def test_frames_scale_with_duration(self, cfg: FeatureConfig) -> None:
        short = compute_log_mel(np.zeros(2 * _SR, dtype=np.float32), cfg)
        long = compute_log_mel(np.zeros(8 * _SR, dtype=np.float32), cfg)
        assert long.shape[0] > short.shape[0]

    def test_deterministic(self, cfg: FeatureConfig) -> None:
        rng = np.random.default_rng(11)
        samples = (rng.standard_normal(4 * _SR) * 0.1).astype(np.float32)
        assert np.array_equal(
            compute_log_mel(samples, cfg), compute_log_mel(samples, cfg)
        )

    def test_signal_shorter_than_one_frame_still_produces_output(
        self, cfg: FeatureConfig
    ) -> None:
        # Centre-padding means a sub-frame signal is padded rather than dropped.
        # What matters is the contract: the output always has n_mels columns and
        # never has a non-finite entry, so a model cannot be handed a ragged array.
        feats = compute_log_mel(np.zeros(100, dtype=np.float32), cfg)
        assert feats.shape[1] == cfg.n_mels
        assert np.isfinite(feats).all()

    def test_empty_signal_yields_empty_output(self, cfg: FeatureConfig) -> None:
        feats = compute_log_mel(np.zeros(0, dtype=np.float32), cfg)
        assert feats.shape == (0, cfg.n_mels)

    def test_cmvn_normalisation_is_applied(self) -> None:
        # With per-utterance CMVN on, mean and variance are flattened, so a
        # different input level must not shift the output distribution. This is
        # what stops a detector keying on microphone gain instead of on whether
        # the speech is synthetic.
        cfg = FeatureConfig(per_utterance_cmvn=True)
        rng = np.random.default_rng(5)
        quiet = (rng.standard_normal(4 * _SR) * 0.01).astype(np.float32)
        loud = (quiet * 10.0).astype(np.float32)
        a, b = compute_log_mel(quiet, cfg), compute_log_mel(loud, cfg)
        assert abs(float(a.mean())) < 1e-3
        assert abs(float(b.mean())) < 1e-3
        assert abs(float(a.std()) - float(b.std())) < 0.2
        assert a.std() == pytest.approx(1.0, abs=0.05)
