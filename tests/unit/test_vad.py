"""Unit tests for the frame-based VAD.

The VAD decides *where* speech is, so a false positive inflates the evidence
and a false negative erases it. The tests therefore assert behaviour a detector
can rely on -- gain invariance, abstention on non-speech, and no fabricated
regions -- rather than a fixed accuracy number that would only hold for one
threshold setting.
"""

from __future__ import annotations

import numpy as np
import pytest

from voxshield.audio.preprocess import preprocess
from voxshield.audio.vad import (
    SpeechMask,
    compute_frame_features,
    detect_speech,
    frame_signal,
)
from voxshield.config import AudioConfig, VadConfig

_SR = 16_000


def _tone(seconds: float, hz: float, sr: int = _SR, amplitude: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class TestFraming:
    def test_shape_and_stride(self) -> None:
        x = np.zeros(1_600, dtype=np.float32)
        frames = frame_signal(x, frame_length=400, hop_length=160)
        assert frames.ndim == 2
        assert frames.shape[1] == 400
        # (1600 - 400) // 160 + 1
        assert frames.shape[0] == 8

    def test_empty_signal(self) -> None:
        assert frame_signal(np.zeros(0, dtype=np.float32), 400, 160).shape[0] == 0

    def test_signal_shorter_than_one_frame(self) -> None:
        assert frame_signal(np.zeros(10, dtype=np.float32), 400, 160).shape[0] == 0

    def test_frames_preserve_values(self) -> None:
        x = np.arange(1_000, dtype=np.float32)
        frames = frame_signal(x, 400, 160)
        assert frames[0, :5].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


class TestFrameFeatures:
    def test_silence_is_quiet_and_flat(self) -> None:
        frames = frame_signal(np.zeros(4_000, dtype=np.float32), 400, 160)
        energy_db, zcr, flatness = compute_frame_features(frames)
        assert energy_db.max() < -100
        assert zcr.max() == 0.0
        assert np.all((flatness >= 0.0) & (flatness <= 1.0))

    def test_tone_has_energy_and_low_zero_crossings(self) -> None:
        frames = frame_signal(_tone(1.0, 220.0), 400, 160)
        energy_db, zcr, _ = compute_frame_features(frames)
        assert energy_db.mean() > -40
        assert zcr.mean() < 0.35

    def test_white_noise_is_spectrally_flat(self) -> None:
        # Flatness is the feature that separates hiss from voiced speech; if it
        # stops working, the noise gate stops working with it.
        rng = np.random.default_rng(2)
        noise = (rng.standard_normal(8_000) * 0.1).astype(np.float32)
        frames = frame_signal(noise, 400, 160)
        _, _, flatness = compute_frame_features(frames)
        assert flatness.mean() > 0.3


class TestDetectSpeech:
    def test_finds_speech(self, speech_samples: np.ndarray) -> None:
        mask = detect_speech(speech_samples, _SR)
        assert isinstance(mask, SpeechMask)
        assert mask.has_speech
        assert mask.speech_seconds > 2.0
        assert 0.0 < mask.speech_ratio <= 1.0

    def test_rejects_silence(self) -> None:
        mask = detect_speech(np.zeros(4 * _SR, dtype=np.float32), _SR)
        assert mask.speech_seconds == 0.0
        assert mask.has_speech is False
        assert mask.regions() == []

    def test_rejects_white_noise(self) -> None:
        # The abstention-leaning direction: better to ask for more audio than to
        # invent a speech region and score it.
        rng = np.random.default_rng(9)
        noise = (rng.standard_normal(4 * _SR) * 0.2).astype(np.float32)
        mask = detect_speech(noise, _SR)
        assert mask.speech_seconds < 4 * 0.5

    def test_is_gain_invariant_after_normalisation(self, speech_samples: np.ndarray) -> None:
        # The pipeline normalises loudness before the VAD runs, so the contract
        # is that a quiet and a loud copy of the same call agree once each has
        # been through preprocessing. Testing the raw gain range instead would
        # only assert that the absolute floor exists.
        quiet = detect_speech(
            preprocess((speech_samples * 0.02).astype(np.float32), _SR).samples, _SR
        )
        loud = detect_speech(
            preprocess((speech_samples * 0.8).astype(np.float32), _SR).samples, _SR
        )
        assert quiet.has_speech and loud.has_speech
        assert quiet.speech_ratio == pytest.approx(loud.speech_ratio, abs=0.1)

    def test_below_absolute_floor_abstains(self, speech_samples: np.ndarray) -> None:
        # The floor is a hard gate by design: a clip so quiet that lifting it
        # would mean amplifying noise must produce no speech, not a guess.
        mask = detect_speech((speech_samples * 0.005).astype(np.float32), _SR)
        assert mask.has_speech is False

    def test_is_deterministic(self, speech_samples: np.ndarray) -> None:
        a = detect_speech(speech_samples, _SR)
        b = detect_speech(speech_samples, _SR)
        assert np.array_equal(a.is_speech, b.is_speech)
        assert a.threshold_dbfs == b.threshold_dbfs

    def test_speech_seconds_matches_mask(self, speech_samples: np.ndarray) -> None:
        # speech_seconds is the authoritative number callers use to decide
        # whether to abstain, so it must be derived from the mask, not estimated.
        mask = detect_speech(speech_samples, _SR)
        assert mask.speech_seconds == pytest.approx(
            int(np.count_nonzero(mask.is_speech)) * mask.hop_seconds
        )

    def test_regions_are_within_bounds(self, speech_samples: np.ndarray) -> None:
        mask = detect_speech(speech_samples, _SR)
        duration = len(speech_samples) / _SR
        for start, end in mask.regions():
            assert 0.0 <= start < end <= duration + mask.hop_seconds

    def test_regions_never_invent_speech_in_silence(self) -> None:
        # A leading and trailing silence band must stay non-speech.
        signal = np.zeros(6 * _SR, dtype=np.float32)
        signal[2 * _SR : 4 * _SR] = _tone(2.0, 220.0)
        mask = detect_speech(signal, _SR)
        regions = mask.regions()
        assert regions, "the embedded tone should be found"
        assert regions[0][0] > 0.5, "leading silence was classified as speech"
        assert regions[-1][1] < 5.5, "trailing silence was classified as speech"

    def test_empty_input(self) -> None:
        mask = detect_speech(np.zeros(0, dtype=np.float32), _SR)
        assert mask.n_frames == 0
        assert mask.speech_ratio == 0.0
        assert mask.has_speech is False

    def test_non_finite_input_does_not_crash(self) -> None:
        # A single bad sample would otherwise poison the percentiles for the
        # whole file, collapse the seed arm, and let the fallback threshold mark
        # near-silence as speech.
        signal = np.zeros(2 * _SR, dtype=np.float32)
        signal[_SR : _SR + 8_000] = _tone(0.5, 220.0)
        signal[10] = np.nan
        signal[11] = np.inf
        mask = detect_speech(signal, _SR)
        assert np.isfinite(mask.threshold_dbfs)
        assert 0.0 < mask.speech_ratio < 0.9
        assert not mask.is_speech[0], "a NaN frame must not read as loud speech"

    def test_threshold_is_reported_for_audit(self, speech_samples: np.ndarray) -> None:
        # The adaptive threshold is a diagnostic; without it an operator cannot
        # tell a wrong decision from an unlucky threshold.
        mask = detect_speech(speech_samples, _SR)
        assert np.isfinite(mask.threshold_dbfs)
        assert -120.0 < mask.threshold_dbfs < 0.0

    def test_threshold_never_below_absolute_floor(self, speech_samples: np.ndarray) -> None:
        cfg = AudioConfig(vad=VadConfig(absolute_floor_dbfs=-20.0, dynamic_range_db=200.0))
        mask = detect_speech(speech_samples, _SR, cfg)
        assert mask.threshold_dbfs >= -20.0

    def test_short_blip_is_dropped(self) -> None:
        # A 50 ms click is an artefact, not speech; the morphological cleanup
        # exists so a single click cannot create a segment to score.
        signal = np.zeros(4 * _SR, dtype=np.float32)
        signal[2 * _SR : 2 * _SR + 800] = _tone(0.05, 300.0)
        mask = detect_speech(signal, _SR)
        assert mask.speech_seconds == pytest.approx(0.0, abs=0.05)

    def test_internal_pause_is_bridged(self) -> None:
        # A 100 ms gap mid-utterance is a stop consonant, not a segment boundary.
        signal = _tone(4.0, 220.0)
        signal[2 * _SR + 500 : 2 * _SR + 1_500] = 0.0
        mask = detect_speech(signal, _SR)
        assert len(mask.regions()) <= 2
