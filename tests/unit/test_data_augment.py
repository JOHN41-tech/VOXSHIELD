"""Unit tests for load-time augmentation.

The load-bearing properties here are the ones that fail *quietly*. A codec that
loses the sign, a seed derived from a visit counter, a noise corpus that silently
becomes synthetic, an evaluation split that gets augmented -- each of those still
returns audio, still trains, and still reports a number. So the tests assert the
transfer curves and the refusal behaviour rather than just "it ran".

The G.711 tests check the round trip against the numbers real G.711 produces
(~36 dB SNR, clipping at +/-0.980 for mu-law). A wrong codec that is merely lossy
passes any round-trip-is-not-identity check; only the measured SNR and clip point
tell the two apart.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxshield.data.augment import (
    G711_ALAW,
    G711_MULAW,
    AugmentedAudio,
    augment,
    available_codec_schemes,
    class_weights,
    derive_seed,
    g711_alaw_decode,
    g711_alaw_encode,
    g711_mulaw_decode,
    g711_mulaw_encode,
)
from voxshield.data.config import (
    BALANCE_NONE,
    BALANCE_WEIGHTED_SAMPLER,
    AugmentationConfig,
)
from voxshield.data.errors import AugmentationError

SAMPLE_RATE = 16_000
SPEECH_ID = "train/sample-001"

CODECS = [
    pytest.param(g711_mulaw_encode, g711_mulaw_decode, 0.980, id="mulaw"),
    pytest.param(g711_alaw_encode, g711_alaw_decode, 0.984, id="alaw"),
]


def aug_config(**overrides: object) -> AugmentationConfig:
    """An enabled configuration with every transform switched off by default.

    Gain is included in that set. ``AugmentationConfig`` ships a real default
    level range, so leaving it alone here would put a second, unreported
    transform in the middle of a test that thought it was measuring one thing --
    which is exactly how the noise-SNR test ended up measuring noise plus gain.
    """
    settings: dict[str, object] = {
        "enabled": True,
        "seed": 11,
        "gain_db_min": 0.0,
        "gain_db_max": 0.0,
    }
    settings.update(overrides)
    return AugmentationConfig(**settings)  # type: ignore[arg-type]


def everything_on(**overrides: object) -> AugmentationConfig:
    """A configuration that applies every transform to every window."""
    return aug_config(
        noise_probability=1.0,
        codec_probability=1.0,
        channel_probability=1.0,
        reverb_probability=1.0,
        **overrides,
    )


def sweep(count: int = 400_001) -> np.ndarray:
    """A full-scale linear sweep, for exercising both halves of the range."""
    return np.linspace(-1.0, 1.0, count, dtype=np.float64)


def snr_db(reference: np.ndarray, degraded: np.ndarray) -> float:
    """Signal-to-noise ratio of a degraded signal against its reference.

    Takes the *degraded signal*, not the residual: the error term is computed
    here as ``reference - degraded``. Handing it a residual subtracts a second
    time and measures the degraded signal against itself, which is 0 dB for any
    SNR at all.
    """
    reference = np.asarray(reference, dtype=np.float64)
    error = reference - np.asarray(degraded, dtype=np.float64)
    return 10.0 * math.log10(float(np.sum(reference**2)) / float(np.sum(error**2)))


# --------------------------------------------------------------------------
# The train-only rule
# --------------------------------------------------------------------------


def test_disabled_configuration_returns_the_input_untouched(speech_samples: np.ndarray) -> None:
    result = augment(speech_samples, SAMPLE_RATE, AugmentationConfig(enabled=False), split="train")

    assert result.applied == ()
    assert np.array_equal(result.waveform, speech_samples)


def test_no_configuration_at_all_returns_the_input_untouched(speech_samples: np.ndarray) -> None:
    result = augment(speech_samples, SAMPLE_RATE, None, split="train")

    assert np.array_equal(result.waveform, speech_samples)


@pytest.mark.parametrize("split", ["dev", "test", "eval", "all"])
def test_augmenting_a_non_train_split_is_refused(speech_samples: np.ndarray, split: str) -> None:
    """The guard is a parameter, not a convention.

    An augmented evaluation split measures the augmentation rather than the
    detector, and reports a number that will not reproduce elsewhere.
    """
    with pytest.raises(AugmentationError, match="train"):
        augment(speech_samples, SAMPLE_RATE, everything_on(), split=split)


def test_a_disabled_configuration_may_be_applied_to_any_split(
    speech_samples: np.ndarray,
) -> None:
    """Disabling augmentation makes it safe everywhere, which is the point."""
    result = augment(
        speech_samples,
        SAMPLE_RATE,
        AugmentationConfig(enabled=False),
        split="test",
    )

    assert np.array_equal(result.waveform, speech_samples)


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_audio(speech_samples: np.ndarray) -> None:
    config = everything_on()

    first = augment(speech_samples, SAMPLE_RATE, config, split="train", sample_id=SPEECH_ID)
    second = augment(speech_samples, SAMPLE_RATE, config, split="train", sample_id=SPEECH_ID)

    assert first.seed == second.seed
    assert np.array_equal(first.waveform, second.waveform)
    assert first.applied == second.applied


def test_a_new_epoch_gives_different_audio(speech_samples: np.ndarray) -> None:
    config = everything_on()

    first = augment(
        speech_samples, SAMPLE_RATE, config, split="train", sample_id=SPEECH_ID, epoch=0
    )
    second = augment(
        speech_samples, SAMPLE_RATE, config, split="train", sample_id=SPEECH_ID, epoch=1
    )

    assert first.seed != second.seed
    assert not np.array_equal(first.waveform, second.waveform)


def test_a_different_sample_gets_different_audio(speech_samples: np.ndarray) -> None:
    config = everything_on()

    first = augment(speech_samples, SAMPLE_RATE, config, split="train", sample_id="a")
    second = augment(speech_samples, SAMPLE_RATE, config, split="train", sample_id="b")

    assert not np.array_equal(first.waveform, second.waveform)


def test_the_seed_does_not_depend_on_visit_order() -> None:
    """A counter-derived seed would change with batch size or sampler.

    That is the failure this guards: the same corpus augmented differently
    depending on how the loader happened to walk it, which cannot be reproduced
    from the manifest alone.
    """
    forward = [derive_seed(11, name) for name in ("a", "b", "c", "d")]
    backward = [derive_seed(11, name) for name in ("d", "c", "b", "a")]

    assert forward == list(reversed(backward))
    assert len(set(forward)) == 4


def test_the_seed_is_reproducible_across_processes() -> None:
    """A seed that only holds inside one interpreter is not reproducible at all.

    This is asserted by actually running the derivation in two subprocesses under
    different ``PYTHONHASHSEED`` values, because a same-process comparison cannot
    see the bug it is aimed at: the built-in ``hash()`` is stable *within* one
    process and only diverges *between* them, so an in-process assertion passes
    against exactly the implementation that is broken across a DataLoader's
    workers and a resumed checkpoint.
    """
    program = (
        "import sys;"
        "from voxshield.data.augment import derive_seed;"
        "print(derive_seed(11, 'train/sample-001', 0))"
    )

    def run(hash_seed: str) -> int:
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return int(result.stdout.strip())

    first, second = run("0"), run("12345")

    assert first == second
    # Pinned so a change of digest algorithm, field encoding, or field order is a
    # visible diff rather than a silent re-shuffle of every augmented epoch.
    assert first == 612883921


# --------------------------------------------------------------------------
# Shape and range contracts
# --------------------------------------------------------------------------


def test_augmentation_never_changes_the_window_length(speech_samples: np.ndarray) -> None:
    result = augment(speech_samples, SAMPLE_RATE, everything_on(), split="train")

    assert result.waveform.shape == speech_samples.shape


def test_augmented_audio_stays_within_full_scale(speech_samples: np.ndarray) -> None:
    """A transform chain must not clip; clipping is distortion nobody asked for."""
    loud = (speech_samples * 0.99).astype(np.float32)

    result = augment(
        loud, SAMPLE_RATE, everything_on(gain_db_min=6.0, gain_db_max=6.0), split="train"
    )

    assert result.waveform.dtype == np.float32
    assert float(np.max(np.abs(result.waveform))) <= 1.0


def test_a_multichannel_input_is_rejected(speech_samples: np.ndarray) -> None:
    stereo = np.stack([speech_samples, speech_samples], axis=1)

    with pytest.raises(AugmentationError, match="1-D"):
        augment(stereo, SAMPLE_RATE, everything_on(), split="train")


def test_an_empty_input_is_rejected() -> None:
    with pytest.raises(AugmentationError, match="non-empty"):
        augment(np.zeros(0, dtype=np.float32), SAMPLE_RATE, everything_on(), split="train")


def test_augmented_audio_rejects_a_non_float_waveform() -> None:
    with pytest.raises(AugmentationError, match="float32"):
        AugmentedAudio(waveform=np.zeros(4, dtype=np.float64))


# --------------------------------------------------------------------------
# G.711 codec correctness
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("encode", "decode", "clip"), CODECS)
def test_codec_round_trip_is_actually_good(encode, decode, clip) -> None:
    """A wrong-but-lossy codec still round-trips; only the SNR tells them apart.

    Real G.711 lands near 36 dB. Anything much below that is a broken transfer
    curve, not a codec.
    """
    x = sweep()

    y = decode(encode(x))

    assert snr_db(x, y) > 30.0


@pytest.mark.parametrize(("encode", "decode", "clip"), CODECS)
def test_codec_preserves_the_sign_of_every_sample(encode, decode, clip) -> None:
    """The failure this catches is invisible: the audio still sounds like speech.

    Collapsing the negative half onto the positive one is a monotone, lossless
    looking transform of the wrong signal.
    """
    x = sweep()

    y = decode(encode(x))

    assert np.all(y[x < -0.05] < 0.0)
    assert np.all(y[x > 0.05] > 0.0)


@pytest.mark.parametrize(("encode", "decode", "clip"), CODECS)
def test_codec_is_monotone_on_each_half(encode, decode, clip) -> None:
    """A codec with a folded transfer curve is not merely lossy, it is wrong."""
    x = sweep()
    y = decode(encode(x))

    assert np.all(np.diff(y[x > 0]) >= 0.0)
    # Read the negative half oldest-to-newest, which is the order ``x`` arrives in:
    # a symmetric magnitude codec makes a nondecreasing signal *more* negative as
    # the input grows toward zero. Asserting "<= 0" here would be asserting that
    # the codec is wrong, and would have passed a folded transfer curve.
    assert np.all(np.diff(y[x < 0]) >= 0.0)


@pytest.mark.parametrize(("encode", "decode", "clip"), CODECS)
def test_codec_output_stays_within_full_scale(encode, decode, clip) -> None:
    y = decode(encode(sweep()))

    assert float(np.max(np.abs(y))) <= 1.0


@pytest.mark.parametrize(("encode", "decode", "clip"), CODECS)
def test_codec_clips_at_the_real_g711_full_scale(encode, decode, clip) -> None:
    """Pins the codec to G.711 rather than to something that merely sounds like it.

    mu-law saturates at 32124/32768 and A-law at 32256/32768; those two constants
    are the fingerprint of the real thing.
    """
    full = decode(encode(np.array([1.0, -1.0])))

    assert full[0] == pytest.approx(clip, abs=1e-3)
    assert full[1] == pytest.approx(-clip, abs=1e-3)


@pytest.mark.parametrize(("encode", "decode", "clip"), CODECS)
def test_codec_near_silence_stays_near_silence(encode, decode, clip) -> None:
    y = decode(encode(np.zeros(16)))

    assert float(np.max(np.abs(y))) < 0.01


@pytest.mark.parametrize(("encode", "decode", "clip"), CODECS)
def test_codec_uses_all_of_its_code_space(encode, decode, clip) -> None:
    """A codec that maps a whole range onto a few codes is broken, not efficient."""
    codes = encode(sweep())

    assert codes.dtype == np.uint8
    assert int(np.unique(codes).size) == 256


@pytest.mark.parametrize(("encode", "decode", "clip"), CODECS)
def test_codec_is_stable_for_identical_input(encode, decode, clip) -> None:
    x = sweep(10_001)

    assert np.array_equal(decode(encode(x)), decode(encode(x)))


# --------------------------------------------------------------------------
# Codec scheme reporting
# --------------------------------------------------------------------------


def test_an_unimplemented_codec_scheme_is_reported_not_hidden(
    speech_samples: np.ndarray,
) -> None:
    """A skipped codec looks exactly like a codec that ran.

    ``opus`` is not implemented here, so asking for it must surface rather than
    quietly produce un-codec'd audio that the report then calls clean.
    """
    config = everything_on(codec_schemes=("opus",))

    result = augment(speech_samples, SAMPLE_RATE, config, split="train")

    assert result.unavailable == ("opus",)
    assert any("opus" in note for note in result.notes)
    assert not any(name.startswith("codec:") for name in result.applied)


def test_a_configured_codec_is_reported_by_name(speech_samples: np.ndarray) -> None:
    config = everything_on(codec_schemes=(G711_ALAW,))

    result = augment(speech_samples, SAMPLE_RATE, config, split="train")

    assert "codec:g711_alaw" in result.applied


def test_available_codec_schemes_keeps_request_order() -> None:
    assert available_codec_schemes(("amr", G711_MULAW, G711_ALAW, "wav")) == (
        G711_MULAW,
        G711_ALAW,
    )


# --------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------


def test_a_missing_noise_corpus_falls_back_and_says_so(speech_samples: np.ndarray) -> None:
    config = aug_config(noise_probability=1.0, noise_corpus_dir="does/not/exist")

    result = augment(speech_samples, SAMPLE_RATE, config, split="train")

    assert "noise" in result.applied
    assert any("synthetic" in note for note in result.notes)


def test_a_real_noise_corpus_is_used_when_present(
    speech_samples: np.ndarray,
    tmp_path: Path,
) -> None:
    noise_dir = tmp_path / "noise"
    noise_dir.mkdir()
    rng = np.random.default_rng(3)
    for index in range(3):
        sf.write(
            noise_dir / f"n{index}.wav",
            rng.standard_normal(SAMPLE_RATE).astype(np.float32) * 0.2,
            SAMPLE_RATE,
        )

    config = aug_config(noise_probability=1.0, noise_corpus_dir=str(noise_dir))

    result = augment(speech_samples, SAMPLE_RATE, config, split="train")

    assert "noise" in result.applied
    assert not any("synthetic" in note for note in result.notes)


def test_added_noise_lands_inside_the_requested_snr_band(speech_samples: np.ndarray) -> None:
    config = aug_config(noise_probability=1.0, noise_snr_db_min=25.0, noise_snr_db_max=25.0)

    result = augment(speech_samples, SAMPLE_RATE, config, split="train")

    assert "noise" in result.applied
    assert snr_db(speech_samples, result.waveform) == pytest.approx(25.0, abs=1.5)


def test_silence_survives_noise_undamaged() -> None:
    """A zero-power signal has no SNR to hit, and must not divide by zero."""
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)

    result = augment(silence, SAMPLE_RATE, aug_config(noise_probability=1.0), split="train")

    assert np.all(np.isfinite(result.waveform))


# --------------------------------------------------------------------------
# Class balancing
# --------------------------------------------------------------------------


def test_class_weights_are_none_when_balancing_is_off() -> None:
    assert class_weights([0, 0, 0, 1], AugmentationConfig(enabled=True)) is None


def test_class_weights_reach_one_on_average_per_class() -> None:
    """The sampler must equalise the draw without inventing rows.

    Weights rather than an index list: a duplicated row would be counted twice in
    the dataset statistics and would trip the duplicate check on the corpus.
    """
    labels = [0] * 90 + [1] * 10
    config = AugmentationConfig(enabled=True, class_balance=BALANCE_WEIGHTED_SAMPLER)

    weights = class_weights(labels, config)

    assert weights is not None
    assert weights.shape == (100,)
    assert float(weights[:90].sum()) == pytest.approx(weights[90:].sum(), rel=0.01)
    assert float(weights.sum()) == pytest.approx(100.0, rel=1e-6)


def test_class_weights_give_a_minority_class_a_larger_draw() -> None:
    """Per-sample weight, not per-class total, is what the sampler actually uses.

    The minority class here holds 10 of 100 rows. Equalising the two classes'
    *totals* necessarily gives each of those 10 rows a weight ten times larger
    than each of the 90 majority rows, so the assertion is inverted relative to a
    reading of the name: 9 out of 10 draws must be able to land on the minority.
    """
    weights = class_weights(
        [0] * 90 + [1] * 10,
        AugmentationConfig(enabled=True, class_balance=BALANCE_WEIGHTED_SAMPLER),
    )

    assert weights is not None
    assert float(weights[99]) > float(weights[0])


def test_class_weights_handle_an_empty_label_list() -> None:
    config = AugmentationConfig(enabled=True, class_balance=BALANCE_WEIGHTED_SAMPLER)

    assert class_weights([], config) is None


def test_class_weights_reject_an_unknown_strategy() -> None:
    hacked = object.__new__(AugmentationConfig)
    object.__setattr__(hacked, "class_balance", "oversample_by_duplication")

    with pytest.raises(AugmentationError, match="unsupported class_balance"):
        class_weights([0, 1], hacked)


def test_the_none_strategy_is_accepted() -> None:
    assert (
        class_weights([0, 1], AugmentationConfig(enabled=True, class_balance=BALANCE_NONE)) is None
    )


# --------------------------------------------------------------------------
# Reported application
# --------------------------------------------------------------------------


def test_transforms_report_themselves(speech_samples: np.ndarray) -> None:
    result = augment(speech_samples, SAMPLE_RATE, everything_on(), split="train")

    assert "noise" in result.applied
    assert "channel" in result.applied
    assert "reverb" in result.applied


def test_gain_is_not_applied_when_the_range_is_flat_at_zero(speech_samples: np.ndarray) -> None:
    config = aug_config(gain_db_min=0.0, gain_db_max=0.0)

    result = augment(speech_samples, SAMPLE_RATE, config, split="train")

    assert "gain" not in result.applied
    assert np.array_equal(result.waveform, speech_samples)


def test_band_limiting_removes_energy_above_the_telephone_band(speech_samples: np.ndarray) -> None:
    config = aug_config(channel_probability=1.0)

    result = augment(speech_samples, SAMPLE_RATE, config, split="train")

    from scipy import signal

    _, original = signal.periodogram(speech_samples.astype(np.float64), SAMPLE_RATE)
    _, filtered = signal.periodogram(result.waveform.astype(np.float64), SAMPLE_RATE)
    frequencies = np.fft.rfftfreq(speech_samples.size, d=1.0 / SAMPLE_RATE)
    high = frequencies > 4000.0
    assert float(filtered[high].sum()) < float(original[high].sum())


def test_reverb_does_not_change_the_length(speech_samples: np.ndarray) -> None:
    config = aug_config(reverb_probability=1.0)

    result = augment(speech_samples, SAMPLE_RATE, config, split="train")

    assert result.waveform.size == speech_samples.size


def test_probability_zero_never_applies(speech_samples: np.ndarray) -> None:
    config = aug_config(
        noise_probability=0.0,
        codec_probability=0.0,
        channel_probability=0.0,
        reverb_probability=0.0,
        gain_db_min=0.0,
        gain_db_max=0.0,
    )

    result = augment(speech_samples, SAMPLE_RATE, config, split="train")

    assert result.applied == ()
    assert np.array_equal(result.waveform, speech_samples)


@pytest.fixture
def every_transform() -> Iterator[AugmentedAudio]:
    """One fully augmented window, for tests that only need the object."""
    samples = np.sin(2.0 * np.pi * 220.0 * np.arange(SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32)
    yield augment(samples, SAMPLE_RATE, everything_on(), split="train", sample_id="fixture")
