"""The unified loader: one entry point, one set of limits, honest provenance."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxshield.audio.decode import summarise_metadata
from voxshield.audio.loader import load_audio
from voxshield.config import AudioConfig
from voxshield.errors import (
    AudioDecodeError,
    AudioIntakeError,
    AudioTooLargeError,
    InvalidAudioSignalError,
    UnsupportedAudioFormatError,
)

_SR = 16_000


def _mono(seconds: float = 2.0, amplitude: float = 0.3) -> np.ndarray:
    t = np.arange(int(_SR * seconds)) / _SR
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _wav_bytes(seconds: float = 2.0, sample_rate: int = _SR, subtype: str = "PCM_16") -> bytes:
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    tone = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, tone, sample_rate, format="WAV", subtype=subtype)
    return buffer.getvalue()


class TestEncodedSources:
    @pytest.mark.parametrize(
        "wrap",
        [
            pytest.param(lambda b: b, id="bytes"),
            pytest.param(bytearray, id="bytearray"),
            pytest.param(memoryview, id="memoryview"),
            pytest.param(lambda b: io.BytesIO(b), id="stream"),
        ],
    )
    def test_every_encoded_shape_loads_identically(self, wrap) -> None:
        reference = load_audio(_wav_bytes())
        loaded = load_audio(wrap(_wav_bytes()))

        assert loaded.sample_rate == reference.sample_rate
        assert loaded.samples.shape == reference.samples.shape
        assert np.allclose(loaded.samples, reference.samples)
        assert loaded.source_kind == "encoded"

    def test_a_path_loads_the_same_as_its_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "clip.wav"
        path.write_bytes(_wav_bytes())

        from_path = load_audio(path)
        from_str = load_audio(str(path))

        assert np.allclose(from_path.samples, from_str.samples)
        assert from_path.source_kind == "encoded"

    def test_a_missing_path_is_an_intake_error(self, tmp_path: Path) -> None:
        with pytest.raises(AudioIntakeError):
            load_audio(tmp_path / "absent.wav")

    def test_a_disallowed_format_is_refused(self) -> None:
        # The container is detected by libsndfile, never by the ``.wav``-style
        # name we hand it, so a real AIFF behind a WAV-looking request is
        # refused on its actual type.
        buffer = io.BytesIO()
        sf.write(buffer, _mono(), _SR, format="AIFF", subtype="PCM_16")
        payload = buffer.getvalue()

        assert "AIFF" not in AudioConfig().allowed_formats
        with pytest.raises(UnsupportedAudioFormatError, match="AIFF"):
            load_audio(payload)

    def test_a_corrupt_header_is_a_decode_error(self) -> None:
        with pytest.raises(AudioDecodeError):
            load_audio(b"RIFF" + b"\x00" * 64)

    def test_empty_bytes_are_refused(self) -> None:
        with pytest.raises(AudioDecodeError):
            load_audio(b"")

    def test_a_caller_supplied_rate_does_not_override_the_container(self) -> None:
        # The WAV is 16 kHz. Passing 8 kHz must not resample anything: a silent
        # rate mismatch would shift every frequency in the clip.
        loaded = load_audio(_wav_bytes(), 8_000)

        assert loaded.sample_rate == _SR


class TestArraySources:
    def test_a_mono_array_loads(self) -> None:
        loaded = load_audio(_mono(), _SR)

        # A 1-D input keeps the channel axis, matching the encoded path for mono
        # exactly. Callers must not have to branch on where the audio came from.
        assert loaded.samples.shape == (_mono().size, 1)
        assert loaded.sample_rate == _SR
        assert loaded.source_kind == "array"
        assert loaded.source_format == "RAW"

    def test_an_array_is_downmixed_by_default(self) -> None:
        stereo = np.stack([_mono(), _mono()], axis=1)

        assert load_audio(stereo, _SR).samples.shape == (_mono().size,)
        assert load_audio(stereo, _SR, downmix=False).samples.shape == stereo.shape

    def test_an_int_array_is_accepted_and_converted(self) -> None:
        loaded = load_audio((np.arange(4_000) % 400).astype(np.int16), _SR)

        assert loaded.samples.dtype == np.float32
        assert np.isfinite(loaded.samples).all()

    def test_a_missing_sample_rate_is_refused(self) -> None:
        with pytest.raises(AudioIntakeError, match="sample_rate is required"):
            load_audio(_mono())

    @pytest.mark.parametrize("rate", [0, -1, -16_000])
    def test_a_non_positive_rate_is_refused(self, rate: int) -> None:
        with pytest.raises(AudioIntakeError, match="sample_rate"):
            load_audio(_mono(), rate)

    def test_a_rate_above_the_limit_is_refused(self) -> None:
        with pytest.raises(AudioTooLargeError, match="sample rate"):
            load_audio(_mono(), AudioConfig().max_sample_rate * 2)

    def test_a_duration_above_the_limit_is_refused(self) -> None:
        with pytest.raises(AudioTooLargeError, match="exceeding"):
            load_audio(_mono(), 1_000)

    def test_too_many_channels_are_refused(self) -> None:
        wide = np.zeros((1_000, 9), dtype=np.float32)

        with pytest.raises(AudioTooLargeError, match="channels"):
            load_audio(wide, _SR)

    def test_a_non_numeric_dtype_is_refused(self) -> None:
        with pytest.raises(AudioIntakeError, match="numeric dtype"):
            load_audio(np.array([{"a": 1}], dtype=object), _SR)

    def test_complex_audio_is_refused(self) -> None:
        with pytest.raises(AudioIntakeError, match="complex"):
            load_audio(np.ones(1_000, dtype=np.complex64), _SR)

    def test_a_higher_dimensional_array_is_refused(self) -> None:
        with pytest.raises(AudioIntakeError, match="1-D or 2-D"):
            load_audio(np.zeros((4, 4, 4), dtype=np.float32), _SR)

    def test_an_empty_array_is_refused(self) -> None:
        with pytest.raises(AudioDecodeError):
            load_audio(np.zeros(0, dtype=np.float32), _SR)

    def test_a_silent_array_is_refused(self) -> None:
        with pytest.raises(InvalidAudioSignalError):
            load_audio(np.zeros(_SR, dtype=np.float32), _SR)

    def test_non_finite_samples_are_refused(self) -> None:
        damaged = _mono()
        damaged[0] = np.nan

        with pytest.raises(InvalidAudioSignalError, match="non-finite"):
            load_audio(damaged, _SR)

    def test_array_limits_match_the_encoded_limits(self) -> None:
        """The same budget must be enforced on both paths, or the trust boundary moves."""
        config = AudioConfig(max_duration_seconds=1.0)

        with pytest.raises(AudioTooLargeError):
            load_audio(_mono(2.0), _SR, config)
        with pytest.raises(AudioTooLargeError):
            load_audio(_wav_bytes(2.0), config=config)


class TestProvenance:
    def test_source_kind_distinguishes_encoded_from_array(self) -> None:
        assert load_audio(_wav_bytes()).source_kind == "encoded"
        assert load_audio(_mono(), _SR).source_kind == "array"

    def test_metadata_stays_sample_free_for_both_paths(self) -> None:
        for loaded in (load_audio(_wav_bytes()), load_audio(_mono(), _SR)):
            metadata = summarise_metadata(loaded)
            assert "source_kind" in metadata
            assert not any(isinstance(v, (np.ndarray, list, bytes)) for v in metadata.values())


class TestUnsupportedSources:
    @pytest.mark.parametrize("source", [None, 42, 3.5, {"a": 1}, ["wav"]])
    def test_an_unsupported_type_is_refused(self, source: object) -> None:
        with pytest.raises(AudioIntakeError):
            load_audio(source)  # type: ignore[arg-type]

    def test_the_same_input_gives_the_same_output(self) -> None:
        first = load_audio(_wav_bytes())
        second = load_audio(_wav_bytes())

        assert np.array_equal(first.samples, second.samples)
        assert summarise_metadata(first) == summarise_metadata(second)
