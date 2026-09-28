"""Unit tests for the audio intake boundary.

The threat this file tests: a caller must not be able to crash the service,
exhaust memory, or reach an unhandled exception by crafting a file. Every case
here is a hostile input, and every case must produce a typed error.

A second, subtler threat is a *dishonest* file: one whose header claims more
audio than it carries. The header is what the size limits are checked against --
that is the point, since the check must happen before allocation -- so the tests
below also pin that the metadata VoxShield ends up auditing always describes what
was actually decoded, never what the header claimed.
"""

from __future__ import annotations

import struct
from collections.abc import Callable

import numpy as np
import pytest

from voxshield.audio.decode import decode_audio_bytes, summarise_metadata
from voxshield.config import AudioConfig
from voxshield.errors import (
    AudioDecodeError,
    AudioTooLargeError,
    InvalidAudioSignalError,
    UnsupportedAudioFormatError,
)

_TYPED_DECODE_ERRORS = (
    AudioDecodeError,
    AudioTooLargeError,
    UnsupportedAudioFormatError,
    InvalidAudioSignalError,
)

_SR = 16_000


def _riff_header(
    *,
    audio_format: int = 1,
    channels: int = 1,
    sample_rate: int = _SR,
    bits_per_sample: int = 16,
    declared_data_bytes: int | None = None,
) -> bytes:
    """Build a WAV header with independently controllable fields.

    Useful for crafting a header that lies about its contents, which is the
    interesting attack: a small file that claims a huge frame count.
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_bytes = (
        declared_data_bytes
        if declared_data_bytes is not None
        else _SR * block_align
    )
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_bytes)
        + b"WAVEfmt "
        + struct.pack("<I", 16)
        + struct.pack(
            "<HHIIHH",
            audio_format,
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        )
        + b"data"
        + struct.pack("<I", data_bytes)
    )


class TestHappyPath:
    def test_decodes_16k_pcm16(self, wav: Callable, samples: Callable) -> None:
        decoded = decode_audio_bytes(wav(samples(1.0)), AudioConfig())
        assert decoded.sample_rate == _SR
        assert decoded.channels == 1
        assert decoded.samples.dtype == np.float32
        assert decoded.samples.size == _SR

    def test_summary_is_metadata_only(self, wav: Callable, samples: Callable) -> None:
        decoded = decode_audio_bytes(wav(samples(1.0)), AudioConfig())
        summary = summarise_metadata(decoded)
        assert set(summary) >= {"source_sample_rate_hz", "frames", "duration_seconds"}
        for value in summary.values():
            assert not isinstance(value, (bytes, bytearray, np.ndarray))

    def test_accepts_float_subtype(self, wav: Callable, samples: Callable) -> None:
        payload = wav(samples(1.0), subtype="FLOAT")
        assert decode_audio_bytes(payload, AudioConfig()).samples.size == _SR

    def test_reports_actual_decoded_length(self, wav: Callable, samples: Callable) -> None:
        # frames/duration_seconds describe the decode, not the header, and are
        # mutually consistent. This is what the audit trail records.
        decoded = decode_audio_bytes(wav(samples(1.0)), AudioConfig())
        assert decoded.frames == decoded.samples.size
        assert decoded.duration_seconds == pytest.approx(
            decoded.frames / decoded.sample_rate
        )


class TestSizeLimits:
    def test_rejects_oversized_upload(self, wav: Callable, samples: Callable) -> None:
        cfg = AudioConfig().with_overrides(max_upload_bytes=1024)
        payload = wav(samples(2.0))
        assert len(payload) > 1024
        with pytest.raises(AudioTooLargeError):
            decode_audio_bytes(payload, cfg)

    def test_rejects_header_declaring_more_frames_than_allowed(self) -> None:
        # A tiny file whose header claims an hour of 8-channel 192 kHz audio.
        # Must be refused on the header, before any allocation proportional to
        # the claim.
        header = _riff_header(
            sample_rate=192_000,
            channels=8,
            bits_per_sample=32,
            declared_data_bytes=192_000 * 8 * 4 * 60,
        )
        assert len(header) < 200
        with pytest.raises(_TYPED_DECODE_ERRORS):
            decode_audio_bytes(header, AudioConfig())

    def test_rejects_too_many_channels(self, wav: Callable) -> None:
        cfg = AudioConfig().with_overrides(max_channels=1)
        stereo = np.zeros((_SR, 2), dtype=np.float32)
        stereo[:, 0] = 0.3
        with pytest.raises(AudioTooLargeError):
            decode_audio_bytes(wav(stereo), cfg)

    def test_rejects_overlong_duration(self, wav: Callable, samples: Callable) -> None:
        cfg = AudioConfig().with_overrides(max_duration_seconds=0.5)
        with pytest.raises(AudioTooLargeError):
            decode_audio_bytes(wav(samples(3.0)), cfg)

    def test_rejects_oversized_sample_rate(self, wav: Callable, samples: Callable) -> None:
        # The ceiling has to stay at or above the resample target, so a 24 kHz
        # ceiling is the tightest one that still leaves 48 kHz input over-limit.
        cfg = AudioConfig().with_overrides(max_sample_rate=24_000)
        with pytest.raises(AudioTooLargeError):
            decode_audio_bytes(wav(samples(1.0), sample_rate=48_000), cfg)
        # ...and it still accepts what is under the ceiling.
        assert decode_audio_bytes(wav(samples(1.0), sample_rate=22_050), cfg).samples.size > 0

    def test_sample_rate_ceiling_cannot_fall_below_target(self) -> None:
        # Otherwise a misconfiguration would make every file fail, or worse,
        # silently skip the check.
        with pytest.raises(ValueError, match="max_sample_rate"):
            AudioConfig().with_overrides(max_sample_rate=8_000)


class TestFormatAllowList:
    @pytest.mark.parametrize("fmt", ["AIFF", "OGG", "AU"])
    def test_rejects_unsupported_container(self, wav: Callable, samples: Callable, fmt: str) -> None:
        payload = wav(samples(0.5), fmt=fmt)
        with pytest.raises(UnsupportedAudioFormatError):
            decode_audio_bytes(payload, AudioConfig())

    def test_rejects_disallowed_subtype(self, wav: Callable, samples: Callable) -> None:
        cfg = AudioConfig().with_overrides(allowed_subtypes=frozenset({"PCM_16"}))
        with pytest.raises(UnsupportedAudioFormatError):
            decode_audio_bytes(wav(samples(0.5), subtype="FLOAT"), cfg)

    def test_allow_list_fails_closed(self, wav: Callable, samples: Callable) -> None:
        # An empty allow-list must reject everything rather than allow all.
        cfg = AudioConfig().with_overrides(allowed_subtypes=frozenset())
        with pytest.raises(UnsupportedAudioFormatError):
            decode_audio_bytes(wav(samples(0.5)), cfg)

    def test_wav_only_when_flac_excluded(self, wav: Callable, samples: Callable) -> None:
        # The MVP decision is WAV-only. If FLAC is removed from the allow-list,
        # FLAC must fail closed rather than being silently accepted.
        cfg = AudioConfig().with_overrides(allowed_formats=frozenset({"WAV"}))
        assert decode_audio_bytes(wav(samples(0.5)), cfg).samples.size > 0
        with pytest.raises(UnsupportedAudioFormatError):
            decode_audio_bytes(wav(samples(0.5), fmt="FLAC"), cfg)

    def test_allowed_container_round_trips(self, wav: Callable, samples: Callable) -> None:
        # Guard against the allow-list drifting out of sync with libsndfile.
        for fmt in sorted(AudioConfig().allowed_formats):
            try:
                payload = wav(samples(0.5), fmt=fmt)
            except Exception:
                pytest.skip(f"{fmt} is not writable in this libsndfile build")
            assert decode_audio_bytes(payload, AudioConfig()).samples.size > 0


class TestMalformedInput:
    @pytest.mark.parametrize(
        "payload",
        [
            b"",
            b"not audio at all",
            b"RIFF",
            b"RIFF\x00\x00\x00\x00WAVE",
            b"\x00" * 512,
            b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00",
        ],
        ids=["empty", "garbage", "riff-magic-only", "riff-zero-size", "nulls", "short-fmt-chunk"],
    )
    def test_malformed_raises_typed_error(self, payload: bytes) -> None:
        with pytest.raises(_TYPED_DECODE_ERRORS):
            decode_audio_bytes(payload, AudioConfig())

    def test_bytearray_accepted(self, wav: Callable, samples: Callable) -> None:
        assert decode_audio_bytes(bytearray(wav(samples(0.5))), AudioConfig()).samples.size > 0

    @pytest.mark.parametrize("keep_fraction", [0.25, 0.5, 0.9])
    def test_truncation_never_reports_the_declared_length(
        self, wav: Callable, samples: Callable, keep_fraction: float
    ) -> None:
        # A truncated file may decode to whatever survived, or be refused. What it
        # must never do is claim the full declared duration, because that value
        # is what gets written to the audit trail.
        full = wav(samples(1.0))
        try:
            decoded = decode_audio_bytes(full[: int(len(full) * keep_fraction)], AudioConfig())
        except _TYPED_DECODE_ERRORS:
            return
        assert decoded.frames <= _SR
        assert decoded.duration_seconds == pytest.approx(
            decoded.frames / decoded.sample_rate
        )

    def test_dishonest_header_cannot_inflate_reported_duration(
        self, wav: Callable, samples: Callable
    ) -> None:
        # Header claims a full second, body carries a few dozen samples.
        honest = wav(samples(1.0))
        lying = _riff_header(declared_data_bytes=2 * _SR * 2) + honest[44:44 + 96]
        assert len(lying) < 200
        try:
            decoded = decode_audio_bytes(lying, AudioConfig())
        except _TYPED_DECODE_ERRORS:
            return
        assert decoded.frames < _SR
        assert decoded.duration_seconds == pytest.approx(
            decoded.frames / decoded.sample_rate
        )


class TestDegenerateSignals:
    def test_rejects_digital_silence(self, wav: Callable) -> None:
        with pytest.raises(InvalidAudioSignalError, match="silent"):
            decode_audio_bytes(wav(np.zeros(_SR, dtype=np.float32)), AudioConfig())

    def test_rejects_inaudible_signal(self, wav: Callable) -> None:
        # Non-zero but below the usable floor. PCM_16 quantises 1e-12 to zero, so
        # the floor check would never see it -- a float container is required to
        # reach the branch under test.
        quiet = np.full(_SR, 1e-12, dtype=np.float32)
        with pytest.raises(InvalidAudioSignalError, match="usable floor"):
            decode_audio_bytes(wav(quiet, subtype="FLOAT"), AudioConfig())
        # The same signal in PCM_16 is indistinguishable from silence, and the
        # silent branch is the correct answer for it.
        with pytest.raises(InvalidAudioSignalError, match="silent"):
            decode_audio_bytes(wav(quiet, subtype="PCM_16"), AudioConfig())

    @pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite(self, wav: Callable, samples: Callable, bad: float) -> None:
        corrupted = np.full(_SR, bad, dtype=np.float32)
        with pytest.raises(InvalidAudioSignalError):
            decode_audio_bytes(wav(corrupted, subtype="FLOAT"), AudioConfig())
        assert samples  # fixture kept in use; generator is shared
