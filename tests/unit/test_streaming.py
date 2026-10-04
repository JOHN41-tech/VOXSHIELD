"""Rolling-buffer streaming: window accounting, padding, and memory bounds.

The contract these tests defend is that a streamed window and a batch window
from the same audio are the same thing. If that stops being true, a caller gets
one verdict from the live path and a different one from the batch path, which
is the failure mode that makes a streaming result untrustworthy.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from voxshield.audio.process import process_audio
from voxshield.audio.streaming import StreamingProcessor
from voxshield.config import AudioConfig, NormalizationConfig

_SR = 16_000

#: A stream cannot measure a whole-clip RMS, so parity is asserted against a
#: batch run that also applies no gain. The gain path is covered separately.
_FLAT = AudioConfig(normalization=NormalizationConfig(enabled=False))


def _tone(seconds: float, freq: float = 200.0, amplitude: float = 0.25) -> np.ndarray:
    n = int(_SR * seconds)
    t = np.arange(n) / _SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _run(stream: StreamingProcessor, audio: np.ndarray, size: int) -> list:
    """Push ``audio`` in ``size``-sample chunks and flush."""
    out: list = []
    for start in range(0, len(audio), size):
        out += list(stream.push(audio[start : start + size]))
    out += list(stream.flush())
    return out


class TestWindowGeometry:
    def test_a_full_window_is_emitted_as_soon_as_it_completes(self) -> None:
        stream = StreamingProcessor(_SR)

        assert list(stream.push(_tone(4.0))) != []
        assert list(stream.push(_tone(0.5))) == []

    def test_a_short_stream_emits_nothing_until_flush(self) -> None:
        stream = StreamingProcessor(_SR)

        assert list(stream.push(_tone(1.0))) == []
        assert len(list(stream.flush())) == 1

    def test_windows_advance_by_the_configured_hop(self) -> None:
        stream = StreamingProcessor(_SR)

        windows = _run(stream, _tone(12.0), size=8_000)

        # Default 4s window with 50% overlap means a 2s hop.
        assert [round(w.start_seconds, 2) for w in windows] == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]

    def test_every_window_reports_the_nominal_length(self) -> None:
        stream = StreamingProcessor(_SR)

        windows = _run(stream, _tone(9.0), size=4_000)

        assert all(w.duration_seconds == pytest.approx(4.0) for w in windows)

    def test_a_hop_override_changes_the_advance(self) -> None:
        stream = StreamingProcessor(_SR, AudioConfig(segment_hop_seconds=3.0))

        windows = _run(stream, _tone(20.0), size=_SR)

        assert [round(w.start_seconds, 2) for w in windows] == [
            0.0,
            3.0,
            6.0,
            9.0,
            12.0,
            15.0,
            18.0,
        ]


class TestFinalWindow:
    def test_a_padded_final_window_is_flagged(self) -> None:
        stream = StreamingProcessor(_SR)

        windows = list(stream.push(_tone(1.5))) + list(stream.flush())

        assert len(windows) == 1
        assert windows[0].is_final is True
        assert windows[0].features.shape == (401, 80)

    def test_flush_emits_at_most_one_window(self) -> None:
        """``push`` drains every complete window, so only the tail remains."""
        stream = StreamingProcessor(_SR, _FLAT)

        windows = list(stream.push(_tone(9.0))) + list(stream.flush())

        final = [w for w in windows if w.is_final]
        assert len(final) == 1
        # Three complete windows at 0/2/4s, then one padded tail at 6s. A
        # hop-strided tail would have added a 5s-of-nothing window at 8s.
        assert len(windows) == 4
        assert final[0].start_seconds == pytest.approx(6.0)

    def test_flush_on_an_empty_stream_emits_nothing(self) -> None:
        stream = StreamingProcessor(_SR)

        assert list(stream.flush()) == []

    def test_a_non_multiple_stream_always_ends_with_a_tail(self) -> None:
        """With hop < window, the remainder lands mid-window and must be flagged.

        This is why ``is_final`` has to be honoured: for the default 4s window
        and 2s hop, almost every stream ends with a half-silent window.
        """
        stream = StreamingProcessor(_SR)

        windows = _run(stream, _tone(10.0), size=_SR)

        assert windows[-1].is_final is True

    def test_final_and_complete_windows_have_the_same_shape(self) -> None:
        stream = StreamingProcessor(_SR)

        windows = _run(stream, _tone(6.5), size=1_000)

        shapes = {w.features.shape for w in windows}
        assert len(shapes) == 1
        assert any(w.is_final for w in windows)


class TestChunking:
    def test_ragged_chunk_sizes_do_not_change_the_output(self) -> None:
        audio = _tone(10.0)

        reference = _run(StreamingProcessor(_SR), audio, size=_SR)
        ragged = _run(StreamingProcessor(_SR), audio, size=3_133)

        assert [w.index for w in ragged] == [w.index for w in reference]
        assert [w.start_seconds for w in ragged] == [w.start_seconds for w in reference]
        for a, b in zip(reference, ragged, strict=True):
            np.testing.assert_allclose(a.features, b.features, atol=1e-4)

    def test_an_empty_chunk_is_ignored(self) -> None:
        stream = StreamingProcessor(_SR)

        assert list(stream.push(np.empty(0, dtype=np.float32))) == []
        assert stream.received_seconds == 0.0

    def test_a_stereo_chunk_is_downmixed(self) -> None:
        stream = StreamingProcessor(_SR)
        stereo = np.column_stack([_tone(4.0), _tone(4.0, freq=300.0)])

        windows = list(stream.push(stereo))

        assert windows
        assert windows[0].features.shape == (401, 80)

    def test_non_finite_values_are_sanitized_not_propagated(self) -> None:
        stream = StreamingProcessor(_SR)
        chunk = _tone(4.0)
        chunk[0] = np.nan
        chunk[1] = np.inf

        windows = list(stream.push(chunk))

        assert np.isfinite(windows[0].features).all()

    def test_a_chunk_rate_other_than_canonical_is_resampled(self) -> None:
        stream = StreamingProcessor(44_100)
        n = 44_100 * 4
        t = np.arange(n) / 44_100
        chunk = (0.1 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)

        windows = list(stream.push(chunk))

        assert stream.sample_rate == _SR
        assert windows and windows[0].features.shape == (401, 80)

    def test_a_bad_sample_rate_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="sample_rate"):
            StreamingProcessor(0)

    def test_a_three_dimensional_chunk_is_rejected(self) -> None:
        stream = StreamingProcessor(_SR)

        with pytest.raises(ValueError, match="1-D or 2-D"):
            list(stream.push(np.zeros((2, 2, 2), dtype=np.float32)))


class TestMemoryAndLifecycle:
    def test_the_buffer_stays_bounded_across_a_long_stream(self) -> None:
        stream = StreamingProcessor(_SR)
        peak = 0.0

        for _ in range(20):
            list(stream.push(_tone(1.0)))  # one second per push
            peak = max(peak, stream.buffered_seconds)

        # A rolling window keeps a couple of windows, not the whole stream.
        assert peak <= stream.window_samples / _SR + 1.0
        assert stream.received_seconds == pytest.approx(20.0)

    def test_drop_audio_keeps_the_stream_position(self) -> None:
        stream = StreamingProcessor(_SR)
        list(stream.push(_tone(4.0)))
        before = (stream.received_seconds, stream.n_emitted)

        stream.drop_audio()

        assert stream.buffered_seconds == 0.0
        assert (stream.received_seconds, stream.n_emitted) == before

    def test_reset_returns_the_stream_to_zero(self) -> None:
        stream = StreamingProcessor(_SR)
        _run(stream, _tone(6.0), size=_SR)

        stream.reset()

        assert stream.n_emitted == 0
        assert stream.received_seconds == 0.0
        assert stream.buffered_seconds == 0.0
        first = list(stream.push(_tone(4.0)))
        assert first[0].index == 0
        assert first[0].start_seconds == 0.0

    def test_a_second_stream_after_flush_starts_fresh(self) -> None:
        stream = StreamingProcessor(_SR)
        list(stream.push(_tone(4.0)))
        list(stream.flush())

        windows = list(stream.push(_tone(4.0)))

        # Window 0 from the first push, window 1 the flushed tail, window 2 here.
        assert windows[0].index == 2
        assert windows[0].start_seconds == pytest.approx(4.0)


def _batch_features(audio: np.ndarray, config: AudioConfig | None = None) -> np.ndarray:
    """Features from the offline path, for a true parity comparison."""
    from voxshield.audio.pipeline import prepare

    prepared = prepare(b"", config or _FLAT, decoded=_as_decoded(audio))
    return np.stack([prepared.segment_features(i) for i in range(len(prepared.segments))])


def _as_decoded(audio: np.ndarray):
    from voxshield.audio.decode import DecodedAudio
    from voxshield.audio.preprocess import sanitize, to_mono

    mono = to_mono(sanitize(audio))
    return DecodedAudio(
        samples=mono,
        sample_rate=_SR,
        channels=1,
        frames=int(mono.size),
        duration_seconds=mono.size / float(_SR),
        source_format="RAW",
        source_subtype="PCM_FLOAT32",
        source_kind="array",
    )


class TestEquivalenceWithBatch:
    def test_streamed_features_match_the_offline_features(self) -> None:
        """The shared prefix must be identical, or the paths disagree."""
        audio = _tone(12.0)

        windows = _run(StreamingProcessor(_SR, _FLAT), audio, size=9_973)
        scored = [w for w in windows if not w.is_final]

        expected = _batch_features(audio)
        assert len(scored) == len(expected)
        np.testing.assert_allclose(np.stack([w.features for w in scored]), expected, atol=1e-6)

    def test_streamed_shapes_match_the_offline_shapes(self) -> None:
        audio = _tone(12.0)
        expected = _batch_features(audio)

        windows = _run(StreamingProcessor(_SR, _FLAT), audio, size=_SR)

        assert all(w.features.shape == expected.shape[1:] for w in windows)

    def test_an_explicit_gain_reproduces_a_normalized_batch_run(self) -> None:
        """The gain is the only thing a stream cannot measure for itself."""
        audio = _tone(12.0)
        gain = -7.5
        scaled = audio * (10.0 ** (gain / 20.0))

        # The flat batch path applies no gain, so it features the scaled audio
        # directly; the stream reaches the same samples via ``gain_db``.
        expected = _batch_features(scaled)
        windows = _run(StreamingProcessor(_SR, _FLAT, gain_db=gain), audio, size=_SR)
        scored = [w for w in windows if not w.is_final]

        np.testing.assert_allclose(np.stack([w.features for w in scored]), expected, atol=1e-6)

    def test_a_default_stream_does_not_match_a_normalizing_batch_run(self) -> None:
        """The documented limitation: no global gain, so levels are not equalized.

        Log-mel is a log scale, so this is not a statement about dynamics. It
        says a quiet clip is scored at its own level by a stream, whereas the
        batch path would have lifted it toward the target RMS first.
        """
        quiet = _tone(12.0, amplitude=0.05)

        expected = _batch_features(quiet, AudioConfig())
        windows = _run(StreamingProcessor(_SR, AudioConfig()), quiet, size=_SR)
        scored = [w for w in windows if not w.is_final]

        assert not np.allclose(np.stack([w.features for w in scored]), expected, atol=1e-2)

    def test_the_streaming_default_config_agrees_with_itself(self) -> None:
        """Sanity: with no gain and no normalization the two paths must match."""
        audio = _tone(12.0, amplitude=0.25)

        expected = _batch_features(audio, _FLAT)
        windows = _run(StreamingProcessor(_SR, _FLAT), audio, size=_SR)
        scored = [w for w in windows if not w.is_final]

        np.testing.assert_allclose(np.stack([w.features for w in scored]), expected, atol=1e-6)

    def test_the_batch_result_exposes_no_feature_matrix(self) -> None:
        """The result object is metadata-only, so it cannot be logged or shipped."""
        audio = _tone(6.0)
        buffer = io.BytesIO()
        sf.write(buffer, audio, _SR, format="WAV", subtype="FLOAT")

        batch = process_audio(buffer.getvalue())

        assert not hasattr(batch, "features")
        assert batch.feature_shape is not None
        assert batch.n_features > 0

    def test_the_only_extra_window_is_the_flagged_tail(self) -> None:
        """Batch anchors its last window; a stream must pad instead."""
        audio = _tone(12.0)

        windows = _run(StreamingProcessor(_SR, _FLAT), audio, size=_SR)
        expected = _batch_features(audio)

        assert len(windows) == len(expected) + 1
        assert [w.is_final for w in windows] == [False] * len(expected) + [True]

    def test_a_stream_ending_on_a_window_boundary_adds_no_tail(self) -> None:
        """With hop == window, a multiple-of-window stream leaves no remainder."""
        config = AudioConfig(segment_hop_seconds=4.0)
        stream = StreamingProcessor(_SR, config)

        windows = _run(stream, _tone(8.0), size=_SR)

        assert len(windows) == 2
        assert not any(w.is_final for w in windows)


class TestWindowContract:
    def test_windows_are_frozen(self) -> None:
        stream = StreamingProcessor(_SR)

        window = next(iter(stream.push(_tone(4.0))))

        with pytest.raises(Exception, match=r"cannot assign|frozen|immutable"):
            window.index = 99  # type: ignore[misc]

    def test_windows_carry_only_scalar_metadata_and_features(self) -> None:
        stream = StreamingProcessor(_SR)

        window = next(iter(stream.push(_tone(4.0))))

        assert isinstance(window.index, int)
        assert isinstance(window.start_seconds, float)
        assert isinstance(window.end_seconds, float)
        assert isinstance(window.is_final, bool)
        assert isinstance(window.features, np.ndarray)
        assert not hasattr(window, "samples")
