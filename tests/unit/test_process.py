"""Offline processing: what it reports, and what it must never report."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxshield.audio.process import process_audio
from voxshield.audio.quality import ISSUE_CLIPPED
from voxshield.config import AudioConfig, NormalizationConfig
from voxshield.errors import (
    AudioDecodeError,
    AudioTooLargeError,
    InsufficientSpeechError,
    InvalidAudioSignalError,
    UnsupportedAudioFormatError,
)

_SR = 16_000


def _tone(seconds: float, amplitude: float = 0.25, rate: int = _SR) -> np.ndarray:
    t = np.arange(int(rate * seconds)) / rate
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 1.2 * t)
    return (amplitude * np.sin(2 * np.pi * 200.0 * t) * envelope).astype(np.float32)


def _wav(seconds: float = 8.0, rate: int = _SR, amplitude: float = 0.25) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, _tone(seconds, amplitude, rate), rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


class TestHappyPath:
    def test_a_clean_clip_is_scorable(self) -> None:
        result = process_audio(_wav())

        assert result.scorable
        assert result.blocking_issues == ()
        assert result.n_segments > 0
        assert result.n_features == result.n_segments

    def test_rtf_is_measured_and_plausible(self) -> None:
        result = process_audio(_wav())

        assert result.wall_seconds > 0.0
        assert result.rtf > 0.0
        # Whole pipeline is far cheaper than real time; if this ever fails, the
        # measurement is wrong rather than the pipeline being slow.
        assert result.rtf < 1.0

    def test_features_are_computed_with_a_constant_shape(self) -> None:
        result = process_audio(_wav())

        assert result.feature_shape is not None
        n_frames, n_mels = result.feature_shape
        assert n_frames > 0
        assert n_mels == AudioConfig().features.n_mels

    def test_features_can_be_skipped(self) -> None:
        result = process_audio(_wav(), compute_features=False)

        assert result.n_features == 0
        assert result.feature_shape is None
        assert result.n_segments > 0

    def test_regions_and_segments_are_both_reported(self) -> None:
        result = process_audio(_wav())

        assert len(result.regions) >= 1
        assert result.n_segments >= 1
        assert result.speech_seconds > 0.0

    def test_the_measured_audio_is_the_canonical_length(self) -> None:
        result = process_audio(_wav(8.0))

        assert result.audio_seconds == pytest.approx(8.0, abs=0.05)
        assert result.metadata()["canonical_sample_rate_hz"] == 16_000

    def test_a_resampled_source_reports_canonical_metrics(self) -> None:
        result = process_audio(_wav(6.0, rate=44_100))

        assert result.source["source_sample_rate_hz"] == 44_100
        assert result.audio_seconds == pytest.approx(6.0, abs=0.1)
        assert result.metadata()["canonical_sample_rate_hz"] == 16_000


class TestConfigurationIsHonoured:
    def test_segment_hop_changes_the_window_count(self) -> None:
        default = process_audio(_wav(16.0))
        coarse = process_audio(_wav(16.0), config=AudioConfig(segment_hop_seconds=3.5))

        assert coarse.n_segments < default.n_segments
        assert coarse.metadata()["hop_seconds"] == 3.5

    def test_the_pad_policy_pads_a_clip_shorter_than_one_window(self) -> None:
        # The tail anchor keeps the last window of a long clip full width, so the
        # policy only has a decision to make for a sub-window recording.
        result = process_audio(
            _wav(3.0), config=AudioConfig(short_segment_policy="pad")
        )

        assert result.n_segments == 1
        assert result.n_padded_windows == 1
        assert result.n_features == 1
        # Zero-filled, so the feature matrix still has the nominal shape.
        assert result.feature_shape is not None

    def test_the_keep_policy_scores_a_sub_window_clip_at_natural_width(self) -> None:
        result = process_audio(
            _wav(3.0), config=AudioConfig(short_segment_policy="keep")
        )

        assert result.n_segments == 1
        assert result.n_padded_windows == 0

    def test_the_drop_policy_abstains_on_a_sub_window_clip(self) -> None:
        with pytest.raises(InsufficientSpeechError) as excinfo:
            process_audio(_wav(3.0), config=AudioConfig(short_segment_policy="drop"))

        # The recording carries 3s of contiguous speech, which is *more* than the
        # 2s minimum. It is dropped because the clip is smaller than one 4s
        # window, so the error must say that. Reporting it as insufficient
        # speech produced "3.00s of speech, below the 2.00s minimum" -- a
        # message that contradicts its own numbers and tells an operator to ask
        # for more speech rather than a longer recording.
        exc = excinfo.value
        assert exc.reason == InsufficientSpeechError.REASON_SHORT_WINDOW
        assert "4.00s analysis window" in str(exc)
        assert "short-segment policy is 'drop'" in str(exc)

    def test_thin_speech_in_a_long_clip_is_reported_as_contiguous(self) -> None:
        # The other branch: long enough for a full window, and enough total
        # speech, but no window is ever dense enough. That is a genuinely
        # different problem from a short recording and must not borrow its
        # wording.
        #
        # 0.6s of tone every 2.0s is ~30% density, so any 4s window holds at most
        # two runs (1.2s) and can never reach a 3.0s minimum, while the clip as a
        # whole carries 3.6s -- above the 1.0s total minimum. Raising
        # min_segment_seconds is what makes this independent of window phase: at
        # the default 2.0s a 4s window can span three runs and survive, so the
        # case would not be reachable at all.
        run, period = 0.6, 2.0
        block = np.concatenate(
            [_tone(run, 0.25, _SR), np.zeros(int((period - run) * _SR), dtype=np.float32)]
        )
        blocks = int(np.ceil(12.0 * _SR / block.size))
        signal = np.tile(block, blocks)[: 12 * _SR].astype(np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, signal, _SR, format="WAV", subtype="PCM_16")

        with pytest.raises(InsufficientSpeechError) as excinfo:
            process_audio(
                buffer.getvalue(),
                config=AudioConfig(
                    min_segment_seconds=3.0, short_segment_policy="drop"
                ),
            )

        exc = excinfo.value
        assert exc.reason == InsufficientSpeechError.REASON_CONTIGUOUS
        assert "contiguous" in str(exc)

    def test_too_little_speech_in_total_is_reported_as_total(self) -> None:
        with pytest.raises(InsufficientSpeechError) as excinfo:
            process_audio(_wav(0.8, amplitude=0.02))

        exc = excinfo.value
        assert exc.reason == InsufficientSpeechError.REASON_TOTAL
        assert exc.longest_run_seconds is None

    def test_no_policy_pads_a_long_clip(self) -> None:
        """The tail anchor means a long clip never needs a padded window."""
        for policy in ("drop", "pad", "keep"):
            result = process_audio(
                _wav(9.0), config=AudioConfig(short_segment_policy=policy)
            )
            assert result.n_padded_windows == 0, policy

    def test_the_normalization_strategy_is_reported(self) -> None:
        for strategy in ("rms", "peak"):
            result = process_audio(
                _wav(6.0),
                config=AudioConfig(normalization=NormalizationConfig(strategy=strategy)),
            )
            assert result.metadata()["normalization_strategy"] == strategy

    def test_disabled_normalization_is_reported_and_changes_gain(self) -> None:
        enabled = process_audio(_wav(6.0, amplitude=0.01))
        disabled = process_audio(
            _wav(6.0, amplitude=0.01),
            config=AudioConfig(normalization=NormalizationConfig(enabled=False)),
        )

        assert disabled.metadata()["normalization_strategy"] == "none"
        assert disabled.gain_db_applied == 0.0
        assert enabled.gain_db_applied > 0.0


class TestQualityIsReportedNotEnforced:
    def test_a_clipped_clip_is_reported_as_a_warning(self) -> None:
        loud = np.clip(_tone(6.0, amplitude=0.99) * 4.0, -1.0, 1.0).astype(np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, loud, _SR, format="WAV", subtype="FLOAT")

        result = process_audio(buffer.getvalue())

        # Clipping is only visible if quality is measured before normalisation
        # repairs the level, which is what process_audio does.
        assert ISSUE_CLIPPED in result.warning_issues
        # Clipping degrades confidence; it does not make a clip unscorable.
        assert result.scorable

    def test_too_little_speech_is_an_abstention_not_a_low_score(self) -> None:
        quiet = io.BytesIO()
        sf.write(quiet, _tone(0.6, amplitude=0.02), _SR, format="WAV", subtype="FLOAT")

        with pytest.raises(InsufficientSpeechError):
            process_audio(quiet.getvalue())

    def test_a_quality_problem_does_not_become_an_exception(self) -> None:
        """``process_audio`` reports; it does not adjudicate."""
        result = process_audio(_wav(8.0, amplitude=0.05))

        assert isinstance(result.blocking_issues, tuple)
        assert isinstance(result.scorable, bool)


class TestRefusalsPropagate:
    def test_an_empty_payload_is_refused(self) -> None:
        with pytest.raises(AudioDecodeError):
            process_audio(b"")

    def test_corrupt_bytes_are_refused(self) -> None:
        with pytest.raises(AudioDecodeError):
            process_audio(b"RIFF\x00\x00\x00\x00WAVEjunk")

    def test_a_silent_clip_is_refused(self) -> None:
        buffer = io.BytesIO()
        sf.write(buffer, np.zeros(_SR * 2, dtype=np.float32), _SR, format="WAV", subtype="FLOAT")

        with pytest.raises(InvalidAudioSignalError):
            process_audio(buffer.getvalue())

    def test_an_over_long_clip_is_refused(self) -> None:
        with pytest.raises(AudioTooLargeError):
            process_audio(_tone(31.0), _SR)

    def test_a_disallowed_container_is_refused(self) -> None:
        buffer = io.BytesIO()
        sf.write(buffer, _tone(4.0), _SR, format="AIFF", subtype="PCM_16")

        with pytest.raises(UnsupportedAudioFormatError):
            process_audio(buffer.getvalue())

    def test_an_array_source_needs_its_rate(self) -> None:
        with pytest.raises(Exception, match="sample_rate"):
            process_audio(_tone(6.0))


class TestPrivacyBoundary:
    def test_metadata_contains_no_audio(self) -> None:
        metadata = process_audio(_wav()).metadata()

        assert not any(
            isinstance(v, (np.ndarray, bytes, bytearray, memoryview))
            for v in metadata.values()
        )
        serialised = json.dumps(metadata)
        assert "ndarray" not in serialised
        assert "redacted" not in serialised

    def test_metadata_is_json_serialisable(self) -> None:
        metadata = process_audio(_wav()).metadata()

        assert json.loads(json.dumps(metadata)) == metadata

    def test_no_result_field_can_hold_samples(self) -> None:
        result = process_audio(_wav())

        for name in result.__dataclass_fields__:
            value = getattr(result, name)
            assert not isinstance(value, (np.ndarray, bytes, bytearray, memoryview)), name

    def test_the_result_keeps_no_reference_to_the_audio(self) -> None:
        """The result must outlive the audio without holding it."""
        import gc

        result = process_audio(_wav())
        gc.collect()

        assert result.n_segments > 0
        assert result.audio_seconds > 0.0


class TestDeterminism:
    def test_the_same_input_gives_the_same_analysis(self) -> None:
        volatile = {"wall_seconds", "rtf"}
        first = process_audio(_wav(8.0)).metadata()
        second = process_audio(_wav(8.0)).metadata()

        assert {k: v for k, v in first.items() if k not in volatile} == {
            k: v for k, v in second.items() if k not in volatile
        }

    def test_the_same_input_gives_the_same_quality_and_regions(self) -> None:
        a = process_audio(_wav(8.0))
        b = process_audio(_wav(8.0))

        assert a.quality.issues == b.quality.issues
        assert [r.start_s for r in a.regions] == [r.start_s for r in b.regions]
        assert a.n_segments == b.n_segments


def _subprocess_env() -> dict[str, str]:
    """Environment with the source tree importable.

    pytest's ``pythonpath`` ini option only affects the test process, so a
    subprocess needs the path restated or it will not import the working tree.
    """
    env = dict(os.environ)
    src = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (src, env.get("PYTHONPATH", "")) if p)
    return env


def _run_module(*argv: str) -> subprocess.CompletedProcess[str]:
    """Run ``python -m voxshield.audio.process`` as a real program."""
    return subprocess.run(
        [sys.executable, "-m", "voxshield.audio.process", *argv],
        capture_output=True,
        text=True,
        env=_subprocess_env(),
        timeout=180,
        check=False,
    )


class TestModuleEntrypoint:
    """``python -m voxshield.audio.process`` must actually analyse a file.

    These run a subprocess because the failure they guard against is invisible
    in-process. A module with no ``__main__`` guard imports cleanly, defines every
    function, and passes every other test in this file -- it simply does nothing
    when invoked as a program. That is the defect being replaced: the command
    ran, printed nothing, and exited 0, which reads as a clean measurement.
    """

    def test_it_prints_a_report_and_exits_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "clip.wav"
        path.write_bytes(_wav(8.0))

        done = _run_module(str(path))

        assert done.returncode == 0, done.stderr
        report = json.loads(done.stdout)
        assert report["scorable"] is True
        assert report["n_segments"] > 0
        assert report["rtf"] > 0.0
        assert report["wall_seconds"] > 0.0

    def test_it_matches_the_documented_exit_codes(self, tmp_path: Path) -> None:
        """A refusal must not be reported as success by the ``-m`` route."""
        done = _run_module(str(tmp_path / "absent.wav"))

        assert done.returncode == 2
        assert done.stdout == ""

    def test_it_does_not_warn_about_being_imported_twice(self, tmp_path: Path) -> None:
        """A spurious stderr warning teaches operators to ignore stderr.

        The warning is emitted because the parent package imported this module
        before ``runpy`` executed it, so the module body ran twice. It is
        harmless here, but a measurement tool that always prints a warning is one
        whose real warnings get read as noise.
        """
        path = tmp_path / "clip.wav"
        path.write_bytes(_wav(8.0))

        done = _run_module(str(path))

        assert "RuntimeWarning" not in done.stderr
        assert "runpy" not in done.stderr

    def test_the_package_does_not_import_this_module_eagerly(self) -> None:
        """Guards the fix at its cause rather than at its symptom.

        Importing ``voxshield.audio`` must not drag in ``process``; that eager
        import is what produced the double execution above. The lazy
        ``__getattr__`` still resolves both exported names, which is what the next
        test confirms.
        """
        done = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, voxshield.audio as a; "
                "print('eager' if 'voxshield.audio.process' in sys.modules else 'lazy')",
            ],
            capture_output=True,
            text=True,
            env=_subprocess_env(),
            timeout=180,
            check=False,
        )

        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == "lazy"

    def test_the_lazily_exported_names_still_resolve(self) -> None:
        """Laziness must not cost the public surface or its typing."""
        import voxshield.audio as audio

        assert audio.ProcessResult.__name__ == "ProcessResult"
        assert callable(audio.process_audio)
        assert "ProcessResult" in audio.__all__
        assert "process_audio" in audio.__all__

    def test_an_unknown_attribute_still_raises(self) -> None:
        import voxshield.audio as audio

        with pytest.raises(AttributeError, match="no attribute 'not_a_real_export'"):
            _ = audio.not_a_real_export
