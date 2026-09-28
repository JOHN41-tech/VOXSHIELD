"""The ``process`` subcommand: exit codes, overrides, and what it prints.

The exit-code contract is the reason these tests exist. An operator scripting
this needs to distinguish "analysed and scorable", "analysed but not scorable",
and "refused before analysis" -- three claims that a shared success code would
flatten into one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxshield.cli import main

_SR = 16_000

#: Analysed and scorable.
EXIT_OK = 0
#: Refused: bad path, unreadable, undecodable, or not enough speech.
EXIT_REFUSED = 1
#: The path itself was unusable.
EXIT_BAD_PATH = 2
#: Analysed, but the quality report blocks scoring.
EXIT_UNSCORABLE = 3


def _write(tmp_path: Path, seconds: float, amplitude: float = 0.25) -> Path:
    t = np.arange(int(_SR * seconds)) / _SR
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 1.2 * t)
    tone = (amplitude * np.sin(2 * np.pi * 200.0 * t) * envelope).astype(np.float32)
    path = tmp_path / "clip.wav"
    with sf.SoundFile(path, "w", samplerate=_SR, channels=1, subtype="PCM_16") as fh:
        fh.write(tone)
    return path


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    code = main(list(argv))
    out = capsys.readouterr().out
    return code, json.loads(out)


class TestSuccessfulRun:
    def test_a_clean_clip_exits_zero_and_prints_json(self, tmp_path: Path, capsys) -> None:
        path = _write(tmp_path, 8.0)

        code, report = _run(capsys, "process", str(path))

        assert code == EXIT_OK
        assert report["scorable"] is True
        assert report["n_segments"] > 0
        assert report["canonical_sample_rate_hz"] == _SR

    def test_the_report_carries_a_measured_rtf(self, tmp_path: Path, capsys) -> None:
        path = _write(tmp_path, 8.0)

        _, report = _run(capsys, "process", str(path))

        assert report["rtf"] > 0.0
        assert report["wall_seconds"] > 0.0

    def test_compact_output_is_a_single_line(self, tmp_path: Path, capsys) -> None:
        path = _write(tmp_path, 6.0)

        code = main(["process", str(path), "--compact"])
        out = capsys.readouterr().out

        assert code == EXIT_OK
        assert len(out.strip().splitlines()) == 1

    def test_feature_extraction_can_be_skipped(self, tmp_path: Path, capsys) -> None:
        path = _write(tmp_path, 8.0)

        _, report = _run(capsys, "process", str(path), "--no-features")

        assert report["n_features"] == 0
        assert report["feature_shape"] is None
        assert report["n_segments"] > 0


class TestOverrides:
    def test_hop_override_is_applied(self, tmp_path: Path, capsys) -> None:
        path = _write(tmp_path, 16.0)

        _, default = _run(capsys, "process", str(path))
        _, coarse = _run(capsys, "process", str(path), "--hop", "3.5")

        assert coarse["hop_seconds"] == 3.5
        assert coarse["n_segments"] < default["n_segments"]

    def test_short_policy_pad_pads_a_sub_window_clip(
        self, tmp_path: Path, capsys
    ) -> None:
        path = _write(tmp_path, 3.0)

        code, report = _run(capsys, "process", str(path), "--short-policy", "pad")

        assert code == EXIT_OK
        assert report["n_padded_windows"] == 1
        # The padded window is still full width, so features match the nominal.
        assert report["feature_shape"][0] > 0

    def test_short_policy_drop_abstains_on_a_sub_window_clip(
        self, tmp_path: Path, capsys
    ) -> None:
        # 3s of audio against a 4s window, with 'drop', yields no window. That is
        # an abstention (exit 3), not a refusal: the file was read and measured
        # successfully. The stderr message names the actual cause so the fix is
        # obvious from the command output.
        path = _write(tmp_path, 3.0)

        code = main(["process", str(path), "--short-policy", "drop"])

        assert code == EXIT_UNSCORABLE
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "shorter than the 4.00s analysis window" in captured.err

    def test_short_policy_keep_scores_a_sub_window_clip(
        self, tmp_path: Path, capsys
    ) -> None:
        path = _write(tmp_path, 3.0)

        code = main(["process", str(path), "--short-policy", "keep"])

        assert code == EXIT_OK
        report = json.loads(capsys.readouterr().out)
        assert report["n_segments"] == 1

    def test_an_unknown_short_policy_is_rejected_by_the_parser(
        self, tmp_path: Path
    ) -> None:
        path = _write(tmp_path, 6.0)

        with pytest.raises(SystemExit):
            main(["process", str(path), "--short-policy", "shrink"])


class TestRefusals:
    def test_a_missing_file_exits_two(self, tmp_path: Path, capsys) -> None:
        code = main(["process", str(tmp_path / "absent.wav")])

        assert code == EXIT_BAD_PATH
        assert "not a file" in capsys.readouterr().err

    def test_a_directory_exits_two(self, tmp_path: Path, capsys) -> None:
        code = main(["process", str(tmp_path)])

        assert code == EXIT_BAD_PATH

    def test_undecodable_bytes_exit_one(self, tmp_path: Path, capsys) -> None:
        path = tmp_path / "junk.wav"
        path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunkjunkjunk")

        code = main(["process", str(path)])

        assert code == EXIT_REFUSED
        assert "AudioDecodeError" in capsys.readouterr().err

    def test_an_empty_file_exits_one(self, tmp_path: Path, capsys) -> None:
        path = tmp_path / "empty.wav"
        path.write_bytes(b"")

        code = main(["process", str(path)])

        assert code == EXIT_REFUSED

    def test_too_little_speech_is_an_abstention_not_a_refusal(
        self, tmp_path: Path, capsys
    ) -> None:
        # Exit 3, not 1. The clip was read, decoded, and measured; it just does
        # not carry enough speech to window. Reporting that the same way as an
        # undecodable file would tell an operator to re-export audio that is
        # perfectly fine and simply too short.
        path = _write(tmp_path, 0.6, amplitude=0.02)

        code = main(["process", str(path)])

        assert code == EXIT_UNSCORABLE
        assert "InsufficientSpeechError" in capsys.readouterr().err

    def test_a_refused_clip_prints_no_report(self, tmp_path: Path, capsys) -> None:
        path = _write(tmp_path, 0.6, amplitude=0.02)

        main(["process", str(path)])
        captured = capsys.readouterr()

        assert captured.out == ""
        assert captured.err != ""


class TestOutputSafety:
    def test_the_report_never_contains_audio(self, tmp_path: Path, capsys) -> None:
        path = _write(tmp_path, 8.0)

        _, report = _run(capsys, "process", str(path))

        serialised = json.dumps(report)
        assert "ndarray" not in serialised
        assert "redacted" not in serialised
        # The only nested structure is region bounds, which are float pairs.
        for pair in report["region_bounds"]:
            assert len(pair) == 2
            assert all(isinstance(x, float) for x in pair)

    def test_the_report_omits_the_file_path(self, tmp_path: Path, capsys) -> None:
        """A path is a filesystem detail, and logging it maps the host."""
        path = _write(tmp_path, 8.0)

        code = main(["process", str(path)])
        out = capsys.readouterr().out

        assert code == EXIT_OK
        assert str(path) not in out
        assert path.name not in out

    def test_existing_subcommands_still_work(self, capsys) -> None:
        assert main(["formats"]) == 0
        assert "allowed_containers" in capsys.readouterr().out


class TestDeclaredEntryPointsResolve:
    """Every console script in ``pyproject.toml`` must actually work.

    A declared entry point is a promise made at install time, to whoever runs
    ``pip install``. Nothing checks it until the script is invoked, and a missing
    module surfaces as a ``ModuleNotFoundError`` traceback on first use rather
    than as a packaging error.

    This class of defect shipped once already: ``voxshield-train`` and
    ``voxshield-evaluate`` both pointed at ``voxshield.scripts_entry``, which was
    never written. Setuptools would have installed both scripts happily.
    """

    @staticmethod
    def _declared_scripts() -> dict[str, str]:
        import tomllib
        from pathlib import Path

        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with pyproject.open("rb") as handle:
            return dict(tomllib.load(handle)["project"].get("scripts", {}))

    def test_at_least_one_script_is_declared(self) -> None:
        """A silently empty table would make the rest of this class vacuous."""
        assert self._declared_scripts()

    def test_every_declared_module_exists(self) -> None:
        import importlib.util

        for name, target in self._declared_scripts().items():
            module_name, _, _attr = target.partition(":")
            assert importlib.util.find_spec(module_name) is not None, (
                f"{name} points at {target}, but module {module_name!r} does not exist"
            )

    def test_every_declared_attribute_exists(self) -> None:
        from importlib import import_module

        for name, target in self._declared_scripts().items():
            module_name, _, attr = target.partition(":")
            module = import_module(module_name)
            assert hasattr(module, attr), f"{name} points at {target}, which is missing"

    def test_the_voxshield_script_invokes_the_cli(self) -> None:
        from importlib import import_module

        target = self._declared_scripts()["voxshield"]
        module_name, _, attr = target.partition(":")
        assert getattr(import_module(module_name), attr) is main
