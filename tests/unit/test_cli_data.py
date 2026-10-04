"""The ``voxshield data`` command group.

The contract these tests pin down is the one a pipeline script depends on: JSON
on stdout, diagnostics on stderr, and an exit code that says whether the command
*found* something or *could not look*. Getting that last distinction wrong is
the dangerous kind of bug -- a leakage check that could not run and one that
found nothing both have to stop a build, but only one of them is a finding about
the corpus.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxshield.cli import DATA_CANNOT_RUN, DATA_FAILED, DATA_OK, main

SAMPLE_RATE = 16_000
SPEAKERS = 6
CONFIG_TEXT = """
root: __ROOT__
random_seed: 20260928
license_policy: require_verified
window_seconds: 4.0
hop_ratio: 0.5
storage_subtype: PCM_16
write_segment_audio: false

datasets:
  - dataset_id: local
    name: Local speech
    source: generated for this test
    version: "1.0"
    license: project-owned
    license_status: VERIFIED
    license_verified_by: fixture
    task: real_speech
    enabled: true
    path: raw/local
    adapter: real_speech
    metadata:
      speaker_pattern: "^spk(?P<speaker>[0-9]{3})_"

  - dataset_id: synth
    name: Synthetic speech
    source: generated for this test
    version: "0.0"
    license: project-owned
    license_status: VERIFIED
    license_verified_by: fixture
    task: spoof_detection
    enabled: true
    path: raw/synth
    adapter: wavefake

paths:
  raw: raw
  interim: interim
  processed: processed
  manifests: manifests
  synthetic: synthetic
  cache: cache
  reports: reports

split:
  train_ratio: 0.70
  dev_ratio: 0.15
  min_train_speakers: 2

validation:
  require_speech: true
  min_duration_seconds: 0.30
  reject_warning_issues: false
  require_metadata: [label]
  reject_duplicates: true
  dedup_scope: both

augmentation:
  enabled: false
  class_balance: none
"""


def speechish(seed: int, seconds: float = 6.0) -> np.ndarray:
    """A speech-shaped signal that survives the validator's speech gate.

    The syllable envelope is what the gate keys on; a signal without one is
    rejected as ``NO_SPEECH`` and the corpus quietly collapses to one class.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    f0 = 110.0 + 8.0 * (seed % 6)
    signal = np.zeros_like(t)
    for harmonic, weight in ((1, 1.0), (2, 0.7), (3, 0.45), (5, 0.25), (7, 0.15)):
        signal += weight * np.sin(2 * np.pi * f0 * harmonic * t + rng.random())
    signal *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    signal += 0.01 * rng.standard_normal(t.size)
    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak * 0.6).astype(np.float32)


def write_corpus(root: Path) -> None:
    """Write a two-class corpus, two differing signals per speaker.

    The synthetic side follows WaveFake's directory-derived labelling -- the
    label *is* the path -- because that is what makes a bona fide/spoof pair
    share a ``parent_id``. Pairing them is the point: it is the arrangement that
    catches a split putting a clip next to its own twin.
    """
    local = root / "raw" / "local"
    spoof = root / "raw" / "synth" / "train" / "gen" / "melgan"
    for directory in (local, spoof):
        directory.mkdir(parents=True, exist_ok=True)

    for index in range(SPEAKERS):
        stem = f"spk{index:03d}"
        for take, jitter in enumerate((0.0, 0.35)):
            audio = speechish(seed=index, seconds=6.0)
            if jitter:
                audio = np.roll(audio, int(jitter * SAMPLE_RATE))
            name = f"{stem}_{take}.wav"
            sf.write(local / name, audio, SAMPLE_RATE, subtype="PCM_16")
            sf.write(spoof / name, audio, SAMPLE_RATE, subtype="PCM_16")


@pytest.fixture(scope="module")
def corpus_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A built two-class corpus with manifests, shared across the tests here.

    Module-scoped because every build in this file would otherwise redo the same
    discovery, validation, and segmentation work.
    """
    root = tmp_path_factory.mktemp("cli_data")
    write_corpus(root)
    # The root has to be stamped in, because ``raw/`` is resolved relative to it
    # and the default is nowhere near the temporary directory. Plain substitution
    # rather than ``format``: the speaker regex below is full of braces.
    (root / "data.yaml").write_text(
        CONFIG_TEXT.replace("__ROOT__", root.as_posix()), encoding="utf-8"
    )
    code = main(["data", "build", "--config", str(root / "data.yaml")])
    assert code == DATA_OK, "fixture corpus failed to build"
    return root


@pytest.fixture
def argv(corpus_root: Path) -> list[str]:
    """The common ``--config`` arguments for the shared corpus."""
    return ["--config", str(corpus_root / "data.yaml")]


def invoke(*args: str) -> tuple[int, str, str]:
    """Run the CLI, returning its exit code, stdout, and stderr."""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(args))
    return code, out.getvalue(), err.getvalue()


def payload_of(out: str) -> dict:
    """Parse stdout as a JSON object."""
    return json.loads(out)


class TestReportsAreMachineReadable:
    def test_paths_prints_json_and_exits_zero(self, argv: list[str]) -> None:
        code, out, err = invoke("data", "paths", *argv)
        assert code == DATA_OK
        assert err == ""
        body = payload_of(out)
        assert body["root"]
        assert body["dataset_build_id"].startswith("vs-")

    def test_discover_reports_the_corpus_it_found(self, argv: list[str]) -> None:
        code, out, _err = invoke("data", "discover", *argv)
        assert code == DATA_OK
        body = payload_of(out)
        assert body["kept"] > 0
        assert body["stats"]["total_records"] >= body["kept"]

    def test_validate_reports_acceptance_counts(self, argv: list[str]) -> None:
        code, out, _err = invoke("data", "validate", *argv)
        assert code == DATA_OK
        body = payload_of(out)
        assert body["accepted"] > 0
        assert body["accepted"] + body["rejected"] == body["discovered"]

    def test_compact_is_one_line(self, argv: list[str]) -> None:
        code, out, _err = invoke("data", "paths", *argv, "--compact")
        assert code == DATA_OK
        assert len(out.strip().splitlines()) == 1

    def test_a_missing_config_is_reported_not_raised(self, tmp_path: Path) -> None:
        """A bad path must not surface as a traceback out of ``main``."""
        code, out, err = invoke("data", "paths", "--config", str(tmp_path / "absent.yaml"))
        assert code == DATA_CANNOT_RUN
        assert out == ""
        assert "DatasetConfigError" in err


class TestBuild:
    def test_build_is_reported_with_manifests(self, corpus_root: Path) -> None:
        code, out, _err = invoke("data", "build", "--config", str(corpus_root / "data.yaml"))
        assert code == DATA_OK
        build = payload_of(out)["build"]
        assert build["rows"] > 0
        assert build["gates_passed"] is True
        for name in ("train", "dev", "test", "all"):
            assert (corpus_root / "manifests" / f"{name}.jsonl").is_file()

    def test_a_second_build_hits_the_cache(self, corpus_root: Path) -> None:
        first = payload_of(invoke("data", "build", "--config", str(corpus_root / "data.yaml"))[1])
        second = payload_of(invoke("data", "build", "--config", str(corpus_root / "data.yaml"))[1])
        assert second["build"]["cache_hits"] > 0
        assert second["build"]["build_id"] == first["build"]["build_id"]

    def test_no_cache_reports_no_hits(self, corpus_root: Path) -> None:
        code, out, _err = invoke(
            "data", "build", "--config", str(corpus_root / "data.yaml"), "--no-cache"
        )
        assert code == DATA_OK
        assert payload_of(out)["build"]["cache_hits"] == 0

    def test_no_gates_reports_the_gate_refusal_instead_of_failing(self, corpus_root: Path) -> None:
        """``--no-gates`` is how a draft corpus is inspected, so it must build."""
        code, out, _err = invoke(
            "data",
            "build",
            "--config",
            str(corpus_root / "data.yaml"),
            "--no-gates",
            "--no-overwrite",
        )
        assert code == DATA_OK
        assert "gate_failures" in payload_of(out)["build"]


class TestInspect:
    def test_all_is_the_default_split(self, argv: list[str]) -> None:
        code, out, _err = invoke("data", "inspect", *argv)
        assert code == DATA_OK
        assert payload_of(out)["requested_split"] == "all"

    def test_a_named_split_is_reported(self, argv: list[str]) -> None:
        code, out, _err = invoke("data", "inspect", *argv, "--split", "train")
        assert code == DATA_OK
        assert payload_of(out)["requested_split"] == "train"

    def test_an_unknown_split_cannot_run(self, argv: list[str]) -> None:
        code, out, err = invoke("data", "inspect", *argv, "--split", "bogus")
        assert code == DATA_CANNOT_RUN
        assert out == ""
        assert "bogus" in err

    def test_an_unknown_sample_id_cannot_run(self, argv: list[str]) -> None:
        """A bad argument, not a finding about the corpus."""
        code, _out, err = invoke("data", "inspect", *argv, "--sample", "no-such-sample")
        assert code == DATA_CANNOT_RUN
        assert "no-such-sample" in err

    def test_a_root_without_manifests_cannot_run(self, corpus_root: Path, argv: list[str]) -> None:
        code, out, err = invoke("data", "inspect", *argv, "--root", str(corpus_root / "void"))
        assert code == DATA_CANNOT_RUN
        assert out == ""
        assert "build" in err


class TestLeakage:
    """The exit code reports the *finding*, and ``--strict`` widens what counts.

    Note this corpus deliberately has one vocoder, so the generator axis really
    does leak. The tests assert the contract -- the code follows the payload --
    rather than a leak-free corpus that would hide a broken mapping.
    """

    def test_the_exit_code_follows_the_finding(self, argv: list[str]) -> None:
        code, out, _err = invoke("data", "check-leakage", *argv)
        body = payload_of(out)
        assert code == (DATA_FAILED if body["has_leakage"] else DATA_OK)

    def test_a_found_leak_names_its_axis(self, argv: list[str]) -> None:
        code, out, _err = invoke("data", "check-leakage", *argv)
        body = payload_of(out)
        if body["has_leakage"]:
            assert code == DATA_FAILED
            assert body["leaked_axes"], "a finding with no axis named is not actionable"
        else:
            assert code == DATA_OK

    def test_strict_refuses_unevaluable_axes(self, argv: list[str]) -> None:
        """``--strict`` turns an unverifiable axis into a failure, not a warning."""
        lenient = payload_of(invoke("data", "check-leakage", *argv)[1])
        code, out, err = invoke("data", "check-leakage", *argv, "--strict")
        strict = payload_of(out)
        assert strict["unavailable_axes"] == lenient["unavailable_axes"]
        assert strict["has_leakage"] == lenient["has_leakage"]

        if lenient["has_leakage"]:
            # A real finding outranks the unevaluable axes, so ``--strict`` adds
            # nothing and must not claim to be the reason for the failure.
            assert code == DATA_FAILED
            assert "strict" not in err
        elif lenient["unavailable_axes"]:
            assert code == DATA_FAILED
            assert "strict" in err
            assert set(strict["unavailable_axes"]) == set(err.split(":", 1)[1].split(", "))
        else:
            assert code == DATA_OK


class TestRootPrecedence:
    """``--root`` > ``VOXSHIELD_DATA_ROOT`` > the configuration document.

    The library's own ``load_data_config`` gives the environment the last word,
    which is right for an unattended run but wrong here: a person typing
    ``--root`` means it. The CLI gets there by dropping the variable before it
    calls the loader, so this ordering is a CLI decision rather than a
    coincidence, and it is pinned here so a future simplification does not
    quietly reverse it.
    """

    def test_the_root_flag_beats_the_environment(
        self, corpus_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elsewhere = corpus_root / "from-env"
        override = corpus_root / "from-flag"
        elsewhere.mkdir(exist_ok=True)
        override.mkdir(exist_ok=True)
        monkeypatch.setenv("VOXSHIELD_DATA_ROOT", str(elsewhere))
        code, out, _err = invoke(
            "data", "paths", "--config", str(corpus_root / "data.yaml"), "--root", str(override)
        )
        assert code == DATA_OK
        assert payload_of(out)["root"] == str(override)

    def test_the_environment_beats_the_config_file(
        self, corpus_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elsewhere = corpus_root / "from-env"
        elsewhere.mkdir(exist_ok=True)
        monkeypatch.setenv("VOXSHIELD_DATA_ROOT", str(elsewhere))
        code, out, _err = invoke("data", "paths", "--config", str(corpus_root / "data.yaml"))
        assert code == DATA_OK
        assert payload_of(out)["root"] == str(elsewhere)

    def test_the_config_file_is_used_when_neither_is_given(
        self, corpus_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VOXSHIELD_DATA_ROOT", raising=False)
        code, out, _err = invoke("data", "paths", "--config", str(corpus_root / "data.yaml"))
        assert code == DATA_OK
        assert payload_of(out)["root"] == str(corpus_root)


class TestTopLevelInspectIsUntouched:
    def test_the_audit_command_still_owns_inspect(self) -> None:
        """``data inspect`` is new; the top-level audit ``inspect`` must not move."""
        from voxshield.cli import build_parser

        parser = build_parser()
        top_level = parser.parse_args(["inspect", "req-1"])
        nested = parser.parse_args(["data", "inspect"])
        assert top_level.func is not nested.func
        assert top_level.request_id == "req-1"
        assert not hasattr(top_level, "split"), "audit inspect must not gain data flags"


@pytest.fixture(autouse=True)
def _quiet_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep build logs out of captured stderr for the exit-code assertions."""
    monkeypatch.setenv("VOXSHIELD_LOG_LEVEL", "ERROR")
    yield