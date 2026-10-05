"""End-to-end smoke of the ``voxshield data`` command group.

Not a test. Generates a small two-class corpus of real WAV files in a temporary
root, then drives every ``data`` subcommand through ``main()`` and prints each
exit code, so the operator contract is exercised the way a script would use it.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(os.environ.get("TEMP", ".")) / "voxshield_cli_smoke"
SPEAKERS = tuple(f"spk{index:03d}" for index in range(1, 7))
TAKES = 2
SECONDS = 6.0
RATE = 16000

CONFIG = f"""
root: {ROOT.as_posix()}
random_seed: 20260928
license_policy: require_verified
window_seconds: 4.0
hop_ratio: 0.5
storage_subtype: PCM_16
write_segment_audio: true

datasets:
  - dataset_id: voxshield_local
    name: VoxShield operator-supplied real speech
    source: local operator recordings
    version: "1.0"
    license: project-owned
    license_status: VERIFIED
    task: real_speech
    enabled: true
    path: raw/voxshield_local
    adapter: real_speech
    metadata:
      layout: flat
      speaker_pattern: "^spk(?P<speaker>[0-9]{{3}})_"

  - dataset_id: wavefake_smoke
    name: WaveFake-shaped synthetic corpus
    source: generated for this smoke run
    version: "0.0"
    license: project-owned
    license_status: VERIFIED
    task: spoof_detection
    enabled: true
    path: raw/wavefake_smoke
    adapter: wavefake
    metadata:
      speaker_pattern: "^spk(?P<speaker>[0-9]{{3}})_"

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
  streaming_max_sources: 512
  partition_by_temporal: false
  min_temporal_coverage: 0.95

validation:
  require_speech: true
  min_duration_seconds: 0.30
  max_duration_seconds: 60.0
  max_file_bytes: 67108864
  reject_warning_issues: false
  require_metadata: [label]
  reject_duplicates: true
  dedup_scope: both

augmentation:
  enabled: true
  class_balance: weighted_sampler
  gain_db_min: -6.0
  gain_db_max: 3.0
  noise_probability: 0.3
  noise_snr_db_min: 5.0
  noise_snr_db_max: 20.0
  noise_corpus_dir: ""
  codec_probability: 0.1
  codec_schemes: [g711_mulaw, g711_alaw]
  channel_probability: 0.1
  reverb_probability: 0.1
  seed: 20260928

cache:
  enabled: true
  max_entries: 200000

gates:
  require_speaker_disjoint: true
  require_generator_disjoint: false
  require_session_disjoint: false
  require_file_disjoint: true
  require_parent_disjoint: true
  require_language_disjoint: false
  fail_on_unavailable: false
  min_test_sources: 1
"""


def voiced(seed: int, seconds: float = SECONDS) -> np.ndarray:
    """A voiced-speech-shaped signal: harmonics under a syllabic envelope."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(RATE * seconds), dtype=np.float64) / RATE
    f0 = 110.0 + 20.0 * rng.random()
    signal = np.zeros_like(t)
    for harmonic in range(1, 26):
        signal += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t + rng.random())
    signal *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    signal += 0.01 * rng.standard_normal(t.size)
    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak * 0.6).astype(np.float32)


def synthetic(seed: int, seconds: float = SECONDS) -> np.ndarray:
    """A vocoder-shaped signal: speech-shaped, but spectrally unlike ``voiced``.

    Two lessons are baked in here. It must keep the syllabic envelope and enough
    low-frequency energy to read as speech, or the validator rejects every file
    with ``NO_SPEECH`` and the corpus silently collapses to one class -- a build
    that reports success while having proven nothing. And it must still differ
    from the bona fide signal, or a two-class build is no better than a one-class
    one. Equal-amplitude harmonics under the same syllable rate gives both: the
    envelope the gate looks for, and the flattened spectrum a real vocoder
    leaves behind.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(RATE * seconds), dtype=np.float64) / RATE
    f0 = 130.0 + 9.0 * (seed % 4)
    signal = np.zeros_like(t)
    for harmonic in range(1, 26):
        signal += np.sin(2 * np.pi * f0 * harmonic * t + rng.random())
    signal *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    signal += 0.015 * rng.standard_normal(t.size)
    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak * 0.6).astype(np.float32)


def clean(path: Path) -> None:
    """Remove a tree, retrying briefly: Windows can still hold a handle."""
    if not path.exists():
        return
    for attempt in range(5):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.4 * (attempt + 1))


def generate() -> tuple[int, int]:
    """Write both corpora. Returns ``(bona fide files, spoof files)``."""
    real = ROOT / "raw" / "voxshield_local"
    fake = ROOT / "raw" / "wavefake_smoke" / "train"
    real.mkdir(parents=True)
    (fake / "gen" / "melgan").mkdir(parents=True)
    (fake / "orig").mkdir(parents=True)

    bona = 0
    spoof = 0
    for speaker_index, speaker in enumerate(SPEAKERS):
        for take in range(TAKES):
            stem = f"{speaker}_{take:02d}"
            seed = speaker_index * 100 + take
            # Same underlying audio across the two corpora, so the paired
            # orig/gen copies share content and the duplicate and leakage
            # checks have something real to catch.
            source = voiced(seed=seed)
            fake_audio = synthetic(seed=seed)
            sf.write(real / f"{stem}.wav", source, RATE, subtype="PCM_16")
            sf.write(fake / "orig" / f"{stem}.wav", source, RATE, subtype="PCM_16")
            sf.write(fake / "gen" / "melgan" / f"{stem}.wav", fake_audio, RATE, subtype="PCM_16")
            bona += 2
            spoof += 1
    return bona, spoof


def run(argv: list[str]) -> tuple[int, str, str]:
    """Invoke ``main`` with stdout and stderr captured."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        from voxshield.cli import main

        code = main(argv)
    return code, out.getvalue(), err.getvalue()


FAILURES: list[str] = []


def check(label: str, code: int, expected: int) -> None:
    """Record a mismatch between the exit code seen and the one required."""
    if code != expected:
        FAILURES.append(f"{label}: expected exit {expected}, got {code}")


def show(label: str, argv: list[str], *, keys: tuple[str, ...] = (), expect: int = 0) -> None:
    """Run one subcommand and print its exit code and a short summary."""
    code, out, err = run(argv)
    print(f"\n=== {label} :: {' '.join(argv)}")
    print(f"    exit {code} (expected {expect})")
    check(label, code, expect)
    if err.strip():
        print(f"    stderr: {err.strip()[:400]}")
    if out.strip():
        try:
            payload = json.loads(out)
        except json.JSONDecodeError:
            print(f"    stdout(raw): {out.strip()[:400]}")
            return
        if keys:
            brief = {key: payload.get(key) for key in keys}
        else:
            brief = {
                key: value
                for key, value in payload.items()
                if not isinstance(value, (list, dict))
            }
        print(f"    stdout: {json.dumps(brief, indent=6, sort_keys=True)[:1200]}")
    return None


def brief(label: str, argv: list[str], *, expect: int = 0) -> None:
    """Run one build and print only its build statistics."""
    code, out, err = run(argv)
    print(f"\n=== {label} :: {' '.join(argv)}")
    print(f"    exit {code} (expected {expect})")
    check(label, code, expect)
    if err.strip():
        print(f"    stderr: {err.strip()[:400]}")
    try:
        build = json.loads(out)["build"]
        summary = {
            key: build[key]
            for key in (
                "build_id",
                "rows",
                "cache_hits",
                "elapsed_seconds",
                "real_time_factor",
                "gates_passed",
            )
        }
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"    could not read build report: {exc}")
        return
    print(json.dumps(summary, indent=4, sort_keys=True))


def _root_seen_by(argv: list[str]) -> tuple[int, str]:
    """Return the exit code and root the CLI actually resolved."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        from voxshield.cli import main

        code = main(argv)
    try:
        return code, str(json.loads(out.getvalue())["root"])
    except (json.JSONDecodeError, KeyError):
        return code, f"<no root in report: {err.getvalue()[:200]}>"


def main() -> int:
    """Generate the corpus and drive every data subcommand."""
    clean(ROOT)
    bona, spoof = generate()
    print(f"wrote {bona} bona fide and {spoof} spoof wav files under {ROOT}")

    config = ROOT / "data.yaml"
    config.write_text(CONFIG, encoding="utf-8")

    common = ["--config", str(config)]
    show(
        "paths",
        ["data", "paths", *common],
        keys=("config_source", "dataset_build_id", "cache_enabled"),
    )
    show(
        "discover",
        ["data", "discover", *common],
        keys=("kept", "stats", "config_source"),
    )
    show(
        "validate",
        ["data", "validate", *common, "--limit", "5"],
        keys=("discovered", "accepted", "rejected", "rejected_per_reason"),
    )
    # ``--no-cache`` neither reads nor writes, so it leaves the cache exactly as
    # it found it. That is the point of the flag, but it also means the build
    # right after one is still cold -- so "warm" has to mean a second build with
    # the default cache, not the next build after a --no-cache build.
    brief("build (cold: reads nothing, writes nothing)", ["data", "build", *common, "--no-cache"])
    brief("build (cold: reads nothing, populates cache)", ["data", "build", *common])
    brief("build (warm: should hit cache)", ["data", "build", *common])

    show("inspect all", ["data", "inspect", *common], keys=("retired_rows",))
    show(
        "inspect train",
        ["data", "inspect", *common, "--split", "train", "--limit", "2"],
        keys=("requested_split", "retired_rows"),
    )
    show(
        "check-leakage",
        ["data", "check-leakage", *common],
        keys=("rows", "has_leakage", "leaked_axes", "unavailable_axes"),
        # Every spoof clip here comes from one generator, so the generator axis
        # really does leak. Exiting 1 is the finding, not a failure of the check.
        expect=1,
    )
    show(
        "check-leakage --strict",
        ["data", "check-leakage", *common, "--strict"],
        keys=("has_leakage", "unavailable_axes"),
        expect=1,
    )
    show(
        "test-loader train",
        [
            "data",
            "test-loader",
            *common,
            "--split",
            "train",
            "--batch-size",
            "4",
            "--batches",
            "3",
            "--samples",
            "1",
        ],
        keys=("split", "row_count", "label_counts", "augmentation", "ok", "reason"),
    )
    show(
        "test-loader test",
        ["data", "test-loader", *common, "--split", "test", "--batch-size", "4", "--batches", "2"],
        keys=("split", "row_count", "label_counts", "ok"),
    )

    print("\n--- negative cases ---")
    show(
        "no manifest for this root",
        ["data", "inspect", *common, "--root", str(ROOT / "empty")],
        expect=2,
    )
    show("bad config path", ["data", "paths", "--config", str(ROOT / "nope.yaml")], expect=2)
    show("bad split", ["data", "inspect", *common, "--split", "bogus"], expect=2)
    show(
        "missing sample id",
        ["data", "inspect", *common, "--sample", "does-not-exist"],
        expect=2,
    )
    show(
        "cold loader with no torch installed is not exercised here",
        ["data", "test-loader", *common, "--split", "dev", "--workers", "0"],
    )

    print("\n--- precedence: --root must beat VOXSHIELD_DATA_ROOT ---")
    saved = os.environ.copy()
    os.environ["VOXSHIELD_DATA_ROOT"] = str(ROOT / "from-env")
    try:
        env_only = _root_seen_by(["data", "paths", *common])
        overridden = _root_seen_by(["data", "paths", *common, "--root", str(ROOT)])
    finally:
        os.environ.clear()
        os.environ.update(saved)

    print(f"    env-only           exit {env_only[0]} root={json.dumps(env_only[1])}")
    print(f"    --root beats env   exit {overridden[0]} root={json.dumps(overridden[1])}")
    ok = (
        env_only[0] == 0
        and env_only[1] == str(ROOT / "from-env")
        and overridden[0] == 0
        and overridden[1] == str(ROOT)
    )
    print(f"    precedence respected: {ok}")
    if not ok:
        FAILURES.append("precedence: --root did not beat VOXSHIELD_DATA_ROOT")

    if FAILURES:
        print(f"\n--- {len(FAILURES)} FAILURE(S) ---")
        for failure in FAILURES:
            print(f"    {failure}")
        return 1
    print("\n--- all expected exit codes matched ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())