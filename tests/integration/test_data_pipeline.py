"""End-to-end dataset pipeline over real audio files.

The unit tests check each stage in isolation, which is exactly what makes them
able to miss the failures that only appear when the stages run in order: a
manifest written for a root the loader does not use, a cache that saves work
while quietly returning the wrong thing, a split that satisfies each rule alone
and violates their intersection.

So this file drives the real thing -- real WAV files on disk, the real config,
the real build -- and asserts the properties a person citing the corpus would
assume. It is slower than the unit tests by design; that cost is the point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxshield.cli import DATA_FAILED, DATA_OK, main
from voxshield.data.build import build_dataset
from voxshield.data.cache import PreprocessingCache
from voxshield.data.config import load_data_config
from voxshield.data.manifest import SPLIT_MANIFESTS, read_manifest
from voxshield.data.schema import UNKNOWN

SAMPLE_RATE = 16_000
SPEAKERS = 8

CONFIG_TEXT = """
root: __ROOT__
random_seed: 20260928
license_policy: require_verified
window_seconds: 1.0
hop_ratio: 0.5
storage_subtype: PCM_16
write_segment_audio: true

datasets:
  - dataset_id: bona
    name: Bona fide speech
    source: generated for this test
    version: "1.0"
    license: project-owned
    license_status: VERIFIED
    license_verified_by: fixture
    task: real_speech
    enabled: true
    path: raw/bona
    adapter: real_speech
    metadata:
      speaker_pattern: "^spk(?P<speaker>[0-9]{3})_"

  - dataset_id: fake
    name: Synthetic speech
    source: generated for this test
    version: "0.0"
    license: project-owned
    license_status: VERIFIED
    license_verified_by: fixture
    task: spoof_detection
    enabled: true
    path: raw/fake
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
  train_ratio: 0.6
  dev_ratio: 0.2
  min_train_speakers: 2

validation:
  require_speech: true
  min_duration_seconds: 0.30
  reject_warning_issues: false
  require_metadata: []
  reject_duplicates: true
  dedup_scope: both

augmentation:
  enabled: false
  class_balance: none

gates:
  min_test_sources: 1
"""


def speech(seed: int, seconds: float) -> np.ndarray:
    """A voiced signal with a syllable envelope, so it reads as speech not noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    f0 = 105.0 + 9.0 * (seed % 7)
    signal = np.zeros_like(t)
    for harmonic, weight in ((1, 1.0), (2, 0.65), (3, 0.4), (4, 0.25), (6, 0.12)):
        signal += weight * np.sin(2 * np.pi * f0 * harmonic * t + rng.random() * 6.28)
    signal *= 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    signal += 0.008 * rng.standard_normal(t.size)
    return (signal / (float(np.max(np.abs(signal))) or 1.0) * 0.55).astype(np.float32)


def write_corpus(root: Path, *, identical_classes: bool) -> Path:
    """Two classes of real audio on disk, plus the config that points at them.

    ``identical_classes`` writes byte-identical files under both labels. That is
    a trap rather than a shortcut: the pipeline is expected to notice and drop
    one copy, so tests can pin that it does.
    """
    bona = root / "raw" / "bona"
    spoof = root / "raw" / "fake" / "train" / "gen" / "melgan"
    for directory in (bona, spoof):
        directory.mkdir(parents=True, exist_ok=True)
    for index in range(SPEAKERS):
        for take in range(2):
            name = f"spk{index:03d}_{take}.wav"
            real = speech(seed=index, seconds=3.0 + take)
            fake = real if identical_classes else speech(seed=1_000 + index, seconds=3.0 + take)
            sf.write(bona / name, real, SAMPLE_RATE, subtype="PCM_16")
            sf.write(spoof / name, fake, SAMPLE_RATE, subtype="PCM_16")
    config = root / "data.yaml"
    config.write_text(CONFIG_TEXT.replace("__ROOT__", root.as_posix()), encoding="utf-8")
    return config


@pytest.fixture(scope="module")
def pipeline_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A built two-class corpus with manifests, reused across the tests here."""
    root = tmp_path_factory.mktemp("pipeline")
    config = write_corpus(root, identical_classes=False)
    assert main(["data", "build", "--config", str(config)]) == DATA_OK
    return root


@pytest.fixture(scope="module")
def splits(pipeline_root: Path) -> dict[str, tuple]:
    """The three split manifests, read once and shared.

    Cross-split invariants cannot be checked one file at a time, so the tests
    that care read all three together.
    """
    return {
        split: read_manifest(pipeline_root / "manifests" / SPLIT_MANIFESTS[split]).samples
        for split in ("train", "dev", "test")
    }


def config_for(root: Path):  # type: ignore[no-untyped-def]
    """The same configuration the CLI used, loaded the library way."""
    return load_data_config(root / "data.yaml")


def identities(result) -> list[dict[str, object]]:  # type: ignore[no-untyped-def]
    """Row identities with every timestamp and path that legitimately moves removed."""
    return [
        {
            "sample_id": row.sample_id,
            "split": row.split,
            "label_index": row.label_index,
            "start_seconds": row.start_seconds,
            "duration_seconds": row.duration_seconds,
            "coverage": row.coverage,
            "speaker_id": row.speaker_id,
            "dataset_build_id": row.dataset_build_id,
        }
        for row in result.rows
    ]


class TestTheCorpusIsUsable:
    def test_it_built_rows_in_every_split(self, pipeline_root: Path) -> None:
        for split in ("train", "dev", "test"):
            manifest = read_manifest(pipeline_root / "manifests" / SPLIT_MANIFESTS[split])
            assert manifest.samples, f"{split} is empty"

    def test_both_labels_survive(self, pipeline_root: Path) -> None:
        manifest = read_manifest(pipeline_root / "manifests" / SPLIT_MANIFESTS["all"])
        assert {row.label_index for row in manifest.samples} == {0, 1}

    def test_every_row_points_at_audio_that_exists(self, pipeline_root: Path) -> None:
        """The most expensive failure mode is a manifest full of unreadable paths."""
        manifest = read_manifest(pipeline_root / "manifests" / SPLIT_MANIFESTS["all"])
        missing = [
            row.sample_id
            for row in manifest.samples
            if not (pipeline_root / row.audio_path).is_file()
        ]
        assert missing == []

    def test_every_row_shares_one_build_id(self, pipeline_root: Path) -> None:
        manifest = read_manifest(pipeline_root / "manifests" / SPLIT_MANIFESTS["all"])
        assert {row.dataset_build_id for row in manifest.samples} == {
            manifest.header.dataset_build_id
        }


class TestSplitInvariantsHoldOnDisk:
    """Rules that hold per manifest can still fail across them, so read all three."""

    def test_no_known_speaker_appears_in_two_splits(self, splits: dict[str, tuple]) -> None:
        """``UNKNOWN`` is the absence of a speaker, so it is not a shared speaker.

        Real WaveFake filenames do not encode one. Counting ``UNKNOWN`` as a
        speaker would report a collision between two files that merely both
        declined to name a speaker.
        """
        seen: dict[str, str] = {}
        collisions = []
        for split, rows in splits.items():
            for row in rows:
                if row.speaker_id == UNKNOWN:
                    continue
                if seen.setdefault(row.speaker_id, split) != split:
                    collisions.append(row.speaker_id)
        assert collisions == []

    def test_no_source_recording_is_split_across_splits(self, splits: dict[str, tuple]) -> None:
        seen: dict[str, str] = {}
        collisions = []
        for split, rows in splits.items():
            for row in rows:
                if seen.setdefault(row.parent_id, split) != split:
                    collisions.append(row.parent_id)
        assert collisions == []

    def test_no_segment_is_in_two_splits(self, splits: dict[str, tuple]) -> None:
        seen: dict[str, str] = {}
        collisions = []
        for split, rows in splits.items():
            for row in rows:
                if seen.setdefault(row.sample_id, split) != split:
                    collisions.append(row.sample_id)
        assert collisions == []

    def test_windows_are_uniformly_sized(self, splits: dict[str, tuple]) -> None:
        """Rectangular batches are what let ``collate_samples`` stack without padding."""
        widths = {row.waveform_samples for rows in splits.values() for row in rows}
        assert len(widths) == 1, f"ragged corpus: {sorted(widths)}"


class TestRebuildsAreDeterministic:
    def test_the_same_config_reproduces_the_same_rows(self, pipeline_root: Path) -> None:
        config = config_for(pipeline_root)
        first = identities(build_dataset(config, overwrite=True))
        second = identities(build_dataset(config, overwrite=True))
        assert first == second

    def test_the_build_id_is_stable(self, pipeline_root: Path) -> None:
        config = config_for(pipeline_root)
        assert build_dataset(config, overwrite=True).build_id == build_dataset(
            config, overwrite=True
        ).build_id

    def test_the_manifest_differs_only_in_its_timestamp(self, pipeline_root: Path) -> None:
        target = pipeline_root / "manifests" / SPLIT_MANIFESTS["train"]
        build_dataset(config_for(pipeline_root), overwrite=True)
        first = target.read_text(encoding="utf-8")

        def without_timestamp(text: str) -> list[dict]:
            return [
                {k: v for k, v in json.loads(line).items() if k != "created_at"}
                for line in text.splitlines()
                if line.strip()
            ]

        build_dataset(config_for(pipeline_root), overwrite=True)
        assert without_timestamp(first) == without_timestamp(target.read_text(encoding="utf-8"))

    def test_a_changed_window_size_changes_the_build(self, pipeline_root: Path) -> None:
        """A different window is a different corpus, so the stamp must say so."""
        narrow = build_dataset(
            config_for(pipeline_root).__class__(
                **{
                    **{
                        field: getattr(config_for(pipeline_root), field)
                        for field in config_for(pipeline_root).__dataclass_fields__
                    },
                    "window_seconds": 0.5,
                }
            ),
            overwrite=True,
        )
        assert narrow.build_id != build_dataset(config_for(pipeline_root), overwrite=True).build_id


class TestCacheReuse:
    def test_a_rebuild_reads_the_cache(self, pipeline_root: Path) -> None:
        config = config_for(pipeline_root)
        build_dataset(config, overwrite=True)
        warm = build_dataset(config, overwrite=True)
        assert warm.cache_hits == len(warm.validation.accepted)
        assert warm.cache_hits > 0

    def test_a_cold_run_reports_no_hits(self, pipeline_root: Path) -> None:
        config = config_for(pipeline_root)
        build_dataset(config, overwrite=True)
        cold = build_dataset(config, overwrite=True, cache=PreprocessingCache(config.data_paths(), enabled=False))
        assert cold.cache_hits == 0
        assert cold.rows, "a cold run must still produce the same corpus"

    def test_a_cold_run_produces_the_same_rows_as_a_warm_one(self, pipeline_root: Path) -> None:
        """The cache is an optimisation; it must not be able to change the answer."""
        config = config_for(pipeline_root)
        warm = identities(build_dataset(config, overwrite=True))
        cold = identities(
            build_dataset(
                config, overwrite=True, cache=PreprocessingCache(config.data_paths(), enabled=False)
            )
        )
        assert warm == cold

    def test_a_cold_run_leaves_the_cache_intact(self, pipeline_root: Path) -> None:
        config = config_for(pipeline_root)
        build_dataset(config, overwrite=True)
        cache_dir = config.data_paths().cache
        before = {p.name for p in cache_dir.rglob("*") if p.is_file()}
        build_dataset(config, overwrite=True, cache=PreprocessingCache(cache_dir, enabled=False))
        assert {p.name for p in cache_dir.rglob("*") if p.is_file()} == before


class TestIdenticalAudioCannotSpanBothClasses:
    """The same bytes under two labels is a leak the dedup pass has to remove.

    A corpus holding one recording as both bona fide and spoof is not a weak
    corpus, it is a broken one: every model scores it perfectly and learns
    nothing. Getting this wrong is silent, so it is pinned here.
    """

    def test_the_duplicate_pair_collapses_to_one_label(self, tmp_path: Path) -> None:
        result = build_dataset(
            load_data_config(write_corpus(tmp_path, identical_classes=True)), overwrite=True
        )
        labels = {row.label for row in result.rows}
        assert len(labels) == 1, f"identical audio survived under {sorted(labels)}"

    def test_it_is_reported_rather_than_silently_dropped(self, tmp_path: Path) -> None:
        """Dropping a duplicate without recording it is indistinguishable from loss."""
        result = build_dataset(
            load_data_config(write_corpus(tmp_path, identical_classes=True)), overwrite=True
        )
        dropped = result.discovery.dropped_ids()
        assert dropped, "the collapse happened but left no record of what it removed"


class TestTheWholeCommandSequence:
    def test_a_pipeline_script_gets_usable_exit_codes(self, pipeline_root: Path) -> None:
        """The order a script runs them in, with the codes it must be able to trust."""
        config = str(pipeline_root / "data.yaml")
        assert main(["data", "build", "--config", config]) == DATA_OK
        assert main(["data", "check-leakage", "--config", config]) in (DATA_OK, DATA_FAILED)
        assert main(["data", "inspect", "--config", config]) == DATA_OK
        assert main(["data", "inspect", "--config", config, "--split", "train"]) == DATA_OK
        assert main(["data", "test-loader", "--config", config, "--split", "train"]) == DATA_OK

    def test_a_later_stage_refuses_to_run_without_a_build(self, pipeline_root: Path) -> None:
        """Checking a corpus that was never built must not report a clean result."""
        empty = pipeline_root / "never-built"
        empty.mkdir(exist_ok=True)
        config = str(pipeline_root / "data.yaml")
        assert main(["data", "check-leakage", "--config", config, "--root", str(empty)]) == 2
        assert main(["data", "test-loader", "--config", config, "--root", str(empty)]) == 2

    def test_preprocessing_is_faster_than_realtime(self, pipeline_root: Path) -> None:
        """A build slower than realtime would make the corpus unbuildable in practice.

        The bound is deliberately loose -- this is a smoke signal that the pipeline
        is not accidentally quadratic, not a performance benchmark to tune against.
        """
        config = config_for(pipeline_root)
        result = build_dataset(config, overwrite=True)
        audio_seconds = sum(row.duration_seconds for row in result.rows)
        assert audio_seconds > 0
        ratio = audio_seconds / max(result.elapsed_seconds, 1e-9)
        assert ratio > 1.0, (
            f"preprocessing ran at {ratio:.2f}x realtime ({result.elapsed_seconds:.2f}s "
            f"for {audio_seconds:.1f}s of audio)"
        )


