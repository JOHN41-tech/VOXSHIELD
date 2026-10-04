"""The Torch-facing half of the dataset layer.

A build that writes manifests is not yet usable. These tests cover the gap
between "the rows exist on disk" and "a trainer can iterate them": that a batch
stacks into one tensor, that the row count the loader reports matches the
manifest, and that the two guards which silently shrink a training set -- the
coverage floor and the padded-window filter -- actually fire.

Manifests are written with the real writer rather than hand-rolled JSONL, so a
schema change cannot leave this file quietly testing a fiction.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

from voxshield.data.errors import ManifestError  # noqa: E402
from voxshield.data.labels import BONA_FIDE, SPOOF, encode_label  # noqa: E402
from voxshield.data.manifest import read_manifest, write_manifest_set  # noqa: E402
from voxshield.data.paths import DataPaths  # noqa: E402
from voxshield.data.schema import SampleRecord  # noqa: E402
from voxshield.data.torch_dataset import (  # noqa: E402
    VoxShieldDataset,
    build_dataloader,
    collate_samples,
    load_split,
    read_segment,
    resolve_split,
    summarise_batches,
)
from voxshield.errors import AudioDecodeError  # noqa: E402

SAMPLE_RATE = 16_000
DURATION = 1.0
BUILD = "vs-torch-fixture"
RELATIVE = "processed/segments/fixture"


def segment(sample_id: str, *, split: str, label: str, coverage: float, is_padded: bool) -> SampleRecord:
    """A manifest row whose audio file is written alongside it."""
    return SampleRecord(
        sample_id=sample_id,
        dataset_id="fixture",
        audio_path=f"{RELATIVE}/{sample_id}.wav",
        label=label,
        label_index=encode_label(label),
        split=split,
        parent_id=f"parent-{sample_id}",
        segment_index=0,
        start_seconds=0.0,
        duration_seconds=DURATION,
        sample_rate=SAMPLE_RATE,
        speech_seconds=DURATION * coverage,
        coverage=coverage,
        is_padded=is_padded,
        waveform_samples=int(DURATION * SAMPLE_RATE),
        speaker_id=f"spk-{sample_id}",
        generator_id="melgan",
        language="en",
        codec="pcm_s16le",
        session_id="s1",
        attack_type="unknown",
        channel="mono",
        device="unknown",
        recorded_at="unknown",
        file_hash=f"sha256:file-{sample_id}",
        content_hash=f"sha256:content-{sample_id}",
        source_split="unknown",
        preprocessing_version="v1",
        dataset_build_id=BUILD,
    )


def write_audio(root, sample_id: str) -> None:
    """Write the WAV a row points at, so decoding has something to read."""
    target = root / RELATIVE / f"{sample_id}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(DURATION * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    audio = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    sf.write(target, audio, SAMPLE_RATE, subtype="PCM_16")


@pytest.fixture
def corpus(tmp_path):
    """A written, two-class train/dev corpus plus its manifest set."""
    root = tmp_path / "data"
    paths = DataPaths.resolve(root)
    rows = [
        segment("train_a", split="train", label=BONA_FIDE, coverage=0.9, is_padded=False),
        segment("train_b", split="train", label=SPOOF, coverage=0.4, is_padded=True),
        segment("dev_a", split="dev", label=BONA_FIDE, coverage=0.8, is_padded=False),
        segment("test_a", split="test", label=SPOOF, coverage=0.7, is_padded=False),
    ]
    for row in rows:
        write_audio(root, row.sample_id)
    written = write_manifest_set(rows, paths)
    assert set(written) >= {"all", "train", "dev", "test"}
    return paths


class TestEmptyManifestsAreRefused:
    """An empty split is a configuration that measured nothing, not a result."""

    def test_the_writer_refuses_an_empty_split(self, corpus) -> None:
        rows = list(read_manifest(corpus.manifests / "all.jsonl").samples)
        trimmed = [row for row in rows if row.split != "dev"]
        with pytest.raises(ManifestError, match="empty dev"):
            write_manifest_set(trimmed, corpus, overwrite=True)


class TestResolveSplit:
    @pytest.mark.parametrize("split", ["train", "dev", "test"])
    def test_known_splits_are_accepted(self, split: str) -> None:
        assert resolve_split(split) == split

    def test_an_unknown_split_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="bogus"):
            resolve_split("bogus")

    def test_all_is_not_a_trainable_split(self) -> None:
        """``all.jsonl`` exists as a report, not as something to iterate."""
        with pytest.raises(ValueError, match="all"):
            resolve_split("all")


class TestRowSelection:
    def test_only_the_named_split_is_read(self, corpus) -> None:
        dataset = load_split("train", paths=corpus)
        assert [row.sample_id for row in dataset.rows] == ["train_a", "train_b"]

    def test_a_coverage_floor_drops_quiet_windows(self, corpus) -> None:
        dataset = VoxShieldDataset(corpus.manifests / "train.jsonl", split="train", paths=corpus, min_coverage=0.5)
        assert [row.sample_id for row in dataset.rows] == ["train_a"]

    def test_padded_windows_can_be_dropped(self, corpus) -> None:
        """A padded window is real audio plus silence; sometimes that tail hurts."""
        dataset = VoxShieldDataset(
            corpus.manifests / "train.jsonl", split="train", paths=corpus, include_padded=False
        )
        assert [row.sample_id for row in dataset.rows] == ["train_a"]

    def test_both_guards_compose(self, corpus) -> None:
        dataset = VoxShieldDataset(
            corpus.manifests / "train.jsonl",
            split="train",
            paths=corpus,
            min_coverage=0.5,
            include_padded=False,
        )
        assert len(dataset) == 1

    def test_repr_reports_the_row_count(self, corpus) -> None:
        dataset = load_split("train", paths=corpus)
        assert "rows=2" in repr(dataset)


class TestAugmentationIsTrainOnly:
    def test_dev_never_augments(self, corpus) -> None:
        assert load_split("dev", paths=corpus).augmentation is None


class TestDecoding:
    def test_one_item_is_a_waveform_and_a_label(self, corpus) -> None:
        sample = load_split("train", paths=corpus)[0]
        assert tuple(sample["waveform"].shape) == (1, int(DURATION * SAMPLE_RATE))
        assert int(sample["label"]) == encode_label(BONA_FIDE)
        assert sample["sample_id"] == "train_a"

    def test_a_truncated_file_fails_loudly(self, corpus) -> None:
        """Silently training on less audio than the manifest describes is worse than failing."""
        (corpus.root / RELATIVE / "train_a.wav").write_bytes(b"not a wav file")
        with pytest.raises(AudioDecodeError):
            load_split("train", paths=corpus)[0]

    def test_a_length_mismatch_is_rejected(self, corpus) -> None:
        target = corpus.root / RELATIVE / "train_a.wav"
        sf.write(target, np.zeros(100, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")
        with pytest.raises(AudioDecodeError):
            read_segment(target, expected_samples=int(DURATION * SAMPLE_RATE))


class TestCollate:
    def test_a_batch_stacks_into_one_tensor(self, corpus) -> None:
        dataset = load_split("train", paths=corpus)
        batch = collate_samples([dataset[0], dataset[1]])
        assert tuple(batch["waveform"].shape) == (2, 1, int(DURATION * SAMPLE_RATE))
        assert list(batch["sample_id"]) == ["train_a", "train_b"]

    def test_labels_stack_as_vectors(self, corpus) -> None:
        dataset = load_split("train", paths=corpus)
        batch = collate_samples([dataset[0], dataset[1]])
        assert batch["label"].tolist() == [encode_label(BONA_FIDE), encode_label(SPOOF)]


class TestDataLoader:
    def test_every_row_is_yielded_exactly_once(self, corpus) -> None:
        dataset = load_split("train", paths=corpus)
        loader = build_dataloader(dataset, batch_size=1, shuffle=False)
        seen = [str(sample_id) for batch in loader for sample_id in batch["sample_id"]]
        assert sorted(seen) == ["train_a", "train_b"]

    def test_batches_are_rectangular(self, corpus) -> None:
        """Every window is zero-filled to the configured width, so the tail is not ragged."""
        dataset = load_split("train", paths=corpus)
        (batch,) = list(build_dataloader(dataset, batch_size=2, shuffle=False))
        assert tuple(batch["waveform"].shape)[:2] == (2, 1)

    def test_a_zero_batch_size_is_rejected(self, corpus) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            build_dataloader(load_split("train", paths=corpus), batch_size=0)

    def test_evaluation_order_is_stable_across_runs(self, corpus) -> None:
        """Two evaluation runs must agree on order, or their numbers will not compare."""
        first = [
            str(i)
            for batch in build_dataloader(load_split("train", paths=corpus), batch_size=1, seed=7)
            for i in batch["sample_id"]
        ]
        second = [
            str(i)
            for batch in build_dataloader(load_split("train", paths=corpus), batch_size=1, seed=7)
            for i in batch["sample_id"]
        ]
        assert first == second

    def test_a_weighted_sampler_draws_every_row(self, corpus) -> None:
        dataset = load_split("train", paths=corpus)
        loader = build_dataloader(dataset, batch_size=1, class_balance="weighted_sampler", seed=3)
        seen = [str(sample_id) for batch in loader for sample_id in batch["sample_id"]]
        assert sorted(seen) == ["train_a", "train_b"]

    def test_workers_read_the_same_rows(self, corpus) -> None:
        """num_workers is a throughput trade; it must not change what is read."""
        dataset = load_split("train", paths=corpus)
        inline = sorted(
            str(i)
            for batch in build_dataloader(dataset, batch_size=1, shuffle=False)
            for i in batch["sample_id"]
        )
        forked = sorted(
            str(i)
            for batch in build_dataloader(dataset, batch_size=1, shuffle=False, num_workers=2)
            for i in batch["sample_id"]
        )
        assert inline == forked


def test_an_empty_dataset_is_refused_before_torch_sees_it(corpus) -> None:
    """Torch sampler errors name neither the split nor the filter that emptied it."""
    filtered = VoxShieldDataset(
        corpus.manifests / "train.jsonl", split="train", paths=corpus, min_coverage=2.0
    )
    with pytest.raises(ValueError, match="min_coverage"):
        build_dataloader(filtered, batch_size=2)


class TestSummariseBatches:
    def test_a_healthy_split_summarises(self, corpus) -> None:
        dataset = load_split("train", paths=corpus)
        summary = summarise_batches(dataset, build_dataloader(dataset, batch_size=2, shuffle=False))
        assert summary["ok"] is True
        assert summary["rows"] == 2
        assert summary["samples_read"] == 2
        assert summary["unique_samples_seen"] == 2
        assert summary["labels_present"] == sorted([encode_label(BONA_FIDE), encode_label(SPOOF)])

    def test_reading_is_bounded(self, corpus) -> None:
        dataset = load_split("train", paths=corpus)
        summary = summarise_batches(
            dataset, build_dataloader(dataset, batch_size=1, shuffle=False), max_batches=1
        )
        assert summary["batches_read"] == 1
        assert summary["samples_read"] == 1


class TestLoadSplit:
    def test_it_resolves_the_manifest_for_you(self, corpus) -> None:
        assert len(load_split("dev", paths=corpus)) == 1

    def test_an_explicit_manifest_wins(self, corpus) -> None:
        other = corpus.manifests / "all.jsonl"
        dataset = load_split("train", paths=corpus, manifest=other)
        assert len(dataset) == 2