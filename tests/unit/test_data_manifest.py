"""Unit tests for the versioned dataset manifest.

The manifest is the only artefact a training run reads, so the tests concentrate
on the ways it can lie: rows that disagree with the file they sit in, a corpus
whose statistics are computed from retired rows, a header that survives its
contents, and a rewrite that quietly changes which dataset a result cites.

Reproducibility is treated as a correctness property rather than a nicety. Two
builds of the same corpus must produce byte-identical files, or a manifest
cannot be reviewed in a diff and "did anything change?" has no cheap answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxshield.data.config import LICENSE_VERIFIED, DataConfig, DatasetEntry
from voxshield.data.errors import ManifestError
from voxshield.data.labels import BONA_FIDE, SPOOF, encode_label
from voxshield.data.manifest import (
    MANIFEST_SCHEMA_VERSION,
    Manifest,
    ManifestEntry,
    compute_statistics,
    dataset_build_id,
    evaluation_view_names,
    read_manifest,
    retire_samples,
    write_manifest,
    write_manifest_set,
)
from voxshield.data.paths import DataPaths
from voxshield.data.schema import UNKNOWN, SampleRecord

FIXED_TIME = "2026-09-29T12:00:00+00:00"
BUILD = "vs-abc123"


def segment(
    sample_id: str,
    *,
    split: str = "train",
    label: str = BONA_FIDE,
    parent_id: str | None = None,
    content_hash: str | None = None,
    dataset_build_id: str = BUILD,
    speaker_id: str = UNKNOWN,
    generator_id: str = UNKNOWN,
    language: str = UNKNOWN,
    codec: str = UNKNOWN,
    duration: float = 4.0,
    speech: float = 3.0,
    coverage: float = 0.75,
    is_padded: bool = False,
) -> SampleRecord:
    """A segment record with only the fields a manifest row actually varies."""
    return SampleRecord(
        sample_id=sample_id,
        dataset_id="corpus",
        audio_path=f"processed/segments/corpus/{sample_id}.wav",
        label=label,
        label_index=encode_label(label),
        split=split,
        parent_id=parent_id or f"parent-{sample_id}",
        segment_index=0,
        start_seconds=0.0,
        duration_seconds=duration,
        sample_rate=16_000,
        speech_seconds=speech,
        coverage=coverage,
        is_padded=is_padded,
        waveform_samples=int(duration * 16_000),
        speaker_id=speaker_id,
        generator_id=generator_id,
        language=language,
        codec=codec,
        session_id=UNKNOWN,
        attack_type=UNKNOWN,
        channel=UNKNOWN,
        device=UNKNOWN,
        recorded_at=UNKNOWN,
        file_hash=f"sha256:file-{sample_id}",
        content_hash=content_hash if content_hash is not None else f"sha256:content-{sample_id}",
        source_split=UNKNOWN,
        preprocessing_version="v1",
        dataset_build_id=dataset_build_id,
    )


def config_for(root: str = "data", **overrides: object) -> DataConfig:
    """A minimal valid build configuration.

    ``DataConfig`` refuses an empty dataset list, because a build with no
    corpora is a configuration mistake rather than an empty run.
    """
    return DataConfig(
        root=root,
        datasets=(
            DatasetEntry(
                dataset_id="corpus",
                name="Example Corpus",
                source="local",
                version="1.0",
                license="CC-BY-4.0",
                license_status=LICENSE_VERIFIED,
                license_verified_by="fixture",
                task="spoof_detection",
                enabled=True,
                path="raw/example",
                adapter="real_speech",
            ),
        ),
        **overrides,  # type: ignore[arg-type]
    )


def corpus() -> list[SampleRecord]:
    """Three splits, two labels, two corpora, one known generator."""
    return [
        segment("a1", split="train", label=BONA_FIDE, speaker_id="corpus:s1", generator_id="g1"),
        segment("a2", split="train", label=SPOOF, speaker_id="corpus:s1", generator_id="g1"),
        segment("b1", split="dev", label=BONA_FIDE, speaker_id="corpus:s2", generator_id="g2"),
        segment("c1", split="test", label=BONA_FIDE, speaker_id="corpus:s3", generator_id="g3"),
    ]


# -- round trip --------------------------------------------------------------


def test_round_trip_preserves_every_field(tmp_path: Path) -> None:
    original = corpus()
    path = tmp_path / "all.jsonl"
    write_manifest(original, path, split="all", created_at=FIXED_TIME)

    manifest = read_manifest(path)

    assert manifest.header.schema_version == MANIFEST_SCHEMA_VERSION
    assert manifest.header.created_at == FIXED_TIME
    assert [row.to_dict() for row in manifest.samples] == [
        row.to_dict() for row in sorted(original, key=lambda r: r.sample_id)
    ]


def test_header_is_first_line_and_declares_the_build(tmp_path: Path) -> None:
    path = tmp_path / "all.jsonl"
    write_manifest(
        corpus(),
        path,
        split="all",
        dataset_build_id=BUILD,
        config_hash="cfg-hash",
        content_fingerprint="fingerprint",
        created_at=FIXED_TIME,
    )

    first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert first["record_type"] == "manifest_header"
    assert first["dataset_build_id"] == BUILD
    assert first["config_hash"] == "cfg-hash"
    assert first["content_fingerprint"] == "fingerprint"
    assert first["n_rows"] == 4
    assert first["n_active"] == 4


def test_header_records_corpus_provenance(tmp_path: Path) -> None:
    path = tmp_path / "all.jsonl"
    write_manifest(
        corpus(),
        path,
        split="all",
        created_at=FIXED_TIME,
        datasets={
            "corpus": {
                "name": "Example Corpus",
                "license": "CC-BY-4.0",
                "permits_training": True,
            }
        },
    )

    manifest = read_manifest(path)

    assert manifest.header.datasets["corpus"]["license"] == "CC-BY-4.0"
    assert manifest.header.datasets["corpus"]["permits_training"] is True


# -- reproducibility ---------------------------------------------------------


def test_row_order_does_not_affect_the_file(tmp_path: Path) -> None:
    rows = corpus()
    forwards = tmp_path / "a.jsonl"
    backwards = tmp_path / "b.jsonl"

    write_manifest(rows, forwards, split="all", created_at=FIXED_TIME)
    write_manifest(list(reversed(rows)), backwards, split="all", created_at=FIXED_TIME)

    assert forwards.read_bytes() == backwards.read_bytes()


def test_same_corpus_and_time_is_byte_reproducible(tmp_path: Path) -> None:
    first = tmp_path / "one.jsonl"
    second = tmp_path / "two.jsonl"

    write_manifest(corpus(), first, split="all", created_at=FIXED_TIME)
    write_manifest(corpus(), second, split="all", created_at=FIXED_TIME)

    assert first.read_bytes() == second.read_bytes()


# -- refusing to lie ---------------------------------------------------------


def test_empty_manifest_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="empty"):
        write_manifest([], tmp_path / "all.jsonl", split="all")


def test_duplicate_sample_id_is_refused(tmp_path: Path) -> None:
    rows = [segment("dup"), segment("dup", parent_id="other")]

    with pytest.raises(ManifestError, match="duplicate sample_id"):
        write_manifest(rows, tmp_path / "all.jsonl", split="all")


def test_split_manifest_refuses_rows_from_another_split(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="assigned to"):
        write_manifest([segment("a1", split="test")], tmp_path / "train.jsonl", split="train")


def test_all_manifest_permits_every_split(tmp_path: Path) -> None:
    path = tmp_path / "all.jsonl"
    write_manifest(corpus(), path, split="all")

    assert len(read_manifest(path).samples) == 4


def test_content_hash_shared_across_splits_is_refused(tmp_path: Path) -> None:
    rows = [
        segment("a1", split="train", content_hash="sha256:same"),
        segment("c1", split="test", content_hash="sha256:same"),
    ]

    with pytest.raises(ManifestError, match="content hash"):
        write_manifest(rows, tmp_path / "all.jsonl", split="all")


def test_duplicate_content_within_one_split_is_allowed_and_reported(tmp_path: Path) -> None:
    """Same audio filed twice in one split is hygiene, not a leak."""
    rows = [
        segment("a1", split="train", content_hash="sha256:same"),
        segment("a2", split="train", content_hash="sha256:same"),
    ]
    path = tmp_path / "train.jsonl"

    write_manifest(rows, path, split="train")

    statistics = compute_statistics(read_manifest(path).entries, dataset_build_id=BUILD)
    assert statistics.duplicate_content_hashes_within_split == 1
    assert statistics.cross_split_content_duplicates == 0


def test_manifest_mixing_two_builds_is_refused(tmp_path: Path) -> None:
    rows = [segment("a1", dataset_build_id=BUILD), segment("a2", dataset_build_id="vs-other")]

    with pytest.raises(ManifestError, match="mixes dataset builds"):
        write_manifest(rows, tmp_path / "all.jsonl", split="all")


def test_manifest_of_only_retired_rows_is_refused(tmp_path: Path) -> None:
    entries = [ManifestEntry(segment("a1"), removed_reason="licence", removed_at=FIXED_TIME)]

    with pytest.raises(ManifestError, match="retired"):
        write_manifest(entries, tmp_path / "all.jsonl", split="all")


# -- overwrite protection ----------------------------------------------------


def test_replacing_a_manifest_from_another_build_needs_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "all.jsonl"
    write_manifest(corpus(), path, split="all", dataset_build_id=BUILD, created_at=FIXED_TIME)
    others = [segment("z1", dataset_build_id="vs-other")]

    with pytest.raises(ManifestError, match="overwrite=True"):
        write_manifest(others, path, split="all", dataset_build_id="vs-other")

    write_manifest(others, path, split="all", dataset_build_id="vs-other", overwrite=True)
    assert read_manifest(path).header.dataset_build_id == "vs-other"


def test_rewriting_the_same_build_needs_no_flag(tmp_path: Path) -> None:
    """Re-running an interrupted build must not require an operator decision."""
    path = tmp_path / "all.jsonl"
    write_manifest(corpus(), path, split="all", dataset_build_id=BUILD, created_at=FIXED_TIME)

    write_manifest(corpus(), path, split="all", dataset_build_id=BUILD, created_at=FIXED_TIME)

    assert read_manifest(path).header.dataset_build_id == BUILD


def test_write_leaves_no_temporary_files(tmp_path: Path) -> None:
    path = tmp_path / "all.jsonl"
    write_manifest(corpus(), path, split="all", created_at=FIXED_TIME)

    assert [item.name for item in tmp_path.iterdir()] == ["all.jsonl"]


# -- read-side validation ----------------------------------------------------


def test_reading_a_missing_manifest_fails(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="no manifest"):
        read_manifest(tmp_path / "absent.jsonl")


def test_reading_a_file_without_a_header_fails(tmp_path: Path) -> None:
    path = tmp_path / "not-a-manifest.jsonl"
    path.write_text(json.dumps(segment("a1").to_dict()) + "\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="first line must be"):
        read_manifest(path)


def test_reading_a_newer_schema_version_fails(tmp_path: Path) -> None:
    path = tmp_path / "future.jsonl"
    header = {
        "record_type": "manifest_header",
        "schema_version": MANIFEST_SCHEMA_VERSION + 1,
        "dataset_build_id": BUILD,
        "config_hash": "",
        "content_fingerprint": "",
        "created_at": FIXED_TIME,
        "split": "all",
        "n_rows": 1,
        "n_active": 1,
    }
    path.write_text(json.dumps(header) + "\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="schema version"):
        read_manifest(path)


def test_header_count_that_disagrees_with_the_file_fails(tmp_path: Path) -> None:
    """A truncated manifest must not read as a shorter, valid one."""
    path = tmp_path / "truncated.jsonl"
    write_manifest(corpus(), path, split="all", created_at=FIXED_TIME)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="truncated or edited"):
        read_manifest(path)


def test_row_missing_a_required_field_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    write_manifest([segment("a1")], path, split="all", created_at=FIXED_TIME)
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[1])
    del row["label_index"]
    path.write_text("\n".join([lines[0], json.dumps(row)]) + "\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="malformed manifest row"):
        read_manifest(path)


# -- append-only retirement --------------------------------------------------


def test_retirement_keeps_the_row_and_annotates_it(tmp_path: Path) -> None:
    path = tmp_path / "all.jsonl"
    write_manifest(corpus(), path, split="all", created_at=FIXED_TIME)

    changed = retire_samples(path, ["a1"], "licence could not be verified", removed_at=FIXED_TIME)
    manifest = read_manifest(path)

    assert changed == 1
    assert [row.sample_id for row in manifest.samples] == ["a2", "b1", "c1"]
    retired = manifest.retired
    assert len(retired) == 1
    assert retired[0].record.sample_id == "a1"
    assert "licence" in retired[0].removed_reason
    assert retired[0].removed_at == FIXED_TIME
    # The row itself is untouched: retirement is a corpus claim, not an edit of
    # the audio's identity.
    assert retired[0].record.content_hash == "sha256:content-a1"


def test_retirement_updates_the_header_counts(tmp_path: Path) -> None:
    path = tmp_path / "all.jsonl"
    write_manifest(corpus(), path, split="all", created_at=FIXED_TIME)

    retire_samples(path, ["a1", "a2"], "superseded", removed_at=FIXED_TIME)
    header = read_manifest(path).header

    assert header.n_rows == 4
    assert header.n_active == 2


def test_retirement_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "all.jsonl"
    write_manifest(corpus(), path, split="all", created_at=FIXED_TIME)
    retire_samples(path, ["a1"], "licence", removed_at=FIXED_TIME)

    assert retire_samples(path, ["a1"], "licence again", removed_at=FIXED_TIME) == 0


def test_retirement_without_a_reason_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "all.jsonl"
    write_manifest(corpus(), path, split="all", created_at=FIXED_TIME)

    with pytest.raises(ManifestError, match="without a reason"):
        retire_samples(path, ["a1"], "   ")


def test_unknown_ids_are_ignored_by_retirement(tmp_path: Path) -> None:
    path = tmp_path / "all.jsonl"
    write_manifest(corpus(), path, split="all", created_at=FIXED_TIME)

    assert retire_samples(path, ["absent"], "typo", removed_at=FIXED_TIME) == 0


# -- statistics --------------------------------------------------------------


def test_statistics_count_only_active_rows() -> None:
    entries = [
        ManifestEntry(segment("a1")),
        ManifestEntry(segment("a2"), removed_reason="licence", removed_at=FIXED_TIME),
    ]

    statistics = compute_statistics(entries, dataset_build_id=BUILD)

    assert statistics.n_rows == 2
    assert statistics.n_active == 1
    assert statistics.n_retired == 1
    assert statistics.by_split == {"train": 1}


def test_statistics_report_the_class_ratio() -> None:
    statistics = compute_statistics(corpus(), dataset_build_id=BUILD)

    assert statistics.by_label == {BONA_FIDE: 3, SPOOF: 1}
    assert statistics.class_ratio == pytest.approx(0.75)


def test_statistics_count_speakers_and_sources() -> None:
    statistics = compute_statistics(corpus(), dataset_build_id=BUILD)

    assert statistics.n_speakers == 3
    assert statistics.n_sources == 4


def test_statistics_report_distributions() -> None:
    rows = [
        segment("a1", duration=2.0, speech=1.0, coverage=0.5),
        segment("a2", duration=6.0, speech=5.0, coverage=0.9, is_padded=True),
    ]

    statistics = compute_statistics(rows, dataset_build_id=BUILD)

    assert statistics.duration.total == pytest.approx(8.0)
    assert statistics.duration.minimum == pytest.approx(2.0)
    assert statistics.duration.maximum == pytest.approx(6.0)
    assert statistics.duration.mean == pytest.approx(4.0)
    assert statistics.coverage.minimum == pytest.approx(0.5)
    assert statistics.n_padded == 1


def test_statistics_of_an_empty_set_are_zeroed_not_undefined() -> None:
    statistics = compute_statistics([], dataset_build_id=BUILD)

    assert statistics.n_active == 0
    assert statistics.duration.count == 0
    assert statistics.class_ratio == 0.0


def test_statistics_separate_within_split_duplicates_from_leaks() -> None:
    """The two findings are measured separately and never summed.

    Two digests each repeated inside one split are hygiene (2 redundant rows);
    one digest straddling train and test is a leak (1 occurrence). Counting them
    as a single total would let a build trade one for the other.
    """
    rows = [
        segment("a1", split="train", content_hash="sha256:dup"),
        segment("a2", split="train", content_hash="sha256:dup"),
        segment("b1", split="dev", content_hash="sha256:dup2"),
        segment("b2", split="dev", content_hash="sha256:dup2"),
        segment("c1", split="test", content_hash="sha256:cross"),
        segment("d1", split="train", content_hash="sha256:cross"),
    ]

    statistics = compute_statistics(rows, dataset_build_id=BUILD)

    assert statistics.duplicate_content_hashes_within_split == 2
    assert statistics.cross_split_content_duplicates == 1


def test_statistics_ignore_rows_without_a_content_hash() -> None:
    rows = [
        segment("a1", content_hash=UNKNOWN),
        segment("a2", content_hash=UNKNOWN),
    ]

    statistics = compute_statistics(rows, dataset_build_id=BUILD)

    assert statistics.duplicate_content_hashes_within_split == 0
    assert statistics.n_distinct_content_hashes == 0


# -- evaluation views --------------------------------------------------------


def test_only_test_rows_belong_to_an_evaluation_view() -> None:
    assert (
        evaluation_view_names(
            segment("a1", split="train", generator_id="g9"), {"generator": ["g9"]}
        )
        == ()
    )


def test_a_held_out_generator_puts_a_test_row_in_the_cross_generator_view() -> None:
    row = segment("c1", split="test", generator_id="g9")

    assert evaluation_view_names(row, {"generator": ["g9"]}) == ("test_cross_generator",)


def test_an_unknown_generator_never_qualifies_as_a_holdout() -> None:
    """Claiming a cross-generator set on an unpublished generator is not one."""
    row = segment("c1", split="test", generator_id=UNKNOWN)

    assert evaluation_view_names(row, {"generator": ["g9"]}) == ()


def test_a_view_with_no_configured_holdout_is_not_claimed() -> None:
    row = segment("c1", split="test", generator_id="g9")

    assert evaluation_view_names(row, {}) == ()


def test_multiple_axes_produce_multiple_views() -> None:
    row = segment("c1", split="test", generator_id="g9", language="fr", codec="amr")

    names = evaluation_view_names(row, {"generator": ["g9"], "language": ["fr"], "codec": ["amr"]})

    assert names == ("test_cross_generator", "test_cross_language", "test_cross_codec")


# -- build id ----------------------------------------------------------------


def test_build_id_ignores_where_the_data_lives() -> None:
    """The same corpus built at two paths is one dataset, not two."""
    base = config_for("data")
    elsewhere = config_for("/srv/corpus")

    assert dataset_build_id(base) == dataset_build_id(elsewhere)


def test_build_id_changes_with_something_that_changes_the_corpus() -> None:
    base = config_for("data", window_seconds=4.0)
    other = config_for("data", window_seconds=3.0)

    assert dataset_build_id(base) != dataset_build_id(other)


def test_build_id_is_stable_across_calls() -> None:
    config = config_for()

    assert dataset_build_id(config) == dataset_build_id(config)


# -- the manifest set --------------------------------------------------------


def test_manifest_set_writes_every_split_and_statistics(tmp_path: Path) -> None:
    paths = DataPaths.resolve(tmp_path).ensure()
    config = config_for(root=str(tmp_path))

    written = write_manifest_set(corpus(), paths, config=config, created_at=FIXED_TIME)

    assert set(written) == {"all", "train", "dev", "test", "statistics"}
    assert len(read_manifest(paths.manifests / "all.jsonl")) == 4
    assert len(read_manifest(paths.manifests / "train.jsonl")) == 2
    assert len(read_manifest(paths.manifests / "dev.jsonl")) == 1
    assert len(read_manifest(paths.manifests / "test.jsonl")) == 1
    assert paths.statistics_path().is_file()


def test_manifest_set_stamps_every_file_with_the_same_build(tmp_path: Path) -> None:
    paths = DataPaths.resolve(tmp_path).ensure()
    config = config_for(root=str(tmp_path))
    build = dataset_build_id(config)

    write_manifest_set(corpus(), paths, config=config, created_at=FIXED_TIME)

    for name in ("all", "train", "dev", "test"):
        manifest = read_manifest(paths.manifests / f"{name}.jsonl")
        assert manifest.header.dataset_build_id == build
        assert {row.dataset_build_id for row in manifest.samples} == {build}


def test_manifest_set_writes_a_populated_evaluation_view(tmp_path: Path) -> None:
    paths = DataPaths.resolve(tmp_path).ensure()

    written = write_manifest_set(
        corpus(),
        paths,
        holdouts={"generator": ["g3"]},
        created_at=FIXED_TIME,
    )

    assert "test_cross_generator" in written
    view = read_manifest(paths.manifests / "test_cross_generator.jsonl")
    assert [row.sample_id for row in view.samples] == ["c1"]


def test_manifest_set_skips_an_evaluation_view_that_selects_nothing(tmp_path: Path) -> None:
    """An empty cross-axis file invites a perfect score on no measurement."""
    paths = DataPaths.resolve(tmp_path).ensure()

    written = write_manifest_set(corpus(), paths, holdouts={"generator": ["absent"]})

    assert "test_cross_generator" not in written
    assert not (paths.manifests / "test_cross_generator.jsonl").exists()


def test_manifest_statistics_cover_the_whole_corpus(tmp_path: Path) -> None:
    paths = DataPaths.resolve(tmp_path).ensure()

    write_manifest_set(corpus(), paths, created_at=FIXED_TIME)
    payload = json.loads(paths.statistics_path().read_text(encoding="utf-8"))

    assert payload["n_active"] == 4
    assert payload["by_split"] == {"dev": 1, "test": 1, "train": 2}
    assert payload["manifests"]["train"] == "train.jsonl"


def test_manifest_rows_are_identical_across_roots_but_the_run_is_not(tmp_path: Path) -> None:
    """Byte-identity of the rows, not of the header, is the reproducibility claim.

    ``config_hash`` deliberately covers ``root`` and the path names: it cites
    *this run on this machine*, and a hash that ignored where the build ran would
    hide a path difference. The corpus-affecting claim is carried by
    ``content_fingerprint`` and the build id, which must match across roots.
    """
    first = tmp_path / "one"
    second = tmp_path / "two"
    paths_a = DataPaths.resolve(first).ensure()
    paths_b = DataPaths.resolve(second).ensure()
    config = config_for(root=str(first))
    config_b = config_for(root=str(second))

    write_manifest_set(corpus(), paths_a, config=config, created_at=FIXED_TIME)
    write_manifest_set(corpus(), paths_b, config=config_b, created_at=FIXED_TIME)

    a = read_manifest(paths_a.manifests / "all.jsonl")
    b = read_manifest(paths_b.manifests / "all.jsonl")

    assert [row.to_dict() for row in a.samples] == [row.to_dict() for row in b.samples]
    assert a.header.content_fingerprint == b.header.content_fingerprint
    assert a.header.dataset_build_id == b.header.dataset_build_id
    # The run-level hashes differ, which is the point: these are two runs.
    assert a.header.config_hash != b.header.config_hash


def test_rebuilding_the_same_root_is_byte_identical(tmp_path: Path) -> None:
    first = DataPaths.resolve(tmp_path / "one").ensure()
    second = DataPaths.resolve(tmp_path / "one-copy").ensure()
    config = config_for(root=str(tmp_path / "one"))
    # Same root, so the run-level configuration is identical too.
    config_b = config_for(root=str(tmp_path / "one"))

    write_manifest_set(corpus(), first, config=config, created_at=FIXED_TIME)
    write_manifest_set(corpus(), second, config=config_b, created_at=FIXED_TIME)

    assert (first.manifests / "all.jsonl").read_bytes() == (
        second.manifests / "all.jsonl"
    ).read_bytes()


# -- the parsed view ---------------------------------------------------------


def test_manifest_view_filters_by_split(tmp_path: Path) -> None:
    path = tmp_path / "all.jsonl"
    write_manifest(corpus(), path, split="all", created_at=FIXED_TIME)

    manifest = read_manifest(path)

    assert [row.sample_id for row in manifest.by_split("test")] == ["c1"]
    assert len(manifest) == 4
    assert [row.sample_id for row in manifest] == ["a1", "a2", "b1", "c1"]


def test_manifest_view_for_split_keeps_a_consistent_header(tmp_path: Path) -> None:
    path = tmp_path / "all.jsonl"
    write_manifest(corpus(), path, split="all", created_at=FIXED_TIME)

    view = read_manifest(path).for_split("test")

    assert view.header.split == "test"
    assert view.header.n_active == 1
    assert [row.sample_id for row in view.samples] == ["c1"]


def test_manifest_is_not_empty_for_an_empty_entry_list(tmp_path: Path) -> None:
    """``Manifest`` can legitimately be empty; a file may not be."""
    path = tmp_path / "all.jsonl"
    write_manifest(corpus(), path, split="all", created_at=FIXED_TIME)

    manifest: Manifest = read_manifest(path)

    assert isinstance(manifest.entries, tuple)
    assert manifest.path == path
