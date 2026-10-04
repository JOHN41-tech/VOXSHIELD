"""Unit tests for corpus inventory and duplicate detection.

The properties tested here are the ones whose failure is invisible in a happy-path
build. A duplicate that is missed lets the same clip answer both sides of a split.
A duplicate that is *wrongly* reported is worse: it deletes real data, the corpus
still looks healthy, and nothing downstream ever learns a sample went missing.
The positional-versus-keyed bug these tests were written against is the second
kind -- a real sample declared a duplicate of an unrelated one.

Files carry deliberately distinct byte payloads rather than real audio, because
these tests are about identity of content and reading real waveforms would only
add encode/decode noise to a property that does not involve the samples.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from voxshield.data.discovery import (
    DuplicateGroup,
    InventoryEntry,
    build_inventory,
    discover,
    file_hash,
    find_duplicates,
)
from voxshield.data.errors import DatasetBuildError
from voxshield.data.schema import UNKNOWN, SourceRecord

BONA_FIDE = "bona_fide"


def write(root: Path, relative: str, payload: bytes) -> Path:
    """Create a file with exact bytes and return its path."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def record(
    sample_id: str,
    dataset_id: str,
    audio_path: str,
    *,
    speaker_id: str = UNKNOWN,
    label: str = BONA_FIDE,
) -> SourceRecord:
    """Build a source record with only the fields discovery cares about."""
    return SourceRecord(
        sample_id=sample_id,
        dataset_id=dataset_id,
        audio_path=audio_path,
        label=label,
        speaker_id=speaker_id,
    )


class TestFileHash:
    def test_labels_the_algorithm(self, tmp_path: Path) -> None:
        """A bare hex digest is ambiguous across algorithms once stored."""
        target = write(tmp_path, "a.bin", b"payload")
        assert file_hash(target) == f"sha256:{hashlib.sha256(b'payload').hexdigest()}"

    def test_is_independent_of_chunk_size(self, tmp_path: Path) -> None:
        """Chunking bounds memory and must not change the digest."""
        target = write(tmp_path, "a.bin", b"0123456789" * 500)
        assert file_hash(target, chunk_bytes=7) == file_hash(target, chunk_bytes=1 << 20)

    def test_matches_hashlib_on_empty_file(self, tmp_path: Path) -> None:
        target = write(tmp_path, "empty.bin", b"")
        assert file_hash(target) == f"sha256:{hashlib.sha256(b'').hexdigest()}"


class TestBuildInventory:
    def test_measures_size_and_digest(self, tmp_path: Path) -> None:
        write(tmp_path, "a.wav", b"abcd")
        entries, unreadable = build_inventory([record("s1", "ds", "a.wav")], tmp_path)
        assert unreadable == []
        assert len(entries) == 1
        assert entries[0].file_bytes == 4
        assert entries[0].file_hash == f"sha256:{hashlib.sha256(b'abcd').hexdigest()}"

    def test_resolves_relative_to_root(self, tmp_path: Path) -> None:
        write(tmp_path, "corpus/nested/a.wav", b"x")
        entries, _ = build_inventory([record("s1", "ds", "corpus/nested/a.wav")], tmp_path)
        assert len(entries) == 1

    def test_accepts_absolute_path(self, tmp_path: Path) -> None:
        """A corpus on another volume is a real deployment, not an error."""
        target = write(tmp_path, "external/a.wav", b"x")
        entries, unreadable = build_inventory(
            [record("s1", "ds", str(target))], tmp_path / "elsewhere"
        )
        assert unreadable == []
        assert len(entries) == 1

    def test_missing_file_is_reported_not_raised(self, tmp_path: Path) -> None:
        """One bad file must not stop the corpus from being reported on."""
        entries, unreadable = build_inventory([record("s1", "ds", "absent.wav")], tmp_path)
        assert entries == []
        assert [item.sample_id for item in unreadable] == ["s1"]
        assert "cannot stat" in unreadable[0].reason

    def test_oversized_file_is_not_hashed(self, tmp_path: Path) -> None:
        """A huge file is rejected before it can dominate build time."""
        write(tmp_path, "big.wav", b"x" * 4096)
        entries, unreadable = build_inventory(
            [record("s1", "ds", "big.wav")], tmp_path, max_file_bytes=1024
        )
        assert entries == []
        assert "exceeds max_file_bytes" in unreadable[0].reason

    def test_shared_path_is_read_once(self, tmp_path: Path) -> None:
        """Two records on one file cost one read; both get the same digest."""
        write(tmp_path, "a.wav", b"shared")
        records = [record("s1", "ds1", "a.wav"), record("s2", "ds2", "a.wav")]
        result = discover(records, tmp_path)
        assert result.stats.hashed_files == 1

    def test_output_is_ordered_by_sample_id(self, tmp_path: Path) -> None:
        for name in "abc":
            write(tmp_path, f"{name}.wav", name.encode())
        records = [record(f"s_{name}", "ds", f"{name}.wav") for name in "cba"]
        entries, _ = build_inventory(records, tmp_path)
        assert [entry.sample_id for entry in entries] == ["s_a", "s_b", "s_c"]


class TestDuplicateDetection:
    def test_detects_identical_bytes_under_different_paths(self, tmp_path: Path) -> None:
        """The real case: one recording reached through two corpora."""
        write(tmp_path, "librispeech/a.wav", b"shared-audio")
        write(tmp_path, "wavefake/orig/a.wav", b"shared-audio")
        records = [
            record("librispeech:a", "librispeech", "librispeech/a.wav"),
            record("wavefake:a", "wavefake", "wavefake/orig/a.wav"),
        ]
        result = discover(records, tmp_path)
        assert result.stats.duplicate_groups == 1
        assert len(result.kept_ids()) == 1
        # The dropped set is the *other* copy, not a subset of both. Stating it as
        # the complement of the kept set is what makes this a claim about one
        # recording entered twice; intersecting the two ids with the dropped set
        # would compare the dropped set to itself and pass whatever discovery did,
        # including deleting a genuine sample. Which copy survives is the
        # tiebreak's business, asserted separately.
        assert (
            result.dropped_ids() == frozenset({"librispeech:a", "wavefake:a"}) - result.kept_ids()
        )

    def test_scope_is_cross_dataset(self, tmp_path: Path) -> None:
        """A per-dataset dedup would miss precisely the interesting leak."""
        write(tmp_path, "one/a.wav", b"same")
        write(tmp_path, "two/a.wav", b"same")
        result = discover(
            [
                record("one:a", "one", "one/a.wav"),
                record("two:a", "two", "two/a.wav"),
            ],
            tmp_path,
        )
        # The group spans both datasets, so exactly one of the two survives and
        # the other dataset is emptied. A per-dataset dedup would have kept both.
        assert len(result.duplicates) == 1
        assert set(result.duplicates[0].dropped) | {result.duplicates[0].kept} == {
            "one:a",
            "two:a",
        }
        assert len(result.stats.per_dataset) == 1

    def test_keeps_the_copy_that_names_a_speaker(self, tmp_path: Path) -> None:
        """Dropping the only speaker-labelled copy silently weakens the split."""
        write(tmp_path, "one/a.wav", b"same")
        write(tmp_path, "two/a.wav", b"same")
        result = discover(
            [
                record("one:a", "one", "one/a.wav"),
                record("two:a", "two", "two/a.wav", speaker_id="two:spk1"),
            ],
            tmp_path,
        )
        assert result.duplicates[0].kept == "two:a"
        assert result.duplicates[0].dropped == ("one:a",)

    def test_tiebreak_is_deterministic(self, tmp_path: Path) -> None:
        """Equally-described copies must resolve identically on every machine."""
        write(tmp_path, "one/a.wav", b"same")
        write(tmp_path, "two/a.wav", b"same")
        records = [
            record("two:a", "two", "two/a.wav"),
            record("one:a", "one", "one/a.wav"),
        ]
        first = discover(records, tmp_path)
        second = discover(list(reversed(records)), tmp_path)
        assert first.duplicates == second.duplicates
        assert first.duplicates[0].kept == "one:a"

    def test_different_bytes_are_not_duplicates(self, tmp_path: Path) -> None:
        write(tmp_path, "a.wav", b"aaaa")
        write(tmp_path, "b.wav", b"aaab")
        result = discover([record("s1", "ds", "a.wav"), record("s2", "ds", "b.wav")], tmp_path)
        assert result.duplicates == ()
        assert len(result.kept_ids()) == 2

    def test_same_length_different_content_is_not_a_duplicate(self, tmp_path: Path) -> None:
        """Size prefiltering must not promote equal-length files to duplicates."""
        write(tmp_path, "a.wav", b"aaaa")
        write(tmp_path, "b.wav", b"bbbb")
        result = discover([record("s1", "ds", "a.wav"), record("s2", "ds", "b.wav")], tmp_path)
        assert result.stats.duplicate_groups == 0

    def test_three_copies_collapse_to_one(self, tmp_path: Path) -> None:
        for name in "abc":
            write(tmp_path, f"{name}.wav", b"same")
        records = [record(f"s_{name}", "ds", f"{name}.wav") for name in "abc"]
        result = discover(records, tmp_path)
        assert result.stats.duplicate_groups == 1
        assert result.duplicates[0].size == 3
        assert len(result.kept_ids()) == 1

    def test_unreadable_files_are_never_grouped(self, tmp_path: Path) -> None:
        """Treating UNKNOWN as a shared value makes every bad file a duplicate."""
        records = [record("s1", "ds", "absent1.wav"), record("s2", "ds", "absent2.wav")]
        result = discover(records, tmp_path)
        assert result.duplicates == ()
        assert result.stats.unreadable_files == 2

    def test_result_is_order_independent(self, tmp_path: Path) -> None:
        for name in "abcd":
            write(tmp_path, f"{name}.wav", b"dup" if name in "ac" else name.encode())
        records = [
            record("s_a", "ds", "a.wav"),
            record("s_b", "ds", "b.wav"),
            record("s_c", "ds", "c.wav"),
            record("s_d", "ds", "d.wav"),
        ]
        forward = discover(records, tmp_path)
        backward = discover(list(reversed(records)), tmp_path)
        assert forward.kept_ids() == backward.kept_ids()
        assert forward.duplicates == backward.duplicates
        assert forward.entries == backward.entries

    def test_repeated_runs_are_identical(self, tmp_path: Path) -> None:
        for name in "abc":
            write(tmp_path, f"{name}.wav", b"dup" if name in "ac" else name.encode())
        records = [record(f"s_{n}", "ds", f"{n}.wav") for n in "abc"]
        assert discover(records, tmp_path) == discover(records, tmp_path)

    def test_all_entries_retain_the_dropped_ones(self, tmp_path: Path) -> None:
        """A duplicate is removed from the corpus but must stay auditable."""
        write(tmp_path, "a.wav", b"same")
        write(tmp_path, "b.wav", b"same")
        result = discover([record("s1", "ds", "a.wav"), record("s2", "ds", "b.wav")], tmp_path)
        assert len(result.entries) == 1
        assert len(result.all_entries) == 2


class TestFindDuplicatesContract:
    def test_rejects_duplicate_sample_ids(self) -> None:
        """Ids must identify one record for deduplication to mean anything."""
        records = [record("s1", "ds", "a.wav"), record("s1", "ds", "b.wav")]
        with pytest.raises(DatasetBuildError, match="unique"):
            find_duplicates(records, [])

    def test_rejects_entry_without_a_record(self) -> None:
        """A mismatch means two different corpora are being described.

        The failure this guards is one-directional on purpose. An *entry* with no
        record means the inventory walked a different corpus than the records
        describe, and deduplicating across that gap attributes a digest to the
        wrong file and deletes real data. A *record* with no entry is legitimate --
        that is an unreadable file, which ``build_inventory`` reports separately --
        so it is not a mismatch and must not raise.
        """
        orphan = InventoryEntry(
            sample_id="not_in_records",
            dataset_id="ds",
            audio_path="a.wav",
            file_bytes=4,
            file_hash="sha256:deadbeef",
            modified_ns=0,
        )
        with pytest.raises(DatasetBuildError, match="no matching source record"):
            find_duplicates([record("s1", "ds", "a.wav")], [orphan])

    def test_accepts_a_record_with_no_entry(self) -> None:
        """An unreadable file yields a record with no entry, and is not a mismatch."""
        result = find_duplicates([record("s1", "ds", "a.wav")], [])
        assert result == ()

    def test_accepts_entries_in_any_order(self, tmp_path: Path) -> None:
        """Keyed matching, not positional: the regression this replaces."""
        write(tmp_path, "a.wav", b"same")
        write(tmp_path, "b.wav", b"same")
        records = [record("z", "ds", "a.wav"), record("a", "ds", "b.wav")]
        entries, _ = build_inventory(records, tmp_path)
        groups = find_duplicates(records, entries)
        assert len(groups) == 1
        assert groups[0].kept in {"a", "z"}
        assert groups[0].size == 2

    def test_group_reports_its_size(self) -> None:
        group = DuplicateGroup(content_hash="sha256:0", kept="a", dropped=("b", "c"))
        assert group.size == 3
