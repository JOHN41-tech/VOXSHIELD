"""Corpus inventory and duplicate detection.

Every later stage -- validation, preprocessing, splitting, the manifest -- reasons
about "the samples", and each of them needs the same two answers first: which
files are actually there, and which of them are the same audio twice. This module
produces those answers once, so that no later stage has to re-walk the corpus and
no two stages can disagree about what the corpus contains.

**Why content hashing is not optional.** The duplicate that matters here is rarely
a file copied to a second name. It is the same audio reached by two independent
paths into the build: LibriSpeech genuine audio redistributed inside WaveFake's
``orig/``, a corpus bundled into two archives extracted side by side, a clip
re-uploaded under a new filename. All of those are byte-identical across a
train/test boundary, and the resulting headline number looks perfectly healthy
while measuring recall of a memorised clip. Path- or name-based deduplication
finds none of them, which is why this is content-addressed.

**Scope is global, not per-dataset.** Deliberately. The interesting leak is the
cross-corpus one, and a per-dataset dedup would find only the trivial
within-corpus copies while reporting a clean bill of health on exactly the case
that matters. Two corpora that share audio are not two corpora; they are one
corpus entered twice, and pretending otherwise produces a validation score that
does not survive contact with unseen data.

**Bounded in memory, exact in content.** Every file is read in full, in fixed-size
chunks, so a 4 GB recording costs the same resident memory as a 4 kB one. Reads
are shared between records pointing at the same path. Nothing is sampled or
truncated: a partial hash would trade a silent false duplicate for a missed one,
and a false duplicate deletes real data, so the bound is on memory and never on
fidelity.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from voxshield.data.errors import DatasetBuildError
from voxshield.data.schema import UNKNOWN, SourceRecord

__all__ = [
    "DiscoveryResult",
    "DiscoveryStats",
    "DuplicateGroup",
    "InventoryEntry",
    "UnreadableFile",
    "build_inventory",
    "discover",
    "file_hash",
]

#: Read granularity. Bounds peak memory during hashing; it has no effect on the
#: digest, since SHA-256 is defined over the concatenated byte stream.
_HASH_CHUNK_BYTES = 1 << 20

#: Fields whose presence makes one copy of a duplicated recording more useful than
#: another. Used only to break a tie, never to decide *that* something is a
#: duplicate.
_METADATA_FIELDS = (
    "speaker_id",
    "generator_id",
    "language",
    "session_id",
    "attack_type",
)


def file_hash(path: Path, *, chunk_bytes: int = _HASH_CHUNK_BYTES) -> str:
    """SHA-256 of a file's bytes, as ``"sha256:<hex>"``.

    The algorithm is named in the output because the value is stored in manifests
    and compared across builds and machines. A bare hex digest is ambiguous
    between algorithms, and an unlabelled digest later found to be MD5 cannot be
    recomputed to check the collision assumption it was chosen under.

    Args:
        path: File to digest.
        chunk_bytes: Read size. Purely a memory bound.

    Returns:
        The digest, prefixed with the algorithm name.

    Raises:
        OSError: If the file cannot be read.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _resolve(root: Path, audio_path: str) -> Path:
    """Locate a file from a record's ``audio_path``.

    ``audio_path`` is relative to the data root so a manifest stays portable. A
    corpus on a separate volume is a real deployment, so an absolute path is used
    as given rather than rejected.
    """
    candidate = Path(audio_path)
    return candidate if candidate.is_absolute() else root / candidate


@dataclass(frozen=True, slots=True)
class InventoryEntry:
    """One source file, measured.

    Attributes:
        sample_id: Namespaced id from the adapter.
        dataset_id: Owning corpus.
        audio_path: Path as recorded, relative to the data root when possible.
        file_bytes: Size on disk.
        file_hash: ``"sha256:<hex>"``, or :data:`UNKNOWN` when unreadable.
        modified_ns: Modification time, reported for staleness checks only.
    """

    sample_id: str
    dataset_id: str
    audio_path: str
    file_bytes: int
    file_hash: str
    modified_ns: int


@dataclass(frozen=True, slots=True)
class UnreadableFile:
    """A file discovery could not measure, and why.

    Kept rather than raised. A corpus with one unreadable file should still be
    reported on, and the report that diagnoses it is the one that must not be
    suppressed by the problem it exists to describe.
    """

    sample_id: str
    dataset_id: str
    audio_path: str
    reason: str


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """A set of records whose source files are byte-identical.

    Attributes:
        content_hash: The shared digest.
        kept: Sample id of the record retained.
        dropped: Sample ids removed, in deterministic order.
    """

    content_hash: str
    kept: str
    dropped: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        """Total records in the group, including the kept one."""
        return len(self.dropped) + 1


@dataclass(frozen=True, slots=True)
class DiscoveryStats:
    """Counts describing what was found, before any exclusion."""

    total_records: int = 0
    unique_files: int = 0
    duplicate_groups: int = 0
    duplicate_records: int = 0
    hashed_files: int = 0
    unreadable_files: int = 0
    per_dataset: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """The outcome of an inventory pass.

    Attributes:
        entries: One :class:`InventoryEntry` per readable, deduplicated record.
        all_entries: Every entry, including those dropped as duplicates.
        duplicates: The groups that were collapsed.
        unreadable: Files that could not be measured.
        stats: Counts.
    """

    entries: tuple[InventoryEntry, ...] = ()
    all_entries: tuple[InventoryEntry, ...] = ()
    duplicates: tuple[DuplicateGroup, ...] = ()
    unreadable: tuple[UnreadableFile, ...] = ()
    stats: DiscoveryStats = field(default_factory=DiscoveryStats)

    def kept_ids(self) -> frozenset[str]:
        """Sample ids surviving duplicate removal."""
        return frozenset(entry.sample_id for entry in self.entries)

    def dropped_ids(self) -> frozenset[str]:
        """Sample ids removed as byte-identical duplicates."""
        return frozenset(sample_id for group in self.duplicates for sample_id in group.dropped)


def _metadata_score(record: SourceRecord) -> int:
    """How much published metadata a record carries.

    Used to choose which copy of a duplicate survives. Dropping the only copy that
    names a speaker silently weakens a speaker-disjoint split, and the resulting
    corpus looks no different from one where duplication was never present. Where
    both copies are equally well described, the choice falls through to a
    deterministic tiebreak so two machines resolve the same conflict the same way.
    """
    return sum(1 for name in _METADATA_FIELDS if getattr(record, name) != UNKNOWN)


def _keep_order(record: SourceRecord) -> tuple[int, str, str]:
    """Sort key deciding which duplicate survives: most metadata, then stable."""
    return (-_metadata_score(record), record.dataset_id, record.sample_id)


def _measure(
    records: Sequence[SourceRecord],
    root: Path,
    *,
    max_file_bytes: int | None,
) -> tuple[list[InventoryEntry], list[UnreadableFile], int]:
    """Stat and hash every record, returning the hashed-file count with the rest.

    Kept private so the optimisation count does not leak into
    :func:`build_inventory`'s return type, which callers should not have to
    unpack differently depending on whether they care about the optimisation.
    """
    entries: list[InventoryEntry] = []
    unreadable: list[UnreadableFile] = []
    readable: list[tuple[SourceRecord, Path, int, int]] = []

    def unreadable_for(record: SourceRecord, reason: str) -> None:
        unreadable.append(
            UnreadableFile(
                sample_id=record.sample_id,
                dataset_id=record.dataset_id,
                audio_path=record.audio_path,
                reason=reason,
            )
        )

    for record in records:
        target = _resolve(root, record.audio_path)
        try:
            stat = target.stat()
        except OSError as exc:
            unreadable_for(record, f"cannot stat: {exc.strerror or type(exc).__name__}")
            continue
        if max_file_bytes is not None and stat.st_size > max_file_bytes:
            unreadable_for(record, f"exceeds max_file_bytes ({stat.st_size} bytes)")
            continue
        readable.append((record, target, stat.st_size, stat.st_mtime_ns))

    # One read per distinct path, so a two-record-one-file corpus costs one read
    # rather than two. Grouping by path is what makes that true, and it is worth
    # doing precisely because the common duplicate is two records pointing at the
    # same bytes: LibriSpeech genuine audio and WaveFake's ``orig/`` view of it.
    by_target: dict[Path, list[int]] = defaultdict(list)
    for position, (_record, target, _size, _mtime) in enumerate(readable):
        by_target[target].append(position)

    hashed = 0
    for target, positions in by_target.items():
        try:
            digest = file_hash(target)
        except OSError as exc:
            reason = f"cannot read: {exc.strerror or type(exc).__name__}"
            for position in positions:
                unreadable_for(readable[position][0], reason)
            continue
        hashed += 1
        for position in positions:
            record, _target, size, mtime = readable[position]
            entries.append(
                InventoryEntry(
                    sample_id=record.sample_id,
                    dataset_id=record.dataset_id,
                    audio_path=record.audio_path,
                    file_bytes=size,
                    file_hash=digest,
                    modified_ns=mtime,
                )
            )

    # Ordering is re-established by sample id because ``entries`` was appended in
    # per-path order, and discovery has to hand every later stage the adapter's
    # deterministic sequence rather than a hash-table accident.
    entries.sort(key=lambda entry: entry.sample_id)
    return entries, unreadable, hashed


def build_inventory(
    records: Iterable[SourceRecord],
    root: str | Path,
    *,
    max_file_bytes: int | None = None,
) -> tuple[list[InventoryEntry], list[UnreadableFile]]:
    """Measure every record's file: size, digest, and modification time.

    Args:
        records: Source records from the registry, in deterministic order.
        root: Data root used to resolve relative audio paths.
        max_file_bytes: Optional hard cap. A larger file is reported as unreadable
            with a reason instead of being hashed, so it cannot dominate build
            time before validation rejects it.

    Returns:
        ``(entries, unreadable)``, both ordered by ``sample_id`` for determinism.
    """
    entries, unreadable, _hashed = _measure(
        tuple(records), Path(root), max_file_bytes=max_file_bytes
    )
    return entries, unreadable


def find_duplicates(
    records: Sequence[SourceRecord],
    entries: Sequence[InventoryEntry],
) -> tuple[DuplicateGroup, ...]:
    """Collapse byte-identical source files across the whole corpus.

    Records are matched to entries by ``sample_id``, never by position. A
    positional pairing looks equivalent and is not: entries are sorted for
    deterministic output while records keep the adapter's order, and the two
    diverge on any corpus whose walk is not already alphabetical. When it
    diverges, a digest is attributed to the wrong record, and a genuine sample
    is reported as a duplicate of an unrelated one and deleted.

    Args:
        records: The source records.
        entries: Their measurements, in any order.

    Returns:
        One :class:`DuplicateGroup` per set of identical files, ordered by digest.
        An unreadable file is never grouped: it has no digest, and treating
        :data:`UNKNOWN` as a shared value would make every unreadable file look
        like a duplicate of every other one.

    Raises:
        DatasetBuildError: If two records share a ``sample_id``, or an entry
            names a record that was not supplied. Both mean the inventory and the
            records describe different corpora, and deduplicating across that gap
            would delete data on the strength of a mismatch.
    """
    records_by_id: dict[str, SourceRecord] = {}
    for record in records:
        if record.sample_id in records_by_id:
            msg = (
                f"discovery was given two records with sample_id "
                f"{record.sample_id!r}; ids must be unique for deduplication "
                "to be meaningful"
            )
            raise DatasetBuildError(msg)
        records_by_id[record.sample_id] = record

    by_hash: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        if entry.sample_id not in records_by_id:
            msg = (
                f"inventory entry {entry.sample_id!r} has no matching source "
                "record; the inventory and the records describe different corpora"
            )
            raise DatasetBuildError(msg)
        if entry.file_hash != UNKNOWN:
            by_hash[entry.file_hash].append(entry.sample_id)

    groups: list[DuplicateGroup] = []
    for digest in sorted(by_hash):
        sample_ids = by_hash[digest]
        if len(sample_ids) < 2:
            continue
        ordered = sorted(sample_ids, key=lambda sid: _keep_order(records_by_id[sid]))
        groups.append(
            DuplicateGroup(
                content_hash=digest,
                kept=ordered[0],
                dropped=tuple(sorted(ordered[1:])),
            )
        )
    return tuple(groups)


def discover(
    records: Sequence[SourceRecord],
    root: str | Path,
    *,
    max_file_bytes: int | None = None,
) -> DiscoveryResult:
    """Inventory a corpus and remove byte-identical duplicates.

    Args:
        records: Source records from the registry, in deterministic order.
        root: Data root used to resolve relative audio paths.
        max_file_bytes: Optional per-file size cap; larger files are reported
            unreadable rather than hashed.

    Returns:
        A :class:`DiscoveryResult`. ``entries`` holds the surviving records
        ordered by ``sample_id``, so the ordering of every later stage is a
        function of the ids and nothing else.
    """
    entries, unreadable, hashed = _measure(records, Path(root), max_file_bytes=max_file_bytes)
    groups = find_duplicates(records, entries)
    dropped = {sample_id for group in groups for sample_id in group.dropped}
    kept = tuple(entry for entry in entries if entry.sample_id not in dropped)

    per_dataset: dict[str, int] = defaultdict(int)
    for entry in kept:
        per_dataset[entry.dataset_id] += 1

    stats = DiscoveryStats(
        total_records=len(records),
        unique_files=len(kept),
        duplicate_groups=len(groups),
        duplicate_records=len(dropped),
        hashed_files=hashed,
        unreadable_files=len(unreadable),
        per_dataset=dict(sorted(per_dataset.items())),
    )
    return DiscoveryResult(
        entries=kept,
        all_entries=tuple(entries),
        duplicates=groups,
        unreadable=tuple(unreadable),
        stats=stats,
    )
