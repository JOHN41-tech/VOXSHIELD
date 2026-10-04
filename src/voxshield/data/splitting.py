"""Source-level split assignment, before anything acoustic is inspected.

The single decision this module exists to make well is *where the boundary goes*.
Everything downstream -- the manifest, the leakage check, the quality gates --
reads :class:`SplitAssignment` and never re-derives it, because a split that
segments can redistribute is a split that leaks.

The ordering rule, restated here because it is load-bearing:

* **The unit assigned is a group, not a file.** A group is a speaker when one is
  published, otherwise the source recording (``parent_id``). Assigning whole
  speakers to one side is what makes speaker-disjointness true by construction
  rather than by luck; ``partition_key`` on the record already expresses this.
* **A group is widened until every required axis is disjoint at once.** When
  channel- or device-disjointness is required, the unit is the *connected
  component* over every required axis, not the speaker. The naive alternative --
  grouping by ``(speaker, channel)`` -- splits one speaker's recordings across
  train and test the moment that speaker appears on two channels, which trades a
  channel leak for a speaker leak and is strictly worse. Merging instead is what
  makes all of the axes simultaneously disjoint, at the cost of coarser groups:
  two speakers sharing a channel cannot be separated, and a corpus where every
  recording is wideband cannot be split at all. Both outcomes are refusals with
  an explanation, never a quiet downgrade of the claim.
* **Unknown values join nothing.** An unpublished channel is not a private
  channel, so it neither merges groups nor claims disjointness -- the same rule
  :mod:`voxshield.data.leakage` applies when it verifies the result.
* **Segments inherit.** :mod:`voxshield.data.preprocess` refuses to run without
  this mapping, so no later step can silently move a test window into training.
* **Holdouts take precedence.** A file whose published metadata lands it in a
  ``cross_*_holdout`` goes to ``test`` before any ratio arithmetic runs, because
  reserving those files *is* the point of the holdout. A file with the metadata
  unpublished can never be "pure" for the axis (:meth:`SourceRecord.is_pure_for`
  encodes exactly that), so it cannot sneak into a holdout by pretending to be
  something it does not claim.
* **Deterministic.** A build with the same configuration, corpus, and seed
  produces the same assignment, every time.
* **Temporal is opt-in and honest.** Ordering the pool oldest-to-newest makes a
  genuinely future-looking evaluation, but only when nearly every file carries a
  capture date; below ``min_temporal_coverage`` the split quietly falls back to
  the seeded random order and says so in :attr:`SplitAssignment.notes`.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from voxshield.data.config import SplitConfig
from voxshield.data.errors import SplitError
from voxshield.data.schema import UNKNOWN, SourceRecord

__all__ = [
    "SPLIT_NAMES",
    "SplitAssignment",
    "SplitReport",
    "assign_splits",
    "build_split_report",
    "write_split_report",
]

TRAIN = "train"
DEV = "dev"
TEST = "test"

#: Ordering that gives ``train`` then ``dev`` then ``test`` meaning: the model
#: tunes on the first two and is judged -- only -- on the last.
SPLIT_NAMES = (TRAIN, DEV, TEST)

#: ``holdout config field -> SourceRecord metadata field``. The publication axis
#: a holdout set reserves.
_HOLDOUT_AXES: tuple[tuple[str, str], ...] = (
    ("cross_generator_holdout", "generator_id"),
    ("cross_codec_holdout", "codec"),
    ("cross_language_holdout", "language"),
)

#: ``config field -> SourceRecord metadata field`` for the axes that are a
#: *grouping* requirement rather than a holdout. There is no holdout set for
#: them: the whole population goes to one side of the boundary, which is what
#: "channel-disjoint" means, and a reserved subset would be a holdout instead.
_GROUP_AXES: tuple[tuple[str, str], ...] = (
    ("require_channel_disjoint", "channel"),
    ("require_device_disjoint", "device"),
)


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """One deterministic source-to-split assignment.

    Attributes:
        splits: ``source.sample_id`` to ``"train" | "dev" | "test"``.
        group_splits: Group key (``SpeakerRecord.partition_key``) to split.
        holdouts: Allowed holdout values per axis, with the sample ids assigned
            to the test split by that rule.
        config: The split configuration that produced this.
        seed: Random seed used, so the assignment is reproducible.
        notes: Decided properties worth surfacing (a fallback, a claim that
            could not be made).
    """

    splits: Mapping[str, str]
    group_splits: Mapping[str, str]
    holdouts: Mapping[str, tuple[str, ...]]
    config: SplitConfig
    seed: int
    notes: tuple[str, ...] = ()

    def split_for(self, sample_id: str) -> str:
        """The split a source (or its segments) belongs to."""
        try:
            return self.splits[sample_id]
        except KeyError:
            msg = f"no split assignment for source {sample_id!r}"
            raise SplitError(msg) from None

    def ids_for(self, split: str) -> tuple[str, ...]:
        """All source ids assigned to ``split``, sorted for stable reports."""
        if split not in SPLIT_NAMES:
            msg = f"unknown split {split!r}; expected one of {SPLIT_NAMES}"
            raise SplitError(msg)
        return tuple(sorted(k for k, value in self.splits.items() if value == split))

    def counts(self) -> dict[str, int]:
        """Source count per split, in split order."""
        return {split: len(self.ids_for(split)) for split in SPLIT_NAMES}

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for a build report and interim records."""
        return {
            "splits": dict(self.splits),
            "group_splits": dict(self.group_splits),
            "holdouts": {axis: list(values) for axis, values in self.holdouts.items()},
            "counts": self.counts(),
            "seed": self.seed,
            "notes": list(self.notes),
        }


def _temporal_seconds(record: SourceRecord) -> float | None:
    """Epoch seconds for ordering, or ``None`` when a capture date is unusable."""
    value = record.recorded_at
    if not value or value == UNKNOWN:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def _required_group_axes(config: SplitConfig) -> tuple[str, ...]:
    """Record attributes that must also be kept whole, from the configuration."""
    return tuple(field for flag, field in _GROUP_AXES if getattr(config, flag))


def _components(
    records: Sequence[SourceRecord],
    axes: Sequence[str],
) -> list[tuple[str, list[SourceRecord]]]:
    """Partition records so every axis in ``axes`` is disjoint across the result.

    Two records end up in the same component when they share a *known* value on
    the record's base :attr:`~voxshield.data.schema.SourceRecord.partition_key` or
    on any axis in ``axes``. Union-find over those shared values gives the
    coarsest partition under which no required axis spans two components, which
    is the only partition that can make them all disjoint simultaneously.

    Args:
        records: The sources to partition.
        axes: Extra record attributes that must be kept whole.

    Returns:
        ``(label, members)`` pairs sorted by label. The label is the set of
        distinguishing values the component spans, so with no extra axes it *is*
        the base partition key -- reports and existing callers see no change when
        the requirement is off.
    """
    parent: dict[str, str] = {record.sample_id: record.sample_id for record in records}

    def find(node: str) -> str:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            # Always attach the larger sample id under the smaller one, so the
            # structure does not depend on the order records were discovered in.
            low, high = sorted((left_root, right_root))
            parent[high] = low

    first_seen: dict[str, str] = {}
    for record in records:
        for value in (record.partition_key, *(getattr(record, axis, "") for axis in axes)):
            if not value or value == UNKNOWN:
                continue
            key = str(value)
            union(first_seen.setdefault(key, record.sample_id), record.sample_id)

    members: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in records:
        members[find(record.sample_id)].append(record)

    components: list[tuple[str, list[SourceRecord]]] = []
    for group in members.values():
        labels = {record.partition_key for record in group}
        for axis in axes:
            for record in group:
                value = str(getattr(record, axis, UNKNOWN) or UNKNOWN)
                if value != UNKNOWN:
                    labels.add(f"{axis}:{value}")
        components.append(("|".join(sorted(labels)), group))
    return sorted(components, key=lambda entry: entry[0])


def assign_splits(
    records: Iterable[SourceRecord],
    config: SplitConfig | None = None,
    *,
    seed: int = 0,
) -> SplitAssignment:
    """Assign every source to exactly one split, by whole groups.

    Args:
        records: Sources to assign.
        config: Split configuration. Defaults to the built-in standard.
        seed: Determinism seed for the non-temporal order.

    Raises:
        SplitError: Fewer than two groups to split; a split comes out empty; or
            fewer known training speakers than ``min_train_speakers``.
    """
    cfg = config or SplitConfig()
    recs = tuple(records)
    if not recs:
        msg = "cannot split an empty source set"
        raise SplitError(msg)

    splits: dict[str, str] = {}
    group_splits: dict[str, str] = {}
    holdout_members: dict[str, list[str]] = defaultdict(list)
    # Which holdout axis reserved each group, so the greedy fill can say plainly
    # why it declined to promote the rest of that group.
    pinned_by: dict[str, str] = {}
    notes: list[str] = []

    # Group labels are computed over the whole corpus before anything is reserved,
    # so a holdout file and a main-pool file that belong to the same group agree
    # on its key. Computing them afterwards would leave ``group_splits`` with two
    # names for one group and a report that cannot be read back.
    group_axes = _required_group_axes(cfg)
    group_of = {
        record.sample_id: label
        for label, members in _components(recs, group_axes)
        for record in members
    }

    # 1. Holdouts: published cross-generator / cross-codec / cross-language
    # files are reserved for test, ahead of any ratio arithmetic.
    for axis_field, meta_field in _HOLDOUT_AXES:
        allowed = frozenset(getattr(cfg, axis_field))
        if not allowed:
            continue
        for record in recs:
            if record.is_pure_for(meta_field, allowed):
                splits[record.sample_id] = TEST
                label = group_of[record.sample_id]
                group_splits.setdefault(label, TEST)
                if label not in pinned_by:
                    pinned_by[label] = axis_field
                holdout_members[axis_field].append(record.sample_id)

    main_pool = [record for record in recs if record.sample_id not in splits]

    if not main_pool:
        msg = (
            "every source was reserved by a cross-*_holdout and nothing remains "
            "for train/dev; review the holdout sets"
        )
        raise SplitError(msg)

    # The main pool's groups are the whole corpus's groups with the reserved
    # files removed. Restricting a partition cannot split a component, so this is
    # still the coarsest valid partition -- and reusing ``group_of`` for the keys
    # is what keeps one group from appearing under two names, with two splits.
    by_label: dict[str, list[SourceRecord]] = defaultdict(list)
    for record in main_pool:
        by_label[group_of[record.sample_id]].append(record)
    groups = sorted(by_label.items(), key=lambda entry: entry[0])
    if len(groups) < 2:
        if group_axes:
            msg = (
                f"requiring {' and '.join(group_axes)}-disjoint splits left "
                f"{len(groups)} indivisible group(s) across {len(main_pool)} source(s); "
                "a whole population is pinned to one side of the boundary, so a "
                "corpus that shares one value of that axis everywhere cannot be "
                "split at all. Disjointness here is a property of the corpus, not "
                "a partitioning choice -- add sources on other "
                f"{'/'.join(group_axes)} values, or turn the requirement off and "
                "drop the claim"
            )
        else:
            msg = (
                "corpus has only one group; a speaker-disjoint split of one speaker is "
                "one fold, not a training set, and cannot be split without leaking"
            )
        raise SplitError(msg)
    if group_axes:
        notes.append(
            f"grouped by {' and '.join(group_axes)} as well as speaker/parent, so "
            f"those axes are disjoint by construction: {len(groups)} group(s) from "
            f"{len(main_pool)} source(s)"
        )
        for axis in group_axes:
            published = {str(getattr(record, axis, UNKNOWN)) for record in main_pool}
            published.discard(UNKNOWN)
            published.discard("")
            if not published:
                notes.append(
                    f"no source publishes {axis}; {axis}-disjointness cannot be "
                    "claimed and the axis will be reported as unavailable"
                )

    # 2. Order the main pool deterministically.
    temporal = cfg.partition_by_temporal
    if temporal:
        dated = sum(1 for record in main_pool if _temporal_seconds(record) is not None)
        coverage = dated / len(main_pool)
        if coverage < cfg.min_temporal_coverage:
            temporal = False
            notes.append(
                f"partition_by_temporal requested but only {coverage:.0%} of sources "
                f"carry recorded_at (min_temporal_coverage={cfg.min_temporal_coverage:.0%}); "
                "fell back to seeded random order"
            )

    # Deterministic shuffle. A seeded PRNG is exactly what a reproducible split
    # needs; cryptographic strength would be worse here (it cannot be seeded).
    rng = random.Random(seed)  # noqa: S311
    if temporal:
        ordered = sorted(
            groups,
            key=lambda entry: (
                min(
                    (t for rec in entry[1] if (t := _temporal_seconds(rec)) is not None),
                    default=float("inf"),
                ),
                entry[0],
            ),
        )
        notes.append("ordered the main pool by recorded_at, oldest to train and newest to test")
    else:
        ordered = list(groups)
        rng.shuffle(ordered)

    # 3. Greedy group fill by remaining capacity, in split order. The index is
    # monotonic: once train is at capacity we move to dev, once dev is full we
    # move to test, and test takes the remainder -- which sums to zero over the
    # whole pool, so nothing is ever left over or duplicated.
    n_main = len(main_pool)
    train_target = round(n_main * cfg.train_ratio)
    dev_target = round(n_main * cfg.dev_ratio)
    remaining = {TRAIN: train_target, DEV: dev_target, TEST: n_main - train_target - dev_target}
    open_index = 0
    promoted: list[str] = []
    for group_key, members in ordered:
        if group_splits.get(group_key) == TEST:
            # A group holding a reserved source stays reserved. The holdout
            # promised that speaker would not train; promoting the rest of the
            # group would train on the speaker the holdout exists to exclude, so
            # the whole group joins the test split instead.
            for record in members:
                splits.setdefault(record.sample_id, TEST)
            promoted.append(group_key)
            continue
        while open_index < 2 and remaining[SPLIT_NAMES[open_index]] <= 0:
            open_index += 1
        split = SPLIT_NAMES[open_index]
        remaining[split] -= len(members)
        group_splits[group_key] = split
        for record in members:
            splits[record.sample_id] = split
    if promoted:
        axes = ", ".join(sorted({pinned_by[key] for key in promoted if key in pinned_by}))
        notes.append(
            f"{len(promoted)} group(s) contained a {axes} source and stayed in "
            "test as a whole: a holdout speaker whose other files sit in the main "
            "pool cannot be trained on in part"
        )

    # 4. Empty-split guard: an empty test split means no evaluation, an empty
    # dev split means blind model selection. Either is fatal.
    for split in SPLIT_NAMES:
        if not any(value == split for value in splits.values()):
            msg = (
                f"split {split!r} came out empty. Groups are indivisible, so a "
                "single oversized group cannot be split further; adjust the "
                "train/dev ratios or reconsider the corpus before the build "
                "silently trains or evaluates on the wrong side of the boundary."
            )
            raise SplitError(msg)

    # 5. Known-speaker floor. Only enforced when the corpus actually publishes
    # speakers: a corpus without speaker metadata cannot overclaim a speaker
    # floor, and that is reported as unavailable rather than faked.
    if any(record.speaker_id != UNKNOWN for record in recs):
        train_speakers = {
            record.speaker_id
            for record in recs
            if record.speaker_id != UNKNOWN and splits.get(record.sample_id) == TRAIN
        }
        if len(train_speakers) < cfg.min_train_speakers:
            msg = (
                f"train has {len(train_speakers)} known speaker(s), below "
                f"split.min_train_speakers={cfg.min_train_speakers}; a speaker-"
                "disjoint split of too few speakers is not a training set"
            )
            raise SplitError(msg)
    else:
        notes.append(
            "no speaker metadata is published anywhere in this corpus; the "
            "speaker-disjoint claim cannot be verified and no speaker floor applies"
        )

    holdouts = {axis: tuple(members) for axis, members in holdout_members.items()}
    return SplitAssignment(
        splits=dict(splits),
        group_splits=dict(group_splits),
        holdouts=holdouts,
        config=cfg,
        seed=seed,
        notes=tuple(notes),
    )


@dataclass(frozen=True, slots=True)
class SplitReport:
    """The assignment plus the cross-cutting counts a build report shows.

    Attributes:
        assignment: The assignment being reported.
        sources: The sources it covers, for counting.
    """

    assignment: SplitAssignment
    sources: tuple[SourceRecord, ...]

    def counts(self) -> dict[str, int]:
        return self.assignment.counts()

    def per_dataset(self) -> dict[str, dict[str, int]]:
        """Source count per split, grouped by ``dataset_id``."""
        out: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(SPLIT_NAMES, 0))
        for record in self.sources:
            out[record.dataset_id][self.assignment.split_for(record.sample_id)] += 1
        return {dataset: dict(counts) for dataset, counts in sorted(out.items())}

    def per_label(self) -> dict[str, dict[str, int]]:
        """``bona_fide`` / ``spoof`` count per split."""
        out: dict[str, dict[str, int]] = defaultdict(lambda: {"bona_fide": 0, "spoof": 0})
        for record in self.sources:
            out[self.assignment.split_for(record.sample_id)][record.label] += 1
        return {split: dict(counts) for split, counts in sorted(out.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts(),
            "assignment": self.assignment.to_dict(),
            "per_dataset": self.per_dataset(),
            "per_label": self.per_label(),
        }


def build_split_report(
    records: Iterable[SourceRecord],
    assignment: SplitAssignment,
) -> SplitReport:
    """Pair records with an assignment for reporting."""
    return SplitReport(assignment=assignment, sources=tuple(records))


def write_split_report(report: SplitReport, path: Path) -> Path:
    """Persist a split report as JSON, for ``interim/splits`` and the build report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), sort_keys=True, indent=2), encoding="utf-8")
    return path
