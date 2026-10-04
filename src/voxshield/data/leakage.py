"""Leakage checks: is any identity unit present on more than one split side?

"Leakage" here means a known value of an identity-carrying axis appearing in at
least two different splits. A speaker in both ``train`` and ``test`` makes every
downstream number optimistic -- the model has heard that voice's spoofing style
before -- and the whole point of the source-level split in
:mod:`voxshield.data.splitting` is to prevent it. This module checks that the
decision held.

Two rules keep the check honest rather than merely thorough:

* **``UNKNOWN`` is never an identity.** A file whose speaker was not published
  cannot be counted as a unique speaker, so it also cannot be counted as a
  *leaked* speaker. Pretending otherwise would manufacture a leak out of missing
  metadata. The reciprocal rule matters too: a file with unknown metadata is
  also not counted as evidence the corpus is *disjoint* -- it is excluded from
  both assertions, and whether the corpus can make any claim at all is
  surfaced as :attr:`LeakageCheck.available`.
* **The unit is the whole recording, not the file.** The parent axis keys on
  ``parent_id`` and the speaker axis on the namespaced ``speaker_id``, so if two
  files of one multi-utterance recording land on opposite sides the check sees
  them. A segment-level manifest is checked the same way -- a segment inherits
  its parent's identity, never a finer one.

The channel and device axes are population-level: disjointness there does not
mean "each file has its own channel", it means "a channel class never appears on
both sides of the boundary", which is what ``docs/dataset-manifest.md`` asks for
across narrowband, wideband, and mobile. :mod:`voxshield.data.splitting` is what
can establish it, by assigning whole populations; this module only reports
whether it held.

The file axis is special: it has two candidates per record. The source ``file
hash`` catches byte-identical copies; the segment ``content_hash`` catches the
same audio re-encoded under a new name. Either one overlapping a boundary is a
leak worth blocking, so both are checked.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from voxshield.data.errors import DatasetConfigError, DatasetLeakageError
from voxshield.data.schema import UNKNOWN, SampleRecord, SourceRecord

__all__ = [
    "AXIS_CHANNEL",
    "AXIS_DEVICE",
    "AXIS_FILE",
    "AXIS_GENERATOR",
    "AXIS_LANGUAGE",
    "AXIS_PARENT",
    "AXIS_SESSION",
    "AXIS_SPEAKER",
    "LEAKAGE_AXES",
    "LeakageCheck",
    "LeakageReport",
    "assert_no_leakage",
    "check_leakage",
]

AXIS_PARENT = "parent"
AXIS_SPEAKER = "speaker"
AXIS_FILE = "file"
AXIS_SESSION = "session"
AXIS_GENERATOR = "generator"
AXIS_LANGUAGE = "language"
AXIS_CHANNEL = "channel"
AXIS_DEVICE = "device"

#: Check order, which is also report order.
LEAKAGE_AXES = (
    AXIS_PARENT,
    AXIS_SPEAKER,
    AXIS_FILE,
    AXIS_SESSION,
    AXIS_GENERATOR,
    AXIS_LANGUAGE,
    AXIS_CHANNEL,
    AXIS_DEVICE,
)

_EXAMPLE_CAP = 8

#: ``axis -> SourceRecord/SampleRecord attribute`` carrying the identity.
_AXIS_FIELD = {
    AXIS_PARENT: "parent_id",
    AXIS_SPEAKER: "speaker_id",
    AXIS_SESSION: "session_id",
    AXIS_GENERATOR: "generator_id",
    AXIS_LANGUAGE: "language",
    AXIS_CHANNEL: "channel",
    AXIS_DEVICE: "device",
}


def _split_of(record: SourceRecord | SampleRecord, splits: Mapping[str, str] | None) -> str | None:
    """The split an identity is measured on, from record or mapping."""
    if splits is not None:
        return splits.get(record.sample_id)
    split = getattr(record, "split", None)
    if isinstance(split, str) and split:
        return split
    return None


def _is_known(value: str) -> bool:
    return bool(value) and value != UNKNOWN


def _key_values(
    record: SourceRecord | SampleRecord,
    axis: str,
    file_hashes: Mapping[str, str] | None,
) -> list[str]:
    """Known identity keys for one record on one axis.

    ``UNKNOWN`` values are filtered out here, not downstream: the number of
    *known* values drives whether the axis is available for a claim at all.
    """
    if axis == AXIS_FILE:
        out: list[str] = []
        file_hash = getattr(record, "file_hash", "") or ""
        if not file_hash and file_hashes is not None:
            file_hash = file_hashes.get(record.sample_id, "") or ""
        if _is_known(file_hash):
            out.append(f"file:{file_hash}")
        content_hash = getattr(record, "content_hash", "") or ""
        if _is_known(content_hash):
            out.append(f"content:{content_hash}")
        return out
    value = str(getattr(record, _AXIS_FIELD[axis], UNKNOWN))
    return [value] if _is_known(value) else []


@dataclass(frozen=True, slots=True)
class LeakageCheck:
    """One axis's outcome.

    Attributes:
        axis: The axis checked.
        available: At least one known identity value exists, so a disjointness
            claim can be made at all. False for a corpus whose metadata is
            entirely unknown.
        leaked: At least one known value appears on more than one split.
        known_values: Distinct known identity values observed.
        overlapping_values: The leaked values, capped for the report.
        detail: Human sentence naming the leakage.
    """

    axis: str
    available: bool
    leaked: bool
    known_values: int
    overlapping_values: tuple[str, ...] = ()
    detail: str = ""

    @property
    def passed(self) -> bool:
        return not self.leaked


def _check_axis(
    axis: str,
    records: tuple[SourceRecord | SampleRecord, ...],
    *,
    splits: Mapping[str, str] | None,
    file_hashes: Mapping[str, str] | None,
) -> LeakageCheck:
    by_value: dict[str, set[str]] = {}
    for record in records:
        split = _split_of(record, splits)
        if not split:
            raise DatasetConfigError(
                f"record {record.sample_id!r} has no split to check leakage on; "
                "supply a split assignment or check SampleRecords, which carry "
                "their split"
            )
        for key in _key_values(record, axis, file_hashes):
            by_value.setdefault(key, set()).add(split)

    overlapping = sorted((value for value, present in by_value.items() if len(present) > 1))
    detail = ""
    if overlapping:
        first = overlapping[0]
        present = sorted(by_value[first])
        detail = (
            f"{len(overlapping)} value(s) of {axis!r} appear on more than one "
            f"split; e.g. {first!r} on {', '.join(sorted(present))}"
        )
    return LeakageCheck(
        axis=axis,
        available=bool(by_value),
        leaked=bool(overlapping),
        known_values=len(by_value),
        overlapping_values=tuple(overlapping[:_EXAMPLE_CAP]),
        detail=detail,
    )


@dataclass(frozen=True, slots=True)
class LeakageReport:
    """All axes checked, plus the shape of the checked population.

    Attributes:
        checks: One :class:`LeakageCheck` per axis, in :data:`LEAKAGE_AXES` order.
        n_records: Records examined.
    """

    checks: tuple[LeakageCheck, ...]
    n_records: int

    @property
    def leaked_axes(self) -> tuple[str, ...]:
        return tuple(check.axis for check in self.checks if check.leaked)

    @property
    def has_leakage(self) -> bool:
        return any(check.leaked for check in self.checks)

    def check_for(self, axis: str) -> LeakageCheck | None:
        """The check for ``axis``, or ``None`` if it was never run."""
        for check in self.checks:
            if check.axis == axis:
                return check
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form for a build report."""
        return {
            "n_records": self.n_records,
            "has_leakage": self.has_leakage,
            "leaked_axes": list(self.leaked_axes),
            "checks": [
                {
                    "axis": check.axis,
                    "available": check.available,
                    "leaked": check.leaked,
                    "known_values": check.known_values,
                    "overlapping_values": list(check.overlapping_values),
                    "detail": check.detail,
                }
                for check in self.checks
            ],
        }


def check_leakage(
    records: Iterable[SourceRecord | SampleRecord],
    *,
    splits: Mapping[str, str] | None = None,
    file_hashes: Mapping[str, str] | None = None,
) -> LeakageReport:
    """Check every identity axis for cross-split overlap.

    Args:
        records: Source records (with ``splits``) or sample records (which carry
            their own ``split``).
        splits: Source ``sample_id`` to split, required when checking source
            records.
        file_hashes: ``sample_id`` to source ``file_hash``, for the file axis on
            records that do not carry one.

    Raises:
        DatasetConfigError: A record has no split to check against.
    """
    recs = tuple(records)
    return LeakageReport(
        checks=tuple(
            _check_axis(axis, recs, splits=splits, file_hashes=file_hashes) for axis in LEAKAGE_AXES
        ),
        n_records=len(recs),
    )


def assert_no_leakage(
    report: LeakageReport,
    *,
    required: tuple[str, ...] = (AXIS_PARENT, AXIS_FILE),
    fail_on_unavailable: bool = False,
) -> None:
    """Raise when a required axis leaks or is unevaluable under the policy.

    Args:
        report: A :class:`LeakageReport`.
        required: Axes whose leak blocks. Other axes' leaks are reported but do
            not raise -- callers decide whether to block on optional axes.
        fail_on_unavailable: Also treat a mandatory axis with no known identity
            values as a failure (the claim cannot be made).

    Raises:
        DatasetLeakageError: A required axis leaked, or was unevaluable while
            ``fail_on_unavailable`` is set.
    """
    failures: list[str] = []
    for axis in required:
        check = report.check_for(axis)
        if check is None:
            failures.append(f"axis {axis!r} was not checked")
            continue
        if check.leaked:
            failures.append(f"{axis}: {check.detail}")
        elif fail_on_unavailable and not check.available:
            failures.append(
                f"{axis}: no known identity values, so a disjoint corpus cannot "
                "be claimed; failing on unavailable as configured"
            )
    if failures:
        lines = "\n".join(f"  - {failure}" for failure in failures)
        raise DatasetLeakageError(f"leakage gates failed:\n{lines}")
