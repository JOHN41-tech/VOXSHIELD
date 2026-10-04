"""Quality gates: the boundary is whole-group, and the axes that must be
disjoint are, or the build refuses to continue.

The split module decides *where* the boundary goes; this module verifies that
the decision held and turns it into a go/no-go with the configured strictness
(:class:`~voxshield.data.config.GateConfig`). It does not re-split anything --
verification that can change what it verifies is not verification.

Three distinct signals are produced:

* **Boundary evaluation.** ``train -> dev``, ``train -> test`` and
  ``dev -> test`` are compared as sets of source ids. ``train -> test`` and
  ``dev -> test`` are *evaluation* boundaries, and any source on both sides
  blocks: one held-out speaker is an anecdote. ``train -> dev`` shares by
  design (the model tunes on dev), so its overlap is a report note, not a gate.
* **Axis leakage.** The identity checks from
  :mod:`voxshield.data.leakage` run on every axis. A leaked *mandatory* axis
  blocks; a leaked *optional* axis is reported as a warning because it was
  individually configured as acceptable.
* **Availability.** A mandatory axis with no known identity values cannot make
  the disjointness claim it was asked to enforce. With ``fail_on_unavailable``
  that blocks; without it, the report names exactly which claim was weakened.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from voxshield.data.config import GateConfig, SplitConfig
from voxshield.data.errors import DatasetConfigError, DatasetLeakageError
from voxshield.data.leakage import (
    AXIS_CHANNEL,
    AXIS_DEVICE,
    AXIS_FILE,
    AXIS_GENERATOR,
    AXIS_LANGUAGE,
    AXIS_PARENT,
    AXIS_SESSION,
    AXIS_SPEAKER,
    LeakageReport,
    check_leakage,
)
from voxshield.data.schema import SampleRecord, SourceRecord
from voxshield.data.splitting import DEV, TEST, TRAIN

__all__ = [
    "BoundaryStatus",
    "GateReport",
    "assert_gates_pass",
    "evaluate_leakage_gates",
]

#: GateConfig flag -> leakage axis it makes mandatory.
_REQUIRED_AXIS = (
    ("require_speaker_disjoint", AXIS_SPEAKER),
    ("require_generator_disjoint", AXIS_GENERATOR),
    ("require_session_disjoint", AXIS_SESSION),
    ("require_file_disjoint", AXIS_FILE),
    ("require_parent_disjoint", AXIS_PARENT),
    ("require_language_disjoint", AXIS_LANGUAGE),
)

#: SplitConfig flag -> leakage axis it makes mandatory. Channel and device are
#: deliberately absent from :class:`~voxshield.data.config.GateConfig`: they are
#: not verifier preferences but grouping constraints the splitter has already
#: enforced, so the axis is mandatory exactly when the split asked for it.
#: Duplicating the flag in both configs would make one claim with two sources of
#: truth, and the copy an operator forgot would silently drop the requirement.
_SPLIT_AXIS = (
    ("require_channel_disjoint", AXIS_CHANNEL),
    ("require_device_disjoint", AXIS_DEVICE),
)

#: Evaluation boundaries, in report order: the set a model is judged on.
_EVALUATION_BOUNDARIES = ((TRAIN, TEST), (DEV, TEST))


@dataclass(frozen=True, slots=True)
class BoundaryStatus:
    """Source-level overlap between one train side and one evaluation split.

    Attributes:
        train_id: The side the model was trained (or tuned) on.
        eval_id: The split being evaluated, ``"dev"`` or ``"test"``.
        n_train_sources: Source files on the train side.
        n_eval_sources: Source files on the evaluation side.
        n_shared_sources: Sources on both sides -- the leak count.
    """

    train_id: str
    eval_id: str
    n_train_sources: int
    n_eval_sources: int
    n_shared_sources: int

    @property
    def leaked(self) -> bool:
        return self.n_shared_sources > 0

    @property
    def blocks(self) -> bool:
        """Evaluation boundaries block; the tuning boundary only reports."""
        return self.eval_id == TEST and self.leaked

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_id": self.train_id,
            "eval_id": self.eval_id,
            "n_train_sources": self.n_train_sources,
            "n_eval_sources": self.n_eval_sources,
            "n_shared_sources": self.n_shared_sources,
            "leaked": self.leaked,
            "blocks": self.blocks,
        }


@dataclass(frozen=True, slots=True)
class GateReport:
    """The full go/no-go picture for one gate evaluation.

    Attributes:
        leakage: Axis-level leakage checks.
        boundaries: Evaluation-boundary statuses, in decision-relevant order.
        required_axes: Axes the configuration makes mandatory.
        test_sources: Known test-set source count.
        min_test_sources: The configured floor.
        fail_on_unavailable: Whether a mandatory axis with no known identity
            values counts as a gate failure.
        warnings: Non-fatal signals worth surfacing (a weakened claim, an
            optional axis that leaked and was configured acceptable).
    """

    leakage: LeakageReport
    boundaries: tuple[BoundaryStatus, ...]
    required_axes: tuple[str, ...]
    test_sources: int
    min_test_sources: int
    fail_on_unavailable: bool = False
    warnings: tuple[str, ...] = ()

    @property
    def failures(self) -> tuple[str, ...]:
        """Blocking gate failures, one string each, in a stable order."""
        failures: list[str] = []
        for axis in self.required_axes:
            check = self.leakage.check_for(axis)
            if check is None:
                failures.append(f"axis {axis!r} was not evaluated")
            elif check.leaked:
                failures.append(f"{check.detail}")
            elif self.fail_on_unavailable and not check.available:
                failures.append(
                    f"{axis}: no known identity values, so a disjoint corpus "
                    "cannot be claimed; failing on unavailable as configured"
                )
        for boundary in self.boundaries:
            if boundary.blocks:
                failures.append(
                    f"whole-group {boundary.eval_id} boundary "
                    f"{boundary.train_id} -> {boundary.eval_id}: "
                    f"{boundary.n_shared_sources} source(s) appear on both sides"
                )
        return tuple(failures)

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "required_axes": list(self.required_axes),
            "test_sources": self.test_sources,
            "min_test_sources": self.min_test_sources,
            "fail_on_unavailable": self.fail_on_unavailable,
            "warnings": list(self.warnings),
            "boundaries": [boundary.to_dict() for boundary in self.boundaries],
            "leakage": self.leakage.to_dict(),
        }


def _required_axes(cfg: GateConfig, split_cfg: SplitConfig | None) -> tuple[str, ...]:
    """Axes that must be disjoint, gate preference first then split grouping.

    Order is stable and append-only so an existing report's axis order does not
    change when a split-level requirement is added.
    """
    axes = [axis for flag, axis in _REQUIRED_AXIS if getattr(cfg, flag)]
    if split_cfg is not None:
        axes.extend(
            axis for flag, axis in _SPLIT_AXIS if getattr(split_cfg, flag) and axis not in axes
        )
    return tuple(axes)


def _split_of(
    records: tuple[SourceRecord | SampleRecord, ...],
    splits: Mapping[str, str] | None,
) -> dict[str, str]:
    if splits is not None:
        return dict(splits)
    out: dict[str, str] = {}
    for record in records:
        split = getattr(record, "split", None)
        if isinstance(split, str) and split:
            out[record.sample_id] = split
    return out


def _boundaries(split_of: Mapping[str, str]) -> tuple[BoundaryStatus, ...]:
    by_split: dict[str, set[str]] = defaultdict(set)
    for sample_id, split in split_of.items():
        by_split[split].add(sample_id)
    boundaries: list[BoundaryStatus] = []
    for train_id, eval_id in _EVALUATION_BOUNDARIES:
        train_set = by_split[train_id]
        eval_set = by_split[eval_id]
        boundaries.append(
            BoundaryStatus(
                train_id=train_id,
                eval_id=eval_id,
                n_train_sources=len(train_set),
                n_eval_sources=len(eval_set),
                n_shared_sources=len(train_set & eval_set),
            )
        )
    return tuple(boundaries)


def evaluate_leakage_gates(
    records: Iterable[SourceRecord | SampleRecord],
    *,
    config: GateConfig | None = None,
    split_config: SplitConfig | None = None,
    splits: Mapping[str, str] | None = None,
    file_hashes: Mapping[str, str] | None = None,
) -> GateReport:
    """Evaluate every gate the configuration makes mandatory.

    Args:
        records: Source records (with ``splits``) or sample records (which carry
            their own ``split``).
        config: Gate policy. Defaults to the built-in standard.
        split_config: The split configuration the corpus was split with. Any
            axis it groups on -- channel, device -- becomes mandatory here, so
            the gate verifies the split's own promise instead of trusting it.
        splits: Source ``sample_id`` to split, used instead of record fields.
        file_hashes: ``sample_id`` to source ``file_hash``, for the file axis.

    Raises:
        DatasetConfigError: A record has no split to evaluate.
    """
    cfg = config or GateConfig()
    recs = tuple(records)
    if not recs:
        msg = "cannot evaluate gates on an empty record set"
        raise DatasetConfigError(msg)

    split_of = _split_of(recs, splits)
    if splits is None:
        missing = sorted(record.sample_id for record in recs if record.sample_id not in split_of)
        if missing:
            msg = "records without a split cannot be gated: " + ", ".join(
                repr(item) for item in missing[:5]
            )
            raise DatasetConfigError(msg)

    required_axes = _required_axes(cfg, split_config)
    leakage = check_leakage(recs, splits=split_of, file_hashes=file_hashes)
    boundaries = _boundaries(split_of)

    test_ids = {sample_id for sample_id, split in split_of.items() if split == TEST}

    warnings: list[str] = []
    for check in leakage.checks:
        if check.axis in required_axes:
            if not check.available and not cfg.fail_on_unavailable:
                warnings.append(
                    f"{check.axis}: no known identity values; the disjoint-"
                    "corpus claim is weakened (set fail_on_unavailable to refuse)"
                )
        elif check.leaked:
            warnings.append(
                f"{check.axis}: optional axis leaked ({check.detail}); not "
                "blocking because it is configured acceptable"
            )
    for boundary in boundaries:
        if boundary.eval_id == DEV and boundary.leaked:
            warnings.append(
                f"tuning boundary {boundary.train_id} -> {boundary.eval_id}: "
                f"{boundary.n_shared_sources} source(s) overlap; expected, since "
                "dev is tuned on"
            )

    return GateReport(
        leakage=leakage,
        boundaries=boundaries,
        required_axes=required_axes,
        test_sources=len(test_ids),
        min_test_sources=cfg.min_test_sources,
        fail_on_unavailable=cfg.fail_on_unavailable,
        warnings=tuple(warnings),
    )


def assert_gates_pass(report: GateReport, *, config: GateConfig | None = None) -> None:
    """Raise unless every gate in ``report`` passes under ``config``.

    The gate policy rides on the report's *verification*, never the reverse:
    this raises when the build's claims cannot be made, and it never re-splits.
    The minimum test-set floor is a config error (a corpus problem), while
    leakage against an evaluation boundary is a ``DatasetLeakageError``.

    Raises:
        DatasetConfigError: ``gates.min_test_sources`` is not met.
        DatasetLeakageError: A mandatory gate failed or was unevaluable under
            ``fail_on_unavailable``.
    """
    cfg = config or GateConfig()
    if report.test_sources < cfg.min_test_sources:
        msg = (
            f"test split has {report.test_sources} source file(s), below "
            f"gates.min_test_sources={cfg.min_test_sources}; held-out evaluation "
            "needs enough sources to be a measurement"
        )
        raise DatasetConfigError(msg)
    if report.failures:
        lines = "\n".join(f"  - {failure}" for failure in report.failures)
        raise DatasetLeakageError(f"quality gates failed:\n{lines}")
