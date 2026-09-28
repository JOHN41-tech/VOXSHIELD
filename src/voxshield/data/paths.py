"""On-disk layout for the data tree.

Every path the pipeline reads or writes is derived here, from one root, so that
"where did this build put its segments" has exactly one answer and relocating the
tree is a configuration change rather than a code change.

The directory names are the vocabulary the project already uses
(``data/raw``, ``data/processed``, ``data/manifests``, ``data/synthetic``). The
two that are new here, ``interim`` and ``cache``, are separated on purpose:
``interim`` holds per-file ingestion output that is meaningful for one build and
meaningless later, while ``cache`` holds results keyed by content hash and
preprocessing version so a second build of unchanged audio does not re-decode it.
Merging them would make "delete the cache" also mean "delete the records of what
was ingested".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from voxshield.data.errors import DatasetConfigError

__all__ = ["DEFAULT_PATHS", "DataPaths"]

#: Directory names under the data root, in creation order.
DEFAULT_PATHS: dict[str, str] = {
    "raw": "raw",
    "interim": "interim",
    "processed": "processed",
    "manifests": "manifests",
    "synthetic": "synthetic",
    "cache": "cache",
    "reports": "reports",
}

#: Every directory is created even when empty, so a fresh clone has the tree the
#: documentation describes instead of a set of paths that appear only after the
#: first successful run.
_ALL_DIRS: tuple[str, ...] = tuple(DEFAULT_PATHS)


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Resolved locations for one build.

    Attributes:
        root: The data root, e.g. ``data``.
        raw: Untouched source corpora. VoxShield never writes here.
        interim: Per-build ingestion records. Disposable.
        processed: Standardised analysis segments produced by Phase 1.
        manifests: Versioned JSONL manifests. Committed.
        synthetic: Generated speech used for negative controls. Read-only here.
        cache: Content-addressed preprocessing results. Disposable.
        reports: JSON reports and statistics. Committed.
    """

    root: Path
    raw: Path
    interim: Path
    processed: Path
    manifests: Path
    synthetic: Path
    cache: Path
    reports: Path

    @classmethod
    def resolve(
        cls,
        root: str | Path,
        overrides: dict[str, str] | None = None,
    ) -> DataPaths:
        """Build a path set from a root and optional per-directory overrides.

        Args:
            root: The data root. Created if missing by :meth:`ensure`.
            overrides: Per-directory name replacements, e.g.
                ``{"processed": "processed_v2"}``. Unknown keys are rejected:
                a silently ignored override would leave the build writing to a
                directory the operator believes it is not using.

        Returns:
            A :class:`DataPaths`. Directories are *not* created here.

        Raises:
            DatasetConfigError: If ``overrides`` names a directory that does not
                exist.
        """
        base = Path(root)
        names = dict(DEFAULT_PATHS)
        if overrides:
            unknown = sorted(set(overrides) - set(DEFAULT_PATHS))
            if unknown:
                msg = (
                    f"unknown data path override(s) {unknown}; "
                    f"expected any of {sorted(DEFAULT_PATHS)}"
                )
                raise DatasetConfigError(msg)
            names.update(overrides)

        resolved = {key: base / value for key, value in names.items()}
        return cls(
            root=base,
            raw=resolved["raw"],
            interim=resolved["interim"],
            processed=resolved["processed"],
            manifests=resolved["manifests"],
            synthetic=resolved["synthetic"],
            cache=resolved["cache"],
            reports=resolved["reports"],
        )

    def ensure(self) -> DataPaths:
        """Create every directory in the set if it does not exist."""
        for path in (
            self.root,
            self.raw,
            self.interim,
            self.processed,
            self.manifests,
            self.synthetic,
            self.cache,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def segments_dir(self, dataset_id: str) -> Path:
        """Where standardised segments for one dataset live."""
        return self.processed / "segments" / dataset_id

    def rejected_path(self) -> Path:
        """JSONL record of every sample that did not survive the build."""
        return self.reports / "rejected_samples.jsonl"

    def inventory_path(self) -> Path:
        return self.reports / "dataset_inventory.json"

    def validation_path(self) -> Path:
        return self.reports / "validation_report.json"

    def statistics_path(self) -> Path:
        return self.reports / "dataset_statistics.json"

    def split_report_path(self) -> Path:
        return self.reports / "split_report.json"

    def quality_path(self) -> Path:
        return self.reports / "quality_report.json"

    def leakage_path(self) -> Path:
        return self.reports / "leakage_report.json"

    def build_report_path(self) -> Path:
        return self.reports / "build_report.json"

    def __str__(self) -> str:
        return str(self.root)
