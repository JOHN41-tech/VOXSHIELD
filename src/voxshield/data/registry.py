"""The dataset registry: which corpora this project knows about, and where.

A registry that only maps a name to a class is a lookup table. This one also
answers the question a build actually has to answer first: *given this
configuration, which entries are usable, and why not the others?* That question
comes up on every run, on machines that hold different subsets of the corpora, and
a registry that raised on the first missing directory would make the pipeline
unusable everywhere except on the machine that happened to be used to write it.

So the registry separates three concerns that are easy to conflate:

* **Resolution** -- ``adapter:`` names resolve to adapter classes. An unknown name
  is a configuration error and raises, because there is no sensible fallback.
* **Availability** -- an entry whose corpus is absent is *skipped with a reason*.
  Never an error, and never silently omitted: the reason is in the report.
* **Licensing** -- governed by the configured policy, and enforced before
  availability, because a corpus whose licence is unverified should be refused
  even when its files are present and the machine would happily read them.

Every skip produces a :class:`SkipReason` rather than a bare string, so the build
report can group them and a reader can tell "not on this machine" from "licence
not verified" -- two problems with two completely different fixes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from voxshield.data.config import (
    LICENSE_POLICIES,
    DataConfig,
    DatasetEntry,
    license_status_is_admissible,
)
from voxshield.data.errors import RegistryError
from voxshield.data.schema import UNKNOWN, SourceRecord

if TYPE_CHECKING:
    from voxshield.data.adapters.base import AdapterDescription, DatasetAdapter

__all__ = [
    "ADAPTERS",
    "DatasetRegistry",
    "SkipReason",
    "adapter_class",
    "adapter_names",
    "iter_records",
]

#: Adapter name -> import path. Imported lazily by :func:`adapter_class` so that a
#: broken third-party adapter cannot make the whole package unimportable, and so
#: that adding a corpus does not cost an import cycle at module load.
ADAPTERS: dict[str, str] = {
    "asvspoof": "voxshield.data.adapters.asvspoof:ASVspoofAdapter",
    "wavefake": "voxshield.data.adapters.wavefake:WaveFakeAdapter",
    "real_speech": "voxshield.data.adapters.real_speech:RealSpeechAdapter",
}

#: Every skip reason, so callers can branch on behaviour rather than on message text.
SKIP_NOT_FOUND = "not_found"
SKIP_DISABLED = "disabled"
SKIP_LICENSE = "license"
SKIP_UNKNOWN_ADAPTER = "unknown_adapter"
SKIP_TASK = "task_mismatch"


def adapter_names() -> tuple[str, ...]:
    """Every registered adapter name, sorted."""
    return tuple(sorted(ADAPTERS))


def adapter_class(name: str) -> type[DatasetAdapter]:
    """Resolve an adapter name to its class.

    Args:
        name: The ``adapter:`` value from a :class:`DatasetEntry`.

    Returns:
        The adapter class.

    Raises:
        RegistryError: If the name is not registered. There is deliberately no
            default adapter: an entry whose adapter is misspelled should fail
            loudly at resolution time, not be read by whichever class happened to
            be first in a lookup order.
    """
    try:
        target = ADAPTERS[name]
    except KeyError:
        msg = (
            f"unknown dataset adapter {name!r}; registered adapters are "
            f"{', '.join(adapter_names())}"
        )
        raise RegistryError(msg) from None
    module_path, _, attribute = target.partition(":")
    try:
        module = import_module(module_path)
    except ImportError as exc:  # pragma: no cover - only on a broken install
        msg = f"adapter {name!r} is registered but {module_path!r} could not be imported: {exc}"
        raise RegistryError(msg) from exc
    return getattr(module, attribute)


@dataclass(frozen=True, slots=True)
class SkipReason:
    """Why one configured entry is not part of this build.

    Attributes:
        dataset_id: The entry that was skipped.
        reason: One of the ``SKIP_*`` constants.
        detail: Human-readable explanation, specific enough to act on.
    """

    dataset_id: str
    reason: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"dataset_id": self.dataset_id, "reason": self.reason, "detail": self.detail}


class DatasetRegistry:
    """Adapter resolution and per-run entry selection, for one :class:`DataConfig`.

    The registry is constructed per build rather than being a module-level
    singleton, because its answers depend on the data root and the licence policy
    of the configuration it was built for. A global registry would make two
    configurations in one process -- a test and a real build, say -- share
    decisions that were made for different inputs.
    """

    def __init__(self, config: DataConfig) -> None:
        """Validate the configuration and prepare for resolution.

        Args:
            config: The dataset build configuration.

        Raises:
            RegistryError: If the licence policy is not recognised, or if two
                entries share a ``dataset_id``. Duplicate ids are rejected here
                rather than in the manifests, because two corpora writing the same
                ids produce a manifest whose rows disagree about which corpus they
                came from -- and the speaker namespace, which disambiguates exactly
                this case, would then also be ambiguous.
        """
        if config.license_policy not in LICENSE_POLICIES:
            msg = (
                f"license_policy {config.license_policy!r} is not one of {sorted(LICENSE_POLICIES)}"
            )
            raise RegistryError(msg)
        self.config = config
        self.data_root = Path(config.root).expanduser()
        seen: dict[str, DatasetEntry] = {}
        for entry in config.datasets:
            previous = seen.get(entry.dataset_id)
            if previous is not None:
                msg = (
                    f"duplicate dataset_id {entry.dataset_id!r}: declared as both "
                    f"adapter {previous.adapter!r} and {entry.adapter!r}. dataset_id "
                    "is the speaker namespace and part of the build id, so it must "
                    "be unique."
                )
                raise RegistryError(msg)
            seen[entry.dataset_id] = entry
        self._entries = seen

    # -- entry access -------------------------------------------------------

    def entries(self) -> tuple[DatasetEntry, ...]:
        """All configured entries, in configuration order."""
        return tuple(self.config.datasets)

    def get(self, dataset_id: str) -> DatasetEntry:
        """The entry for ``dataset_id``.

        Raises:
            RegistryError: If no such entry is configured.
        """
        try:
            return self._entries[dataset_id]
        except KeyError:
            msg = (
                f"no dataset entry for {dataset_id!r}; configured entries are "
                f"{', '.join(sorted(self._entries)) or '(none)'}"
            )
            raise RegistryError(msg) from None

    def admits(self, entry: DatasetEntry) -> bool:
        """Whether the licence policy permits this entry.

        Delegates to :func:`~voxshield.data.config.license_status_is_admissible`
        rather than re-deciding the policy here. The decision is a licensing one,
        it is already written down once, and a second table in this module would be
        free to disagree with the first -- and the direction it would drift matters:
        admitting a corpus the policy meant to refuse is a licence incident, while
        refusing one it meant to admit is only an inconvenience.
        """
        return license_status_is_admissible(entry.license_status, self.config.license_policy)

    def classify(self, entry: DatasetEntry) -> SkipReason | None:
        """Why this entry cannot join the build, or ``None`` if it can.

        Order matters and is deliberate. The adapter name is resolved first
        (a misspelling is a bug and must not be reported as "not on this machine"),
        then ``enabled``, then licensing, then presence on disk. Licensing precedes
        presence so that a corpus which is both unlicensed and absent is reported as
        unlicensed -- the licence is a property of the project and would still block
        the build after someone downloads the files, whereas the absence is a
        property of the machine that goes away on the next one.
        """
        if entry.adapter not in ADAPTERS:
            return SkipReason(
                entry.dataset_id,
                SKIP_UNKNOWN_ADAPTER,
                f"adapter {entry.adapter!r} is not registered "
                f"(known: {', '.join(adapter_names())})",
            )
        if not entry.enabled:
            return SkipReason(
                entry.dataset_id, SKIP_DISABLED, "entry is disabled in the configuration"
            )
        if not self.admits(entry):
            return SkipReason(
                entry.dataset_id,
                SKIP_LICENSE,
                f"license_status {entry.license_status!r} is not admitted by policy "
                f"{self.config.license_policy!r}",
            )
        root = self.resolve_path(entry)
        if not root.is_dir():
            return SkipReason(
                entry.dataset_id,
                SKIP_NOT_FOUND,
                f"no directory at {root}",
            )
        return None

    def resolve_path(self, entry: DatasetEntry) -> Path:
        """The absolute path an entry's files live at."""
        declared = Path(entry.path).expanduser()
        return declared if declared.is_absolute() else self.data_root / declared

    # -- selection ----------------------------------------------------------

    def selected(self) -> tuple[DatasetEntry, ...]:
        """Entries usable in this build, in configuration order."""
        return tuple(entry for entry in self.config.datasets if self.classify(entry) is None)

    def skipped(self) -> tuple[SkipReason, ...]:
        """Why each excluded entry was excluded, in configuration order."""
        return tuple(
            reason for entry in self.config.datasets if (reason := self.classify(entry)) is not None
        )

    def build_adapters(self, *, respect_max_files: bool = True) -> tuple[DatasetAdapter, ...]:
        """Instantiate an adapter for every selected entry.

        Args:
            respect_max_files: Apply the config's ``max_files_per_dataset`` as each
                adapter's discovery cap. Left off for a build, on for tests and
                smoke runs, where silently processing ten thousand files is not the
                point.

        Returns:
            One adapter per selected entry, in configuration order.

        Raises:
            RegistryError: If an entry's adapter and task disagree. That is caught
                here rather than during discovery so the failure names the entry
                rather than surfacing as a per-file adapter error.
        """
        adapters: list[DatasetAdapter] = []
        for entry in self.selected():
            cls = adapter_class(entry.adapter)
            declared_task = getattr(cls, "task", UNKNOWN)
            if entry.task != declared_task:
                msg = (
                    f"dataset {entry.dataset_id!r} declares task {entry.task!r} but "
                    f"adapter {entry.adapter!r} produces {declared_task!r} records"
                )
                raise RegistryError(msg)
            adapters.append(cls(entry, self.data_root))
        return tuple(adapters)

    def discover_all(
        self, *, respect_max_files: bool = True
    ) -> tuple[tuple[SourceRecord, ...], tuple[SkipReason, ...], tuple[AdapterDescription, ...]]:
        """Discover every selected corpus in one pass.

        Returns:
            ``(records, skips, descriptions)``. Skips are returned rather than
            raised because an absent corpus is a normal condition, and descriptions
            are returned because the build report needs the per-corpus layout
            evidence regardless of whether any records were found.

        Raises:
            AdapterError: If an adapter fails. Deliberately not caught: a corpus
                that cannot be read is a build failure, whereas a corpus that is
                merely absent is not. Collapsing the two would hide a broken adapter
                behind a skip reason that says "not found".
        """
        records: list[SourceRecord] = []
        descriptions: list[AdapterDescription] = []
        cap = self.config.max_files_per_dataset if respect_max_files else None
        for adapter in self.build_adapters(respect_max_files=respect_max_files):
            descriptions.append(adapter.describe())
            records.extend(adapter.discover(max_files=cap))
        return tuple(records), self.skipped(), tuple(descriptions)

    def report(self) -> dict[str, object]:
        """A JSON-serialisable summary of what this registry would do.

        Used by ``inspect`` and written into the build report. Includes the skips,
        because a build that silently used one of three configured corpora is
        indistinguishable from one where the other two were forgotten.
        """
        return {
            "root": str(self.data_root),
            "license_policy": self.config.license_policy,
            "adapters_registered": list(adapter_names()),
            "configured": [entry.dataset_id for entry in self.config.datasets],
            "selected": [entry.dataset_id for entry in self.selected()],
            "skipped": [reason.to_dict() for reason in self.skipped()],
        }


def iter_records(adapters: Iterable[DatasetAdapter]) -> Iterator[SourceRecord]:
    """Flatten discovery across several adapters."""
    for adapter in adapters:
        yield from adapter.discover()
