"""Dataset build configuration.

Every decision that changes which samples enter a corpus, and therefore every
number a later model reports, is made here and recorded in a hash. The rule this
module enforces is narrow and absolute: **a dataset build must be a function of
its configuration**. Two runs with the same config produce the same build id, the
same split, and the same manifests; a run with a changed config produces a
different build id and refuses to overwrite the old manifests. That is what makes
a reported result reproducible after the fact, when nobody remembers which
version of the config produced it.

The configuration is a frozen dataclass tree rather than a loose dict for the
usual reason: a typo in a string key is a runtime ``KeyError`` halfway through a
long build, while a typo in a field name is an import-time failure. Values are
validated in ``__post_init__`` so a contradictory configuration is rejected
before any audio is read, not after.

YAML support is optional. PyYAML is not a runtime dependency of the API, and
adding one to the service to satisfy a build tool would be a poor trade. The
loader imports it lazily and raises a clear, actionable error naming the extra
to install, while every configuration is equally constructible in Python for
tests and programmatic builds.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from voxshield.config import AudioConfig
from voxshield.data.errors import DatasetConfigError
from voxshield.data.paths import DEFAULT_PATHS, DataPaths

__all__ = [
    "AugmentationConfig",
    "CacheConfig",
    "DataConfig",
    "DatasetEntry",
    "GateConfig",
    "PathsConfig",
    "SplitConfig",
    "ValidationConfig",
    "license_status_is_admissible",
    "load_data_config",
]

#: How a dataset's licence is treated at build time. Recorded per entry, never
#: inferred from a URL.
LICENSE_VERIFIED = "VERIFIED"
LICENSE_REQUIRES_VERIFICATION = "REQUIRES_VERIFICATION"
LICENSE_RESTRICTED = "RESTRICTED"
LICENSE_UNKNOWN = "UNKNOWN"

LICENSE_STATUSES: frozenset[str] = frozenset(
    {
        LICENSE_VERIFIED,
        LICENSE_REQUIRES_VERIFICATION,
        LICENSE_RESTRICTED,
        LICENSE_UNKNOWN,
    }
)

#: Licence policies, in increasing order of permissiveness.
#:
#: ``require_verified`` is the default and is the conservative choice: a corpus
#: whose licence has not been checked is not one this project should be
#: redistributable, and a build is a redistribution of a derived artefact.
#: ``allow_unverified`` admits a corpus that is present and permitted-looking but
#: unconfirmed, and *must* be set deliberately. ``allow_all_except_restricted``
#: still refuses anything explicitly restricted.
LICENSE_POLICY_REQUIRE_VERIFIED = "require_verified"
LICENSE_POLICY_ALLOW_UNVERIFIED = "allow_unverified"
LICENSE_POLICY_ALLOW_ALL_EXCEPT_RESTRICTED = "allow_all_except_restricted"

LICENSE_POLICIES: frozenset[str] = frozenset(
    {
        LICENSE_POLICY_REQUIRE_VERIFIED,
        LICENSE_POLICY_ALLOW_UNVERIFIED,
        LICENSE_POLICY_ALLOW_ALL_EXCEPT_RESTRICTED,
    }
)

#: Class-balancing strategies for the training manifest.
BALANCE_NONE = "none"
BALANCE_WEIGHTED_SAMPLER = "weighted_sampler"

BALANCE_STRATEGIES: frozenset[str] = frozenset({BALANCE_NONE, BALANCE_WEIGHTED_SAMPLER})


def license_status_is_admissible(status: str, policy: str) -> bool:
    """Whether a licence status may enter a build under ``policy``.

    Args:
        status: One of :data:`LICENSE_STATUSES`.
        policy: One of :data:`LICENSE_POLICIES`.

    Returns:
        ``True`` if the dataset may be ingested.

    Raises:
        DatasetConfigError: If either argument is not a known value.
    """
    if status not in LICENSE_STATUSES:
        msg = f"unknown license_status {status!r}; expected one of {sorted(LICENSE_STATUSES)}"
        raise DatasetConfigError(msg)
    if policy not in LICENSE_POLICIES:
        msg = f"unknown license_policy {policy!r}; expected one of {sorted(LICENSE_POLICIES)}"
        raise DatasetConfigError(msg)

    if status == LICENSE_RESTRICTED:
        # Restricted is refused under every policy. There is no configuration in
        # which "we are not allowed to use this" becomes "we may use this".
        return False
    if policy == LICENSE_POLICY_REQUIRE_VERIFIED:
        return status == LICENSE_VERIFIED
    if policy == LICENSE_POLICY_ALLOW_ALL_EXCEPT_RESTRICTED:
        return True
    return status in (LICENSE_VERIFIED, LICENSE_REQUIRES_VERIFICATION)


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Per-directory names under the data root."""

    raw: str = DEFAULT_PATHS["raw"]
    interim: str = DEFAULT_PATHS["interim"]
    processed: str = DEFAULT_PATHS["processed"]
    manifests: str = DEFAULT_PATHS["manifests"]
    synthetic: str = DEFAULT_PATHS["synthetic"]
    cache: str = DEFAULT_PATHS["cache"]
    reports: str = DEFAULT_PATHS["reports"]

    def overrides(self) -> dict[str, str]:
        return {key: getattr(self, key) for key in DEFAULT_PATHS}

    def to_dict(self) -> dict[str, str]:
        return self.overrides()


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """How recordings are divided before segmentation.

    Attributes:
        train_ratio: Fraction of the main pool assigned to ``train``.
        dev_ratio: Fraction of the main pool assigned to ``dev``. The remainder
            is ``test``.
        min_train_speakers: A build with fewer known training speakers than this
            is refused. A speaker-disjoint split of four speakers is four folds,
            not a training set, and the failure is silent otherwise.
        cross_generator_holdout: Generator ids reserved for the cross-generator
            test. Empty means no generator holdout is configured, and the
            cross-generator manifest is then empty with the check reported as
            unavailable.
        cross_codec_holdout: Codec ids reserved for the cross-codec test.
        cross_language_holdout: Language tags reserved for the cross-language
            test.
        streaming_max_sources: Cap on the streaming test's source count. Capped
            because a streaming evaluation replays audio in arrival order and an
            unbounded one turns into an overnight job.
        partition_by_temporal: Order the main pool by capture date when every
            file has one, oldest to train and newest to test. This trades label
            balance for a genuine future-looking evaluation, so it is opt-in.
        min_temporal_coverage: Fraction of files that must carry ``recorded_at``
            for temporal ordering to be used at all.
    """

    train_ratio: float = 0.70
    dev_ratio: float = 0.15
    min_train_speakers: int = 2
    cross_generator_holdout: tuple[str, ...] = ()
    cross_codec_holdout: tuple[str, ...] = ()
    cross_language_holdout: tuple[str, ...] = ()
    streaming_max_sources: int = 512
    partition_by_temporal: bool = False
    min_temporal_coverage: float = 0.95

    def __post_init__(self) -> None:
        for name in ("train_ratio", "dev_ratio"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                msg = f"split.{name} must be within [0, 1], got {value}"
                raise DatasetConfigError(msg)
        if self.train_ratio + self.dev_ratio > 1.0:
            msg = (
                f"split.train_ratio ({self.train_ratio}) + split.dev_ratio "
                f"({self.dev_ratio}) exceeds 1.0"
            )
            raise DatasetConfigError(msg)
        if self.train_ratio <= 0.0:
            msg = "split.train_ratio must be greater than 0"
            raise DatasetConfigError(msg)
        if self.min_train_speakers < 1:
            msg = "split.min_train_speakers must be at least 1"
            raise DatasetConfigError(msg)
        if self.streaming_max_sources < 1:
            msg = "split.streaming_max_sources must be at least 1"
            raise DatasetConfigError(msg)
        if not 0.0 <= self.min_temporal_coverage <= 1.0:
            msg = "split.min_temporal_coverage must be within [0, 1]"
            raise DatasetConfigError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_ratio": self.train_ratio,
            "dev_ratio": self.dev_ratio,
            "min_train_speakers": self.min_train_speakers,
            "cross_generator_holdout": list(self.cross_generator_holdout),
            "cross_codec_holdout": list(self.cross_codec_holdout),
            "cross_language_holdout": list(self.cross_language_holdout),
            "streaming_max_sources": self.streaming_max_sources,
            "partition_by_temporal": self.partition_by_temporal,
            "min_temporal_coverage": self.min_temporal_coverage,
        }


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Which discovered files are allowed to become training samples.

    Attributes:
        require_speech: Reject a file with less speech than the Phase 1
            minimum. On by default: a training sample with no speech teaches a
            detector to key on channel noise.
        min_duration_seconds: Reject files shorter than this regardless of VAD.
        max_duration_seconds: Reject files longer than this. Also raises Phase 1's
            defensive ``max_duration_seconds`` for the duration of a build -- see
            :meth:`DataConfig.audio_config`.
        max_file_bytes: Largest source file to read. Also raises Phase 1's
            defensive ``max_upload_bytes`` for the duration of a build. The
            Phase 1 limits bound what a hostile *caller* can push at a service;
            a build reads the operator's own curated corpus, so the dataset
            policy is the right bound here. It is still a bound.
        reject_warning_issues: Reject files carrying any non-blocking quality
            issue (clipping, low SNR, DC offset). Off by default, because those
            describe a suspect recording rather than an unusable one, and
            rejecting them all biases the corpus toward studio-quality audio.
        require_metadata: Source fields that must be known. ``"label"`` is always
            required. ``speaker_id`` is deliberately *not* in the default: a
            corpus without speakers is still trainable, it just cannot make a
            speaker-disjointness claim, and that is reported as unavailable
            rather than blocked.
        reject_duplicates: Drop samples whose content hash matches an earlier
            sample, keeping the first.
        dedup_scope: ``"content"`` compares stored segment audio (catches the
            same audio re-encoded under a new name), ``"file"`` compares source
            file bytes (cheaper, catches byte-identical copies only), ``"both"``
            applies either.
    """

    require_speech: bool = True
    min_duration_seconds: float = 0.30
    max_duration_seconds: float = 60.0
    max_file_bytes: int = 64 * 1024 * 1024
    reject_warning_issues: bool = False
    require_metadata: tuple[str, ...] = ("label",)
    reject_duplicates: bool = True
    dedup_scope: str = "both"

    def __post_init__(self) -> None:
        if self.min_duration_seconds < 0.0:
            msg = "validation.min_duration_seconds cannot be negative"
            raise DatasetConfigError(msg)
        if self.max_duration_seconds <= self.min_duration_seconds:
            msg = "validation.max_duration_seconds must exceed validation.min_duration_seconds"
            raise DatasetConfigError(msg)
        if self.max_file_bytes < 1024:
            msg = "validation.max_file_bytes must be at least 1024"
            raise DatasetConfigError(msg)
        if self.dedup_scope not in ("content", "file", "both"):
            msg = (
                f"validation.dedup_scope must be 'content', 'file', or 'both', "
                f"got {self.dedup_scope!r}"
            )
            raise DatasetConfigError(msg)
        if "label" not in self.require_metadata:
            # A supervised corpus needs a label. Enforcing it here rather than
            # trusting the caller means an unlabelled file cannot be admitted by
            # forgetting to list the requirement.
            object.__setattr__(self, "require_metadata", (*self.require_metadata, "label"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "require_speech": self.require_speech,
            "min_duration_seconds": self.min_duration_seconds,
            "max_duration_seconds": self.max_duration_seconds,
            "max_file_bytes": self.max_file_bytes,
            "reject_warning_issues": self.reject_warning_issues,
            "require_metadata": list(self.require_metadata),
            "reject_duplicates": self.reject_duplicates,
            "dedup_scope": self.dedup_scope,
        }


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Robustness augmentation applied at load time, to training data only.

    Off by default and applied only to ``train``. Augmenting an evaluation set
    changes the question being answered, and a corpus split into ``train`` and
    ``test`` where ``test`` was augmented is a test set that measures the
    augmentation rather than the detector.

    Attributes:
        enabled: Master switch. ``False`` makes the loader return stored audio
            untouched regardless of the other fields.
        gain_db_min, gain_db_max: Level change range. Rides on top of Phase 1's
            loudness normalisation rather than replacing it, so the stored corpus
            stays canonical and the augmentation is reproducible from parameters.
        noise_probability: Chance of adding noise to a training window.
        noise_snr_db_min, noise_snr_db_max: SNR range for the added noise.
        noise_corpus_dir: Optional directory of real noise recordings. When
            absent, synthetic noise is used. Nothing is downloaded.
        codec_probability: Chance of a codec round-trip.
        codec_schemes: Which codec simulations may be applied. Only schemes this
            module can actually implement are listed; an unavailable scheme is
            reported rather than skipped silently.
        channel_probability: Chance of a channel-filter simulation.
        reverb_probability: Chance of a synthetic room impulse response.
        seed: Base seed. The loader derives a per-sample, per-epoch seed from it
            so a run is reproducible and an epoch still differs from the last.
        class_balance: ``"none"`` or ``"weighted_sampler"``. Never oversamples by
            duplicating rows: a duplicated manifest row would be counted twice in
            the dataset statistics and would break the duplicate check that
            guards the corpus.
    """

    enabled: bool = False
    gain_db_min: float = -6.0
    gain_db_max: float = 3.0
    noise_probability: float = 0.0
    noise_snr_db_min: float = 20.0
    noise_snr_db_max: float = 40.0
    noise_corpus_dir: str = ""
    codec_probability: float = 0.0
    codec_schemes: tuple[str, ...] = ("g711_mulaw", "g711_alaw")
    channel_probability: float = 0.0
    reverb_probability: float = 0.0
    seed: int = 0
    class_balance: str = BALANCE_NONE

    def __post_init__(self) -> None:
        if self.gain_db_min > self.gain_db_max:
            msg = "augmentation.gain_db_min cannot exceed gain_db_max"
            raise DatasetConfigError(msg)
        if not 0.0 <= self.noise_probability <= 1.0:
            msg = "augmentation.noise_probability must be within [0, 1]"
            raise DatasetConfigError(msg)
        if not 0.0 <= self.codec_probability <= 1.0:
            msg = "augmentation.codec_probability must be within [0, 1]"
            raise DatasetConfigError(msg)
        if not 0.0 <= self.channel_probability <= 1.0:
            msg = "augmentation.channel_probability must be within [0, 1]"
            raise DatasetConfigError(msg)
        if not 0.0 <= self.reverb_probability <= 1.0:
            msg = "augmentation.reverb_probability must be within [0, 1]"
            raise DatasetConfigError(msg)
        if self.noise_snr_db_min > self.noise_snr_db_max:
            msg = "augmentation.noise_snr_db_min cannot exceed noise_snr_db_max"
            raise DatasetConfigError(msg)
        if self.class_balance not in BALANCE_STRATEGIES:
            msg = (
                f"augmentation.class_balance must be one of "
                f"{sorted(BALANCE_STRATEGIES)}, got {self.class_balance!r}"
            )
            raise DatasetConfigError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "gain_db_min": self.gain_db_min,
            "gain_db_max": self.gain_db_max,
            "noise_probability": self.noise_probability,
            "noise_snr_db_min": self.noise_snr_db_min,
            "noise_snr_db_max": self.noise_snr_db_max,
            "noise_corpus_dir": self.noise_corpus_dir,
            "codec_probability": self.codec_probability,
            "codec_schemes": list(self.codec_schemes),
            "channel_probability": self.channel_probability,
            "reverb_probability": self.reverb_probability,
            "seed": self.seed,
            "class_balance": self.class_balance,
        }


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Preprocessing cache.

    Attributes:
        enabled: Reuse results for unchanged audio. The key always includes the
            source file hash, the Phase 1 preprocessing signature, and the audio
            configuration, so a stale hit is not possible by construction.
        max_entries: Soft cap. The cache is a build accelerator, and an
            unbounded one on a workstation quietly becomes the largest thing on
            the disk.
    """

    enabled: bool = True
    max_entries: int = 200_000

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            msg = "cache.max_entries must be at least 1"
            raise DatasetConfigError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "max_entries": self.max_entries}


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Which quality gates are mandatory, and what counts as a pass.

    Attributes:
        require_speaker_disjoint: Mandatory when the check is evaluable. A
            speaker appearing on both sides of a boundary makes every downstream
            number optimistic, so this is on by default.
        require_generator_disjoint: Off by default. ASVspoof 2021 DF does not
            publish a per-file attack type, so this check is frequently
            unevaluable, and a mandatory unevaluable gate would block the build
            for a property the corpus simply does not claim. The check still
            runs and is still reported.
        require_session_disjoint: Off by default, for the same reason.
        require_file_disjoint: Mandatory. Byte-identical files across a boundary
            are always a defect.
        require_parent_disjoint: Mandatory. Same source recording on both sides.
        require_language_disjoint: Off by default; often unevaluable.
        fail_on_unavailable: Refuse the build when a *mandatory* gate is
            unevaluable. Off by default so a missing optional metadata field
            degrades the report instead of blocking ingestion, and the build
            report names exactly which claim was weakened.
        min_test_sources: A test split with fewer than this many source files is
            refused. One held-out speaker is an anecdote, not an evaluation set.
    """

    require_speaker_disjoint: bool = True
    require_generator_disjoint: bool = False
    require_session_disjoint: bool = False
    require_file_disjoint: bool = True
    require_parent_disjoint: bool = True
    require_language_disjoint: bool = False
    fail_on_unavailable: bool = False
    min_test_sources: int = 1

    def __post_init__(self) -> None:
        if self.min_test_sources < 1:
            msg = "gates.min_test_sources must be at least 1"
            raise DatasetConfigError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "require_speaker_disjoint": self.require_speaker_disjoint,
            "require_generator_disjoint": self.require_generator_disjoint,
            "require_session_disjoint": self.require_session_disjoint,
            "require_file_disjoint": self.require_file_disjoint,
            "require_parent_disjoint": self.require_parent_disjoint,
            "require_language_disjoint": self.require_language_disjoint,
            "fail_on_unavailable": self.fail_on_unavailable,
            "min_test_sources": self.min_test_sources,
        }


@dataclass(frozen=True, slots=True)
class DatasetEntry:
    """One registry entry: a dataset this project knows how to ingest.

    Attributes:
        dataset_id: Stable slug. Becomes the speaker namespace, so changing it
            changes every speaker identifier in the corpus.
        name: Human-readable name for reports.
        source: Where the corpus comes from. A URL or a citation, never a
            license claim.
        version: The corpus version. Part of the build id, so a different
            release is a different dataset even under the same id.
        license: The licence as published. ``"unknown"`` when it cannot be
            determined -- never invented.
        license_status: One of :data:`LICENSE_STATUSES`.
        license_note: Free text, typically the evidence a human checked.
        task: ``"spoof_detection"`` or ``"real_speech"``.
        enabled: Whether a build should use it at all. An entry can be present
            and correct while the corpus is not on this machine.
        path: Location relative to the data root, or absolute.
        adapter: Adapter name from :mod:`voxshield.data.adapters`.
        notes: Free text.
        metadata: Adapter-specific settings, string-valued.
    """

    dataset_id: str
    name: str
    source: str
    version: str
    license: str
    license_status: str
    task: str
    enabled: bool
    path: str
    adapter: str
    license_note: str = ""
    notes: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.dataset_id or any(ch.isspace() for ch in self.dataset_id):
            msg = f"dataset_id {self.dataset_id!r} must be non-empty and contain no whitespace"
            raise DatasetConfigError(msg)
        if not self.adapter:
            msg = f"dataset {self.dataset_id!r} declares no adapter"
            raise DatasetConfigError(msg)
        if self.license_status not in LICENSE_STATUSES:
            msg = (
                f"dataset {self.dataset_id!r} has license_status "
                f"{self.license_status!r}; expected one of {sorted(LICENSE_STATUSES)}"
            )
            raise DatasetConfigError(msg)
        if not self.license.strip():
            msg = (
                f"dataset {self.dataset_id!r} has an empty license; use 'unknown' "
                "explicitly rather than leaving it blank"
            )
            raise DatasetConfigError(msg)
        if self.task not in ("spoof_detection", "real_speech"):
            msg = (
                f"dataset {self.dataset_id!r} has task {self.task!r}; expected "
                "'spoof_detection' or 'real_speech'"
            )
            raise DatasetConfigError(msg)
        if self.license_status == LICENSE_VERIFIED and self.license.strip().lower() == "unknown":
            # A verified licence that reads "unknown" is a bookkeeping error that
            # would otherwise wave a corpus through the gate.
            msg = (
                f"dataset {self.dataset_id!r} is marked VERIFIED but its license is "
                "'unknown'; fix the license or downgrade license_status"
            )
            raise DatasetConfigError(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "source": self.source,
            "version": self.version,
            "license": self.license,
            "license_status": self.license_status,
            "license_note": self.license_note,
            "task": self.task,
            "enabled": self.enabled,
            "path": self.path,
            "adapter": self.adapter,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DatasetEntry:
        """Build an entry from a config mapping, with required keys checked."""
        missing = [
            key
            for key in (
                "dataset_id",
                "source",
                "version",
                "license",
                "license_status",
                "task",
                "path",
                "adapter",
            )
            if not str(payload.get(key, "")).strip()
        ]
        if missing:
            msg = f"dataset entry is missing required field(s): {', '.join(missing)}"
            raise DatasetConfigError(msg)
        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            msg = f"dataset {payload.get('dataset_id')!r} metadata must be a mapping"
            raise DatasetConfigError(msg)
        return cls(
            dataset_id=str(payload["dataset_id"]),
            name=str(payload.get("name") or payload["dataset_id"]),
            source=str(payload["source"]),
            version=str(payload["version"]),
            license=str(payload["license"]),
            license_status=str(payload["license_status"]).upper(),
            license_note=str(payload.get("license_note", "")),
            task=str(payload["task"]),
            enabled=bool(payload.get("enabled", True)),
            path=str(payload["path"]),
            adapter=str(payload["adapter"]),
            notes=str(payload.get("notes", "")),
            metadata={str(k): str(v) for k, v in metadata.items()},
        )


@dataclass(frozen=True, slots=True)
class DataConfig:
    """The complete dataset build configuration.

    Attributes:
        root: Data root directory.
        random_seed: Seed for split assignment and augmentation. Part of the
            build id, because a build with a different seed is a different
            dataset even over identical audio.
        datasets: Registry entries.
        license_policy: How licence statuses are filtered. See
            :data:`LICENSE_POLICIES`.
        window_seconds: Analysis window for dataset segmentation. ``None``
            inherits the Phase 1 default, which is the right answer: a dataset
            window that differs from the inference window produces segments the
            detector cannot score.
        hop_ratio: Window hop as a fraction of the window. ``None`` inherits the
            Phase 1 default of 0.5.
        max_files_per_dataset: Cap for testing and smoke runs. ``None`` ingests
            everything.
        write_segment_audio: Persist standardised segments as WAV under
            ``processed/segments``. Required for a reusable dataset, so it is on
            by default; the files are audio-derived and are git-excluded.
        storage_subtype: WAV sample subtype for stored segments.
        paths: Per-directory names.
        split, validation, augmentation, cache, gates: Sub-configurations.
    """

    root: str = "data"
    random_seed: int = 20260928
    datasets: tuple[DatasetEntry, ...] = ()
    license_policy: str = LICENSE_POLICY_REQUIRE_VERIFIED
    window_seconds: float | None = None
    hop_ratio: float | None = None
    max_files_per_dataset: int | None = None
    write_segment_audio: bool = True
    storage_subtype: str = "PCM_16"
    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    gates: GateConfig = field(default_factory=GateConfig)

    def __post_init__(self) -> None:
        if self.license_policy not in LICENSE_POLICIES:
            msg = (
                f"license_policy must be one of {sorted(LICENSE_POLICIES)}, "
                f"got {self.license_policy!r}"
            )
            raise DatasetConfigError(msg)
        if self.window_seconds is not None and self.window_seconds <= 0.0:
            msg = f"window_seconds must be positive, got {self.window_seconds}"
            raise DatasetConfigError(msg)
        if self.hop_ratio is not None and not 0.0 < self.hop_ratio <= 1.0:
            msg = f"hop_ratio must be within (0, 1], got {self.hop_ratio}"
            raise DatasetConfigError(msg)
        if self.max_files_per_dataset is not None and self.max_files_per_dataset < 1:
            msg = "max_files_per_dataset must be at least 1 when set"
            raise DatasetConfigError(msg)
        if not self.datasets:
            msg = (
                "no datasets configured; add at least one entry under 'datasets:' "
                "in configs/data.yaml"
            )
            raise DatasetConfigError(msg)

        seen: set[str] = set()
        for entry in self.datasets:
            if entry.dataset_id in seen:
                msg = f"duplicate dataset_id {entry.dataset_id!r} in configuration"
                raise DatasetConfigError(msg)
            seen.add(entry.dataset_id)

    # -- derived views -----------------------------------------------------

    def enabled_datasets(self) -> tuple[DatasetEntry, ...]:
        """Entries marked ``enabled``, in configuration order."""
        return tuple(entry for entry in self.datasets if entry.enabled)

    def admissible_datasets(self) -> tuple[DatasetEntry, ...]:
        """Enabled entries whose licence status is admissible under the policy."""
        return tuple(
            entry
            for entry in self.enabled_datasets()
            if license_status_is_admissible(entry.license_status, self.license_policy)
        )

    def excluded_datasets(self) -> tuple[tuple[DatasetEntry, str], ...]:
        """Enabled entries excluded, each with the reason. For the build report.

        Reported rather than silently dropped: an operator who enabled a dataset
        and saw it contribute nothing needs to know which rule excluded it.
        """
        out: list[tuple[DatasetEntry, str]] = []
        for entry in self.enabled_datasets():
            if license_status_is_admissible(entry.license_status, self.license_policy):
                continue
            if entry.license_status == LICENSE_RESTRICTED:
                reason = "license_status is RESTRICTED; not admissible under any policy"
            elif self.license_policy == LICENSE_POLICY_REQUIRE_VERIFIED:
                reason = (
                    f"license_status is {entry.license_status} and license_policy is "
                    f"{LICENSE_POLICY_REQUIRE_VERIFIED}; verify the licence and set "
                    "license_status to VERIFIED, or set license_policy explicitly"
                )
            else:
                reason = f"license_status {entry.license_status} is not admissible"
            out.append((entry, reason))
        return tuple(out)

    def data_paths(self) -> DataPaths:
        """Resolve the directory layout for this configuration."""
        return DataPaths.resolve(self.root, self.paths.overrides())

    def audio_config(self) -> AudioConfig:
        """The Phase 1 :class:`~voxshield.config.AudioConfig` for dataset work.

        Phase 1's defaults are the *inference* defaults, and four of them are
        wrong for a build. Every adjustment is made through public configuration
        -- this method never edits Phase 1 -- and each is deliberate and
        reversible.

        **Defensive intake limits are raised** to the dataset policy's bounds.
        ``max_duration_seconds`` defaults to 30s and ``max_upload_bytes`` to
        10 MiB because those cap what a hostile API caller can request, and
        ``load_audio`` refuses anything larger before a build ever sees it. A
        build reads the operator's own curated corpus, where a 45-second
        consolidated recording is legitimate. Left at the inference defaults, each
        such file is rejected with ``AudioTooLargeError`` and a message about
        *uploads* -- wrong, and unactionable during a build. The Phase 1 limits
        are restored for inference because this is a per-call value, not a
        mutation of any shared object.

        **Window bounds scale proportionally.** ``min_segment_seconds`` is applied
        as a fraction of the window (0.5 by default) rather than clamped to an
        absolute value, so narrowing the window does not produce a window that
        requires itself to be entirely speech.

        **``short_segment_policy`` becomes ``"pad"``.** Inference can afford to
        drop a short clip. A training batch cannot, because segments must stack
        into one tensor. Padded windows record their real ``coverage`` in the
        manifest, so a trainer can filter them.

        **``min_speech_seconds`` becomes the per-window minimum.** A file that
        cannot fill one window with speech is not a training sample, so rejecting
        it here gives one clear reason instead of a segmenter that returns nothing
        and leaves the file reported as unusable for a reason that never mentions
        speech.

        ``window_seconds`` and ``hop_ratio`` may additionally override the
        analysis window. Left as ``None``, Phase 1's defaults apply and dataset
        segments match inference windows exactly, which is the default because a
        dataset window the detector cannot score is not useful.

        Returns:
            A validated :class:`~voxshield.config.AudioConfig`.
        """
        base = AudioConfig()

        window = base.segment_seconds if self.window_seconds is None else self.window_seconds
        # ``hop_ratio`` is a fraction of the window, so it must be scaled by the
        # *resolved* window rather than copied from the default. Reading
        # ``base.segment_hop`` directly would silently pin a narrowed window to
        # the old 2.0s hop and change the overlap ratio behind the caller's back.
        hop_seconds = base.segment_hop_seconds
        if self.hop_ratio is not None:
            hop_seconds = self.hop_ratio * window
        elif self.window_seconds is not None:
            hop_seconds = base.segment_hop * window / base.segment_seconds

        # Proportional, not absolute: the default 2.0s minimum on a 4.0s window is
        # a 50% floor, and the ratio is the part that carries the meaning.
        min_ratio = base.min_segment_seconds / base.segment_seconds
        max_ratio = base.max_segment_seconds / base.segment_seconds
        max_segment = min(max_ratio * window, window)
        min_segment = min(min_ratio * window, max_segment)

        return replace(
            base,
            segment_seconds=window,
            segment_hop_seconds=hop_seconds,
            max_segment_seconds=max_segment,
            min_segment_seconds=min_segment,
            min_speech_seconds=min_segment,
            short_segment_policy="pad",
            max_duration_seconds=max(
                base.max_duration_seconds, self.validation.max_duration_seconds
            ),
            max_upload_bytes=max(base.max_upload_bytes, self.validation.max_file_bytes),
        )

    # -- provenance --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Canonical, JSON-serialisable form of the whole configuration.

        Every field that can change a sample or a split is present, so
        :meth:`config_hash` covers the entire decision surface.
        """
        return {
            "root": self.root,
            "random_seed": self.random_seed,
            "license_policy": self.license_policy,
            "window_seconds": self.window_seconds,
            "hop_ratio": self.hop_ratio,
            "max_files_per_dataset": self.max_files_per_dataset,
            "write_segment_audio": self.write_segment_audio,
            "storage_subtype": self.storage_subtype,
            "datasets": [entry.to_dict() for entry in self.datasets],
            "paths": self.paths.to_dict(),
            "split": self.split.to_dict(),
            "validation": self.validation.to_dict(),
            "augmentation": self.augmentation.to_dict(),
            "cache": self.cache.to_dict(),
            "gates": self.gates.to_dict(),
        }

    def config_hash(self) -> str:
        """Stable hash of :meth:`to_dict`, recorded in every manifest header.

        Covers the full configuration, including ``root`` and the directory
        names. That is correct for provenance: the header is a statement about
        *this* run on *this* machine, and a hash that ignored where the build ran
        would hide a path difference.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def content_fingerprint(self) -> str:
        """Hash of only the settings that can change a sample or a split.

        Deliberately distinct from :meth:`config_hash`. ``config_hash`` records
        the entire run and is what a manifest header cites. This one excludes
        ``root``, ``paths``, and ``cache``: those change *where* output lands and
        how fast it is produced, not *what* the corpus is. The dataset build id is
        derived from this, so the same corpus built at two different paths is
        recognised as one dataset rather than as two competing ones.
        """
        payload = self.to_dict()
        for key in ("root", "paths", "cache"):
            payload.pop(key, None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _coerce_ratio(raw: Any, path: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        msg = f"{path} must be a number, got {raw!r}"
        raise DatasetConfigError(msg) from exc


def _coerce_bool(raw: Any, path: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in ("true", "yes", "1", "on"):
            return True
        if lowered in ("false", "no", "0", "off"):
            return False
    msg = f"{path} must be a boolean, got {raw!r}"
    raise DatasetConfigError(msg)


def _coerce_str_tuple(raw: Any, path: str) -> tuple[str, ...]:
    """Accept a comma-separated string or a sequence, and return a clean tuple.

    Blank entries are dropped rather than preserved, so ``"a,,b,"`` becomes
    ``("a", "b")`` and a trailing comma in YAML does not silently become an
    empty holdout id that would match no generator.
    """
    items: Sequence[Any]
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = raw.split(",")
    elif isinstance(raw, Sequence):
        items = raw
    else:
        msg = f"{path} must be a list of strings, got {raw!r}"
        raise DatasetConfigError(msg)
    return tuple(str(item).strip() for item in items if str(item).strip())


def _build_section(cls: type[Any], payload: Mapping[str, Any] | None, path: str) -> Any:
    """Instantiate a config section from a mapping, rejecting unknown keys.

    Unknown keys are an error rather than a warning. A misspelled
    ``cross_generator_holdout`` that is silently ignored produces a build with no
    generator holdout, an empty cross-generator manifest, and a report that looks
    complete.
    """
    if payload is None:
        return cls()
    if not isinstance(payload, Mapping):
        msg = f"{path} must be a mapping, got {type(payload).__name__}"
        raise DatasetConfigError(msg)

    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = sorted(set(payload) - known)
    if unknown:
        msg = f"{path} has unknown key(s) {unknown}; expected any of {sorted(known)}"
        raise DatasetConfigError(msg)

    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        full = f"{path}.{key}"
        target = getattr(cls, "__dataclass_fields__", {})[key].type
        name = str(target)
        if "bool" in name and "str" not in name:
            kwargs[key] = _coerce_bool(value, full)
        elif "float" in name:
            kwargs[key] = _coerce_ratio(value, full)
        elif "int" in name and "str" not in name:
            kwargs[key] = int(value)
        elif "tuple" in name:
            kwargs[key] = _coerce_str_tuple(value, full)
        elif "str" in name:
            kwargs[key] = "" if value is None else str(value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def parse_data_config(payload: Mapping[str, Any]) -> DataConfig:
    """Build a :class:`DataConfig` from a plain mapping.

    Args:
        payload: The parsed configuration document.

    Returns:
        A validated configuration.

    Raises:
        DatasetConfigError: On an unknown key, a bad type, or a self-contradictory
            value.
    """
    if not isinstance(payload, Mapping):
        msg = f"dataset configuration must be a mapping, got {type(payload).__name__}"
        raise DatasetConfigError(msg)

    top_known = set(DataConfig.__dataclass_fields__)
    unknown = sorted(set(payload) - top_known)
    if unknown:
        msg = f"dataset configuration has unknown top-level key(s) {unknown}"
        raise DatasetConfigError(msg)

    datasets_raw = payload.get("datasets")
    if not isinstance(datasets_raw, Sequence) or isinstance(datasets_raw, (str, bytes)):
        msg = "'datasets' must be a list of dataset entries"
        raise DatasetConfigError(msg)
    entries = tuple(DatasetEntry.from_mapping(item) for item in datasets_raw)

    kwargs: dict[str, Any] = {"datasets": entries}

    if "root" in payload:
        kwargs["root"] = str(payload["root"])
    if "random_seed" in payload:
        kwargs["random_seed"] = int(payload["random_seed"])
    if "license_policy" in payload:
        kwargs["license_policy"] = str(payload["license_policy"])
    if "window_seconds" in payload and payload["window_seconds"] is not None:
        kwargs["window_seconds"] = _coerce_ratio(payload["window_seconds"], "window_seconds")
    if "hop_ratio" in payload and payload["hop_ratio"] is not None:
        kwargs["hop_ratio"] = _coerce_ratio(payload["hop_ratio"], "hop_ratio")
    if "max_files_per_dataset" in payload and payload["max_files_per_dataset"] is not None:
        kwargs["max_files_per_dataset"] = int(payload["max_files_per_dataset"])
    if "write_segment_audio" in payload:
        kwargs["write_segment_audio"] = _coerce_bool(
            payload["write_segment_audio"], "write_segment_audio"
        )
    if "storage_subtype" in payload:
        kwargs["storage_subtype"] = str(payload["storage_subtype"])

    kwargs["paths"] = _build_section(PathsConfig, payload.get("paths"), "paths")
    kwargs["split"] = _build_section(SplitConfig, payload.get("split"), "split")
    kwargs["validation"] = _build_section(ValidationConfig, payload.get("validation"), "validation")
    kwargs["augmentation"] = _build_section(
        AugmentationConfig, payload.get("augmentation"), "augmentation"
    )
    kwargs["cache"] = _build_section(CacheConfig, payload.get("cache"), "cache")
    kwargs["gates"] = _build_section(GateConfig, payload.get("gates"), "gates")
    return DataConfig(**kwargs)


def _load_yaml(path: Path) -> Mapping[str, Any]:
    """Parse a YAML document, importing PyYAML lazily.

    Kept out of the module import so the service does not pay for a build-time
    dependency, and given its own function so the error message can say exactly
    which extra to install.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on install extras
        msg = (
            f"reading {path} requires PyYAML. Install it with "
            "'pip install voxshield[data]', or construct DataConfig in Python."
        )
        raise DatasetConfigError(msg) from exc

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"cannot read dataset configuration at {path}: {exc}"
        raise DatasetConfigError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"dataset configuration at {path} is not valid YAML: {exc}"
        raise DatasetConfigError(msg) from exc

    if loaded is None:
        msg = f"dataset configuration at {path} is empty"
        raise DatasetConfigError(msg)
    if not isinstance(loaded, Mapping):
        msg = f"dataset configuration at {path} must be a mapping at the top level"
        raise DatasetConfigError(msg)
    return loaded


def load_data_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> DataConfig:
    """Load, merge, and validate a dataset configuration.

    Precedence, lowest to highest: dataclass defaults, the YAML document,
    ``overrides``, then environment variables. The environment sits highest so an
    operator can point a build at a different corpus for one run without editing
    a tracked file.

    Recognised environment variables:

    * ``VOXSHIELD_DATA_ROOT`` -- data root directory.
    * ``VOXSHIELD_DATA_SEED`` -- random seed.
    * ``VOXSHIELD_DATA_LICENSE_POLICY`` -- licence policy.
    * ``VOXSHIELD_DATA_MAX_FILES`` -- per-dataset file cap, for smoke runs.

    Args:
        path: A YAML document. ``None`` uses the defaults plus environment.
        overrides: In-memory overrides applied after the document.
        env: Environment mapping. Defaults to ``os.environ``.

    Returns:
        A validated :class:`DataConfig`.

    Raises:
        DatasetConfigError: On any configuration problem.
    """
    environ = os.environ if env is None else env
    payload: dict[str, Any] = {}
    if path is not None:
        payload.update(dict(_load_yaml(Path(path))))
    if overrides:
        payload.update(dict(overrides))

    if environ.get("VOXSHIELD_DATA_ROOT"):
        payload["root"] = environ["VOXSHIELD_DATA_ROOT"]
    if environ.get("VOXSHIELD_DATA_SEED"):
        try:
            payload["random_seed"] = int(environ["VOXSHIELD_DATA_SEED"])
        except ValueError as exc:
            msg = f"VOXSHIELD_DATA_SEED must be an integer, got {environ['VOXSHIELD_DATA_SEED']!r}"
            raise DatasetConfigError(msg) from exc
    if environ.get("VOXSHIELD_DATA_LICENSE_POLICY"):
        payload["license_policy"] = environ["VOXSHIELD_DATA_LICENSE_POLICY"]
    if environ.get("VOXSHIELD_DATA_MAX_FILES"):
        try:
            payload["max_files_per_dataset"] = int(environ["VOXSHIELD_DATA_MAX_FILES"])
        except ValueError as exc:
            msg = (
                "VOXSHIELD_DATA_MAX_FILES must be an integer, got "
                f"{environ['VOXSHIELD_DATA_MAX_FILES']!r}"
            )
            raise DatasetConfigError(msg) from exc

    return parse_data_config(payload)
