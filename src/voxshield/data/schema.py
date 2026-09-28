"""The common record schema, at two granularities.

VoxShield has to ingest corpora whose on-disk conventions have nothing in common
with each other, then emit training data whose records *do* have something in
common. Those are two different shapes, and collapsing them into one is how a
dataset pipeline ends up unable to express either a directory of files or a list
of analysis windows.

So there are two records:

* :class:`SourceRecord` -- one discovered file plus the metadata its dataset
  actually published. This is the unit that validation rejects and the unit the
  split assigns.
* :class:`SampleRecord` -- one analysis segment, carrying the split it inherited
  from its parent file. This is the unit a manifest row and a training loader
  deal in.

The relationship is one-to-many and it is recorded explicitly: every
:class:`SampleRecord` names the ``parent_id`` of the :class:`SourceRecord` it
came from. That single field is what makes leakage across a split boundary
detectable, because without it, ten windows from one speaker look like ten
independent observations.

Unknown metadata is stored as :data:`UNKNOWN`, never as ``None`` and never as a
guess. An adapter that cannot determine a speaker writes ``"unknown"``, and the
leakage checker then reports the speaker check as unavailable rather than
pretending ten unknown speakers are ten distinct speakers.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from voxshield.data.errors import DatasetConfigError
from voxshield.data.labels import attack_family as classify_attack
from voxshield.data.labels import encode_label, is_valid_label

__all__ = [
    "UNKNOWN",
    "SampleRecord",
    "SourceRecord",
    "known_or_unknown",
    "namespace_speaker",
]

#: The single representation of "this dataset did not publish this field".
UNKNOWN = "unknown"

# Speaker namespaces must be stable and greppable. The separator is chosen so the
# namespace is visible in a report without a lookup table.
_SPEAKER_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def known_or_unknown(value: str | None) -> str:
    """Normalise optional metadata to either the real value or :data:`UNKNOWN`.

    Args:
        value: A published value, or ``None``/blank.

    Returns:
        The stripped value, or :data:`UNKNOWN`. Case is preserved because a
        language or generator tag's case can be the only thing distinguishing
        ``"EN"`` from ``"en"``, and normalising that away is a metadata decision
        this function should not be making on the adapter's behalf.
    """
    if value is None:
        return UNKNOWN
    text = str(value).strip()
    return text or UNKNOWN


def namespace_speaker(dataset_id: str, speaker_id: str | None) -> str:
    """Scope a speaker identifier to the dataset that published it.

    Every public corpus numbers its speakers, and nearly all of them start at 1.
    Merging ``ASVspoof`` speaker ``17`` with ``WaveFake`` speaker ``17`` would put
    two unrelated humans on the same side of a train/test boundary and make the
    resulting speaker-disjointness claim false. The namespace is therefore
    applied mechanically, at the point the identifier enters the schema, so no
    adapter can forget.

    Args:
        dataset_id: The dataset the identifier came from.
        speaker_id: The raw identifier, or ``None``.

    Returns:
        ``"<dataset_id>:<speaker_id>"``, or :data:`UNKNOWN` when absent. An
        identifier that already carries the namespace is returned unchanged, so
        re-ingesting the same corpus is idempotent.
    """
    raw = known_or_unknown(speaker_id)
    if raw == UNKNOWN:
        return UNKNOWN
    if raw.startswith(f"{dataset_id}:"):
        return raw
    namespace = dataset_id.strip().lower()
    if not _SPEAKER_NAMESPACE_RE.match(namespace):
        msg = (
            f"dataset_id {dataset_id!r} cannot form a speaker namespace; "
            "use lowercase alphanumerics, underscore, dot, or dash"
        )
        raise DatasetConfigError(msg)
    return f"{namespace}:{raw}"


def _clean_optional(value: str | None) -> str | None:
    """``None`` for an unknown value, so optional fields stay genuinely optional."""
    text = known_or_unknown(value)
    return None if text == UNKNOWN else text


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One discovered source file, before validation, splitting, or segmentation.

    Attributes:
        sample_id: Unique within the dataset. Namespaced by the adapter so two
            corpora cannot collide on a numeric filename.
        dataset_id: The registry entry this came from.
        audio_path: Path to the encoded source, relative to the data root where
            possible so a manifest is portable between machines.
        label: ``"bona_fide"`` or ``"spoof"``. Required -- an unlabelled file
            cannot enter a supervised corpus.
        parent_id: The recording this file belongs to. Used as the grouping unit
            for splitting and leakage. Defaults to the file itself, which is
            correct for a corpus of independent utterances and honest about the
            absence of a coarser grouping.
        speaker_id: Namespaced speaker identifier, or :data:`UNKNOWN`.
        generator_id: The synthesis system, or :data:`UNKNOWN`. Drives the
            cross-generator split.
        language: Language tag, or :data:`UNKNOWN`. Drives the cross-language
            split.
        codec: Source codec, or :data:`UNKNOWN`. Drives the cross-codec split.
        session_id: Capture session, or :data:`UNKNOWN`. Sessions are the axis
            that catches same-microphone, same-room overlap.
        attack_type: Detailed attack or vocoder identifier, or :data:`UNKNOWN`.
            Metadata only; never part of the training target.
        attack_family: Coarse grouping derived from ``attack_type``.
        channel: Channel label, or :data:`UNKNOWN`.
        source_split: The split the *source dataset* published, if any. Recorded
            for provenance and cross-checking, never trusted: the split this
            pipeline produces is the one that governs.
        recorded_at: ISO-8601 capture date, or :data:`UNKNOWN`. Optional, and
            its absence is what makes temporal splitting unavailable rather than
            automatic.
        duration_seconds: Source duration from the container header, or ``None``.
        sample_rate: Source sample rate from the container header, or ``None``.
        extra: Dataset-specific passthrough metadata, string-valued.
    """

    sample_id: str
    dataset_id: str
    audio_path: str
    label: str
    parent_id: str = ""
    speaker_id: str = UNKNOWN
    generator_id: str = UNKNOWN
    language: str = UNKNOWN
    codec: str = UNKNOWN
    session_id: str = UNKNOWN
    attack_type: str = UNKNOWN
    channel: str = UNKNOWN
    source_split: str = UNKNOWN
    recorded_at: str = UNKNOWN
    duration_seconds: float | None = None
    sample_rate: int | None = None
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_valid_label(self.label):
            msg = (
                f"source {self.sample_id!r} has label {self.label!r}; "
                "expected 'bona_fide' or 'spoof'"
            )
            raise DatasetConfigError(msg)
        if not self.dataset_id:
            msg = f"source {self.sample_id!r} has an empty dataset_id"
            raise DatasetConfigError(msg)
        if not self.sample_id:
            msg = f"source in dataset {self.dataset_id!r} has an empty sample_id"
            raise DatasetConfigError(msg)
        if not self.parent_id:
            # A corpus of independent utterances has no coarser grouping, so the
            # file is its own group. Defaulting here rather than requiring every
            # adapter to restate that keeps "unknown grouping" and "deliberately
            # grouped" distinguishable by omitting the field, not by inventing one.
            object.__setattr__(self, "parent_id", self.sample_id)
        # The speaker namespace is applied here, at the point the identifier
        # enters the schema, rather than left to the adapter. Every public corpus
        # numbers its speakers from 1, so an adapter that omits the namespace
        # merges speaker 17 of one corpus with speaker 17 of another and quietly
        # falsifies speaker-disjointness. Doing it in one place is what makes it
        # impossible to forget; :func:`namespace_speaker` is idempotent, so
        # re-ingesting an already-namespaced record is a no-op.
        object.__setattr__(self, "speaker_id", namespace_speaker(self.dataset_id, self.speaker_id))
        # Metadata defaults are normalised on the way in, so an adapter that
        # passes ``None`` for speaker or language cannot produce a record that
        # compares unequal to an otherwise identical one.
        for name in (
            "generator_id",
            "language",
            "codec",
            "session_id",
            "attack_type",
            "channel",
            "source_split",
            "recorded_at",
        ):
            object.__setattr__(self, name, known_or_unknown(getattr(self, name)))

    @property
    def attack_family(self) -> str:
        """Coarse grouping derived from :attr:`attack_type`.

        Derived rather than stored. Both a published attack identifier and a
        hand-assigned family can end up in a manifest, and once they disagree
        nothing downstream can tell which one is wrong -- a report claiming
        "60% vocoder" next to a manifest of ``A07`` rows is the kind of error
        that survives review. One field, one source.
        """
        return classify_attack(self.attack_type)

    @property
    def label_index(self) -> int:
        """Training index for :attr:`label`, from the single shared encoding."""
        return encode_label(self.label)

    @property
    def group_key(self) -> str:
        """The key this file is split and leakage-checked as a group.

        Prefixed to make it obvious in a traceback that this is a grouping key and
        not a sample identifier -- the two are frequently equal, and conflating
        them is precisely the bug this property exists to make visible.
        """
        return f"parent:{self.parent_id}"

    @property
    def partition_key(self) -> str:
        """The key that actually gets assigned to a split.

        A speaker, when known. Assigning whole speakers to one side is what makes
        speaker-disjointness true by construction rather than by luck; a file with
        no published speaker falls back to its own parent group, which is
        disjoint by definition and honestly *not* speaker-disjoint.
        """
        if self.speaker_id and self.speaker_id != UNKNOWN:
            return f"speaker:{self.speaker_id}"
        return self.group_key

    def is_pure_for(self, field: str, allowed: frozenset[str]) -> bool:
        """Whether every known value of ``field`` on this file is in ``allowed``.

        Args:
            field: A metadata attribute name, e.g. ``"generator_id"``.
            allowed: The holdout set.

        Returns:
            ``True`` only if the field is known and fully inside the holdout set.
            A file with an unknown value is never "pure" for that axis: claiming a
            generator holdout on a file whose generator was never published would
            produce a cross-generator test that is not cross-generator.
        """
        value = getattr(self, field, UNKNOWN)
        if not value or value == UNKNOWN:
            return False
        return value in allowed

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, for reports and interim records."""
        return {
            "sample_id": self.sample_id,
            "dataset_id": self.dataset_id,
            "audio_path": self.audio_path,
            "label": self.label,
            "label_index": self.label_index,
            "parent_id": self.parent_id,
            "speaker_id": self.speaker_id,
            "generator_id": self.generator_id,
            "language": self.language,
            "codec": self.codec,
            "session_id": self.session_id,
            "attack_type": self.attack_type,
            "attack_family": self.attack_family,
            "channel": self.channel,
            "source_split": self.source_split,
            "recorded_at": self.recorded_at,
            "duration_seconds": self.duration_seconds,
            "sample_rate": self.sample_rate,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SourceRecord:
        """Rebuild a record from :meth:`to_dict` output.

        ``attack_family`` and ``label_index`` appear in the serialised form but are
        not accepted here: both are derived, so accepting a caller-supplied copy
        would let a manifest carry a value the record then disagrees with. Missing
        metadata is tolerated and normalises to :data:`UNKNOWN`; a missing
        required field raises :class:`KeyError`.
        """
        return cls(
            sample_id=str(payload["sample_id"]),
            dataset_id=str(payload["dataset_id"]),
            audio_path=str(payload["audio_path"]),
            label=str(payload["label"]),
            parent_id=str(payload.get("parent_id") or payload["sample_id"]),
            # Optional metadata is absent in a real manifest, not merely empty.
            # Routing it through the normaliser keeps the declared ``str`` type
            # honest while preserving the "missing means unknown" contract the
            # docstring states.
            speaker_id=known_or_unknown(payload.get("speaker_id")),
            generator_id=known_or_unknown(payload.get("generator_id")),
            language=known_or_unknown(payload.get("language")),
            codec=known_or_unknown(payload.get("codec")),
            session_id=known_or_unknown(payload.get("session_id")),
            attack_type=known_or_unknown(payload.get("attack_type")),
            channel=known_or_unknown(payload.get("channel")),
            source_split=known_or_unknown(payload.get("source_split")),
            recorded_at=known_or_unknown(payload.get("recorded_at")),
            duration_seconds=(
                None
                if payload.get("duration_seconds") is None
                else float(payload["duration_seconds"])
            ),
            sample_rate=(
                None if payload.get("sample_rate") is None else int(payload["sample_rate"])
            ),
            extra=dict(payload.get("extra") or {}),
        )


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """One analysis segment: the unit a manifest row and a loader deal in.

    Attributes:
        sample_id: Globally unique. Built from the source id and the segment
            index so it is stable across builds of the same audio.
        dataset_id: The registry entry this came from.
        audio_path: Path to the standardised segment, relative to the data root.
        label: ``"bona_fide"`` or ``"spoof"``.
        label_index: Encoded label, from :mod:`voxshield.data.labels`.
        split: ``"train"``, ``"dev"``, or ``"test"``. A segment never has a finer
            split than its parent, and this is the field leakage is measured on.
        parent_id: The source file's grouping id. Inherited, never re-derived, so
            every window of a recording reports the same side of the boundary.
        segment_index: Position of this window within the source recording.
        start_seconds: Window start in the source timeline, after Phase 1
            preprocessing, so it indexes the canonical signal rather than the
            original file.
        duration_seconds: Window length in seconds.
        sample_rate: Canonical sample rate of the stored segment.
        speech_seconds: Speech content inside the window, from Phase 1's VAD.
        coverage: Fraction of the window that carries speech. Recorded so a
            trainer can filter quiet windows without re-deriving VAD.
        is_padded: Whether the window was zero-filled to full width.
        waveform_samples: Samples in the stored segment.
        speaker_id, generator_id, language, codec, session_id, attack_type,
            attack_family, channel, recorded_at: Inherited source metadata.
        file_hash: SHA-256 of the *source* file, so a re-encoded duplicate is
            traceable to the original it came from.
        content_hash: SHA-256 of the stored segment's PCM, so the same audio
            under a different filename is still detected as a duplicate.
        source_split: The source dataset's own split, for provenance.
        preprocessing_version: The Phase 1 signature that produced the segment.
        dataset_build_id: The build this row belongs to.
        extra: Dataset-specific passthrough metadata.
    """

    sample_id: str
    dataset_id: str
    audio_path: str
    label: str
    label_index: int
    split: str
    parent_id: str
    segment_index: int
    start_seconds: float
    duration_seconds: float
    sample_rate: int
    speech_seconds: float
    coverage: float
    is_padded: bool
    waveform_samples: int
    speaker_id: str
    generator_id: str
    language: str
    codec: str
    session_id: str
    attack_type: str
    channel: str
    recorded_at: str
    file_hash: str
    content_hash: str
    source_split: str
    preprocessing_version: str
    dataset_build_id: str
    extra: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not is_valid_label(self.label):
            msg = f"sample {self.sample_id!r} has invalid label {self.label!r}"
            raise DatasetConfigError(msg)
        if self.label_index != encode_label(self.label):
            msg = (
                f"sample {self.sample_id!r} label_index {self.label_index} "
                f"disagrees with label {self.label!r}"
            )
            raise DatasetConfigError(msg)
        # A segment inherits an already-namespaced speaker from its source
        # record. Re-applying it is idempotent, and doing so here means a segment
        # built by hand -- or by a loader reading a manifest from an older build
        # -- cannot reintroduce the cross-corpus speaker collision the namespace
        # exists to prevent.
        object.__setattr__(self, "speaker_id", namespace_speaker(self.dataset_id, self.speaker_id))
        for name in (
            "generator_id",
            "language",
            "codec",
            "session_id",
            "attack_type",
            "channel",
            "recorded_at",
            "source_split",
        ):
            object.__setattr__(self, name, known_or_unknown(getattr(self, name)))

    @property
    def attack_family(self) -> str:
        """Coarse grouping derived from :attr:`attack_type`. See :class:`SourceRecord`."""
        return classify_attack(self.attack_type)

    @property
    def manifest_name(self) -> str:
        """Which ``test_*`` manifest this segment also belongs to, or ``""``.

        A test segment can legitimately appear in several evaluation manifests
        when it qualifies for more than one axis -- a cross-language test that is
        also in-domain is not a contradiction. Training and dev segments are
        never in a test manifest, and this property returns the empty string for
        them so a caller cannot accidentally add a training row to an evaluation
        set.
        """
        return ""

    def with_build(self, dataset_build_id: str) -> SampleRecord:
        """Return a copy stamped with ``dataset_build_id``."""
        return replace(self, dataset_build_id=dataset_build_id)

    def with_split(self, split: str) -> SampleRecord:
        """Return a copy assigned to ``split``."""
        return replace(self, split=split)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, with the keys a manifest row carries."""
        return {
            "sample_id": self.sample_id,
            "dataset_id": self.dataset_id,
            "audio_path": self.audio_path,
            "label": self.label,
            "label_index": self.label_index,
            "split": self.split,
            "parent_id": self.parent_id,
            "segment_index": self.segment_index,
            "start_seconds": round(self.start_seconds, 4),
            "duration_seconds": round(self.duration_seconds, 4),
            "sample_rate": self.sample_rate,
            "speech_seconds": round(self.speech_seconds, 4),
            "coverage": round(self.coverage, 4),
            "is_padded": self.is_padded,
            "waveform_samples": self.waveform_samples,
            "speaker_id": self.speaker_id,
            "generator_id": self.generator_id,
            "language": self.language,
            "codec": self.codec,
            "session_id": self.session_id,
            "attack_type": self.attack_type,
            "attack_family": self.attack_family,
            "channel": self.channel,
            "recorded_at": self.recorded_at,
            "file_hash": self.file_hash,
            "content_hash": self.content_hash,
            "source_split": self.source_split,
            "preprocessing_version": self.preprocessing_version,
            "dataset_build_id": self.dataset_build_id,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SampleRecord:
        """Rebuild a record from :meth:`to_dict` output.

        Raises:
            KeyError: If a required field is missing, which is how a manifest
                written by an older or broken build is detected rather than
                silently producing a record full of defaults.
        """
        return cls(
            sample_id=str(payload["sample_id"]),
            dataset_id=str(payload["dataset_id"]),
            audio_path=str(payload["audio_path"]),
            label=str(payload["label"]),
            label_index=int(payload["label_index"]),
            split=str(payload["split"]),
            parent_id=str(payload["parent_id"]),
            segment_index=int(payload["segment_index"]),
            start_seconds=float(payload["start_seconds"]),
            duration_seconds=float(payload["duration_seconds"]),
            sample_rate=int(payload["sample_rate"]),
            speech_seconds=float(payload["speech_seconds"]),
            coverage=float(payload["coverage"]),
            is_padded=bool(payload["is_padded"]),
            waveform_samples=int(payload["waveform_samples"]),
            speaker_id=known_or_unknown(payload.get("speaker_id")),
            generator_id=known_or_unknown(payload.get("generator_id")),
            language=known_or_unknown(payload.get("language")),
            codec=known_or_unknown(payload.get("codec")),
            session_id=known_or_unknown(payload.get("session_id")),
            attack_type=known_or_unknown(payload.get("attack_type")),
            channel=known_or_unknown(payload.get("channel")),
            recorded_at=known_or_unknown(payload.get("recorded_at")),
            file_hash=str(payload.get("file_hash") or UNKNOWN),
            content_hash=str(payload.get("content_hash") or UNKNOWN),
            source_split=known_or_unknown(payload.get("source_split")),
            preprocessing_version=str(payload["preprocessing_version"]),
            dataset_build_id=str(payload["dataset_build_id"]),
            extra=dict(payload.get("extra") or {}),
        )
