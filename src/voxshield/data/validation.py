"""Which discovered files may become training samples, and why the rest may not.

Discovery answers *what is there*. This module answers *what may be used*, and it
never answers that without an explanation. Every rejected file carries one of the
stable codes in :data:`REJECTION_CODES` plus a sentence naming the measurement
that failed and the threshold it failed against. The alternative -- a build that
quietly drops files and reports the size of what survived -- is indistinguishable
from a build that worked, and the operator has no way to recover a corpus that was
discarded for a reason nobody recorded.

Three decisions shape everything below.

**A rejection is a report, not an exception.** A corpus is expected to contain
undecodable downloads, truncated transfers, and recordings of a doorbell. Raising
on the first one would make the build unusable on the corpora it is meant to
digest, so every per-file failure is caught and recorded. The exceptions this
module *does* raise are the ones where continuing would silently corrupt the
result: a duplicated ``sample_id`` and a ``require_metadata`` entry that names no
field on the schema.

**One check chain, a fixed precedence.** A file can fail several checks at once,
and only one reason is reported, so the order in
:func:`_first_failure` is itself a contract: the cheapest and most fundamental
condition wins, and it is always checked before the conditions that depend on it.
Duration precedes speech because a file below the minimum duration is rejected on
duration regardless of how much speech it holds, and reporting "not enough
speech" there would send an operator to tune a threshold that was never the
problem.

**Nothing here retains audio.** :class:`FileMeasurement` holds durations, ratios,
and a :class:`~voxshield.audio.quality.QualityReport` -- all scalars. The decoded
waveform and the per-frame VAD mask exist only inside :func:`measure_file` and go
out of scope with it, so a :class:`ValidationResult` covering a large corpus can
be held, serialised, and attached to a build report without becoming a second copy
of the corpus in memory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from voxshield.audio.loader import load_audio
from voxshield.audio.preprocess import to_mono
from voxshield.audio.quality import QualityReport, assess_quality
from voxshield.audio.vad import detect_speech
from voxshield.config import AudioConfig
from voxshield.data.config import ValidationConfig
from voxshield.data.discovery import DiscoveryResult, InventoryEntry, build_inventory
from voxshield.data.errors import DatasetBuildError, DatasetConfigError
from voxshield.data.schema import UNKNOWN, SourceRecord
from voxshield.errors import (
    AudioTooLargeError,
    InvalidAudioSignalError,
    UnsupportedAudioFormatError,
    VoxShieldError,
)

__all__ = [
    "REJECTION_CODES",
    "REJECT_AUDIO_BUDGET",
    "REJECT_DUPLICATE_CONTENT",
    "REJECT_DUPLICATE_FILE",
    "REJECT_FILE_TOO_LARGE",
    "REJECT_MISSING_METADATA",
    "REJECT_NO_SPEECH",
    "REJECT_QUALITY_WARNING",
    "REJECT_TOO_LONG",
    "REJECT_TOO_SHORT",
    "REJECT_UNDECODABLE",
    "REJECT_UNREADABLE",
    "REJECT_UNSUPPORTED_FORMAT",
    "REJECT_UNUSABLE_SIGNAL",
    "Acceptance",
    "FileMeasurement",
    "Rejection",
    "ValidationResult",
    "ValidationStats",
    "measure_file",
    "validate_measurement",
    "validate_sources",
]

#: The file could not be measured at all -- missing, or unreadable on this
#: filesystem. Carried over from discovery rather than re-derived here.
REJECT_UNREADABLE = "UNREADABLE"

#: The bytes claimed to be audio and were not decodable.
REJECT_UNDECODABLE = "UNDECODABLE"

#: The container or codec is not on the allow-list. Distinct from
#: :data:`REJECT_UNDECODABLE` because a WAV-only project rejecting an MP3 is a
#: configuration choice, while a truncated WAV is a damaged file, and the two
#: warrant different operator responses.
REJECT_UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"

#: Decodable, but the signal itself carries no analysable content -- digital
#: silence, or a recording that is almost entirely silence.
REJECT_UNUSABLE_SIGNAL = "UNUSABLE_SIGNAL"

#: A Phase 1 intake budget was exceeded: bytes, duration, sample rate, or
#: channels. The message names which one.
REJECT_AUDIO_BUDGET = "AUDIO_BUDGET_EXCEEDED"

#: A field named by ``validation.require_metadata`` is unknown for this record.
REJECT_MISSING_METADATA = "MISSING_METADATA"

#: The source file is larger than ``validation.max_file_bytes``.
REJECT_FILE_TOO_LARGE = "FILE_TOO_LARGE"

#: Shorter than ``validation.min_duration_seconds``.
REJECT_TOO_SHORT = "TOO_SHORT"

#: Longer than ``validation.max_duration_seconds``.
REJECT_TOO_LONG = "TOO_LONG"

#: Less speech than one analysis window requires.
REJECT_NO_SPEECH = "NO_SPEECH"

#: Carries a non-blocking quality issue while ``validation.reject_warning_issues``
#: is on.
REJECT_QUALITY_WARNING = "QUALITY_WARNING"

#: Byte-identical to an earlier accepted file.
REJECT_DUPLICATE_FILE = "DUPLICATE_FILE"

#: The same audio as an earlier accepted record, under a different file.
REJECT_DUPLICATE_CONTENT = "DUPLICATE_CONTENT"

#: Every code :func:`validate_sources` can emit. Declared so a report or a test
#: can assert that nothing escapes the vocabulary, and so that a code cannot be
#: invented at a call site the way an issue code in
#: :mod:`voxshield.audio.quality` deliberately is not.
REJECTION_CODES: frozenset[str] = frozenset(
    {
        REJECT_UNREADABLE,
        REJECT_UNDECODABLE,
        REJECT_UNSUPPORTED_FORMAT,
        REJECT_UNUSABLE_SIGNAL,
        REJECT_AUDIO_BUDGET,
        REJECT_MISSING_METADATA,
        REJECT_FILE_TOO_LARGE,
        REJECT_TOO_SHORT,
        REJECT_TOO_LONG,
        REJECT_NO_SPEECH,
        REJECT_QUALITY_WARNING,
        REJECT_DUPLICATE_FILE,
        REJECT_DUPLICATE_CONTENT,
    }
)

#: ``validation.require_metadata`` may only name real :class:`SourceRecord` fields.
#: ``label`` is included: it is a dataclass field, and it is checked first, so the
#: generic loop skips it rather than reporting it twice under two codes.
_METADATA_FIELDS: frozenset[str] = frozenset(SourceRecord.__dataclass_fields__)

#: Duplicate scopes, named as ``ValidationConfig.dedup_scope`` names them.
_SCOPE_FILE = "file"
_SCOPE_CONTENT = "content"


def _seconds(value: float) -> str:
    """Format a duration for a message, at the precision the thresholds use."""
    return f"{value:.2f}s"


@dataclass(frozen=True, slots=True)
class FileMeasurement:
    """One candidate file, measured. Metadata only, never audio.

    Attributes:
        duration_seconds: Length of the decoded signal.
        speech_seconds: Speech found by the Phase 1 VAD. The authoritative measure
            of how much analysable speech the file holds.
        speech_ratio: Fraction of frames classified as speech, for the report.
        sample_rate: Rate the file was decoded at, before any resampling.
        file_bytes: Size on disk.
        quality: Phase 1's quality assessment, including the issue codes that
            separate "unusable" from "suspect".
    """

    duration_seconds: float
    speech_seconds: float
    speech_ratio: float
    sample_rate: int
    file_bytes: int
    quality: QualityReport

    @property
    def blocking_issues(self) -> tuple[str, ...]:
        """Quality issues that make the signal unusable rather than suspect."""
        return self.quality.blocking_issues

    @property
    def warnings(self) -> tuple[str, ...]:
        """Quality issues describing a low-confidence recording without blocking it."""
        return self.quality.warnings


@dataclass(frozen=True, slots=True)
class Rejection:
    """One file that may not enter the corpus, and the single reason why.

    Attributes:
        sample_id: The record that was rejected.
        dataset_id: Owning corpus, so a report can group by origin.
        audio_path: Path as recorded on the record.
        code: One of :data:`REJECTION_CODES`. Stable; match on it, do not parse
            :attr:`reason`.
        reason: A sentence naming the measurement and the threshold. The part a
            human acts on, and the reason it is generated rather than stored.
    """

    sample_id: str
    dataset_id: str
    audio_path: str
    code: str
    reason: str

    def __post_init__(self) -> None:
        if self.code not in REJECTION_CODES:
            msg = (
                f"rejection for {self.sample_id!r} uses code {self.code!r}, which "
                f"is not one of {sorted(REJECTION_CODES)}"
            )
            raise DatasetBuildError(msg)

    def to_dict(self) -> dict[str, str]:
        """JSON-serialisable form, for the build report."""
        return {
            "sample_id": self.sample_id,
            "dataset_id": self.dataset_id,
            "audio_path": self.audio_path,
            "code": self.code,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class Acceptance:
    """One file admitted to the corpus, with the measurements that admitted it.

    Carrying the record alongside its measurement keeps the two from becoming
    parallel collections that can desynchronise, and gives the splitting stage a
    single list to walk.

    Attributes:
        record: The surviving :class:`SourceRecord`.
        measurement: What the file was measured as.
    """

    record: SourceRecord
    measurement: FileMeasurement

    @property
    def sample_id(self) -> str:
        """The record's sample id."""
        return self.record.sample_id

    @property
    def dataset_id(self) -> str:
        """The record's dataset id."""
        return self.record.dataset_id

    @property
    def duration_seconds(self) -> float:
        """Decoded duration of the file."""
        return self.measurement.duration_seconds

    @property
    def speech_seconds(self) -> float:
        """Speech content found in the file."""
        return self.measurement.speech_seconds

    @property
    def warnings(self) -> tuple[str, ...]:
        """Quality issues tolerated on admission, for the build report."""
        return self.measurement.warnings


@dataclass(frozen=True, slots=True)
class ValidationStats:
    """Counts describing what survived and why the rest did not.

    Attributes:
        total: Records offered for validation.
        accepted: Records admitted.
        rejected: Records refused, equal to ``sum(rejected_per_reason.values())``.
        accepted_per_dataset: Admitted records per corpus, sorted by name.
        rejected_per_reason: Rejections grouped by code, sorted by code. The view
            that turns three thousand individual sentences into the two rules
            actually responsible.
        duration_seconds, speech_seconds: Totals over admitted files, so a build
            report can state how much audio the corpus actually contains.
        unavailable_scopes: Duplicate scopes that were requested but could not be
            evaluated, e.g. ``"content"`` before segmentation has produced stored
            segment hashes. Recorded rather than silently treated as a pass.
    """

    total: int = 0
    accepted: int = 0
    rejected: int = 0
    accepted_per_dataset: dict[str, int] = field(default_factory=dict)
    rejected_per_reason: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0
    speech_seconds: float = 0.0
    unavailable_scopes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """The outcome of a validation pass.

    Attributes:
        accepted: Admitted files, ordered by ``sample_id``.
        rejected: Refused files, ordered by ``(code, sample_id)`` so a report
            reads as a list of reasons with examples rather than a shuffled one.
        stats: Counts.
    """

    accepted: tuple[Acceptance, ...] = ()
    rejected: tuple[Rejection, ...] = ()
    stats: ValidationStats = field(default_factory=ValidationStats)

    def accepted_ids(self) -> frozenset[str]:
        """Sample ids admitted to the corpus."""
        return frozenset(item.sample_id for item in self.accepted)

    def rejected_ids(self) -> frozenset[str]:
        """Sample ids refused."""
        return frozenset(item.sample_id for item in self.rejected)

    def accepted_records(self) -> tuple[SourceRecord, ...]:
        """The surviving records, for the splitting stage."""
        return tuple(item.record for item in self.accepted)

    def rejections_by_code(self) -> dict[str, tuple[Rejection, ...]]:
        """Rejections grouped by code, for a human-facing report."""
        grouped: dict[str, list[Rejection]] = {}
        for rejection in self.rejected:
            grouped.setdefault(rejection.code, []).append(rejection)
        return {code: tuple(grouped[code]) for code in sorted(grouped)}


def measure_file(
    path: str | Path,
    audio_config: AudioConfig | None = None,
    *,
    file_bytes: int | None = None,
) -> FileMeasurement:
    """Decode one file and measure what validation needs to judge it.

    Args:
        path: The encoded file.
        audio_config: Phase 1 limits and thresholds. Should be the build's audio
            configuration -- :meth:`~voxshield.data.config.DataConfig.audio_config`
            -- because ``min_speech_seconds`` set there is the per-window minimum
            a training file must reach.
        file_bytes: Size already known from an inventory pass, to avoid a second
            ``stat``. Measured here when omitted.

    Returns:
        A :class:`FileMeasurement`.

    Raises:
        voxshield.errors.AudioIntakeError: The file is undecodable, of an
            unsupported format, over an intake budget, or carries no analysable
            signal. :func:`validate_sources` catches these and turns each into a
            coded rejection; a direct caller sees them as they are.
    """
    cfg = audio_config or AudioConfig()
    target = Path(path)

    decoded = load_audio(target, config=cfg)
    # A mono container keeps its channel axis, so ``samples`` arrives as
    # ``(n_frames, 1)``. Both the VAD and the quality metrics frame a 1-D signal,
    # so handing them the 2-D form fails inside the framing rather than at the
    # call. ``to_mono`` collapses it, and is a no-op on an already-mono 1-D array.
    samples = to_mono(decoded.samples)
    mask = detect_speech(samples, decoded.sample_rate, cfg)
    quality = assess_quality(samples, decoded.sample_rate, cfg)

    # The waveform and the frame mask go out of scope here. Only the scalars and
    # the scalar-only quality report survive, so holding a result for a whole
    # corpus never holds the corpus.
    return FileMeasurement(
        duration_seconds=decoded.duration_seconds,
        speech_seconds=mask.speech_seconds,
        speech_ratio=mask.speech_ratio,
        sample_rate=decoded.sample_rate,
        file_bytes=target.stat().st_size if file_bytes is None else file_bytes,
        quality=quality,
    )


def _rejection(record: SourceRecord, code: str, reason: str) -> Rejection:
    """Build a rejection, keeping the record's identity in one place."""
    return Rejection(
        sample_id=record.sample_id,
        dataset_id=record.dataset_id,
        audio_path=record.audio_path,
        code=code,
        reason=reason,
    )


def _first_failure(
    record: SourceRecord,
    measurement: FileMeasurement,
    config: ValidationConfig,
    audio_config: AudioConfig,
) -> Rejection | None:
    """The highest-precedence failing check for one measured file, or ``None``.

    The order below is the contract. Each entry is cheaper or more fundamental
    than the ones after it, so the reported reason is the one an operator can act
    on directly rather than a downstream symptom.

    1. Required metadata -- a file that cannot be described cannot be analysed.
    2. File size, which is also checked before decoding by
       :func:`validate_sources`; there it is an optimisation, here it is part of
       the rule so a direct caller gets the same answer.
    3. Duration bounds. A file under the minimum is rejected on duration whatever
       its speech content, so reporting the speech shortfall would point at a
       threshold that was never the constraint.
    4. Blocking quality issues, then the speech floor. "This recording is
       silence" is the more fundamental statement than "this recording has too
       little speech".
    5. Quality warnings, which are warnings by construction and are only fatal
       when the configuration says so.

    There is no label check here, and the absence is deliberate.
    :meth:`SourceRecord.__post_init__` and :meth:`SourceRecord.from_dict` both
    refuse to build a record whose label is not in
    :data:`~voxshield.data.labels.LABELS`, so a mislabelled file is a
    construction error raised where it can be corrected, not a corpus-level
    rejection. A ``MISSING_LABEL`` code in the vocabulary would promise a
    rejection the build is structurally incapable of producing, and a report that
    offered it would be making a claim about its own coverage.
    """
    for name in config.require_metadata:
        if name == "label":
            # Always required, and guaranteed to hold by construction, so the
            # generic loop must not report it a second time under a second code.
            continue
        value = getattr(record, name, None)
        if value is None or not str(value).strip() or str(value) == UNKNOWN:
            return _rejection(
                record,
                REJECT_MISSING_METADATA,
                f"required metadata {name!r} is unknown; publish it in the adapter "
                f"or remove {name!r} from validation.require_metadata",
            )

    if measurement.file_bytes > config.max_file_bytes:
        return _rejection(
            record,
            REJECT_FILE_TOO_LARGE,
            f"file is {measurement.file_bytes} bytes, above "
            f"validation.max_file_bytes of {config.max_file_bytes}",
        )

    if measurement.duration_seconds < config.min_duration_seconds:
        return _rejection(
            record,
            REJECT_TOO_SHORT,
            f"duration {_seconds(measurement.duration_seconds)} is below "
            f"validation.min_duration_seconds of "
            f"{_seconds(config.min_duration_seconds)}",
        )

    if measurement.duration_seconds > config.max_duration_seconds:
        return _rejection(
            record,
            REJECT_TOO_LONG,
            f"duration {_seconds(measurement.duration_seconds)} is above "
            f"validation.max_duration_seconds of "
            f"{_seconds(config.max_duration_seconds)}",
        )

    blocking = measurement.blocking_issues
    if blocking:
        return _rejection(
            record,
            REJECT_UNUSABLE_SIGNAL,
            f"audio carries blocking quality issues {list(blocking)} and contains "
            "no analysable signal",
        )

    # The Phase 1 minimum, not a validation-local number. A training file has to
    # fill one analysis window with speech, and ``DataConfig.audio_config`` is
    # what sets that minimum to the window, so reading it from there keeps one
    # definition of "enough speech" across Phase 1 and Phase 2.
    minimum_speech = audio_config.min_speech_seconds
    if config.require_speech and measurement.speech_seconds < minimum_speech:
        return _rejection(
            record,
            REJECT_NO_SPEECH,
            f"{_seconds(measurement.speech_seconds)} of speech is below the "
            f"{_seconds(minimum_speech)} needed to fill one analysis window; "
            "lower audio_config.min_speech_seconds, shorten the window, or turn "
            "off validation.require_speech",
        )

    warnings = measurement.warnings
    if config.reject_warning_issues and warnings:
        return _rejection(
            record,
            REJECT_QUALITY_WARNING,
            f"quality issues {list(warnings)} are warnings and "
            "validation.reject_warning_issues is on",
        )

    return None


def validate_measurement(
    record: SourceRecord,
    measurement: FileMeasurement,
    config: ValidationConfig | None = None,
    audio_config: AudioConfig | None = None,
) -> Rejection | None:
    """Judge one already-measured file.

    Split out from :func:`validate_sources` so a later stage that has already
    measured a file -- preprocessing, or a cache hit -- reuses the measurement
    instead of decoding the audio again. The rules and the precedence are the
    same; only the reading of the file is skipped.

    Args:
        record: The candidate.
        measurement: What :func:`measure_file` returned for it.
        config: Validation policy. Defaults to :class:`ValidationConfig`.
        audio_config: Phase 1 configuration supplying the speech minimum.

    Returns:
        The first failing check, or ``None`` when the file is admissible.
    """
    cfg = config or ValidationConfig()
    audio = audio_config or AudioConfig()
    return _first_failure(record, measurement, cfg, audio)


def _decode_rejection(record: SourceRecord, exc: Exception) -> Rejection:
    """Map an intake failure onto a code that names the kind of failure.

    The message from :mod:`voxshield.errors` is kept verbatim in the reason: it is
    already written to be acted on ("audio is a 120.0s recording, exceeding the
    60.0s limit"), and paraphrasing it here would only add a chance to disagree
    with it.
    """
    if isinstance(exc, UnsupportedAudioFormatError):
        code = REJECT_UNSUPPORTED_FORMAT
    elif isinstance(exc, InvalidAudioSignalError):
        code = REJECT_UNUSABLE_SIGNAL
    elif isinstance(exc, AudioTooLargeError):
        code = REJECT_AUDIO_BUDGET
    else:
        code = REJECT_UNDECODABLE
    return _rejection(record, code, str(exc) or type(exc).__name__)


def _resolve(root: Path, audio_path: str) -> Path:
    """Locate a file from a record's ``audio_path``.

    Mirrors :func:`voxshield.data.discovery._resolve`: manifests are portable, so
    a relative path is resolved against the data root, while a corpus on a
    separate volume is used as given.
    """
    candidate = Path(audio_path)
    return candidate if candidate.is_absolute() else root / candidate


def _require_metadata_fields(config: ValidationConfig) -> None:
    """Refuse a requirement that names no field on the schema.

    A typo'd entry in ``require_metadata`` matches nothing and is therefore
    satisfied by every record, producing a build where the requirement was never
    applied and no report mentions it. That is a configuration error, not a
    per-file rejection, and it is raised before any audio is read.

    Raises:
        DatasetConfigError: If a required field is not a :class:`SourceRecord`
            field.
    """
    unknown = sorted(set(config.require_metadata) - _METADATA_FIELDS)
    if unknown:
        msg = (
            f"validation.require_metadata names field(s) {unknown} that are not "
            f"SourceRecord attributes; valid fields are {sorted(_METADATA_FIELDS)}"
        )
        raise DatasetConfigError(msg)


def _duplicates(
    accepted: Sequence[Acceptance],
    config: ValidationConfig,
    entries_by_id: Mapping[str, InventoryEntry],
    content_hashes: Mapping[str, str] | None,
) -> tuple[Rejection, ...]:
    """Reject repeated audio among the files that passed every other check.

    Duplicates are resolved *after* the other checks, and that ordering is
    deliberate: a copy of a file that is itself being rejected must not consume
    the kept slot, or a corpus could lose both copies of a recording over a
    problem that only one of them had.

    ``dedup_scope`` selects which identity is compared. ``"file"`` is the source
    file's bytes, which discovery has already measured. ``"content"`` is the
    stored segment's PCM, which does not exist until segmentation has run, so a
    file-stage build that asks for it is told the scope was unavailable rather
    than being handed a clean result it did not earn.

    Which copy survives is decided by ``sample_id`` order, so two machines
    resolve the conflict identically. A build that has run discovery already has
    no file duplicates left to find -- discovery keeps the better-described copy
    -- so this pass matters most for a caller that skipped it.

    Returns:
        One rejection per dropped copy, in ``sample_id`` order.
    """
    if not config.reject_duplicates:
        return ()

    file_scope = config.dedup_scope in (_SCOPE_FILE, "both")
    content_scope = config.dedup_scope in (_SCOPE_CONTENT, "both")

    seen: dict[str, str] = {}
    rejections: list[Rejection] = []
    for item in sorted(accepted, key=lambda entry: entry.sample_id):
        sample_id = item.sample_id
        file_hash = entries_by_id[sample_id].file_hash if file_scope else None
        content_hash = content_hashes.get(sample_id) if content_scope and content_hashes else None

        if file_hash is not None:
            owner = seen.get(f"file:{file_hash}")
            if owner is not None:
                rejections.append(
                    _rejection(
                        item.record,
                        REJECT_DUPLICATE_FILE,
                        f"file is byte-identical to {owner!r} ({file_hash}); the "
                        "same audio reached the build twice",
                    )
                )
                continue
            seen[f"file:{file_hash}"] = sample_id

        if content_hash is not None:
            owner = seen.get(f"content:{content_hash}")
            if owner is not None:
                rejections.append(
                    _rejection(
                        item.record,
                        REJECT_DUPLICATE_CONTENT,
                        f"stored audio is identical to {owner!r} "
                        f"({content_hash}); the same audio reached the build twice",
                    )
                )
                continue
            seen[f"content:{content_hash}"] = sample_id

    return tuple(rejections)


def _unavailable_scopes(
    config: ValidationConfig,
    accepted: Sequence[Acceptance],
    entries_by_id: Mapping[str, InventoryEntry],
    content_hashes: Mapping[str, str] | None,
) -> tuple[str, ...]:
    """Duplicate scopes that were requested but have nothing to compare.

    A scope with no keys is not a scope that passed. Reporting it as unavailable
    is what lets a build say "content-level duplicates were not checked at the
    file stage" instead of implying the corpus is duplicate-free at a level it was
    never examined at.

    An empty result is not unavailable, though, and the distinction is worth
    keeping: a corpus with nothing in it has no duplicate to find, so there is no
    check to report as skipped. Listing it anyway would put a permanent false
    alarm in the report of every empty or fully-refused build, and a build report
    that always carries one warning is a report nobody reads.
    """
    if not config.reject_duplicates or not accepted:
        return ()
    unavailable: list[str] = []
    if config.dedup_scope in (_SCOPE_FILE, "both") and not entries_by_id:
        unavailable.append(_SCOPE_FILE)
    if config.dedup_scope in (_SCOPE_CONTENT, "both") and not content_hashes:
        unavailable.append(_SCOPE_CONTENT)
    return tuple(unavailable)


def validate_sources(
    records: Sequence[SourceRecord],
    root: str | Path,
    config: ValidationConfig | None = None,
    *,
    audio_config: AudioConfig | None = None,
    inventory: DiscoveryResult | Sequence[InventoryEntry] | None = None,
    content_hashes: Mapping[str, str] | None = None,
) -> ValidationResult:
    """Decide which discovered files may become training samples.

    Args:
        records: Source records from the registry, in deterministic order.
        root: Data root used to resolve relative audio paths.
        config: Validation policy. Defaults to :class:`ValidationConfig`.
        audio_config: Phase 1 configuration. Should come from
            :meth:`~voxshield.data.config.DataConfig.audio_config` so the speech
            minimum is the analysis window.
        inventory: Discovery's measurements. A
            :class:`~voxshield.data.discovery.DiscoveryResult` or a plain sequence
            of entries. Reused rather than recomputed, so a build does not hash a
            corpus twice; discovered and measured on demand when omitted. Note
            that ``max_file_bytes`` is *not* forwarded to that pass: a file over
            the policy limit must be reported as :data:`REJECT_FILE_TOO_LARGE`
            here, not hidden as an unreadable file by discovery's own cap.
        content_hashes: Stored-segment digests by sample id, for the ``"content"``
            duplicate scope. Not available at the file stage; see
            :func:`_duplicates`.

    Returns:
        A :class:`ValidationResult`. No file is ever refused with an exception:
        everything that did not pass is in ``rejected`` with a code and a reason.

    Raises:
        DatasetBuildError: Two records share a ``sample_id``. Every result here is
            keyed by it, so duplicates would corrupt the counts and let one file
            answer for another.
        DatasetConfigError: ``require_metadata`` names a field that is not a
            :class:`SourceRecord` attribute.
    """
    cfg = config or ValidationConfig()
    audio = audio_config or AudioConfig()
    base = Path(root)
    _require_metadata_fields(cfg)

    seen_ids: set[str] = set()
    for record in records:
        if record.sample_id in seen_ids:
            msg = (
                f"validation was given two records with sample_id "
                f"{record.sample_id!r}; ids must be unique for the accepted and "
                "rejected sets to describe the corpus"
            )
            raise DatasetBuildError(msg)
        seen_ids.add(record.sample_id)

    if isinstance(inventory, DiscoveryResult):
        entries: Sequence[InventoryEntry] = inventory.entries
        unreadable = {item.sample_id: item.reason for item in inventory.unreadable}
    elif inventory is not None:
        entries = inventory
        unreadable = {}
    else:
        built, missing = build_inventory(records, base)
        entries = built
        unreadable = {item.sample_id: item.reason for item in missing}
    entries_by_id = {entry.sample_id: entry for entry in entries}

    accepted: list[Acceptance] = []
    rejected: list[Rejection] = []

    for record in records:
        entry = entries_by_id.get(record.sample_id)
        if entry is None:
            reason = unreadable.get(
                record.sample_id, "discovery produced no measurement for this file"
            )
            rejected.append(_rejection(record, REJECT_UNREADABLE, reason))
            continue

        # Size is checked before the decode purely to avoid spending decode time
        # on a file that is going to be refused. It reappears inside
        # _first_failure, so the outcome is identical either way -- the only thing
        # the pre-screen changes is how long a 4 GB file takes to be refused.
        if entry.file_bytes > cfg.max_file_bytes:
            rejected.append(
                _rejection(
                    record,
                    REJECT_FILE_TOO_LARGE,
                    f"file is {entry.file_bytes} bytes, above "
                    f"validation.max_file_bytes of {cfg.max_file_bytes}",
                )
            )
            continue

        try:
            measurement = measure_file(
                _resolve(base, record.audio_path), audio, file_bytes=entry.file_bytes
            )
        except (VoxShieldError, OSError) as exc:
            rejected.append(_decode_rejection(record, exc))
            continue

        failure = _first_failure(record, measurement, cfg, audio)
        if failure is None:
            accepted.append(Acceptance(record=record, measurement=measurement))
        else:
            rejected.append(failure)

    duplicates = _duplicates(accepted, cfg, entries_by_id, content_hashes)
    if duplicates:
        dropped = {item.sample_id for item in duplicates}
        accepted = [item for item in accepted if item.sample_id not in dropped]
        rejected.extend(duplicates)

    per_dataset: dict[str, int] = {}
    for item in accepted:
        per_dataset[item.dataset_id] = per_dataset.get(item.dataset_id, 0) + 1
    per_reason: dict[str, int] = {}
    for refusal in rejected:
        per_reason[refusal.code] = per_reason.get(refusal.code, 0) + 1

    stats = ValidationStats(
        total=len(records),
        accepted=len(accepted),
        rejected=len(rejected),
        accepted_per_dataset=dict(sorted(per_dataset.items())),
        rejected_per_reason=dict(sorted(per_reason.items())),
        duration_seconds=sum(item.duration_seconds for item in accepted),
        speech_seconds=sum(item.speech_seconds for item in accepted),
        unavailable_scopes=_unavailable_scopes(cfg, accepted, entries_by_id, content_hashes),
    )
    return ValidationResult(
        accepted=tuple(sorted(accepted, key=lambda item: item.sample_id)),
        rejected=tuple(sorted(rejected, key=lambda item: (item.code, item.sample_id))),
        stats=stats,
    )
