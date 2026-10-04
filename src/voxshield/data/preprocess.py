"""Phase 2 segmentation: turn a validated source file into manifest rows.

Validation admitted a source; this module turns it into the rows that actually
enter the dataset. The ordering rule that governs this module is stated plainly
because it is the one that most corrupts a corpus when ignored:

    **Splits are assigned to sources, never to segments.**

Every segment inherits its split from its parent source file. The batch entry
point :func:`preprocess_sources` refuses to run without an explicit source-to-
split mapping, so a build cannot "assign segments to splits at the end" as an
afterthought that quietly ignores which speaker is which. This is what keeps the
test split a true speaker-disjoint group: the split decision is made before the
acoustic content is ever inspected, and no later step redistributes rows.

A source becomes either an acceptance (a tuple of :class:`SampleRecord`, one per
second of windowed speech) or a rejection carrying a structured code and reason.
Rejections are returned, not raised, matching the validation stage's contract:
a build ingests thousands of files and wants a tally, not a first failure.

The cache integration is deliberately narrow. The expensive half of Phase 1 --
decode and resample -- is a pure function of the file bytes and the audio
configuration, and is stored under ``cache_key(file_hash, signature)`` where
``signature`` is a digest of every audio flag that can change a sample. The
cheap half -- VAD and windowing -- is always re-run on a cache hit, so a segment
manifest is bit-for-bit what a fresh build would produce, including this build's
own ``preprocessing_version`` stamp.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from voxshield.audio.pipeline import PreparedAudio, prepare_from_bytes
from voxshield.audio.preprocess import PreprocessedAudio
from voxshield.audio.segmentation import segment_speech
from voxshield.audio.vad import detect_speech
from voxshield.config import AudioConfig
from voxshield.data.cache import (
    CachedAudio,
    PreprocessingCache,
    cache_key,
    preprocessing_signature,
)
from voxshield.data.config import DataConfig
from voxshield.data.errors import SplitError
from voxshield.data.paths import DataPaths
from voxshield.data.schema import SampleRecord, SourceRecord
from voxshield.errors import (
    AudioDecodeError,
    AudioTooLargeError,
    InsufficientSpeechError,
    InvalidAudioSignalError,
    UnsupportedAudioFormatError,
)

__all__ = [
    "PREPROCESS_REJECTIONS",
    "PreprocessedSource",
    "build_sample_records",
    "preprocess_source",
    "preprocess_source_bytes",
    "preprocess_sources",
    "preprocessing_version",
    "segment_content_hash",
    "write_segment_audio",
]

#: Rejection codes this stage can produce. Distinct from the validation-stage
#: codes so a manifest reader can tell which phase turned a file away.
REJECT_INSUFFICIENT_SPEECH = "insufficient_speech"
REJECT_TOO_LARGE = "too_large"
REJECT_UNSUPPORTED_FORMAT = "unsupported_format"
REJECT_UNDECODABLE = "undecodable"
REJECT_INVALID_SIGNAL = "invalid_signal"
REJECT_UNREADABLE = "unreadable"

PREPROCESS_REJECTIONS = frozenset(
    {
        REJECT_INSUFFICIENT_SPEECH,
        REJECT_TOO_LARGE,
        REJECT_UNSUPPORTED_FORMAT,
        REJECT_UNDECODABLE,
        REJECT_INVALID_SIGNAL,
        REJECT_UNREADABLE,
    }
)


def _hash_bytes(payload: bytes) -> str:
    """The ``"sha256:<hex>"`` form ``file_hash`` uses, for bytes we already hold.

    Computing the digest over the payload avoids a second read of a file that is
    about to be decoded. For the same bytes this is identical to
    :func:`voxshield.data.discovery.file_hash`'s output -- the discovery-stage
    hash and the preprocessing-stage hash both mean "this exact byte stream".
    """
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def segment_content_hash(samples: np.ndarray) -> str:
    """Content hash of one segment window, indexing the canonical signal.

    The window is the float32 signal at the canonical sample rate, hashed as
    little-endian IEEE-754 bytes. Hashing the canonical window rather than the
    stored WAV's PCM keeps the digest stable across storage subtypes (``PCM_16``
    vs ``PCM_24``) so a re-encode of the same window is still the same content.
    """
    arr = np.ascontiguousarray(samples, dtype="<f4")
    return f"sha256:{hashlib.sha256(arr.tobytes()).hexdigest()}"


@dataclass(frozen=True, slots=True)
class PreprocessedSource:
    """One source file's outcome in the segmentation phase.

    Exactly one of :attr:`segments` (accepted) or :attr:`rejected_code`
    (rejected) is meaningful; :attr:`accepted` is the single source of truth.

    Attributes:
        source: The source record this outcome describes.
        split: The split its segments belong to.
        file_hash: ``"sha256:<hex>"`` of the source bytes.
        preprocessing_version: The Phase 1 signature the segments were built with.
        segments: Accepted rows, empty on rejection.
        from_cache: Whether the canonical audio came from the preprocessing cache.
        rejected_code: A :data:`PREPROCESS_REJECTIONS` code, or ``None``.
        rejected_reason: Human-readable explanation for a rejection.
    """

    source: SourceRecord
    split: str
    file_hash: str
    preprocessing_version: str
    segments: tuple[SampleRecord, ...] = ()
    from_cache: bool = False
    rejected_code: str | None = None
    rejected_reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.rejected_code is None

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary for a build report."""
        return {
            "sample_id": self.source.sample_id,
            "split": self.split,
            "status": "accepted" if self.accepted else "rejected",
            "rejected_code": self.rejected_code,
            "rejected_reason": self.rejected_reason,
            "n_segments": self.n_segments,
            "from_cache": self.from_cache,
            "file_hash": self.file_hash,
            "preprocessing_version": self.preprocessing_version,
        }


def preprocessing_version(audio_config: AudioConfig | None = None) -> str:
    """Short, stable stamp for the Phase 1 configuration a manifest row claims.

    The full architectural guard is the 64-hex :func:`cache_key`; this is the
    human-facing cousin recorded on every ``SampleRecord`` so a manifest row can
    be traced to the exact audio flags that produced it, even when the cache is
    long gone. The first 16 hex characters keep manifests printable while still
    uniquely identifying a configuration in practice.
    """
    return f"phase1.{preprocessing_signature(audio_config)[:16]}"


def _resolve(path: str, root: Path) -> Path:
    """Absolute path for ``path``, honouring ``DataPaths`` portability rules."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / path


def _prepare_from_canonical(audio: PreprocessedAudio, cfg: AudioConfig) -> PreparedAudio:
    """Rebuild a :class:`PreparedAudio` from already-canonical samples.

    Used on a cache hit: the samples are already at the canonical rate and
    loudness, so the expensive decode/resample half is skipped, while VAD and
    windowing re-run over the exact same values a fresh build would feed them.
    The speech and window gates are enforced here, not assumed to be satisfied
    by cache membership -- a corrupted or hand-edited cache entry is recomputed,
    never trusted.
    """
    speech = detect_speech(audio.samples, audio.sample_rate, cfg)
    if speech.speech_seconds < cfg.min_speech_seconds:
        raise InsufficientSpeechError(
            speech_seconds=speech.speech_seconds,
            minimum_seconds=cfg.min_speech_seconds,
        )
    segments = segment_speech(
        n_samples=len(audio.samples),
        speech_mask=speech,
        sample_rate=audio.sample_rate,
        config=cfg,
    )
    if not segments:
        regions = speech.regions()
        longest_run = max((end - start for start, end in regions), default=0.0)
        if longest_run < cfg.min_segment_seconds:
            reason = InsufficientSpeechError.REASON_CONTIGUOUS
        elif audio.duration_seconds < cfg.segment_seconds:
            reason = InsufficientSpeechError.REASON_SHORT_WINDOW
        else:
            reason = InsufficientSpeechError.REASON_CONTIGUOUS
        raise InsufficientSpeechError(
            speech_seconds=speech.speech_seconds,
            minimum_seconds=cfg.min_segment_seconds,
            reason=reason,
            longest_run_seconds=longest_run,
            window_seconds=cfg.segment_seconds,
        )
    return PreparedAudio(
        preprocessed=audio,
        speech=speech,
        segments=segments,
        feature_config=cfg.features,
        source_metadata={"source_kind": "cached", "decoded_from_cache": True},
        window_samples=round(cfg.segment_seconds * audio.sample_rate),
    )


def build_sample_records(
    source: SourceRecord,
    prepared: PreparedAudio,
    *,
    split: str,
    dataset_build_id: str,
    file_hash: str,
    preprocessing_version: str,
    audio_paths: Sequence[str] | None = None,
) -> tuple[SampleRecord, ...]:
    """Build one :class:`SampleRecord` per window, inheriting the source.

    Every acoustic measurement comes from the segment the detector would score,
    never from re-derivation: ``speech_seconds`` and ``coverage`` are the window's
    own VAD totals, ``start_seconds`` indexes the canonical signal, and
    ``content_hash`` covers exactly the samples that are stored. Metadata is
    inherited wholesale from the source -- a window of a recording cannot have a
    different speaker than its parent, and storing the parent's ``speaker_id``
    (already namespaced at ingest) is what keeps leakage checks honest.

    Args:
        source: The parent source record.
        prepared: Phase 1 output for the source.
        split: The split these segments inherit.
        dataset_build_id: Build stamp for every row.
        file_hash: Source file hash, ``"sha256:<hex>"``.
        preprocessing_version: Phase 1 signature stamp.
        audio_paths: Stored segment paths, parallel to windows. When absent,
            every row falls back to the source's own path -- acceptable for
            provenance-only manifests, documented as *not* a data index.

    Returns:
        One record per window, in window order.
    """
    rows: list[SampleRecord] = []
    sample_rate = prepared.preprocessed.sample_rate
    for index, seg in enumerate(prepared.segments):
        window = prepared.segment_window(index)
        stored_audio = (
            audio_paths[index]
            if audio_paths is not None and index < len(audio_paths)
            else source.audio_path
        )
        rows.append(
            SampleRecord(
                sample_id=f"{source.sample_id}#{index:04d}",
                dataset_id=source.dataset_id,
                audio_path=stored_audio,
                label=source.label,
                label_index=source.label_index,
                split=split,
                parent_id=source.parent_id,
                segment_index=index,
                start_seconds=seg.start_sample / sample_rate,
                duration_seconds=window.size / sample_rate,
                sample_rate=sample_rate,
                speech_seconds=seg.speech_seconds,
                coverage=seg.coverage,
                is_padded=seg.is_padded,
                waveform_samples=window.size,
                speaker_id=source.speaker_id,
                generator_id=source.generator_id,
                language=source.language,
                codec=source.codec,
                session_id=source.session_id,
                attack_type=source.attack_type,
                channel=source.channel,
                device=source.device,
                recorded_at=source.recorded_at,
                file_hash=file_hash,
                content_hash=segment_content_hash(window),
                source_split=source.source_split,
                preprocessing_version=preprocessing_version,
                dataset_build_id=dataset_build_id,
                extra=source.extra,
            )
        )
    return tuple(rows)


def write_segment_audio(
    source: SourceRecord,
    prepared: PreparedAudio,
    *,
    segments_dir: Path,
    root: Path,
    storage_subtype: str = "PCM_16",
) -> list[str]:
    """Persist each window as a standardised WAV under ``processed/segments``.

    Arguments:
        source: Parent source, for the filename.
        prepared: Phase 1 output whose windows are written.
        segments_dir: Target directory; created if missing.
        root: Data root, for converting absolute paths to portable relative ones.
        storage_subtype: WAV subtype, mirrored into the manifest's storage note.

    Returns:
        Stored paths, one per window, relative to ``root`` when possible.
    """
    segments_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index in range(len(prepared.segments)):
        window = prepared.segment_window(index)
        name = f"{source.sample_id}__s{index:04d}.wav"
        target = segments_dir / name
        sf.write(target, window, prepared.preprocessed.sample_rate, subtype=storage_subtype)
        try:
            rel = target.relative_to(root)
        except ValueError:
            rel = target
        paths.append(rel.as_posix())
    return paths


def preprocess_source(
    source: SourceRecord,
    *,
    config: DataConfig,
    paths: DataPaths | None = None,
    cache: PreprocessingCache | None = None,
    split: str = "train",
    dataset_build_id: str = "",
) -> PreprocessedSource:
    """Segment one source file, respecting split-before-segmentation.

    Args:
        source: Validated source record.
        config: Build configuration. Supplies the audio configuration, the
            validation size bound, and the storage settings.
        paths: Resolved data layout. Defaults to ``config.data_paths()``.
        cache: Optional preprocessing cache; a enabled cache serves and stores
            canonical audio but never changes output.
        split: Split the segments inherit. Batch builds must go through
            :func:`preprocess_sources`, which enforces that every source has an
            assignment.
        dataset_build_id: Build stamp for the rows.

    Returns:
        An accepted or rejected :class:`PreprocessedSource`.
    """
    data_paths = paths or config.data_paths()
    root = Path(data_paths.root)
    target = _resolve(source.audio_path, root)

    try:
        size = target.stat().st_size
    except OSError as exc:
        return _reject(source, split, REJECT_UNREADABLE, f"cannot stat source: {exc}")
    if config.validation.max_file_bytes is not None and size > config.validation.max_file_bytes:
        return _reject(
            source,
            split,
            REJECT_TOO_LARGE,
            f"{size} bytes exceeds validation.max_file_bytes ({config.validation.max_file_bytes})",
        )

    try:
        payload = target.read_bytes()
    except OSError as exc:
        return _reject(source, split, REJECT_UNREADABLE, f"cannot read source: {exc}")

    return _preprocess_payload(
        payload,
        source,
        config=config,
        data_paths=data_paths,
        cache=cache,
        split=split,
        dataset_build_id=dataset_build_id,
        file_hash=_hash_bytes(payload),
    )


def preprocess_source_bytes(
    payload: bytes,
    source: SourceRecord,
    *,
    config: DataConfig,
    paths: DataPaths | None = None,
    cache: PreprocessingCache | None = None,
    split: str = "train",
    dataset_build_id: str = "",
) -> PreprocessedSource:
    """Segment a source given its bytes, for adapters that stage audio in memory.

    The size bound and file hash are computed over the supplied bytes. Stored
    segment audio still lands under ``processed/segments`` in the data layout.
    """
    return _preprocess_payload(
        payload,
        source,
        config=config,
        data_paths=paths or config.data_paths(),
        cache=cache,
        split=split,
        dataset_build_id=dataset_build_id,
        file_hash=_hash_bytes(payload),
    )


def _reject(
    source: SourceRecord,
    split: str,
    code: str,
    reason: str,
) -> PreprocessedSource:
    return PreprocessedSource(
        source=source,
        split=split,
        file_hash="",
        preprocessing_version="",
        rejected_code=code,
        rejected_reason=reason,
    )


def _preprocess_payload(
    payload: bytes,
    source: SourceRecord,
    *,
    config: DataConfig,
    data_paths: DataPaths,
    cache: PreprocessingCache | None,
    split: str,
    dataset_build_id: str,
    file_hash: str,
) -> PreprocessedSource:
    audio_config = config.audio_config()
    stamp = preprocessing_version(audio_config)
    signature = preprocessing_signature(audio_config)
    key = cache_key(file_hash, signature)

    cached_audio: CachedAudio | None = None
    if cache is not None:
        cached_audio = cache.get(key)

    if cached_audio is not None:
        try:
            prepared = _prepare_from_canonical(cached_audio.to_preprocessed(), audio_config)
        except InsufficientSpeechError as exc:
            return _reject(source, split, REJECT_INSUFFICIENT_SPEECH, str(exc))
        from_cache = True
    else:
        try:
            prepared = prepare_from_bytes(payload, audio_config)
        except AudioTooLargeError as exc:
            return _reject(source, split, REJECT_TOO_LARGE, str(exc))
        except UnsupportedAudioFormatError as exc:
            return _reject(source, split, REJECT_UNSUPPORTED_FORMAT, str(exc))
        except AudioDecodeError as exc:
            return _reject(source, split, REJECT_UNDECODABLE, str(exc))
        except InvalidAudioSignalError as exc:
            return _reject(source, split, REJECT_INVALID_SIGNAL, str(exc))
        except InsufficientSpeechError as exc:
            return _reject(source, split, REJECT_INSUFFICIENT_SPEECH, str(exc))
        from_cache = False
        if cache is not None:
            cache.put(key, prepared.preprocessed)

    audio_paths: Sequence[str] | None = None
    if config.write_segment_audio:
        segments_dir = data_paths.segments_dir(source.dataset_id)
        root = Path(data_paths.root)
        audio_paths = write_segment_audio(
            source,
            prepared,
            segments_dir=segments_dir,
            root=root,
            storage_subtype=config.storage_subtype,
        )

    rows = build_sample_records(
        source,
        prepared,
        split=split,
        dataset_build_id=dataset_build_id,
        file_hash=file_hash,
        preprocessing_version=stamp,
        audio_paths=audio_paths,
    )
    return PreprocessedSource(
        source=source,
        split=split,
        file_hash=file_hash,
        preprocessing_version=stamp,
        segments=rows,
        from_cache=from_cache,
    )


def preprocess_sources(
    records: Sequence[SourceRecord],
    *,
    config: DataConfig,
    split_of: Mapping[str, str],
    paths: DataPaths | None = None,
    cache: PreprocessingCache | None = None,
    dataset_build_id: str = "",
) -> list[PreprocessedSource]:
    """Segment a batch, enforcing that every source has a split assignment.

    Args:
        records: Validated source records.
        config: Build configuration.
        split_of: Source ``sample_id`` to split. Missing entries are a defect:
            this is the enforcement point of split-before-segmentation.
        paths: Resolved data layout.
        cache: Optional preprocessing cache.
        dataset_build_id: Build stamp for the rows.

    Raises:
        SplitError: A source has no split assignment.
    """
    data_paths = paths or config.data_paths()
    cache_instance = cache if cache is not None else PreprocessingCache.for_config(config)
    outcomes: list[PreprocessedSource] = []
    for record in records:
        split = split_of.get(record.sample_id)
        if split is None:
            msg = (
                f"source {record.sample_id!r} has no split assignment; assign splits "
                "to sources before segmentation so the test split stays a whole-group, "
                "speaker-disjoint set"
            )
            raise SplitError(msg)
        outcomes.append(
            preprocess_source(
                record,
                config=config,
                paths=data_paths,
                cache=cache_instance,
                split=split,
                dataset_build_id=dataset_build_id,
            )
        )
    return outcomes
