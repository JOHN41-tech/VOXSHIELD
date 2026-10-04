"""The adapter contract: one place that knows a corpus's on-disk conventions.

    Everything downstream of discovery -- validation, splitting, manifests, leakage
    checks -- speaks only :class:`~voxshield.data.schema.SourceRecord`. Nothing
    downstream knows what a ``progress.txt`` column means or that WaveFake puts its
    vocoders in the directory name. That knowledge is quarantined here, in one
    subclass per corpus, so a new corpus costs one file rather than a sweep through
    the pipeline.

    Three responsibilities, in the order they matter:

    * **Availability.** :meth:`DatasetAdapter.is_available` answers "is this corpus on
      this machine" without decoding any audio, so a build on a laptop holding
      LibriSpeech but not ASVspoof degrades to a report instead of an exception.
* **Discovery.** :meth:`DatasetAdapter.discover` yields one
  :class:`SourceRecord` per audio file, carrying only the metadata the corpus
  actually published. Metadata the corpus does not publish is :data:`UNKNOWN`,
  and an adapter that guesses instead is the single most damaging thing it could
  do -- see :func:`~voxshield.data.schema.namespace_speaker` for why.
* **Self-description.** :meth:`DatasetAdapter.describe` states what the adapter
  expects to find, so ``inspect`` can distinguish "not installed" from
  "installed but laid out differently than this adapter assumes".

**There is deliberately no ``download()``.** Every corpus in the registry is
gated behind a click-through licence agreement, several of them forbid
redistribution, and two are tens of gigabytes. A pipeline that fetches its own
training data cannot tell a *missing* licence from a *missing* disk, and
"REQUIRES_VERIFICATION" would then be a string in a config that nothing enforces.
Acquisition stays a human step; this module verifies what arrived.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from voxshield.data.config import DatasetEntry
from voxshield.data.errors import AdapterError, DatasetUnavailableError
from voxshield.data.schema import UNKNOWN, SourceRecord, known_or_unknown

__all__ = [
    "AUDIO_EXTENSIONS",
    "PHASE1_FORMATS",
    "AdapterDescription",
    "DatasetAdapter",
    "as_sequence",
    "container_format_for",
]

#: Extensions an adapter will consider at all. Deliberately wider than Phase 1's
#: allow-list: a corpus that ships MP3 should be *discovered* and then rejected
#: by validation with a reason naming the format, rather than being invisible in
#: the inventory. A file that cannot be seen cannot be diagnosed.
AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".wav",
        ".flac",
        ".ogg",
        ".oga",
        ".opus",
        ".mp3",
        ".m4a",
        ".mp4",
        ".aac",
        ".aiff",
        ".aif",
        ".aifc",
        ".au",
        ".snd",
        ".w64",
        ".caf",
        ".pcm",
        ".raw",
    }
)

#: Extensions Phase 1 can actually decode. Used to pre-classify files in the
#: inventory so the ratio of "incompatible container" to "corrupt file" is visible
#: without decoding anything.
PHASE1_FORMATS: frozenset[str] = frozenset({".wav", ".flac"})

#: Container names as libsndfile reports them, keyed by extension. Only the
#: formats Phase 1 accepts are named; anything else is reported by the extension
#: so a reader can see which codec a file claimed to be without decoding it.
_CONTAINER_NAMES: dict[str, str] = {
    ".wav": "WAV",
    ".wave": "WAV",
    ".flac": "FLAC",
    ".ogg": "OGG",
    ".oga": "OGG",
    ".opus": "OGG",
    ".aiff": "AIFF",
    ".aif": "AIFF",
    ".aifc": "AIFF",
    ".au": "AU",
    ".snd": "AU",
    ".w64": "WAV",
    ".caf": "CAF",
}


def container_format_for(path: Path | str) -> str:
    """The container a file's extension claims, or :data:`UNKNOWN`.

    Reported as *claimed*, never as verified. The container is confirmed during
    validation by Phase 1's own decoder, and a ``codec`` field filled in from the
    extension would let a mislabelled ``.wav`` that is really an MP3 carry a codec
    the dataset statistics then trust.

    Args:
        path: File path or name.

    Returns:
        A libsndfile-style container name, or :data:`UNKNOWN`.
    """
    return _CONTAINER_NAMES.get(Path(path).suffix.lower(), UNKNOWN)


@dataclass(frozen=True, slots=True)
class AdapterDescription:
    """What an adapter found, or expected to find, on disk.

    Every field is a scalar or a list of strings. This object is written into the
    build report and printed by ``inspect``, so it must be safe to publish --
    which is why it describes *layout*, never the audio itself.
    """

    dataset_id: str
    adapter: str
    task: str
    version: str
    root: str
    available: bool
    audio_file_count: int
    total_bytes: int
    extensions: tuple[str, ...]
    expected_layout: tuple[str, ...]
    layout_hints_found: tuple[str, ...] = ()
    metadata_files: tuple[str, ...] = ()
    notes: str = ""
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form, for the build report."""
        return {
            "dataset_id": self.dataset_id,
            "adapter": self.adapter,
            "task": self.task,
            "version": self.version,
            "root": self.root,
            "available": self.available,
            "audio_file_count": self.audio_file_count,
            "total_bytes": self.total_bytes,
            "extensions": list(self.extensions),
            "expected_layout": list(self.expected_layout),
            "layout_hints_found": list(self.layout_hints_found),
            "metadata_files": list(self.metadata_files),
            "notes": self.notes,
            "warnings": list(self.warnings),
        }


class DatasetAdapter(ABC):
    """Base class for a corpus-specific reader.

    Subclasses implement :meth:`discover` and usually :meth:`expected_layout`.
    The directory-walking, id-minting, and header-probing helpers here are
    shared, because getting them subtly differently per corpus is how two datasets
    end up with incompatible ``sample_id`` conventions.

    Attributes:
        name: Registry name. Must match the ``adapter:`` field of a
            :class:`~voxshield.data.config.DatasetEntry`.
        task: ``"spoof_detection"`` or ``"real_speech"``. Checked against the
            entry at construction, because an entry declaring a real-speech corpus
            behind a spoof-detection adapter produces a corpus with no negatives.
    """

    #: Registry name for this adapter.
    name: ClassVar[str] = ""

    #: Which task this adapter produces records for.
    task: ClassVar[str] = "spoof_detection"

    def __init__(self, entry: DatasetEntry, data_root: Path) -> None:
        """Bind the adapter to a configured entry.

        Args:
            entry: The registry entry.
            data_root: Absolute data root. An entry ``path`` is resolved against
                it, unless the entry gives an absolute path, which is how a corpus
                on a separate volume is pointed at.

        Raises:
            AdapterError: If the entry names a different adapter, or a task that
                disagrees with this class.
        """
        if entry.adapter != self.name:
            msg = (
                f"dataset {entry.dataset_id!r} declares adapter {entry.adapter!r}, "
                f"but adapter {self.name!r} was requested"
            )
            raise AdapterError(msg)
        if entry.task != self.task:
            msg = (
                f"dataset {entry.dataset_id!r} declares task {entry.task!r}, but "
                f"adapter {self.name!r} produces {self.task!r} records"
            )
            raise AdapterError(msg)

        self.entry = entry
        self.dataset_id = entry.dataset_id
        self.metadata: dict[str, str] = dict(entry.metadata)
        # Stored rather than re-derived: a record's ``audio_path`` is relative to
        # the *data root*, which is a property of the configuration and not of the
        # entry, so recovering it from the entry later is a bug waiting to happen
        # for any entry whose ``path`` is absolute.
        self.data_root = data_root
        self.root = self._resolve_root(entry, data_root)

    @staticmethod
    def _resolve_root(entry: DatasetEntry, data_root: Path) -> Path:
        declared = Path(entry.path)
        return declared if declared.is_absolute() else data_root / declared

    # -- configuration knobs ----------------------------------------------

    @property
    def extensions(self) -> tuple[str, ...]:
        """Extensions to consider.

        Overridable per entry through ``metadata.extensions`` (comma-separated),
        because a corpus delivered in a non-standard container should be
        ingestible without editing this file.
        """
        raw = self.metadata.get("extensions", "")
        if raw.strip():
            cleaned = tuple(
                f".{part.strip().lower().lstrip('.')}" for part in raw.split(",") if part.strip()
            )
            return cleaned
        return tuple(sorted(AUDIO_EXTENSIONS))

    def flag(self, name: str, default: str = "") -> str:
        """Read a string-valued metadata setting, falling back to ``default``."""
        value = self.metadata.get(name, "")
        return value.strip() if value.strip() else default

    def int_flag(self, name: str, default: int) -> int:
        """Read an integer metadata setting, falling back to ``default``.

        A malformed value is an :class:`AdapterError` rather than a silent
        fallback: an operator who wrote ``max_files: 100x`` needs to be told, not
        quietly given the default.
        """
        raw = self.metadata.get(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError as exc:
            msg = f"dataset {self.dataset_id!r} metadata {name}={raw!r} is not an integer"
            raise AdapterError(msg) from exc

    def bool_flag(self, name: str, default: bool) -> bool:
        """Read a boolean metadata setting, falling back to ``default``."""
        raw = self.metadata.get(name, "").strip().lower()
        if not raw:
            return default
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        msg = f"dataset {self.dataset_id!r} metadata {name}={raw!r} is not a boolean"
        raise AdapterError(msg)

    # -- availability ------------------------------------------------------

    def is_available(self) -> bool:
        """Whether the corpus is present on this machine."""
        return self.root.is_dir()

    def require_available(self) -> Path:
        """Return :attr:`root`, or raise if the corpus is absent.

        Raises:
            DatasetUnavailableError: If the directory does not exist.
        """
        if not self.is_available():
            msg = (
                f"dataset {self.dataset_id!r} is not available at {self.root}; place "
                f"the corpus there, or set dataset {self.dataset_id!r} .path to its "
                "location"
            )
            raise DatasetUnavailableError(msg)
        return self.root

    def expected_layout(self) -> tuple[str, ...]:
        """Relative paths or glob patterns this adapter looks for.

        Purely descriptive: used by ``inspect`` and by
        :meth:`describe` to report which conventions were actually found. An
        adapter whose layout matches is trustworthy; one whose layout does not is
        a candidate for the mislabelled-corpus failure, and saying so is more
        useful than emitting zero rows.
        """
        return ()

    def layout_hints(self) -> tuple[str, ...]:
        """Which of :meth:`expected_layout` patterns are present."""
        if not self.is_available():
            return ()
        root = self.root
        found: list[str] = []
        for hint in self.expected_layout():
            if any(ch in hint for ch in "*?["):
                if any(root.glob(hint)):
                    found.append(hint)
            elif (root / hint).exists():
                found.append(hint)
        return tuple(found)

    def metadata_files(self) -> tuple[str, ...]:
        """Dataset-provided sidecar files this adapter parsed, relative to root."""
        return ()

    # -- walking -----------------------------------------------------------

    def iter_audio_files(self) -> Iterator[Path]:
        """Yield audio files under :attr:`root`, in a deterministic order.

        Sorting is not cosmetic. Discovery order decides which copy of a duplicate
        survives ``dedup``, and a filesystem's own enumeration order is not stable
        across machines, so a duplicate policy that keeps "the first one seen"
        keeps a *different* file on a different machine unless the order is fixed.

        Symbolic links to directories are not followed: a corpus that links to its
        parent would otherwise walk forever, and a self-referential link would
        hang a build rather than fail it.
        """
        if not self.is_available():
            return
        allowed = set(self.extensions)
        candidates: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self.root, followlinks=False):
            # Prune in place so os.walk does not descend into them. Hidden
            # directories are skipped because a corpus's ``.git`` and cache
            # directories are not part of its data.
            dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
            for filename in sorted(filenames):
                candidate = Path(dirpath) / filename
                if candidate.suffix.lower() in allowed:
                    candidates.append(candidate)
        yield from sorted(candidates)

    def mint_sample_id(self, path: Path) -> str:
        """A stable, dataset-unique id for one file.

        Derived from the path relative to :attr:`root`, not from the file's
        metadata: an id that changed when a file moved would silently orphan every
        cache entry and manifest row that referred to it. The dataset id is
        included because two corpora both containing ``1.wav`` must not collide.
        """
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            # An entry whose ``path`` points at a subdirectory can still yield
            # files outside it only if the walk were misconfigured; fall back to
            # the resolved absolute path rather than crashing discovery.
            relative = path
        flattened = "_".join(part for part in relative.parts)
        stem = flattened[: -len(path.suffix)] if path.suffix else flattened
        return f"{self.dataset_id}:{stem}"

    def relative_audio_path(self, path: Path) -> str:
        """The path as recorded in a record, relative to the data root.

        Kept relative so a manifest built on a workstation loads on a training
        host. An entry whose ``path`` is absolute and lives outside the data root
        necessarily yields an absolute path, which is a real situation -- a corpus
        on a separate volume -- so the absolute result is returned rather than
        raising, and the build report can see it.
        """
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return path.as_posix()
        return (Path(self.entry.path) / relative).as_posix()

    # -- record construction ------------------------------------------------

    def make_record(
        self,
        path: Path,
        *,
        label: str,
        sample_id: str | None = None,
        parent_id: str = "",
        speaker_id: str = UNKNOWN,
        generator_id: str = UNKNOWN,
        language: str = UNKNOWN,
        codec: str = "",
        session_id: str = UNKNOWN,
        attack_type: str = UNKNOWN,
        channel: str = UNKNOWN,
        device: str = UNKNOWN,
        source_split: str = UNKNOWN,
        recorded_at: str = UNKNOWN,
        extra: dict[str, str] | None = None,
    ) -> SourceRecord:
        """Build a :class:`SourceRecord` with the adapter's conventions applied.

        The speaker namespace, the unknown-value normalisation, and the
        parent-id default all live in
        :class:`~voxshield.data.schema.SourceRecord`. An adapter's job is to pass
        what the corpus published, and this helper exists so that it cannot forget
        the parts that are not its business.

        Args:
            path: The audio file.
            label: ``"bona_fide"`` or ``"spoof"``.
            sample_id: Override the derived id.
            parent_id: Grouping id for splitting. Defaults to the sample id, which
                is the honest answer for a corpus of independent utterances.
            speaker_id, generator_id, language, session_id, attack_type, channel,
            device, source_split, recorded_at: Published metadata, or
            :data:`UNKNOWN`.
            codec: Override the container inferred from the file extension. An
                adapter that knows the real container better than the extension
                does should pass it here.
            extra: Dataset-specific passthrough.

        Returns:
            A :class:`SourceRecord` with duration and sample rate probed from the
            container header when the probe succeeds, and left as ``None`` when it
            does not.
        """
        duration, sample_rate = self.probe_header(path)
        record_extra = {
            "adapter": self.name,
            "dataset_version": self.entry.version,
        }
        if extra:
            record_extra.update(extra)
        # The container is taken from the file's own extension. An adapter may know
        # better (a corpus that stores WAV under a ``.data`` suffix), and may
        # override via the ``codec`` argument; it may not know better, in which
        # case ``unknown`` is the honest answer. Recorded as a claim either way --
        # Phase 1's decoder is what verifies it, and a verified container that
        # disagreed with the claim would surface during validation.
        if self.bool_flag("codec_from_extension", True):
            codec_value = container_format_for(path)
        else:
            codec_value = UNKNOWN
        return SourceRecord(
            sample_id=sample_id or self.mint_sample_id(path),
            dataset_id=self.dataset_id,
            audio_path=self.relative_audio_path(path),
            label=label,
            parent_id=parent_id or sample_id or self.mint_sample_id(path),
            speaker_id=known_or_unknown(speaker_id),
            generator_id=known_or_unknown(generator_id),
            language=known_or_unknown(language),
            codec=known_or_unknown(codec or codec_value),
            session_id=known_or_unknown(session_id),
            attack_type=known_or_unknown(attack_type),
            channel=known_or_unknown(self.declared("channel", channel)),
            device=known_or_unknown(self.declared("device", device)),
            source_split=known_or_unknown(source_split),
            recorded_at=known_or_unknown(recorded_at),
            duration_seconds=duration,
            sample_rate=sample_rate,
            extra=record_extra,
        )

    def declared(self, name: str, found: str = UNKNOWN) -> str:
        """An entry-level ``metadata`` value, used only when discovery found none.

        Some corpora publish channel or handset populations once, in a paper or a
        licence appendix, rather than per file, so no adapter can extract them
        during discovery and every record would be :data:`UNKNOWN` -- which makes
        ``split.require_channel_disjoint`` unsatisfiable for a corpus that in fact
        has one documented population. An operator can state it once in the entry
        instead of forking an adapter.

        The fallback applies to *every* file in the entry, uniformly. That is what
        makes it usable for a whole-population claim and exactly why it must not be
        used to label a mixed corpus: a mixed corpus declared as one value is not
        half-right, it is a false statement that survives into every manifest row.
        Published per-file metadata always wins over the declaration.

        Args:
            name: The metadata key, e.g. ``"channel"``.
            found: What discovery extracted for this file.

        Returns:
            ``found`` when it is a real value, otherwise the entry-level
            declaration, otherwise :data:`UNKNOWN`.
        """
        if known_or_unknown(found) != UNKNOWN:
            return found
        return self.flag(name, UNKNOWN)

    def probe_header(self, path: Path) -> tuple[float | None, int | None]:
        """Read duration and sample rate from a container header.

        Header-only, so it is cheap, and failures are swallowed: a file Phase 1
        will reject anyway should appear in the inventory as a file with unknown
        duration, not vanish from it. Turning a header probe into a hard failure
        would mean a single truncated file prevented the discovery report from
        being written, which is exactly the report needed to diagnose it.
        """
        if self.bool_flag("probe_headers", True):
            try:
                import soundfile as sf

                info = sf.info(str(path))
            except Exception:
                return None, None
            frames = int(info.frames)
            rate = int(info.samplerate)
            if rate <= 0:
                return None, None
            return frames / float(rate), rate
        return None, None

    # -- discovery ----------------------------------------------------------

    @abstractmethod
    def discover(self, *, max_files: int | None = None) -> Iterator[SourceRecord]:
        """Yield one :class:`SourceRecord` per usable candidate file.

        Adapters are expected to yield *candidates*: files whose label and
        metadata are as published, without yet applying dataset policy. Whether a
        file is admitted is validation's decision, and keeping the two apart is
        what makes the rejection report meaningful.

        Args:
            max_files: Stop after this many records. Used by smoke runs and
                tests; ``None`` means everything.

        Yields:
            :class:`SourceRecord` instances in a deterministic order.
        """

    # -- self-description --------------------------------------------------

    def notes(self) -> str:
        """Adapter-specific note for the build report. Empty by default."""
        return ""

    def describe(self) -> AdapterDescription:
        """Describe what is on disk for this entry.

        The walk is bounded by ``metadata.describe_max_files`` so that ``inspect``
        on a ten-terabyte corpus answers in seconds; the returned
        ``audio_file_count`` is then the number examined, not the number present,
        and ``total_bytes`` is the corresponding partial sum. The ``notes`` field
        says so when the cap bit, because a truncated count that reads as a total
        is a number somebody will put in a slide.
        """
        cap = self.int_flag("describe_max_files", 20_000)
        count = 0
        total_bytes = 0
        extensions: set[str] = set()
        for path in self.iter_audio_files():
            count += 1
            extensions.add(path.suffix.lower())
            try:
                total_bytes += path.stat().st_size
            except OSError:
                # A dangling entry or a permission problem on one file must not
                # abort the inventory; the file is still counted as present.
                pass
            if count >= cap:
                break

        hints = self.layout_hints()
        expected = self.expected_layout()
        warnings: list[str] = []
        if not self.is_available():
            warnings.append(
                f"corpus directory {self.root} does not exist; the dataset is "
                "configured but not present on this machine"
            )
        else:
            if expected and not hints:
                warnings.append(
                    "none of the expected layout markers were found; this corpus "
                    f"does not look like the {self.name} layout, so a zero-row "
                    "discovery here means a wrong path rather than an empty corpus"
                )
            elif expected and len(hints) < len(expected):
                missing = sorted(set(expected) - set(hints))
                warnings.append(
                    f"expected layout marker(s) not found: {missing}; partial metadata may result"
                )
            if count == 0:
                warnings.append(
                    f"no files with a recognised audio extension were found under {self.root}"
                )
            if count >= cap:
                warnings.append(
                    f"counted {cap} files; the cap metadata.describe_max_files was "
                    "reached, so the counts below are a lower bound"
                )

        truncated_note = " (lower bound: describe_max_files reached)" if count >= cap else ""
        return AdapterDescription(
            dataset_id=self.dataset_id,
            adapter=self.name,
            task=self.task,
            version=self.entry.version,
            root=str(self.root),
            available=self.is_available(),
            audio_file_count=count,
            total_bytes=total_bytes,
            extensions=tuple(sorted(extensions)),
            expected_layout=expected,
            layout_hints_found=hints,
            metadata_files=self.metadata_files(),
            notes=self.notes() + truncated_note,
            warnings=tuple(warnings),
        )


def as_sequence(value: object) -> Sequence[str]:
    """Coerce a metadata value to a sequence of strings.

    Adapters receive every ``metadata`` value as a string, so a list-valued
    setting arrives comma-separated. This is the one place that convention is
    decoded, rather than each adapter reimplementing ``split(",")`` with its own
    idea of what to do with blanks.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(part).strip() for part in value if str(part).strip())
    return (str(value).strip(),)
