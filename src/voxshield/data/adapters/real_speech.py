"""Real-speech corpus ingestion: LibriSpeech, Common Voice, and flat directories.

Spoof detection needs a negative class that is *not* a negative from a spoofing
corpus. ASVspoof's bona fide clips are the counterfeits of ASVspoof's spoofs --
same speakers, same microphones, same transmission pipeline, only the label
differs -- and a detector can learn that difference instead of learning synthesis.
A separately collected real-speech corpus breaks the coupling, which is what makes
these entries useful as the "unseen bona fide" side of an evaluation.

Three layouts, because that is how the corpora actually arrive:

* **LibriSpeech** -- ``<subset>/<speaker>/<chapter>/<utterance>.flac``, so the
  speaker is a directory component and the split-relevant unit is a chapter.
* **Common Voice** -- a ``.tsv`` sidecar joined against a ``clips/`` tree. Carries
  sentence text and vote counts, neither of which is used here, and it does
  **not** carry a speaker.
* **Flat** -- a directory of audio with no metadata at all. The fallback of last
  resort, and honest about what it yields: bona fide labels and nothing else.

The Common Voice speaker deserves a specific warning. Its ``client_id`` column
identifies a *volunteer who contributed clips*, not a single speaker, and in the
validated release the mapping is deliberately broken. Treating ``client_id`` as a
speaker id would produce a field that looks like speaker-disjointness support and
provides none: the same human is not tracked, and neither is the same microphone.
So ``speaker_id`` stays :data:`UNKNOWN` there, and the cross-language holdout
selects on locale instead.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator, Mapping
from pathlib import Path

from voxshield.data.adapters.base import DatasetAdapter
from voxshield.data.errors import AdapterError
from voxshield.data.labels import BONA_FIDE
from voxshield.data.schema import UNKNOWN, SourceRecord

__all__ = ["RealSpeechAdapter"]

#: ISO-639-1 shape. A two-letter path component in Common Voice's tree is a locale
#: code, which is the only published language signal either corpus has.
_LOCALE_RE = re.compile(r"^[a-z]{2}(?:[-_][A-Za-z]{2,4})?$")

#: Common Voice's top-level directory is ``cv-corpus-<language>-<date>``, where the
#: language is numeric. Map the handful of language codes that appear in practice
#: rather than shipping a full table that would look authoritative while being
#: wrong for the codes it omits. An unmapped code becomes UNKNOWN, not a guess.
_COMMON_VOICE_LANGUAGE_CODES: dict[str, str] = {
    "6": "ar",
    "10": "cs",
    "13": "en",
    "18": "hi",
    "21": "ja",
    "24": "ko",
    "28": "ru",
    "33": "es",
    "34": "tr",
    "38": "de",
    "43": "fr",
    "44": "it",
    "52": "pt",
}

#: Placeholder values meaning "not published".
_NULL_TOKENS: frozenset[str] = frozenset({"-", "--", "", "n/a", "na", "none", "null"})


def _clean(value: str | None) -> str:
    text = (value or "").strip()
    return UNKNOWN if text.lower() in _NULL_TOKENS else text


def _source_split_from_parts(parts: tuple[str, ...]) -> str:
    """The corpus's own subset designation, as provenance only.

    LibriSpeech publishes ``dev-clean``/``test-clean``/``train-clean-100`` and
    Common Voice publishes ``validated``/``dev``/``test``. Recorded because it is
    published, never used to assign a split: this pipeline's split is seeded and
    group-aware, and adopting a corpus's fixed per-file split would make the build
    irreproducible under a different seed while looking more authoritative.
    """
    for part in parts[:-1]:
        lowered = part.lower()
        if lowered in (
            "train",
            "dev",
            "valid",
            "validation",
            "validated",
            "test",
            "eval",
        ) or lowered.startswith(("train-", "dev-", "test-")):
            return lowered
    return UNKNOWN


class RealSpeechAdapter(DatasetAdapter):
    """Reads LibriSpeech, Common Voice, or a flat real-speech directory.

    The layout is chosen explicitly via ``metadata.layout``, because inferring it
    from directory names means a mislaid corpus gets read with the wrong rules
    and produces plausible records that are quietly wrong. ``auto`` is available
    and is the only mode that guesses; the report says which mode ran.

    Configuration keys, all optional, read from ``metadata``:

    * ``layout`` -- ``"librispeech"``, ``"common_voice"``, ``"flat"``, or
      ``"auto"`` (default ``"auto"``).
    * ``locale`` -- force a locale for the whole entry, e.g. ``"en"``.
    * ``language`` -- force a language, overriding locale detection. Used for
      LibriSpeech, which is monolingual English by construction but publishes no
      language field.
    * ``metadata_file`` -- the ``.tsv`` for Common Voice, relative to the root.
    * ``max_files`` -- cap on records yielded, for smoke runs.
    """

    name = "real_speech"
    task = "real_speech"

    def expected_layout(self) -> tuple[str, ...]:
        layout = self.resolved_layout()
        if layout == "librispeech":
            return ("dev-clean", "test-clean", "train-clean-100", "validated.tsv")
        if layout == "common_voice":
            return ("clips", "validated.tsv", "train.tsv", "dev.tsv", "test.tsv")
        return ()

    def notes(self) -> str:
        return (
            f"layout={self.resolved_layout()}. All records are bona fide. "
            "Common Voice publishes no speaker identifier, so speaker_id is "
            f"'{UNKNOWN}' there and speaker-disjointness is unavailable for this "
            "corpus rather than approximated with client_id."
        )

    # -- layout -------------------------------------------------------------

    def resolved_layout(self) -> str:
        """The layout in force, resolving ``auto``.

        Detection order is fixed: LibriSpeech's speaker-number directories and
        ``.trans.txt`` files are distinctive, then Common Voice's ``.tsv``, then
        flat. A fixed order keeps two builds of the same corpus on the same rules.
        """
        declared = self.flag("layout", "auto").lower()
        known = {"librispeech", "common_voice", "flat", "auto"}
        if declared not in known:
            msg = (
                f"dataset {self.dataset_id!r} metadata layout={declared!r} is not "
                f"one of {sorted(known)}"
            )
            raise AdapterError(msg)
        if declared != "auto":
            return declared
        if not self.is_available():
            return "flat"
        for candidate in self.root.glob("**/*.trans.txt"):
            if candidate.is_file():
                return "librispeech"
        for name in ("validated.tsv", "train.tsv", "dev.tsv", "test.tsv", "invalid.tsv"):
            if (self.root / name).is_file() or any(self.root.glob(f"**/{name}")):
                return "common_voice"
        return "flat"

    # -- sidecar ------------------------------------------------------------

    def _tsv_rows(self) -> dict[str, dict[str, str]]:
        """Index a Common Voice ``.tsv`` by clip filename.

        An absent or unreadable sidecar yields ``{}``. Common Voice clips remain
        discoverable from the filesystem; what is lost is the locale published in
        the row, which then falls back to the directory name.
        """
        override = self.flag("metadata_file")
        candidates = (
            [self.root / override]
            if override
            else sorted(
                child
                for name in ("validated.tsv", "train.tsv", "dev.tsv", "test.tsv")
                for child in self.root.glob(f"**/{name}")
            )
        )
        index: dict[str, dict[str, str]] = {}
        for candidate in candidates:
            if not candidate.is_file():
                continue
            with candidate.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t", skipinitialspace=True)
                for row in reader:
                    if not row:
                        continue
                    keyed = {
                        str(key).strip().lower(): (value or "")
                        for key, value in row.items()
                        if key is not None
                    }
                    name = keyed.get("path", "").strip()
                    if not name:
                        continue
                    index[name.lower()] = keyed
                    index[Path(name).name.lower()] = keyed
                    index[Path(name).stem.lower()] = keyed
        return index

    def metadata_files(self) -> tuple[str, ...]:
        override = self.flag("metadata_file")
        names = [override] if override else ["validated.tsv", "train.tsv", "dev.tsv", "test.tsv"]
        found: list[str] = []
        for name in names:
            for child in sorted(self.root.glob(f"**/{name}")):
                if child.is_file():
                    found.append(child.relative_to(self.root).as_posix())
        return tuple(found)

    # -- per-layout field extraction ----------------------------------------

    def _relative_parts(self, path: Path) -> tuple[str, ...]:
        if path.is_relative_to(self.root):
            return path.relative_to(self.root).parts
        return (path.name,)

    def _librispeech_speaker(self, parts: tuple[str, ...]) -> tuple[str, str, str]:
        """Return ``(speaker, chapter, parent_stem)`` from LibriSpeech's tree.

        The path is ``<subset>/<speaker>/<chapter>/<utterance>.flac``. The speaker is
        the *first* purely numeric component: a speaker id is a number and a chapter
        is a number, so position is the only thing that distinguishes them, and
        LibriSpeech puts the speaker first. The chapter is the second, and becomes
        the ``parent_id`` -- utterances from one chapter are one continuous reading
        session, which is the unit that must not straddle a split, so it is grouped
        more tightly than the speaker.
        """
        numbers = [part for part in parts[:-1] if part.isdigit()]
        speaker = numbers[0] if numbers else UNKNOWN
        chapter = numbers[1] if len(numbers) > 1 else UNKNOWN
        return speaker, chapter, (f"{speaker}-{chapter}" if chapter != UNKNOWN else "")

    def _common_voice_locale(self, parts: tuple[str, ...], row: Mapping[str, str] | None) -> str:
        """The locale, from the sidecar row, the directory, or configuration.

        Checked in that order because the row is the most explicit, then the
        directory (which is how newer releases encode it), then the numeric
        top-level directory name.
        """
        forced = self.flag("locale")
        if forced:
            return forced.lower()
        if row:
            published = _clean(row.get("locale", "")) or _clean(row.get("language", ""))
            if published != UNKNOWN and _LOCALE_RE.match(published.lower()):
                return published.lower()
        for part in parts[:-1]:
            if _LOCALE_RE.match(part.lower()):
                return part.lower()
        for part in parts:
            match = re.match(r"^cv-corpus-(\d+)-", part.lower())
            if match:
                return _COMMON_VOICE_LANGUAGE_CODES.get(match.group(1), UNKNOWN)
        return UNKNOWN

    def _flat_speaker(self, path: Path) -> str:
        """A speaker from the filename, when a pattern is configured.

        Off unless ``metadata.speaker_pattern`` is set. A flat directory has no
        published speaker, and inferring one from an arbitrary filename convention
        would give the split a grouping that the corpus never asserted.
        """
        pattern_text = self.flag("speaker_pattern")
        if not pattern_text or pattern_text.lower() == "none":
            return UNKNOWN
        match = re.match(pattern_text, path.stem)
        if match is None:
            return UNKNOWN
        try:
            return match.group("speaker")
        except IndexError:  # pragma: no cover - defensive against a bad pattern
            return UNKNOWN

    # -- discovery ----------------------------------------------------------

    def discover(self, *, max_files: int | None = None) -> Iterator[SourceRecord]:
        """Yield one bona fide :class:`SourceRecord` per candidate file.

        Every record is labelled :data:`~voxshield.data.labels.BONA_FIDE`. A real
        speech corpus is a negative class, not a spoof class, and a record claiming
        otherwise from a directory named ``spoof/`` would be a layout confusion
        worth catching in the validation report.
        """
        self.require_available()
        layout = self.resolved_layout()
        rows = self._tsv_rows() if layout == "common_voice" else {}
        forced_language = self.flag("language")
        limit = max_files if max_files is not None else self.int_flag("max_files", 0) or None

        emitted = 0
        for path in self.iter_audio_files():
            if limit is not None and emitted >= limit:
                return
            parts = self._relative_parts(path)
            row = rows.get(path.name.lower()) or rows.get(path.stem.lower()) or None

            speaker = UNKNOWN
            parent_stem = ""
            language = forced_language.lower() if forced_language else UNKNOWN
            extra: dict[str, str] = {"layout": layout}

            if layout == "librispeech":
                speaker, chapter, parent_stem = self._librispeech_speaker(parts)
                if language == UNKNOWN:
                    # LibriSpeech is English-only by construction. Stating that is a
                    # fact about the corpus, not an inference from a file, and it is
                    # what makes the cross-language holdout selectable at all. It is
                    # overridable via ``metadata.language`` for any derivative
                    # corpus that is not.
                    language = "en"
                extra["chapter"] = chapter
            elif layout == "common_voice":
                # Deliberately not client_id: see the module docstring.
                language = self._common_voice_locale(parts, row) or language
                extra["votes"] = _clean(row.get("up_votes", "")) if row else UNKNOWN
            else:
                speaker = self._flat_speaker(path)

            yield self.make_record(
                path,
                label=BONA_FIDE,
                parent_id=parent_stem or path.stem,
                speaker_id=speaker,
                language=language,
                source_split=_source_split_from_parts(parts),
                extra=extra,
            )
            emitted += 1
