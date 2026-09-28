"""ASVspoof 2019 / 2021 ingestion.

ASVspoof is the closest thing the field has to a reference corpus, and the layout
changed between its editions. 2019 partitions by subset and label
(``LA/eval/bona``, ``LA/eval/spoof``); 2021 flattens the labels and ships a
``progress.txt`` whose columns are the only published source for speaker, codec,
and attack. Reading 2021 by directory name yields a corpus with no speaker
metadata at all, which then cannot be split speaker-disjointly -- the split
degrades to file-disjoint, the guarantee in the docs quietly stops being true,
and nothing in the build fails.

So this adapter prefers ``progress.txt`` and falls back to the directory tree,
recording which it used. Both paths are implemented, and the fallback is
explicit in the record rather than silent.

What it will not do is guess. The attack column is ``-`` for bona fide rows;
``"-"`` is not an attack identifier, so it becomes :data:`~voxshield.data.schema.UNKNOWN`.
Likewise, when only the directory layout is available, the attack is left unknown
rather than inferred from the spoof directory, because every ASVspoof spoof in a
given subset can come from a different codec condition and pretending otherwise
would manufacture a cross-codec holdout that does not exist.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator, Mapping
from pathlib import Path

from voxshield.data.adapters.base import DatasetAdapter
from voxshield.data.errors import AdapterError
from voxshield.data.labels import BONA_FIDE, SPOOF
from voxshield.data.schema import UNKNOWN, SourceRecord

__all__ = ["ASVspoofAdapter"]

#: ASVspoof publishes attack identifiers as ``A01``..``A17``. Recognising the shape
#: lets a redistributable that bakes the attack into the directory name
#: (``spoof/A07/``) be read correctly, without accepting any other directory name
#: as an attack.
_ATTACK_DIR_RE = re.compile(r"^A\d{2}$", re.IGNORECASE)

#: Same shape, case-sensitive, for validating a value that came *out* of the
#: progress file. The directory rule is case-insensitive because directory names
#: vary in case between redistributions; the published identifier is upper case, and
#: accepting ``a07`` from a file as an attack would mean accepting a value the
#: corpus does not publish.
_ATTACK_ID_RE = re.compile(r"^A\d{2}$")

#: Placeholder ASVspoof writes in a populated field it has no value for. Treated
#: as absent: these are the dataset's "no value" markers, so preserving them would
#: make ``"-"`` a real codec that no other corpus has.
_NULL_TOKENS: frozenset[str] = frozenset({"-", "--", "", "n/a", "na", "none", "null"})

#: ``progress.txt`` column names, normalised. ASVspoof has shipped these under
#: several spellings ("SpeakerID", "speaker_id", "SPEAKERID"); normalising by
#: lowercasing and dropping non-alphanumerics absorbs the variation instead of
#: failing on a header this adapter cannot parse.
_COLUMN_KEYS: dict[str, tuple[str, ...]] = {
    "speaker_id": ("speakerid", "speaker", "spkid"),
    "filename": ("filename", "file", "name"),
    "codec": ("codec", "codeccondition", "codecconditionname"),
    # Deliberately *not* aliased to "source". The source column names the corpus
    # edition (ASVspoof2019LA, ASVspoof2021DF) -- provenance, not a transmission
    # condition. Folding it into channel would both invent a channel value and
    # displace the real one recorded in the codec column.
    "channel": ("channel",),
    "source": ("source",),
    "attack": ("attack", "attacktype", "attackid"),
    "subset": ("subset", "trialset", "split"),
    "trim": ("trim",),
}

#: Published codec conditions name the transmission, and are better modelled as
#: the channel/condition axis than as a container format. Mapping them out keeps
#: ``codec`` meaning "container" everywhere, so the cross-codec holdout selects
#: genuinely different encoders rather than ASVspoof's bitrate grid.
_CODEC_CONDITIONS: frozenset[str] = frozenset(
    {
        "low_mdbmp3",
        "low_mdbmp3_music",
        "low_mdbmp3_speech",
        "high_mdbmp3",
        "high_mdbmp3_music",
        "high_mdbmp3_speech",
    }
)


def _normalise_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


def _clean(value: str | None) -> str:
    """Strip, and map ASVspoof's null placeholders to :data:`UNKNOWN`."""
    text = (value or "").strip()
    return UNKNOWN if text.lower() in _NULL_TOKENS else text


def _cell(row: list[str], position: int) -> str:
    """One cell of a progress row, or :data:`UNKNOWN` if the row is short.

    Rows in these files are ragged: a trailing empty field is routinely omitted, so
    a fixed-width read would index past the end of a row. Bounds-checking here
    turns that into an unknown value instead of an ``IndexError`` partway through a
    corpus.

    Args:
        row: The parsed row.
        position: Column index from the header.

    Returns:
        The cleaned cell value, or :data:`UNKNOWN`.
    """
    if position < 0 or position >= len(row):
        return UNKNOWN
    return _clean(row[position])


def _label_from_row(row: Mapping[str, str] | None) -> str:
    """Derive the label from a progress row.

    ASVspoof's ``progress.txt`` has no label column. The label is encoded in the
    ``attack`` column: bona fide rows carry the null placeholder, spoof rows carry
    an ``A<nn>`` identifier. That is the corpus's own encoding, not an inference,
    so using it is faithful -- but it is a convention, and the rule is stated
    explicitly here rather than inlined at the call site where it would be
    unreadable.

    Args:
        row: One parsed row, or ``None`` if the file had no progress entry.

    Returns:
        :data:`BONA_FIDE` for a row whose attack is absent, :data:`SPOOF` for a row
        with an attack identifier, :data:`UNKNOWN` when the row is missing or
        carries an unrecognised attack value.
    """
    if not row:
        return UNKNOWN
    attack = row.get("attack", UNKNOWN)
    if attack == UNKNOWN:
        return BONA_FIDE
    if _ATTACK_ID_RE.match(attack):
        return SPOOF
    return UNKNOWN


def _reconcile_labels(directory_label: str, published_label: str) -> str:
    """Combine the two label sources, treating disagreement as a defect.

    When both sources are available and they disagree -- a file under ``spoof/``
    that the progress file lists as bona fide -- the label is :data:`UNKNOWN`
    rather than either value. Preferring one side would make the choice invisibly
    arbitrary, and picking the wrong side mislabels a training example, which for
    this task means training the model to invert its own output. An unlabelled
    record is rejected by validation with a reason; a mislabelled one is not
    detectable downstream at all.

    Args:
        directory_label: Label implied by the path, or :data:`UNKNOWN`.
        published_label: Label derived from the progress row, or :data:`UNKNOWN`.

    Returns:
        The agreed label, or :data:`UNKNOWN`.
    """
    if directory_label == published_label:
        return directory_label
    if directory_label == UNKNOWN:
        return published_label
    if published_label == UNKNOWN:
        return directory_label
    return UNKNOWN


class ASVspoofAdapter(DatasetAdapter):
    """Reads ASVspoof 2019 and 2021 layouts, preferring ``progress.txt``.

    Configuration keys, all optional, read from ``metadata``:

    * ``progress_file`` -- path to the metadata file relative to the dataset root.
      Defaults to auto-detection: the first ``progress.txt`` found, in a fixed
      order, so a corpus that ships one at the root and another under a subset does
      not silently use the wrong one.
    * ``bona_dirname`` / ``spoof_dirname`` -- directory names carrying the label.
      Default ``bona`` and ``spoof``.
    * ``directory_labels`` -- ``"1"`` (default) to accept a label read from a
      ``bona``/``spoof`` directory when the progress file has none, or ``"0"`` to
      require the progress file for every label.
    * ``max_files`` -- cap on records yielded, for smoke runs.
    """

    name = "asvspoof"
    task = "spoof_detection"

    def expected_layout(self) -> tuple[str, ...]:
        return ("LA", "PA", "progress.txt", "LA/eval", "PA/eval")

    def notes(self) -> str:
        return (
            "ASVspoof labels and speaker/codec/attack metadata are read from "
            "progress.txt when present, otherwise from the directory layout. "
            "Attack identifiers that the corpus does not publish are recorded as "
            f"'{UNKNOWN}', not guessed from the spoof directory."
        )

    # -- metadata file ------------------------------------------------------

    def _progress_candidates(self) -> list[Path]:
        """Metadata files to try, in a fixed order.

        The order is fixed rather than filesystem-dependent for the same reason
        discovery sorts its walk: a corpus shipping more than one ``progress.txt``
        must resolve to the same one on every machine, or two builds of identical
        audio produce different metadata.
        """
        override = self.flag("progress_file")
        if override:
            return [self.root / override]
        preferred = (
            "progress.txt",
            "LA/progress.txt",
            "PA/progress.txt",
            "LA/eval/progress.txt",
            "PA/eval/progress.txt",
            "LA/train/progress.txt",
            "PA/train/progress.txt",
        )
        found = [self.root / name for name in preferred if (self.root / name).is_file()]
        if found:
            return found
        # An unrecognised location is still better than nothing, as long as the
        # search is bounded and deterministic.
        return sorted(self.root.glob("**/progress.txt"))

    def _read_progress(self) -> tuple[Path, dict[str, dict[str, str]]]:
        """Parse the first readable ``progress.txt``.

        Returns:
            ``(path, index)`` where ``index`` maps both a bare filename stem and a
            repository-relative posix path to that row's fields. Two keys per row
            because ASVspoof's ``filename`` column sometimes carries a full
            relative path and sometimes only a basename, and matching on both avoids
            a corpus silently losing every row to a join mismatch.

        Raises:
            AdapterError: If a candidate exists but has no recognisable header. A
                malformed file is an error rather than a reason to fall back to
                directory labels, because falling back produces a corpus that looks
                fine and has lost its speaker metadata.
        """
        for candidate in self._progress_candidates():
            if not candidate.is_file():
                continue
            index = self._parse_progress(candidate)
            if index:
                return candidate, index
            msg = (
                f"ASVspoof metadata file {candidate} was found but has no "
                "recognisable header row; expected columns such as SpeakerID and "
                "filename. Refusing to fall back to directory labels, which would "
                "drop speaker, codec, and attack metadata without failing loudly."
            )
            raise AdapterError(msg)
        return Path(""), {}

    def _parse_progress(self, path: Path) -> dict[str, dict[str, str]]:
        """Parse one ``progress.txt`` into a filename-keyed index."""
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=",", skipinitialspace=True)
            rows = [row for row in reader if row]
        if not rows:
            return {}
        header = [_normalise_column(cell) for cell in rows[0]]
        if "filename" not in _COLUMN_KEYS and "filename" not in header:
            return {}
        positions = {
            field: next(
                (i for i, cell in enumerate(header) if cell in aliases),
                -1,
            )
            for field, aliases in _COLUMN_KEYS.items()
        }
        if positions["filename"] < 0:
            return {}
        index: dict[str, dict[str, str]] = {}
        for row in rows[1:]:
            if len(row) <= positions["filename"]:
                continue
            values = {field: _cell(row, positions[field]) for field in _COLUMN_KEYS}
            filename = row[positions["filename"]].strip()
            if not filename:
                continue
            index[filename.lower()] = values
            # Basename key, so a row whose filename column carries a path still
            # matches a record identified by the file on disk.
            index[Path(filename).name.lower()] = values
            index[Path(filename).stem.lower()] = values
        return index

    def _lookup(self, path: Path, index: Mapping[str, dict[str, str]]) -> dict[str, str]:
        """Find a file's progress row, or all-unknown values."""
        if not index:
            return {}
        relative = (
            path.relative_to(self.root).as_posix() if path.is_relative_to(self.root) else path.name
        )
        for key in (
            relative,
            relative.lower(),
            path.name,
            path.name.lower(),
            path.stem,
            path.stem.lower(),
        ):
            if key in index:
                return index[key]
        return {}

    def metadata_files(self) -> tuple[str, ...]:
        try:
            path, index = self._read_progress()
        except AdapterError:
            return ()
        if not path or not index:
            return ()
        return (path.relative_to(self.root).as_posix(),)

    # -- label --------------------------------------------------------------

    def _label_from_path(self, path: Path) -> str:
        """Read the label from the directory layout, or :data:`UNKNOWN`.

        :data:`UNKNOWN` is returned rather than raising when the directory carries
        no label, because validation is where a candidate becomes a rejection, and
        an adapter that raised would abort discovery for the whole corpus over one
        file in an unexpected place.
        """
        bona = self.flag("bona_dirname", "bona").lower()
        spoof = self.flag("spoof_dirname", "spoof").lower()
        for part in path.relative_to(self.root).parts if path.is_relative_to(self.root) else ():
            lowered = part.lower()
            if lowered == bona:
                return BONA_FIDE
            if lowered == spoof:
                return SPOOF
        return UNKNOWN

    def _attack_from_path(self, path: Path) -> str:
        """Recover an attack id from a path component, if one is there.

        Some redistributions lay ASVspoof out as ``spoof/A07/``. Matching the
        published ``A<nn>`` shape is a narrow rule; anything looser would turn
        ordinary directory names into attack identifiers.
        """
        parts = path.relative_to(self.root).parts if path.is_relative_to(self.root) else ()
        for part in reversed(parts):
            if _ATTACK_DIR_RE.match(part):
                return part.upper()
        return UNKNOWN

    # -- discovery ----------------------------------------------------------

    def discover(self, *, max_files: int | None = None) -> Iterator[SourceRecord]:
        """Yield one :class:`SourceRecord` per candidate file.

        Files whose label cannot be determined from either source are yielded with
        an empty label, which the schema rejects. That is intended: a file in an
        unrecognised directory has no defensible label, and dropping it here would
        hide it from the validation report.
        """
        self.require_available()
        _progress_path, index = self._read_progress()
        limit = max_files if max_files is not None else self.int_flag("max_files", 0) or None
        use_directory_labels = self.bool_flag("directory_labels", True)

        emitted = 0
        for path in self.iter_audio_files():
            if limit is not None and emitted >= limit:
                return
            row = self._lookup(path, index)
            directory_label = self._label_from_path(path) if use_directory_labels else UNKNOWN
            label = _reconcile_labels(directory_label, _label_from_row(row))
            attack = row.get("attack", UNKNOWN) if row else UNKNOWN
            if attack == UNKNOWN:
                attack = self._attack_from_path(path)
            attack = attack.upper() if _ATTACK_DIR_RE.match(attack) else attack

            channel = row.get("channel", UNKNOWN) if row else UNKNOWN
            codec_condition = row.get("codec", UNKNOWN) if row else UNKNOWN
            if channel == UNKNOWN and codec_condition in _CODEC_CONDITIONS:
                # A published bitrate grid is a channel condition, not a
                # container. Recording it as the codec would make the cross-codec
                # holdout select ASVspoof's own conditions rather than encoders.
                channel = codec_condition

            subset = row.get("subset", UNKNOWN) if row else UNKNOWN

            yield self.make_record(
                path,
                label=label,
                speaker_id=row.get("speaker_id", UNKNOWN) if row else UNKNOWN,
                generator_id=UNKNOWN,
                attack_type=attack,
                channel=channel,
                # The container is deliberately not taken from the progress file.
                # Its codec column names a transmission condition, not a container,
                # so it is left to be probed from the file itself.
                source_split=subset,
                extra={
                    "metadata_source": "progress.txt" if row else "directory",
                    "source": row.get("source", UNKNOWN) if row else UNKNOWN,
                    "trim": row.get("trim", UNKNOWN) if row else UNKNOWN,
                },
            )
            emitted += 1
