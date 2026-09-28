"""WaveFake ingestion.

WaveFake is the friendlier of the two spoofing corpora: the label *is* the
top-level directory, and the vocoder *is* the next one down. ``train/gen/melgan``
and ``train/orig`` make the ground truth readable from the filesystem, so this
adapter needs no sidecar file at all.

That readability is a trap in one specific way. WaveFake pairs each synthetic clip
with a genuine clip carrying the same filename::

    train/gen/melgan/LJ001-0001.wav
    train/orig/LJ001-0001.wav

Same stem, opposite labels, same speaker. If these land on opposite sides of a
split, the model is being tested on a clip whose bona fide twin it memorised, and
the reported number measures recall of a filename rather than detection of a
vocoder. The two files share a ``parent_id`` here for exactly that reason: the
pairing is a real relationship in the corpus, and the split must respect it even
though the dataset never published a field for it. :mod:`voxshield.data.splitting`
assigns whole ``parent_id`` groups, so the pair stays together by construction.

The vocoder directory is also the ``generator_id``, which is the field the
cross-generator holdout selects on -- so a typo in a directory name becomes a
distinct "generator" in the dataset statistics. Vocoder names are therefore
normalised against a known vocabulary, and an unrecognised name is preserved
lowercased rather than rejected: WaveFake ships new vocoders over time, and a
corpus that adds one should still be usable, but two cases of the same name must
not become two generators.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from pathlib import Path

from voxshield.data.adapters.base import DatasetAdapter
from voxshield.data.labels import BONA_FIDE, SPOOF
from voxshield.data.schema import UNKNOWN, SourceRecord

__all__ = ["WaveFakeAdapter"]

#: Vocoders WaveFake publishes. Used to normalise casing and to catch the case
#: where a directory name has been mangled. Membership is not required -- see the
#: module docstring -- it only decides whether to warn.
_KNOWN_VOCODERS: frozenset[str] = frozenset(
    {
        "melgan",
        "melgan2",
        "full_band",
        "parallel_wavegan",
        "waveglow",
        "hi_fi_gan",
        "wavemul",
    }
)

#: Default speaker pattern. LibriSpeech filenames are ``<speaker>-<utterance>``
#: with a two-letter book code, so ``LJ001-0001`` yields speaker ``LJ001``. JSUT
#: filenames (``common_voice``-derived and the Japanese half) follow a different
#: convention, and are left unknown rather than matched by a looser rule that would
#: also match arbitrary text.
_DEFAULT_SPEAKER_RE = re.compile(r"^(?P<speaker>[A-Za-z]{2}\d+)(?=[-_])")

#: Directories that hold originals, at any depth. Compared case-insensitively
#: because the same corpus has been redistributed as both ``orig`` and ``original``.
_ORIGINAL_DIRS: frozenset[str] = frozenset({"orig", "original", "real", "bona"})
_GENERATED_DIRS: frozenset[str] = frozenset({"gen", "generated", "fake", "spoof"})

#: Placeholder values for "the corpus published nothing here".
_NULL_TOKENS: frozenset[str] = frozenset({"-", "--", "", "n/a", "na", "none", "null", "nan"})


def _clean(value: str | None) -> str:
    text = (value or "").strip()
    return UNKNOWN if text.lower() in _NULL_TOKENS else text


class WaveFakeAdapter(DatasetAdapter):
    """Reads the WaveFake ``{gen,orig}/<vocoder>/`` layout.

    Configuration keys, all optional, read from ``metadata``:

    * ``speaker_pattern`` -- regex with a ``speaker`` group applied to the file
      stem. Defaults to the LibriSpeech convention. Set to ``"none"`` to disable
      speaker extraction, which is the honest setting for a corpus whose filenames
      carry no speaker.
    * ``metadata_file`` -- optional ``metadata.csv`` relative to the dataset root,
      with ``file,vocoder,dataset,language`` columns. Used only to *add* published
      fields; the directory layout remains authoritative for the label, because a
      filename in a metadata file can be stale while the tree it points into
      cannot.
    * ``language_from_directory`` -- when ``"1"``, treat a two-letter directory
      above the vocoder as a language code. Off by default: WaveFake's structure has
      changed between releases and an enabled-by-default guess would attach the
      wrong language to a whole corpus silently.
    * ``max_files`` -- cap on records yielded, for smoke runs.
    """

    name = "wavefake"
    task = "spoof_detection"

    def expected_layout(self) -> tuple[str, ...]:
        return ("train", "test", "train/gen", "train/orig", "test/gen", "test/orig")

    def notes(self) -> str:
        return (
            "Label and generator are read from the {gen,orig}/<vocoder> tree. "
            "Paired clips that share a filename across gen/ and orig/ are given a "
            "shared parent_id so the split keeps them together."
        )

    # -- optional sidecar ---------------------------------------------------

    def _metadata_rows(self) -> dict[str, dict[str, str]]:
        """Index an optional ``metadata.csv`` by lowercase basename.

        An absent or unreadable file yields ``{}`` rather than raising. This is
        the one place in the pipeline where optional metadata is allowed to be
        missing without complaint, because the directory layout alone is a complete
        source of label and generator.
        """
        override = self.flag("metadata_file")
        candidates = (
            [self.root / override]
            if override
            else [
                self.root / "metadata.csv",
                self.root / "wavefake_metadata.csv",
            ]
        )
        index: dict[str, dict[str, str]] = {}
        for candidate in candidates:
            if not candidate.is_file():
                continue
            with candidate.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                for row in csv.DictReader(handle, skipinitialspace=True):
                    if not row:
                        continue
                    keyed = {
                        str(key).strip().lower(): (value or "")
                        for key, value in row.items()
                        if key is not None
                    }
                    name = keyed.get("file", "").strip()
                    if not name:
                        continue
                    index[name.lower()] = keyed
                    index[Path(name).name.lower()] = keyed
                    index[Path(name).stem.lower()] = keyed
            break
        return index

    def metadata_files(self) -> tuple[str, ...]:
        override = self.flag("metadata_file")
        for candidate in [self.root / override] if override else [self.root / "metadata.csv"]:
            if candidate.is_file():
                try:
                    return (candidate.relative_to(self.root).as_posix(),)
                except ValueError:
                    return (candidate.as_posix(),)
        return ()

    # -- path parsing -------------------------------------------------------

    def _relative_parts(self, path: Path) -> tuple[str, ...]:
        if path.is_relative_to(self.root):
            return path.relative_to(self.root).parts
        return (path.name,)

    def _label_from_parts(self, parts: tuple[str, ...]) -> str:
        """The label is a directory name, at any depth.

        Scans every component rather than a fixed index because releases differ:
        ``train/orig`` and ``test/gen/melgan`` are both common, and so is a flat
        ``orig/`` at the root. A fixed index would silently mislabel an entire
        release.
        """
        for part in parts[:-1]:
            lowered = part.lower()
            if lowered in _ORIGINAL_DIRS:
                return BONA_FIDE
            if lowered in _GENERATED_DIRS:
                return SPOOF
        return UNKNOWN

    def _vocoder_from_parts(self, parts: tuple[str, ...], labelled: str) -> str:
        """The generator, taken from the directory under ``gen/``.

        For genuine clips the answer is legitimately :data:`UNKNOWN` -- there is no
        generator, and writing ``"none"`` or the corpus name would create a
        generator category that a cross-generator holdout could then select on.
        """
        if labelled != SPOOF:
            return UNKNOWN
        for index, part in enumerate(parts):
            if part.lower() in _GENERATED_DIRS and index + 1 < len(parts) - 1:
                return _normalise_vocoder(parts[index + 1])
        return UNKNOWN

    def _language_from_parts(self, parts: tuple[str, ...]) -> str:
        """A language code from the directory tree, when explicitly enabled.

        Off by default for the reason in the class docstring. When enabled, only a
        two-letter component counts, which is the ISO-639-1 shape the corpus uses;
        ``en`` is returned as written because the corpus writes it that way and
        normalising case here would fork ``en`` and ``EN`` into two languages.
        """
        if not self.bool_flag("language_from_directory", False):
            return UNKNOWN
        for part in parts[:-1]:
            if len(part) == 2 and part.isalpha():
                return part.lower()
        return UNKNOWN

    def _speaker_from_stem(self, stem: str) -> str:
        """The speaker, from the filename, when the configured pattern matches."""
        pattern_text = self.flag("speaker_pattern")
        if pattern_text.lower() == "none":
            return UNKNOWN
        pattern = re.compile(pattern_text) if pattern_text else _DEFAULT_SPEAKER_RE
        match = pattern.match(stem)
        if match is None:
            return UNKNOWN
        try:
            return match.group("speaker")
        except IndexError:  # pragma: no cover - defensive against a bad pattern
            return UNKNOWN

    def parent_id_for_stem(self, stem: str) -> str:
        """Grouping id for the utterance a filename stem names.

        Derived from the stem alone, so the genuine and synthetic copies of one
        utterance resolve to the same value. Namespaced by dataset for the same
        reason sample ids are: two corpora both containing ``1.wav`` must not
        merge into one group.

        Stems that repeat across WaveFake's own ``train``/``test`` directories are
        deliberately merged. Grouping them is conservative -- it can only make a
        group larger -- whereas treating them as distinct would let a split
        separate two copies of one utterance, which is the leak this exists to
        prevent.

        Args:
            stem: The audio filename without its suffix.

        Returns:
            A dataset-namespaced parent id.
        """
        return f"{self.dataset_id}:{stem}"

    def _source_split_from_parts(self, parts: tuple[str, ...]) -> str:
        """WaveFake's own ``train``/``test`` designation, as provenance only.

        Recorded because it is published, and never used to assign a split: this
        pipeline's split is seeded and group-aware, whereas the corpus's is fixed
        and per-file. Adopting the corpus split would make the build
        irreproducible under a different seed while appearing to be more
        authoritative.
        """
        for part in parts[:-1]:
            if part.lower() in ("train", "dev", "valid", "validation", "eval", "test"):
                return part.lower()
        return UNKNOWN

    # -- discovery ----------------------------------------------------------

    def discover(self, *, max_files: int | None = None) -> Iterator[SourceRecord]:
        """Yield one :class:`SourceRecord` per candidate file."""
        self.require_available()
        rows = self._metadata_rows()
        limit = max_files if max_files is not None else self.int_flag("max_files", 0) or None

        emitted = 0
        for path in self.iter_audio_files():
            if limit is not None and emitted >= limit:
                return
            parts = self._relative_parts(path)
            label = self._label_from_parts(parts)
            vocoder = self._vocoder_from_parts(parts, label)
            row = rows.get(path.name.lower()) or rows.get(path.stem.lower()) or {}

            generator = _normalise_vocoder(row.get("vocoder", "")) if row else UNKNOWN
            if generator == UNKNOWN:
                # The sidecar said nothing, and the directory is the corpus's own
                # encoding of the same fact.
                generator = vocoder

            language = _clean(row.get("language", "")) if row else UNKNOWN
            if language == UNKNOWN:
                language = self._language_from_parts(parts)

            # A metadata row naming a different generator than the directory it sits
            # in means the sidecar is stale. The tree wins, and the disagreement is
            # recorded, because a wrong generator_id silently corrupts the
            # cross-generator holdout.
            if row and vocoder != UNKNOWN:
                declared = _normalise_vocoder(row.get("vocoder", ""))
                if declared not in (UNKNOWN, vocoder):
                    row = {**row, "generator_conflict": f"{declared} vs {vocoder}"}
                    generator = vocoder

            yield self.make_record(
                path,
                label=label,
                # The pairing parent is the *filename stem*, not the file and not
                # its directory. Both twins derive it identically, which is the
                # whole point: a gen/ and an orig/ clip for one utterance must land
                # in the same group, and deriving it from the path would put the
                # gen/ copy under "..._gen_melgan_<stem>" and the orig/ copy under
                # "..._orig_<stem>". Two groups for one utterance, and the split
                # separates them.
                parent_id=self.parent_id_for_stem(path.stem) if label in (BONA_FIDE, SPOOF) else "",
                speaker_id=self._speaker_from_stem(path.stem),
                generator_id=generator,
                language=language,
                attack_type=generator,
                source_split=self._source_split_from_parts(parts),
                extra={
                    "metadata_source": "metadata.csv" if row else "directory",
                    "corpus_dataset": _clean(row.get("dataset", "")) if row else UNKNOWN,
                    **(
                        {"generator_conflict": row["generator_conflict"]}
                        if row and "generator_conflict" in row
                        else {}
                    ),
                },
            )
            emitted += 1


def _vocoder_key(name: str) -> str:
    """Separators removed entirely, for comparing one vocoder under any spelling.

    ``HiFi-GAN``, ``hi_fi_gan``, and ``hifigan`` all reduce to ``hifigan``. Folding
    separators to underscores instead is not enough: the vocabulary entry
    ``hi_fi_gan`` and the directory ``HiFi-GAN`` disagree about where the word
    boundary falls, so a separator-preserving comparison calls them two vocoders.
    """
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _normalise_vocoder(name: str) -> str:
    """Normalise a vocoder name to the canonical spelling from the vocabulary.

    ``HiFi-GAN``, ``hifigan``, and ``hi_fi_gan`` are the same vocoder. Left
    distinct, they become three generators and a cross-generator holdout can
    select one spelling while reporting coverage of all three.

    Separators are folded to underscores and matched against the known vocabulary;
    an unrecognised name is kept as-is (lowercased) rather than dropped, because a
    corpus that adds a vocoder should not become unusable.
    """
    text = (name or "").strip().lower()
    if not text or text in _NULL_TOKENS:
        return UNKNOWN
    key = _vocoder_key(text)
    if not key:
        return UNKNOWN
    for known in _KNOWN_VOCODERS:
        if _vocoder_key(known) == key:
            return known
    return re.sub(r"[\s\-.]+", "_", text)
