"""The label taxonomy and its single encoding.

Two separate decisions live here, and conflating them is a well-known way to
produce a model that looks trained and is not.

**The classification target.** Every record carries a binary ``bona_fide`` /
``spoof`` label, because "does this contain synthetic speech" is the only
question VoxShield answers. Detailed attack taxonomies exist in every public
spoofing corpus, but folding them into the target means a model is trained to
separate vocoders rather than to detect synthesis, and a novel generator then
reads as bona fide. The detailed attack is therefore *metadata*: preserved,
queryable, and never the training target.

**The encoding.** ``bona_fide = 0`` and ``spoof = 1``. This is fixed here, in one
place, and read from here by the dataset, the manifests, and the CLI. A mapping
that is defined at each use site eventually disagrees with itself, and the
disagreement surfaces as a detector that is confidently inverted.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "ATTACK_FAMILIES",
    "ATTACK_TYPES",
    "BONA_FIDE",
    "INDEX_TO_LABEL",
    "LABELS",
    "LABEL_TO_INDEX",
    "SPOOF",
    "UNCLASSIFIED",
    "attack_family",
    "encode_label",
    "is_valid_label",
]

#: Human speech that was not synthesised. Index 0.
BONA_FIDE: Final = "bona_fide"

#: Machine-generated or vocoded speech. Index 1.
SPOOF: Final = "spoof"

#: The complete set of valid binary labels. A label outside this set is a data
#: error, not a new class, and is rejected at ingestion rather than mapped to a
#: near-miss.
LABELS: Final[frozenset[str]] = frozenset({BONA_FIDE, SPOOF})

#: The one authoritative label encoding. Order is meaningful only through this
#: mapping; nothing may depend on a bare ``int`` meaning.
LABEL_TO_INDEX: Final[dict[str, int]] = {BONA_FIDE: 0, SPOOF: 1}

INDEX_TO_LABEL: Final[dict[int, str]] = {index: label for label, index in LABEL_TO_INDEX.items()}

#: ASVspoof 2019 / 2021 attack identifiers, as published (upper case). The
#: authoritative source for both the public vocabulary and the internal family
#: lookup, so the two cannot drift apart.
_ASVSPOOF_ATTACK_IDS: Final[frozenset[str]] = frozenset(f"A{index:02d}" for index in range(1, 18))

#: Known attack identifiers, kept as a controlled vocabulary so a typo in an
#: adapter's metadata cannot invent a category. The set is deliberately
#: non-exhaustive: :func:`attack_family` falls back to :data:`UNCLASSIFIED` rather
#: than rejecting an unrecognised-but-real attack type, because a corpus
#: published after this list was written should still be usable.
ATTACK_TYPES: Final[frozenset[str]] = frozenset(
    {
        *_ASVSPOOF_ATTACK_IDS,
        # WaveFake vocoders, as published.
        "melgan",
        "melgan2",
        "full_band",
        "parallel_wavegan",
        "waveglow",
        "hi_fi_gan",
        "wavemul",
        # TTS systems, as published.
        "tacotron2",
        "fastspeech",
        "fastspeech2",
        "wave_tts",
        "vits",
        "ljspeech_vits",
    }
)

#: Family for an attack type this module does not recognise, and for the one
#: identifier that names a condition rather than a generator. Carries no
#: information, so a report can count it without interpreting it.
UNCLASSIFIED: Final = "unclassified"

#: Coarse grouping of attack types. Useful for a report ("half the spoof set is
#: vocoder-based") without any of it becoming part of the target.
ATTACK_FAMILIES: Final[frozenset[str]] = frozenset(
    {"vocoder", "neural_waveform", "tts", UNCLASSIFIED}
)

# Attack sets are keyed in the same normalised form :func:`attack_family` looks
# them up in, which is lower case. Storing the ASVspoof identifiers as ``"a07"``
# rather than as published (``"A07"``) is deliberate: publishing them is a
# manifest concern, and a lookup table that disagrees with its own query case
# silently matches nothing.
_VOCODER_ATTACKS: Final[frozenset[str]] = frozenset(
    {
        "melgan",
        "melgan2",
        "parallel_wavegan",
        "waveglow",
        "hi_fi_gan",
        "wavemul",
    }
)

_WAVEFORM_ATTACKS: Final[frozenset[str]] = frozenset(
    identifier.lower() for identifier in _ASVSPOOF_ATTACK_IDS
)

_TTS_ATTACKS: Final[frozenset[str]] = frozenset(
    {"tacotron2", "fastspeech", "fastspeech2", "wave_tts", "vits", "ljspeech_vits"}
)


def is_valid_label(label: str) -> bool:
    """Whether ``label`` is one of the two valid binary labels."""
    return label in LABELS


def encode_label(label: str) -> int:
    """Map a binary label to its training index.

    Args:
        label: ``"bona_fide"`` or ``"spoof"``.

    Returns:
        ``0`` for bona fide, ``1`` for spoof.

    Raises:
        ValueError: If ``label`` is not a valid binary label. Raising rather than
            defaulting matters: an unmapped label would otherwise become a
            silent ``0``, and a corpus of mislabelled spoofs trained as bona fide
            is a model that reports synthetic speech as human.
    """
    try:
        return LABEL_TO_INDEX[label]
    except KeyError:
        msg = f"unknown label {label!r}; expected one of {sorted(LABELS)}"
        raise ValueError(msg) from None


def attack_family(attack_type: str | None) -> str:
    """Group a detailed attack type into a coarse family.

    Lookup is case-insensitive, because ASVspoof publishes its identifiers in
    upper case (``A07``) while directory names and vocoder names are lower case,
    and a family report that silently bucketed all 17 ASVspoof attacks as
    ``unclassified`` would still look plausible.

    Args:
        attack_type: A published attack or vocoder identifier, or ``None``.

    Returns:
        One of :data:`ATTACK_FAMILIES`.

    Notes:
        ``"full_band"`` is a bandwidth condition rather than a synthesis method,
        so it classifies as ``"unclassified"`` rather than being folded into
        ``"vocoder"``; it is the one identifier in :data:`ATTACK_TYPES` that names
        a condition instead of a generator. The unrecognised value itself is
        still preserved in the manifest, so nothing is lost.
    """
    if not attack_type:
        return UNCLASSIFIED
    key = attack_type.strip().lower()
    if key in _VOCODER_ATTACKS:
        return "vocoder"
    if key in _WAVEFORM_ATTACKS:
        return "neural_waveform"
    if key in _TTS_ATTACKS:
        return "tts"
    return UNCLASSIFIED
