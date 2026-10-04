"""Content-addressed preprocessing cache.

A dataset build re-decodes and re-resamples every source file. Most of that work
is wasted on the second build: decode and resampling dominate the cost of Phase 1,
and they are pure functions of the source bytes and the audio configuration. The
canonical output -- one float32 16 kHz mono signal -- can therefore be stored
under a key that makes a stale hit impossible by construction:

``cache_key = hash(format_version | file_hash | preprocessing_signature)``

The **file hash** pins the value to exactly one byte stream; the
**preprocessing signature** pins it to exactly one audio configuration. Change
either one and the key changes, so "I changed the config but got last build's
audio" cannot happen by accident. ``format_version`` is the third ingredient for
the same reason this module's serialisation format exists: a future on-disk
layout change must invalidate, not silently reinterpret, old entries.

What is cached is deliberately only the *canonical waveform* and its few
measurements, not segments or VAD decisions. Re-running VAD and segmentation over
a 16 kHz signal costs frames, not seconds, next to a decode and a polyphase
resample, and keeping them out means a change to one audio flag that affects only
segmentation (a smaller window, say) redoes the cheap half while the expensive
half stays cached. The waveform and measurements are stored with numpy's native
format (exact, no codec round-trip) plus a JSON sidecar naming what produced it,
with ``allow_pickle=False`` on load so an on-disk entry can never execute code.

The cache is a soft accelerator, never a correctness mechanism: a missing,
corrupt, or unreadable entry is a miss, not a failure. ``max_entries`` is a soft
cap enforced on insert, mirroring :class:`~voxshield.data.config.CacheConfig`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from voxshield.audio.preprocess import PreprocessedAudio
from voxshield.config import AudioConfig
from voxshield.data.config import CacheConfig, DataConfig
from voxshield.data.errors import DatasetBuildError, DatasetConfigError
from voxshield.data.paths import DataPaths

__all__ = [
    "CACHE_FORMAT_VERSION",
    "CachedAudio",
    "PreprocessingCache",
    "cache_key",
    "preprocessing_signature",
]

#: Identifies the on-disk layout of this cache. Bumped only when a change would
#: make an old entry silently wrong to load; rasterised into both the cache key
#: and the sidecar metadata so a layout bump invalidates everything at once.
CACHE_FORMAT_VERSION = "preprocess-1"


def _canonical(value: Any) -> Any:
    """Fold any configuration value into a JSON-serialisable, order-stable form.

    ``AudioConfig`` and its sub-configs are frozen dataclasses with frozenset
    members, which ``json.dumps`` cannot serialise directly. Reducing to
    plain JSON makes the signature a pure function of the configuration's
    values rather than of dictionary insertion order or set iteration order.
    Anything this cannot classify falls back to ``repr``, which is stable for
    the immutable values a configuration holds.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (Sequence, AbstractSet)) and not isinstance(value, (str, bytes)):
        items = sorted((_canonical(item) for item in value), key=str)
        return {"#items": items}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonical(dataclasses.asdict(value))
    return repr(value)


def preprocessing_signature(audio_config: AudioConfig | None = None) -> str:
    """Stable digest of every ``AudioConfig`` value that can change a sample.

    The signature covers the whole configuration tree -- not a hand-picked
    subset -- because the "…the FULL meaning, not a summary" rule which governs
    the manifest applies here too: a quiet change to a VAD threshold alters which
    samples survive, and a signature that ignored it would serve cached audio
    built under the old threshold. Keeping every value costs a few microseconds
    of hashing and removes the class of bug where "just one field was left out".

    Returns:
        A 64-hex SHA-256 digest of the canonicalised configuration.
    """
    cfg = audio_config or AudioConfig()
    payload = {
        "cache_format": CACHE_FORMAT_VERSION,
        "audio": _canonical(cfg),
        "resolved_normalization": _canonical(cfg.normalization_settings),
        "resolved_segment_hop": cfg.segment_hop_seconds,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cache_key(
    file_hash: str,
    signature: str,
    *,
    format_version: str = CACHE_FORMAT_VERSION,
) -> str:
    """Stable key for one source file under one preprocessing configuration.

    Args:
        file_hash: Content hash of the source file, ``"sha256:<hex>"``.
        signature: Output of :func:`preprocessing_signature`.
        format_version: On-disk layout version. Overridable for tests.

    Returns:
        The 64-hex digest that names this audio under this configuration.
    """
    material = "\0".join((format_version, file_hash, signature))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CachedAudio:
    """The canonical waveform plus the measurements needed to rebuild Phase 1.

    This is exactly the information :class:`PreprocessedAudio` carries, in a form
    that can be serialised to disk. Arrays are never logged; this object exists to
    cross the cache boundary.

    Attributes:
        samples: Float32 mono samples at the canonical rate.
        sample_rate: Rate the samples were canonicalised to.
        gain_db_applied: Loudness gain applied, for audit.
        peak_before_normalize: Peak amplitude prior to gain, for diagnostics.
        clipped: Whether the peak ceiling reduced the signal.
    """

    samples: np.ndarray
    sample_rate: int
    gain_db_applied: float
    peak_before_normalize: float
    clipped: bool

    @classmethod
    def from_preprocessed(cls, audio: PreprocessedAudio) -> CachedAudio:
        """Capture one :class:`PreprocessedAudio` for storage."""
        return cls(
            samples=audio.samples,
            sample_rate=audio.sample_rate,
            gain_db_applied=audio.gain_db_applied,
            peak_before_normalize=audio.peak_before_normalize,
            clipped=audio.clipped,
        )

    def to_preprocessed(self) -> PreprocessedAudio:
        """Rebuild the canonical representation this entry was stored from.

        ``PreprocessedAudio.__post_init__`` re-validates finiteness on the way
        in, so a corrupt entry cannot manufacture an all-NaN signal silently.
        """
        return PreprocessedAudio(
            samples=self.samples,
            sample_rate=self.sample_rate,
            gain_db_applied=self.gain_db_applied,
            peak_before_normalize=self.peak_before_normalize,
            clipped=self.clipped,
        )

    def metadata(self) -> dict[str, Any]:
        """JSON-serialisable record of what this entry is."""
        return {
            "format": CACHE_FORMAT_VERSION,
            "sample_rate": self.sample_rate,
            "gain_db_applied": self.gain_db_applied,
            "peak_before_normalize": self.peak_before_normalize,
            "clipped": self.clipped,
        }


class PreprocessingCache:
    """Content-addressed store for canonical waveforms on disk.

    Lifetimes are disposable: ``cache/`` sits next to ``processed/`` and
    ``interim/`` precisely so that "delete the cache" has exactly one meaning.
    Correctness never depends on this class -- a miss or a corrupt entry flows
    through to a fresh decode and a re-``put``.

    Attributes:
        hits: Entries served from disk during this process.
        misses: Keys looked up and not found.
        corrupt: Entries present but unusable, dropped and re-computed.
    """

    def __init__(
        self,
        root: Path | DataPaths,
        *,
        max_entries: int = 200_000,
        enabled: bool = True,
    ) -> None:
        if max_entries < 1:
            msg = f"max_entries must be at least 1, got {max_entries}"
            raise DatasetConfigError(msg)
        base = root.cache if isinstance(root, DataPaths) else Path(root)
        base.mkdir(parents=True, exist_ok=True)
        self._dir = base / "preprocess"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max_entries = max_entries
        self._enabled = enabled
        self.hits = 0
        self.misses = 0
        self.corrupt = 0
        self._known: set[str] | None = None

    @classmethod
    def for_config(cls, config: DataConfig) -> PreprocessingCache:
        """Build the cache a :class:`~voxshield.data.config.DataConfig` describes.

        ``enabled`` and ``max_entries`` come from ``config.cache``, so switching
        the cache off in configuration disables it here without a code change.
        """
        settings: CacheConfig = config.cache
        return cls(
            config.data_paths(),
            max_entries=settings.max_entries,
            enabled=settings.enabled,
        )

    # -- filesystem layout -------------------------------------------------

    def audio_path(self, key: str) -> Path:
        """Where this key's waveform is stored."""
        return self._dir / key[:2] / f"{key}.npy"

    def meta_path(self, key: str) -> Path:
        """Where this key's metadata sidecar is stored."""
        return self._dir / key[:2] / f"{key}.json"

    # -- read / write ------------------------------------------------------

    def get(self, key: str) -> CachedAudio | None:
        """Return the cached audio for ``key``, or ``None`` on any miss reason.

        A missing file, a missing sidecar, a format mismatch, or an unreadable or
        non-finite payload are all treated equally: ``None``. The cache is an
        accelerator, so any doubt resolves in favour of recomputing rather than
        serving possibly-wrong bytes.
        """
        if not self._enabled:
            return None
        audio_path = self.audio_path(key)
        meta_path = self.meta_path(key)
        if not audio_path.exists() or not meta_path.exists():
            self.misses += 1
            return None
        try:
            samples = np.load(audio_path, allow_pickle=False)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            self._drop(key)
            self.corrupt += 1
            return None
        if (
            not isinstance(meta, Mapping)
            or meta.get("format") != CACHE_FORMAT_VERSION
            or not isinstance(samples, np.ndarray)
            or samples.ndim != 1
            or samples.dtype != np.float32
            or not np.isfinite(samples).all()
        ):
            self._drop(key)
            self.corrupt += 1
            return None
        try:
            cached = CachedAudio(
                samples=samples,
                sample_rate=int(meta["sample_rate"]),
                gain_db_applied=float(meta["gain_db_applied"]),
                peak_before_normalize=float(meta["peak_before_normalize"]),
                clipped=bool(meta["clipped"]),
            )
        except (KeyError, TypeError, ValueError):
            self._drop(key)
            self.corrupt += 1
            return None
        self.hits += 1
        return cached

    def put(self, key: str, audio: CachedAudio | PreprocessedAudio) -> Path:
        """Store ``audio`` under ``key`` and return the path written.

        The soft cap is enforced against the real entry count, so the cache
        never settles above ``max_entries``; the oldest entries (by modification
        time) are removed once the count exceeds it.
        """
        if not self._enabled:
            return self.audio_path(key)
        cached = audio if isinstance(audio, CachedAudio) else CachedAudio.from_preprocessed(audio)
        if cached.samples.ndim != 1 or cached.samples.size == 0:
            msg = "refusing to cache an empty or non-mono waveform"
            raise DatasetBuildError(msg)
        audio_path = self.audio_path(key)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(audio_path, np.ascontiguousarray(cached.samples, dtype=np.float32))
        meta_path = self.meta_path(key)
        meta_path.write_text(
            json.dumps({**cached.metadata(), "created_ns": time.time_ns()}, sort_keys=True),
            encoding="utf-8",
        )
        known = self._keys()
        known.add(key)
        if len(known) > self._max_entries:
            self.prune()
        return audio_path

    def _drop(self, key: str) -> None:
        """Remove a key's files best-effort; a failed removal is a future miss.

        The tracked key set is updated too: a dropped entry that lingers in it
        would inflate ``len(cache)`` and make the count disagree with disk.
        """
        for path in (self.audio_path(key), self.meta_path(key)):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        if self._known is not None:
            self._known.discard(key)

    # -- lifecycle ---------------------------------------------------------

    def _keys(self) -> set[str]:
        """Cache keys present on disk, discovered lazily once per process."""
        if self._known is None:
            self._known = {path.stem for path in self._dir.glob("*/*.npy")}
        return self._known

    def _age_key(self, key: str) -> tuple[int, int, str]:
        """Sort key for eviction: modification time, then write order, then key.

        Modification time is the primary signal because age is the property that
        matters -- a backdated entry is stale regardless of when it was written.
        It is not, however, a total order: a filesystem with coarse timestamp
        granularity stamps several writes in the same tick, and ties then fall
        back to comparing key hashes, which is arbitrary relative to insertion.
        ``created_ns`` from the sidecar breaks those ties in the order the
        entries were actually written, so eviction is deterministic instead of
        filesystem-dependent.
        """
        try:
            mtime = self.audio_path(key).stat().st_mtime_ns
        except OSError:
            mtime = 0
        created = 0
        try:
            payload = json.loads(self.meta_path(key).read_text(encoding="utf-8"))
            created = int(payload.get("created_ns", 0))
        except (OSError, ValueError, TypeError):
            created = 0
        return (mtime, created, key)

    def prune(self) -> int:
        """Delete the oldest entries until the count is within ``max_entries``.

        Returns:
            Number of entries removed.
        """
        keys = self._keys()
        if len(keys) <= self._max_entries:
            return 0
        aged: list[tuple[tuple[int, int, str], str]] = []
        for key in keys:
            age = self._age_key(key)
            if age[0] == 0:
                # Tracked but already gone from disk; drop the stale key rather
                # than failing the prune.
                keys.discard(key)
                continue
            aged.append((age, key))
        aged.sort()
        removed = 0
        for _, key in aged:
            if len(keys) <= self._max_entries:
                break
            self._drop(key)
            removed += 1
        return removed

    def clear(self) -> None:
        """Delete every entry. The empty cache is still a working cache."""
        for path in self._dir.glob("*/*.npy"):
            path.unlink(missing_ok=True)
        for path in self._dir.glob("*/*.json"):
            path.unlink(missing_ok=True)
        if self._known is not None:
            self._known.clear()

    def __len__(self) -> int:
        """Number of entries currently on disk."""
        return len(self._keys())

    def __contains__(self, key: str) -> bool:
        return key in self._keys()
