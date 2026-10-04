"""Unit tests for the content-addressed preprocessing cache.

Two properties matter enough to test directly, because either failure is
silent in a happy-path build:

* a stale hit is impossible -- the key covers the file hash, the preprocessing
  signature, and the format version, so changing any one changes the key and
  the build recomputes instead of serving old audio;
* a bad entry is a miss, never a result -- the cache is an accelerator, so a
  corrupt, truncated, or wrongly-formatted entry must flow through to a fresh
  decode rather than to possibly-wrong bytes.

The rest is bookkeeping: the round-trip of the waveform is exact, the soft cap
is respected, and the refundable "disable" switch actually stops both reads and
writes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from tests.conftest import harmonic_speech

from voxshield.audio.preprocess import PreprocessedAudio
from voxshield.config import AudioConfig, FeatureConfig
from voxshield.data.cache import (
    CACHE_FORMAT_VERSION,
    CachedAudio,
    PreprocessingCache,
    cache_key,
    preprocessing_signature,
)
from voxshield.data.config import CacheConfig, DataConfig, DatasetEntry
from voxshield.data.errors import DatasetBuildError, DatasetConfigError
from voxshield.data.paths import DataPaths

SAMPLE_RATE = 16_000


def canned_audio(seconds: float = 4.0) -> PreprocessedAudio:
    """A canonical waveform with the measurements Phase 1 would record."""
    return PreprocessedAudio(
        samples=harmonic_speech(seconds),
        sample_rate=SAMPLE_RATE,
        gain_db_applied=4.2,
        peak_before_normalize=0.7,
        clipped=False,
    )


def dataset_entry(dataset_id: str = "corpus") -> DatasetEntry:
    """A minimal enabled registry entry, mirroring the validation tests."""
    return DatasetEntry(
        dataset_id=dataset_id,
        name="Corpus",
        source="local",
        version="1",
        license="unknown",
        license_status="UNKNOWN",
        task="spoof_detection",
        enabled=True,
        path="raw",
        adapter="generic",
    )


class TestPreprocessingSignature:
    def test_is_stable_across_calls(self) -> None:
        assert preprocessing_signature() == preprocessing_signature()

    def test_same_config_from_different_callers_is_equal(self) -> None:
        config = AudioConfig(
            target_sample_rate=8_000,
            min_speech_seconds=1.5,
            features=FeatureConfig(sample_rate=8_000),
        )
        assert preprocessing_signature(config) == preprocessing_signature(config)

    def test_changes_when_an_audio_flag_changes(self) -> None:
        base = AudioConfig()
        assert preprocessing_signature(base) != preprocessing_signature(
            AudioConfig(min_speech_seconds=2.0)
        )

    def test_changes_when_the_sample_rate_changes(self) -> None:
        assert preprocessing_signature() != preprocessing_signature(
            AudioConfig(target_sample_rate=8_000, features=FeatureConfig(sample_rate=8_000))
        )

    def test_resolved_hop_is_part_of_the_signature(self) -> None:
        """A change to meaningful segmentation output must invalidate the cache."""
        explicit_hop = AudioConfig(segment_hop_seconds=2.0)
        assert preprocessing_signature(explicit_hop) != preprocessing_signature()

    def test_explicit_none_hop_matches_the_default(self) -> None:
        """``None`` hop means half the window; writing it explicitly must not
        rotate the signature away from the default configuration."""
        assert preprocessing_signature(AudioConfig(segment_hop_seconds=None)) == (
            preprocessing_signature()
        )

    def test_covers_the_full_configuration_tree(self) -> None:
        """A quiet VAD change must rotate the signature, not a hand-picked subset."""
        with_norm = AudioConfig(target_rms_dbfs=-26.0)
        assert preprocessing_signature() != preprocessing_signature(with_norm)


class TestCacheKey:
    def test_is_stable(self) -> None:
        key = cache_key("sha256:aaaa", "sig")
        assert key == cache_key("sha256:aaaa", "sig")

    def test_changes_with_the_file_hash(self) -> None:
        assert cache_key("sha256:aaaa", "sig") != cache_key("sha256:bbbb", "sig")

    def test_changes_with_the_signature(self) -> None:
        assert cache_key("sha256:aaaa", "sig-one") != cache_key("sha256:aaaa", "sig-two")

    def test_changes_with_the_format_version(self) -> None:
        assert cache_key("sha256:aaaa", "sig", format_version="preprocess-1") != cache_key(
            "sha256:aaaa", "sig", format_version="preprocess-2"
        )

    def test_the_format_version_is_rasterised_the_default(self) -> None:
        assert cache_key("sha256:aaaa", "sig") == cache_key(
            "sha256:aaaa", "sig", format_version=CACHE_FORMAT_VERSION
        )

    def test_key_is_a_hex_digest(self) -> None:
        key = cache_key("sha256:aaaa", "sig")
        assert len(key) == 64
        int(key, 16)  # raises for anything but hex


class TestCachedAudio:
    def test_round_trip_is_bit_exact(self) -> None:
        original = canned_audio()
        cached = CachedAudio.from_preprocessed(original)
        rebuilt = cached.to_preprocessed()
        np.testing.assert_array_equal(rebuilt.samples, original.samples)
        assert rebuilt.sample_rate == original.sample_rate
        assert rebuilt.gain_db_applied == original.gain_db_applied
        assert rebuilt.peak_before_normalize == original.peak_before_normalize
        assert rebuilt.clipped == original.clipped

    def test_metadata_is_json_serialisable(self) -> None:
        meta = CachedAudio.from_preprocessed(canned_audio()).metadata()
        json.dumps(meta)
        assert meta["format"] == CACHE_FORMAT_VERSION
        assert meta["sample_rate"] == SAMPLE_RATE


class TestPreprocessingCacheLifecycle:
    def test_miss_on_a_fresh_cache(self, tmp_path: Path) -> None:
        cache = PreprocessingCache(tmp_path)
        assert cache.get(cache_key("sha256:aaaa", "sig")) is None
        assert cache.misses == 1

    def test_put_then_get_round_trips_the_waveform(self, tmp_path: Path) -> None:
        cache = PreprocessingCache(tmp_path)
        audio = canned_audio()
        key = cache_key("sha256:aaaa", "sig")
        cache.put(key, audio)
        served = cache.get(key)
        assert served is not None
        np.testing.assert_array_equal(served.samples, audio.samples)
        assert served.sample_rate == SAMPLE_RATE
        assert cache.hits == 1

    def test_contains_and_length_track_disk(self, tmp_path: Path) -> None:
        cache = PreprocessingCache(tmp_path)
        first = cache_key("sha256:aaaa", "sig")
        second = cache_key("sha256:bbbb", "sig")
        cache.put(first, canned_audio())
        assert first in cache
        assert second not in cache
        assert len(cache) == 1
        cache.put(second, canned_audio())
        assert len(cache) == 2

    def test_put_after_get_ids_are_stable(self, tmp_path: Path) -> None:
        """The stored file names are the input key, never a derived one."""
        cache = PreprocessingCache(tmp_path)
        key = cache_key("sha256:aaaa", "sig")
        cache.put(key, canned_audio())
        assert cache.audio_path(key).exists()
        assert cache.meta_path(key).exists()

    def test_a_disabled_cache_stores_nothing_and_serves_nothing(self, tmp_path: Path) -> None:
        cache = PreprocessingCache(tmp_path, enabled=False)
        key = cache_key("sha256:aaaa", "sig")
        cache.put(key, canned_audio())
        assert cache.get(key) is None
        assert key not in cache
        assert len(cache) == 0

    def test_for_config_carries_the_enabled_switch(self, tmp_path: Path) -> None:
        config = DataConfig(
            root=str(tmp_path),
            datasets=(dataset_entry(),),
            cache=CacheConfig(enabled=False),
        )
        cache = PreprocessingCache.for_config(config)
        key = cache_key("sha256:aaaa", "sig")
        cache.put(key, canned_audio())
        assert len(cache) == 0

    def test_for_config_accepts_a_paths_root(self, tmp_path: Path) -> None:
        paths = DataPaths.resolve(str(tmp_path), {})
        cache = PreprocessingCache(paths)
        assert paths.cache == (tmp_path / "cache")
        # A DataPaths root resolves through .cache, not through the root itself.
        assert cache._dir == paths.cache / "preprocess"
        assert cache._dir.is_dir()

    def test_max_entries_must_be_at_least_one(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetConfigError, match="max_entries"):
            PreprocessingCache(tmp_path, max_entries=0)

    def test_refuses_non_mono_or_empty_waveforms(self, tmp_path: Path) -> None:
        cache = PreprocessingCache(tmp_path)
        stereo = canned_audio()
        stereo = PreprocessedAudio(
            samples=np.stack([stereo.samples, stereo.samples], axis=1),
            sample_rate=SAMPLE_RATE,
            gain_db_applied=0.0,
            peak_before_normalize=0.0,
            clipped=False,
        )
        with pytest.raises(DatasetBuildError, match="empty or non-mono"):
            cache.put(cache_key("sha256:aaaa", "sig"), stereo)
        empty = PreprocessedAudio(
            samples=np.zeros(0, dtype=np.float32),
            sample_rate=SAMPLE_RATE,
            gain_db_applied=0.0,
            peak_before_normalize=0.0,
            clipped=False,
        )
        with pytest.raises(DatasetBuildError, match="empty or non-mono"):
            cache.put(cache_key("sha256:bbbb", "sig"), empty)


class TestCorruptEntries:
    def test_a_truncated_waveform_is_a_miss_and_drops_the_entry(self, tmp_path: Path) -> None:
        cache = PreprocessingCache(tmp_path)
        key = cache_key("sha256:aaaa", "sig")
        cache.put(key, canned_audio())
        cache.audio_path(key).write_bytes(b"XX-short")
        assert cache.get(key) is None
        assert key not in cache
        assert cache.corrupt == 1

    def test_a_wrong_format_sidecar_is_a_miss(self, tmp_path: Path) -> None:
        cache = PreprocessingCache(tmp_path)
        key = cache_key("sha256:aaaa", "sig")
        cache.put(key, canned_audio())
        meta_path = cache.meta_path(key)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["format"] = "preprocess-2"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        assert cache.get(key) is None
        assert cache.corrupt == 1

    def test_a_unreadable_json_sidecar_is_a_miss(self, tmp_path: Path) -> None:
        cache = PreprocessingCache(tmp_path)
        key = cache_key("sha256:aaaa", "sig")
        cache.put(key, canned_audio())
        cache.meta_path(key).write_text("{not json", encoding="utf-8")
        assert cache.get(key) is None
        assert cache.corrupt == 1

    def test_a_missing_sidecar_is_a_miss_but_not_corrupt(self, tmp_path: Path) -> None:
        cache = PreprocessingCache(tmp_path)
        key = cache_key("sha256:aaaa", "sig")
        cache.put(key, canned_audio())
        cache.meta_path(key).unlink()
        assert cache.get(key) is None
        assert cache.misses == 1
        assert cache.corrupt == 0

    def test_a_non_finite_waveform_is_a_miss(self, tmp_path: Path) -> None:
        """A corrupted payload with NaN samples is a miss, never a served hit: the
        cache's contract is "any doubt resolves in favour of recomputing"."""
        cache = PreprocessingCache(tmp_path)
        key = cache_key("sha256:aaaa", "sig")
        cache.put(key, canned_audio())
        np.save(cache.audio_path(key), np.full(10, np.nan, dtype=np.float32))
        served = cache.get(key)
        assert served is None
        assert cache.corrupt == 1


class TestPrune:
    def test_removes_oldest_beyond_the_soft_cap(self, tmp_path: Path) -> None:
        cache = PreprocessingCache(tmp_path, max_entries=2)
        keys = [cache_key(f"sha256:{item:04d}", "sig") for item in range(4)]
        for key in keys:
            cache.put(key, canned_audio())
        assert len(cache) == 2
        assert keys[0] not in cache
        assert keys[1] not in cache
        assert keys[2] in cache
        assert keys[3] in cache

    def test_uses_modification_time_not_insertion_time(self, tmp_path: Path) -> None:
        """Age is what matters for eviction, so a backdated entry loses first.

        The cap is enforced as soon as a write takes the count over it, so the
        eviction happens inside the third ``put`` rather than waiting for a
        later explicit prune.
        """
        cache = PreprocessingCache(tmp_path, max_entries=2)
        old = cache_key("sha256:aaaa", "sig")
        young = cache_key("sha256:bbbb", "sig")
        newest = cache_key("sha256:cccc", "sig")
        cache.put(old, canned_audio())
        cache.put(young, canned_audio())
        os.utime(cache.audio_path(old), (1_700_000_000, 1_700_000_000))
        cache.put(newest, canned_audio())

        # Old is the true oldest despite being written first; young and newest survive.
        assert old not in cache
        assert young in cache
        assert newest in cache
        assert len(cache) == 2

    def test_prune_with_room_to_spare_removes_nothing(self, tmp_path: Path) -> None:
        cache = PreprocessingCache(tmp_path, max_entries=10)
        cache.put(cache_key("sha256:aaaa", "sig"), canned_audio())
        assert cache.prune() == 0
        assert len(cache) == 1

    @pytest.mark.parametrize("cap", [1, 2, 3, 7, 25])
    def test_cap_holds_across_a_long_write_batch(self, tmp_path: Path, cap: int) -> None:
        """The cap is enforced on the real entry count, not a write counter.

        A counter-driven trigger resets after each prune and therefore lets the
        cache settle well above the cap; a cap that only holds for small write
        batches is not a cap.
        """
        cache = PreprocessingCache(tmp_path, max_entries=cap)
        for item in range(cap * 3):
            cache.put(cache_key(f"sha256:{item:04d}", "sig"), canned_audio())
        assert len(cache) <= cap

    def test_eviction_keeps_the_newest_and_leaves_no_orphans(self, tmp_path: Path) -> None:
        cache = PreprocessingCache(tmp_path, max_entries=3)
        keys = [cache_key(f"sha256:{item:04d}", "sig") for item in range(8)]
        for key in keys:
            cache.put(key, canned_audio())

        assert [key for key in keys if key in cache] == keys[-3:]
        # Both halves of an entry must go together, or a later read finds a
        # waveform with no metadata beside it.
        assert len(list(cache._dir.glob("*/*.npy"))) == 3
        assert len(list(cache._dir.glob("*/*.json"))) == 3

    def test_cap_holds_when_a_fresh_handle_reopens_an_overfull_cache(self, tmp_path: Path) -> None:
        """A cache written by an older, looser run is brought back under the cap."""
        writer = PreprocessingCache(tmp_path, max_entries=50)
        for item in range(12):
            writer.put(cache_key(f"sha256:{item:04d}", "sig"), canned_audio())

        reopened = PreprocessingCache(tmp_path, max_entries=4)
        assert len(reopened) == 12

        reopened.put(cache_key("sha256:9999", "sig"), canned_audio())
        assert len(reopened) <= 4

    def test_clear_empties_everything(self, tmp_path: Path) -> None:
        cache = PreprocessingCache(tmp_path)
        for item in range(3):
            cache.put(cache_key(f"sha256:{item:04d}", "sig"), canned_audio())
        cache.clear()
        assert len(cache) == 0
        assert not any(cache._dir.glob("*/*"))

    def test_eviction_is_insertion_ordered_when_writes_share_a_timestamp(
        self, tmp_path: Path
    ) -> None:
        """Ties on modification time must not be broken by key order.

        A filesystem with coarse timestamp granularity stamps several writes in
        the same tick, so modification time is not a total order. Breaking those
        ties by comparing keys picks an arbitrary victim, which makes eviction
        nondeterministic across filesystems and can keep a stale entry while
        dropping a fresh one. The sidecar's ``created_ns`` is the tie-break.
        """
        cache = PreprocessingCache(tmp_path, max_entries=8)
        keys = [cache_key(f"sha256:{item:04d}", "sig") for item in range(4)]
        for key in keys:
            cache.put(key, canned_audio())

        # Force the tie that a coarse-granularity filesystem would produce.
        for key in keys:
            os.utime(cache.audio_path(key), (1_700_000_000, 1_700_000_000))
        for key in keys:
            cache.put(key, canned_audio())

        cache = PreprocessingCache(tmp_path, max_entries=2)
        assert cache.prune() == 2
        assert [key for key in keys if key in cache] == keys[-2:]
