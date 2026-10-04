"""Build orchestration, and the cache contract ``build_dataset`` advertises.

The behaviour under test is the one an operator is told to expect in the
docstring: ``cache=None`` reuses the cache the configuration describes, while a
disabled cache is a genuine no-op that neither reads nor writes. Getting that
backwards is invisible in a unit test of either piece alone -- it only shows up
as a second build that is mysteriously just as slow as the first.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxshield.data import manifest as manifest_io
from voxshield.data import splitting
from voxshield.data.build import (
    build_dataset,
    discover_corpus,
    jsonable,
    segment_corpus,
    split_corpus,
)
from voxshield.data.cache import PreprocessingCache
from voxshield.data.config import (
    CacheConfig,
    DataConfig,
    DatasetEntry,
    SplitConfig,
    ValidationConfig,
)
from voxshield.data.errors import (
    DatasetBuildError,
    DatasetConfigError,
    ManifestError,
)
from voxshield.data.validation import validate_sources

SAMPLE_RATE = 16_000


def speechish(seed: int, seconds: float = 6.0) -> np.ndarray:
    """A speech-shaped signal that survives the validator's speech gate.

    The envelope matters more than the timbre here: a flat buzz with no syllabic
    modulation is rejected as ``NO_SPEECH``, and a corpus built from those is
    silently single-class.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    f0 = 110.0 + 8.0 * (seed % 6)
    signal = np.zeros_like(t)
    for harmonic, weight in ((1, 1.0), (2, 0.7), (3, 0.45), (5, 0.25), (7, 0.15)):
        signal += weight * np.sin(2 * np.pi * f0 * harmonic * t + rng.random())
    signal *= 0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * t)
    signal += 0.01 * rng.standard_normal(t.size)
    peak = float(np.max(np.abs(signal))) or 1.0
    return (signal / peak * 0.6).astype(np.float32)


def dataset_entry(dataset_id: str, *, path: str, label: int) -> DatasetEntry:
    """One enabled registry entry carrying a fixed label in its metadata.

    ``task`` has to agree with the adapter: the registry refuses an entry whose
    declared task and adapter disagree, which is a useful guard but means the
    real-speech side has to declare ``real_speech``, not ``spoof_detection``.
    """
    adapter = "real_speech" if label == 0 else "wavefake"
    task = "real_speech" if label == 0 else "spoof_detection"
    return DatasetEntry(
        dataset_id=dataset_id,
        name=dataset_id,
        source="generated for this test",
        version="1",
        license="project-owned",
        license_status="VERIFIED",
        license_verified_by="fixture",
        task=task,
        enabled=True,
        path=path,
        adapter=adapter,
        metadata={"speaker_pattern": "^spk(?P<speaker>[0-9]{3})_", "label": label},
    )


def write_corpus(root: Path, *, speakers: int = 6) -> None:
    """Write a two-class corpus: real speech plus a differently-shaped signal."""
    real = root / "raw" / "real"
    fake = root / "raw" / "fake" / "train" / "gen" / "melgan"
    real.mkdir(parents=True, exist_ok=True)
    fake.mkdir(parents=True, exist_ok=True)
    for index in range(speakers):
        stem = f"spk{index:03d}"
        source = speechish(seed=index)
        spoof = speechish(seed=index + 100)
        sf.write(real / f"{stem}_a.wav", source, SAMPLE_RATE, subtype="PCM_16")
        sf.write(real / f"{stem}_b.wav", source, SAMPLE_RATE, subtype="PCM_16")
        sf.write(fake / f"{stem}_a.wav", spoof, SAMPLE_RATE, subtype="PCM_16")
        sf.write(fake / f"{stem}_b.wav", spoof, SAMPLE_RATE, subtype="PCM_16")


def build_config(root: Path, *, cache: CacheConfig | None = None) -> DataConfig:
    """A configuration small enough to build in a unit test."""
    return DataConfig(
        root=str(root),
        datasets=(
            dataset_entry("real", path="raw/real", label=0),
            dataset_entry("fake", path="raw/fake", label=1),
        ),
        cache=cache if cache is not None else CacheConfig(enabled=True),
        split=SplitConfig(train_ratio=0.7, dev_ratio=0.15, min_train_speakers=2),
        validation=ValidationConfig(min_duration_seconds=0.3),
    )


@pytest.fixture
def corpus(tmp_path: Path) -> DataConfig:
    """A written two-class corpus and the configuration that describes it."""
    write_corpus(tmp_path)
    return build_config(tmp_path)


def _accepted(config: DataConfig) -> tuple[tuple, splitting.SplitAssignment]:
    """Run discovery and validation, returning accepted records and their splits."""
    paths = config.data_paths()
    _inventory, surviving, _registry = discover_corpus(config, paths)
    result = validate_sources(
        surviving,
        config.root,
        config.validation,
        audio_config=config.audio_config(),
    )
    records = tuple(item.record for item in result.accepted)
    return records, splitting.assign_splits(records, config.split)


class TestConfiguredCacheIsUsedByDefault:
    def test_a_second_build_reports_cache_hits(self, corpus: DataConfig) -> None:
        first = build_dataset(corpus, overwrite=True)
        second = build_dataset(corpus, overwrite=True)
        assert first.cache_hits == 0
        assert second.cache_hits > 0, "a default build must reuse the configured cache"

    def test_the_second_build_matches_the_first(self, corpus: DataConfig) -> None:
        first = build_dataset(corpus, overwrite=True)
        second = build_dataset(corpus, overwrite=True)
        assert second.build_id == first.build_id
        assert [row.sample_id for row in second.rows] == [row.sample_id for row in first.rows]

    def test_a_disabled_cache_is_a_no_op(self, tmp_path: Path) -> None:
        """``--no-cache`` must neither read nor write, so it leaves no warm cache."""
        write_corpus(tmp_path)
        config = build_config(tmp_path, cache=CacheConfig(enabled=False))
        cold = build_dataset(config, overwrite=True)
        assert cold.cache_hits == 0
        assert len(list((tmp_path / "cache" / "preprocess").glob("*.npy"))) == 0

    def test_caching_off_in_configuration_is_honoured(self, tmp_path: Path) -> None:
        write_corpus(tmp_path)
        config = build_config(tmp_path, cache=CacheConfig(enabled=False))
        build_dataset(config, overwrite=True)
        assert len(list((tmp_path / "cache" / "preprocess").glob("*.npy"))) == 0

    def test_an_explicit_disabled_cache_overrides_an_enabled_configuration(
        self, corpus: DataConfig
    ) -> None:
        """An explicit disabled cache wins, so ``--no-cache`` cannot be ignored."""
        disabled = PreprocessingCache(corpus.data_paths(), enabled=False)
        result = build_dataset(corpus, cache=disabled, overwrite=True)
        assert result.cache_hits == 0

    def test_changing_the_audio_config_invalidates_the_cache(self, corpus: DataConfig) -> None:
        """Cache keys carry the preprocessing signature, so edits must miss."""
        build_dataset(corpus, overwrite=True)
        warmed = build_dataset(corpus, overwrite=True)
        assert warmed.cache_hits > 0

        wider = dataclasses.replace(corpus, window_seconds=(corpus.window_seconds or 4.0) + 1.0)
        result = build_dataset(wider, overwrite=True)
        assert result.cache_hits == 0


class TestSegmentCorpusCacheArgument:
    """``None`` means "use the configured cache" at every level, not "no cache".

    This is the subtle one. A caller reaching ``segment_corpus`` directly would
    reasonably read ``None`` as "do not cache", but the default is resolved one
    layer down in ``preprocess_sources``, so a ``None`` here still hits the
    cache. Only an explicitly disabled cache makes a run genuinely cold.
    """

    def test_none_resolves_to_the_configured_cache(self, corpus: DataConfig) -> None:
        paths = corpus.data_paths()
        records, assignment = _accepted(corpus)
        build_id = manifest_io.dataset_build_id(corpus)
        first = segment_corpus(records, corpus, paths, assignment, build_id=build_id, cache=None)
        second = segment_corpus(records, corpus, paths, assignment, build_id=build_id, cache=None)
        assert all(outcome.from_cache is False for outcome in first)
        assert any(outcome.from_cache for outcome in second)

    def test_a_disabled_cache_never_hits(self, corpus: DataConfig) -> None:
        paths = corpus.data_paths()
        records, assignment = _accepted(corpus)
        build_id = manifest_io.dataset_build_id(corpus)
        disabled = PreprocessingCache(paths, enabled=False)
        first = segment_corpus(records, corpus, paths, assignment, build_id=build_id, cache=disabled)
        second = segment_corpus(records, corpus, paths, assignment, build_id=build_id, cache=disabled)
        assert all(outcome.from_cache is False for outcome in first)
        assert all(outcome.from_cache is False for outcome in second)

    def test_a_shared_cache_is_reused(self, corpus: DataConfig) -> None:
        paths = corpus.data_paths()
        records, assignment = _accepted(corpus)
        build_id = manifest_io.dataset_build_id(corpus)
        cache = PreprocessingCache.for_config(corpus)
        first = segment_corpus(records, corpus, paths, assignment, build_id=build_id, cache=cache)
        second = segment_corpus(records, corpus, paths, assignment, build_id=build_id, cache=cache)
        assert all(outcome.from_cache is False for outcome in first)
        assert any(outcome.from_cache for outcome in second)


class TestJsonable:
    def test_preserves_booleans_and_integers(self) -> None:
        payload = jsonable({"flag": True, "count": 3, "ratio": 0.5, "text": "x"})
        assert payload == {"flag": True, "count": 3, "ratio": 0.5, "text": "x"}

    def test_replaces_unserialisable_values_with_a_description(self) -> None:
        payload = jsonable({"path": Path("a/b.wav")})
        assert isinstance(payload["path"], str)

    def test_the_result_round_trips_through_json(self) -> None:
        text = json.dumps(jsonable({"path": Path("a/b.wav"), "flag": True}))
        assert json.loads(text)["flag"] is True


class TestSplitSeed:
    """The configuration owns a ``random_seed`` that is hashed into the build id.

    If the split ignored it, two builds that differed only in that field would
    report different dataset ids while producing identical splits -- the build
    id claiming a difference that does not exist.
    """

    def test_the_default_seed_is_the_configured_one(self, corpus: DataConfig) -> None:
        records, _ = _accepted(corpus)
        assert split_corpus(records, corpus).seed == corpus.random_seed

    def test_an_explicit_seed_overrides_the_configuration(self, corpus: DataConfig) -> None:
        records, _ = _accepted(corpus)
        assert split_corpus(records, corpus, seed=7).seed == 7

    def test_the_configured_seed_is_never_silently_zero(self, corpus: DataConfig) -> None:
        """``0`` is a real seed, so it must be asked for rather than defaulted to."""
        records, _ = _accepted(corpus)
        seeded = split_corpus(records, corpus)
        assert seeded.seed == corpus.random_seed
        assert seeded.to_dict()["seed"] != 0
        for candidate in (0, 5, corpus.random_seed):
            other = split_corpus(records, dataclasses.replace(corpus, random_seed=candidate))
            assert other.seed == candidate, f"random_seed={candidate} did not reach assign_splits"
        restated = split_corpus(records, dataclasses.replace(corpus, random_seed=corpus.random_seed))
        assert restated.to_dict() == seeded.to_dict()


class TestBuildRefusals:
    def test_the_same_build_is_idempotent_without_overwrite(self, corpus: DataConfig) -> None:
        """Re-running one build replaces its own output, which is not a conflict."""
        first = build_dataset(corpus, overwrite=False)
        second = build_dataset(corpus, overwrite=False)
        assert second.build_id == first.build_id

    def test_a_different_build_is_refused_without_overwrite(self, corpus: DataConfig) -> None:
        """Overwriting *someone else's* corpus is the case that needs consent."""
        build_dataset(corpus, overwrite=False)
        changed = dataclasses.replace(corpus, random_seed=corpus.random_seed + 1)
        with pytest.raises((ManifestError, DatasetBuildError)):
            build_dataset(changed, overwrite=False)

    def test_overwrite_true_replaces_a_different_build(self, corpus: DataConfig) -> None:
        build_dataset(corpus, overwrite=False)
        changed = dataclasses.replace(corpus, random_seed=corpus.random_seed + 1)
        assert build_dataset(changed, overwrite=True).cache_hits > 0

    def test_a_single_class_corpus_fails_a_gate(self, tmp_path: Path) -> None:
        """A build that accepted only one class has not produced a usable corpus."""
        real = tmp_path / "raw" / "real"
        real.mkdir(parents=True)
        for index in range(4):
            sf.write(real / f"spk{index:03d}_a.wav", speechish(index), SAMPLE_RATE, subtype="PCM_16")
        config = DataConfig(
            root=str(tmp_path),
            datasets=(dataset_entry("real", path="raw/real", label=0),),
            cache=CacheConfig(enabled=True),
            split=SplitConfig(train_ratio=0.7, dev_ratio=0.15, min_train_speakers=2),
            validation=ValidationConfig(min_duration_seconds=0.3),
        )
        with pytest.raises(DatasetBuildError):
            build_dataset(config, overwrite=True)

    def test_a_missing_dataset_directory_is_reported_not_ignored(self, tmp_path: Path) -> None:
        config = build_config(tmp_path)
        with pytest.raises((DatasetBuildError, DatasetConfigError)):
            build_dataset(config, overwrite=True)