"""Unit tests for the segmentation phase.

These tests exercise :mod:`voxshield.data.preprocess` from the record boundary
inward: what a file becomes a segment, and what a file does *not* become one.
The rejection paths here matter because this phase sits after validation --
a file admitted there can still be turned away here (too long after resampling,
inaudible after phase 1, a container the decoder rejects), and each of those
refusals must produce a code and a measured reason rather than a silent drop.

Audio is synthesised rather than read from fixtures, on the same logic as the
validation tests: a waveform's duration, level, and speech content are values
the test chooses exactly, and that is what segmentation is about.
"""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from tests.conftest import harmonic_speech, to_wav_bytes, white_noise

from voxshield.data.cache import PreprocessingCache
from voxshield.data.config import CacheConfig, DataConfig, DatasetEntry, ValidationConfig
from voxshield.data.errors import SplitError
from voxshield.data.paths import DataPaths
from voxshield.data.preprocess import (
    PREPROCESS_REJECTIONS,
    REJECT_INSUFFICIENT_SPEECH,
    REJECT_INVALID_SIGNAL,
    REJECT_TOO_LARGE,
    REJECT_UNDECODABLE,
    REJECT_UNREADABLE,
    REJECT_UNSUPPORTED_FORMAT,
    preprocess_source,
    preprocess_source_bytes,
    preprocess_sources,
    preprocessing_version,
    segment_content_hash,
)
from voxshield.data.schema import SourceRecord

BONA_FIDE = "bona_fide"

SAMPLE_RATE = 16_000


def source(sample_id: str = "s1", audio_path: str = "raw/s1.wav") -> SourceRecord:
    """A validated source record with provenance this phase forwards."""
    return SourceRecord(
        sample_id=sample_id,
        dataset_id="corpus",
        audio_path=audio_path,
        label=BONA_FIDE,
        speaker_id="spk1",
        generator_id="g1",
        language="en",
        codec="pcm",
        session_id="ses1",
        attack_type="vocoder",
        channel="mono",
        recorded_at="2024-01-01",
    )


def dataset_entry(dataset_id: str = "corpus") -> DatasetEntry:
    """A minimal enabled registry entry."""
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


def make_config(
    tmp_path: Path,
    *,
    validation: ValidationConfig | None = None,
    cache: CacheConfig | None = None,
    write_segment_audio: bool = True,
) -> DataConfig:
    """A build config rooted in the test's sandbox directory."""
    return DataConfig(
        root=str(tmp_path),
        datasets=(dataset_entry(),),
        validation=validation or ValidationConfig(),
        cache=cache or CacheConfig(),
        write_segment_audio=write_segment_audio,
    )


def write_wav(root: Path, relative: str, samples: np.ndarray) -> Path:
    """Write mono float samples as a WAV and return the path."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")
    return path


def samples_from_wav(payload: bytes) -> np.ndarray:
    """Decode in-memory WAV bytes back to samples, for rewriting to disk."""
    return sf.read(BytesIO(payload), dtype="float32")[0]


class TestAcceptance:
    """A speech-like file becomes segments that inherit the source wholesale."""

    def test_segments_inherit_split_and_source_identity(
        self, tmp_path: Path, speech_wav: bytes
    ) -> None:
        config = make_config(tmp_path)
        outcome = preprocess_source_bytes(speech_wav, source(), config=config, split="train")

        assert outcome.accepted
        assert outcome.rejected_code is None
        assert outcome.split == "train"
        assert outcome.n_segments >= 1
        assert outcome.from_cache is False
        assert outcome.file_hash.startswith("sha256:")
        assert outcome.preprocessing_version.startswith("phase1.")

        first = outcome.segments[0]
        assert first.sample_id == "s1#0000"
        assert first.split == "train"
        assert first.parent_id == "s1"
        assert first.speaker_id == source().speaker_id
        assert first.generator_id == "g1"
        assert first.dataset_build_id == ""
        assert first.content_hash.startswith("sha256:")
        assert first.sample_rate == SAMPLE_RATE

    def test_rejection_code_comes_from_the_closed_vocabulary(
        self, tmp_path: Path, speech_wav: bytes
    ) -> None:
        config = make_config(tmp_path)
        for payload, expected in [
            (to_wav_bytes(white_noise(4.0)), REJECT_INSUFFICIENT_SPEECH),
            (
                to_wav_bytes(white_noise(4.0), fmt="OGG", subtype="VORBIS"),
                REJECT_UNSUPPORTED_FORMAT,
            ),
            (b"this is not audio", REJECT_UNDECODABLE),
        ]:
            outcome = preprocess_source_bytes(payload, source(), config=config)
            assert outcome.rejected_code == expected
            assert outcome.rejected_code in PREPROCESS_REJECTIONS
            assert outcome.segments == ()

    def test_digital_silence_is_rejected(self, tmp_path: Path) -> None:
        """A wholly silent file has no signal to judge at all.

        This is distinct from ``insufficient_speech``: decode refuses an
        entirely silent signal before any speech estimate is taken, so the
        rejection is ``invalid_signal``.
        """
        config = make_config(tmp_path)
        outcome = preprocess_source_bytes(
            to_wav_bytes(np.zeros(4 * SAMPLE_RATE, dtype=np.float32)),
            source(),
            config=config,
        )

        assert outcome.rejected_code == REJECT_INVALID_SIGNAL
        assert "silent" in (outcome.rejected_reason or "").lower()

    def test_missing_file_is_unreadable_and_named(self, tmp_path: Path) -> None:
        config = make_config(tmp_path)
        outcome = preprocess_source(source("s1", "raw/absent.wav"), config=config)

        assert outcome.rejected_code == REJECT_UNREADABLE
        assert "stat" in (outcome.rejected_reason or "").lower()

    def test_file_over_the_size_bound_is_rejected(self, tmp_path: Path, speech_wav: bytes) -> None:
        config = make_config(tmp_path, validation=ValidationConfig(max_file_bytes=1024))
        write_wav(tmp_path, "raw/s1.wav", samples_from_wav(speech_wav))

        outcome = preprocess_source(source(), config=config)

        assert outcome.rejected_code == REJECT_TOO_LARGE
        assert "max_file_bytes" in (outcome.rejected_reason or "")

    def test_size_bound_is_not_reapplied_on_the_bytes_variant(
        self, tmp_path: Path, speech_wav: bytes
    ) -> None:
        # The stat path applies validation.max_file_bytes; the in-memory adapter
        # path has its own separate 10 MiB upload floor. The two must not
        # disagree on the same bytes.
        config = make_config(tmp_path, validation=ValidationConfig(max_file_bytes=1024))

        outcome = preprocess_source_bytes(speech_wav, source(), config=config)

        assert outcome.accepted


class TestCacheIntegration:
    """The preprocessing cache changes nothing except where the samples came from."""

    def test_second_build_is_a_cache_hit_with_identical_segments(
        self, tmp_path: Path, speech_wav: bytes
    ) -> None:
        config = make_config(tmp_path)
        cache = PreprocessingCache.for_config(config)
        rec = source()

        first = preprocess_source_bytes(speech_wav, rec, config=config, cache=cache)
        second = preprocess_source_bytes(speech_wav, rec, config=config, cache=cache)

        assert first.accepted and second.accepted
        assert first.from_cache is False
        assert second.from_cache is True
        assert [seg.sample_id for seg in first.segments] == [
            seg.sample_id for seg in second.segments
        ]
        assert first.preprocessing_version == second.preprocessing_version

    def test_cached_build_matches_an_uncached_build(
        self, tmp_path: Path, speech_wav: bytes
    ) -> None:
        config = make_config(tmp_path)
        cache = PreprocessingCache.for_config(config)
        rec = source()

        cached = preprocess_source_bytes(speech_wav, rec, config=config, cache=cache)
        plain = preprocess_source_bytes(speech_wav, rec, config=config, cache=None)

        assert cached.accepted and plain.accepted
        assert cached.preprocessing_version == plain.preprocessing_version
        assert [seg.content_hash for seg in cached.segments] == [
            seg.content_hash for seg in plain.segments
        ]

    def test_disabled_cache_is_never_touched(self, tmp_path: Path, speech_wav: bytes) -> None:
        config = make_config(tmp_path, cache=CacheConfig(enabled=False))
        cache = PreprocessingCache.for_config(config)

        outcome = preprocess_source_bytes(speech_wav, source(), config=config, cache=cache)

        assert outcome.accepted
        assert outcome.from_cache is False
        assert cache.hits == 0 and cache.misses == 0


class TestBatch:
    """Batch builds enforce split-before-segmentation, then handle the split."""

    def test_missing_assignment_is_a_split_error(self, tmp_path: Path, speech_wav: bytes) -> None:
        write_wav(tmp_path, "raw/s1.wav", samples_from_wav(speech_wav))
        config = make_config(tmp_path)

        with pytest.raises(SplitError):
            preprocess_sources([source()], config=config, split_of={})

    def test_batch_assigns_each_source_its_split(self, tmp_path: Path, speech_wav: bytes) -> None:
        write_wav(tmp_path, "raw/s1.wav", samples_from_wav(speech_wav))
        config = make_config(tmp_path)

        outcomes = preprocess_sources([source()], config=config, split_of={"s1": "test"})

        assert len(outcomes) == 1
        assert outcomes[0].accepted
        assert outcomes[0].split == "test"
        assert outcomes[0].segments[0].split == "test"


class TestStoredSegments:
    """Stored audio is standardised WAV under processed/segments, or refused."""

    def test_written_segments_are_valid_wav(self, tmp_path: Path, speech_wav: bytes) -> None:
        config = make_config(tmp_path)
        outcome = preprocess_source_bytes(speech_wav, source(), config=config)
        paths = DataPaths.resolve(tmp_path)

        assert outcome.accepted
        for index, segment in enumerate(outcome.segments):
            relative = f"processed/segments/corpus/s1__s{index:04d}.wav"
            assert segment.audio_path == relative
            stored = paths.root / segment.audio_path
            assert stored.exists()
            samples, rate = sf.read(stored, dtype="float32")
            assert rate == SAMPLE_RATE
            assert samples.ndim == 1

    def test_no_stored_audio_when_disabled(self, tmp_path: Path, speech_wav: bytes) -> None:
        config = make_config(tmp_path, write_segment_audio=False)
        outcome = preprocess_source_bytes(speech_wav, source(), config=config)

        assert outcome.accepted
        assert outcome.segments
        assert all(segment.audio_path == "raw/s1.wav" for segment in outcome.segments)


class TestVersions:
    """The preprocessing stamp and content hash are determinism guarantees."""

    def test_preprocessing_version_is_stable_and_scoped(self) -> None:
        from voxshield.config import AudioConfig as CoreAudioConfig

        first = preprocessing_version(CoreAudioConfig())
        second = preprocessing_version(CoreAudioConfig())

        assert first == second
        assert first.startswith("phase1.")
        assert len(first.split(".")[1]) == 16

    def test_preprocessing_version_changes_with_audio_config(self) -> None:
        from voxshield.config import AudioConfig as CoreAudioConfig
        from voxshield.config import FeatureConfig as CoreFeatureConfig

        base = CoreAudioConfig()
        changed = replace(
            CoreAudioConfig(),
            target_sample_rate=8_000,
            features=CoreFeatureConfig(sample_rate=8_000),
        )

        assert preprocessing_version(base) != preprocessing_version(changed)

    def test_segment_content_hash_indexes_the_canonical_signal(self) -> None:
        window = harmonic_speech(4.0)
        other = harmonic_speech(4.0, base_f0=150.0)

        assert segment_content_hash(window).startswith("sha256:")
        assert segment_content_hash(window) == segment_content_hash(window.copy())
        assert segment_content_hash(window) != segment_content_hash(other)
