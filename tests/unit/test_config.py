"""Unit tests for the configuration contract.

A configuration mistake is the one class of bug that is invisible in normal
operation and catastrophic under load, so the invariants are tested directly
rather than inferred from the code that happens to use them.
"""

from __future__ import annotations

import pytest

from voxshield.config import (
    AudioConfig,
    FeatureConfig,
    VadConfig,
    load_audio_config,
)


class TestDefaults:
    def test_frozen(self) -> None:
        cfg = AudioConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.max_upload_bytes = 1  # type: ignore[misc]

    def test_defensive_limits_are_bounded(self) -> None:
        cfg = AudioConfig()
        # Every intake limit must have a finite, non-trivial ceiling: an
        # unbounded limit is how a single upload takes the process down.
        assert 0 < cfg.max_upload_bytes <= 100 * 1024 * 1024
        assert 0 < cfg.max_duration_seconds <= 300
        assert 0 < cfg.max_channels <= 32
        assert 0 < cfg.max_sample_rate <= 384_000

    def test_feature_rate_must_match_pipeline_rate(self) -> None:
        # The feature extractor is hard-wired to the canonical rate; a mismatch
        # would produce a spectrogram whose time axis does not match the audio.
        with pytest.raises(ValueError, match=r"features\.sample_rate"):
            AudioConfig(
                target_sample_rate=8_000,
                features=FeatureConfig(sample_rate=16_000),
            )

    def test_segment_bounds_are_ordered(self) -> None:
        cfg = AudioConfig()
        assert cfg.min_segment_seconds <= cfg.max_segment_seconds
        assert cfg.max_segment_seconds <= cfg.segment_seconds


class TestValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"max_duration_seconds": 0.0}, "max_duration_seconds"),
            ({"max_duration_seconds": -1.0}, "max_duration_seconds"),
            ({"target_sample_rate": 0}, "target_sample_rate"),
            ({"min_speech_seconds": 0.0}, "min_speech_seconds"),
            ({"min_speech_seconds": -2.0}, "min_speech_seconds"),
            (
                {"target_sample_rate": 32_000, "max_sample_rate": 16_000},
                "max_sample_rate",
            ),
            ({"min_segment_seconds": 9.0}, "min_segment_seconds cannot exceed"),
            ({"max_segment_seconds": 99.0}, "max_segment_seconds cannot exceed"),
        ],
    )
    def test_rejects_contradictory_limits(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            AudioConfig(**kwargs)

    @pytest.mark.parametrize("percentile", [-1.0, 101.0])
    def test_rejects_out_of_range_percentile(self, percentile: float) -> None:
        with pytest.raises(ValueError, match="seed_percentile"):
            AudioConfig(vad=VadConfig(seed_percentile=percentile))

    @pytest.mark.parametrize("percentile", [-0.1, 100.1])
    def test_rejects_out_of_range_reference_percentile(self, percentile: float) -> None:
        with pytest.raises(ValueError, match="seed_reference_percentile"):
            AudioConfig(vad=VadConfig(seed_reference_percentile=percentile))

    def test_rejects_non_positive_dynamic_range(self) -> None:
        with pytest.raises(ValueError, match="dynamic_range_db"):
            AudioConfig(vad=VadConfig(dynamic_range_db=0.0))


class TestOverrides:
    def test_returns_a_copy(self) -> None:
        base = AudioConfig()
        derived = base.with_overrides(min_speech_seconds=2.5)
        assert base.min_speech_seconds != 2.5
        assert derived.min_speech_seconds == 2.5
        # The original must be untouched, including its nested configs.
        assert base.features == derived.features
        assert base.vad == derived.vad

    def test_validated_after_override(self) -> None:
        # replace() re-runs __post_init__, so an override cannot smuggle in a
        # configuration the constructor would have rejected.
        with pytest.raises(ValueError, match="min_speech_seconds"):
            AudioConfig().with_overrides(min_speech_seconds=-1.0)

    def test_unknown_field_rejected(self) -> None:
        with pytest.raises(TypeError):
            AudioConfig().with_overrides(no_such_field=1)  # type: ignore[call-arg]


class TestEnvironmentLoading:
    def test_defaults_without_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "VOXSHIELD_MAX_UPLOAD_BYTES",
            "VOXSHIELD_MAX_DURATION_SECONDS",
            "VOXSHIELD_MIN_SPEECH_SECONDS",
            "VOXSHIELD_VAD_FLOOR_DBFS",
        ):
            monkeypatch.delenv(name, raising=False)
        assert load_audio_config() == AudioConfig()

    def test_reads_deployment_limits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VOXSHIELD_MAX_UPLOAD_BYTES", "1024")
        monkeypatch.setenv("VOXSHIELD_MAX_DURATION_SECONDS", "12.5")
        monkeypatch.setenv("VOXSHIELD_MIN_SPEECH_SECONDS", "3")
        monkeypatch.setenv("VOXSHIELD_VAD_FLOOR_DBFS", "-70")
        cfg = load_audio_config()
        assert cfg.max_upload_bytes == 1024
        assert cfg.max_duration_seconds == 12.5
        assert cfg.min_speech_seconds == 3.0
        assert cfg.vad.absolute_floor_dbfs == -70.0

    @pytest.mark.parametrize("raw", ["not-a-number", "", "  "])
    def test_malformed_value_uses_or_fails_safely(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv("VOXSHIELD_MAX_DURATION_SECONDS", raw)
        if raw.strip() == "":
            # Blank means "not set", which must fall back to the default.
            assert load_audio_config().max_duration_seconds == AudioConfig().max_duration_seconds
        else:
            # Garbage must fail loudly at startup, not silently open a limit.
            with pytest.raises(ValueError, match="VOXSHIELD_MAX_DURATION_SECONDS"):
                load_audio_config()

    @pytest.mark.parametrize("raw", ["0", "-5"])
    def test_non_positive_limit_rejected(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        monkeypatch.setenv("VOXSHIELD_MAX_UPLOAD_BYTES", raw)
        with pytest.raises(ValueError, match="must be positive"):
            load_audio_config()

    @pytest.mark.parametrize("raw", ["12.0", "-500.0", "nan"])
    def test_dbfs_floor_must_be_a_plausible_level(
        self, monkeypatch: pytest.MonkeyPatch, raw: str
    ) -> None:
        # dBFS is signed, so the floor accepts negatives -- but a positive or
        # non-finite floor would make every frame pass, or fail comparison.
        monkeypatch.setenv("VOXSHIELD_VAD_FLOOR_DBFS", raw)
        with pytest.raises(ValueError, match="VOXSHIELD_VAD_FLOOR_DBFS"):
            load_audio_config()

    def test_dsp_parameters_are_not_env_tunable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Feature parameters are part of the model's input contract. Allowing
        # them to drift per deployment would invalidate stored evaluations.
        monkeypatch.setenv("VOXSHIELD_N_MELS", "40")
        monkeypatch.setenv("VOXSHIELD_N_FFT", "512")
        cfg = load_audio_config()
        assert cfg.features.n_mels == 80
        assert cfg.features.n_fft == 400
