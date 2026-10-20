"""Validated configuration for deterministic prosody analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ProsodyConfig:
    """Engineering defaults; values are not calibrated risk thresholds."""

    sample_rate_hz: int = 16_000
    frame_ms: float = 40.0
    hop_ms: float = 10.0
    min_duration_seconds: float = 0.5
    min_voiced_seconds: float = 0.25
    min_valid_pitch_ratio: float = 0.10
    silence_dbfs: float = -45.0
    min_pause_seconds: float = 0.15
    pitch_min_hz: float = 50.0
    pitch_max_hz: float = 500.0
    max_clipping_ratio: float = 0.01
    minimum_baseline_samples: int = 3
    minimum_feature_observations: int = 3
    deviation_z_threshold: float = 3.0
    deviation_mad_threshold: float = 3.5
    storage_path: Path = Path("data/prosody_baselines")
    enabled_features: tuple[str, ...] = field(
        default_factory=lambda: (
            "pitch_median_hz",
            "pitch_mean_hz",
            "pitch_std_hz",
            "pitch_range_hz",
            "pitch_frame_variation_hz",
            "voiced_frame_ratio",
            "valid_pitch_frame_ratio",
            "energy_median_rms",
            "energy_std_rms",
            "energy_p10_rms",
            "energy_p90_rms",
            "energy_change_rate",
            "duration_seconds",
            "speech_duration_seconds",
            "voiced_duration_seconds",
            "silence_duration_seconds",
            "speech_activity_ratio",
            "pause_count",
            "median_pause_seconds",
            "pause_std_seconds",
            "speech_segment_count",
            "speech_segment_duration_std_seconds",
            "segment_energy_change_std",
        )
    )

    def __post_init__(self) -> None:
        positive = {
            "sample_rate_hz": self.sample_rate_hz,
            "frame_ms": self.frame_ms,
            "hop_ms": self.hop_ms,
            "min_duration_seconds": self.min_duration_seconds,
            "min_voiced_seconds": self.min_voiced_seconds,
            "pitch_min_hz": self.pitch_min_hz,
            "pitch_max_hz": self.pitch_max_hz,
            "minimum_baseline_samples": self.minimum_baseline_samples,
            "minimum_feature_observations": self.minimum_feature_observations,
            "deviation_z_threshold": self.deviation_z_threshold,
            "deviation_mad_threshold": self.deviation_mad_threshold,
        }
        if any(float(value) <= 0.0 for value in positive.values()):
            raise ValueError("positive prosody configuration values must be > 0")
        if self.hop_ms > self.frame_ms:
            raise ValueError("hop_ms cannot exceed frame_ms")
        if self.pitch_min_hz >= self.pitch_max_hz:
            raise ValueError("pitch_min_hz must be below pitch_max_hz")
        if not 0.0 <= self.min_valid_pitch_ratio <= 1.0:
            raise ValueError("min_valid_pitch_ratio must be between 0 and 1")
        if not 0.0 <= self.max_clipping_ratio <= 1.0:
            raise ValueError("max_clipping_ratio must be between 0 and 1")
        if self.silence_dbfs > 0.0:
            raise ValueError("silence_dbfs must be <= 0")
        if self.min_pause_seconds <= 0.0:
            raise ValueError("min_pause_seconds must be > 0")
        object.__setattr__(self, "storage_path", Path(self.storage_path))
        object.__setattr__(self, "enabled_features", tuple(dict.fromkeys(self.enabled_features)))

    @property
    def extraction_fingerprint(self) -> str:
        """Stable human-readable identity for compatibility checks."""
        return (
            f"sr={self.sample_rate_hz};frame={self.frame_ms:g};hop={self.hop_ms:g};"
            f"silence={self.silence_dbfs:g};pitch={self.pitch_min_hz:g}-{self.pitch_max_hz:g}"
        )

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> ProsodyConfig:
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unknown prosody configuration keys: {sorted(unknown)}")
        return cls(**values)


def load_prosody_config(path: str | Path | None = None) -> ProsodyConfig:
    """Load a simple YAML mapping when PyYAML is available, else use defaults."""
    if path is None:
        return ProsodyConfig()
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    try:
        import yaml
    except ImportError:
        return ProsodyConfig()
    loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("prosody configuration root must be a mapping")
    return ProsodyConfig.from_mapping(loaded)
