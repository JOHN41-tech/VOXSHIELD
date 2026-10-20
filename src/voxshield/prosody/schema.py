"""Versioned, JSON-safe schemas for prosodic measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

FEATURE_SCHEMA_VERSION = "1.0"


class QualityStatus(StrEnum):
    PASS = "PASS"  # noqa: S105 - status label, not a credential
    WARN = "WARN"
    INCONCLUSIVE = "INCONCLUSIVE"


class AnalysisStatus(StrEnum):
    COMPLETED = "COMPLETED"
    INCONCLUSIVE = "INCONCLUSIVE"
    FAILED = "FAILED"


class BaselineComparisonStatus(StrEnum):
    NOT_REQUESTED = "NOT_REQUESTED"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    AVAILABLE = "AVAILABLE"
    INCONCLUSIVE = "INCONCLUSIVE"


def _finite(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("feature values must be finite or None")
    return value


@dataclass(frozen=True, slots=True)
class FeatureValue:
    """One measurement with explicit availability and units."""

    value: float | int | None
    unit: str
    valid: bool = True
    reason: str | None = None
    quality_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite(self.value))
        if self.valid and self.value is None:
            raise ValueError("a valid feature must have a value")
        if not self.valid and self.value is not None:
            raise ValueError("an unavailable feature must have value=None")
        if not self.unit.strip():
            raise ValueError("feature units must not be empty")
        if not self.valid and not (self.reason and self.reason.strip()):
            raise ValueError("an unavailable feature must explain why")

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "valid": self.valid,
            "reason": self.reason,
            "quality_notes": list(self.quality_notes),
        }


@dataclass(frozen=True, slots=True)
class ProsodyFeatures:
    """Collection of named measurements plus extraction metadata."""

    values: dict[str, FeatureValue] = field(default_factory=dict)
    schema_version: str = FEATURE_SCHEMA_VERSION
    sample_rate_hz: int = 16_000
    duration_seconds: float = 0.0
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(f"unsupported feature schema: {self.schema_version}")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        _finite(self.duration_seconds)
        if self.duration_seconds < 0.0:
            raise ValueError("duration_seconds cannot be negative")
        object.__setattr__(self, "values", dict(self.values))

    @property
    def valid_values(self) -> dict[str, float | int]:
        return {
            name: item.value
            for name, item in self.values.items()
            if item.valid and item.value is not None
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_schema_version": self.schema_version,
            "sample_rate_hz": self.sample_rate_hz,
            "duration_seconds": self.duration_seconds,
            "features": {name: item.as_dict() for name, item in sorted(self.values.items())},
            "warnings": list(self.warnings),
        }
