"""Privacy-conscious behavioral baseline enrollment and local storage."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from voxshield.prosody.config import ProsodyConfig
from voxshield.prosody.schema import FEATURE_SCHEMA_VERSION, ProsodyFeatures


class BaselineError(Exception):
    """Base error for baseline lifecycle operations."""


class BaselineNotFoundError(BaselineError):
    pass


class BaselineCompatibilityError(BaselineError):
    pass


class BaselineAlreadyExistsError(BaselineError):
    pass


@dataclass(frozen=True, slots=True)
class FeatureDistribution:
    count: int
    mean: float
    std: float
    median: float
    mad: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "mad": self.mad,
        }


@dataclass(frozen=True, slots=True)
class BehavioralBaseline:
    baseline_id: str
    feature_schema_version: str
    extraction_fingerprint: str
    created_at: str
    sample_count: int
    accepted_sample_count: int
    rejected_sample_count: int
    distributions: dict[str, FeatureDistribution]
    limitations: tuple[str, ...] = ()
    active: bool = True
    version: int = 1
    speaker_profile_id: str | None = None

    def __post_init__(self) -> None:
        uuid.UUID(self.baseline_id)
        if self.sample_count < 1 or self.accepted_sample_count < 1:
            raise ValueError("baseline must contain accepted samples")
        if self.accepted_sample_count > self.sample_count:
            raise ValueError("accepted samples cannot exceed total samples")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("unsupported baseline feature schema")

    def compatible_with(self, features: ProsodyFeatures, config: ProsodyConfig) -> bool:
        return (
            self.active
            and self.feature_schema_version == features.schema_version == FEATURE_SCHEMA_VERSION
            and self.extraction_fingerprint == config.extraction_fingerprint
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_id": self.baseline_id,
            "feature_schema_version": self.feature_schema_version,
            "extraction_fingerprint": self.extraction_fingerprint,
            "created_at": self.created_at,
            "sample_count": self.sample_count,
            "accepted_sample_count": self.accepted_sample_count,
            "rejected_sample_count": self.rejected_sample_count,
            "distributions": {name: item.as_dict() for name, item in self.distributions.items()},
            "limitations": list(self.limitations),
            "active": self.active,
            "version": self.version,
            "speaker_profile_id": self.speaker_profile_id,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BehavioralBaseline:
        distributions = {
            name: FeatureDistribution(**distribution)
            for name, distribution in value["distributions"].items()
        }
        return cls(
            baseline_id=value["baseline_id"],
            feature_schema_version=value["feature_schema_version"],
            extraction_fingerprint=value["extraction_fingerprint"],
            created_at=value["created_at"],
            sample_count=int(value["sample_count"]),
            accepted_sample_count=int(value["accepted_sample_count"]),
            rejected_sample_count=int(value["rejected_sample_count"]),
            distributions=distributions,
            limitations=tuple(value.get("limitations", [])),
            active=bool(value.get("active", True)),
            version=int(value.get("version", 1)),
            speaker_profile_id=value.get("speaker_profile_id"),
        )


def _distribution(values: list[float]) -> FeatureDistribution:
    center = float(median(values))
    deviations = [abs(value - center) for value in values]
    return FeatureDistribution(
        count=len(values),
        mean=float(sum(values) / len(values)),
        std=float(
            (sum((value - sum(values) / len(values)) ** 2 for value in values) / len(values))
            ** 0.5
        ),
        median=center,
        mad=float(median(deviations)),
    )


def enroll_baseline(
    samples: list[ProsodyFeatures],
    config: ProsodyConfig | None = None,
    *,
    baseline_id: str | None = None,
    speaker_profile_id: str | None = None,
) -> BehavioralBaseline:
    """Aggregate suitable feature observations; never stores raw audio."""
    cfg = config or ProsodyConfig()
    if len(samples) < cfg.minimum_baseline_samples:
        raise ValueError(f"at least {cfg.minimum_baseline_samples} samples are required")
    observations: dict[str, list[float]] = {}
    accepted = 0
    rejected = 0
    for features in samples:
        if features.schema_version != FEATURE_SCHEMA_VERSION:
            rejected += 1
            continue
        valid = features.valid_values
        if len(valid) < 3:
            rejected += 1
            continue
        accepted += 1
        for name, value in valid.items():
            observations.setdefault(name, []).append(float(value))
    if accepted < cfg.minimum_baseline_samples:
        raise ValueError("too few suitable samples were available for baseline enrollment")
    distributions = {
        name: _distribution(values)
        for name, values in observations.items()
        if len(values) >= cfg.minimum_feature_observations
    }
    if not distributions:
        raise ValueError("no feature has enough observations for a baseline")
    return BehavioralBaseline(
        baseline_id=str(uuid.UUID(baseline_id)) if baseline_id else str(uuid.uuid4()),
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        extraction_fingerprint=cfg.extraction_fingerprint,
        created_at=datetime.now(UTC).isoformat(),
        sample_count=len(samples),
        accepted_sample_count=accepted,
        rejected_sample_count=rejected,
        distributions=distributions,
        limitations=("Behavioral variation is not an identity proof or deception signal.",),
        speaker_profile_id=speaker_profile_id,
    )


class LocalBaselineStore:
    """Atomic JSON store containing aggregate measurements, not raw audio."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(self.root, 0o700)
        except OSError:
            pass
        self._lock = threading.RLock()

    def _path(self, baseline_id: str) -> Path:
        return self.root / f"{uuid.UUID(baseline_id)}.json"

    def create(self, baseline: BehavioralBaseline) -> None:
        path = self._path(baseline.baseline_id)
        with self._lock:
            if path.exists():
                raise BaselineAlreadyExistsError("baseline already exists")
            self._write(path, baseline)

    def _write(self, path: Path, baseline: BehavioralBaseline) -> None:
        payload = json.dumps(baseline.as_dict(), sort_keys=True, separators=(",", ":"))
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.root, delete=False
        ) as handle:
            temporary = Path(handle.name)
            try:
                os.chmod(temporary, 0o600)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                os.replace(temporary, path)
                os.chmod(path, 0o600)
            finally:
                temporary.unlink(missing_ok=True)

    def get(self, baseline_id: str) -> BehavioralBaseline:
        path = self._path(baseline_id)
        with self._lock:
            if not path.is_file():
                raise BaselineNotFoundError("baseline was not found")
            try:
                baseline = BehavioralBaseline.from_dict(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                raise BaselineError("stored baseline is invalid") from exc
        if baseline.baseline_id != str(uuid.UUID(baseline_id)):
            raise BaselineError("baseline storage key does not match record")
        return baseline

    def update(self, baseline: BehavioralBaseline) -> None:
        path = self._path(baseline.baseline_id)
        with self._lock:
            previous = self.get(baseline.baseline_id)
            if baseline.version != previous.version + 1:
                raise BaselineError("baseline update must increment version by one")
            self._write(path, baseline)

    def deactivate(self, baseline_id: str) -> BehavioralBaseline:
        from dataclasses import replace

        current = self.get(baseline_id)
        if not current.active:
            return current
        updated = replace(current, active=False, version=current.version + 1)
        self.update(updated)
        return updated

    def delete(self, baseline_id: str) -> bool:
        path = self._path(baseline_id)
        with self._lock:
            if not path.exists():
                return False
            path.unlink()
            return True
