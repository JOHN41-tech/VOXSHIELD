"""Speaker-sample quality gates built on VoxShield's existing audio pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from voxshield.audio.loader import AudioSource, load_audio
from voxshield.audio.preprocess import preprocess_decoded
from voxshield.audio.quality import assess_quality
from voxshield.audio.vad import detect_speech
from voxshield.config import AudioConfig
from voxshield.errors import VoxShieldError
from voxshield.speaker.schemas import SampleQuality

SPEAKER_QUALITY_REASONS = frozenset(
    {
        "INSUFFICIENT_SPEECH",
        "EXCESSIVE_SILENCE",
        "INVALID_AUDIO",
        "CLIPPING_DETECTED",
        "LOW_QUALITY",
    }
)


@dataclass(frozen=True, slots=True)
class SpeakerQualityPolicy:
    """Configurable minimum voiced speech and signal-quality rejection gates."""

    minimum_voiced_seconds: float = 2.0
    maximum_silence_ratio: float = 0.60
    maximum_clipping_ratio: float = 0.01

    def __post_init__(self) -> None:
        if self.minimum_voiced_seconds <= 0:
            raise ValueError("minimum voiced duration must be positive")
        if not 0 <= self.maximum_silence_ratio <= 1:
            raise ValueError("maximum silence ratio must be within [0, 1]")
        if not 0 <= self.maximum_clipping_ratio <= 1:
            raise ValueError("maximum clipping ratio must be within [0, 1]")


@dataclass(slots=True)
class AssessedSample:
    """Transient canonical audio and metadata; never persist or log the waveform."""

    samples: np.ndarray
    sample_rate: int
    quality: SampleQuality


def assess_sample(
    source: AudioSource,
    *,
    sample_rate: int | None = None,
    config: AudioConfig | None = None,
    quality_policy: SpeakerQualityPolicy | None = None,
) -> AssessedSample:
    """Validate an input with shared loaders and return transient canonical audio."""
    cfg = config or AudioConfig()
    policy = quality_policy or SpeakerQualityPolicy()
    started = time.perf_counter()
    try:
        decoded = load_audio(source, sample_rate=sample_rate, config=cfg)
        raw_quality = assess_quality(decoded.samples, decoded.sample_rate, cfg)
        preprocessed = preprocess_decoded(decoded, cfg)
        speech = detect_speech(preprocessed.samples, preprocessed.sample_rate, cfg)
    except (VoxShieldError, ValueError, TypeError) as exc:
        del exc
        empty = np.empty(0, dtype=np.float32)
        quality = SampleQuality(
            acceptable=False,
            rejection_reasons=("INVALID_AUDIO",),
            duration_seconds=0.0,
            voiced_seconds=0.0,
            voiced_ratio=0.0,
            silence_ratio=1.0,
            clipping_ratio=0.0,
            rms_dbfs=-120.0,
            snr_db=None,
            preprocessing_ms=(time.perf_counter() - started) * 1000.0,
        )
        return AssessedSample(empty, cfg.target_sample_rate, quality)

    reasons: list[str] = []
    duration = float(decoded.duration_seconds)
    voiced_seconds = float(speech.speech_seconds)
    voiced_ratio = voiced_seconds / duration if duration > 0 else 0.0
    if voiced_seconds < policy.minimum_voiced_seconds:
        reasons.append("INSUFFICIENT_SPEECH")
    if raw_quality.silence_ratio > policy.maximum_silence_ratio:
        reasons.append("EXCESSIVE_SILENCE")
    if raw_quality.clipping_ratio > policy.maximum_clipping_ratio:
        reasons.append("CLIPPING_DETECTED")
    if not raw_quality.is_scorable:
        reasons.append("LOW_QUALITY")
    report = SampleQuality(
        acceptable=not reasons,
        rejection_reasons=tuple(dict.fromkeys(reasons)),
        duration_seconds=duration,
        voiced_seconds=voiced_seconds,
        voiced_ratio=voiced_ratio,
        silence_ratio=raw_quality.silence_ratio,
        clipping_ratio=raw_quality.clipping_ratio,
        rms_dbfs=raw_quality.rms_dbfs,
        snr_db=raw_quality.snr_db,
        preprocessing_ms=(time.perf_counter() - started) * 1000.0,
    )
    return AssessedSample(preprocessed.samples, preprocessed.sample_rate, report)
