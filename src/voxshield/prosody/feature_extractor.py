"""Deterministic, interpretable prosodic feature extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voxshield.audio.quality import QualityReport, assess_quality
from voxshield.audio.vad import SpeechMask, detect_speech, frame_signal
from voxshield.prosody.config import ProsodyConfig
from voxshield.prosody.schema import FeatureValue, ProsodyFeatures


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    features: ProsodyFeatures
    quality: QualityReport
    speech: SpeechMask


def _unavailable(unit: str, reason: str, notes: tuple[str, ...] = ()) -> FeatureValue:
    return FeatureValue(None, unit, valid=False, reason=reason, quality_notes=notes)


def _metric(value: float | int, unit: str, notes: tuple[str, ...] = ()) -> FeatureValue:
    return FeatureValue(value, unit, quality_notes=notes)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(changes[0::2], changes[1::2], strict=True)]


def _pitch_track(frames: np.ndarray, sample_rate: int, cfg: ProsodyConfig) -> np.ndarray:
    """Estimate F0 using normalized autocorrelation and reject weak periodicity."""
    result = np.full(frames.shape[0], np.nan, dtype=np.float64)
    min_lag = max(1, int(sample_rate / cfg.pitch_max_hz))
    max_lag = min(frames.shape[1] - 1, int(sample_rate / cfg.pitch_min_hz))
    if max_lag <= min_lag:
        return result
    window = np.hanning(frames.shape[1])
    for index, frame in enumerate(frames):
        centered = (frame - np.mean(frame)) * window
        energy = float(np.dot(centered, centered))
        if energy <= 1e-10:
            continue
        correlation = np.correlate(centered, centered, mode="full")[frames.shape[1] - 1 :]
        correlation /= max(float(correlation[0]), 1e-12)
        region = correlation[min_lag : max_lag + 1]
        if region.size == 0:
            continue
        lag = min_lag + int(np.argmax(region))
        peak = float(correlation[lag])
        if peak < 0.30:
            continue
        result[index] = sample_rate / lag
    return result


def _frame_energy(frames: np.ndarray) -> np.ndarray:
    if frames.size == 0:
        return np.empty(0, dtype=np.float64)
    return np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))


def extract_prosodic_features(
    samples: np.ndarray,
    sample_rate: int,
    config: ProsodyConfig | None = None,
    *,
    speech: SpeechMask | None = None,
) -> ExtractionResult:
    """Extract measurable features from mono floating-point audio in memory."""
    cfg = config or ProsodyConfig()
    arr = np.asarray(samples, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError("prosody extraction requires a mono 1-D waveform")
    if sample_rate != cfg.sample_rate_hz:
        raise ValueError(
            f"prosody extraction requires {cfg.sample_rate_hz} Hz audio, got {sample_rate}"
        )
    quality = assess_quality(arr, sample_rate)
    if not np.all(np.isfinite(arr)):
        raise ValueError("waveform contains non-finite values")
    speech_mask = speech or detect_speech(arr, sample_rate)
    frame_length = max(2, round(sample_rate * cfg.frame_ms / 1000.0))
    hop_length = max(1, round(sample_rate * cfg.hop_ms / 1000.0))
    frames = frame_signal(arr, frame_length, hop_length)
    energy = _frame_energy(frames)
    speech_flags = speech_mask.is_speech[: energy.size]
    duration = arr.size / sample_rate
    values: dict[str, FeatureValue] = {}
    warnings: list[str] = list(quality.warnings)
    speech_seconds = speech_mask.speech_seconds
    regions = _runs(speech_flags)
    voiced_energy = energy[speech_flags] if energy.size else np.empty(0)

    def add(name: str, value: FeatureValue) -> None:
        if name in cfg.enabled_features:
            values[name] = value

    pitch = _pitch_track(frames, sample_rate, cfg)
    valid_pitch = np.isfinite(pitch) & speech_flags
    pitch_values = pitch[valid_pitch]
    speech_frame_count = max(1, int(np.count_nonzero(speech_flags)))
    valid_pitch_ratio = float(np.count_nonzero(valid_pitch) / speech_frame_count)
    if pitch_values.size:
        add("pitch_median_hz", _metric(float(np.median(pitch_values)), "Hz"))
        add("pitch_mean_hz", _metric(float(np.mean(pitch_values)), "Hz"))
        add("pitch_std_hz", _metric(float(np.std(pitch_values)), "Hz"))
        add("pitch_range_hz", _metric(float(np.ptp(pitch_values)), "Hz"))
        changes = np.abs(np.diff(pitch_values))
        add(
            "pitch_frame_variation_hz",
            _metric(float(np.median(changes)) if changes.size else 0.0, "Hz"),
        )
    else:
        for name in (
            "pitch_median_hz",
            "pitch_mean_hz",
            "pitch_std_hz",
            "pitch_range_hz",
            "pitch_frame_variation_hz",
        ):
            add(name, _unavailable("Hz", "no reliable pitch frames"))
    add(
        "voiced_frame_ratio",
        _metric(float(np.mean(speech_flags)) if speech_flags.size else 0.0, "ratio"),
    )
    add("valid_pitch_frame_ratio", _metric(valid_pitch_ratio, "ratio"))

    if voiced_energy.size:
        add("energy_median_rms", _metric(float(np.median(voiced_energy)), "linear RMS"))
        add("energy_std_rms", _metric(float(np.std(voiced_energy)), "linear RMS"))
        add("energy_p10_rms", _metric(float(np.percentile(voiced_energy, 10)), "linear RMS"))
        add("energy_p90_rms", _metric(float(np.percentile(voiced_energy, 90)), "linear RMS"))
        energy_changes = np.abs(np.diff(voiced_energy))
        add(
            "energy_change_rate",
            _metric(float(np.mean(energy_changes)) if energy_changes.size else 0.0, "RMS/frame"),
        )
    else:
        for name in (
            "energy_median_rms",
            "energy_std_rms",
            "energy_p10_rms",
            "energy_p90_rms",
            "energy_change_rate",
        ):
            add(name, _unavailable("linear RMS", "no detected speech frames"))

    add("duration_seconds", _metric(duration, "seconds"))
    add("speech_duration_seconds", _metric(speech_seconds, "seconds"))
    add(
        "voiced_duration_seconds",
        _metric(float(np.count_nonzero(valid_pitch) * cfg.hop_ms / 1000), "seconds"),
    )
    add("silence_duration_seconds", _metric(max(0.0, duration - speech_seconds), "seconds"))
    add("speech_activity_ratio", _metric(speech_seconds / duration if duration else 0.0, "ratio"))

    pause_lengths = [
        (start - previous_end) * cfg.hop_ms / 1000.0
        for previous_end, (start, _end) in zip(
            [region[1] for region in regions[:-1]], regions[1:], strict=False
        )
        if (start - previous_end) * cfg.hop_ms / 1000.0 >= cfg.min_pause_seconds
    ]
    add("pause_count", _metric(len(pause_lengths), "count"))
    add(
        "median_pause_seconds",
        _metric(float(np.median(pause_lengths)) if pause_lengths else 0.0, "seconds"),
    )
    add(
        "pause_std_seconds",
        _metric(float(np.std(pause_lengths)) if pause_lengths else 0.0, "seconds"),
    )
    segment_lengths = np.asarray([(end - start) * cfg.hop_ms / 1000.0 for start, end in regions])
    add("speech_segment_count", _metric(len(regions), "count"))
    add(
        "speech_segment_duration_std_seconds",
        _metric(float(np.std(segment_lengths)) if segment_lengths.size else 0.0, "seconds"),
    )
    segment_energy = np.asarray(
        [np.mean(energy[start:end]) for start, end in regions if end > start]
    )
    changes = np.diff(segment_energy) if segment_energy.size > 1 else np.empty(0)
    add(
        "segment_energy_change_std",
        _metric(float(np.std(changes)) if changes.size else 0.0, "linear RMS"),
    )

    if duration < cfg.min_duration_seconds:
        warnings.append("SHORT_RECORDING")
    if speech_seconds < cfg.min_voiced_seconds:
        warnings.append("INSUFFICIENT_VOICED_SPEECH")
    if valid_pitch_ratio < cfg.min_valid_pitch_ratio:
        warnings.append("LOW_VALID_PITCH_RATIO")
    return ExtractionResult(
        features=ProsodyFeatures(
            values=values,
            sample_rate_hz=sample_rate,
            duration_seconds=duration,
            warnings=tuple(dict.fromkeys(warnings)),
        ),
        quality=quality,
        speech=speech_mask,
    )
