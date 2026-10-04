"""Runtime configuration for the VoxShield audio preprocessing contract.

Every limit that protects the service from untrusted input lives here rather
than being scattered through the pipeline as magic numbers. Values can be
overridden from the environment for deployment tuning, but every limit has a
conservative default so that a missing or malformed environment variable fails
safe rather than silently opening the service up.

Preprocessing is pinned to 16 kHz mono PCM because that is the convention of
the ASVspoof 2021 DF corpus, which is our primary evaluation benchmark
(https://arxiv.org/html/2109.00535v1). Matching the benchmark's sample rate
removes one avoidable confound from every measurement we make.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal

__all__ = [
    "AudioConfig",
    "FeatureConfig",
    "NormalizationConfig",
    "QualityConfig",
    "VadConfig",
    "load_audio_config",
]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        msg = f"{name} must be a number, got {raw!r}"
        raise ValueError(msg) from exc
    if value <= 0:
        msg = f"{name} must be positive, got {value!r}"
        raise ValueError(msg)
    return value


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        msg = f"{name} must be an integer, got {raw!r}"
        raise ValueError(msg) from exc
    if value <= 0:
        msg = f"{name} must be positive, got {value!r}"
        raise ValueError(msg)
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# A dBFS floor is negative by definition, so it cannot go through _env_float,
# which demands positivity. The bound is the representable range of a useful
# floor: below -120 dBFS nothing is measurable, above 0 dBFS nothing passes.
_DBFS_RANGE = (-120.0, 0.0)


def _env_dbfs(name: str, default: float) -> float:
    """Read a dBFS value, which is a signed quantity bounded by the unit interval."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        msg = f"{name} must be a number, got {raw!r}"
        raise ValueError(msg) from exc
    low, high = _DBFS_RANGE
    if not low <= value <= high:
        msg = f"{name} must be between {low} and {high} dBFS, got {value!r}"
        raise ValueError(msg)
    return value


NormalizationStrategy = Literal["rms", "peak", "none"]

NORMALIZATION_STRATEGIES: frozenset[str] = frozenset({"rms", "peak", "none"})

#: Smallest speech-to-noise ratio that is distinguishable from no measurement at
#: all. Below this, the estimated floor sits inside the frame-to-frame spread of
#: the speech itself, so the ratio is undefined rather than small.
MIN_MEASURABLE_SNR_DB = 3.0

# What to do with a window shorter than ``min_segment_seconds``. ``drop`` is the
# default because padding a mostly-silent window feeds the detector exactly the
# input it is least reliable on; ``pad`` exists for callers that would rather
# over-generate windows than lose a tail utterance.
SHORT_SEGMENT_POLICIES: frozenset[str] = frozenset({"drop", "pad", "keep"})


@dataclass(frozen=True, slots=True)
class VadConfig:
    """Frame-based voice activity detection parameters.

    The MVP detector is a deterministic **energy + zero-crossing + spectral
    flatness** frame VAD, not a neural VAD. That is a conscious Phase 0 choice:
    it is reproducible, has no model download, and its failure modes are
    inspectable. Silero-VAD is a Phase 1 upgrade once the detector is real --
    see ``docs/decision-log/ADR-002-vad-baseline.md``.

    Thresholds are expressed relative to the *signal's own* frame-energy
    distribution rather than as absolute dBFS. That makes the VAD invariant to
    recording gain, which matters because callers submit audio from wildly
    different devices and the pipeline normalises loudness upstream.
    """

    frame_ms: float = 30.0
    hop_ms: float = 10.0

    # Absolute floor, in dBFS. Frames quieter than this are never speech.
    absolute_floor_dbfs: float = -55.0

    # --- Seed arm (hysteresis) --------------------------------------------
    # Frames within seed_margin_db of the seed_percentile energy are
    # confidently speech, provided they also pass the spectral gates.
    seed_percentile: float = 90.0
    seed_margin_db: float = 6.0
    # Quantile within the seed frames that defines the seed's quiet end.
    seed_reference_percentile: float = 20.0

    # --- Spread arm --------------------------------------------------------
    # How far below the seed reference the speech threshold may fall. This is
    # what admits speech quieter than the seed: the tails of syllables, and
    # whole low-volume recordings, without descending into room tone.
    dynamic_range_db: float = 40.0

    # A frame is speech if energy clears the threshold AND zero-crossing rate is
    # below this. Broadband noise and hiss sit well above it.
    max_zero_crossing_rate: float = 0.35

    # Spectral flatness ceiling. White-ish noise and hiss are spectrally flat
    # (flatness near 1.0); voiced speech, which is strongly harmonic and peaky,
    # sits far lower.
    max_spectral_flatness: float = 0.60

    # Morphological cleanup, in seconds.
    min_speech_duration_s: float = 0.20
    min_silence_duration_s: float = 0.20

    # Fraction of total frames that must be speech for the file to be considered
    # to contain speech at all.
    min_speech_ratio: float = 0.02

    # --- Region post-processing -------------------------------------------
    # A raw frame run is not yet a usable speech region: the onset of an
    # utterance tends to fall below the speech threshold and the final syllable
    # decays out of it, so padding each run before windowing keeps those edges
    # inside the analysed window. Merging then stops one utterance split by a
    # brief pause from becoming two regions.
    region_padding_s: float = 0.10
    region_merge_gap_s: float = 0.20
    min_region_seconds: float = 0.20


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Log-mel spectrogram parameters.

    Defaults are the conventional anti-spoofing configuration at 16 kHz:
    ``n_fft=400`` (25 ms), ``hop_length=160`` (10 ms), ``n_mels=80``, HTK mel
    scale spanning 20 Hz to 7600 Hz.

    These are code-defined and not environment-tunable. Feature parameters are
    part of the model's input contract, so they must be a function of the code
    version rather than of ambient configuration; a deployment that silently
    changed ``n_mels`` would invalidate every stored evaluation result.
    :func:`voxshield.config.load_audio_config` reads only intake and VAD limits.
    """

    sample_rate: int = 16_000
    n_fft: int = 400  # 25 ms window at 16 kHz
    win_length: int = 400
    hop_length: int = 160  # 10 ms hop
    n_mels: int = 80
    f_min: float = 20.0
    f_max: float = 7_600.0
    log_eps: float = 1e-6
    preemphasis: float = 0.97
    center: bool = True
    per_utterance_cmvn: bool = True


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Loudness and amplitude policy for the canonical representation.

    The three targets on :class:`AudioConfig` (``target_rms_dbfs``,
    ``max_gain_db``, ``peak_ceiling``) remain the defaults. This object exists so
    a caller can supply a *complete, self-consistent* alternative in one field
    instead of overriding three independently and landing on a combination that
    was never validated together -- a peak target above the ceiling, say, which
    would be silently undone on every call.

    Attributes:
        enabled: When false, no gain is applied. The peak ceiling still applies,
            because a signal that already exceeds full scale is a malformed input
            rather than a loud one.
        strategy: ``rms`` targets ``target_rms_dbfs``; ``peak`` targets
            ``target_peak``; ``none`` applies the ceiling only.
        target_rms_dbfs: Loudness target for the ``rms`` strategy.
        target_peak: Amplitude target for the ``peak`` strategy.
        max_gain_db: Hard ceiling on applied gain, in both directions. Lifting a
            near-silent recording by 40 dB would amplify its own noise floor and
            then classify that noise.
        peak_ceiling: Highest amplitude permitted in the output. Reaching it is
            reported as clipping.
        silence_rms: RMS below which the signal counts as digital silence and is
            left alone, so silence is never scaled into audible noise.
    """

    enabled: bool = True
    strategy: NormalizationStrategy = "rms"
    target_rms_dbfs: float = -23.0
    target_peak: float = 0.95
    max_gain_db: float = 20.0
    peak_ceiling: float = 0.99
    silence_rms: float = 1e-7

    def __post_init__(self) -> None:
        if self.strategy not in NORMALIZATION_STRATEGIES:
            msg = (
                "normalization.strategy must be one of "
                f"{sorted(NORMALIZATION_STRATEGIES)}, got {self.strategy!r}"
            )
            raise ValueError(msg)
        if not _DBFS_RANGE[0] <= self.target_rms_dbfs <= _DBFS_RANGE[1]:
            msg = (
                "normalization.target_rms_dbfs must be between "
                f"{_DBFS_RANGE[0]} and {_DBFS_RANGE[1]} dBFS, "
                f"got {self.target_rms_dbfs!r}"
            )
            raise ValueError(msg)
        if not 0.0 < self.target_peak <= 1.0:
            msg = f"normalization.target_peak must be within (0, 1], got {self.target_peak!r}"
            raise ValueError(msg)
        if not 0.0 < self.peak_ceiling <= 1.0:
            msg = f"normalization.peak_ceiling must be within (0, 1], got {self.peak_ceiling!r}"
            raise ValueError(msg)
        if self.target_peak > self.peak_ceiling:
            msg = (
                "normalization.target_peak cannot exceed normalization.peak_ceiling "
                f"({self.target_peak} > {self.peak_ceiling}): the ceiling would undo "
                "the target on every call"
            )
            raise ValueError(msg)
        if self.max_gain_db <= 0.0:
            msg = f"normalization.max_gain_db must be positive, got {self.max_gain_db!r}"
            raise ValueError(msg)
        if self.silence_rms < 0.0:
            msg = f"normalization.silence_rms must be non-negative, got {self.silence_rms!r}"
            raise ValueError(msg)
        if not self.enabled and self.strategy != "none":
            # Permitted, but resolved rather than left contradictory: a disabled
            # policy that still names a strategy reads as if the strategy runs.
            object.__setattr__(self, "strategy", "none")

    @property
    def applies_gain(self) -> bool:
        """Whether this policy will apply a gain at all."""
        return self.enabled and self.strategy != "none"


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """Thresholds for the audio quality assessment.

    Quality assessment is *diagnostic*: it describes how trustworthy a clip is
    so an analyst can discount a verdict, and it refuses audio that cannot be
    scored at all. The thresholds live here rather than in the assessment code
    for the same reason the intake limits do -- they are deployment-tunable
    without touching DSP behaviour, and every one of them has a default that
    fails safe.

    Attributes:
        enabled: When false, :func:`voxshield.audio.quality.assess_quality`
            reports metrics and no issues.
        frame_ms: Analysis frame length for the frame-level ratios.
        silence_dbfs: Frames quieter than this count toward ``silence_ratio``.
        clip_threshold: Samples at or above this amplitude count as clipped.
        low_level_dbfs: Overall level below which the clip is flagged as too
            quiet to be trusted even though it is not silent.
        max_silence_ratio: Silence ratio at or above which the clip is treated as
            unusable rather than merely quiet.
        min_snr_db: Estimated signal-to-noise ratio below which the clip is
            flagged.
        dc_offset_limit: Mean sample value above which a DC offset is flagged.
        max_crest_factor_db: Peak-to-RMS ratio above which impulsive noise or a
            single loud transient is suspected.
    """

    enabled: bool = True
    frame_ms: float = 20.0
    silence_dbfs: float = -60.0
    clip_threshold: float = 0.99
    low_level_dbfs: float = -50.0
    max_silence_ratio: float = 0.95
    min_snr_db: float = 10.0
    dc_offset_limit: float = 0.01
    max_crest_factor_db: float = 30.0

    def __post_init__(self) -> None:
        if self.frame_ms <= 0.0:
            msg = "quality.frame_ms must be positive"
            raise ValueError(msg)
        if not _DBFS_RANGE[0] <= self.silence_dbfs <= _DBFS_RANGE[1]:
            msg = "quality.silence_dbfs must be a plausible dBFS level"
            raise ValueError(msg)
        if not 0.0 < self.clip_threshold <= 1.0:
            msg = "quality.clip_threshold must be within (0, 1]"
            raise ValueError(msg)
        if not _DBFS_RANGE[0] <= self.low_level_dbfs <= _DBFS_RANGE[1]:
            msg = "quality.low_level_dbfs must be a plausible dBFS level"
            raise ValueError(msg)
        if not 0.0 < self.max_silence_ratio <= 1.0:
            msg = "quality.max_silence_ratio must be within (0, 1]"
            raise ValueError(msg)
        if self.silence_dbfs > self.low_level_dbfs:
            # A silence gate above the "too quiet" level would make the second
            # check unreachable: every frame that trips the first trips the
            # second, and the low-level issue could never be reported alone.
            msg = "quality.silence_dbfs must not exceed quality.low_level_dbfs"
            raise ValueError(msg)
        if self.min_snr_db <= MIN_MEASURABLE_SNR_DB:
            # Any threshold at or below the measurability floor would be
            # unsatisfiable by construction: ratios that small are never
            # estimated at all, so the check would never fire and an operator
            # would have no way to tell "configured" from "broken".
            msg = (
                "quality.min_snr_db must exceed "
                f"{MIN_MEASURABLE_SNR_DB} dB, got {self.min_snr_db!r}"
            )
            raise ValueError(msg)
        if self.dc_offset_limit < 0.0:
            msg = "quality.dc_offset_limit must be non-negative"
            raise ValueError(msg)
        if self.max_crest_factor_db <= 0.0:
            msg = "quality.max_crest_factor_db must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Limits and targets for the audio preprocessing contract.

    All of these are *defensive limits*. They exist to bound the work an
    untrusted caller can request, not to tune quality.
    """

    # --- Intake limits -----------------------------------------------------
    max_upload_bytes: int = 10 * 1024 * 1024  # 10 MiB
    max_duration_seconds: float = 30.0
    max_channels: int = 8
    # Bounds frames-per-second so that a legal duration at a pathological
    # sample rate cannot still demand a multi-hundred-megabyte allocation.
    max_sample_rate: int = 192_000

    # --- Canonical internal format ----------------------------------------
    target_sample_rate: int = 16_000

    # --- Normalisation targets --------------------------------------------
    target_rms_dbfs: float = -23.0
    max_gain_db: float = 20.0
    peak_ceiling: float = 0.99

    # --- Verdict requirements ---------------------------------------------
    min_speech_seconds: float = 1.0

    # --- Segmentation ------------------------------------------------------
    segment_seconds: float = 4.0
    min_segment_seconds: float = 2.0
    max_segment_seconds: float = 4.0
    segment_merge_gap_s: float = 0.30

    # ``None`` means "half the window", which is the 50% overlap the segmenter
    # has always used. It stays derived rather than pinned to a literal, so
    # changing ``segment_seconds`` cannot silently change the overlap ratio.
    segment_hop_seconds: float | None = None
    short_segment_policy: str = "drop"

    # --- Allowed containers (allow-list, never a deny-list) ----------------
    allowed_subtypes: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "PCM_16",
                "PCM_24",
                "PCM_32",
                "FLOAT",
                "DOUBLE",
            }
        )
    )
    allowed_formats: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "WAV",
                "FLAC",
            }
        )
    )

    normalization: NormalizationConfig | None = None
    quality: QualityConfig = field(default_factory=QualityConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    @property
    def normalization_settings(self) -> NormalizationConfig:
        """Effective normalization policy, resolved from whichever field was set.

        Resolution happens here and nowhere else, so the normalization stage never
        has to know whether a caller used the top-level targets or supplied a
        complete :class:`NormalizationConfig`.
        """
        if self.normalization is not None:
            return self.normalization
        return NormalizationConfig(
            target_rms_dbfs=self.target_rms_dbfs,
            max_gain_db=self.max_gain_db,
            peak_ceiling=self.peak_ceiling,
        )

    @property
    def segment_hop(self) -> float:
        """Window advance in seconds. Defaults to 50% overlap."""
        if self.segment_hop_seconds is None:
            return self.segment_seconds / 2.0
        return self.segment_hop_seconds

    def __post_init__(self) -> None:
        if self.max_duration_seconds <= 0:
            msg = "max_duration_seconds must be positive"
            raise ValueError(msg)
        if self.target_sample_rate <= 0:
            msg = "target_sample_rate must be positive"
            raise ValueError(msg)
        if self.max_sample_rate < self.target_sample_rate:
            msg = "max_sample_rate must be >= target_sample_rate"
            raise ValueError(msg)
        if self.min_speech_seconds <= 0:
            msg = "min_speech_seconds must be positive"
            raise ValueError(msg)
        if not 0.0 <= self.vad.seed_percentile <= 100.0:
            msg = "vad.seed_percentile must be within [0, 100]"
            raise ValueError(msg)
        if not 0.0 <= self.vad.seed_reference_percentile <= 100.0:
            msg = "vad.seed_reference_percentile must be within [0, 100]"
            raise ValueError(msg)
        if self.vad.dynamic_range_db <= 0.0:
            msg = "vad.dynamic_range_db must be positive"
            raise ValueError(msg)
        if self.vad.region_padding_s < 0.0:
            msg = "vad.region_padding_s must be non-negative"
            raise ValueError(msg)
        if self.vad.region_merge_gap_s < 0.0:
            msg = "vad.region_merge_gap_s must be non-negative"
            raise ValueError(msg)
        if self.vad.min_region_seconds < 0.0:
            msg = "vad.min_region_seconds must be non-negative"
            raise ValueError(msg)
        if self.short_segment_policy not in SHORT_SEGMENT_POLICIES:
            msg = (
                "short_segment_policy must be one of "
                f"{sorted(SHORT_SEGMENT_POLICIES)}, got {self.short_segment_policy!r}"
            )
            raise ValueError(msg)
        hop = self.segment_hop_seconds
        if hop is not None and not 0.0 < hop <= self.segment_seconds:
            msg = "segment_hop_seconds must be within (0, segment_seconds]"
            raise ValueError(msg)
        if self.min_segment_seconds <= 0:
            msg = "min_segment_seconds must be positive"
            raise ValueError(msg)
        if self.min_segment_seconds > self.max_segment_seconds:
            msg = "min_segment_seconds cannot exceed max_segment_seconds"
            raise ValueError(msg)
        if self.max_segment_seconds > self.segment_seconds:
            msg = "max_segment_seconds cannot exceed segment_seconds"
            raise ValueError(msg)
        if self.features.sample_rate != self.target_sample_rate:
            msg = (
                "features.sample_rate must match target_sample_rate "
                f"({self.features.sample_rate} != {self.target_sample_rate})"
            )
            raise ValueError(msg)

    def with_overrides(self, **kwargs: object) -> AudioConfig:
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)  # type: ignore[arg-type]


def _env_optional_float(name: str, default: float | None) -> float | None:
    """Read an optional positive float, keeping unset distinct from set."""
    if os.environ.get(name, "").strip() == "":
        return default
    return _env_float(name, default if default is not None else 0.0)


def _env_peak(name: str, default: float) -> float:
    """Read a linear amplitude in ``(0, 1]``."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        msg = f"{name} must be a number, got {raw!r}"
        raise ValueError(msg) from exc
    if not 0.0 < value <= 1.0:
        msg = f"{name} must be within (0, 1], got {value!r}"
        raise ValueError(msg)
    return value


def _env_normalization(default: NormalizationConfig) -> NormalizationConfig | None:
    """Read the normalization policy, or ``None`` when nothing is configured.

    ``None`` means "use the code defaults" and is deliberately distinct from "use
    these values", so a deployment that sets no normalization variables produces
    exactly :class:`AudioConfig` -- the preprocessed signal stays a function of
    the code rather than of which variables happen to be set.
    """
    strategy = os.environ.get("VOXSHIELD_NORMALIZATION_STRATEGY", "").strip()
    enabled = os.environ.get("VOXSHIELD_NORMALIZATION_ENABLED", "").strip()
    target_rms = os.environ.get("VOXSHIELD_NORMALIZATION_TARGET_RMS_DBFS", "").strip()
    peak_ceiling = os.environ.get("VOXSHIELD_NORMALIZATION_PEAK_CEILING", "").strip()

    if not any([strategy, enabled, target_rms, peak_ceiling]):
        return None

    if strategy and strategy not in NORMALIZATION_STRATEGIES:
        msg = (
            "VOXSHIELD_NORMALIZATION_STRATEGY must be one of "
            f"{sorted(NORMALIZATION_STRATEGIES)}, got {strategy!r}"
        )
        raise ValueError(msg)

    return replace(
        default,
        enabled=_env_bool("VOXSHIELD_NORMALIZATION_ENABLED", default.enabled),
        strategy=(strategy or default.strategy),  # type: ignore[arg-type]
        target_rms_dbfs=_env_dbfs(
            "VOXSHIELD_NORMALIZATION_TARGET_RMS_DBFS", default.target_rms_dbfs
        ),
        peak_ceiling=_env_peak("VOXSHIELD_NORMALIZATION_PEAK_CEILING", default.peak_ceiling),
    )


def _env_quality(default: QualityConfig) -> QualityConfig:
    """Read the quality-assessment thresholds.

    Quality thresholds are deployment-tunable, so unlike the feature parameters
    they are read from the environment -- and because the defaults are unchanged
    when nothing is set, an unconfigured deployment still equals
    :class:`AudioConfig`.
    """
    return replace(
        default,
        enabled=_env_bool("VOXSHIELD_QUALITY_ENABLED", default.enabled),
        silence_dbfs=_env_dbfs("VOXSHIELD_QUALITY_SILENCE_DBFS", default.silence_dbfs),
        min_snr_db=_env_float("VOXSHIELD_QUALITY_MIN_SNR_DB", default.min_snr_db),
    )


def load_audio_config() -> AudioConfig:
    """Build an :class:`AudioConfig` from the environment.

    Only the fields that are genuinely deployment-specific are read. DSP
    parameters stay in code so that a feature extractor's output is a function
    of the code version, not of ambient environment drift.
    """
    base = AudioConfig()
    vad = replace(
        base.vad,
        absolute_floor_dbfs=_env_dbfs("VOXSHIELD_VAD_FLOOR_DBFS", base.vad.absolute_floor_dbfs),
        seed_margin_db=_env_float("VOXSHIELD_VAD_SEED_MARGIN_DB", base.vad.seed_margin_db),
        dynamic_range_db=_env_float("VOXSHIELD_VAD_DYNAMIC_RANGE_DB", base.vad.dynamic_range_db),
        max_spectral_flatness=_env_float(
            "VOXSHIELD_VAD_MAX_FLATNESS", base.vad.max_spectral_flatness
        ),
    )
    return replace(
        base,
        max_upload_bytes=_env_int("VOXSHIELD_MAX_UPLOAD_BYTES", base.max_upload_bytes),
        max_duration_seconds=_env_float(
            "VOXSHIELD_MAX_DURATION_SECONDS", base.max_duration_seconds
        ),
        min_speech_seconds=_env_float("VOXSHIELD_MIN_SPEECH_SECONDS", base.min_speech_seconds),
        normalization=_env_normalization(base.normalization_settings),
        quality=_env_quality(base.quality),
        segment_hop_seconds=_env_optional_float(
            "VOXSHIELD_SEGMENT_HOP_SECONDS", base.segment_hop_seconds
        ),
        vad=vad,
    )
