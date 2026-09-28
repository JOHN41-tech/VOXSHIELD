"""The VoxShield audio preprocessing contract.

One call to :func:`prepare` turns untrusted encoded bytes into the exact
internal state the detector, risk fusion, and policy engine expect, or it fails
with a typed error. Nothing else in the codebase decodes, resamples, or detects
speech regions, so this module is the single definition of what "valid audio"
means.

The stages, in order:

1. **Decode** -- size, container, and rate limits enforced before allocation.
2. **Preprocess** -- 16 kHz mono float32, DC-free, loudness-normalised.
3. **VAD** -- frame-level speech/non-speech decisions.
4. **Segment** -- overlapping fixed-width analysis windows.
5. **Features** -- log-mel spectrogram per window.

Stages 1 through 4 are metadata-safe: their outputs are timings, counts, and
normalized arrays. No stage writes to disk. Stage 5 is the last point at which
audio-derived data exists in memory, and the API layer drops those references as
soon as inference returns.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from voxshield.audio.decode import DecodedAudio, decode_audio_bytes, summarise_metadata
from voxshield.audio.features import compute_log_mel
from voxshield.audio.preprocess import PreprocessedAudio, preprocess_decoded
from voxshield.audio.segmentation import Segment, segment_speech
from voxshield.audio.vad import SpeechMask, detect_speech
from voxshield.config import AudioConfig, FeatureConfig
from voxshield.errors import InsufficientSpeechError

__all__ = ["PreparedAudio", "allowed_upload_formats", "prepare", "prepare_from_bytes"]


@dataclass(slots=True)
class PreparedAudio:
    """Everything downstream stages need, and nothing they do not.

    Attributes:
        preprocessed: Canonical 16 kHz mono audio.
        speech: Frame-level VAD result.
        segments: Analysis windows over speech regions.
        feature_config: Feature parameters aligned with the canonical rate.
        source_metadata: Metadata-only description of the source container.
        window_samples: Nominal window length in canonical samples. Segments that
            run past the end of the audio are zero-filled to exactly this many
            samples, so every window yields a feature matrix of identical shape
            regardless of the short-window policy in force.
    """

    preprocessed: PreprocessedAudio
    speech: SpeechMask
    segments: list[Segment]
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    source_metadata: dict[str, object] = field(default_factory=dict)
    window_samples: int = 0

    @property
    def n_segments(self) -> int:
        """Number of scorable windows."""
        return len(self.segments)

    @property
    def speech_seconds(self) -> float:
        """Total detected speech, in seconds."""
        return self.speech.speech_seconds

    def segment_window(self, index: int) -> np.ndarray:
        """Canonical samples for segment ``index``, zero-filled to full width.

        A window that runs off the end of the audio is the normal case at the
        tail of every clip, and it is the entire point of the ``pad`` policy. The
        fill is zeros rather than a repeat of the last sample: repeating would
        fabricate a tone the speaker never produced, and a fabricated low-level
        hum is exactly what the clipping and SNR checks would then flag.

        Args:
            index: Zero-based segment index.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        if not 0 <= index < len(self.segments):
            msg = f"segment index {index} out of range (n_segments={len(self.segments)})"
            raise IndexError(msg)

        seg = self.segments[index]
        samples = self.preprocessed.samples
        start = min(max(0, seg.start_sample), samples.size)
        end = min(max(start, seg.end_sample), samples.size)
        window = samples[start:end]

        target = self.window_samples or int(window.size)
        if window.size >= target:
            return np.ascontiguousarray(window, dtype=np.float32)

        padded = np.zeros(target, dtype=np.float32)
        padded[: window.size] = window
        return padded

    def segment_features(self, index: int) -> np.ndarray:
        """Log-mel spectrogram for segment ``index``.

        Args:
            index: Zero-based segment index.

        Returns:
            Float32 array of shape ``(n_frames, n_mels)``. Constant across
            segments, including tail windows that were zero-filled.

        Raises:
            IndexError: If ``index`` is out of range.
        """
        window = self.segment_window(index)
        return compute_log_mel(window, self.feature_config)

    def iter_segment_features(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(index, features)`` for every segment, in order."""
        for index in range(len(self.segments)):
            yield index, self.segment_features(index)

    def metadata(self) -> dict[str, object]:
        """Metadata-only summary safe for logs, audit store, and dashboard.

        Contains timings, counts, and signal statistics. Contains no audio
        samples, no transcript, and no identity-bearing field. This is the only
        representation of a analysed call that may leave the process.
        """
        return {
            **self.source_metadata,
            "canonical_sample_rate_hz": self.preprocessed.sample_rate,
            "speech_seconds": round(self.speech.speech_seconds, 4),
            "speech_ratio": round(self.speech.speech_ratio, 4),
            "n_segments": self.n_segments,
            "vad_threshold_dbfs": round(self.speech.threshold_dbfs, 2),
            "gain_db_applied": round(self.preprocessed.gain_db_applied, 2),
            "clipped": self.preprocessed.clipped,
        }

    def drop_audio_references(self) -> None:
        """Release in-memory audio buffers held by this object.

        Called once inference completes so the decoded and normalised arrays
        become collectable promptly rather than lingering for the request's
        full lifetime or, worse, until an exception traceback pins them.
        """
        self.preprocessed = _EMPTY_PREPROCESSED
        self.segments = []


_EMPTY_PREPROCESSED = PreprocessedAudio(
    samples=np.empty(0, dtype=np.float32),
    sample_rate=16_000,
    gain_db_applied=0.0,
    peak_before_normalize=0.0,
    clipped=False,
)


def prepare(
    payload: bytes | bytearray,
    config: AudioConfig | None = None,
    *,
    decoded: DecodedAudio | None = None,
) -> PreparedAudio:
    """Run the full preprocessing contract on uploaded audio.

    Args:
        payload: Encoded audio bytes from an untrusted caller.
        config: Pipeline configuration.
        decoded: Pre-decoded audio, to skip the decode stage. Used by tests and
            by callers that already hold validated samples.

    Returns:
        A :class:`PreparedAudio`.

    Raises:
        AudioTooLargeError: Upload exceeds a defensive limit.
        UnsupportedAudioFormatError: Container or encoding not allow-listed.
        AudioDecodeError: Bytes are not decodable audio.
        InvalidAudioSignalError: Signal decodes but is unusable.
        InsufficientSpeechError: Less contiguous speech than required for a
            verdict. This is an abstention, not a risk judgement, and callers
            must surface it as ``INSUFFICIENT_SPEECH`` rather than mapping it to
            either a low or a high risk score.
    """
    cfg = config or AudioConfig()

    if decoded is None:
        decoded = decode_audio_bytes(payload, cfg)

    preprocessed = preprocess_decoded(decoded, cfg)
    speech = detect_speech(preprocessed.samples, preprocessed.sample_rate, cfg)

    if speech.speech_seconds < cfg.min_speech_seconds:
        # Drop the reference before raising: the caller gets a clean failure and
        # the buffer becomes collectable immediately.
        del preprocessed
        raise InsufficientSpeechError(
            speech_seconds=speech.speech_seconds,
            minimum_seconds=cfg.min_speech_seconds,
        )

    segments = segment_speech(
        n_samples=len(preprocessed.samples),
        speech_mask=speech,
        sample_rate=preprocessed.sample_rate,
        config=cfg,
    )

    if not segments:
        # No window survived. Different causes reach this point and the caller
        # acts on each differently, so the attribution matters. Reporting them
        # all as "not enough speech" produced messages that contradicted their
        # own numbers and pointed operators at the wrong fix.
        regions = speech.regions()
        longest_run = max((end - start for start, end in regions), default=0.0)
        canonical_seconds = preprocessed.duration_seconds
        if longest_run < cfg.min_segment_seconds:
            reason = InsufficientSpeechError.REASON_CONTIGUOUS
        elif canonical_seconds < cfg.segment_seconds:
            # Enough contiguous speech, yet no window survived: the recording is
            # smaller than one window and ``drop`` discarded it. Blaming the
            # speech here would tell an operator to ask the customer to speak
            # more, when the recording already holds enough and the fix is a
            # longer clip or a different short-segment policy.
            reason = InsufficientSpeechError.REASON_SHORT_WINDOW
        else:
            # Long enough for a full window, and enough contiguous speech, yet
            # nothing survived. The speech must be spread thinly enough that no
            # window is dense enough.
            reason = InsufficientSpeechError.REASON_CONTIGUOUS
        # Drop the buffer reference before raising: the caller gets a clean
        # failure and the samples become collectable immediately.
        del preprocessed
        raise InsufficientSpeechError(
            speech_seconds=speech.speech_seconds,
            minimum_seconds=cfg.min_segment_seconds,
            reason=reason,
            longest_run_seconds=longest_run,
            window_seconds=cfg.segment_seconds,
        )

    return PreparedAudio(
        preprocessed=preprocessed,
        speech=speech,
        segments=segments,
        feature_config=cfg.features,
        source_metadata=summarise_metadata(decoded),
        window_samples=round(cfg.segment_seconds * preprocessed.sample_rate),
    )


def prepare_from_bytes(
    payload: bytes | bytearray,
    config: AudioConfig | None = None,
) -> PreparedAudio:
    """:func:`prepare` for the standard byte-upload path."""
    return prepare(payload, config)


def allowed_upload_formats(config: AudioConfig | None = None) -> list[str]:
    """Allow-listed container formats, for API documentation and error text."""
    cfg = config or AudioConfig()
    return sorted(cfg.allowed_formats)
