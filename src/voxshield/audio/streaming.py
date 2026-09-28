"""Chunked streaming over the offline preprocessing contract.

VoxShield's analysis is window-based, which is a property that suits streaming
better than it first appears: a fixed-width window is a self-contained unit, so
a caller can push audio in arbitrary chunks and receive scored windows without
the whole call ever being resident.

What this module is:

* A **rolling buffer** that accepts chunks of any size, resamples them to the
  canonical rate, and emits fixed-width windows at a configured hop.
* A **reuse of the offline path** -- :func:`~voxshield.audio.features.compute_log_mel`
  and the configuration resolve to the same objects the offline pipeline uses, so
  a streamed window and a batch-processed window from the same audio are equal.
* An explicit :meth:`StreamingProcessor.flush` for the tail, so a caller decides
  when a stream is over instead of guessing from silence. The tail is zero-filled
  to full width and flagged :attr:`StreamWindow.is_final`.

The one intentional difference from the batch path: batch segmentation anchors
its last window to end exactly at the clip end, while a stream cannot know where
it ends until :meth:`~StreamingProcessor.flush`. So a streamed tail is a
zero-filled window, and a stream therefore yields at least as many windows as
the batch path for the same audio, with a flagged tail beyond it. Consumers
should drop or down-weight ``is_final`` windows rather than score them as speech.

A second difference is loudness normalization, and it is unavoidable in the same
way. The batch path measures the whole clip's RMS and applies one gain before
windowing. A stream has not heard the end of the clip, so it cannot know that
gain. :class:`StreamingProcessor` therefore applies no gain by default and its
features will not match a normalized batch run. Pass ``gain_db`` -- typically
``ProcessResult.gain_db_applied`` from a prior calibration pass, or 0.0 to match
a batch run with normalization disabled -- to get exact parity.

What this module is not, stated plainly because the distinction is where
streaming designs usually mislead:

* **VAD is not incremental here.** The offline VAD thresholds over whole windows
  of context; a streaming decision made on a partial window can be revised when
  the rest arrives. This module therefore does *not* run VAD. It segments
  unconditionally and lets the offline path's abstention rules apply once a full
  clip exists. Adding a streaming VAD would mean carrying a heuristic whose
  outputs differ from the batch ones, which is worse than not having one.
* **Per-chunk resampling is stateless.** :func:`~voxshield.audio.preprocess.resample_to`
  is applied to each chunk independently, so a resampling filter's edge transient
  can appear at every chunk boundary. Fixed-size chunks and a small amount of
  overlap between them reduce this. A production deployment that cannot tolerate
  boundary artefacts should resample once upstream and feed canonical-rate
  chunks, which this module accepts and is the recommended mode.

Emitted windows hold real audio-derived features. They belong in memory for the
duration of scoring and nowhere else; :meth:`StreamingProcessor.drop_audio` exists
so a caller can release the buffer deliberately. See ``docs/privacy-design.md``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from voxshield.audio.features import compute_log_mel
from voxshield.audio.preprocess import resample_to, sanitize, to_mono
from voxshield.config import AudioConfig, FeatureConfig

__all__ = ["StreamWindow", "StreamingProcessor"]


@dataclass(frozen=True, slots=True)
class StreamWindow:
    """One emitted analysis window.

    Attributes:
        index: Monotonic window counter for this stream.
        start_seconds: Offset from the start of the stream.
        end_seconds: Offset of the window end, always a full window after
            ``start_seconds``.
        features: Log-mel matrix for the window. In memory only.
        is_final: True for a window emitted by :meth:`StreamingProcessor.flush`
            that had to be zero-filled, because the stream ended before a full
            window of audio existed.

    A final window contains silence the speaker never produced. ``push`` always
    drains every complete window, so a ``flush`` window is always zero-filled;
    there is no "final but complete" state to distinguish.
    """

    index: int
    start_seconds: float
    end_seconds: float
    features: np.ndarray
    is_final: bool = False

    @property
    def duration_seconds(self) -> float:
        """Window length in seconds, including any zero fill."""
        return self.end_seconds - self.start_seconds


class StreamingProcessor:
    """Rolling window buffer over a stream of audio chunks.

    Example:
        >>> import numpy as np
        >>> stream = StreamingProcessor(16_000)
        >>> windows = list(stream.push(np.zeros(16_000, dtype=np.float32)))
        >>> windows += list(stream.flush())
        >>> len(windows)
        1
    """

    __slots__ = (
        "_buffer",
        "_buffer_start",
        "_config",
        "_consumed",
        "_feature_config",
        "_gain_scale",
        "_hop_samples",
        "_index",
        "_source_rate",
        "_total_samples",
        "_window_samples",
    )

    def __init__(
        self,
        sample_rate: int,
        config: AudioConfig | None = None,
        *,
        feature_config: FeatureConfig | None = None,
        gain_db: float | None = None,
    ) -> None:
        """Create a processor for one stream.

        Args:
            sample_rate: Rate of the chunks that will be pushed.
            config: Pipeline configuration; supplies the window and hop.
            feature_config: Feature parameters. Defaults to ``config.features``.
            gain_db: Input gain in dB applied to every window, or ``None`` for
                no gain. A stream cannot measure a whole-clip RMS the way the
                batch path does, so this is how a caller reproduces a batch
                result. ``0.0`` matches a batch run with normalization disabled.

        Raises:
            ValueError: If ``sample_rate`` is not positive.
        """
        if sample_rate <= 0:
            msg = f"sample_rate must be positive, got {sample_rate}"
            raise ValueError(msg)

        self._config = config or AudioConfig()
        self._feature_config = feature_config or self._config.features
        self._source_rate = int(sample_rate)
        self._gain_scale = 10.0 ** (gain_db / 20.0) if gain_db else 1.0

        rate = self._config.target_sample_rate
        self._window_samples = round(self._config.segment_seconds * rate)
        self._hop_samples = max(1, round(self._config.segment_hop * rate))

        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0
        self._consumed = 0
        self._index = 0
        self._total_samples = 0

    @property
    def sample_rate(self) -> int:
        """Canonical rate every window is expressed at."""
        return self._config.target_sample_rate

    @property
    def window_samples(self) -> int:
        """Nominal window length in canonical samples."""
        return self._window_samples

    @property
    def received_seconds(self) -> float:
        """Audio accepted so far, in seconds of canonical-rate audio."""
        return self._total_samples / float(self.sample_rate)

    @property
    def buffered_seconds(self) -> float:
        """Audio held in the buffer awaiting a full window."""
        return len(self._buffer) / float(self.sample_rate)

    @property
    def n_emitted(self) -> int:
        """Windows emitted so far."""
        return self._index

    def push(self, chunk: np.ndarray) -> Iterator[StreamWindow]:
        """Accept one chunk and yield every window it completes.

        Args:
            chunk: Samples at the rate given to the constructor. Any length,
                including zero. Shape ``(n,)`` or ``(n, channels)``.

        Yields:
            :class:`StreamWindow` for each newly completed window, in order.

        Raises:
            ValueError: If the chunk has more than two dimensions.
        """
        canonical = self._to_canonical(chunk)
        if canonical.size == 0:
            return

        self._buffer = (
            canonical.copy()
            if self._buffer.size == 0
            else np.concatenate((self._buffer, canonical))
        )
        self._total_samples += int(canonical.size)

        yield from self._drain()

    def flush(self) -> Iterator[StreamWindow]:
        """Emit the zero-filled tail of the stream and mark it final.

        The tail is padded to full width so it produces a feature matrix of the
        same shape as every other window. That padding is real silence, so the
        window is flagged :attr:`StreamWindow.is_final` and a consumer that scores
        it as if it were speech is making a claim about nothing.

        Exactly one window is emitted, if any audio remains. Striding by the hop
        here would manufacture further overlapping windows out of the same short
        remainder -- for a 9s stream, a second window holding 1s of speech and 3s
        of nothing. A caller that wants those overlapping views can restart a
        stream with a smaller hop.

        Yields:
            Zero or one :class:`StreamWindow` with ``is_final=True``.
        """
        rate = float(self.sample_rate)
        offset = self._consumed - self._buffer_start
        available = len(self._buffer) - offset

        if available > 0:
            window = np.zeros(self._window_samples, dtype=np.float32)
            filled = min(available, self._window_samples)
            window[:filled] = self._buffer[offset : offset + filled]

            start_s = self._consumed / rate
            yield StreamWindow(
                index=self._index,
                start_seconds=start_s,
                end_seconds=start_s + self._config.segment_seconds,
                features=compute_log_mel(window, self._feature_config),
                is_final=True,
            )
            self._index += 1
            self._consumed += filled

        self._buffer = np.empty(0, dtype=np.float32)
        # The tail is gone, but the stream position is not: audio arriving after
        # a flush continues the same numbering and offset.
        self._buffer_start = self._consumed

    def reset(self) -> None:
        """Drop all state and start a new stream from zero.

        Releases the buffer, so this is also the way to end a stream early
        without holding its tail in memory.
        """
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start = 0
        self._consumed = 0
        self._index = 0
        self._total_samples = 0

    def drop_audio(self) -> None:
        """Release buffered audio while keeping the stream position.

        Unlike :meth:`reset`, the window counter and total duration are kept, so
        a caller may free memory and continue without renumbering windows.
        """
        self._buffer = np.empty(0, dtype=np.float32)

    def _to_canonical(self, chunk: np.ndarray) -> np.ndarray:
        """Sanitize, downmix, and resample one chunk to the canonical rate."""
        arr = np.asarray(chunk)
        if arr.ndim > 2:
            msg = f"stream chunk must be 1-D or 2-D, got shape {arr.shape}"
            raise ValueError(msg)
        mono = to_mono(sanitize(arr))
        if mono.size == 0:
            return np.empty(0, dtype=np.float32)
        if self._source_rate != self.sample_rate:
            mono = resample_to(mono, self._source_rate, self.sample_rate)
        if self._gain_scale != 1.0:
            mono = mono * self._gain_scale
        return np.ascontiguousarray(mono, dtype=np.float32)

    def _drain(self) -> Iterator[StreamWindow]:
        """Emit every window the buffer can now support.

        ``_consumed`` is the absolute stream offset of the next window to emit,
        while ``_buffer_start`` is the absolute offset of ``buffer[0]``. Keeping
        them separate is what lets the buffer be trimmed for bounded memory
        without losing the window's position in the stream.
        """
        rate = float(self.sample_rate)
        while len(self._buffer) - (self._consumed - self._buffer_start) >= (
            self._window_samples
        ):
            offset = self._consumed - self._buffer_start
            window = self._buffer[offset : offset + self._window_samples]

            start_s = self._consumed / rate
            yield StreamWindow(
                index=self._index,
                start_seconds=start_s,
                end_seconds=start_s + self._config.segment_seconds,
                features=compute_log_mel(window, self._feature_config),
                is_final=False,
            )
            self._index += 1
            self._consumed += self._hop_samples

            # Release consumed audio. This trim is what keeps memory bounded
            # during normal operation; ``drop_audio`` is for the operator.
            consumed_in_buffer = self._consumed - self._buffer_start
            if consumed_in_buffer > 0:
                self._buffer = self._buffer[consumed_in_buffer:]
                self._buffer_start = self._consumed
