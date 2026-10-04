"""Offline processing: audio in, one auditable result out.

:mod:`voxshield.audio.pipeline` defines *what valid audio is*. This module runs
that contract end to end and reports what happened, in a shape that is safe to
log, store, and put on a dashboard.

The distinction from :func:`~voxshield.audio.pipeline.prepare` is deliberate.
``prepare`` returns live audio-derived state -- samples, spectrograms -- for a
caller that is about to run inference. ``process_audio`` returns **no audio at
all**: only scalars, counts, and durations. That makes it usable from a CLI, a
batch job, or a health check without a reviewer having to audit whether any
sample array escaped into the result.

Three properties are worth stating because they are the reason this module
exists rather than a three-line wrapper:

* **Real-time factor is measured, not estimated.** RTF here is wall-clock time
  divided by audio duration, measured around the same stages a production call
  would run. An RTF below 1.0 means a call could be analysed faster than it is
  recorded. Numbers that are not measured are not reported.
* **Quality gates scoring without hijacking it.** A clip whose quality report
  carries blocking issues is reported as ``scorable=False`` with the reasons
  attached. This module never converts a quality problem into an exception,
  because "we could not score this" and "this scored low" are different claims
  and a caller that cannot tell them apart will eventually report one as the
  other.
* **Regions and windows are both reported.** Detected speech regions and the
  analysis windows derived from them are separate facts about the same clip, and
  a reviewer debugging a surprising verdict needs both.

The audio is released before returning. See ``docs/privacy-design.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from voxshield.audio.loader import AudioSource, load_audio
from voxshield.audio.pipeline import PreparedAudio, prepare
from voxshield.audio.quality import QualityReport, assess_quality
from voxshield.audio.vad import SpeechRegion, find_speech_regions
from voxshield.config import AudioConfig
from voxshield.monitoring.logging import safe_metadata

__all__ = ["ProcessResult", "main", "process_audio"]


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Scalar-only outcome of one offline processing run.

    Every field is a number, a string, or a list of strings. There is no field
    that can hold a sample array, and :meth:`metadata` passes the whole result
    through :func:`~voxshield.monitoring.logging.safe_metadata` so a future
    field added carelessly is caught rather than trusted.

    ``quality`` is required rather than defaulted. A result with no quality
    report would have to fabricate one, and a fabricated all-zero report reads
    as "clean" rather than "unknown" -- so the type refuses to be built at all.
    """

    quality: QualityReport
    source: dict[str, object] = field(default_factory=dict)
    regions: tuple[SpeechRegion, ...] = ()
    speech_seconds: float = 0.0
    speech_ratio: float = 0.0
    n_segments: int = 0
    n_padded_windows: int = 0
    n_features: int = 0
    feature_shape: tuple[int, int] | None = None
    window_seconds: float = 0.0
    hop_seconds: float = 0.0
    normalization_strategy: str = "rms"
    gain_db_applied: float = 0.0
    audio_seconds: float = 0.0
    wall_seconds: float = 0.0
    rtf: float = 0.0

    @property
    def scorable(self) -> bool:
        """Whether a verdict may be derived from this clip.

        False when the quality report carries a blocking issue. An unscorable
        clip is an abstention, not a low-risk result.
        """
        return self.quality.is_scorable

    @property
    def blocking_issues(self) -> tuple[str, ...]:
        """Stable issue codes that prevent scoring."""
        return self.quality.blocking_issues

    @property
    def warning_issues(self) -> tuple[str, ...]:
        """Stable issue codes worth surfacing that do not block scoring."""
        return self.quality.warnings

    def metadata(self) -> dict[str, object]:
        """Flat, log-safe summary of the run.

        Safe by construction: built from scalars, then filtered through
        :func:`~voxshield.monitoring.logging.safe_metadata` before returning.
        """
        return safe_metadata(
            {
                **self.source,
                "audio_seconds": round(self.audio_seconds, 4),
                "canonical_sample_rate_hz": 16_000,
                "speech_seconds": round(self.speech_seconds, 4),
                "speech_ratio": round(self.speech_ratio, 4),
                "n_regions": len(self.regions),
                "region_bounds": [[round(r.start_s, 3), round(r.end_s, 3)] for r in self.regions],
                "n_segments": self.n_segments,
                "n_padded_windows": self.n_padded_windows,
                "n_features": self.n_features,
                "feature_shape": list(self.feature_shape) if self.feature_shape else None,
                "window_seconds": round(self.window_seconds, 3),
                "hop_seconds": round(self.hop_seconds, 3),
                "normalization_strategy": self.normalization_strategy,
                "gain_db_applied": round(self.gain_db_applied, 2),
                "wall_seconds": round(self.wall_seconds, 4),
                "rtf": round(self.rtf, 6),
                "scorable": self.scorable,
                "quality_issues": list(self.quality.issues),
                "quality_blocking_issues": list(self.blocking_issues),
                "quality_warning_issues": list(self.warning_issues),
            }
        )


def _measure_features(
    prepared: PreparedAudio,
) -> tuple[int, tuple[int, int] | None, int]:
    """Compute every window's features, returning counts and a shape check.

    Features are computed and discarded rather than accumulated. Holding them
    would make peak memory scale with clip length for no benefit here, and the
    useful result -- that every window yields the same shape -- is just as true
    when they are computed one at a time.
    """
    n = 0
    n_padded = 0
    shape: tuple[int, int] | None = None
    consistent = True
    for index, segment in enumerate(prepared.segments):
        if segment.is_padded:
            n_padded += 1
        features = prepared.segment_features(index)
        if shape is None:
            shape = (int(features.shape[0]), int(features.shape[1]))
        elif (int(features.shape[0]), int(features.shape[1])) != shape:
            consistent = False
        n += 1
    if not consistent:
        msg = (
            "analysis windows produced inconsistent feature shapes; the "
            "short-window policy and window length are inconsistent"
        )
        raise ValueError(msg)
    return n, shape, n_padded


def process_audio(
    source: AudioSource,
    sample_rate: int | None = None,
    config: AudioConfig | None = None,
    *,
    compute_features: bool = True,
) -> ProcessResult:
    """Load, analyse, and summarise one clip. Returns no audio.

    Args:
        source: Anything :func:`~voxshield.audio.loader.load_audio` accepts --
            encoded bytes, a path, a stream, or an array with its rate.
        sample_rate: Required for array sources, ignored for encoded ones.
        config: Pipeline configuration. Defaults to :class:`AudioConfig`.
        compute_features: Run feature extraction over every window. Leave on to
            get a real RTF and a verified feature shape; turn off only to time
            the earlier stages in isolation.

    Returns:
        A :class:`ProcessResult` holding scalars, counts, and durations.

    Raises:
        AudioTooLargeError: A size, rate, channel, or duration budget is
            exceeded.
        UnsupportedAudioFormatError: The container or encoding is not
            allow-listed.
        AudioDecodeError: The source is not decodable audio.
        InvalidAudioSignalError: The audio decodes but is unusable.
        InsufficientSpeechError: Too little contiguous speech for a window.
    """
    cfg = config or AudioConfig()
    started = time.perf_counter()

    # prepare() owns decode, preprocess, VAD, and segmentation. Running it first
    # and deriving everything else from its output keeps the expensive stages
    # running exactly once -- calling preprocess separately as well would double
    # the real work and quietly inflate the RTF reported below.
    decoded = load_audio(source, sample_rate, cfg)
    prepared = prepare(b"", cfg, decoded=decoded)

    # Quality is assessed on the *decoded* signal, before DC removal,
    # resampling, and loudness normalisation. That placement is the whole point:
    # normalising is designed to fix level, so a clip that arrived clipped or
    # far too quiet comes out of the gain stage looking fine and every
    # level-based check silently passes. Measuring before the repair reports what
    # the recording chain actually delivered, which is what a reviewer needs.
    quality = assess_quality(decoded.samples, decoded.sample_rate, cfg)

    regions = tuple(
        find_speech_regions(
            prepared.speech,
            prepared.preprocessed.sample_rate,
            len(prepared.preprocessed.samples),
            cfg,
        )
    )

    n_features = 0
    n_padded = 0
    feature_shape: tuple[int, int] | None = None
    if compute_features:
        n_features, feature_shape, n_padded = _measure_features(prepared)
    else:
        n_padded = sum(1 for s in prepared.segments if s.is_padded)

    audio_seconds = prepared.preprocessed.duration_seconds
    wall_seconds = time.perf_counter() - started
    rtf = wall_seconds / audio_seconds if audio_seconds > 0 else 0.0

    result = ProcessResult(
        source=prepared.source_metadata,
        quality=quality,
        regions=regions,
        speech_seconds=prepared.speech.speech_seconds,
        speech_ratio=prepared.speech.speech_ratio,
        n_segments=prepared.n_segments,
        n_padded_windows=n_padded,
        n_features=n_features,
        feature_shape=feature_shape,
        window_seconds=cfg.segment_seconds,
        hop_seconds=cfg.segment_hop,
        normalization_strategy=cfg.normalization_settings.strategy,
        gain_db_applied=prepared.preprocessed.gain_db_applied,
        audio_seconds=audio_seconds,
        wall_seconds=wall_seconds,
        rtf=rtf,
    )

    # Release the audio before the result leaves this function. The result holds
    # no reference to it, so nothing above needs to remember to clean up.
    del prepared, decoded
    return result


def main(argv: list[str] | None = None) -> int:
    """Analyse one file and print a metadata-only report.

    Exists so ``python -m voxshield.audio.process`` is a working entry point. It
    was previously not: the module had no ``__main__`` guard, so the interpreter
    imported it, defined the functions above, and exited 0 without analysing
    anything. A command that prints nothing and reports success is the worst
    possible failure mode for a measurement tool -- it reads as "no problems" and
    gets trusted.

    The work is delegated to :func:`voxshield.cli.main` rather than reimplemented
    here. The 0/1/2/3 exit codes are documented as part of the contract, and a
    second copy of that mapping would be free to drift from the first.

    Args:
        argv: Argument vector, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code, identical to ``voxshield process``.
    """
    import sys

    from voxshield.cli import main as cli_main

    return cli_main(["process", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
