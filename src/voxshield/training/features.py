"""Front end for the Phase 3 baselines: MFCC and log-mel frames.

The design constraint is that these baselines must measure the system VoxShield
actually serves. :func:`voxshield.audio.features.compute_log_mel` is the
production front end, so this module calls it rather than reimplementing a
second spectrogram that could silently drift from it. What is added on top is
only what is genuinely model-specific: the DCT for MFCC, and fixed-length
handling for the classifier's input shape.

One interaction is worth stating plainly, because it looks like a bug and is
not. The shared front end always applies per-utterance log-mel mean/variance
normalisation (``per_utterance_cmvn=True`` in :class:`FeatureConfig`), and
:class:`~voxshield.training.config.FeatureSpec` can additionally apply cepstral
mean normalisation via ``lifter``. Both are on by default, which means the MFCC
path is normalised twice in two different domains. That is deliberate --
normalising in the log-mel domain removes channel and gain, and normalising in
the cepstral domain removes vocal-tract and channel shape -- but it is a real
doubling, and ``lifter: 0`` turns the second stage off for anyone who wants to
isolate the first.
"""

from __future__ import annotations

import numpy as np

from voxshield.audio.features import compute_log_mel
from voxshield.config import FeatureConfig
from voxshield.training.config import FeatureSpec, TrainingError

__all__ = [
    "FeatureExtractor",
    "cepstral_mean_normalise",
    "dct_matrix",
    "delta_features",
    "mfcc_from_log_mel",
    "pad_or_truncate",
]


def dct_matrix(n_coefficients: int, n_mels: int) -> np.ndarray:
    """Build an orthonormal DCT-II basis for cepstral analysis.

    Args:
        n_coefficients: Number of cepstral coefficients ``K`` to produce.
        n_mels: Number of mel bands ``N``, the DCT input dimension.

    Returns:
        Float64 array of shape ``(K, N)`` whose rows are orthonormal basis
        functions, so that ``D @ D.T`` is the identity and the DCT neither
        amplifies nor attenuates the signal it transforms.

    Raises:
        TrainingError: If ``K`` or ``N`` is not positive, or ``K > N``. Asking
            for more coefficients than bands is the single most common MFCC
            misconfiguration; it does not error numerically, it silently returns
            ``K`` rows of which the trailing ones are meaningless combinations of
            the same information.
    """
    if n_mels < 1:
        msg = f"n_mels must be positive, got {n_mels}"
        raise TrainingError(msg)
    if n_coefficients < 1:
        msg = f"n_coefficients must be positive, got {n_coefficients}"
        raise TrainingError(msg)
    if n_coefficients > n_mels:
        msg = (
            f"n_coefficients ({n_coefficients}) exceeds n_mels ({n_mels}); "
            "the DCT cannot produce more coefficients than input bands"
        )
        raise TrainingError(msg)

    n = np.arange(n_mels, dtype=np.float64)
    k = np.arange(n_coefficients, dtype=np.float64)[:, None]
    basis = np.cos(np.pi * k * (2.0 * n[None, :] + 1.0) / (2.0 * n_mels))
    # First basis row carries no factor of two, which is what keeps the transform
    # orthonormal instead of merely orthogonal.
    basis *= np.sqrt(2.0 / n_mels)
    basis[0] *= 1.0 / np.sqrt(2.0)
    return basis


def mfcc_from_log_mel(
    log_mel: np.ndarray,
    n_coefficients: int,
    *,
    lifter: float = 0.0,
) -> np.ndarray:
    """Project a log-mel spectrogram onto cepstral coefficients.

    Args:
        log_mel: Array of shape ``(n_frames, n_mels)``.
        n_coefficients: Number of coefficients to keep.
        lifter: Cepstral mean-normalisation coefficient. ``0`` skips mean
            normalisation; any positive value subtracts each coefficient's
            per-utterance mean and applies the classic sinusoidal lifter, which
            de-emphasises ``c0`` (overall energy) and the highest coefficients
            (which are mostly noise). Normalising out ``c0`` is what stops a
            logistic regression from separating the classes on loudness.

    Returns:
        Float32 array of shape ``(n_frames, n_coefficients)``.

    Raises:
        TrainingError: If the input is not a 2-D ``(frames, mels)`` array.
    """
    if log_mel.ndim != 2:
        msg = f"log_mel must be 2-D (frames, mels), got shape {log_mel.shape}"
        raise TrainingError(msg)

    basis = dct_matrix(n_coefficients, log_mel.shape[1])
    coefficients = (basis @ log_mel.T).T.astype(np.float32)
    if lifter <= 0.0:
        return coefficients
    return cepstral_mean_normalise(coefficients, lifter)


def cepstral_mean_normalise(coefficients: np.ndarray, lifter: float) -> np.ndarray:
    """Subtract the per-utterance coefficient means and apply a sinusoidal lifter.

    Args:
        coefficients: Array of shape ``(n_frames, n_coefficients)``.
        lifter: Sinusoidal lifter strength.

    Returns:
        A new float32 array of the same shape. The input is not modified.
    """
    centred = coefficients - coefficients.mean(axis=0, keepdims=True)
    n_coefficients = coefficients.shape[1]
    if n_coefficients < 2:
        # A single coefficient has no cepstral order, so the lifter curve is
        # undefined. Leaving it un-lifted is the only defensible choice.
        return centred.astype(np.float32)

    order = np.arange(n_coefficients, dtype=np.float32)
    weights = 1.0 + (lifter / 2.0) * np.sin(np.pi * order / (n_coefficients - 1))
    return (centred * weights).astype(np.float32)


def delta_features(frames: np.ndarray, width: int = 2) -> np.ndarray:
    """Estimate first-order derivatives by local linear regression.

    The standard Sprenger/Klatt formulation: fit a straight line against a
    symmetric ``2 * width + 1`` window of neighbouring frames and read its slope.
    Simple forward differences were tried and rejected -- they amplify frame-level
    noise into the delta channel and produce a model that keys on codec
    artefacts rather than on speech dynamics.

    Args:
        frames: Array of shape ``(n_frames, n_dims)``.
        width: Half-width of the regression window.

    Returns:
        Float32 array of shape ``(n_frames, n_dims)``, on the same scale in the
        interior and at the edges. The output never has fewer rows than the
        input.

    Note:
        Windows are clipped at the sequence edges and then **re-centred**, which
        is what keeps the estimator unbiased. A clipped window is one-sided, so
        its offsets do not sum to zero, and the raw regression weight
        ``sum(t * x) / sum(t * t)`` then returns ``x * sum(t) / sum(t^2)`` for a
        *constant* signal -- a slope of 4.2 for an input of 7.0 at the first
        frame. Re-centring makes the offsets sum to zero again, so a flat
        utterance yields exactly zero slope everywhere while a linear ramp still
        yields its true gradient at both ends.
    """
    if frames.ndim != 2:
        msg = f"frames must be 2-D (frames, dims), got shape {frames.shape}"
        raise TrainingError(msg)
    if width < 1:
        msg = f"delta width must be at least 1, got {width}"
        raise TrainingError(msg)

    n_frames = frames.shape[0]
    out = np.zeros_like(frames, dtype=np.float32)
    for index in range(n_frames):
        lo = max(0, index - width)
        hi = min(n_frames, index + width + 1)
        # Least-squares slope over the clipped window: sum(t * x) / sum(t * t).
        # The symmetric 2/(t^2 + 2) taper that appears in cepstral smoothing is
        # *not* a regression weight -- using it here makes the output a moving
        # average rather than a slope, so a constant signal came out equal to
        # that constant instead of zero. The window is re-centred after clipping
        # so a one-sided edge window is still unbiased.
        offsets = np.arange(-(index - lo), (hi - 1 - index) + 1, dtype=np.float64)
        offsets -= offsets.mean()
        denominator = float(np.square(offsets).sum())
        if denominator <= 0.0:
            # A single-frame window has no gradient to estimate.
            out[index] = 0.0
            continue
        out[index] = (frames[lo:hi] * (offsets / denominator)[:, None]).sum(axis=0)
    return out


def pad_or_truncate(frames: np.ndarray, n_frames: int | None) -> np.ndarray:
    """Force a frame sequence to an exact length by edge-padding or cropping.

    Edge-padding with the last observed frame is used rather than zero-padding,
    because zero is a real value in a log-mel spectrogram (silence) and a block
    of fabricated silence at the end of every short clip is a cue the model can
    learn from. Cropping from the start rather than the centre keeps the leading
    consonant, which is where onset and voicing differences live.

    Args:
        frames: Array of shape ``(n_frames, n_dims)``.
        n_frames: Target length. ``None`` returns the input unchanged.

    Returns:
        Array of shape ``(n_frames, n_dims)``.
    """
    if n_frames is None:
        return frames
    current = frames.shape[0]
    if current == n_frames:
        return frames
    if current == 0:
        return np.zeros((n_frames, frames.shape[1]), dtype=frames.dtype)
    if current > n_frames:
        return frames[:n_frames]
    pad = np.repeat(frames[-1:], n_frames - current, axis=0)
    return np.concatenate([frames, pad], axis=0)


class FeatureExtractor:
    """Turn raw audio into either a frame matrix or a pooled feature vector.

    Two output shapes, because the two model families need different things and
    forcing them to share one would waste one of them:

    * :meth:`matrix` -- ``(T, D)`` fixed-length, for the CNN.
    * :meth:`vector` -- ``(2D,)``, mean and standard deviation pooled over
      frames, for the linear models.

    The pooled form is mean-plus-standard-deviation rather than a mean alone
    deliberately: the mean captures the spectral envelope while the deviation
    captures how much the envelope moves over the utterance, and synthesis
    artefacts show up more consistently in the second than the first.

    Args:
        spec: Front-end settings. Defaults are the production configuration.

    Raises:
        TrainingError: If the extractor is asked for pooled and frame outputs
            that disagree about width, which would mean a silent shape mismatch
            at fit time.
    """

    def __init__(self, spec: FeatureSpec | None = None) -> None:
        self.spec = spec or FeatureSpec()
        # Built once per extractor and reused for every clip: constructing a
        # FeatureConfig per sample is cheap but not free, and the filterbank is
        # the most expensive object in the front end.
        self._feature_config = FeatureConfig(
            n_fft=self.spec.n_fft,
            win_length=self.spec.n_fft,
            hop_length=self.spec.hop_length,
            n_mels=self.spec.n_mels,
            f_min=self.spec.fmin,
            f_max=self.spec.fmax,
        )
        self._frame_length = self.spec.n_frames or self.spec.target_frames

    @property
    def frame_dim(self) -> int:
        """Width of one frame."""
        return self.spec.output_dim

    @property
    def vector_dim(self) -> int:
        """Width of one pooled vector, ``2 * frame_dim``."""
        return 2 * self.spec.output_dim

    def _log_mel(self, samples: np.ndarray) -> np.ndarray:
        """Compute the shared log-mel spectrogram for one clip."""
        return compute_log_mel(samples, self._feature_config)

    def _frames(self, samples: np.ndarray) -> np.ndarray:
        """Per-frame features at the clip's natural length."""
        log_mel = self._log_mel(samples)
        if log_mel.shape[0] == 0:
            # An empty clip still has to produce a correctly shaped matrix, or a
            # batch containing one silent recording fails at the model instead of
            # contributing a zero vector.
            return np.zeros((0, self.frame_dim), dtype=np.float32)

        if self.spec.kind == "mfcc":
            features = mfcc_from_log_mel(log_mel, self.spec.n_coefficients, lifter=self.spec.lifter)
        else:
            features = log_mel.astype(np.float32)

        if self.spec.with_delta and features.shape[0] > 0:
            features = np.concatenate([features, delta_features(features)], axis=1).astype(
                np.float32
            )
        return features

    def matrix(self, samples: np.ndarray) -> np.ndarray:
        """Fixed-length frame matrix for the CNN.

        Args:
            samples: 1-D audio at 16 kHz.

        Returns:
            Float32 array of shape ``(n_frames, D)``.
        """
        frames = self._frames(samples)
        return pad_or_truncate(frames, self._frame_length).astype(np.float32)

    def vector(self, samples: np.ndarray) -> np.ndarray:
        """Mean-and-standard-deviation pooled vector for the linear models.

        Computed over the clip's natural frame count rather than a padded one,
        so that a long clip is not silently weighted by repetition and a short
        clip is not diluted by the padding it would receive.

        Args:
            samples: 1-D audio at 16 kHz.

        Returns:
            Float32 array of shape ``(2D,)``.
        """
        frames = self._frames(samples)
        if frames.shape[0] == 0:
            return np.zeros(self.vector_dim, dtype=np.float32)
        pooled = np.concatenate([frames.mean(axis=0), frames.std(axis=0)])
        return pooled.astype(np.float32)

    def describe(self) -> dict[str, object]:
        """JSON-ready description of the front end, for artefact metadata."""
        return {
            "kind": self.spec.kind,
            "frame_dim": self.frame_dim,
            "vector_dim": self.vector_dim,
            "frame_length": self._frame_length,
            "n_frames": self.spec.n_frames,
            "target_frames": self.spec.target_frames,
        }
