"""Shared fixtures.

The synthetic-speech generator here is intentionally crude. It produces a
harmonic stack under a slowly modulated envelope, which is enough to exercise
the energy, zero-crossing, and flatness gates in the VAD and to give the
segmenter contiguous speech. It is not a claim about what synthetic speech
sounds like, and no detector is trained or evaluated against it.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator

import numpy as np
import pytest
import soundfile as sf

from voxshield.config import AudioConfig
from voxshield.models.interface import SegmentScore
from voxshield.storage import InMemoryAuditStore

SAMPLE_RATE = 16_000


def harmonic_speech(
    seconds: float = 4.0,
    sample_rate: int = SAMPLE_RATE,
    *,
    base_f0: float = 120.0,
    harmonics: int = 12,
    modulation_hz: float = 3.1,
    peak: float = 0.5,
) -> np.ndarray:
    """Generate a speech-like harmonic signal.

    Args:
        seconds: Duration.
        sample_rate: Sample rate.
        base_f0: Fundamental frequency in Hz.
        harmonics: Number of harmonics in the stack.
        modulation_hz: Envelope modulation rate; controls pause duration.
        peak: Target peak amplitude.

    Returns:
        Float32 mono samples normalised to ``peak``.
    """
    t = np.arange(int(seconds * sample_rate)) / float(sample_rate)
    f0 = base_f0 + 25.0 * np.sin(2.0 * np.pi * 1.7 * t)
    signal = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        signal += np.sin(2.0 * np.pi * f0 * k * t) / k
    envelope = (0.5 + 0.5 * np.sin(2.0 * np.pi * modulation_hz * t)) ** 2
    signal = signal * envelope
    # A quiet broadband component keeps spectral flatness below the noise gate.
    signal += 0.01 * np.sin(2.0 * np.pi * 4000.0 * t)
    signal = signal / np.max(np.abs(signal))
    return (signal * peak).astype(np.float32)


def white_noise(seconds: float = 4.0, sample_rate: int = SAMPLE_RATE, seed: int = 7) -> np.ndarray:
    """Generate white noise, which the VAD should reject as non-speech."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(seconds * sample_rate)).astype(np.float32) * 0.1


def to_wav_bytes(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    subtype: str = "PCM_16",
    fmt: str = "WAV",
) -> bytes:
    """Encode samples to an in-memory container."""
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format=fmt, subtype=subtype)
    return buffer.getvalue()


# Subtype libsndfile accepts per container. OGG is Vorbis-only, so a test that
# writes PCM_16 into it fails in the encoder and never reaches the decoder
# behaviour it was meant to exercise.
_DEFAULT_SUBTYPE = {
    "WAV": "PCM_16",
    "FLAC": "PCM_16",
    "AIFF": "PCM_16",
    "AU": "PCM_16",
    "OGG": "VORBIS",
}


@pytest.fixture
def speech_samples() -> np.ndarray:
    """Four seconds of speech-like audio."""
    return harmonic_speech(4.0)


@pytest.fixture
def speech_wav() -> bytes:
    """Four seconds of speech-like audio as 16 kHz PCM_16 WAV."""
    return to_wav_bytes(harmonic_speech(4.0))


@pytest.fixture
def noise_wav() -> bytes:
    """Four seconds of white noise as 16 kHz PCM_16 WAV."""
    return to_wav_bytes(white_noise(4.0))


@pytest.fixture
def silent_wav() -> bytes:
    """Four seconds of digital silence."""
    return to_wav_bytes(np.zeros(4 * SAMPLE_RATE, dtype=np.float32))


@pytest.fixture
def samples() -> Callable[..., np.ndarray]:
    """Factory for synthetic signal arrays."""
    return harmonic_speech


@pytest.fixture
def wav() -> Callable[..., bytes]:
    """Factory that encodes samples to an in-memory container.

    Tests that need a container libsndfile can write but the config does not
    allow, or an unusual subtype, use this instead of a bespoke encoder.
    """

    def _encode(
        samples: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        subtype: str | None = None,
        fmt: str = "WAV",
    ) -> bytes:
        return to_wav_bytes(
            samples, sample_rate, subtype or _DEFAULT_SUBTYPE.get(fmt, "PCM_16"), fmt
        )

    return _encode


@pytest.fixture
def config() -> AudioConfig:
    """Default configuration."""
    return AudioConfig()


@pytest.fixture
def store() -> InMemoryAuditStore:
    """Empty in-memory audit store."""
    return InMemoryAuditStore()


@pytest.fixture
def client(store: InMemoryAuditStore) -> Iterator:
    """A TestClient wired to an inspectable audit store.

    Uses ``raise_server_exceptions=False`` so that an unhandled exception
    surfaces as a 500 in the test rather than propagating, which is what a real
    client would see.
    """
    from fastapi.testclient import TestClient

    from voxshield.app import create_app

    with TestClient(create_app(audit_store=store), raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def client_for() -> Iterator[Callable[..., object]]:
    """Factory for TestClients with a specific detector and audit store.

    Yields a callable that returns a context-managed client, so a test can inject
    a stub detector and still inspect what was written to the store afterwards.
    """
    from contextlib import ExitStack

    from fastapi.testclient import TestClient

    from voxshield.app import create_app

    with ExitStack() as stack:

        def _make(detector=None, audit_store=None, **kwargs):
            return stack.enter_context(
                TestClient(
                    create_app(detector=detector, audit_store=audit_store, **kwargs),
                    raise_server_exceptions=False,
                )
            )

        yield _make


class StubDetector:
    """Deterministic detector for tests.

    Returns a fixed score for every window, so policy and API behaviour can be
    asserted exactly without a trained model.
    """

    def __init__(self, score: float = 0.9, version: str = "stub-1.0.0") -> None:
        self._score = score
        self._version = version
        self.calls: list[int] = []

    @property
    def model_version(self) -> str:
        """Reported version."""
        return self._version

    def score_segments(self, features: list[np.ndarray]) -> list[SegmentScore]:
        """Return a fixed score per window."""
        self.calls.append(len(features))
        return [
            SegmentScore(index=i, synthetic_probability=self._score, model_version=self._version)
            for i in range(len(features))
        ]


class BrokenDetector:
    """Detector that raises, to exercise the operational-failure path."""

    def __init__(self) -> None:
        self.model_version = "broken-0.0.0"

    def score_segments(self, features: list[np.ndarray]) -> list[SegmentScore]:
        """Always fail."""
        msg = "weights file is corrupt"
        raise RuntimeError(msg)


@pytest.fixture
def stub_detector() -> StubDetector:
    """Detector returning a high synthetic score."""
    return StubDetector(score=0.9)


@pytest.fixture
def detector() -> Callable[..., StubDetector]:
    """Factory for stub detectors with a chosen score."""
    return StubDetector


@pytest.fixture
def broken_detector() -> BrokenDetector:
    """Detector that always raises, for operational-failure paths."""
    return BrokenDetector()
