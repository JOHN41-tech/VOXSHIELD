"""Optional SpeechBrain ECAPA-TDNN adapter backed only by local model files.

No network-capable model identifier is accepted. Provision the upstream model
into a local directory explicitly before constructing this adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from voxshield.speaker.embeddings.encoder import (
    PREPROCESSING_VERSION,
    normalized_embedding,
)
from voxshield.speaker.exceptions import EmbeddingExtractionError, SpeakerVerificationError
from voxshield.speaker.schemas import Embedding


class SpeechBrainEcapaEncoder:
    """ECAPA-TDNN speaker encoder using user-provisioned local model files."""

    encoder_id = "speechbrain-ecapa-voxceleb"
    encoder_version = "speechbrain/spkrec-ecapa-voxceleb"
    sample_rate = 16_000
    embedding_dimension = 192

    def __init__(self, model_dir: str | Path, *, device: str = "cpu") -> None:
        path = Path(model_dir).expanduser().resolve()
        if not path.is_dir():
            raise SpeakerVerificationError(
                f"local speaker model directory does not exist: {path}; "
                "download/provision the model explicitly first"
            )
        if not any(path.glob("*.yaml")):
            raise SpeakerVerificationError(
                f"local speaker model directory has no YAML hyperparameters: {path}"
            )
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise SpeakerVerificationError(
                "speaker model support is optional; install voxshield[speaker]"
            ) from exc
        try:
            self._model: Any = EncoderClassifier.from_hparams(
                source=str(path), savedir=str(path), run_opts={"device": device}
            )
        except Exception as exc:
            # Keep paths/tracebacks out of the caller-facing error; do not expose
            # model contents or audio through an exception message.
            raise SpeakerVerificationError(
                "could not initialize the local ECAPA-TDNN model; check its files and dependencies"
            ) from exc

    def encode(self, audio: np.ndarray, sample_rate: int) -> Embedding:
        if sample_rate != self.sample_rate:
            raise EmbeddingExtractionError(
                f"ECAPA-TDNN expects {self.sample_rate} Hz audio, got {sample_rate} Hz"
            )
        signal = np.asarray(audio, dtype=np.float32)
        if signal.ndim != 1 or signal.size == 0 or not np.isfinite(signal).all():
            raise EmbeddingExtractionError("encoder input must be finite, non-empty mono audio")
        try:
            import torch

            waveform = torch.from_numpy(np.ascontiguousarray(signal)).unsqueeze(0)
            with torch.inference_mode():
                raw = self._model.encode_batch(waveform)
            vector = raw.detach().cpu().numpy().reshape(-1)
            normal = normalized_embedding(vector, dimension=self.embedding_dimension)
        except EmbeddingExtractionError:
            raise
        except Exception as exc:
            raise EmbeddingExtractionError("ECAPA-TDNN embedding extraction failed") from exc
        return Embedding(
            vector=tuple(float(value) for value in normal),
            encoder_id=self.encoder_id,
            encoder_version=self.encoder_version,
            preprocessing_version=PREPROCESSING_VERSION,
        )
