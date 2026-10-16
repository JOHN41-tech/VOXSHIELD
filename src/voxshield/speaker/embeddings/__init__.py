"""Speaker embedding interfaces and optional pretrained implementations."""

from voxshield.speaker.embeddings.encoder import PREPROCESSING_VERSION, SpeakerEncoder
from voxshield.speaker.embeddings.pretrained_encoder import SpeechBrainEcapaEncoder

__all__ = ["PREPROCESSING_VERSION", "SpeakerEncoder", "SpeechBrainEcapaEncoder"]
