"""Corpus adapters: everything that knows a dataset's on-disk conventions.

Each adapter converts one corpus's layout into :class:`SourceRecord` objects and
describes what it found. Everything downstream -- validation, splitting,
manifests, leakage checks -- consumes only the schema, so adding a corpus means
adding one file here and one entry in :mod:`voxshield.data.registry`, not editing
the pipeline.

Import order is deliberately shallow. This package re-exports the concrete
adapter classes for convenience, which means importing any adapter pulls in all
three. That is a deliberate trade for a package this small and a registry that
resolves by name: the alternative, lazy resolution, is already implemented in
:func:`voxshield.data.registry.adapter_class` for the paths where it matters.
"""

from __future__ import annotations

from voxshield.data.adapters.asvspoof import ASVspoofAdapter
from voxshield.data.adapters.base import (
    AUDIO_EXTENSIONS,
    PHASE1_FORMATS,
    AdapterDescription,
    DatasetAdapter,
    container_format_for,
)
from voxshield.data.adapters.real_speech import RealSpeechAdapter
from voxshield.data.adapters.wavefake import WaveFakeAdapter

__all__ = [
    "AUDIO_EXTENSIONS",
    "PHASE1_FORMATS",
    "ASVspoofAdapter",
    "AdapterDescription",
    "DatasetAdapter",
    "RealSpeechAdapter",
    "WaveFakeAdapter",
    "container_format_for",
]
