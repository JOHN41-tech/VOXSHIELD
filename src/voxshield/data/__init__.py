"""Phase 2: reproducible, versioned, leakage-resistant dataset construction.

The public surface is deliberately small. A build has a handful of stages, and
each one is a module that can be run and inspected on its own:

* :mod:`~voxshield.data.registry` -- which corpora this project knows about, which
  of them are usable here, and why the others are not.
* :mod:`~voxshield.data.adapters` -- one reader per corpus layout.
* :mod:`~voxshield.data.discovery` -- find files, hash them, deduplicate.
* :mod:`~voxshield.data.validation` -- decide which candidates are usable, with a
  reason for every rejection.
* :mod:`~voxshield.data.splitting` -- assign whole groups to splits, before
  segmentation.
* :mod:`~voxshield.data.preprocess` and :mod:`~voxshield.data.cache` -- Phase 1
  audio processing, cached on content.
* :mod:`~voxshield.data.leakage` and :mod:`~voxshield.data.gates` -- refuse to
  write manifests that would measure nothing.
* :mod:`~voxshield.data.manifest` -- the versioned JSONL outputs.

Import order is lazy for everything except the schema. Importing this package
must stay cheap and side-effect free: it is imported by CLI startup, by tests
that only want a label constant, and by the model code once Phase 3 exists, and
none of those should pay for ``soundfile`` or a dataset scan.
"""

from __future__ import annotations

from voxshield.data.config import DataConfig, DatasetEntry
from voxshield.data.errors import (
    AdapterError,
    DatasetBuildError,
    DatasetConfigError,
    DatasetLeakageError,
    DatasetUnavailableError,
    ManifestError,
    QualityGateError,
    RegistryError,
    SplitError,
)
from voxshield.data.labels import (
    ATTACK_FAMILIES,
    BONA_FIDE,
    LABEL_TO_INDEX,
    LABELS,
    SPOOF,
    UNCLASSIFIED,
    attack_family,
    encode_label,
)
from voxshield.data.paths import DataPaths
from voxshield.data.schema import UNKNOWN, SampleRecord, SourceRecord, namespace_speaker

__all__ = [
    "ATTACK_FAMILIES",
    "BONA_FIDE",
    "LABELS",
    "LABEL_TO_INDEX",
    "SPOOF",
    "UNCLASSIFIED",
    "UNKNOWN",
    "AdapterError",
    "DataConfig",
    "DataPaths",
    "DatasetBuildError",
    "DatasetConfigError",
    "DatasetEntry",
    "DatasetLeakageError",
    "DatasetUnavailableError",
    "ManifestError",
    "QualityGateError",
    "RegistryError",
    "SampleRecord",
    "SourceRecord",
    "SplitError",
    "attack_family",
    "encode_label",
    "namespace_speaker",
]
