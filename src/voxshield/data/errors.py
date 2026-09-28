"""Typed errors for dataset construction.

These sit alongside the audio errors in :mod:`voxshield.errors` rather than
inside them, because a dataset build fails for a different class of reason than
an inference request does. An ``AudioDecodeError`` means *this file is bad*; a
:class:`DatasetLeakageError` means *the corpus is unusable as configured* and
that distinction has to survive into the operator's log, or the first instinct
will be to go looking for a corrupt file instead of a duplicated speaker.

Every error here is a configuration or integrity failure, and every one of them
is expected to be actionable by a human without reading source. The message is
part of the contract.
"""

from __future__ import annotations

from voxshield.errors import VoxShieldError

__all__ = [
    "AdapterError",
    "DatasetBuildError",
    "DatasetConfigError",
    "DatasetLeakageError",
    "DatasetUnavailableError",
    "ManifestError",
    "QualityGateError",
    "RegistryError",
    "SplitError",
]


class DatasetConfigError(VoxShieldError):
    """A dataset configuration is missing, malformed, or self-contradictory."""


class DatasetUnavailableError(DatasetConfigError):
    """A registered dataset's files are not present on this machine.

    Distinct from a configuration error because the configuration may be perfectly
    correct and the operator simply has not placed the corpus yet. The build
    should skip it and say so, not abort on a machine that was never meant to hold
    every dataset.
    """


class RegistryError(DatasetConfigError):
    """The dataset registry is inconsistent -- duplicate or unknown entry."""


class AdapterError(VoxShieldError):
    """A dataset adapter could not interpret a source file.

    Adapters are the only place that knows a dataset's on-disk conventions, so
    this error is the boundary between VoxShield's schema and somebody else's
    directory layout.
    """


class DatasetBuildError(VoxShieldError):
    """A dataset build could not be completed."""


class SplitError(DatasetBuildError):
    """The corpus cannot be split as configured.

    Raised before any manifest is written. A split that cannot be honoured must
    abort the build rather than emit manifests that claim a separation the
    manifest data does not actually have.
    """


class DatasetLeakageError(DatasetBuildError):
    """Forbidden overlap between the training set and an evaluation set.

    The build refuses to write manifests when this is raised. Overlapping
    speakers or files across a train/test boundary produce a headline number that
    looks fine while measuring nothing, which is worse than having no number.
    """


class QualityGateError(DatasetBuildError):
    """A mandatory quality gate did not pass."""


class ManifestError(VoxShieldError):
    """A manifest is missing, malformed, or would be silently overwritten."""
