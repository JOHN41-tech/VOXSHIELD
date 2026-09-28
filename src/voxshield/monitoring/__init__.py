"""Observability that cannot leak audio.

The privacy boundary is enforced by the logging layer's inability to accept a
payload, not by remembering to be careful at each call site. See
:mod:`voxshield.monitoring.logging`.
"""

from voxshield.monitoring.logging import (
    configure_logging,
    get_logger,
    safe_metadata,
    scrub_mapping,
)

__all__ = [
    "configure_logging",
    "get_logger",
    "safe_metadata",
    "scrub_mapping",
]
