"""FastAPI application factory.

The app owns the three pieces of state the routes depend on: the audio config,
the audit store, and the detector. They are injected here rather than created at
import time so that a test can substitute any of them, and so that a deployment
can swap the in-memory store for a durable one without touching route code.

Phase 0 ships :class:`~voxshield.models.interface.UnavailableDetector`. The
service starts, accepts audio, reports audio quality, and returns a ``null``
score. That is the intended state, not a degraded one.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from voxshield.api.errors import install_exception_handlers
from voxshield.api.routes import router
from voxshield.config import AudioConfig, load_audio_config
from voxshield.models.interface import UnavailableDetector
from voxshield.monitoring import configure_logging, get_logger
from voxshield.policy.evaluator import POLICY_VERSION
from voxshield.storage import AuditStore, InMemoryAuditStore, JsonlAuditStore

__all__ = ["app", "create_app"]

logger = get_logger("app")

AUDIT_STORE_ENV = "VOXSHIELD_AUDIT_STORE_PATH"


def _build_audit_store() -> AuditStore:
    """Choose an audit store.

    An explicit path selects the durable JSONL store. Absent a path, the
    in-memory store is used, which is correct for tests and local development
    and wrong for production, where records would vanish on restart. The
    deployment config must set the path.
    """
    import os

    path = os.environ.get(AUDIT_STORE_ENV, "").strip()
    if path:
        logger.info("audit store: jsonl")
        return JsonlAuditStore(path)
    logger.warning(
        "audit store: in-memory; records will be lost on restart. "
        f"Set {AUDIT_STORE_ENV} for durable audit storage."
    )
    return InMemoryAuditStore()


def create_app(
    *,
    config: AudioConfig | None = None,
    audit_store: AuditStore | None = None,
    detector: object | None = None,
) -> FastAPI:
    """Build the application.

    Args:
        config: Audio configuration. Defaults to the environment-derived one.
        audit_store: Audit store. Defaults to the environment-selected one.
        detector: Detector. Defaults to :class:`UnavailableDetector`.

    Returns:
        A configured FastAPI application.
    """
    configure_logging(logging.INFO)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "voxshield starting",
            extra={
                "policy_version": POLICY_VERSION,
                "model_loaded": application.state.detector.model_version != "unavailable",
            },
        )
        yield
        logger.info("voxshield stopping")

    application = FastAPI(
        title="VoxShield",
        version="0.1.0",
        summary=(
            "Advisory synthetic-speech detection. Reports a risk band and a "
            "recommended next step; never blocks a transaction."
        ),
        lifespan=lifespan,
    )

    # Explicit ``is None`` checks, not ``or``. A store with zero records is
    # falsy because it defines __len__, so ``store or _build_audit_store()``
    # silently discards an injected empty store and writes audit records to a
    # different instance than the caller is inspecting.
    application.state.audio_config = load_audio_config() if config is None else config
    application.state.audit_store = _build_audit_store() if audit_store is None else audit_store
    application.state.detector = UnavailableDetector() if detector is None else detector

    install_exception_handlers(application)
    application.include_router(router)

    @application.get("/health", tags=["ops"], include_in_schema=False)
    def root_health() -> JSONResponse:
        """Unversioned health check for load balancers."""
        return JSONResponse({"status": "ok"})

    return application


app = create_app()
