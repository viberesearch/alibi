"""FastAPI application factory for Alibi."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from alibi import __version__
from alibi.api.middleware import (
    RateLimitMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
)
from alibi.api.routers import (
    analytics_router,
    annotations_router,
    artifacts_router,
    budgets_router,
    corrections_router,
    distribute_router,
    enrichment_router,
    export_router,
    facts_router,
    health_router,
    identities_router,
    items_router,
    line_items_router,
    nutrition_router,
    predictions_router,
    process_router,
    reports_router,
    search_router,
    users_router,
)

_STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Alibi",
        description="Life Tracker API - headless engine for document and transaction management",
        version=__version__,
        redirect_slashes=False,
    )

    # Global exception handler: prevent internal details leaking
    @app.exception_handler(Exception)
    async def _unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        logging.getLogger(__name__).error(
            "Unhandled error on %s %s",
            request.method,
            request.url.path,
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # Middleware (order matters: first added = outermost)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv(
            "ALIBI_CORS_ORIGINS", "http://localhost:3100,http://127.0.0.1:3100"
        ).split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers
    app.include_router(health_router, tags=["health"])
    app.include_router(process_router, prefix="/api/v1/process", tags=["process"])
    app.include_router(artifacts_router, prefix="/api/v1/artifacts", tags=["artifacts"])
    app.include_router(
        line_items_router, prefix="/api/v1/line-items", tags=["line-items"]
    )
    app.include_router(items_router, prefix="/api/v1/items", tags=["items"])
    app.include_router(reports_router, prefix="/api/v1/reports", tags=["reports"])
    app.include_router(search_router, prefix="/api/v1/search", tags=["search"])
    app.include_router(export_router, prefix="/api/v1/export", tags=["export"])
    app.include_router(facts_router, prefix="/api/v1/facts", tags=["facts"])
    app.include_router(analytics_router, prefix="/api/v1/analytics", tags=["analytics"])
    app.include_router(
        annotations_router, prefix="/api/v1/annotations", tags=["annotations"]
    )
    app.include_router(
        corrections_router, prefix="/api/v1/corrections", tags=["corrections"]
    )
    app.include_router(
        identities_router, prefix="/api/v1/identities", tags=["identities"]
    )
    app.include_router(
        distribute_router, prefix="/api/v1/distribute", tags=["distribute"]
    )
    app.include_router(users_router, prefix="/api/v1/users", tags=["users"])
    app.include_router(nutrition_router, prefix="/api/v1/nutrition", tags=["nutrition"])
    app.include_router(
        predictions_router, prefix="/api/v1/predictions", tags=["predictions"]
    )
    app.include_router(budgets_router, prefix="/api/v1/budgets", tags=["budgets"])
    app.include_router(enrichment_router)

    # Web UI — SPA served from /web/
    @app.get("/web", include_in_schema=False)
    async def web_index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    if _STATIC_DIR.is_dir():
        app.mount(
            "/web",
            StaticFiles(directory=str(_STATIC_DIR), html=True),
            name="web",
        )

    return app


# Module-level app instance for `uvicorn alibi.api.app:app`
app = create_app()
