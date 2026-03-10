"""Health check endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from alibi import __version__
from alibi.api.deps import get_database
from alibi.db.connection import DatabaseManager

router = APIRouter()


@router.get("/health")
async def health(
    db: DatabaseManager = Depends(get_database),
) -> dict[str, Any]:
    """Health check with database status."""
    db_ok = db.is_initialized()
    stats = db.get_stats() if db_ok else {}

    return {
        "status": "ok" if db_ok else "degraded",
        "version": __version__,
        "database": {
            "initialized": db_ok,
            "schema_version": db.get_schema_version() if db_ok else 0,
            **stats,
        },
    }
