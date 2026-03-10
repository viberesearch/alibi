"""Correction event log API endpoints."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query

from alibi.api.deps import get_database
from alibi.db.connection import DatabaseManager
from alibi.services import correction_log

router = APIRouter()


@router.get("")
def list_corrections(
    db: Annotated[DatabaseManager, Depends(get_database)],
    entity_type: str | None = Query(None),
    entity_id: str | None = Query(None),
    field: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, Any]]:
    """List correction events with optional filters."""
    return correction_log.list_corrections(
        db, entity_type=entity_type, entity_id=entity_id, field=field, limit=limit
    )


@router.get("/rate/{vendor_key}")
def get_vendor_correction_rate(
    vendor_key: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    window_days: int = Query(90, ge=1, le=365),
) -> dict[str, Any]:
    """Get correction rate for a vendor."""
    return correction_log.get_vendor_correction_rate(db, vendor_key, window_days)


@router.get("/vendor/{vendor_key}")
def get_vendor_corrections(
    vendor_key: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    limit: int = Query(50, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Get corrections for all facts/items belonging to a vendor."""
    return correction_log.get_vendor_corrections(db, vendor_key, limit)
