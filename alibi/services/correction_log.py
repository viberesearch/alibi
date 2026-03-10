"""Correction event query service.

Provides query functions for the correction event log — the foundation
for adaptive learning feedback loops.
"""

from __future__ import annotations

from typing import Any

from alibi.db.connection import DatabaseManager
from alibi.db import v2_store


def list_corrections(
    db: DatabaseManager,
    entity_type: str | None = None,
    entity_id: str | None = None,
    field: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List correction events with optional filters.

    If entity_type and entity_id given, filters to that entity.
    If field given, filters to that field.
    Otherwise returns all corrections newest first.
    """
    if entity_type and entity_id:
        return v2_store.get_corrections_by_entity(db, entity_type, entity_id, limit)
    if field:
        return v2_store.get_corrections_by_field(db, field, limit)
    rows = db.fetchall(
        "SELECT * FROM correction_events ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    return [dict(row) for row in rows]


def get_vendor_correction_rate(
    db: DatabaseManager,
    vendor_key: str,
    window_days: int = 90,
) -> dict[str, Any]:
    """Get correction rate for a specific vendor."""
    return v2_store.get_correction_rate(db, vendor_key, window_days)


def get_vendor_corrections(
    db: DatabaseManager,
    vendor_key: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Get all corrections for facts/items belonging to a vendor."""
    return v2_store.get_corrections_by_vendor(db, vendor_key, limit)


def get_vendor_unreliable_fields(
    db: DatabaseManager,
    vendor_key: str,
    min_corrections: int = 3,
    window_days: int = 90,
) -> list[str]:
    """Derive fields that are frequently corrected for a vendor.

    Returns field names that have been corrected at least `min_corrections`
    times within the window. Used to feed back into template learning.
    """
    rows = db.fetchall(
        "SELECT ce.field, COUNT(*) AS cnt "
        "FROM correction_events ce "
        "JOIN facts f ON ce.entity_id = f.id OR ce.entity_id IN ("
        "    SELECT fi.id FROM fact_items fi WHERE fi.fact_id = f.id"
        ") "
        "WHERE ce.entity_type IN ('fact', 'fact_item') "
        "AND f.vendor_key = ? "
        "AND ce.created_at >= strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ? || ' days') "
        "GROUP BY ce.field "
        "HAVING COUNT(*) >= ? "
        "ORDER BY cnt DESC",
        (vendor_key, f"-{window_days}", min_corrections),
    )
    return [row["field"] for row in rows]


def should_suggest_reprocessing(
    db: DatabaseManager,
    vendor_key: str,
    threshold: float = 0.3,
    recent_docs: int = 10,
) -> bool:
    """Check if a vendor's correction rate warrants reprocessing.

    Returns True if more than `threshold` fraction of recent docs
    needed corrections.
    """
    rate_info = get_vendor_correction_rate(db, vendor_key)
    if rate_info["total_facts"] < recent_docs:
        return False
    return bool(rate_info["rate"] > threshold)
