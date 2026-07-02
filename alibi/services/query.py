"""Query service for facts, documents, and items.

Consolidates query patterns from CLI, API, and MCP into a shared
service layer. All consumers should call these functions instead
of hitting v2_store directly.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from alibi.db.connection import DatabaseManager
from alibi.db import v2_store

logger = logging.getLogger(__name__)


def get_fact(db: DatabaseManager, fact_id: str) -> dict[str, Any] | None:
    """Get a fact by ID with its items loaded.

    Args:
        db: Database manager instance.
        fact_id: UUID of the fact to retrieve.

    Returns:
        Dict with fact fields and an 'items' key containing the line
        items, or None if the fact does not exist.
    """
    fact = v2_store.get_fact_by_id(db, fact_id)
    if fact is None:
        return None
    fact["items"] = v2_store.get_fact_items(db, fact_id)
    return fact


def inspect_fact(db: DatabaseManager, fact_id: str) -> dict[str, Any] | None:
    """Full drill-down of a fact.

    Returns the fact, its cloud metadata, all bundles with their atoms
    and source documents, and the fact items.  Delegates entirely to
    v2_store.inspect_fact which already builds this nested structure.

    Args:
        db: Database manager instance.
        fact_id: UUID of the fact to inspect.

    Returns:
        Nested dict with keys 'fact', 'cloud', 'bundles', 'items',
        or None if the fact does not exist.
    """
    return v2_store.inspect_fact(db, fact_id)


def list_facts(
    db: DatabaseManager,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    """List facts with optional filters and server-side pagination.

    Args:
        db: Database manager instance.
        filters: Optional dict of filter criteria. Recognised keys:
            - vendor (str): case-insensitive substring match on vendor name
            - date_from (str | date): ISO date lower bound (inclusive)
            - date_to (str | date): ISO date upper bound (inclusive)
            - min_amount (float | Decimal): minimum total_amount
            - max_amount (float | Decimal): maximum total_amount
            - fact_type (str): exact match on fact_type value
        offset: Number of rows to skip (zero-based).
        limit: Maximum number of rows to return.

    Returns:
        Dict with keys:
            - facts: list of fact dicts
            - total: total matching row count (before pagination)
            - offset: the requested offset
            - limit: the requested limit
    """
    filters = filters or {}

    conditions: list[str] = ["1=1"]
    params: list[Any] = []

    vendor = filters.get("vendor")
    if vendor:
        conditions.append("LOWER(vendor) LIKE ?")
        params.append(f"%{str(vendor).lower()}%")

    date_from = filters.get("date_from")
    if date_from is not None:
        if isinstance(date_from, date):
            date_from = date_from.isoformat()
        conditions.append("event_date >= ?")
        params.append(str(date_from))

    date_to = filters.get("date_to")
    if date_to is not None:
        if isinstance(date_to, date):
            date_to = date_to.isoformat()
        conditions.append("event_date <= ?")
        params.append(str(date_to))

    min_amount = filters.get("min_amount")
    if min_amount is not None:
        conditions.append("CAST(total_amount AS REAL) >= ?")
        params.append(float(min_amount))

    max_amount = filters.get("max_amount")
    if max_amount is not None:
        conditions.append("CAST(total_amount AS REAL) <= ?")
        params.append(float(max_amount))

    fact_type = filters.get("fact_type")
    if fact_type:
        conditions.append("fact_type = ?")
        params.append(str(fact_type))

    where = " AND ".join(conditions)

    # Count total matching rows (no LIMIT/OFFSET)
    count_row = db.fetchone(
        f"SELECT COUNT(*) AS cnt FROM facts WHERE {where}",  # noqa: S608
        tuple(params),
    )
    total: int = count_row["cnt"] if count_row else 0

    # Fetch paginated rows, including a per-fact PRODUCT line-item count and the
    # source document type(s) (receipt / payment_confirmation / invoice / ...) so
    # list views can show what a fact was built from, not just its fact_type.
    # Non_Item lines (totals, tax/VAT analysis, payment-method and footer text the
    # structurer sometimes captures) are excluded from the count, matching how
    # basket analytics treat them, so a 2-item receipt reads as 2, not 10.
    data_rows = db.fetchall(
        f"""SELECT *,
                   (SELECT COUNT(*) FROM fact_items fi
                    WHERE fi.fact_id = facts.id
                      AND COALESCE(fi.category, '') != 'Non_Item') AS item_count,
                   (SELECT GROUP_CONCAT(DISTINCT
                              json_extract(d.raw_extraction, '$.document_type'))
                    FROM bundles b JOIN documents d ON d.id = b.document_id
                    WHERE b.cloud_id = facts.cloud_id) AS document_type
            FROM facts WHERE {where}
            ORDER BY event_date DESC LIMIT ? OFFSET ?""",  # noqa: S608
        tuple(params) + (limit, offset),
    )

    return {
        "facts": [dict(r) for r in data_rows],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


def search_facts(
    db: DatabaseManager,
    query: str,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    """Search facts across vendor names and item names.

    Performs a case-insensitive LIKE search on both the vendor field of
    the facts table and the name field of the fact_items table.  A fact
    is included if it matches on either field.

    Args:
        db: Database manager instance.
        query: Search string; % wildcards are added automatically.
        offset: Number of rows to skip (zero-based).
        limit: Maximum number of rows to return.

    Returns:
        Dict with keys:
            - facts: list of distinct fact dicts ordered by event_date DESC
            - total: total distinct matching facts (before pagination)
            - offset: the requested offset
            - limit: the requested limit
    """
    pattern = f"%{query.lower()}%"

    # Use UNION of two queries to get fact IDs that match on vendor OR item name
    count_row = db.fetchone(
        """
        SELECT COUNT(*) AS cnt FROM (
            SELECT id FROM facts
            WHERE LOWER(vendor) LIKE ?
            UNION
            SELECT DISTINCT fi.fact_id AS id
            FROM fact_items fi
            WHERE LOWER(fi.name) LIKE ?
        ) matched
        """,
        (pattern, pattern),
    )
    total: int = count_row["cnt"] if count_row else 0

    data_rows = db.fetchall(
        """
        SELECT f.* FROM facts f
        WHERE f.id IN (
            SELECT id FROM facts
            WHERE LOWER(vendor) LIKE ?
            UNION
            SELECT DISTINCT fi.fact_id AS id
            FROM fact_items fi
            WHERE LOWER(fi.name) LIKE ?
        )
        ORDER BY f.event_date DESC
        LIMIT ? OFFSET ?
        """,
        (pattern, pattern, limit, offset),
    )

    return {
        "facts": [dict(r) for r in data_rows],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


def list_unassigned(db: DatabaseManager) -> list[dict[str, Any]]:
    """List bundles with no cloud assignment.

    These are bundles that have been detached from all clouds and need
    re-matching or manual correction.

    Args:
        db: Database manager instance.

    Returns:
        List of bundle dicts, each containing id, document_id,
        bundle_type, and file_path.
    """
    return v2_store.get_unassigned_bundles(db)


def get_document(db: DatabaseManager, document_id: str) -> dict[str, Any] | None:
    """Get a document by ID.

    Args:
        db: Database manager instance.
        document_id: UUID of the document.

    Returns:
        Document metadata dict, or None if not found.
    """
    row = db.fetchone("SELECT * FROM documents WHERE id = ?", (document_id,))
    return dict(row) if row else None


def get_primary_fact_id_for_document(
    db: DatabaseManager, document_id: str
) -> str | None:
    """Resolve the fact a document contributed to (document → … → fact).

    A document's bundle joins a cloud that collapses to a fact; when the upload
    is recognised as the same transaction as an earlier document, both share one
    fact. Returns that fact's id (the first if several), or None if the document
    has not collapsed into a fact yet. Used to attach a location to the right
    fact after a Telegram/API upload.
    """
    facts = v2_store.get_facts_for_document(db, document_id)
    return facts[0]["id"] if facts else None


def list_documents(
    db: DatabaseManager,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    """List documents with optional filters and server-side pagination.

    Args:
        db: Database manager instance.
        filters: Optional dict of filter criteria. Recognised keys:
            - date_from (str): ISO date/datetime lower bound (inclusive)
            - date_to (str): ISO date/datetime upper bound (inclusive)
        offset: Number of rows to skip (zero-based).
        limit: Maximum number of rows to return.

    Returns:
        Dict with keys:
            - documents: list of document dicts
            - total: total document count
            - offset: the requested offset
            - limit: the requested limit
    """
    filters = filters or {}
    conditions: list[str] = ["1=1"]
    params: list[Any] = []

    date_from = filters.get("date_from")
    if date_from is not None:
        conditions.append("created_at >= ?")
        params.append(str(date_from))

    date_to = filters.get("date_to")
    if date_to is not None:
        conditions.append("created_at <= ?")
        params.append(str(date_to))

    where = " AND ".join(conditions)

    count_row = db.fetchone(
        f"SELECT COUNT(*) AS cnt FROM documents WHERE {where}",  # noqa: S608
        tuple(params),
    )
    total: int = count_row["cnt"] if count_row else 0

    data_rows = db.fetchall(
        f"SELECT * FROM documents WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",  # noqa: S608
        tuple(params) + (limit, offset),
    )

    return {
        "documents": [dict(r) for r in data_rows],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


def get_document_line_items(
    db: DatabaseManager, document_id: str
) -> list[dict[str, Any]]:
    """Get fact items linked to a document via the bundle/cloud chain.

    Args:
        db: Database manager instance.
        document_id: UUID of the document.

    Returns:
        List of fact item dicts linked to the document.
    """
    rows = db.fetchall(
        """SELECT fi.*
           FROM fact_items fi
           JOIN facts f ON fi.fact_id = f.id
           JOIN cloud_bundles cb ON f.cloud_id = cb.cloud_id
           JOIN bundles b ON cb.bundle_id = b.id
           WHERE b.document_id = ?""",
        (document_id,),
    )
    return [dict(row) for row in rows]


def delete_document(db: DatabaseManager, document_id: str) -> bool:
    """Delete a document and all related chain records.

    Cascade order respects FK constraints:
    fact_items (by atom) → bundle_atoms → cloud_bundles → bundles → atoms → documents.

    Args:
        db: Database manager instance.
        document_id: UUID of the document to delete.

    Returns:
        True if the document existed and was deleted, False if not found.
    """
    row = db.fetchone("SELECT id FROM documents WHERE id = ?", (document_id,))
    if not row:
        return False

    with db.transaction() as cursor:
        cursor.execute(
            "DELETE FROM fact_items WHERE atom_id IN "
            "(SELECT id FROM atoms WHERE document_id = ?)",
            (document_id,),
        )
        cursor.execute(
            "DELETE FROM bundle_atoms WHERE bundle_id IN "
            "(SELECT id FROM bundles WHERE document_id = ?)",
            (document_id,),
        )
        cursor.execute(
            "DELETE FROM cloud_bundles WHERE bundle_id IN "
            "(SELECT id FROM bundles WHERE document_id = ?)",
            (document_id,),
        )
        cursor.execute(
            "DELETE FROM bundles WHERE document_id = ?",
            (document_id,),
        )
        cursor.execute(
            "DELETE FROM atoms WHERE document_id = ?",
            (document_id,),
        )
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    return True


def delete_fact(db: DatabaseManager, fact_id: str) -> bool:
    """Delete a fact and clean up its cloud if orphaned.

    Args:
        db: Database manager instance.
        fact_id: UUID of the fact to delete.

    Returns:
        True if the fact existed and was deleted, False if not found.
    """
    row = db.fetchone("SELECT id, cloud_id FROM facts WHERE id = ?", (fact_id,))
    if not row:
        return False

    cloud_id = row["cloud_id"]

    with db.transaction() as cursor:
        cursor.execute("DELETE FROM fact_items WHERE fact_id = ?", (fact_id,))
        cursor.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        # Clean up cloud only when fully orphaned. bundles.cloud_id also
        # references clouds, so the cloud must stay while any bundle points
        # at it (delete_fact called before delete_document) or the delete
        # hits a FOREIGN KEY error and rolls back the whole transaction.
        remaining = cursor.execute(
            "SELECT (SELECT COUNT(*) FROM facts WHERE cloud_id = :c) "
            "+ (SELECT COUNT(*) FROM bundles WHERE cloud_id = :c)",
            {"c": cloud_id},
        ).fetchone()
        if remaining and remaining[0] == 0:
            cursor.execute("DELETE FROM cloud_bundles WHERE cloud_id = ?", (cloud_id,))
            cursor.execute("DELETE FROM clouds WHERE id = ?", (cloud_id,))

    return True


def list_fact_items(
    db: DatabaseManager,
    filters: dict[str, Any] | None = None,
    offset: int = 0,
    limit: int = 50,
) -> dict[str, Any]:
    """List fact items with optional filters and server-side pagination.

    Args:
        db: Database manager instance.
        filters: Optional dict. Recognised keys:
            - category (str): exact match on category
            - name (str): case-insensitive substring match
            - brand (str): case-insensitive substring match
            - fact_id (str): exact match on fact_id
            - date_from (str): ISO date lower bound (via parent fact)
            - date_to (str): ISO date upper bound (via parent fact)
        offset: Number of rows to skip.
        limit: Maximum number of rows to return.

    Returns:
        Dict with keys: fact_items, total, offset, limit.
    """
    filters = filters or {}
    conditions: list[str] = ["1=1"]
    params: list[Any] = []

    if filters.get("category"):
        conditions.append("fi.category = ?")
        params.append(filters["category"])
    if filters.get("name"):
        conditions.append("fi.name LIKE ?")
        params.append(f"%{filters['name']}%")
    if filters.get("brand"):
        conditions.append("fi.brand LIKE ?")
        params.append(f"%{filters['brand']}%")
    if filters.get("fact_id"):
        conditions.append("fi.fact_id = ?")
        params.append(filters["fact_id"])
    if filters.get("date_from"):
        conditions.append("fi.fact_id IN (SELECT id FROM facts WHERE event_date >= ?)")
        params.append(str(filters["date_from"]))
    if filters.get("date_to"):
        conditions.append("fi.fact_id IN (SELECT id FROM facts WHERE event_date <= ?)")
        params.append(str(filters["date_to"]))

    where = " AND ".join(conditions)

    count_row = db.fetchone(
        f"SELECT COUNT(*) AS cnt FROM fact_items fi WHERE {where}",  # noqa: S608
        tuple(params),
    )
    total: int = count_row["cnt"] if count_row else 0

    data_rows = db.fetchall(
        f"SELECT fi.* FROM fact_items fi WHERE {where} ORDER BY fi.created_at DESC LIMIT ? OFFSET ?",  # noqa: S608
        tuple(params) + (limit, offset),
    )

    return {
        "fact_items": [dict(r) for r in data_rows],
        "total": total,
        "offset": offset,
        "limit": limit,
    }


def get_fact_item(db: DatabaseManager, fact_item_id: str) -> dict[str, Any] | None:
    """Get a single fact item by ID.

    Args:
        db: Database manager instance.
        fact_item_id: UUID of the fact item.

    Returns:
        Fact item dict, or None if not found.
    """
    row = db.fetchone("SELECT * FROM fact_items WHERE id = ?", (fact_item_id,))
    return dict(row) if row else None


def list_fact_items_with_fact(
    db: DatabaseManager,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """List fact items joined with parent fact fields.

    Returns all fields needed for CSV export: item fields plus
    vendor, vendor_key, event_date, event_time, currency, country from the
    parent fact — so each item is self-describing along every filter axis.

    Treat each fact_item as a "star" that can be filtered along any axis.

    Args:
        db: Database manager instance.
        filters: Optional dict. Recognised keys (all optional, AND-combined):
            - name (str): substring match on fi.name
            - category (str): exact match on fi.category
            - brand (str): substring match on fi.brand
            - vendor (str): substring match on f.vendor
            - vendor_key (str): exact match on f.vendor_key
            - currency (str): exact match on f.currency (ISO 4217)
            - country (str): exact match on f.country (jurisdiction)
            - date_from / date_to (str | date): f.event_date bound (inclusive)
            - datetime_from / datetime_to (str): date+time bound, e.g.
              "2026-12-01 12:00:00" — uses event_date+event_time for finer
              granularity than the date-only bounds
            - price_min / price_max (number): bound on fi.total_price
            - category_path (str): hierarchical prefix match — "food" matches
              everything at or under "food" (e.g. "food > dairy > milk")

    Returns:
        List of dicts with item + parent fact fields.
    """
    filters = filters or {}
    conditions: list[str] = ["1=1"]
    params: list[Any] = []

    # Substring (LIKE) filters
    for key, column in (
        ("name", "fi.name"),
        ("brand", "fi.brand"),
        ("vendor", "f.vendor"),
    ):
        if filters.get(key):
            conditions.append(f"LOWER({column}) LIKE ?")
            params.append(f"%{str(filters[key]).lower()}%")

    # Exact-match filters
    for key, column in (
        ("category", "fi.category"),
        ("vendor_key", "f.vendor_key"),
        ("currency", "f.currency"),
        ("country", "f.country"),
    ):
        if filters.get(key):
            conditions.append(f"{column} = ?")
            params.append(filters[key])

    date_from = filters.get("date_from")
    if date_from is not None:
        if isinstance(date_from, date):
            date_from = date_from.isoformat()
        conditions.append("f.event_date >= ?")
        params.append(str(date_from))

    date_to = filters.get("date_to")
    if date_to is not None:
        if isinstance(date_to, date):
            date_to = date_to.isoformat()
        conditions.append("f.event_date <= ?")
        params.append(str(date_to))

    # Date+time bounds (finer granularity). event_date and event_time are
    # separate columns; concatenate them into a comparable timestamp string.
    if filters.get("datetime_from") is not None:
        conditions.append(
            "(f.event_date || ' ' || COALESCE(f.event_time, '00:00:00')) >= ?"
        )
        params.append(str(filters["datetime_from"]))
    if filters.get("datetime_to") is not None:
        conditions.append(
            "(f.event_date || ' ' || COALESCE(f.event_time, '23:59:59')) <= ?"
        )
        params.append(str(filters["datetime_to"]))

    if filters.get("price_min") is not None:
        conditions.append("CAST(fi.total_price AS REAL) >= ?")
        params.append(float(filters["price_min"]))
    if filters.get("price_max") is not None:
        conditions.append("CAST(fi.total_price AS REAL) <= ?")
        params.append(float(filters["price_max"]))

    # Hierarchical category prefix: "food" matches "food", "food > dairy",
    # "food > dairy > milk" — i.e. everything at or under the given node.
    category_path = filters.get("category_path")
    if category_path:
        prefix = str(category_path).strip().lower()
        conditions.append(
            "(LOWER(fi.category_path) = ? OR LOWER(fi.category_path) LIKE ?)"
        )
        params.append(prefix)
        params.append(f"{prefix} > %")

    where = " AND ".join(conditions)

    rows = db.fetchall(
        f"""SELECT fi.id, fi.name, fi.quantity, fi.unit_price, fi.total_price,
                   fi.category, fi.category_path, fi.brand, f.vendor, f.vendor_key,
                   f.event_date, f.event_time, f.currency, f.country
            FROM fact_items fi
            JOIN facts f ON fi.fact_id = f.id
            WHERE {where}
            ORDER BY f.event_date DESC, f.event_time DESC""",  # noqa: S608
        tuple(params),
    )
    return [dict(row) for row in rows]


def category_summary(
    db: DatabaseManager,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Get spending summary grouped by item category.

    Args:
        db: Database manager instance.
        filters: Optional dict. Recognised keys:
            - date_from (str): ISO date lower bound (via parent fact)
            - date_to (str): ISO date upper bound (via parent fact)

    Returns:
        List of dicts with category, item_count, total_spent.
    """
    filters = filters or {}
    conditions: list[str] = ["fi.category IS NOT NULL"]
    params: list[Any] = []

    if filters.get("date_from"):
        conditions.append("fi.fact_id IN (SELECT id FROM facts WHERE event_date >= ?)")
        params.append(str(filters["date_from"]))
    if filters.get("date_to"):
        conditions.append("fi.fact_id IN (SELECT id FROM facts WHERE event_date <= ?)")
        params.append(str(filters["date_to"]))

    where = " AND ".join(conditions)

    rows = db.fetchall(
        f"""SELECT fi.category, COUNT(*) as item_count,
                   SUM(CAST(fi.total_price AS REAL)) as total_spent
            FROM fact_items fi
            WHERE {where}
            GROUP BY fi.category
            ORDER BY total_spent DESC""",  # noqa: S608
        tuple(params),
    )
    return [dict(row) for row in rows]
