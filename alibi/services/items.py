"""Item (asset) service layer.

Wraps the raw SQL from the items API router into reusable service
functions. All consumers (CLI, API, MCP) should call these functions
instead of writing SQL directly.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

# Fields that are allowed in update_item calls, mapping to their column names.
_UPDATABLE_FIELDS = {
    "name",
    "category",
    "status",
    "current_value",
    "warranty_expires",
}


def _row_to_item(row: Any) -> dict[str, Any]:
    """Convert a database row to an item dict."""
    return {
        "id": row["id"],
        "space_id": row["space_id"],
        "name": row["name"],
        "category": row["category"],
        "model": row["model"],
        "serial_number": row["serial_number"],
        "purchase_date": str(row["purchase_date"]) if row["purchase_date"] else None,
        "purchase_price": str(row["purchase_price"]) if row["purchase_price"] else None,
        "current_value": str(row["current_value"]) if row["current_value"] else None,
        "currency": row["currency"] or "EUR",
        "status": row["status"] or "active",
        "warranty_expires": (
            str(row["warranty_expires"]) if row["warranty_expires"] else None
        ),
        "warranty_type": row["warranty_type"],
        "insurance_covered": bool(row["insurance_covered"]),
        "created_at": str(row["created_at"]) if row["created_at"] else None,
    }


def list_items(
    db: DatabaseManager,
    category: str | None = None,
    status: str | None = None,
    warranty_expiring: bool = False,
) -> list[dict[str, Any]]:
    """List items with optional filters.

    Args:
        db: Database manager instance.
        category: Optional exact match on category.
        status: Optional exact match on status.
        warranty_expiring: When True, only return items whose warranty
            expires within the next 30 days.

    Returns:
        List of item dicts ordered by created_at descending.
    """
    conditions: list[str] = []
    params: list[Any] = []

    if category:
        conditions.append("category = ?")
        params.append(category)
    if status:
        conditions.append("status = ?")
        params.append(status)
    if warranty_expiring:
        conditions.append(
            "warranty_expires IS NOT NULL AND "
            "warranty_expires <= date('now', '+30 days') AND "
            "warranty_expires >= date('now')"
        )

    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT * FROM items{where} ORDER BY created_at DESC"  # noqa: S608

    rows = db.fetchall(sql, tuple(params))
    return [_row_to_item(row) for row in rows]


def get_item(db: DatabaseManager, item_id: str) -> dict[str, Any] | None:
    """Get a single item by ID.

    Args:
        db: Database manager instance.
        item_id: UUID of the item to retrieve.

    Returns:
        Item dict, or None if not found.
    """
    row = db.fetchone("SELECT * FROM items WHERE id = ?", (item_id,))
    return _row_to_item(row) if row else None


def get_item_documents(db: DatabaseManager, item_id: str) -> list[dict[str, Any]]:
    """Get documents linked to an item.

    Args:
        db: Database manager instance.
        item_id: UUID of the item.

    Returns:
        List of dicts with id, file_path, created_at, and link_type.
    """
    rows = db.fetchall(
        """SELECT d.id, d.file_path, d.created_at, id2.link_type
           FROM item_documents id2
           JOIN documents d ON id2.document_id = d.id
           WHERE id2.item_id = ?""",
        (item_id,),
    )
    return [dict(r) for r in rows]


def get_item_facts(db: DatabaseManager, item_id: str) -> list[dict[str, Any]]:
    """Get facts linked to an item.

    Args:
        db: Database manager instance.
        item_id: UUID of the item.

    Returns:
        List of dicts with id, fact_type, total_amount, event_date,
        and link_type.
    """
    rows = db.fetchall(
        """SELECT f.id, f.fact_type, f.total_amount, f.event_date, if2.link_type
           FROM item_facts if2
           JOIN facts f ON if2.fact_id = f.id
           WHERE if2.item_id = ?""",
        (item_id,),
    )
    return [dict(r) for r in rows]


def create_item(db: DatabaseManager, item_data: dict[str, Any]) -> str:
    """Create a new item.

    Args:
        db: Database manager instance.
        item_data: Dict of item fields. Recognised keys:
            - name (str, required): display name of the item
            - space_id (str): defaults to "default"
            - category (str | None)
            - model (str | None)
            - serial_number (str | None)
            - purchase_date (str | None): ISO date string
            - purchase_price (str | None): numeric string
            - currency (str): defaults to "EUR"
            - warranty_expires (str | None): ISO date string
            - warranty_type (str | None)
            - created_by (str | None): user ID to record as creator

    Returns:
        The UUID string of the newly created item.
    """
    item_id = str(uuid.uuid4())

    with db.transaction() as cursor:
        cursor.execute(
            """
            INSERT INTO items (id, space_id, name, category, model,
                               serial_number, purchase_date, purchase_price,
                               currency, warranty_expires, warranty_type,
                               status, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
            """,
            (
                item_id,
                item_data.get("space_id", "default"),
                item_data["name"],
                item_data.get("category"),
                item_data.get("model"),
                item_data.get("serial_number"),
                item_data.get("purchase_date"),
                item_data.get("purchase_price"),
                item_data.get("currency", "EUR"),
                item_data.get("warranty_expires"),
                item_data.get("warranty_type"),
                item_data.get("created_by"),
            ),
        )

    return item_id


def update_item(db: DatabaseManager, item_id: str, updates: dict[str, Any]) -> bool:
    """Update an existing item.

    Only fields present in the updates dict are changed. Accepted field
    names: name, category, status, current_value, warranty_expires.
    Unknown keys are silently ignored.

    Args:
        db: Database manager instance.
        item_id: UUID of the item to update.
        updates: Dict of field names to new values.

    Returns:
        True if the item existed and was updated, False if not found.
        Returns False (not an error) when updates contains no recognised
        fields or is empty.
    """
    row = db.fetchone("SELECT id FROM items WHERE id = ?", (item_id,))
    if not row:
        return False

    set_clauses: list[str] = []
    params: list[Any] = []

    for field, value in updates.items():
        if field in _UPDATABLE_FIELDS:
            set_clauses.append(f"{field} = ?")
            params.append(value)

    if not set_clauses:
        return False

    set_clauses.append("modified_at = CURRENT_TIMESTAMP")
    params.append(item_id)

    sql = f"UPDATE items SET {', '.join(set_clauses)} WHERE id = ?"  # noqa: S608
    db.execute(sql, tuple(params))
    db.get_connection().commit()

    return True


def delete_item(db: DatabaseManager, item_id: str) -> bool:
    """Delete an item and its junction-table rows.

    Removes the item from item_documents, item_facts, and items.

    Args:
        db: Database manager instance.
        item_id: UUID of the item to delete.

    Returns:
        True if the item existed and was deleted, False if not found.
    """
    row = db.fetchone("SELECT id FROM items WHERE id = ?", (item_id,))
    if not row:
        return False

    with db.transaction() as cursor:
        cursor.execute("DELETE FROM item_documents WHERE item_id = ?", (item_id,))
        cursor.execute("DELETE FROM item_facts WHERE item_id = ?", (item_id,))
        cursor.execute("DELETE FROM items WHERE id = ?", (item_id,))

    return True
