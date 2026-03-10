"""Identity CRUD operations.

Stores and queries identities + identity_members in SQLite.
All write operations use db.transaction() for atomicity.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)


def create_identity(
    db: DatabaseManager,
    entity_type: str,
    canonical_name: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Create a new identity.

    Args:
        db: Database manager.
        entity_type: 'vendor' or 'item'.
        canonical_name: User-chosen display name.
        metadata: Optional type-specific metadata (barcode, legal_name, etc.).

    Returns:
        The identity ID.
    """
    identity_id = str(uuid4())
    now = datetime.now().isoformat()
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO identities (id, entity_type, canonical_name, metadata, "
            "active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                identity_id,
                entity_type,
                canonical_name,
                json.dumps(metadata) if metadata else None,
                True,
                now,
                now,
            ),
        )
    logger.info(
        f"Created {entity_type} identity '{canonical_name}' ({identity_id[:8]})"
    )
    return identity_id


def add_member(
    db: DatabaseManager,
    identity_id: str,
    member_type: str,
    value: str,
    source: str = "user",
) -> str:
    """Add a member to an identity.

    Args:
        identity_id: The identity to add to.
        member_type: One of: name, normalized_name, registration, vendor_key, barcode.
        value: The member value.
        source: 'user' (manual) or 'auto' (system-learned).

    Returns:
        The member ID.
    """
    member_id = str(uuid4())
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT OR IGNORE INTO identity_members "
            "(id, identity_id, member_type, value, source) "
            "VALUES (?, ?, ?, ?, ?)",
            (member_id, identity_id, member_type, value, source),
        )
        # Update identity timestamp
        cursor.execute(
            "UPDATE identities SET updated_at = ? WHERE id = ?",
            (datetime.now().isoformat(), identity_id),
        )
    return member_id


def remove_member(db: DatabaseManager, member_id: str) -> bool:
    """Remove a member from an identity.

    Returns:
        True if a row was deleted.
    """
    with db.transaction() as cursor:
        cursor.execute("DELETE FROM identity_members WHERE id = ?", (member_id,))
        return cursor.rowcount > 0


def delete_identity(db: DatabaseManager, identity_id: str) -> bool:
    """Delete an identity and all its members (CASCADE).

    Returns:
        True if a row was deleted.
    """
    with db.transaction() as cursor:
        cursor.execute("DELETE FROM identities WHERE id = ?", (identity_id,))
        return cursor.rowcount > 0


def update_identity(
    db: DatabaseManager,
    identity_id: str,
    canonical_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    active: bool | None = None,
) -> bool:
    """Update identity fields.

    Only non-None arguments are updated.

    Returns:
        True if a row was updated.
    """
    sets: list[str] = []
    params: list[Any] = []

    if canonical_name is not None:
        sets.append("canonical_name = ?")
        params.append(canonical_name)
    if metadata is not None:
        sets.append("metadata = ?")
        params.append(json.dumps(metadata))
    if active is not None:
        sets.append("active = ?")
        params.append(active)

    if not sets:
        return False

    sets.append("updated_at = ?")
    params.append(datetime.now().isoformat())
    params.append(identity_id)

    with db.transaction() as cursor:
        cursor.execute(
            f"UPDATE identities SET {', '.join(sets)} WHERE id = ?",
            params,
        )
        return cursor.rowcount > 0


def get_identity(db: DatabaseManager, identity_id: str) -> dict[str, Any] | None:
    """Get an identity by ID with all its members."""
    conn = db.get_connection()
    row = conn.execute(
        "SELECT * FROM identities WHERE id = ?", (identity_id,)
    ).fetchone()
    if not row:
        return None

    members = conn.execute(
        "SELECT * FROM identity_members WHERE identity_id = ? ORDER BY created_at",
        (identity_id,),
    ).fetchall()

    return _row_to_dict(row, members)


def list_identities(
    db: DatabaseManager,
    entity_type: str | None = None,
    active_only: bool = False,
) -> list[dict[str, Any]]:
    """List identities with optional filtering."""
    conn = db.get_connection()

    where: list[str] = []
    params: list[Any] = []

    if entity_type:
        where.append("i.entity_type = ?")
        params.append(entity_type)
    if active_only:
        where.append("i.active = 1")

    where_clause = f"WHERE {' AND '.join(where)}" if where else ""

    rows = conn.execute(
        f"SELECT i.* FROM identities i {where_clause} ORDER BY i.canonical_name",
        params,
    ).fetchall()

    result = []
    for row in rows:
        members = conn.execute(
            "SELECT * FROM identity_members WHERE identity_id = ?",
            (row["id"],),
        ).fetchall()
        result.append(_row_to_dict(row, members))

    return result


def get_members_by_type(
    db: DatabaseManager,
    identity_id: str,
    member_type: str,
) -> list[dict[str, Any]]:
    """Get all members of a specific type for an identity."""
    conn = db.get_connection()
    rows = conn.execute(
        "SELECT * FROM identity_members "
        "WHERE identity_id = ? AND member_type = ? ORDER BY value",
        (identity_id, member_type),
    ).fetchall()
    return [dict(r) for r in rows]


def _row_to_dict(identity_row: Any, member_rows: list[Any]) -> dict[str, Any]:
    """Convert identity + members rows to dict."""
    identity = dict(identity_row)
    if identity.get("metadata") and isinstance(identity["metadata"], str):
        identity["metadata"] = json.loads(identity["metadata"])
    identity["members"] = [dict(m) for m in member_rows]
    return identity
