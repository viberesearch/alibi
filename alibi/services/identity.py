"""Identity service for vendor and item identity management.

Wraps identity matching and store operations. Adds merge_vendors
for manual identity consolidation.
"""

from __future__ import annotations

import logging
from typing import Any

from alibi.db.connection import DatabaseManager
from alibi.identities.matching import find_vendor_identity

logger = logging.getLogger(__name__)


def resolve_vendor(
    db: DatabaseManager,
    vendor_name: str | None = None,
    vendor_key: str | None = None,
    registration: str | None = None,
) -> dict[str, Any] | None:
    """Find a vendor identity by any matching signal.

    Resolution priority (strongest first):
    1. vendor_key match
    2. registration match (vat_number > tax_id > legacy registration)
    3. exact name match
    4. normalized_name match

    Args:
        db: Database manager.
        vendor_name: Raw vendor name from extraction.
        vendor_key: Canonical vendor key (VAT number used as primary key).
        registration: VAT or tax ID registration number.

    Returns:
        Identity dict with members, or None if not found.
    """
    return find_vendor_identity(
        db,
        vendor_name=vendor_name,
        vendor_key=vendor_key,
        registration=registration,
    )


def list_identities(
    db: DatabaseManager,
    entity_type: str | None = None,
) -> list[dict[str, Any]]:
    """List all identities, optionally filtered by entity_type.

    Args:
        db: Database manager.
        entity_type: Filter by entity type ('vendor' or 'item').
            Pass None to return all identities.

    Returns:
        List of identity dicts (each includes a 'members' list),
        ordered by canonical_name.
    """
    from alibi.identities import store as _store

    return _store.list_identities(db, entity_type=entity_type)


def get_identity(db: DatabaseManager, identity_id: str) -> dict[str, Any] | None:
    """Get a single identity with its members.

    Args:
        db: Database manager.
        identity_id: The identity UUID.

    Returns:
        Identity dict with 'members' list, or None if not found.
    """
    from alibi.identities import store as _store

    return _store.get_identity(db, identity_id)


def merge_vendors(
    db: DatabaseManager,
    identity_id_a: str,
    identity_id_b: str,
) -> bool:
    """Merge two vendor identities into one.

    All identity_members from identity_id_b are moved to identity_id_a,
    then identity_id_b is deleted. The operation is atomic; if either
    identity does not exist the function returns False without modifying
    the database.

    Args:
        db: Database manager.
        identity_id_a: The identity to merge into (survives).
        identity_id_b: The identity to merge from (deleted).

    Returns:
        True on success, False if either identity was not found.
    """
    from alibi.identities import store as _store

    # Verify both identities exist before touching anything.
    a = _store.get_identity(db, identity_id_a)
    b = _store.get_identity(db, identity_id_b)

    if a is None or b is None:
        missing = []
        if a is None:
            missing.append(identity_id_a[:8])
        if b is None:
            missing.append(identity_id_b[:8])
        logger.warning("merge_vendors: identity not found: %s", ", ".join(missing))
        return False

    with db.transaction() as cursor:
        # Move all members from b -> a.
        cursor.execute(
            "UPDATE identity_members SET identity_id = ? WHERE identity_id = ?",
            (identity_id_a, identity_id_b),
        )
        # Remove the now-empty identity b.
        cursor.execute(
            "DELETE FROM identities WHERE id = ?",
            (identity_id_b,),
        )

    logger.info(
        "merge_vendors: merged %s into %s",
        identity_id_b[:8],
        identity_id_a[:8],
    )
    return True
