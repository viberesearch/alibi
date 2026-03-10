"""User and API key management service.

Provides user CRUD, mnemonic API key lifecycle, and contact management
(Telegram, email). All functions are synchronous and accept a DatabaseManager.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from alibi.auth.keys import generate_mnemonic, generate_salt, hash_key, key_prefix
from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------


def create_user(
    db: DatabaseManager,
    name: str | None = None,
) -> dict[str, Any]:
    """Create a new user.

    Args:
        name: Optional display name. NULL by default (zero PII).

    Returns:
        Dict with id, name, is_active, created_at.
    """
    user_id = str(uuid4())
    db.execute(
        "INSERT INTO users (id, name) VALUES (?, ?)",
        (user_id, name),
    )
    db.get_connection().commit()
    logger.info("Created user %s", user_id[:8])
    return {"id": user_id, "name": name, "is_active": 1}


def update_user(
    db: DatabaseManager,
    user_id: str,
    name: str | None = None,
) -> bool:
    """Update user fields. Currently only name is updatable.

    Returns True if the user was found and updated.
    """
    cursor = db.execute(
        "UPDATE users SET name = ? WHERE id = ?",
        (name, user_id),
    )
    db.get_connection().commit()
    updated: int = cursor.rowcount
    if updated:
        logger.info("Updated user %s name", user_id[:8])
    return updated > 0


def get_user(db: DatabaseManager, user_id: str) -> dict[str, Any] | None:
    """Look up a user by ID."""
    row = db.fetchone("SELECT * FROM users WHERE id = ?", (user_id,))
    return dict(row) if row else None


def list_users(db: DatabaseManager) -> list[dict[str, Any]]:
    """List all active users."""
    rows = db.fetchall(
        "SELECT id, name, is_active, created_at "
        "FROM users WHERE is_active = 1 ORDER BY created_at"
    )
    return [dict(r) for r in rows]


def get_display_name(user: dict[str, Any]) -> str:
    """Get a display-friendly name for a user.

    Returns the user's name if set, otherwise 'user'.
    """
    return user.get("name") or "user"


# ---------------------------------------------------------------------------
# Contact management (1:N — telegram, email)
# ---------------------------------------------------------------------------


def add_contact(
    db: DatabaseManager,
    user_id: str,
    contact_type: str,
    value: str,
    label: str | None = None,
) -> dict[str, Any] | None:
    """Add a contact to a user (telegram account, email address, etc.).

    Returns the created contact dict, or None if the user doesn't exist.
    Raises sqlite3.IntegrityError if the contact is already linked to another user.
    """
    # Verify user exists
    user = get_user(db, user_id)
    if not user:
        return None

    contact_id = str(uuid4())
    db.execute(
        "INSERT INTO user_contacts (id, user_id, contact_type, value, label) "
        "VALUES (?, ?, ?, ?, ?)",
        (contact_id, user_id, contact_type, value, label),
    )
    db.get_connection().commit()
    logger.info(
        "Added %s contact %s to user %s",
        contact_type,
        value[:20],
        user_id[:8],
    )
    return {
        "id": contact_id,
        "user_id": user_id,
        "contact_type": contact_type,
        "value": value,
        "label": label,
    }


def remove_contact(
    db: DatabaseManager,
    contact_id: str,
) -> bool:
    """Remove a contact by its ID.

    Returns True if the contact was found and deleted.
    """
    cursor = db.execute(
        "DELETE FROM user_contacts WHERE id = ?",
        (contact_id,),
    )
    db.get_connection().commit()
    deleted: int = cursor.rowcount
    if deleted:
        logger.info("Removed contact %s", contact_id[:8])
    return deleted > 0


def remove_contact_by_value(
    db: DatabaseManager,
    contact_type: str,
    value: str,
) -> bool:
    """Remove a contact by type and value (e.g., unlink a specific Telegram ID).

    Returns True if the contact was found and deleted.
    """
    cursor = db.execute(
        "DELETE FROM user_contacts WHERE contact_type = ? AND value = ?",
        (contact_type, value),
    )
    db.get_connection().commit()
    deleted: int = cursor.rowcount
    if deleted:
        logger.info("Removed %s contact %s", contact_type, value[:20])
    return deleted > 0


def list_contacts(
    db: DatabaseManager,
    user_id: str,
) -> list[dict[str, Any]]:
    """List all contacts for a user."""
    rows = db.fetchall(
        "SELECT id, contact_type, value, label, created_at "
        "FROM user_contacts WHERE user_id = ? ORDER BY created_at",
        (user_id,),
    )
    return [dict(r) for r in rows]


def find_user_by_contact(
    db: DatabaseManager,
    contact_type: str,
    value: str,
) -> dict[str, Any] | None:
    """Find a user by a contact value (e.g., telegram ID, email).

    Returns the user dict if found and active, None otherwise.
    """
    row = db.fetchone(
        "SELECT u.* FROM users u "
        "JOIN user_contacts uc ON u.id = uc.user_id "
        "WHERE uc.contact_type = ? AND uc.value = ? AND u.is_active = 1",
        (contact_type, value),
    )
    return dict(row) if row else None


def find_user_by_telegram(
    db: DatabaseManager,
    telegram_user_id: str,
) -> dict[str, Any] | None:
    """Find a user by their linked Telegram user ID.

    Convenience wrapper around find_user_by_contact.
    """
    return find_user_by_contact(db, "telegram", telegram_user_id)


# Backward compatibility alias
def link_telegram(
    db: DatabaseManager,
    user_id: str,
    telegram_user_id: str,
) -> bool:
    """Link a Telegram account to a user. Backward compat wrapper."""
    try:
        result = add_contact(db, user_id, "telegram", telegram_user_id)
        return result is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------
# API key lifecycle
# ---------------------------------------------------------------------------


def create_api_key(
    db: DatabaseManager,
    user_id: str,
    label: str = "default",
) -> dict[str, Any]:
    """Generate a mnemonic API key for a user.

    The plaintext mnemonic is returned exactly once. Only the PBKDF2 hash
    and salt are stored.

    Returns:
        Dict with mnemonic (plaintext), id, prefix, label.
    """
    mnemonic = generate_mnemonic()
    salt = generate_salt()
    hashed = hash_key(mnemonic, salt)
    prefix = key_prefix(mnemonic)
    key_id = str(uuid4())

    db.execute(
        "INSERT INTO api_keys (id, user_id, key_hash, key_prefix, label, salt) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (key_id, user_id, hashed, prefix, label, salt),
    )
    db.get_connection().commit()
    logger.info("Created API key %s for user %s", key_id[:8], user_id[:8])

    return {
        "mnemonic": mnemonic,
        "id": key_id,
        "prefix": prefix,
        "label": label,
    }


def validate_api_key(
    db: DatabaseManager,
    mnemonic: str,
) -> dict[str, Any] | None:
    """Validate a mnemonic API key.

    Supports both PBKDF2-salted keys (new) and unsalted SHA-256 keys (legacy).
    Legacy keys are automatically upgraded to PBKDF2 on successful validation.

    Returns the associated user dict if valid, None otherwise.
    """
    # Try salted keys first: fetch all active keys and check each
    rows = db.fetchall(
        "SELECT ak.id AS key_id, ak.user_id, ak.key_hash, ak.salt, "
        "u.name, u.is_active AS user_active "
        "FROM api_keys ak "
        "JOIN users u ON ak.user_id = u.id "
        "WHERE ak.is_active = 1",
    )
    for row in rows:
        if not row["user_active"]:
            continue
        salt = row["salt"]
        hashed = hash_key(mnemonic, salt)
        if hashed == row["key_hash"]:
            # Upgrade legacy unsalted key to PBKDF2 on successful match
            if salt is None:
                new_salt = generate_salt()
                new_hash = hash_key(mnemonic, new_salt)
                db.execute(
                    "UPDATE api_keys SET key_hash = ?, salt = ? WHERE id = ?",
                    (new_hash, new_salt, row["key_id"]),
                )
            # Update last_used_at
            db.execute(
                "UPDATE api_keys SET last_used_at = ? WHERE id = ?",
                (datetime.now().isoformat(), row["key_id"]),
            )
            db.get_connection().commit()
            return {"id": row["user_id"], "name": row["name"]}

    return None


def list_api_keys(
    db: DatabaseManager,
    user_id: str,
) -> list[dict[str, Any]]:
    """List API keys for a user (prefix + label + last_used, no plaintext)."""
    rows = db.fetchall(
        "SELECT id, key_prefix, label, created_at, last_used_at, is_active "
        "FROM api_keys WHERE user_id = ? ORDER BY created_at",
        (user_id,),
    )
    return [dict(r) for r in rows]


def revoke_api_key(db: DatabaseManager, key_id: str) -> bool:
    """Revoke an API key by setting is_active=0.

    Returns True if the key was found and revoked.
    """
    cursor = db.execute(
        "UPDATE api_keys SET is_active = 0 WHERE id = ? AND is_active = 1",
        (key_id,),
    )
    db.get_connection().commit()
    revoked: int = cursor.rowcount
    if revoked:
        logger.info("Revoked API key %s", key_id[:8])
    return revoked > 0
