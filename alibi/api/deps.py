"""Dependency injection for FastAPI endpoints."""

from __future__ import annotations

import hmac
from typing import Annotated, Any, Optional

from fastapi import Depends, Header, HTTPException, Query

from alibi.config import Config, get_config
from alibi.db.connection import DatabaseManager


_db_instance: DatabaseManager | None = None


def get_database() -> DatabaseManager:
    """Get a singleton initialized DatabaseManager instance.

    Reuses the same connection across requests (SQLite check_same_thread=False).
    """
    global _db_instance
    if _db_instance is None:
        config = get_config()
        _db_instance = DatabaseManager(config)
        if not _db_instance.is_initialized():
            _db_instance.initialize()
    return _db_instance


def reset_database() -> None:
    """Reset the singleton DB (for testing)."""
    global _db_instance
    if _db_instance is not None:
        _db_instance.close()
        _db_instance = None


def get_config_dep() -> Config:
    """Get config for dependency injection (overridable in tests)."""
    return get_config()


async def get_current_user(
    x_api_key: Annotated[Optional[str], Header()] = None,
    db: DatabaseManager = Depends(get_database),
    config: Config = Depends(get_config_dep),
) -> dict[str, Any] | None:
    """Extract current user from API key header.

    Auth modes (checked in order):
    1. No key configured + no key sent -> system user (single-user mode)
    2. Legacy ALIBI_API_KEY match -> system user (backward compat)
    3. Mnemonic key -> validate via api_keys table -> return user or 401
    """
    # No API key configured and none sent = single-user mode
    if not config.api_key and not x_api_key:
        return _get_default_user(db)

    # No key sent but one is required
    if not x_api_key:
        return None

    # Legacy API key match
    if config.api_key and hmac.compare_digest(config.api_key, x_api_key):
        return _get_default_user(db)

    # Try mnemonic API key validation
    from alibi.services.auth import validate_api_key

    user = validate_api_key(db, x_api_key)
    if user:
        return user

    # Key provided but invalid
    raise HTTPException(status_code=401, detail="Invalid API key")


def _get_default_user(db: DatabaseManager) -> dict[str, Any]:
    """Get or create the default user."""
    row = db.fetchone("SELECT id, name FROM users LIMIT 1")
    if row:
        return {"id": row["id"], "name": row["name"]}
    # Create default user if none exists
    db.execute(
        "INSERT OR IGNORE INTO users (id, name) VALUES (?, ?)",
        ("default", None),
    )
    db.get_connection().commit()
    return {"id": "default", "name": None}


async def require_user(
    user: Annotated[Optional[dict[str, Any]], Depends(get_current_user)],
) -> dict[str, Any]:
    """Require an authenticated user."""
    if user is None:
        raise HTTPException(status_code=401, detail="API key required")
    return user


class PaginationParams:
    """Standard pagination parameters."""

    def __init__(
        self,
        page: int = Query(1, ge=1, description="Page number"),
        per_page: int = Query(50, ge=1, le=200, description="Items per page"),
    ):
        self.page = page
        self.per_page = per_page
        self.offset = (page - 1) * per_page


def paginate(items: list[Any], params: PaginationParams) -> dict[str, Any]:
    """Apply pagination to a list and return paginated response."""
    total = len(items)
    start = params.offset
    end = start + params.per_page
    page_items = items[start:end]

    return {
        "items": page_items,
        "total": total,
        "page": params.page,
        "per_page": params.per_page,
        "pages": (total + params.per_page - 1) // params.per_page if total > 0 else 0,
    }
