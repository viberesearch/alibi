"""User management API endpoints.

Provides REST endpoints for:
- User CRUD (create, list, get, update)
- Contact management (add, list, remove)
- API key lifecycle (create, list, revoke)
"""

from __future__ import annotations

import sqlite3
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from alibi.api.deps import get_database, require_user
from alibi.db.connection import DatabaseManager
from alibi.services import auth

router = APIRouter()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CreateUserRequest(BaseModel):
    """Request body for creating a user."""

    name: Optional[str] = None


class UpdateUserRequest(BaseModel):
    """Request body for updating a user."""

    name: Optional[str] = None


class AddContactRequest(BaseModel):
    """Request body for adding a contact."""

    contact_type: str
    value: str
    label: Optional[str] = None


class CreateKeyRequest(BaseModel):
    """Request body for creating an API key."""

    label: str = "default"


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------


@router.get("")
async def list_users_endpoint(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> list[dict[str, Any]]:
    """List all active users."""
    return auth.list_users(db)


@router.post("", status_code=201)
async def create_user_endpoint(
    request: CreateUserRequest,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Create a new user."""
    return auth.create_user(db, name=request.name)


@router.get("/{user_id}")
async def get_user_endpoint(
    user_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Get a user by ID, including their contacts."""
    found = auth.get_user(db, user_id)
    if not found:
        raise HTTPException(status_code=404, detail="User not found")
    contacts = auth.list_contacts(db, user_id)
    return {**found, "contacts": contacts}


@router.patch("/{user_id}")
async def update_user_endpoint(
    user_id: str,
    request: UpdateUserRequest,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Update a user's display name."""
    updated = auth.update_user(db, user_id, name=request.name)
    if not updated:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": user_id, "status": "updated"}


# ---------------------------------------------------------------------------
# Contact management
# ---------------------------------------------------------------------------


@router.get("/{user_id}/contacts")
async def list_contacts_endpoint(
    user_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> list[dict[str, Any]]:
    """List all contacts for a user."""
    found = auth.get_user(db, user_id)
    if not found:
        raise HTTPException(status_code=404, detail="User not found")
    return auth.list_contacts(db, user_id)


@router.post("/{user_id}/contacts", status_code=201)
async def add_contact_endpoint(
    user_id: str,
    request: AddContactRequest,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Add a contact (telegram, email, etc.) to a user."""
    try:
        result = auth.add_contact(
            db,
            user_id=user_id,
            contact_type=request.contact_type,
            value=request.value,
            label=request.label,
        )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Contact already linked to a user")
    if result is None:
        raise HTTPException(status_code=404, detail="User not found")
    return result


@router.delete("/{user_id}/contacts/{contact_id}", status_code=204)
async def remove_contact_endpoint(
    user_id: str,
    contact_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> None:
    """Remove a contact from a user."""
    deleted = auth.remove_contact(db, contact_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Contact not found")


# ---------------------------------------------------------------------------
# API key lifecycle
# ---------------------------------------------------------------------------


@router.get("/{user_id}/keys")
async def list_keys_endpoint(
    user_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> list[dict[str, Any]]:
    """List API keys for a user (prefix + label, no plaintext)."""
    found = auth.get_user(db, user_id)
    if not found:
        raise HTTPException(status_code=404, detail="User not found")
    return auth.list_api_keys(db, user_id)


@router.post("/{user_id}/keys", status_code=201)
async def create_key_endpoint(
    user_id: str,
    request: CreateKeyRequest,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Create a new API key for a user. The mnemonic is returned once."""
    found = auth.get_user(db, user_id)
    if not found:
        raise HTTPException(status_code=404, detail="User not found")
    return auth.create_api_key(db, user_id, label=request.label)


@router.delete("/{user_id}/keys/{key_id}", status_code=204)
async def revoke_key_endpoint(
    user_id: str,
    key_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> None:
    """Revoke an API key."""
    revoked = auth.revoke_api_key(db, key_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="API key not found")
