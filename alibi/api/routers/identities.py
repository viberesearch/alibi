"""Identity management API endpoints.

Provides REST endpoints for:
- Listing identities (optionally filtered by entity_type)
- Fetching a single identity with members
- Merging two vendor identities
- Resolving a vendor identity by name, key, or registration
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from alibi.api.deps import get_database, require_user
from alibi.db.connection import DatabaseManager
from alibi.services import identity

router = APIRouter()


class MergeRequest(BaseModel):
    """Request to merge two vendor identities."""

    identity_id_a: str
    identity_id_b: str


# NOTE: /resolve MUST be declared before /{identity_id} so FastAPI matches
# the literal path before the path-parameter catch-all.
@router.get("/resolve")
async def resolve_identity(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    vendor_name: Optional[str] = Query(None, description="Vendor name to resolve"),
    vendor_key: Optional[str] = Query(None, description="Vendor key (VAT number)"),
    registration: Optional[str] = Query(None, description="Registration number"),
) -> dict[str, Any]:
    """Resolve a vendor identity by any matching signal."""
    if not any([vendor_name, vendor_key, registration]):
        raise HTTPException(
            status_code=422,
            detail="At least one of vendor_name, vendor_key, or registration required",
        )

    result = identity.resolve_vendor(
        db,
        vendor_name=vendor_name,
        vendor_key=vendor_key,
        registration=registration,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Identity not found")
    return result


@router.get("")
async def list_identities(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    entity_type: Optional[str] = Query(
        None, description="Filter by entity type: 'vendor' or 'item'"
    ),
) -> list[dict[str, Any]]:
    """List all identities, optionally filtered by entity type."""
    return identity.list_identities(db, entity_type=entity_type)


@router.get("/{identity_id}")
async def get_identity(
    identity_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Get a single identity with its members."""
    result = identity.get_identity(db, identity_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"Identity not found: {identity_id}"
        )
    return result


@router.post("/merge")
async def merge_identities(
    request: MergeRequest,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Merge two vendor identities into one."""
    ok = identity.merge_vendors(
        db,
        identity_id_a=request.identity_id_a,
        identity_id_b=request.identity_id_b,
    )
    if not ok:
        raise HTTPException(
            status_code=404,
            detail=(
                f"One or both identities not found: "
                f"{request.identity_id_a}, {request.identity_id_b}"
            ),
        )
    return {"success": True, "merged_into": request.identity_id_a}
