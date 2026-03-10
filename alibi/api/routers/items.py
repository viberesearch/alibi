"""Item (asset) CRUD endpoints.

Delegates all business logic to the items service layer.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from alibi.api.deps import (
    PaginationParams,
    get_database,
    paginate,
    require_user,
)
from alibi.db.connection import DatabaseManager
from alibi.services.items import (
    create_item,
    delete_item,
    get_item,
    get_item_documents,
    get_item_facts,
    list_items,
    update_item,
)

router = APIRouter()


class ItemCreate(BaseModel):
    """Item creation request."""

    space_id: str = "default"
    name: str
    category: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None
    purchase_date: Optional[str] = None
    purchase_price: Optional[str] = None
    currency: str = "EUR"
    warranty_expires: Optional[str] = None
    warranty_type: Optional[str] = None


class ItemUpdate(BaseModel):
    """Item update request."""

    name: Optional[str] = None
    category: Optional[str] = None
    status: Optional[str] = None
    current_value: Optional[str] = None
    warranty_expires: Optional[str] = None


@router.get("", response_model=dict[str, Any])
async def list_items_endpoint(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    pagination: Annotated[PaginationParams, Depends()],
    category: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    warranty_expiring: Optional[bool] = Query(
        None, description="Filter items with warranties expiring in 30 days"
    ),
) -> dict[str, Any]:
    """List items with optional filters."""
    items = list_items(
        db,
        category=category,
        status=status,
        warranty_expiring=bool(warranty_expiring),
    )
    return paginate(items, pagination)


@router.get("/{item_id}")
async def get_item_endpoint(
    item_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Get a single item by ID."""
    item = get_item(db, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    item["documents"] = get_item_documents(db, item_id)
    item["facts"] = get_item_facts(db, item_id)
    return item


@router.post("", status_code=201)
async def create_item_endpoint(
    data: ItemCreate,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Create a new item."""
    item_data = data.model_dump()
    item_data["created_by"] = user["id"]
    item_id = create_item(db, item_data)
    return {"id": item_id, "status": "created"}


@router.patch("/{item_id}")
async def update_item_endpoint(
    item_id: str,
    data: ItemUpdate,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Update an item."""
    update_data = data.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")

    updated = update_item(db, item_id, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Item not found")

    return {"id": item_id, "status": "updated"}


@router.delete("/{item_id}", status_code=204)
async def delete_item_endpoint(
    item_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> None:
    """Delete an item."""
    deleted = delete_item(db, item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Item not found")
