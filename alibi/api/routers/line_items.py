"""Fact item (line item) query endpoints."""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from alibi.api.deps import (
    PaginationParams,
    get_database,
    require_user,
)
from alibi.db.connection import DatabaseManager
from alibi.services import (
    category_summary as svc_category_summary,
    get_fact_item as svc_get_fact_item,
    list_fact_items as svc_list_fact_items,
)

router = APIRouter()


@router.get("", response_model=dict[str, Any])
async def list_line_items(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    pagination: Annotated[PaginationParams, Depends()],
    category: Optional[str] = Query(None),
    name: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    fact_id: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Query fact items across all purchases with filters."""
    filters: dict[str, Any] = {}
    if category:
        filters["category"] = category
    if name:
        filters["name"] = name
    if brand:
        filters["brand"] = brand
    if fact_id:
        filters["fact_id"] = fact_id
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to

    return svc_list_fact_items(db, filters, pagination.offset, pagination.per_page)


# NOTE: /categories/summary MUST be declared before /{line_item_id}
# so FastAPI does not match "categories" as a line_item_id.
@router.get("/categories/summary")
async def category_summary(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
) -> list[dict[str, Any]]:
    """Get spending summary grouped by category."""
    filters: dict[str, Any] = {}
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to

    return svc_category_summary(db, filters)


@router.get("/{line_item_id}")
async def get_line_item(
    line_item_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Get a single fact item by ID."""
    item = svc_get_fact_item(db, line_item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Line item not found")
    return item


class LineItemUpdate(BaseModel):
    barcode: str | None = None
    brand: str | None = None
    category: str | None = None
    name: str | None = None
    unit_quantity: float | None = None
    unit: str | None = None
    product_variant: str | None = None


@router.patch("/{line_item_id}")
async def update_line_item(
    line_item_id: str,
    body: LineItemUpdate,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Update fields on a fact item (barcode, brand, category, name, unit_quantity, unit)."""
    from alibi.services import update_fact_item as svc_update_fact_item

    fields = body.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    try:
        ok = svc_update_fact_item(db, line_item_id, fields)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not ok:
        raise HTTPException(status_code=404, detail="Line item not found")

    # Return updated item
    from alibi.services import get_fact_item as svc_get_fact_item_inner

    updated = svc_get_fact_item_inner(db, line_item_id)
    assert updated is not None  # guaranteed by ok=True above
    return updated
