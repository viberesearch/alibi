"""Item (asset) CRUD endpoints.

Delegates all business logic to the items service layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
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
    list_warranty_expiring,
    update_item,
)

router = APIRouter()

# Image types accepted by the barcode scan endpoint.
_BARCODE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
_MAX_SCAN_SIZE = 25 * 1024 * 1024  # 25MB


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


# NOTE: Literal paths (/warranty/..., /barcode/...) MUST be declared before
# /{item_id} so FastAPI matches them before the path-param catch-all.


@router.get("/warranty/expiring")
async def list_warranty_expiring_endpoint(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    ahead_days: int = Query(90, ge=1, le=365),
    expired_days: int = Query(30, ge=0, le=365),
) -> list[dict[str, Any]]:
    """List active items with warranties expiring soon (or recently expired)."""
    return list_warranty_expiring(db, ahead_days=ahead_days, expired_days=expired_days)


@router.post("/barcode/scan")
async def scan_barcode_endpoint(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Detect barcodes in an uploaded image, with any cached product matches."""
    from alibi.services import barcode as barcode_svc

    if not barcode_svc.has_support():
        raise HTTPException(
            status_code=503,
            detail="Barcode scanning unavailable (pyzbar not installed)",
        )

    ext = Path(file.filename or "").suffix.lower()
    if ext and ext not in _BARCODE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {ext}")

    data = await file.read()
    if len(data) > _MAX_SCAN_SIZE:
        raise HTTPException(status_code=413, detail="Image too large (max 25MB)")

    scanned = barcode_svc.scan_image(db, data)
    return {
        "count": len(scanned),
        "barcodes": [
            {
                "data": s.data,
                "type": s.type,
                "valid_ean": s.valid_ean,
                "product": s.product,
            }
            for s in scanned
        ],
    }


@router.get("/barcode/{barcode}")
async def barcode_lookup_endpoint(
    barcode: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Look up a barcode in Open Food Facts (falls back to local cache)."""
    from alibi.services import barcode as barcode_svc

    product = barcode_svc.lookup_off_product(barcode)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    return product


@router.post("/barcode/{barcode}/enrich")
async def enrich_by_barcode_endpoint(
    barcode: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Enrich all unenriched fact_items carrying this barcode."""
    from alibi.services import barcode as barcode_svc

    result = barcode_svc.enrich_items_by_barcode(db, barcode)
    return {"barcode": barcode, **result}


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
