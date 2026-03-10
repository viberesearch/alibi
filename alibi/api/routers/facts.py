"""Fact inspection and correction API endpoints.

Provides REST endpoints for:
- Listing and inspecting facts (collapsed clouds)
- Listing clouds and unassigned bundles
- Moving bundles between clouds (correction)
- Setting bundle cloud_id directly (user edit)
- Re-collapsing clouds and marking disputes
"""

from __future__ import annotations

from datetime import date
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from alibi.api.deps import PaginationParams, get_database, paginate, require_user
from alibi.db.connection import DatabaseManager


router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class FactSummary(BaseModel):
    """Fact list item."""

    id: str
    fact_type: Optional[str] = None
    vendor: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    event_date: Optional[str] = None
    status: Optional[str] = None
    cloud_id: Optional[str] = None


class AtomDetail(BaseModel):
    """Atom within a bundle."""

    id: str
    atom_type: str
    role: Optional[str] = None
    data: dict[str, Any]
    confidence: Optional[float] = None


class BundleDetail(BaseModel):
    """Bundle with atoms and source document."""

    id: str
    bundle_type: str
    match_type: Optional[str] = None
    match_confidence: Optional[float] = None
    document: dict[str, Any]
    atoms: list[AtomDetail]


class FactInspection(BaseModel):
    """Full fact drill-down."""

    fact: dict[str, Any]
    cloud: dict[str, Any]
    bundles: list[BundleDetail]
    items: list[dict[str, Any]]


class CloudSummary(BaseModel):
    """Cloud list item."""

    id: str
    status: Optional[str] = None
    confidence: Optional[float] = None
    bundle_count: int = 0
    fact_id: Optional[str] = None
    fact_vendor: Optional[str] = None
    total_amount: Optional[float] = None
    event_date: Optional[str] = None
    fact_status: Optional[str] = None


class UnassignedBundle(BaseModel):
    """Unassigned bundle."""

    id: str
    bundle_type: str
    document_id: Optional[str] = None
    file_path: Optional[str] = None


class MoveBundleRequest(BaseModel):
    """Request to move a bundle to a cloud."""

    bundle_id: str
    target_cloud_id: Optional[str] = None  # None = create new cloud


class SetBundleCloudRequest(BaseModel):
    """Request to set bundle cloud_id directly."""

    bundle_id: str
    cloud_id: Optional[str] = None  # None = detach


class CorrectionResponse(BaseModel):
    """Response from a correction operation."""

    success: bool
    error: Optional[str] = None
    source_cloud_id: Optional[str] = None
    target_cloud_id: Optional[str] = None
    source_fact_id: Optional[str] = None
    target_fact_id: Optional[str] = None
    deleted_clouds: int = 0


class RecollapseResponse(BaseModel):
    """Response from recollapse."""

    success: bool
    fact_id: Optional[str] = None


class UpdateFactRequest(BaseModel):
    """Request to update fields on a fact."""

    vendor: Optional[str] = None
    amount: Optional[str] = None
    date: Optional[str] = None
    fact_type: Optional[str] = None
    vendor_key: Optional[str] = None


class CorrectVendorRequest(BaseModel):
    """Request to correct vendor name on a fact."""

    vendor: str


# ---------------------------------------------------------------------------
# List / query endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=dict[str, Any])
async def list_facts(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    pagination: Annotated[PaginationParams, Depends()],
    vendor: Optional[str] = Query(None, description="Filter by vendor name"),
    fact_type: Optional[str] = Query(None, description="Filter by fact type"),
    date_from: Optional[str] = Query(None, description="Since date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Until date (YYYY-MM-DD)"),
) -> dict[str, Any]:
    """List facts with optional filters and server-side pagination."""
    from alibi.services import query

    filters: dict[str, Any] = {}
    if vendor:
        filters["vendor"] = vendor
    if fact_type:
        filters["fact_type"] = fact_type
    if date_from:
        filters["date_from"] = date.fromisoformat(date_from)
    if date_to:
        filters["date_to"] = date.fromisoformat(date_to)

    result = query.list_facts(
        db,
        filters=filters,
        offset=pagination.offset,
        limit=pagination.per_page,
    )

    # Adapt service response to paginated API format
    facts = [_serialize_fact(f) for f in result["facts"]]
    total = result["total"]
    return {
        "items": facts,
        "total": total,
        "page": pagination.page,
        "per_page": pagination.per_page,
        "pages": (
            (total + pagination.per_page - 1) // pagination.per_page if total > 0 else 0
        ),
    }


# NOTE: Literal paths (/clouds, /unassigned, /move-bundle) MUST be declared
# before /{fact_id} so FastAPI matches them before the path-param catch-all.


@router.get("/clouds", response_model=dict[str, Any])
async def list_clouds(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    pagination: Annotated[PaginationParams, Depends()],
    status: Optional[str] = Query(None, description="Filter by status"),
) -> dict[str, Any]:
    """List clouds with bundle count and fact summary."""
    from alibi.db import v2_store

    rows = v2_store.list_clouds(db, status=status)
    items = [dict(r) for r in rows]
    return paginate(items, pagination)


@router.get("/unassigned", response_model=list[UnassignedBundle])
async def list_unassigned(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> list[dict[str, Any]]:
    """List bundles with no cloud assignment."""
    from alibi.services import query

    return query.list_unassigned(db)


@router.get("/locations/recent", response_model=list[dict[str, Any]])
async def recent_vendor_locations(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    limit: int = Query(20, ge=1, le=100, description="Max locations to return"),
) -> list[dict[str, Any]]:
    """Get recent unique vendor+location pairs for location picker."""
    from alibi.services.correction import get_recent_vendor_locations

    return get_recent_vendor_locations(db, limit=limit)


@router.get("/{fact_id}", response_model=dict[str, Any])
async def get_fact(
    fact_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Get a single fact with its items."""
    from alibi.services import query

    result = query.get_fact(db, fact_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Fact not found: {fact_id}")
    return _serialize_fact(result)


@router.get("/{fact_id}/inspect", response_model=FactInspection)
async def inspect_fact(
    fact_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Inspect a fact: cloud, bundles, atoms, source documents."""
    from alibi.services import query

    result = query.inspect_fact(db, fact_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Fact not found: {fact_id}")
    return result


# ---------------------------------------------------------------------------
# Location endpoints
# ---------------------------------------------------------------------------


class SetLocationRequest(BaseModel):
    """Request body for setting a fact's location."""

    map_url: str


@router.get("/{fact_id}/location", response_model=dict[str, Any])
async def get_fact_location(
    fact_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Get the location annotation for a fact."""
    from alibi.services.correction import get_fact_location as _get_loc

    result = _get_loc(db, fact_id)
    if not result:
        raise HTTPException(status_code=404, detail="No location set for this fact")
    return result


@router.post("/{fact_id}/location", response_model=dict[str, Any])
async def set_fact_location(
    fact_id: str,
    request: SetLocationRequest,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Set or update the location for a fact from a Google Maps URL."""
    from alibi.services.correction import set_fact_location as _set_loc

    result = _set_loc(db, fact_id, request.map_url)
    if not result:
        raise HTTPException(
            status_code=400,
            detail="Could not parse map URL or fact not found",
        )
    return result


# ---------------------------------------------------------------------------
# Correction endpoints
# ---------------------------------------------------------------------------


@router.post("/move-bundle", response_model=CorrectionResponse)
async def move_bundle_endpoint(
    request: MoveBundleRequest,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Move a bundle to a different cloud (or create a new one)."""
    from alibi.services import correction

    result = correction.move_bundle(db, request.bundle_id, request.target_cloud_id)
    return {
        "success": result.success,
        "error": result.error,
        "source_cloud_id": result.source_cloud_id,
        "target_cloud_id": result.target_cloud_id,
        "source_fact_id": result.source_fact_id,
        "target_fact_id": result.target_fact_id,
        "deleted_clouds": result.deleted_clouds,
    }


@router.post("/set-cloud", response_model=dict[str, Any])
async def set_bundle_cloud(
    request: SetBundleCloudRequest,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Set the cloud_id on a bundle directly."""
    from alibi.db import v2_store

    ok = v2_store.set_bundle_cloud(db, request.bundle_id, request.cloud_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Bundle or cloud not found")
    return {
        "success": True,
        "bundle_id": request.bundle_id,
        "cloud_id": request.cloud_id,
    }


@router.post("/clouds/{cloud_id}/recollapse", response_model=RecollapseResponse)
async def recollapse_cloud_endpoint(
    cloud_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Force re-collapse a cloud into a fact."""
    from alibi.services import correction

    result = correction.recollapse_cloud(db, cloud_id)
    return {"success": True, "fact_id": result.target_fact_id}


@router.post("/clouds/{cloud_id}/dispute")
async def dispute_cloud(
    cloud_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Mark a cloud as disputed (needs human review)."""
    from alibi.services import correction

    result = correction.mark_disputed(db, cloud_id)
    return {"success": result.success, "cloud_id": cloud_id, "status": "disputed"}


@router.patch("/{fact_id}", response_model=dict[str, Any])
async def update_fact_endpoint(
    fact_id: str,
    request: UpdateFactRequest,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Update fields on a fact."""
    from alibi.services import correction

    fields = request.model_dump(exclude_none=True)
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")

    try:
        ok = correction.update_fact(db, fact_id, fields)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not ok:
        raise HTTPException(status_code=404, detail=f"Fact not found: {fact_id}")
    return {"id": fact_id, "status": "updated"}


@router.post("/{fact_id}/correct-vendor", response_model=dict[str, Any])
async def correct_vendor_endpoint(
    fact_id: str,
    request: CorrectVendorRequest,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Correct vendor name and teach identity system."""
    from alibi.services import correction

    ok = correction.correct_vendor(db, fact_id, request.vendor)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Fact not found: {fact_id}")
    return {"id": fact_id, "vendor": request.vendor, "status": "corrected"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_fact(row: dict[str, Any]) -> dict[str, Any]:
    """Convert DB row to serializable dict."""
    result = dict(row)
    if "total_amount" in result and result["total_amount"] is not None:
        result["total_amount"] = float(result["total_amount"])
    if "event_date" in result and result["event_date"] is not None:
        result["event_date"] = str(result["event_date"])
    return result
