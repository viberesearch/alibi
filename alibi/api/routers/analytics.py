"""Analytics API endpoints.

Provides REST endpoints for:
- Spending summaries (by month or vendor)
- Subscription detection
- Anomaly detection
- Vendor deduplication report
"""

from __future__ import annotations

import dataclasses
from datetime import date
from decimal import Decimal
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, Query

from alibi.api.deps import get_database, require_user
from alibi.db.connection import DatabaseManager
from alibi.services import analytics

router = APIRouter()


def _convert_types(obj: Any) -> Any:
    """Recursively convert Decimal and date to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _convert_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_types(i) for i in obj]
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, date):
        return obj.isoformat()
    return obj


def _serialize(obj: Any) -> Any:
    """Convert a dataclass (or list of dataclasses) to JSON-serializable form."""
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _convert_types(dataclasses.asdict(obj))
    return _convert_types(obj)


@router.get("/spending")
async def spending(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    period: str = Query("month", pattern="^(month|vendor)$"),
    date_from: Optional[str] = Query(None, description="Since date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Until date (YYYY-MM-DD)"),
    limit: int = Query(50, ge=1, le=500, description="Max vendors (vendor period)"),
) -> list[dict[str, Any]]:
    """Spending summary grouped by month or vendor."""
    filters: dict[str, Any] = {}
    if date_from:
        filters["date_from"] = date.fromisoformat(date_from)
    if date_to:
        filters["date_to"] = date.fromisoformat(date_to)
    if period == "vendor":
        filters["limit"] = limit

    result = analytics.spending_summary(db, period=period, filters=filters)
    serialized: list[dict[str, Any]] = _serialize(result)
    return serialized


@router.get("/subscriptions")
async def subscriptions(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> list[dict[str, Any]]:
    """Detect subscription and recurring payment patterns."""
    result = analytics.detect_subscriptions(db)
    serialized: list[dict[str, Any]] = _serialize(result)
    return serialized


@router.get("/anomalies")
async def anomalies(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    lookback_days: int = Query(90, ge=1, le=365),
    std_threshold: float = Query(2.0, ge=0.5, le=10.0),
) -> list[dict[str, Any]]:
    """Detect unusual spending patterns."""
    result = analytics.detect_anomalies(
        db,
        lookback_days=lookback_days,
        std_threshold=std_threshold,
    )
    serialized: list[dict[str, Any]] = _serialize(result)
    return serialized


@router.get("/vendors")
async def vendors(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Vendor alias / deduplication report."""
    result = analytics.vendor_report(db)
    serialized: dict[str, Any] = _serialize(result)
    return serialized


@router.post("/verify")
async def verify_extractions_endpoint(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    doc_ids: list[str] | None = None,
    limit: int = Query(20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """Cross-validate extracted documents via Gemini."""
    from alibi.services import verify_extractions as _verify

    results = _verify(db, doc_ids=doc_ids, limit=limit)
    return [
        {
            "doc_id": r.doc_id,
            "all_ok": r.all_ok,
            "issues": r.issues,
        }
        for r in results
    ]


@router.get("/corrections/matrix")
async def corrections_matrix(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    limit: int = Query(1000, ge=1, le=5000),
    min_count: int = Query(2, ge=1),
) -> dict[str, Any]:
    """Get correction confusion matrix."""
    from alibi.services import correction_confusion_matrix

    matrix = correction_confusion_matrix(db, limit=limit, min_count=min_count)
    return {
        "total_corrections": matrix.total_corrections,
        "top_corrected_fields": matrix.top_corrected_fields,
        "category_confusions": [
            {"original": c.original, "corrected": c.corrected, "count": c.count}
            for c in matrix.category_confusions
        ],
        "vendor_stats": [
            {
                "vendor_key": v.vendor_key,
                "vendor_name": v.vendor_name,
                "total_corrections": v.total_corrections,
                "field_corrections": v.field_corrections,
            }
            for v in matrix.vendor_stats
        ],
        "refinement_candidates": matrix.refinement_candidates,
    }


@router.get("/corrections/suggestions")
async def corrections_suggestions(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    limit: int = Query(1000, ge=1, le=5000),
) -> list[dict[str, Any]]:
    """Get actionable refinement suggestions from correction analysis."""
    from alibi.services import get_refinement_suggestions

    return get_refinement_suggestions(db, limit=limit)


@router.get("/location/spending")
async def location_spending_endpoint(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    cluster_radius: float = Query(100.0, ge=10.0, le=10000.0),
) -> list[dict[str, Any]]:
    """Spending aggregated by location."""
    from alibi.services import location_spending

    results = location_spending(db, cluster_radius_m=cluster_radius)
    serialized: list[dict[str, Any]] = _serialize(results)
    return serialized


@router.get("/location/branches")
async def vendor_branches_endpoint(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    vendor_key: Optional[str] = Query(None),
) -> list[dict[str, Any]]:
    """Compare vendor branches across locations."""
    from alibi.services import vendor_branches

    results = vendor_branches(db, vendor_key=vendor_key)
    serialized: list[dict[str, Any]] = _serialize(results)
    return serialized


@router.get("/location/nearby")
async def nearby_vendors_endpoint(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    lat: float = Query(..., description="Current latitude"),
    lng: float = Query(..., description="Current longitude"),
    radius: float = Query(2000.0, ge=100.0, le=50000.0),
    limit: int = Query(10, ge=1, le=50),
) -> list[dict[str, Any]]:
    """Suggest vendors near a location."""
    from alibi.services import nearby_vendors

    results = nearby_vendors(db, lat, lng, radius_m=radius, limit=limit)
    serialized: list[dict[str, Any]] = _serialize(results)
    return serialized
