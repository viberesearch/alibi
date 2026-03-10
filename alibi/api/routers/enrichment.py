"""Enrichment review and trigger API endpoints."""

from __future__ import annotations

import logging
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from alibi.api.deps import get_database, require_user
from alibi.db.connection import DatabaseManager
from alibi.services.enrichment_review import (
    confirm_enrichment as svc_confirm,
    get_enrichment_trends as svc_trends,
    get_review_queue as svc_queue,
    get_review_stats as svc_stats,
    get_vendor_coverage as svc_coverage,
    reject_enrichment as svc_reject,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/enrichment", tags=["enrichment"])


@router.get("/review", response_model=list[dict[str, Any]])
async def review_queue(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    threshold: float = Query(0.8, ge=0.0, le=1.0, description="Confidence threshold"),
    limit: int = Query(50, ge=1, le=500, description="Maximum items to return"),
) -> list[dict[str, Any]]:
    """Get fact items with enrichment confidence below threshold.

    Items are ordered worst-first (lowest confidence at the top).
    """
    return svc_queue(db, threshold=threshold, limit=limit)


class ConfirmBody(BaseModel):
    brand: Optional[str] = None
    category: Optional[str] = None


@router.post("/review/{fact_item_id}/confirm")
async def confirm_item_enrichment(
    fact_item_id: str,
    body: ConfirmBody,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Confirm (and optionally correct) the enrichment for a fact item.

    Sets enrichment_source='user_confirmed' and enrichment_confidence=1.0.
    Provide brand/category in the request body to overwrite the current values.
    """
    ok = svc_confirm(db, fact_item_id, brand=body.brand, category=body.category)
    if not ok:
        raise HTTPException(status_code=404, detail="Fact item not found")
    return {"status": "confirmed", "fact_item_id": fact_item_id}


@router.post("/review/{fact_item_id}/reject")
async def reject_item_enrichment(
    fact_item_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Reject the enrichment for a fact item.

    Clears brand, category, enrichment_source and enrichment_confidence back
    to NULL so the item can be re-enriched or left unenriched.
    """
    ok = svc_reject(db, fact_item_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Fact item not found")
    return {"status": "rejected", "fact_item_id": fact_item_id}


@router.get("/stats", response_model=dict[str, Any])
async def enrichment_stats(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Return enrichment statistics grouped by source with average confidence."""
    return svc_stats(db)


@router.get("/analytics/trends", response_model=dict[str, Any])
async def enrichment_trends(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    start_date: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    period: str = Query("month", description="Grouping period: day, week, month"),
) -> dict[str, Any]:
    """Return enrichment activity over time grouped by source.

    Results are bucketed by the requested period granularity and include a
    per-source breakdown for each bucket.
    """
    return svc_trends(db, start_date=start_date, end_date=end_date, period=period)


@router.get("/analytics/coverage", response_model=dict[str, Any])
async def vendor_coverage(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """Return enrichment coverage percentage per vendor.

    Vendors are ordered by total item count descending.  Each entry includes
    the per-source breakdown so callers can see which enrichment sources
    contributed to a vendor's coverage.
    """
    return svc_coverage(db, limit=limit)


# -- Enrichment trigger endpoints ------------------------------------------


@router.post("/run/cloud")
async def run_cloud_enrichment(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    """Trigger cloud-based enrichment (Anthropic API) for unenriched items.

    Requires ALIBI_CLOUD_ENRICHMENT_ENABLED=true and ALIBI_ANTHROPIC_API_KEY.
    """
    from alibi.enrichment.cloud_enrichment import enrich_pending_by_cloud

    results = enrich_pending_by_cloud(db, limit=limit)
    enriched = sum(1 for r in results if r.success)
    return {
        "source": "cloud_api",
        "processed": len(results),
        "enriched": enriched,
    }


@router.post("/run/llm")
async def run_llm_enrichment(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    """Trigger local LLM enrichment (Ollama) for unenriched items.

    Uses the configured Ollama model (default: qwen3:8b).
    """
    from alibi.enrichment.llm_enrichment import enrich_pending_by_llm

    results = enrich_pending_by_llm(db, limit=limit)
    enriched = sum(1 for r in results if r.success)
    return {
        "source": "llm_inference",
        "processed": len(results),
        "enriched": enriched,
    }


@router.post("/run/refine")
async def run_refine_enrichment(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    limit: int = Query(100, ge=1, le=1000),
) -> dict[str, Any]:
    """Trigger cloud category refinement for LLM-inferred items.

    Sends items with enrichment_source='llm_inference' to Claude Sonnet
    for category verification. Only corrects items where cloud disagrees
    with the local LLM assignment.

    Requires ALIBI_ANTHROPIC_API_KEY. Uses the configured
    cloud_enrichment_model (default: claude-sonnet-4-6).
    """
    from alibi.enrichment.cloud_enrichment import refine_categories_by_cloud

    results = refine_categories_by_cloud(db, limit=limit)
    return {
        "source": "cloud_refined",
        "processed": limit,
        "corrected": len(results),
    }


@router.post("/run/gemini")
async def run_gemini_enrichment(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    limit: int = Query(default=500, le=2000),
) -> dict[str, Any]:
    """Trigger Gemini mega-batch enrichment for unenriched items."""
    from alibi.enrichment.gemini_enrichment import enrich_pending_by_gemini

    results = enrich_pending_by_gemini(db, limit=limit)
    return {
        "total": len(results),
        "enriched": sum(1 for r in results if r.success),
        "with_unit_quantity": sum(1 for r in results if r.unit_quantity is not None),
    }


@router.get("/product-matches")
async def product_matches(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    category: str | None = Query(None, description="Filter by category"),
    limit: int = Query(200, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Find cross-vendor product matches via Gemini."""
    from alibi.services import find_product_matches

    groups = find_product_matches(db, category=category, limit=limit)
    return [
        {
            "canonical_name": g.canonical_name,
            "confidence": g.confidence,
            "reasoning": g.reasoning,
            "products": [
                {
                    "item_id": p.item_id,
                    "name": p.name,
                    "vendor": p.vendor_name,
                    "brand": p.brand,
                    "barcode": p.barcode,
                }
                for p in g.products
            ],
        }
        for g in groups
    ]
