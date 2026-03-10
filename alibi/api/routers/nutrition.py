"""Nutritional analytics API endpoints.

Provides REST endpoints for:
- Nutrition summary aggregated by period
- Single fact_item nutrition lookup
- Nutrition breakdown by product category
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from alibi.api.deps import get_database, require_user
from alibi.db.connection import DatabaseManager

router = APIRouter()


@router.get("/summary")
async def nutrition_summary(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    period: str = Query("month", pattern="^(day|week|month)$"),
) -> dict[str, Any]:
    """Aggregate nutritional data from purchased items, grouped by period.

    Only items with a barcode that has a cached Open Food Facts entry contribute.
    Negative cache entries (products not found in OFF) are excluded.
    """
    from alibi.analytics.nutrition import nutrition_summary as _summary

    return _summary(db, start_date=start_date, end_date=end_date, period=period)


@router.get("/item/{fact_item_id}")
async def item_nutrition(
    fact_item_id: str,
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Get nutritional data for a single fact_item.

    Looks up the item's barcode in product_cache and returns per-100g nutriments
    plus computed totals when unit weight is available.

    Returns 404 if the fact_item does not exist.
    """
    from alibi.analytics.nutrition import item_nutrition as _item_nutrition

    result = _item_nutrition(db, fact_item_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Fact item not found")
    return result


@router.get("/by-category")
async def nutrition_by_category(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
) -> list[dict[str, Any]]:
    """Aggregate nutritional data grouped by product category.

    Categories are taken from the fact_item.category field (set by enrichment)
    or fall back to the OFF product categories string.  Results are sorted by
    item count descending.
    """
    from alibi.analytics.nutrition import nutrition_by_category as _by_category

    return _by_category(db, start_date=start_date, end_date=end_date)
