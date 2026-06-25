"""Item-as-star analytics endpoints over the materialised ``item_stars`` table.

Mirrors the A-axis filters of the line-items API but serves the aggregated
"item sky" surface: a filtered scatter/grid of items, average comparable unit
price grouped along any axis, price trends across vendors, and basket
composition. All endpoints route through :mod:`alibi.services.item_stars`.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from alibi.api.deps import get_database, require_user
from alibi.db.connection import DatabaseManager
from alibi.services import (
    avg_comparable_price as svc_avg_comparable_price,
    basket_composition as svc_basket_composition,
    list_attribute_facets as svc_list_attribute_facets,
    list_item_stars as svc_list_item_stars,
    price_by_state as svc_price_by_state,
    price_trend as svc_price_trend,
    rebuild_item_stars as svc_rebuild_item_stars,
)

router = APIRouter()


def _parse_attr(attr: Optional[str]) -> dict[str, Any]:
    """Parse an ``attr`` query string into a facet filter dict.

    Format: comma-separated ``key:value`` pairs, e.g.
    ``organic:true,size:L,free_range`` -> {"organic": True, "size": "L",
    "free_range": None}. A bare key (no value) means "facet present". Values
    "true"/"false" become booleans and numerics become numbers.
    """
    out: dict[str, Any] = {}
    if not attr:
        return out
    for part in attr.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            key, raw = part.split(":", 1)
            key, raw = key.strip(), raw.strip()
            low = raw.lower()
            value: Any
            if low in ("true", "false"):
                value = low == "true"
            else:
                try:
                    value = float(raw) if ("." in raw) else int(raw)
                except ValueError:
                    value = raw
            out[key] = value
        else:
            out[part] = None  # facet present
    return out


def _collect_filters(
    name: Optional[str],
    comparable_name: Optional[str],
    category_path: Optional[str],
    vendor: Optional[str],
    vendor_key: Optional[str],
    country: Optional[str],
    currency: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    price_min: Optional[float],
    price_max: Optional[float],
    attr: Optional[str] = None,
) -> dict[str, Any]:
    """Collect non-empty A-axis query params into a filter dict."""
    filters: dict[str, Any] = {}
    for key, value in (
        ("name", name),
        ("comparable_name", comparable_name),
        ("category_path", category_path),
        ("vendor", vendor),
        ("vendor_key", vendor_key),
        ("country", country),
        ("currency", currency),
        ("date_from", date_from),
        ("date_to", date_to),
        ("price_min", price_min),
        ("price_max", price_max),
    ):
        if value is not None and value != "":
            filters[key] = value
    attrs = _parse_attr(attr)
    if attrs:
        filters["attributes"] = attrs
    return filters


@router.get("")
async def list_stars(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    name: Optional[str] = Query(None),
    comparable_name: Optional[str] = Query(None),
    category_path: Optional[str] = Query(None),
    vendor: Optional[str] = Query(None),
    vendor_key: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    price_min: Optional[float] = Query(None),
    price_max: Optional[float] = Query(None),
    attr: Optional[str] = Query(
        None, description="Facet filter, e.g. organic:true,size:L"
    ),
    limit: int = Query(2000, ge=1, le=20000),
) -> list[dict[str, Any]]:
    """List item stars matching the A-axis filters (the 'item sky' points)."""
    filters = _collect_filters(
        name,
        comparable_name,
        category_path,
        vendor,
        vendor_key,
        country,
        currency,
        date_from,
        date_to,
        price_min,
        price_max,
        attr,
    )
    try:
        return svc_list_item_stars(db, filters, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/avg-price")
async def avg_price(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    group_by: str = Query("comparable_name"),
    name: Optional[str] = Query(None),
    comparable_name: Optional[str] = Query(None),
    category_path: Optional[str] = Query(None),
    vendor: Optional[str] = Query(None),
    vendor_key: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    price_min: Optional[float] = Query(None),
    price_max: Optional[float] = Query(None),
    attr: Optional[str] = Query(None, description="Facet filter, e.g. organic:true"),
) -> list[dict[str, Any]]:
    """Average comparable unit price grouped along the requested dimensions.

    ``group_by`` is a comma-separated list of: comparable_name, comparable_unit,
    product_variant, vendor, vendor_key, brand, category, category_path,
    currency, country, period buckets year / month / quarter, and any facet via
    ``attr:<key>`` (e.g. ``attr:size``). comparable_unit is always included so
    units never blend.
    """
    filters = _collect_filters(
        name,
        comparable_name,
        category_path,
        vendor,
        vendor_key,
        country,
        currency,
        date_from,
        date_to,
        price_min,
        price_max,
        attr,
    )
    dims = [d.strip() for d in group_by.split(",") if d.strip()]
    try:
        return svc_avg_comparable_price(db, filters, dims)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/price-by-state")
async def price_by_state(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    min_states: int = Query(
        2,
        ge=2,
        description="Only products appearing in at least this many distinct states",
    ),
    name: Optional[str] = Query(None),
    comparable_name: Optional[str] = Query(None),
    category_path: Optional[str] = Query(None),
    vendor: Optional[str] = Query(None),
    vendor_key: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    price_min: Optional[float] = Query(None),
    price_max: Optional[float] = Query(None),
    attr: Optional[str] = Query(None, description="Facet filter, e.g. organic:true"),
) -> list[dict[str, Any]]:
    """Comparable unit price by product STATE, within a product (the #58 facet).

    For each comparable product sold in at least ``min_states`` distinct states
    (fresh / frozen / canned / dried / cured / pickled / roasted / cooked), the
    normalised price per state -- fresh vs canned vs frozen, side by side. Scoped
    to genuine comparisons (a product seen in only one state is omitted), grouped
    within ``comparable_unit`` so EUR/kg never blends with EUR/L or EUR/pcs.
    """
    filters = _collect_filters(
        name,
        comparable_name,
        category_path,
        vendor,
        vendor_key,
        country,
        currency,
        date_from,
        date_to,
        price_min,
        price_max,
        attr,
    )
    return svc_price_by_state(db, filters, min_states=min_states)


@router.get("/trend")
async def trend(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    comparable_name: str = Query(..., description="Product to trend"),
    period: str = Query("month"),
    by_vendor: bool = Query(True),
    category_path: Optional[str] = Query(None),
    vendor: Optional[str] = Query(None),
    vendor_key: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    attr: Optional[str] = Query(None, description="Facet filter, e.g. organic:true"),
) -> list[dict[str, Any]]:
    """Comparable unit-price trend for one product over time across vendors."""
    filters = _collect_filters(
        None,
        None,
        category_path,
        vendor,
        vendor_key,
        country,
        currency,
        date_from,
        date_to,
        None,
        None,
        attr,
    )
    try:
        return svc_price_trend(
            db, comparable_name, filters, period=period, by_vendor=by_vendor
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/basket")
async def basket(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    by: str = Query("category"),
    name: Optional[str] = Query(None),
    comparable_name: Optional[str] = Query(None),
    category_path: Optional[str] = Query(None),
    vendor: Optional[str] = Query(None),
    vendor_key: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    attr: Optional[str] = Query(None, description="Facet filter, e.g. organic:true"),
) -> list[dict[str, Any]]:
    """Basket composition: spend grouped by a categorical axis."""
    filters = _collect_filters(
        name,
        comparable_name,
        category_path,
        vendor,
        vendor_key,
        country,
        currency,
        date_from,
        date_to,
        None,
        None,
        attr,
    )
    try:
        return svc_basket_composition(db, filters, by=by)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/facets")
async def facets(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
    name: Optional[str] = Query(None),
    comparable_name: Optional[str] = Query(None),
    category_path: Optional[str] = Query(None),
    vendor: Optional[str] = Query(None),
    country: Optional[str] = Query(None),
    currency: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    attr: Optional[str] = Query(None),
) -> dict[str, Any]:
    """Discover the attribute facets present (key -> values + counts).

    Powers the Item Sky facet chips. Honours the current A-axis / facet filters
    so the offered facets reflect the visible items.
    """
    filters = _collect_filters(
        name,
        comparable_name,
        category_path,
        vendor,
        None,
        country,
        currency,
        date_from,
        date_to,
        None,
        None,
        attr,
    )
    try:
        return svc_list_attribute_facets(db, filters)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/rebuild")
async def rebuild(
    db: Annotated[DatabaseManager, Depends(get_database)],
    user: Annotated[dict[str, Any], Depends(require_user)],
) -> dict[str, Any]:
    """Fully rebuild item_stars from fact_items + facts (drift safety net)."""
    count = svc_rebuild_item_stars(db)
    return {"rebuilt": count}
