"""Nutritional analytics from purchased items with Open Food Facts data.

Joins fact_items (with barcode) against the product_cache table to aggregate
nutrimental data across periods.  Only items that have a cached OFF product
(non-negative cache entry) contribute to the totals.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

# OFF nutriment key → our canonical name
_NUTRIMENT_MAP: dict[str, str] = {
    "energy-kcal_100g": "energy_kcal",
    "fat_100g": "fat_g",
    "saturated-fat_100g": "saturated_fat_g",
    "carbohydrates_100g": "carbohydrates_g",
    "sugars_100g": "sugars_g",
    "fiber_100g": "fiber_g",
    "proteins_100g": "proteins_g",
    "salt_100g": "salt_g",
}

_EMPTY_TOTALS: dict[str, float] = {
    "energy_kcal": 0.0,
    "fat_g": 0.0,
    "saturated_fat_g": 0.0,
    "carbohydrates_g": 0.0,
    "sugars_g": 0.0,
    "fiber_g": 0.0,
    "proteins_g": 0.0,
    "salt_g": 0.0,
}

_PERIOD_FMT: dict[str, str] = {
    "day": "%Y-%m-%d",
    "week": "%Y-W%W",
    "month": "%Y-%m",
}


def _period_key(event_date: str | date, period: str) -> str:
    """Convert an ISO date string (or date object) to a period bucket string."""
    if isinstance(event_date, str):
        d = date.fromisoformat(event_date)
    else:
        d = event_date
    fmt = _PERIOD_FMT.get(period, "%Y-%m")
    return d.strftime(fmt)


def _parse_nutriments(data: dict[str, Any]) -> dict[str, float]:
    """Extract canonical nutriment values (per 100 g) from an OFF product dict."""
    raw = data.get("nutriments") or {}
    result: dict[str, float] = {}
    for off_key, our_key in _NUTRIMENT_MAP.items():
        val = raw.get(off_key)
        if val is not None:
            try:
                result[our_key] = float(val)
            except (TypeError, ValueError):
                pass
    return result


def _fetch_nutrition_rows(
    db: "DatabaseManager",
    start_date: str | None,
    end_date: str | None,
) -> list[Any]:
    """Run the main join query and return raw rows.

    Each row contains: fact_item id, name, barcode, category, quantity,
    unit_quantity, fact event_date, and the raw product_cache data JSON.
    """
    conditions: list[str] = ["pc.data NOT LIKE '%\"_not_found\": true%'"]
    params: list[Any] = []

    if start_date:
        conditions.append("f.event_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("f.event_date <= ?")
        params.append(end_date)

    where = " AND ".join(conditions)

    sql = f"""
SELECT
    fi.id          AS item_id,
    fi.name        AS item_name,
    fi.barcode     AS barcode,
    fi.category    AS category,
    fi.quantity    AS quantity,
    fi.unit_quantity AS unit_quantity,
    f.event_date   AS event_date,
    pc.data        AS product_data
FROM fact_items fi
JOIN facts f         ON fi.fact_id = f.id
JOIN product_cache pc ON fi.barcode = pc.barcode
WHERE fi.barcode IS NOT NULL
  AND {where}
ORDER BY f.event_date
"""
    return db.fetchall(sql, tuple(params))


def _compute_item_nutrition(
    nutriments_100g: dict[str, float],
    unit_quantity: float | None,
    quantity: float | None,
) -> dict[str, float] | None:
    """Compute total nutriment contribution for a single line item.

    If unit_quantity (grams per unit) is known, multiply by purchase quantity.
    Otherwise return None to signal that we can only report per-100g figures.
    """
    if not nutriments_100g:
        return None

    # If we have unit weight in grams, compute totals
    if unit_quantity and unit_quantity > 0:
        qty = quantity if (quantity and quantity > 0) else 1.0
        factor = qty * unit_quantity / 100.0
        return {k: round(v * factor, 3) for k, v in nutriments_100g.items()}

    return None  # cannot compute absolute totals without weight


def nutrition_summary(
    db: "DatabaseManager",
    start_date: str | None = None,
    end_date: str | None = None,
    period: str = "month",
) -> dict[str, Any]:
    """Aggregate nutritional data from purchased items with cached OFF data.

    Joins fact_items (barcode) -> product_cache (data JSON with nutriments).
    Groups by period.

    Args:
        db: Database manager
        start_date: ISO date string filter (inclusive)
        end_date: ISO date string filter (inclusive)
        period: Grouping period — "day", "week", or "month"

    Returns:
        Dict with "periods", "top_sugar_items", "top_calorie_items".
    """
    rows = _fetch_nutrition_rows(db, start_date, end_date)

    # Also count total items per period (including those without barcode / cache)
    total_conditions: list[str] = []
    total_params: list[Any] = []
    if start_date:
        total_conditions.append("f.event_date >= ?")
        total_params.append(start_date)
    if end_date:
        total_conditions.append("f.event_date <= ?")
        total_params.append(end_date)

    total_where = "WHERE " + " AND ".join(total_conditions) if total_conditions else ""
    total_sql = f"""
SELECT f.event_date, COUNT(*) AS cnt
FROM fact_items fi
JOIN facts f ON fi.fact_id = f.id
{total_where}
GROUP BY f.event_date
"""
    total_rows = db.fetchall(total_sql, tuple(total_params))

    # Build total-items-per-period map
    total_by_period: dict[str, int] = defaultdict(int)
    for tr in total_rows:
        pk = _period_key(tr["event_date"], period)
        total_by_period[pk] += tr["cnt"]

    # Aggregate per period
    period_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: dict(_EMPTY_TOTALS)
    )
    period_nutriscore: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    period_items_with_nutrition: dict[str, int] = defaultdict(int)

    # For top-N tracking
    item_sugar: list[dict[str, Any]] = []
    item_calorie: list[dict[str, Any]] = []

    for row in rows:
        try:
            product = json.loads(row["product_data"])
        except (json.JSONDecodeError, TypeError):
            logger.debug(
                "Skipping unparseable product_cache entry for %s", row["barcode"]
            )
            continue

        if product.get("_not_found"):
            continue

        pk = _period_key(row["event_date"], period)
        period_items_with_nutrition[pk] += 1

        nutriments = _parse_nutriments(product)
        if not nutriments:
            # Count as having nutrition entry but no numeric data
            continue

        # Nutriscore
        grade = (product.get("nutriscore_grade") or "").lower()
        if grade in ("a", "b", "c", "d", "e"):
            period_nutriscore[pk][grade] += 1

        # Compute absolute totals if possible
        totals_for_item = _compute_item_nutrition(
            nutriments,
            row["unit_quantity"],
            row["quantity"],
        )

        if totals_for_item:
            for key, val in totals_for_item.items():
                period_totals[pk][key] = round(period_totals[pk][key] + val, 3)

            # Track for top-N
            item_calorie.append(
                {
                    "item_id": row["item_id"],
                    "name": row["item_name"] or "",
                    "barcode": row["barcode"],
                    "energy_kcal": totals_for_item.get("energy_kcal", 0.0),
                    "sugars_g": totals_for_item.get("sugars_g", 0.0),
                }
            )
            item_sugar.append(
                {
                    "item_id": row["item_id"],
                    "name": row["item_name"] or "",
                    "barcode": row["barcode"],
                    "energy_kcal": totals_for_item.get("energy_kcal", 0.0),
                    "sugars_g": totals_for_item.get("sugars_g", 0.0),
                }
            )

    # Build period list (union of all periods that appear in any bucket)
    all_periods: set[str] = set(total_by_period.keys()) | set(
        period_items_with_nutrition.keys()
    )

    period_list: list[dict[str, Any]] = []
    for pk in sorted(all_periods):
        items_with = period_items_with_nutrition.get(pk, 0)
        items_total = total_by_period.get(pk, 0)
        coverage = round(items_with / items_total * 100.0, 1) if items_total else 0.0

        nutriscore_dist: dict[str, int] = {}
        raw_ns = period_nutriscore.get(pk)
        if raw_ns:
            nutriscore_dist = dict(raw_ns)

        period_list.append(
            {
                "period": pk,
                "items_with_nutrition": items_with,
                "items_total": items_total,
                "coverage_pct": coverage,
                "totals": dict(period_totals.get(pk, _EMPTY_TOTALS)),
                "nutriscore_distribution": nutriscore_dist,
            }
        )

    # Top-10 by sugar / calorie (across all periods)
    top_sugar = sorted(item_sugar, key=lambda x: x["sugars_g"], reverse=True)[:10]
    top_calorie = sorted(item_calorie, key=lambda x: x["energy_kcal"], reverse=True)[
        :10
    ]

    return {
        "periods": period_list,
        "top_sugar_items": top_sugar,
        "top_calorie_items": top_calorie,
    }


def item_nutrition(
    db: "DatabaseManager",
    fact_item_id: str,
) -> dict[str, Any] | None:
    """Get nutritional data for a single fact_item by looking up its barcode in product_cache.

    Args:
        db: Database manager
        fact_item_id: The fact_items.id to look up

    Returns:
        Dict with item info, per-100g nutriments, computed totals (when possible),
        and nutriscore; or None if barcode not in cache or no nutritional data.
    """
    row = db.fetchone(
        """
SELECT
    fi.id          AS item_id,
    fi.name        AS item_name,
    fi.barcode     AS barcode,
    fi.category    AS category,
    fi.brand       AS brand,
    fi.quantity    AS quantity,
    fi.unit_quantity AS unit_quantity,
    pc.data        AS product_data
FROM fact_items fi
LEFT JOIN product_cache pc ON fi.barcode = pc.barcode
WHERE fi.id = ?
""",
        (fact_item_id,),
    )

    if not row:
        return None

    if not row["barcode"]:
        return {
            "item_id": fact_item_id,
            "item_name": row["item_name"],
            "barcode": None,
            "error": "no_barcode",
        }

    if not row["product_data"]:
        return {
            "item_id": fact_item_id,
            "item_name": row["item_name"],
            "barcode": row["barcode"],
            "error": "not_cached",
        }

    try:
        product = json.loads(row["product_data"])
    except (json.JSONDecodeError, TypeError):
        return None

    if product.get("_not_found"):
        return {
            "item_id": fact_item_id,
            "item_name": row["item_name"],
            "barcode": row["barcode"],
            "error": "not_found_in_off",
        }

    nutriments_100g = _parse_nutriments(product)
    totals = _compute_item_nutrition(
        nutriments_100g,
        row["unit_quantity"],
        row["quantity"],
    )

    return {
        "item_id": fact_item_id,
        "item_name": row["item_name"],
        "barcode": row["barcode"],
        "brand": row["brand"],
        "category": row["category"],
        "product_name": product.get("product_name"),
        "product_quantity": product.get("quantity"),
        "nutriscore_grade": (product.get("nutriscore_grade") or "").lower() or None,
        "nutriments_per_100g": nutriments_100g,
        "totals": totals,
    }


def nutrition_by_category(
    db: "DatabaseManager",
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate nutrition by product category.

    Groups all fact_items (with cached OFF data) by their category field,
    summing up absolute nutriment totals where unit_quantity is available.

    Args:
        db: Database manager
        start_date: ISO date string filter (inclusive)
        end_date: ISO date string filter (inclusive)

    Returns:
        List of dicts sorted by item count descending, each with:
        category, item_count, items_with_totals, totals, nutriscore_distribution.
    """
    rows = _fetch_nutrition_rows(db, start_date, end_date)

    by_cat: dict[str, dict[str, Any]] = {}

    for row in rows:
        try:
            product = json.loads(row["product_data"])
        except (json.JSONDecodeError, TypeError):
            continue

        if product.get("_not_found"):
            continue

        # Category: prefer fact_item.category, fall back to OFF categories string
        category = (
            row["category"]
            or product.get("categories", "").split(",")[-1].strip()
            or "Uncategorized"
        )
        category = category.strip() or "Uncategorized"

        if category not in by_cat:
            by_cat[category] = {
                "category": category,
                "item_count": 0,
                "items_with_totals": 0,
                "totals": dict(_EMPTY_TOTALS),
                "nutriscore_distribution": defaultdict(int),
            }

        entry = by_cat[category]
        entry["item_count"] += 1

        grade = (product.get("nutriscore_grade") or "").lower()
        if grade in ("a", "b", "c", "d", "e"):
            entry["nutriscore_distribution"][grade] += 1

        nutriments = _parse_nutriments(product)
        item_totals = _compute_item_nutrition(
            nutriments,
            row["unit_quantity"],
            row["quantity"],
        )
        if item_totals:
            entry["items_with_totals"] += 1
            for k, v in item_totals.items():
                entry["totals"][k] = round(entry["totals"][k] + v, 3)

    # Convert defaultdicts to plain dicts for clean serialisation
    result: list[dict[str, Any]] = []
    for entry in by_cat.values():
        entry["nutriscore_distribution"] = dict(entry["nutriscore_distribution"])
        result.append(entry)

    result.sort(key=lambda x: x["item_count"], reverse=True)
    return result
