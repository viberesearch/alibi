"""Price factor analysis -- discover which product attributes influence price.

Uses SQL aggregation and basic arithmetic to compute the marginal price
impact of attributes like product_variant, brand, and product annotations
(organic, free-range, etc.) on comparable_unit_price.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager


@dataclass
class PriceFactor:
    """A discovered price-influencing attribute."""

    attribute: str
    avg_premium: float
    pct_premium: float
    observations: int
    baseline_observations: int
    confidence: float


@dataclass
class ProductPriceProfile:
    """Price analysis for a single product across all vendors and variants."""

    comparable_name: str
    category: str | None
    total_observations: int
    baseline_price: float
    factors: list[PriceFactor]
    price_range: tuple[float, float]
    vendors: list[str]


def analyze_price_factors(
    db: DatabaseManager,
    comparable_name: str | None = None,
    category: str | None = None,
    min_observations: int = 3,
) -> list[ProductPriceProfile]:
    """Discover which attributes influence price for products.

    For each product (grouped by comparable_name), collects all observations
    with their attributes (product_variant + annotations), then computes
    the marginal price impact of each attribute.

    Args:
        db: Database manager.
        comparable_name: Analyze a specific product (or None for all with enough data).
        category: Filter to a category (e.g., "Dairy", "Eggs").
        min_observations: Minimum total observations to analyze a product.

    Returns:
        List of ProductPriceProfile sorted by total_observations descending.
    """
    # Step 1: Query all items with comparable_unit_price
    conditions = [
        "fi.comparable_unit_price IS NOT NULL",
        "fi.comparable_name IS NOT NULL",
    ]
    params: list[Any] = []

    if comparable_name is not None:
        conditions.append("fi.comparable_name = ?")
        params.append(comparable_name)
    if category is not None:
        conditions.append("fi.category = ?")
        params.append(category)

    where = " AND ".join(conditions)
    sql = f"""
SELECT fi.id, fi.comparable_name, fi.comparable_unit_price,
       fi.product_variant, fi.brand, fi.category,
       f.vendor
FROM fact_items fi
JOIN facts f ON fi.fact_id = f.id
WHERE {where}
ORDER BY fi.comparable_name
"""  # noqa: S608

    rows = db.fetchall(sql, tuple(params))
    if not rows:
        return []

    # Group by comparable_name
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["comparable_name"]].append(dict(row))

    # Step 2: Get all annotations for these items
    item_ids = [row["id"] for row in rows]
    annotations = _fetch_product_annotations(db, item_ids)

    # Step 3: Build profiles
    profiles: list[ProductPriceProfile] = []
    for name, items in groups.items():
        if len(items) < min_observations:
            continue
        profile = _build_profile(name, items, annotations)
        if profile is not None:
            profiles.append(profile)

    profiles.sort(key=lambda p: p.total_observations, reverse=True)
    return profiles


def _fetch_product_annotations(
    db: DatabaseManager,
    item_ids: list[str],
) -> dict[str, list[str]]:
    """Fetch product_attribute annotations for a list of item IDs.

    Returns:
        Dict mapping item_id -> list of attribute keys.
    """
    if not item_ids:
        return {}

    result: dict[str, list[str]] = defaultdict(list)

    # SQLite has a variable limit; batch in chunks of 500
    batch_size = 500
    for i in range(0, len(item_ids), batch_size):
        batch = item_ids[i : i + batch_size]
        placeholders = ",".join("?" * len(batch))
        sql = f"""
SELECT target_id, key
FROM annotations
WHERE annotation_type = 'product_attribute'
  AND target_type = 'fact_item'
  AND target_id IN ({placeholders})
"""  # noqa: S608
        rows = db.fetchall(sql, tuple(batch))
        for row in rows:
            result[row["target_id"]].append(row["key"])

    return result


def _build_profile(
    name: str,
    items: list[dict[str, Any]],
    annotations: dict[str, list[str]],
) -> ProductPriceProfile | None:
    """Build a ProductPriceProfile for a single product group."""
    # Collect attributes for each item
    item_attrs: list[tuple[float, set[str]]] = []
    vendors: set[str] = set()
    prices: list[float] = []
    category = None

    for item in items:
        price = float(item["comparable_unit_price"])
        prices.append(price)
        if item["vendor"]:
            vendors.add(item["vendor"])
        if item["category"] and category is None:
            category = item["category"]

        attrs: set[str] = set()
        if item["product_variant"]:
            attrs.add(f"variant:{item['product_variant']}")
        if item["brand"]:
            attrs.add(f"brand:{item['brand']}")
        for ann_key in annotations.get(item["id"], []):
            attrs.add(ann_key)

        item_attrs.append((price, attrs))

    # Find all distinct attributes
    all_attrs: set[str] = set()
    for _, attrs in item_attrs:
        all_attrs.update(attrs)

    # Compute baseline: items with NO attributes
    bare_prices = [p for p, a in item_attrs if not a]
    if bare_prices:
        baseline_price = sum(bare_prices) / len(bare_prices)
    else:
        baseline_price = sum(prices) / len(prices)

    # Compute factors
    factors: list[PriceFactor] = []
    for attr in all_attrs:
        with_prices = [p for p, a in item_attrs if attr in a]
        without_prices = [p for p, a in item_attrs if attr not in a]

        n_with = len(with_prices)
        n_without = len(without_prices)

        if n_with < 2 or n_without < 1:
            continue

        avg_with = sum(with_prices) / n_with
        avg_without = sum(without_prices) / n_without

        if avg_without == 0:
            continue

        avg_premium = avg_with - avg_without
        pct_premium = avg_premium / avg_without
        confidence = min(1.0, (n_with * n_without) / (n_with + n_without) / 5)

        factors.append(
            PriceFactor(
                attribute=attr,
                avg_premium=round(avg_premium, 4),
                pct_premium=round(pct_premium, 4),
                observations=n_with,
                baseline_observations=n_without,
                confidence=round(confidence, 4),
            )
        )

    factors.sort(key=lambda f: abs(f.pct_premium), reverse=True)

    return ProductPriceProfile(
        comparable_name=name,
        category=category,
        total_observations=len(items),
        baseline_price=round(baseline_price, 4),
        factors=factors,
        price_range=(min(prices), max(prices)),
        vendors=sorted(vendors),
    )


def get_category_price_factors(
    db: DatabaseManager,
    category: str,
    min_observations: int = 5,
) -> list[PriceFactor]:
    """Get price factors aggregated across all products in a category.

    Useful for discovering category-level patterns like "organic products
    in Dairy cost 30% more on average".
    """
    profiles = analyze_price_factors(
        db, category=category, min_observations=min_observations
    )

    # Aggregate factors across products by attribute name
    attr_data: dict[str, list[tuple[float, float, int, int]]] = defaultdict(list)
    for profile in profiles:
        for factor in profile.factors:
            attr_data[factor.attribute].append(
                (
                    factor.avg_premium,
                    factor.pct_premium,
                    factor.observations,
                    factor.baseline_observations,
                )
            )

    result: list[PriceFactor] = []
    for attr, entries in attr_data.items():
        total_obs = sum(e[2] for e in entries)
        total_baseline = sum(e[3] for e in entries)
        # Weighted average by observation count
        weighted_avg_premium = sum(e[0] * e[2] for e in entries) / total_obs
        weighted_pct_premium = sum(e[1] * e[2] for e in entries) / total_obs
        confidence = min(
            1.0, (total_obs * total_baseline) / (total_obs + total_baseline) / 5
        )

        result.append(
            PriceFactor(
                attribute=attr,
                avg_premium=round(weighted_avg_premium, 4),
                pct_premium=round(weighted_pct_premium, 4),
                observations=total_obs,
                baseline_observations=total_baseline,
                confidence=round(confidence, 4),
            )
        )

    result.sort(key=lambda f: abs(f.pct_premium), reverse=True)
    return result


def price_factor_summary(
    db: DatabaseManager,
    min_observations: int = 5,
) -> dict[str, Any]:
    """Get a high-level summary of price factors across all categories.

    Returns:
        Dict with:
        - categories: list of category names with enough data
        - top_factors: list of the most impactful factors across all products
        - products_analyzed: total number of products analyzed
    """
    profiles = analyze_price_factors(db, min_observations=min_observations)

    categories: set[str] = set()
    all_factors: list[PriceFactor] = []

    for profile in profiles:
        if profile.category:
            categories.add(profile.category)
        all_factors.extend(profile.factors)

    # Deduplicate and aggregate factors by attribute
    attr_data: dict[str, list[tuple[float, float, int, int]]] = defaultdict(list)
    for factor in all_factors:
        attr_data[factor.attribute].append(
            (
                factor.avg_premium,
                factor.pct_premium,
                factor.observations,
                factor.baseline_observations,
            )
        )

    top_factors: list[PriceFactor] = []
    for attr, entries in attr_data.items():
        total_obs = sum(e[2] for e in entries)
        total_baseline = sum(e[3] for e in entries)
        weighted_pct = sum(e[1] * e[2] for e in entries) / total_obs
        weighted_avg = sum(e[0] * e[2] for e in entries) / total_obs
        confidence = min(
            1.0, (total_obs * total_baseline) / (total_obs + total_baseline) / 5
        )
        top_factors.append(
            PriceFactor(
                attribute=attr,
                avg_premium=round(weighted_avg, 4),
                pct_premium=round(weighted_pct, 4),
                observations=total_obs,
                baseline_observations=total_baseline,
                confidence=round(confidence, 4),
            )
        )

    top_factors.sort(key=lambda f: abs(f.pct_premium), reverse=True)

    return {
        "categories": sorted(categories),
        "top_factors": top_factors[:20],
        "products_analyzed": len(profiles),
    }
