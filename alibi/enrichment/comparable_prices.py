"""Comparable-price recompute pass for fact items.

``comparable_unit_price`` (normalised EUR/L, EUR/kg, EUR/pcs) is what makes the
item-analytics surface comparable across vendors. It is computed at extraction
time from ``total_price / quantity / unit / unit_quantity`` -- but when the
structurer failed to capture a package size it defaults ``unit`` to ``pcs``,
and the item lands in a meaningless "per piece" bucket even though its real
size ("OLIVE OIL 2L", "PASTA 450G") is right there in the name.

This is a deterministic, decoupled recompute pass (a sibling of the
:mod:`alibi.enrichment.comparable_names` LLM pass, but **no LLM** -- it never
invents a size). For items whose comparable price is weak (NULL, or stuck on
``pcs``/``other``), it re-parses the name with the same
:func:`alibi.atoms.parser._extract_unit_from_name` used at extraction, and if a
weight/volume size is present recomputes ``comparable_unit_price`` /
``comparable_unit`` via the canonical
:func:`alibi.atoms.parser._calculate_comparable_price`. Items with no parseable
size are left as-is (genuine count items stay ``pcs``).

Idempotent: a row is rewritten only when the recompute actually changes it, so
a second run is a no-op. Run ``lt items rebuild`` afterwards to sync item_stars.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from alibi.atoms.parser import (
    _calculate_comparable_price,
    _extract_unit_from_name,
)
from alibi.db.connection import DatabaseManager
from alibi.db.models import UnitType

logger = logging.getLogger(__name__)

# Weight/volume unit families that carry a meaningful normalisable size. A
# name-parsed size in one of these is worth overriding a pcs default for.
_SIZED_UNITS = {
    UnitType.GRAM,
    UnitType.KILOGRAM,
    UnitType.POUND,
    UnitType.OUNCE,
    UnitType.MILLILITER,
    UnitType.LITER,
    UnitType.GALLON,
}

# Sanity ceiling: a normalised unit price above this almost certainly comes from
# a corrupt total_price (OCR garble), not a real product, so we refuse to write
# it. Calibrated against the live data: genuine items top out around 390 EUR/L
# (small-package cosmetics) and 125 EUR/kg (loose premium tea), while OCR-garble
# outliers land in the thousands (5000-9900). 200 sits comfortably above the
# everyday grocery range and rejects the gross outliers early, at the enrichment
# layer, instead of relying on after-the-fact remediation. Lowering this only
# blocks NEW writes above the ceiling; it never deletes a price already stored.
_MAX_PLAUSIBLE_UNIT_PRICE = Decimal("200")


@dataclass
class ComparablePriceResult:
    """Outcome of a comparable-price recompute for a single fact item."""

    item_id: str
    comparable_unit: str | None
    comparable_unit_price: str | None
    changed: bool


def _dec(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def recompute_for_row(row: dict[str, Any]) -> dict[str, str] | None:
    """Recompute comparable price for one fact_item row.

    Args:
        row: Mapping with name, quantity, unit, unit_quantity, total_price,
            comparable_unit, comparable_unit_price.

    Returns:
        A dict of changed columns (subset of unit, unit_quantity,
        comparable_unit, comparable_unit_price) to write, or None when nothing
        should change (no parseable size, identical result, or implausible).
    """
    total_price = _dec(row.get("total_price"))
    if total_price is None or total_price == 0:
        return None

    unit = row.get("unit")
    unit_quantity = row.get("unit_quantity")

    # Try to recover a higher-fidelity unit + size from the name. Only override
    # when the name yields a weight/volume size and the current unit lacks one.
    name = row.get("name") or ""
    extracted = _extract_unit_from_name(name)
    recovered = False
    if extracted and extracted[1] in _SIZED_UNITS and extracted[3] is not None:
        try:
            current_unit = UnitType(unit) if unit else None
        except ValueError:
            current_unit = None
        # Override only if we don't already have a sized unit + quantity.
        if current_unit not in _SIZED_UNITS or unit_quantity in (None, ""):
            unit = extracted[1].value
            unit_quantity = str(extracted[3])
            recovered = True

    data: dict[str, Any] = {
        "total_price": str(total_price),
        "quantity": str(_dec(row.get("quantity")) or Decimal("1")),
    }
    if unit is not None:
        data["unit"] = unit
    if unit_quantity is not None:
        data["unit_quantity"] = str(unit_quantity)

    _calculate_comparable_price(data)
    new_price = data.get("comparable_unit_price")
    new_unit = data.get("comparable_unit")
    if new_price is None or new_unit is None:
        return None

    if _dec(new_price) and _dec(new_price) > _MAX_PLAUSIBLE_UNIT_PRICE:  # type: ignore[operator]
        logger.warning(
            "Skipping implausible comparable_unit_price %s for item %s (%s)",
            new_price,
            row.get("id"),
            name[:40],
        )
        return None

    old_price = _dec(row.get("comparable_unit_price"))
    old_unit = row.get("comparable_unit")
    # No-op if the result matches what's already stored (idempotency). Compare
    # the price numerically so stored REAL 2.0 vs recomputed "2.00" is a no-op.
    if (
        not recovered
        and old_price is not None
        and old_price == _dec(new_price)
        and new_unit == old_unit
    ):
        return None

    changes: dict[str, str] = {
        "comparable_unit": str(new_unit),
        "comparable_unit_price": str(new_price),
    }
    if recovered:
        changes["unit"] = str(unit)
        changes["unit_quantity"] = str(unit_quantity)
    return changes


def recompute_pending_comparable_prices(
    db: DatabaseManager,
    limit: int = 2000,
) -> list[ComparablePriceResult]:
    """Recompute comparable prices for fact_items with a weak/missing one.

    Selects priced items whose comparable price is NULL or stuck on a
    non-normalised unit (pcs/other), re-parses the name for a weight/volume
    size, and recomputes. Writes only the rows that actually change.

    Args:
        db: Database manager.
        limit: Max items to scan in this run.

    Returns:
        A ComparablePriceResult per scanned item (changed flag set on writes).
    """
    rows = db.fetchall(
        "SELECT id, name, quantity, unit, unit_quantity, total_price, "
        "       comparable_unit, comparable_unit_price "
        "FROM fact_items "
        "WHERE total_price IS NOT NULL "
        "AND (comparable_unit_price IS NULL "
        "     OR comparable_unit IS NULL OR comparable_unit IN ('pcs', 'other') "
        "     OR unit_quantity IS NULL) "
        "LIMIT ?",
        (limit,),
    )

    results: list[ComparablePriceResult] = []
    for row in rows:
        rowd = dict(row)
        changes = recompute_for_row(rowd)
        if not changes:
            results.append(ComparablePriceResult(rowd["id"], None, None, changed=False))
            continue

        # Note: enrichment_source is intentionally left untouched -- it records
        # how brand/category/comparable_name were determined; recomputing a
        # price from existing inputs is a mechanical fix, not a provenance change.
        set_clause = ", ".join(f"{col} = ?" for col in changes)
        params = [*changes.values(), rowd["id"]]
        with db.transaction() as cur:
            cur.execute(
                f"UPDATE fact_items SET {set_clause} WHERE id = ?",  # noqa: S608
                tuple(params),
            )
        results.append(
            ComparablePriceResult(
                rowd["id"],
                changes["comparable_unit"],
                changes["comparable_unit_price"],
                changed=True,
            )
        )

    changed = sum(1 for r in results if r.changed)
    if changed:
        logger.info(
            "Recomputed comparable price for %d/%d items", changed, len(results)
        )
    return results
