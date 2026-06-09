"""Flexible, fact-based product-attribute enrichment for fact items.

A single scalar ``product_variant`` can't express that one item carries several
price-influencing parameters at once (eggs: size=L + organic + free_range;
milk: fat + lactose_free; cheese: fat + type). This decoupled, local-first LLM
pass reads whatever facets are present in the item name into a flexible
``attributes`` JSON map (migration 040), so any facet can be filtered or grouped
independently downstream (``json_extract(attributes,'$.organic') = 1``).

It also captures the COUNT for count-packs (eggs come in 6/12/30): the count is
the size, so it is written as ``unit_quantity`` (unit=pcs) and the comparable
price is recomputed to EUR/piece (per egg) — making 6/12/30 packs comparable.

Keys are normalised to a standard vocabulary (so filtering is consistent) but
extra keys the model finds are kept. ``product_variant`` is back-filled from the
most salient facet (size > type > fat) when empty, so existing variant grouping
keeps working. Idempotent: only items with a NULL ``attributes`` are processed
(an item with no facet gets ``{}`` so it is not re-scanned). Run
``lt items rebuild`` afterwards to sync the analytics mirror.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from alibi.atoms.parser import _calculate_comparable_price
from alibi.db.connection import DatabaseManager
from alibi.enrichment._batch import call_enrichment_llm, run_vendor_batches

logger = logging.getLogger(__name__)

_BATCH_SIZE = 10

# Refuse a recomputed comparable price above this — corrupt total_price.
_MAX_PLAUSIBLE_UNIT_PRICE = Decimal("1000")

# Standard facet keys + their value kind, used to normalise the model output.
# "bool"/"num"/"str". Extra keys the model returns are kept as strings.
_STD_KEYS: dict[str, str] = {
    "size": "str",  # L / M / S / XL (eggs, apparel)
    "type": "str",  # gouda/feta; whole/skimmed; basmati
    "color": "str",
    "fat_pct": "num",  # 3.6, 0.5
    "organic": "bool",
    "free_range": "bool",
    "lactose_free": "bool",
    "gluten_free": "bool",
    "light": "bool",  # reduced fat/calorie
    "decaf": "bool",
}

# Structural / non-product keys the model sometimes invents. These are NOT
# product facets (they are identifiers, prices, dates, vendor/quantity data that
# live in their own columns) and only pollute the facet space, so they are
# dropped at write time. The facet vocabulary otherwise stays open/flexible.
_DENY_KEYS = frozenset(
    {
        "name",
        "full_name",
        "code",
        "not_code",
        "kot_code",
        "id",
        "barcode",
        "currency",
        "institution",
        "payment_method",
        "vat_rate",
        "tax_rate",
        "expiry_date",
        "date",
        "year",
        "quantity",
        "count",
        "weight",
        "weight_grams",
        "volume",
        "unit",
        "price",
        "total",
        "category",
        "brand",
        "vendor",
        "status",
    }
)

# Order to derive the legacy product_variant scalar from the facet map.
_VARIANT_PRIORITY = ("size", "type", "fat_pct")

_PROMPT_TEMPLATE = """\
For each retail/receipt line item, extract its product ATTRIBUTES (only the \
facts visible in the name — never guess) and, if it is a counted pack, the \
piece count.
Store: {vendor}

Items (idx. name | comparable | brand):
{items_block}

Return JSON only:
{{"items": [
  {{"idx": 1, "attributes": {{"size": "L", "organic": true, "free_range": true}}, "pack_count": 12}},
  {{"idx": 2, "attributes": {{"fat_pct": 3.6, "lactose_free": false}}, "pack_count": null}},
  {{"idx": 3, "attributes": {{}}, "pack_count": null}}
]}}

Rules:
- One entry for EVERY item (idx 1..N).
- attributes: an object of the facts present in the name. Prefer these standard
  keys when they apply: size ("L"/"M"/"S"/"XL"), type (e.g. "gouda", "skimmed",
  "basmati"), color, fat_pct (number, e.g. 3.6), organic (bool), free_range
  (bool), lactose_free (bool), gluten_free (bool), light (bool), decaf (bool).
  You may add other clear facet keys. Use {{}} if the name states no facet.
- Only include a key when the name actually states it. Do NOT invent values.
- Only PRODUCT facets. Never emit identifiers, codes, barcodes, prices, totals,
  tax/VAT, currency, dates, quantities, weights, units, the vendor or brand as
  attributes — those are not facets. Prefer the standard keys above; add a new
  key only for a genuine, reusable product property.
- pack_count: the number of pieces in a counted pack ("EGGS X12" -> 12,
  "30 EGGS" -> 30, "6 pack" -> 6). null if the item is not a counted pack.
- Only the JSON object, no explanation."""


@dataclass
class AttributeResult:
    """Outcome of attribute enrichment for a single fact item."""

    item_id: str
    attributes: dict[str, Any]
    pack_count: int | None
    changed: bool


def _build_items_block(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, item in enumerate(items):
        idx = i + 1
        name = item.get("name") or ""
        comp = item.get("comparable_name") or ""
        brand = item.get("brand") or ""
        lines.append(f"{idx}. {name} | {comp} | {brand}")
    return "\n".join(lines)


def _coerce(kind: str, value: Any) -> Any:
    """Coerce a model value to the standard kind, or None if implausible."""
    if value is None:
        return None
    if kind == "bool":
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in {"true", "yes", "1"}:
            return True
        if s in {"false", "no", "0"}:
            return False
        return None
    if kind == "num":
        try:
            return float(str(value).replace(",", "."))
        except (ValueError, TypeError):
            return None
    s = str(value).strip()
    return s.lower() if s else None


def _clean_attributes(raw: Any) -> dict[str, Any]:
    """Validate/normalise a model attribute map into clean scalar facets."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            continue
        k = key.strip().lower().replace(" ", "_")
        if k in _DENY_KEYS:
            continue  # structural / non-product noise
        kind = _STD_KEYS.get(k, "str")
        coerced = _coerce(kind, value)
        if coerced is None or coerced == "":
            continue
        if isinstance(coerced, str) and len(coerced) > 40:
            continue
        out[k] = coerced
    return out


def _clean_pack_count(value: Any) -> int | None:
    if value is None:
        return None
    try:
        n = int(float(str(value)))
    except (ValueError, TypeError):
        return None
    return n if 1 < n <= 1000 else None


def _derive_variant(attrs: dict[str, Any]) -> str | None:
    """Pick the most salient facet to back-fill the legacy product_variant."""
    for key in _VARIANT_PRIORITY:
        if key in attrs:
            val = attrs[key]
            if key == "fat_pct":
                return f"{val}%"
            return str(val)
    return None


def _dec(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def infer_attributes(
    items: list[dict[str, Any]],
    vendor_name: str = "Unknown",
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: float | None = None,
) -> dict[int, tuple[dict[str, Any], int | None]]:
    """Call the LLM to extract (attributes, pack_count) per item.

    Returns a mapping of 1-based idx -> (clean attribute map, pack_count). Items
    the model omitted are absent; an item with no facets maps to ({}, None).
    """
    if not items:
        return {}

    prompt = _PROMPT_TEMPLATE.format(
        vendor=vendor_name,
        items_block=_build_items_block(items),
    )

    inferred = call_enrichment_llm(
        prompt,
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
        label="Attribute enrichment",
    )

    out: dict[int, tuple[dict[str, Any], int | None]] = {}
    for raw in inferred:
        if not isinstance(raw, dict):
            continue
        idx = raw.get("idx")
        if not isinstance(idx, int):
            continue
        attrs = _clean_attributes(raw.get("attributes"))
        pack = _clean_pack_count(raw.get("pack_count"))
        out[idx] = (attrs, pack)
    return out


def write_attributes(
    db: DatabaseManager,
    item: dict[str, Any],
    attrs: dict[str, Any],
    pack: int | None,
) -> bool:
    """Persist the ``attributes`` map (+ derived variant / count-pack) for one item.

    Always writes the ``attributes`` JSON (``{}`` when no facet, so the item is
    marked processed). Back-fills ``product_variant`` from the salient facet when
    empty, and for a counted pack still lacking ``unit_quantity`` sets unit=pcs,
    unit_quantity=pack_count and recomputes the comparable price to EUR/piece.
    Shared by this pass and the combined enrichment pass. Returns whether the
    item carried any facet or pack count (i.e. the row meaningfully changed).
    """
    item_id = item["id"]
    changes: dict[str, Any] = {"attributes": json.dumps(attrs)}

    # Back-fill the legacy scalar variant from the facet map when empty.
    if not item.get("product_variant"):
        variant = _derive_variant(attrs)
        if variant:
            changes["product_variant"] = variant

    # Count-pack -> per-piece: the pack count is the size for count items.
    total_price = _dec(item.get("total_price"))
    if (
        pack is not None
        and item.get("unit_quantity") is None
        and total_price is not None
        and total_price > 0
    ):
        data: dict[str, Any] = {
            "total_price": str(total_price),
            "quantity": str(_dec(item.get("quantity")) or Decimal("1")),
            "unit": "pcs",
            "unit_quantity": str(pack),
        }
        _calculate_comparable_price(data)
        cup = _dec(data.get("comparable_unit_price"))
        cu = data.get("comparable_unit")
        if cup is not None and cu and cup <= _MAX_PLAUSIBLE_UNIT_PRICE:
            changes["unit"] = "pcs"
            changes["unit_quantity"] = str(pack)
            changes["comparable_unit"] = str(cu)
            changes["comparable_unit_price"] = str(cup)

    set_clause = ", ".join(f"{col} = ?" for col in changes)
    params = [*changes.values(), item_id]
    with db.transaction() as cur:
        cur.execute(
            f"UPDATE fact_items SET {set_clause} WHERE id = ?",  # noqa: S608
            tuple(params),
        )
    return bool(attrs) or pack is not None


def enrich_items(
    db: DatabaseManager,
    items: list[dict[str, Any]],
    vendor_name: str = "Unknown",
    model: str | None = None,
    ollama_url: str | None = None,
) -> list[AttributeResult]:
    """Extract attributes for a batch and write them back.

    Writes the ``attributes`` JSON (``{}`` when no facet, so the item is marked
    processed). For a counted pack lacking ``unit_quantity`` it sets unit=pcs,
    unit_quantity=pack_count and recomputes the comparable price to EUR/piece.
    Back-fills ``product_variant`` from the salient facet when empty.
    """
    if not items:
        return []

    inferred = infer_attributes(
        items, vendor_name=vendor_name, model=model, ollama_url=ollama_url
    )

    results: list[AttributeResult] = []
    for i, item in enumerate(items):
        idx = i + 1
        item_id = item["id"]
        if idx not in inferred:
            results.append(AttributeResult(item_id, {}, None, changed=False))
            continue

        attrs, pack = inferred[idx]
        changed = write_attributes(db, item, attrs, pack)
        results.append(AttributeResult(item_id, attrs, pack, changed=changed))

    enriched = sum(1 for r in results if r.attributes or r.pack_count)
    if enriched:
        logger.info(
            "Attribute-enriched %d/%d items for vendor %s",
            enriched,
            len(items),
            vendor_name,
        )
    return results


def enrich_pending_attributes(
    db: DatabaseManager,
    limit: int = 200,
    model: str | None = None,
    ollama_url: str | None = None,
) -> list[AttributeResult]:
    """Find and attribute-enrich fact_items lacking an ``attributes`` map.

    Groups items by vendor for context, then calls the LLM in sub-batches.
    Re-runnable: only items with a NULL ``attributes`` are selected (an item
    processed with no facet gets ``{}`` and is skipped thereafter).

    Returns an AttributeResult per processed item.
    """
    rows = db.fetchall(
        "SELECT fi.id, fi.name, fi.comparable_name, fi.brand, fi.quantity, "
        "       fi.total_price, fi.unit_quantity, fi.product_variant, f.vendor "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE fi.attributes IS NULL "
        "AND fi.name IS NOT NULL AND fi.name != '' "
        "LIMIT ?",
        (limit,),
    )
    return run_vendor_batches(
        rows,
        _BATCH_SIZE,
        lambda vendor_name, batch: enrich_items(
            db, batch, vendor_name=vendor_name, model=model, ollama_url=ollama_url
        ),
    )
