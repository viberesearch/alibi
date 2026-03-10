"""Gemini-based product enrichment with mega-batch architecture.

Uses Google Gemini 2.0 Flash with Pydantic structured output to infer
brand, category, unit_quantity, and unit for ALL pending items in a
single API call. Mega-batching amortizes the fixed overhead (system
prompt + schema) across hundreds of items.

Privacy-preserving: sends ONLY product name and barcode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

_ENRICHMENT_SOURCE = "gemini"
_ENRICHMENT_CONFIDENCE = 0.85


class ItemEnrichment(BaseModel):
    """Enrichment result for a single item."""

    idx: int
    brand: str | None = None
    category: str | None = None
    unit_quantity: float | None = None
    unit: str | None = None
    comparable_name: str | None = None


class EnrichmentBatchResponse(BaseModel):
    """Mega-batch response containing all item enrichments."""

    items: list[ItemEnrichment]


@dataclass
class GeminiEnrichmentResult:
    """Result of Gemini inference for a single item."""

    item_id: str
    brand: str | None
    category: str | None
    unit_quantity: float | None
    unit: str | None
    comparable_name: str | None
    success: bool


_SYSTEM_PROMPT = """\
You are a product data specialist. For each numbered grocery/retail product, infer:
- brand: the manufacturer/brand name (null if store-brand, generic, or unknown)
- category: broad category from this exact list: Dairy, Bakery, Beverages, \
Meat, Fish, Fruits, Vegetables, Snacks, Sweets, Cereals, Pasta, Rice, \
Condiments, Oils, Canned, Frozen, Cleaning, Personal Care, Baby, Pet, \
Alcohol, Tobacco, Household, Other (null if truly unknown)
- unit_quantity: the standard package size as a number (e.g., 500 for 500ml milk, \
330 for a 330ml can, 1 for a single baguette). null if unknown.
- unit: the measurement unit (g, kg, ml, l, pcs). null if unknown.
- comparable_name: the product name translated to English, standardized for cross-language \
comparison (e.g., "Γάλα Πλήρες 1L" → "Full Fat Milk 1L"). null if already English.

Include every idx in the response. Be concise and accurate."""


def _is_enabled() -> bool:
    from alibi.config import get_config

    return get_config().gemini_enrichment_enabled


def _get_api_key(api_key: str | None) -> str | None:
    if api_key:
        return api_key
    from alibi.config import get_config

    return get_config().gemini_api_key


def _get_model() -> str:
    from alibi.config import get_config

    return get_config().gemini_enrichment_model


def infer_batch(
    items: list[dict[str, Any]],
    api_key: str | None = None,
    model: str | None = None,
) -> list[ItemEnrichment]:
    """Call Gemini for brand/category/unit_quantity inference on a batch.

    Args:
        items: List of dicts with 'idx' (int), 'name' (str), optional 'barcode'.
        api_key: Gemini API key. Falls back to config.
        model: Model ID. Defaults to gemini-2.5-flash.

    Returns:
        List of ItemEnrichment Pydantic models. Empty on failure.
    """
    if not items:
        return []

    resolved_key = _get_api_key(api_key)
    if not resolved_key:
        logger.warning("Gemini enrichment: GEMINI_API_KEY not set, skipping")
        return []

    resolved_model = model or _get_model()

    # Build items block
    lines = []
    for item in items:
        barcode = item.get("barcode", "")
        if barcode:
            lines.append(f"{item['idx']}. {item['name']} (barcode: {barcode})")
        else:
            lines.append(f"{item['idx']}. {item['name']}")

    items_text = "\n".join(lines)
    prompt = f"Products:\n{items_text}"

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=resolved_key)

        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=EnrichmentBatchResponse,
                temperature=0.1,
            ),
        )

        if response.parsed and isinstance(response.parsed, EnrichmentBatchResponse):
            return list(response.parsed.items)

        # Fallback: parse text if structured parsing failed
        import json

        raw_text = response.text or ""
        parsed = json.loads(raw_text)
        return [ItemEnrichment(**item) for item in parsed.get("items", [])]

    except Exception:
        logger.exception("Gemini enrichment: API call failed")
        return []


def enrich_items_by_gemini(
    db: DatabaseManager,
    items: list[dict[str, Any]],
    api_key: str | None = None,
    model: str | None = None,
) -> list[GeminiEnrichmentResult]:
    """Enrich a batch of fact_items using Gemini mega-batch.

    Updates DB for each item where data is inferred.
    """
    if not items:
        return []

    # Deduplicate by name+barcode (same as cloud_enrichment.py pattern)
    seen_keys: dict[str, dict[str, str]] = {}
    item_to_key: dict[str, str] = {}

    for item in items:
        name = (item.get("name") or "").strip().lower()
        barcode = (item.get("barcode") or "").strip()
        dedup_key = f"{name}|{barcode}"
        item_to_key[item["id"]] = dedup_key
        if dedup_key not in seen_keys:
            seen_keys[dedup_key] = {
                "name": item.get("name") or "",
                "barcode": barcode,
            }

    # Build indexed list
    unique_products = list(seen_keys.items())
    indexed: list[dict[str, Any]] = []
    key_to_idx: dict[str, int] = {}
    for idx, (key, product) in enumerate(unique_products, start=1):
        key_to_idx[key] = idx
        entry: dict[str, Any] = {"idx": idx, "name": product["name"]}
        if product["barcode"]:
            entry["barcode"] = product["barcode"]
        indexed.append(entry)

    inferred = infer_batch(indexed, api_key=api_key, model=model)

    # Build lookup by idx
    inferred_by_idx: dict[int, ItemEnrichment] = {item.idx: item for item in inferred}

    # Map dedup key -> result
    key_to_result: dict[str, ItemEnrichment] = {}
    for key, idx in key_to_idx.items():
        if idx in inferred_by_idx:
            key_to_result[key] = inferred_by_idx[idx]

    results: list[GeminiEnrichmentResult] = []

    # Pre-fetch existing unit_quantity to avoid overwriting extraction values.
    # Gemini guesses unit from product name (e.g., "PEANUT BRITTLE 25" → 25g)
    # which can be wrong when extraction already set correct weight.
    _item_ids = [item["id"] for item in items]
    _has_uq: set[str] = set()
    if _item_ids:
        _ph = ",".join("?" for _ in _item_ids)
        _uq_rows = db.fetchall(
            f"SELECT id FROM fact_items WHERE id IN ({_ph}) "
            "AND unit_quantity IS NOT NULL",
            tuple(_item_ids),
        )
        _has_uq = {r["id"] for r in _uq_rows}

    for item in items:
        item_id = item["id"]
        dedup_key = item_to_key[item_id]
        matched = key_to_result.get(dedup_key)

        if not matched:
            results.append(
                GeminiEnrichmentResult(
                    item_id=item_id,
                    brand=None,
                    category=None,
                    unit_quantity=None,
                    unit=None,
                    comparable_name=None,
                    success=False,
                )
            )
            continue

        # Build update fields
        fields: dict[str, object] = {}
        if matched.brand:
            fields["brand"] = matched.brand
        if matched.category:
            fields["category"] = matched.category
        if item_id not in _has_uq:
            if matched.unit_quantity is not None and matched.unit_quantity > 0:
                fields["unit_quantity"] = matched.unit_quantity
            if matched.unit:
                fields["unit"] = matched.unit
        if matched.comparable_name:
            fields["comparable_name"] = matched.comparable_name

        if not fields:
            results.append(
                GeminiEnrichmentResult(
                    item_id=item_id,
                    brand=None,
                    category=None,
                    unit_quantity=None,
                    unit=None,
                    comparable_name=None,
                    success=False,
                )
            )
            continue

        fields["enrichment_source"] = _ENRICHMENT_SOURCE
        fields["enrichment_confidence"] = _ENRICHMENT_CONFIDENCE

        from alibi.services.correction import update_fact_item

        update_fact_item(db, item_id, fields)

        results.append(
            GeminiEnrichmentResult(
                item_id=item_id,
                brand=matched.brand,
                category=matched.category,
                unit_quantity=matched.unit_quantity,
                unit=matched.unit,
                comparable_name=matched.comparable_name,
                success=True,
            )
        )

    enriched = sum(1 for r in results if r.success)
    if enriched:
        logger.info("Gemini enriched %d/%d items", enriched, len(items))

    return results


def normalize_names_by_gemini(
    db: DatabaseManager,
    limit: int = 500,
    api_key: str | None = None,
    model: str | None = None,
) -> list[GeminiEnrichmentResult]:
    """Find items where name_normalized == name and send to Gemini for comparable_name.

    Items whose name_normalized was set to the raw name as a baseline (no
    English translation) are sent to Gemini. The returned comparable_name is
    applied via update_fact_item which auto-backfills name_normalized.
    """
    if not _is_enabled():
        logger.debug("Gemini normalize-names: disabled")
        return []

    resolved_key = _get_api_key(api_key)
    if not resolved_key:
        logger.warning("Gemini normalize-names: GEMINI_API_KEY not set")
        return []

    rows = db.fetchall(
        "SELECT fi.id, fi.name, fi.barcode "
        "FROM fact_items fi "
        "WHERE fi.name_normalized = fi.name "
        "AND fi.name IS NOT NULL AND fi.name != '' "
        "LIMIT ?",
        (limit,),
    )

    if not rows:
        return []

    items = [
        {"id": row["id"], "name": row["name"], "barcode": row["barcode"] or ""}
        for row in rows
    ]

    return enrich_items_by_gemini(db, items, api_key=resolved_key, model=model)


def enrich_pending_by_gemini(
    db: DatabaseManager,
    limit: int = 500,
    api_key: str | None = None,
    model: str | None = None,
) -> list[GeminiEnrichmentResult]:
    """Find and enrich fact_items without brand/category via Gemini mega-batch.

    Historical-first: items with known unit_quantity from historical data
    get that applied without calling Gemini. Only truly unknown items go
    to the LLM.
    """
    if not _is_enabled():
        logger.debug(
            "Gemini enrichment: disabled (ALIBI_GEMINI_ENRICHMENT_ENABLED not set)"
        )
        return []

    resolved_key = _get_api_key(api_key)
    if not resolved_key:
        logger.warning("Gemini enrichment: GEMINI_API_KEY not set")
        return []

    # Get items needing enrichment (no brand AND no category)
    rows = db.fetchall(
        "SELECT fi.id, fi.name, fi.barcode, f.vendor_key, fi.brand, "
        "fi.unit_quantity AS current_uq "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE (fi.brand IS NULL OR fi.brand = '') "
        "AND (fi.category IS NULL OR fi.category = '') "
        "AND fi.name IS NOT NULL AND fi.name != '' "
        "LIMIT ?",
        (limit,),
    )

    if not rows:
        return []

    # Phase 1: Historical unit_quantity + comparable_name lookup
    from alibi.db.v2_store import (
        get_canonical_comparable_name,
        get_canonical_unit_quantity,
    )

    needs_gemini: list[dict[str, Any]] = []

    for row in rows:
        # Skip historical unit_quantity lookup if item already has one from
        # extraction — weighed/loose items have varying quantities per purchase
        # and the "most frequent" historical value would be wrong.
        if row["current_uq"] is not None:
            historical_uq = None
        else:
            historical_uq = get_canonical_unit_quantity(
                db,
                item_name=row["name"],
                barcode=row["barcode"],
                vendor_key=row["vendor_key"],
                brand=row["brand"],
            )
        historical_cn = get_canonical_comparable_name(
            db,
            item_name=row["name"],
            barcode=row["barcode"],
            vendor_key=row["vendor_key"],
            brand=row["brand"],
        )

        # Items still need Gemini for brand/category even if unit_quantity is known
        needs_gemini.append(
            {
                "id": row["id"],
                "name": row["name"],
                "barcode": row["barcode"] or "",
                "_historical_uq": historical_uq,  # Carry historical data for merge
                "_historical_cn": historical_cn,
            }
        )

    # Phase 2: Gemini mega-batch call for brand/category/unit inference
    gemini_items = [
        {"id": item["id"], "name": item["name"], "barcode": item["barcode"]}
        for item in needs_gemini
    ]

    gemini_results = enrich_items_by_gemini(
        db, gemini_items, api_key=resolved_key, model=model
    )

    # Phase 3: Apply historical unit_quantity where Gemini didn't provide one
    result_by_id = {r.item_id: r for r in gemini_results}

    for item in needs_gemini:
        historical = item.get("_historical_uq")
        if not historical:
            continue

        result = result_by_id.get(item["id"])
        if result and result.success and result.unit_quantity is None:
            # Gemini enriched brand/category but no unit_quantity — use historical
            fields: dict[str, object] = {}
            if historical.get("unit_quantity"):
                fields["unit_quantity"] = historical["unit_quantity"]
            if historical.get("unit"):
                fields["unit"] = historical["unit"]

            if fields:
                from alibi.services.correction import update_fact_item

                update_fact_item(db, item["id"], fields)

                # Update result object
                result.unit_quantity = historical["unit_quantity"]
                result.unit = historical.get("unit")

    # Phase 4: Apply historical comparable_name where Gemini didn't provide one
    for item in needs_gemini:
        historical_cn = item.get("_historical_cn")
        if not historical_cn:
            continue

        result = result_by_id.get(item["id"])
        if result and result.success and result.comparable_name is None:
            cn = historical_cn.get("comparable_name")
            if cn:
                from alibi.services.correction import update_fact_item

                update_fact_item(db, item["id"], {"comparable_name": cn})
                result.comparable_name = cn

    return gemini_results
