"""LLM-based brand/category inference for product enrichment.

Uses a local LLM (qwen3:8b by default) to infer brand and product
category from item names when no barcode or fuzzy match is available.
Items are batched per vendor for context-aware inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

# Max items per LLM call (keeps prompt under ~4k tokens)
_BATCH_SIZE = 25

_ENRICHMENT_PROMPT_TEMPLATE = """\
Extract brand and product category for each grocery/retail item below.
Store: {vendor}

Items:
{items_block}

Return JSON:
{{"items": [
  {{"idx": 1, "brand": "Example Brand", "category": "Dairy"}},
  {{"idx": 2, "brand": null, "category": "Beverages"}}
]}}

Rules:
- brand: product manufacturer name. null if store-brand, generic, or unknown.
- category: broad product category. Use one of: Dairy, Bakery, Beverages, \
Meat, Fish, Fruits, Vegetables, Snacks, Sweets, Cereals, Pasta, Rice, \
Condiments, Oils, Canned, Frozen, Cleaning, Personal Care, Baby, Pet, \
Alcohol, Tobacco, Household, Other. null if truly unknown.
- Be concise. Only the JSON object, no explanation."""


@dataclass
class LlmEnrichmentResult:
    """Result of LLM inference for a single item."""

    item_id: str
    brand: str | None
    category: str | None
    success: bool


def _get_llm_timeout() -> float:
    """Get LLM enrichment timeout from Config."""
    from alibi.config import get_config

    return get_config().llm_enrichment_timeout


def infer_brand_category(
    items: list[dict[str, str]],
    vendor_name: str = "Unknown",
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: float | None = None,
) -> list[dict[str, Any]]:
    """Call LLM to infer brand/category for a batch of items.

    Args:
        items: List of dicts with 'idx' (int) and 'name' (str) keys.
        vendor_name: Store/vendor name for context.
        model: Ollama model name (default: config.ollama_structure_model).
        ollama_url: Ollama URL (default: config).
        timeout: LLM call timeout.

    Returns:
        List of dicts with 'idx', 'brand', 'category' keys.
        Empty list on LLM failure.
    """
    if not items:
        return []

    items_block = "\n".join(f"{item['idx']}. {item['name']}" for item in items)

    prompt = _ENRICHMENT_PROMPT_TEMPLATE.format(
        vendor=vendor_name,
        items_block=items_block,
    )

    resolved_timeout = timeout if timeout is not None else _get_llm_timeout()

    try:
        from alibi.extraction.structurer import structure_ocr_text

        result = structure_ocr_text(
            raw_text="",
            emphasis_prompt=prompt,
            model=model,
            ollama_url=ollama_url,
            timeout=resolved_timeout,
        )

        inferred = result.get("items", [])
        if not isinstance(inferred, list):
            logger.warning("LLM returned non-list items: %s", type(inferred))
            return []

        return inferred

    except Exception:
        logger.exception("LLM enrichment call failed")
        return []


def enrich_items_by_llm(
    db: DatabaseManager,
    items: list[dict[str, Any]],
    vendor_name: str = "Unknown",
    model: str | None = None,
    ollama_url: str | None = None,
) -> list[LlmEnrichmentResult]:
    """Enrich a batch of fact_items using LLM inference.

    Calls the LLM with item names, then updates fact_items in the DB.

    Args:
        db: Database connection.
        items: List of dicts with 'id' and 'name' keys.
        vendor_name: Store name for context.
        model: Ollama model override.
        ollama_url: Ollama URL override.

    Returns:
        List of LlmEnrichmentResult for each item.
    """
    if not items:
        return []

    # Build indexed item list
    indexed = [{"idx": i + 1, "name": item["name"]} for i, item in enumerate(items)]

    inferred = infer_brand_category(
        indexed,
        vendor_name=vendor_name,
        model=model,
        ollama_url=ollama_url,
    )

    # Build lookup by idx
    inferred_by_idx: dict[int, dict[str, Any]] = {}
    for raw in inferred:
        raw_idx = raw.get("idx")
        if isinstance(raw_idx, int):
            inferred_by_idx[raw_idx] = raw

    results: list[LlmEnrichmentResult] = []
    for i, item in enumerate(items):
        idx = i + 1
        item_id = item["id"]
        entry = inferred_by_idx.get(idx)

        if not entry:
            results.append(
                LlmEnrichmentResult(
                    item_id=item_id, brand=None, category=None, success=False
                )
            )
            continue

        brand = entry.get("brand")
        category = entry.get("category")

        # Skip if LLM returned nothing useful
        if not brand and not category:
            results.append(
                LlmEnrichmentResult(
                    item_id=item_id, brand=brand, category=category, success=False
                )
            )
            continue

        # Update DB
        fields: dict[str, object] = {}
        if brand:
            fields["brand"] = brand
        if category:
            fields["category"] = category

        if fields:
            fields["enrichment_source"] = "llm_inference"
            fields["enrichment_confidence"] = 0.7

            from alibi.services.correction import update_fact_item

            update_fact_item(db, item_id, fields)

        results.append(
            LlmEnrichmentResult(
                item_id=item_id,
                brand=brand,
                category=category,
                success=True,
            )
        )

    enriched = sum(1 for r in results if r.success)
    if enriched:
        logger.info(
            "LLM enriched %d/%d items for vendor %s",
            enriched,
            len(items),
            vendor_name,
        )

    return results


def enrich_pending_by_llm(
    db: DatabaseManager,
    limit: int = 100,
    model: str | None = None,
    ollama_url: str | None = None,
) -> list[LlmEnrichmentResult]:
    """Find and enrich fact_items without brand/category using LLM.

    Groups items by vendor and calls LLM in batches.

    Args:
        db: Database connection.
        limit: Max items to process.
        model: Ollama model override.
        ollama_url: Ollama URL override.

    Returns:
        List of LlmEnrichmentResult for all processed items.
    """
    rows = db.fetchall(
        "SELECT fi.id, fi.name, f.vendor, f.vendor_key "
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

    # Group by vendor for context-aware batching
    vendor_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        vendor = row["vendor"] or "Unknown"
        vendor_groups.setdefault(vendor, []).append(
            {
                "id": row["id"],
                "name": row["name"],
            }
        )

    all_results: list[LlmEnrichmentResult] = []

    for vendor_name, items in vendor_groups.items():
        # Process in sub-batches if vendor has many items
        for batch_start in range(0, len(items), _BATCH_SIZE):
            batch = items[batch_start : batch_start + _BATCH_SIZE]
            results = enrich_items_by_llm(
                db,
                batch,
                vendor_name=vendor_name,
                model=model,
                ollama_url=ollama_url,
            )
            all_results.extend(results)

    return all_results
