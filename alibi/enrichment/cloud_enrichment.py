"""Cloud-based brand/category inference using Anthropic Claude API.

Privacy-preserving enrichment: sends ONLY product name and barcode —
no financial data, dates, vendor information, or user identifiers.

Tier 3 enrichment, used after local methods (OFF, product resolver,
local LLM) have already been attempted and failed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

# Max items per API call (keeps prompt under ~4k tokens)
_BATCH_SIZE = 30

# Default timeout for cloud API calls (seconds)
_CLOUD_TIMEOUT = 30.0

# Enrichment confidence for cloud API results
_CLOUD_CONFIDENCE = 0.85

# Enrichment source tag written to DB
_ENRICHMENT_SOURCE = "cloud_api"

# Default model — cheapest/fastest Haiku model for initial enrichment
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Anthropic Messages API endpoint
_API_ENDPOINT = "https://api.anthropic.com/v1/messages"

# Enrichment confidence for cloud-refined category corrections
_CLOUD_REFINED_CONFIDENCE = 0.9

# Enrichment source tag for category refinement
_ENRICHMENT_SOURCE_REFINED = "cloud_refined"

_CLOUD_PROMPT_TEMPLATE = """\
For each numbered product below, infer the brand and product category.
Only the product name (and optional barcode) are provided — no other context.

Products:
{items_block}

Return ONLY a JSON object in exactly this format:
{{"items": [
  {{"idx": 1, "brand": "Example Brand", "category": "Dairy"}},
  {{"idx": 2, "brand": null, "category": "Beverages"}}
]}}

Rules:
- brand: the manufacturer or brand name. Use null if store-brand, generic, or unknown.
- category: broad product category. Use exactly one of: Dairy, Bakery, Beverages, \
Meat, Fish, Fruits, Vegetables, Snacks, Sweets, Cereals, Pasta, Rice, \
Condiments, Oils, Canned, Frozen, Cleaning, Personal Care, Baby, Pet, \
Alcohol, Tobacco, Household, Other. Use null if truly unknown.
- Include every idx in the response, even if both brand and category are null.
- Return only the JSON object — no explanation, no markdown fences."""


@dataclass
class CloudEnrichmentResult:
    """Result of cloud API inference for a single item."""

    item_id: str
    brand: str | None
    category: str | None
    success: bool


_REFINE_PROMPT_TEMPLATE = """\
Review the category assignment for each product below. The current category was \
assigned by a local LLM and may be incorrect.

Products:
{items_block}

Return ONLY a JSON object:
{{"items": [
  {{"idx": 1, "corrected_category": "Fish", "reason": "salmon is fish not meat"}},
  {{"idx": 2, "corrected_category": null}}
]}}

Rules:
- corrected_category: the correct category if the current one is wrong. null if current is correct.
- Use exactly one of: Dairy, Bakery, Beverages, Meat, Fish, Fruits, Vegetables, Snacks, \
  Sweets, Cereals, Pasta, Rice, Condiments, Oils, Canned, Frozen, Cleaning, \
  Personal Care, Baby, Pet, Alcohol, Tobacco, Household, Other.
- Common LLM errors to watch for: salmon/tuna classified as Meat (should be Fish), \
  peppers/lettuce classified as Dairy (should be Vegetables), mustard/sauce classified \
  as Snacks (should be Condiments), eggs classified as other categories (should be Dairy).
- Only suggest correction if you are confident the current category is wrong.
- Return only the JSON object — no explanation, no markdown fences."""


def _is_enabled() -> bool:
    """Return True when cloud enrichment is enabled via Config."""
    from alibi.config import get_config

    return get_config().cloud_enrichment_enabled


def _get_api_key(api_key: str | None) -> str | None:
    """Resolve API key from argument or Config."""
    if api_key:
        return api_key
    from alibi.config import get_config

    return get_config().anthropic_api_key


def _get_model() -> str:
    """Resolve cloud enrichment model from Config."""
    from alibi.config import get_config

    return get_config().cloud_enrichment_model


def infer_cloud_brand_category(
    items: list[dict[str, str]],
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = _CLOUD_TIMEOUT,
) -> list[dict[str, Any]]:
    """Call Claude API for brand/category inference.

    Privacy contract: items must contain ONLY 'idx', 'name', and optionally
    'barcode'. No financial data, dates, vendor names, or user info.

    Args:
        items: List of dicts with 'idx' (int), 'name' (str), and optional
            'barcode' (str) keys.
        api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
        model: Claude model ID to use. Defaults to _DEFAULT_MODEL (Haiku).
        timeout: HTTP request timeout in seconds.

    Returns:
        List of dicts with 'idx', 'brand', 'category' keys.
        Empty list on API failure or missing key.
    """
    if not items:
        return []

    resolved_key = _get_api_key(api_key)
    if not resolved_key:
        logger.warning("cloud enrichment: ANTHROPIC_API_KEY not set, skipping")
        return []

    resolved_model = model if model is not None else _DEFAULT_MODEL

    # Build items block — include barcode when available for better accuracy
    lines: list[str] = []
    for item in items:
        barcode = item.get("barcode", "")
        if barcode:
            lines.append(f"{item['idx']}. {item['name']} (barcode: {barcode})")
        else:
            lines.append(f"{item['idx']}. {item['name']}")

    items_block = "\n".join(lines)
    prompt = _CLOUD_PROMPT_TEMPLATE.format(items_block=items_block)

    try:
        import httpx

        response = httpx.post(
            _API_ENDPOINT,
            headers={
                "x-api-key": resolved_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": resolved_model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        raw_text = data["content"][0]["text"]

        # Strip markdown fences if the model added them despite instructions
        stripped = raw_text.strip()
        if stripped.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = stripped.find("\n")
            if first_newline != -1:
                stripped = stripped[first_newline + 1 :]
            # Remove closing fence
            if stripped.endswith("```"):
                stripped = stripped[: stripped.rfind("```")]

        parsed = json.loads(stripped.strip())
        inferred = parsed.get("items", [])
        if not isinstance(inferred, list):
            logger.warning(
                "cloud enrichment: API returned non-list items: %s", type(inferred)
            )
            return []

        return inferred

    except Exception:
        logger.exception("cloud enrichment: API call failed")
        return []


def enrich_items_by_cloud(
    db: DatabaseManager,
    items: list[dict[str, Any]],
    api_key: str | None = None,
    model: str | None = None,
) -> list[CloudEnrichmentResult]:
    """Enrich a batch of fact_items using the Claude cloud API.

    Updates the DB for each item where brand or category is inferred.

    Args:
        db: Database connection.
        items: List of dicts with 'id', 'name', and optional 'barcode' keys.
        api_key: Anthropic API key override.
        model: Claude model ID override.

    Returns:
        List of CloudEnrichmentResult for each item.
    """
    if not items:
        return []

    # Deduplicate by identity: same name+barcode pair resolved once.
    # Build a canonical key for each item and collect unique ones.
    seen_keys: dict[str, dict[str, str]] = {}  # key -> first representative
    item_to_key: dict[str, str] = {}  # item_id -> dedup key

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

    # Build indexed list of unique products
    unique_products = list(seen_keys.items())  # [(key, {name, barcode})]
    indexed: list[dict[str, str]] = []
    key_to_idx: dict[str, int] = {}
    for idx, (key, product) in enumerate(unique_products, start=1):
        key_to_idx[key] = idx
        entry: dict[str, str] = {"idx": str(idx), "name": product["name"]}
        if product["barcode"]:
            entry["barcode"] = product["barcode"]
        indexed.append(entry)

    # Convert idx to int as the prompt template expects
    indexed_typed: list[dict[str, Any]] = [{**e, "idx": int(e["idx"])} for e in indexed]

    inferred = infer_cloud_brand_category(indexed_typed, api_key=api_key, model=model)

    # Build lookup by idx
    inferred_by_idx: dict[int, dict[str, Any]] = {}
    for raw in inferred:
        raw_idx = raw.get("idx")
        if isinstance(raw_idx, int):
            inferred_by_idx[raw_idx] = raw

    # Map from dedup key to inferred result
    key_to_result: dict[str, dict[str, Any]] = {}
    for key, idx in key_to_idx.items():
        inferred_entry = inferred_by_idx.get(idx)
        if inferred_entry:
            key_to_result[key] = inferred_entry

    results: list[CloudEnrichmentResult] = []

    for item in items:
        item_id = item["id"]
        dedup_key = item_to_key[item_id]
        matched: dict[str, Any] | None = key_to_result.get(dedup_key)

        if not matched:
            results.append(
                CloudEnrichmentResult(
                    item_id=item_id, brand=None, category=None, success=False
                )
            )
            continue

        brand = matched.get("brand") or None
        category = matched.get("category") or None

        if not brand and not category:
            results.append(
                CloudEnrichmentResult(
                    item_id=item_id, brand=None, category=None, success=False
                )
            )
            continue

        # Persist to DB
        fields: dict[str, object] = {}
        if brand:
            fields["brand"] = brand
        if category:
            fields["category"] = category

        fields["enrichment_source"] = _ENRICHMENT_SOURCE
        fields["enrichment_confidence"] = _CLOUD_CONFIDENCE

        from alibi.services.correction import update_fact_item

        update_fact_item(db, item_id, fields)

        results.append(
            CloudEnrichmentResult(
                item_id=item_id,
                brand=brand,
                category=category,
                success=True,
            )
        )

    enriched = sum(1 for r in results if r.success)
    if enriched:
        logger.info(
            "cloud enrichment: enriched %d/%d items",
            enriched,
            len(items),
        )

    return results


def enrich_pending_by_cloud(
    db: DatabaseManager,
    limit: int = 100,
    api_key: str | None = None,
    model: str | None = None,
) -> list[CloudEnrichmentResult]:
    """Find and enrich fact_items without brand/category via cloud API.

    Selects items that have no brand AND no category (regardless of whether
    local enrichment was previously attempted). Processes in batches.

    Returns an empty list when ALIBI_CLOUD_ENRICHMENT_ENABLED is not set
    to a truthy value, or when ANTHROPIC_API_KEY is unavailable.

    Args:
        db: Database connection.
        limit: Max items to process per call.
        api_key: Anthropic API key override.
        model: Claude model ID override.

    Returns:
        List of CloudEnrichmentResult for all processed items.
    """
    if not _is_enabled():
        logger.debug(
            "cloud enrichment: disabled (ALIBI_CLOUD_ENRICHMENT_ENABLED not set)"
        )
        return []

    resolved_key = _get_api_key(api_key)
    if not resolved_key:
        logger.warning("cloud enrichment: ANTHROPIC_API_KEY not set, cannot enrich")
        return []

    rows = db.fetchall(
        "SELECT fi.id, fi.name, fi.barcode "
        "FROM fact_items fi "
        "WHERE (fi.brand IS NULL OR fi.brand = '') "
        "AND (fi.category IS NULL OR fi.category = '') "
        "AND fi.name IS NOT NULL AND fi.name != '' "
        "LIMIT ?",
        (limit,),
    )

    if not rows:
        return []

    items = [
        {
            "id": row["id"],
            "name": row["name"],
            "barcode": row["barcode"] or "",
        }
        for row in rows
    ]

    all_results: list[CloudEnrichmentResult] = []

    # Process in sub-batches
    for batch_start in range(0, len(items), _BATCH_SIZE):
        batch = items[batch_start : batch_start + _BATCH_SIZE]
        results = enrich_items_by_cloud(db, batch, api_key=resolved_key, model=model)
        all_results.extend(results)

    return all_results


def refine_categories_by_cloud(
    db: DatabaseManager,
    limit: int = 100,
    api_key: str | None = None,
    model: str | None = None,
) -> list[CloudEnrichmentResult]:
    """Refine LLM-inferred categories using the cloud API (Sonnet).

    Queries items with enrichment_source='llm_inference' that already have a
    category, then sends them to the cloud model for verification. Only
    updates items where the cloud model disagrees with the local LLM category.

    Uses the configured cloud_enrichment_model (default: claude-sonnet-4-6),
    which has higher category accuracy than the local LLM (qwen3:8b).

    Args:
        db: Database connection.
        limit: Max items to evaluate per call.
        api_key: Anthropic API key override.
        model: Claude model ID override. Defaults to Config.cloud_enrichment_model.

    Returns:
        List of CloudEnrichmentResult for items whose category was corrected.
        Items where cloud agreed with the local LLM are excluded from results.
    """
    resolved_key = _get_api_key(api_key)
    if not resolved_key:
        logger.warning("cloud refinement: ANTHROPIC_API_KEY not set, cannot refine")
        return []

    resolved_model = model if model is not None else _get_model()

    rows = db.fetchall(
        "SELECT fi.id, fi.name, fi.barcode, fi.category, fi.brand "
        "FROM fact_items fi "
        "WHERE fi.enrichment_source = 'llm_inference' "
        "AND fi.category IS NOT NULL AND fi.category != '' "
        "LIMIT ?",
        (limit,),
    )

    if not rows:
        logger.debug("cloud refinement: no llm_inference items with category found")
        return []

    all_results: list[CloudEnrichmentResult] = []

    # Process in sub-batches to stay within prompt token limits
    for batch_start in range(0, len(rows), _BATCH_SIZE):
        batch_rows = rows[batch_start : batch_start + _BATCH_SIZE]

        # Build indexed items block for the refine prompt
        lines: list[str] = []
        indexed_rows: list[Any] = []
        for idx, row in enumerate(batch_rows, start=1):
            brand_str = row["brand"] or "null"
            lines.append(
                f'{idx}. "{row["name"]}" [current: {row["category"]}, brand: {brand_str}]'
            )
            indexed_rows.append((idx, row))

        items_block = "\n".join(lines)
        prompt = _REFINE_PROMPT_TEMPLATE.format(items_block=items_block)

        try:
            import httpx

            response = httpx.post(
                _API_ENDPOINT,
                headers={
                    "x-api-key": resolved_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": resolved_model,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=_CLOUD_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            raw_text = data["content"][0]["text"]

            # Strip markdown fences if the model added them despite instructions
            stripped = raw_text.strip()
            if stripped.startswith("```"):
                first_newline = stripped.find("\n")
                if first_newline != -1:
                    stripped = stripped[first_newline + 1 :]
                if stripped.endswith("```"):
                    stripped = stripped[: stripped.rfind("```")]

            parsed = json.loads(stripped.strip())
            inferred = parsed.get("items", [])
            if not isinstance(inferred, list):
                logger.warning(
                    "cloud refinement: API returned non-list items: %s", type(inferred)
                )
                continue

        except Exception:
            logger.exception("cloud refinement: API call failed for batch")
            continue

        # Build lookup by idx
        inferred_by_idx: dict[int, dict[str, Any]] = {}
        for raw in inferred:
            raw_idx = raw.get("idx")
            if isinstance(raw_idx, int):
                inferred_by_idx[raw_idx] = raw

        from alibi.services.correction import update_fact_item

        for idx, row in indexed_rows:
            entry = inferred_by_idx.get(idx)
            if not entry:
                continue

            corrected_category = entry.get("corrected_category")
            if not corrected_category:
                # Cloud agrees with local LLM — no update needed
                continue

            # Cloud disagrees: update category with refined source tag
            fields: dict[str, object] = {
                "category": corrected_category,
                "enrichment_source": _ENRICHMENT_SOURCE_REFINED,
                "enrichment_confidence": _CLOUD_REFINED_CONFIDENCE,
            }
            update_fact_item(db, row["id"], fields)

            logger.info(
                "cloud refinement: item %s category %s -> %s",
                row["id"][:8],
                row["category"],
                corrected_category,
            )

            all_results.append(
                CloudEnrichmentResult(
                    item_id=row["id"],
                    brand=row["brand"],
                    category=corrected_category,
                    success=True,
                )
            )

    refined_count = len(all_results)
    if refined_count:
        logger.info(
            "cloud refinement: corrected %d/%d categories",
            refined_count,
            len(rows),
        )
    else:
        logger.debug("cloud refinement: no corrections needed for %d items", len(rows))

    return all_results
