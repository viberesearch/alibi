"""Cross-vendor product matching via Gemini.

Identifies when the same product is sold at different vendors under
different names (e.g., "Sourdough Baguette" at A == "Artisan Bread" at B).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Config helpers ---


def _get_api_key(api_key: str | None = None) -> str | None:
    if api_key:
        return api_key
    from alibi.config import get_config

    return get_config().gemini_api_key


def _get_model() -> str:
    from alibi.config import get_config

    return get_config().gemini_enrichment_model


# --- Pydantic models ---


class ProductMatchResult(BaseModel):
    """A pair of products that Gemini considers the same."""

    product_a_idx: int = Field(description="Index of first product in batch")
    product_b_idx: int = Field(description="Index of second product in batch")
    confidence: float = Field(description="Match confidence 0-1")
    reasoning: str = Field(description="Brief explanation of why they match")
    suggested_canonical: str | None = Field(
        default=None,
        description="Suggested canonical English name for the product",
    )


class MatchBatchResponse(BaseModel):
    """Batch response for product matching."""

    matches: list[ProductMatchResult]


@dataclass
class ProductCandidate:
    """A product to be matched."""

    item_id: str
    name: str
    vendor_name: str
    brand: str | None = None
    category: str | None = None
    barcode: str | None = None
    comparable_name: str | None = None
    unit_quantity: float | None = None
    unit: str | None = None


@dataclass
class MatchedProductGroup:
    """A group of products matched as the same item."""

    canonical_name: str
    products: list[ProductCandidate]
    confidence: float
    reasoning: str


_SYSTEM_PROMPT = """\
You are a product matching expert for grocery and retail items. \
You will receive a list of products from different vendors. \
Identify products that are actually the SAME item sold under different names \
at different stores.

Rules:
- Only match products that are genuinely the same product (same brand, same type, same size)
- "Store brand milk 1L" at Store A and "Store brand milk 1L" at Store B are NOT the same (different store brands)
- "Coca-Cola 330ml" at Store A and "Coca-Cola Can 330ml" at Store B ARE the same
- Products must be same brand+type+size to match. Similar products from different brands don't count
- Confidence: 1.0 = certain same product, 0.8 = very likely, 0.6 = probable
- Only return matches with confidence >= 0.6
- Suggest a canonical English name for each matched group\
"""


def _build_matching_prompt(products: list[ProductCandidate]) -> str:
    """Build prompt listing all products for matching."""
    lines = []
    for i, p in enumerate(products):
        parts = [f'[{i}] "{p.name}" at {p.vendor_name}']
        if p.brand:
            parts.append(f"brand={p.brand}")
        if p.category:
            parts.append(f"category={p.category}")
        if p.barcode:
            parts.append(f"barcode={p.barcode}")
        if p.unit_quantity and p.unit:
            parts.append(f"size={p.unit_quantity}{p.unit}")
        if p.comparable_name:
            parts.append(f"en={p.comparable_name}")
        lines.append(", ".join(parts))
    return "\n".join(lines)


def match_products_batch(
    products: list[ProductCandidate],
    api_key: str | None = None,
    model: str | None = None,
) -> list[ProductMatchResult]:
    """Send products to Gemini for cross-vendor matching."""
    resolved_key = _get_api_key(api_key)
    if not resolved_key:
        logger.warning("Product matching: no API key configured")
        return []

    if len(products) < 2:
        return []

    resolved_model = model or _get_model()

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        logger.error("google-genai package not installed")
        return []

    prompt = _build_matching_prompt(products)
    client = genai.Client(api_key=resolved_key)

    try:
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=MatchBatchResponse,
                temperature=0.1,
            ),
        )

        if response.parsed and isinstance(response.parsed, MatchBatchResponse):
            return response.parsed.matches

        raw_text = response.text or ""
        parsed = json.loads(raw_text)
        return [ProductMatchResult(**m) for m in parsed.get("matches", [])]

    except Exception:
        logger.exception("Product matching: API call failed")
        return []


def find_cross_vendor_matches(
    db: Any,
    category: str | None = None,
    min_vendors: int = 2,
    limit: int = 200,
    api_key: str | None = None,
) -> list[MatchedProductGroup]:
    """Find products that appear at multiple vendors under different names."""
    query = """
        SELECT fi.id, fi.name, fi.brand, fi.category, fi.barcode,
               fi.comparable_name, fi.unit_quantity, fi.unit,
               f.vendor_key,
               COALESCE(
                   (SELECT im.value FROM identity_members im
                    JOIN identities i ON im.identity_id = i.id
                    WHERE i.entity_type = 'vendor'
                    AND im.member_type = 'name'
                    AND im.identity_id = (
                        SELECT identity_id FROM identity_members
                        WHERE member_type = 'vendor_key'
                        AND value = f.vendor_key
                        LIMIT 1
                    )
                    LIMIT 1),
                   f.vendor_key
               ) as vendor_name
        FROM fact_items fi
        JOIN facts f ON fi.fact_id = f.id
        WHERE fi.name IS NOT NULL
        AND f.vendor_key IS NOT NULL
    """
    params: list[Any] = []

    if category:
        query += " AND fi.category = ?"
        params.append(category)

    query += " ORDER BY fi.name LIMIT ?"
    params.append(limit)

    rows = db.fetchall(query, tuple(params))
    if not rows:
        return []

    candidates = []
    for row in rows:
        candidates.append(
            ProductCandidate(
                item_id=row[0],
                name=row[1],
                brand=row[2],
                category=row[3],
                barcode=row[4],
                comparable_name=row[5],
                unit_quantity=float(row[6]) if row[6] else None,
                unit=row[7],
                vendor_name=row[9] or row[8] or "Unknown",
            )
        )

    vendors = {c.vendor_name for c in candidates}
    if len(vendors) < min_vendors:
        logger.info("Only %d vendors found, need %d", len(vendors), min_vendors)
        return []

    matches = match_products_batch(candidates, api_key=api_key)

    groups = []
    for m in matches:
        if m.product_a_idx >= len(candidates) or m.product_b_idx >= len(candidates):
            continue
        groups.append(
            MatchedProductGroup(
                canonical_name=(
                    m.suggested_canonical or candidates[m.product_a_idx].name
                ),
                products=[
                    candidates[m.product_a_idx],
                    candidates[m.product_b_idx],
                ],
                confidence=m.confidence,
                reasoning=m.reasoning,
            )
        )

    return groups
