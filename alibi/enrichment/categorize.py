"""Hierarchical category enrichment for fact items.

A decoupled LLM pass — like the barcode/OFF and brand/category inference
passes — that assigns a normalised, hierarchical ``category_path`` from the
controlled taxonomy (see :mod:`alibi.enrichment.taxonomy`) to each FactItem.

It is independent of extraction: it batches items lacking a ``category_path``,
prompts the local-first structuring LLM with ``name / name_normalized /
comparable_name / brand`` against the taxonomy, validates the returned path,
and writes back ``category_path`` plus the leaf into the flat ``category``.

Idempotent, re-runnable and convergent: a row is processed while it lacks a
``category_path`` and has not been answered under the current
``taxonomy.TAXONOMY_VERSION`` (recorded in ``category_taxonomy_version``), so a
line the model can't map is marked once and not re-sent every run. Bumping
``taxonomy.TAXONOMY_VERSION`` makes both unmapped rows (re-selected directly) and
successfully-categorised rows (after their ``category_path`` is cleared) eligible
to recategorise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from alibi.db.connection import DatabaseManager
from alibi.enrichment import taxonomy
from alibi.enrichment._batch import (
    apply_answered,
    call_enrichment_llm,
    run_vendor_batches,
)

logger = logging.getLogger(__name__)

# Max items per LLM call. Kept small: local models reliably return a path for
# every item in a short list but start dropping items from long ones. Since the
# pass is idempotent, any dropped item is retried on the next run, but smaller
# batches converge in fewer passes.
_BATCH_SIZE = 12

_ENRICHMENT_SOURCE = "llm_category"
_CONFIDENCE = 0.7

_CATEGORIZE_PROMPT_TEMPLATE = """\
Assign the single best category to each retail/receipt line item below.
Store: {vendor}

You MUST choose a category PATH from this taxonomy (and nothing else):
{taxonomy}

Items (idx. name | normalized | comparable | brand):
{items_block}

Return JSON only:
{{"items": [
  {{"idx": 1, "category_path": "food > dairy > milk"}},
  {{"idx": 2, "category_path": "food > produce > vegetables"}}
]}}

Rules:
- Return one entry for EVERY item (idx 1..N). Do not omit any.
- category_path: a full path from the taxonomy, lowercase, " > " separated.
  Use the deepest path that fits; a top-level-only path is allowed when no
  sub-branch fits (e.g. "household").
- Non-product lines (tax, tip, service charge, totals, payment info, receipt
  footers, OCR noise) go under "adjustment" (e.g. "adjustment > tax",
  "adjustment > non_item").
- If genuinely undecidable, use "other".
- Only the JSON object, no explanation."""


@dataclass
class CategoryResult:
    """Outcome of category enrichment for a single fact item."""

    item_id: str
    category_path: str | None
    category: str | None
    success: bool


def _build_items_block(items: list[dict[str, Any]]) -> str:
    """Render the indexed item lines for the prompt."""
    lines: list[str] = []
    for i, item in enumerate(items):
        idx = i + 1
        name = item.get("name") or ""
        norm = item.get("name_normalized") or ""
        comp = item.get("comparable_name") or ""
        brand = item.get("brand") or ""
        lines.append(f"{idx}. {name} | {norm} | {comp} | {brand}")
    return "\n".join(lines)


def infer_categories(
    items: list[dict[str, Any]],
    vendor_name: str = "Unknown",
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: float | None = None,
) -> dict[int, str | None]:
    """Call the LLM to assign a taxonomy path to each item.

    Args:
        items: List of dicts with name/name_normalized/comparable_name/brand.
        vendor_name: Store/vendor name for context.
        model: Ollama model override (default: config structuring model).
        ollama_url: Ollama URL override.
        timeout: LLM call timeout.

    Returns:
        Mapping of 1-based idx -> canonical category_path, or ``None`` for an
        item the model answered with a path that failed taxonomy validation.
        Items the model dropped from its response are absent entirely — that
        distinction lets the caller mark answered rows processed while leaving
        dropped ones to be retried. Empty on failure.
    """
    if not items:
        return {}

    prompt = _CATEGORIZE_PROMPT_TEMPLATE.format(
        vendor=vendor_name,
        taxonomy=taxonomy.render_for_prompt(),
        items_block=_build_items_block(items),
    )

    inferred = call_enrichment_llm(
        prompt,
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
        label="Category enrichment",
    )

    out: dict[int, str | None] = {}
    for raw in inferred:
        if not isinstance(raw, dict):
            continue
        idx = raw.get("idx")
        if isinstance(idx, int):
            out[idx] = taxonomy.normalize_path(raw.get("category_path"))
    return out


def enrich_items(
    db: DatabaseManager,
    items: list[dict[str, Any]],
    vendor_name: str = "Unknown",
    model: str | None = None,
    ollama_url: str | None = None,
) -> list[CategoryResult]:
    """Categorise a batch of fact_items and write the results back.

    Args:
        db: Database manager.
        items: List of dicts with 'id', 'name' and optional context fields.
        vendor_name: Store name for context.
        model: Ollama model override.
        ollama_url: Ollama URL override.

    Returns:
        A CategoryResult per input item.
    """
    if not items:
        return []

    paths_by_idx = infer_categories(
        items, vendor_name=vendor_name, model=model, ollama_url=ollama_url
    )

    from alibi.services.correction import update_fact_item

    def _write(item_id: str, item: dict[str, Any], path: str) -> CategoryResult:
        leaf = taxonomy.leaf_of(path)
        update_fact_item(
            db,
            item_id,
            {
                "category_path": path,
                "category": leaf,
                "enrichment_source": _ENRICHMENT_SOURCE,
                "enrichment_confidence": _CONFIDENCE,
            },
        )
        return CategoryResult(item_id, path, leaf, success=True)

    results = apply_answered(
        db,
        items,
        paths_by_idx,
        mark_column="category_taxonomy_version",
        mark_value=taxonomy.TAXONOMY_VERSION,
        on_value=_write,
        on_skip=lambda item_id, item: CategoryResult(
            item_id, None, None, success=False
        ),
    )

    enriched = sum(1 for r in results if r.success)
    if enriched:
        logger.info(
            "Categorised %d/%d items for vendor %s", enriched, len(items), vendor_name
        )
    return results


def enrich_pending_categories(
    db: DatabaseManager,
    limit: int = 200,
    model: str | None = None,
    ollama_url: str | None = None,
) -> list[CategoryResult]:
    """Find and categorise fact_items lacking a ``category_path``.

    Groups items by vendor for context-aware batching, then calls the LLM in
    sub-batches. Re-runnable and convergent: a row is selected while it lacks a
    ``category_path`` AND was either never processed or processed under an older
    ``TAXONOMY_VERSION``. A line the model can't map to a valid path is marked
    with the current version and then skipped, so it is no longer re-sent every
    run; bumping ``taxonomy.TAXONOMY_VERSION`` makes such rows eligible again.

    Args:
        db: Database manager.
        limit: Max items to process in this run.
        model: Ollama model override.
        ollama_url: Ollama URL override.

    Returns:
        A CategoryResult per processed item.
    """
    rows = db.fetchall(
        "SELECT fi.id, fi.name, fi.name_normalized, fi.comparable_name, "
        "       fi.brand, f.vendor "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE (fi.category_path IS NULL OR fi.category_path = '') "
        "AND (fi.category_taxonomy_version IS NULL "
        "     OR fi.category_taxonomy_version < ?) "
        "AND fi.name IS NOT NULL AND fi.name != '' "
        "LIMIT ?",
        (taxonomy.TAXONOMY_VERSION, limit),
    )
    return run_vendor_batches(
        rows,
        _BATCH_SIZE,
        lambda vendor_name, batch: enrich_items(
            db, batch, vendor_name=vendor_name, model=model, ollama_url=ollama_url
        ),
    )
