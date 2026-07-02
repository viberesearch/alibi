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
import re
from dataclasses import dataclass
from typing import Any, Iterable

from alibi.db.connection import DatabaseManager
from alibi.enrichment import taxonomy
from alibi.enrichment._batch import (
    apply_answered,
    call_enrichment_llm,
    mark_processed,
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

# Deterministic keyword guard -------------------------------------------------
# A small, conservative set of high-signal product words mapped straight to a
# taxonomy path, applied BEFORE the pure-LLM pass. The motivating bug: the LLM
# bucketed "coffee beans" under ``food > pantry > baking``. For items whose name
# unambiguously identifies the product, a regex assignment is both more accurate
# and cheaper than an LLM call. Keep this list tight — only add a rule when a
# false positive is implausible. Multilingual (EN/EL/RU/TR) to match the corpus.
_KEYWORD_SOURCE = "keyword_category"
_KEYWORD_CONFIDENCE = 0.95

_KEYWORD_CATEGORY_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # Coffee — the motivating mis-categorisation (beans -> baking). Latin/Russian
    # words take a full \b...\b; the Greek stem (καφές/καφέ/καφεδες — the suffix
    # blocks a trailing \b) is matched stem-first with a leading boundary only.
    (
        re.compile(
            r"\b(?:coffee|espresso|cappuccino|americano|decaf|кофе|kahve)\b"
            r"|\bκαφ[εέ]",
            re.IGNORECASE,
        ),
        "food > beverages > coffee_tea",
    ),
    # Tea — same taxonomy leaf. The Greek stem τσά(ι|γ) (τσάι / τσάγια / τσαγιού)
    # is constrained to ι/γ after the vowel so it can't catch τσάντα ("bag").
    (
        re.compile(
            r"\b(?:tea|chai|matcha|чай|çay)\b" r"|\bτσ[αά][ιγ]",
            re.IGNORECASE,
        ),
        "food > beverages > coffee_tea",
    ),
)

# Fail fast at import on a typo'd or taxonomy-drifted target path.
for _pat, _path in _KEYWORD_CATEGORY_RULES:
    assert taxonomy.is_valid_path(_path), f"invalid keyword category path: {_path}"
del _pat, _path


def match_keyword_category(item: dict[str, Any]) -> str | None:
    """Deterministically map a high-signal item to a taxonomy path, or None.

    Searches the item's ``name`` / ``name_normalized`` / ``comparable_name`` for
    a keyword rule and returns the first match's canonical path. Returns None
    when no rule is confident, leaving the item to the LLM pass.
    """
    haystack = " ".join(
        str(item.get(field) or "")
        for field in ("name", "name_normalized", "comparable_name")
    )
    if not haystack.strip():
        return None
    for pattern, path in _KEYWORD_CATEGORY_RULES:
        if pattern.search(haystack):
            return path
    return None


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


# JSON schema for Ollama constrained decoding: forces a well-formed
# ``{"items": [{"idx", "category_path"}]}`` envelope so a garbled OCR batch
# can't yield unparseable JSON (the same root-cause guard PR #59 added to the
# product-state pass). Structural only — taxonomy validation stays in
# ``taxonomy.normalize_path``; gated by ``config.ollama_structured_output``.
_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "category_path": {"type": ["string", "null"]},
                },
                "required": ["idx", "category_path"],
            },
        },
    },
    "required": ["items"],
}


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
        response_format=_RESPONSE_FORMAT,
    )

    out: dict[int, str | None] = {}
    for raw in inferred:
        if not isinstance(raw, dict):
            continue
        idx = raw.get("idx")
        if isinstance(idx, int):
            out[idx] = taxonomy.normalize_path(raw.get("category_path"))
    return out


def _apply_category(
    db: DatabaseManager,
    item_id: str,
    path: str,
    *,
    source: str,
    confidence: float,
) -> CategoryResult:
    """Write a resolved category path + leaf back to a single fact_item."""
    from alibi.services.correction import update_fact_item

    leaf = taxonomy.leaf_of(path)
    update_fact_item(
        db,
        item_id,
        {
            "category_path": path,
            "category": leaf,
            "enrichment_source": source,
            "enrichment_confidence": confidence,
        },
    )
    return CategoryResult(item_id, path, leaf, success=True)


def apply_keyword_categories(
    db: DatabaseManager, rows: Iterable[Any]
) -> tuple[list[CategoryResult], list[dict[str, Any]]]:
    """Resolve high-signal items deterministically before the LLM pass.

    Walks ``rows`` (raw DB rows or dicts), assigns a category to every item a
    keyword rule matches (writing it back and stamping the idempotency marker),
    and returns ``(results, remaining)`` where ``remaining`` are the item dicts
    the LLM still needs to categorise.
    """
    matched: list[CategoryResult] = []
    matched_ids: list[str] = []
    remaining: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        path = match_keyword_category(item)
        if path:
            matched.append(
                _apply_category(
                    db,
                    item["id"],
                    path,
                    source=_KEYWORD_SOURCE,
                    confidence=_KEYWORD_CONFIDENCE,
                )
            )
            matched_ids.append(item["id"])
        else:
            remaining.append(item)
    mark_processed(
        db, "category_taxonomy_version", matched_ids, taxonomy.TAXONOMY_VERSION
    )
    if matched:
        logger.info("Keyword-categorised %d high-signal item(s)", len(matched))
    return matched, remaining


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

    results = apply_answered(
        db,
        items,
        paths_by_idx,
        mark_column="category_taxonomy_version",
        mark_value=taxonomy.TAXONOMY_VERSION,
        on_value=lambda item_id, item, path: _apply_category(
            db, item_id, path, source=_ENRICHMENT_SOURCE, confidence=_CONFIDENCE
        ),
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
    document_id: str | None = None,
) -> list[CategoryResult]:
    """Find and categorise fact_items lacking a ``category_path``.

    High-signal items (coffee, tea, ...) are resolved deterministically by
    :func:`apply_keyword_categories` first — both more accurate than the LLM on
    those (which once put coffee beans under "baking") and cheaper. The rest are
    grouped by vendor for context-aware batching and sent to the LLM in
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
        document_id: When given, restrict to the items of that one document
            (used by the ingestion finalizer so a fresh upload is categorised
            without dragging in the global backlog).

    Returns:
        A CategoryResult per processed item.
    """
    params: list[Any] = [taxonomy.TAXONOMY_VERSION]
    doc_join = ""
    doc_filter = ""
    if document_id:
        doc_join = (
            "JOIN cloud_bundles cb ON cb.cloud_id = f.cloud_id "
            "JOIN bundles b ON b.id = cb.bundle_id "
        )
        doc_filter = "AND b.document_id = ? "
        params.append(document_id)
    params.append(limit)
    rows = db.fetchall(
        "SELECT DISTINCT fi.id, fi.name, fi.name_normalized, fi.comparable_name, "
        "       fi.brand, f.vendor "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        f"{doc_join}"
        "WHERE (fi.category_path IS NULL OR fi.category_path = '') "
        "AND (fi.category_taxonomy_version IS NULL "
        "     OR fi.category_taxonomy_version < ?) "
        "AND fi.name IS NOT NULL AND fi.name != '' "
        f"{doc_filter}"
        "LIMIT ?",
        tuple(params),
    )
    # Deterministic high-signal items first, then the LLM on the remainder.
    keyword_results, remaining = apply_keyword_categories(db, rows)
    llm_results = run_vendor_batches(
        remaining,
        _BATCH_SIZE,
        lambda vendor_name, batch: enrich_items(
            db, batch, vendor_name=vendor_name, model=model, ollama_url=ollama_url
        ),
    )
    return keyword_results + llm_results
