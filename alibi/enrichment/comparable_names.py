"""Comparable-name enrichment for fact items.

A decoupled, local-first LLM pass -- a sibling of
:mod:`alibi.enrichment.categorize` -- that fills the ``comparable_name`` field:
a generic, brand-stripped, English product descriptor used for cross-vendor /
cross-language comparison (e.g. raw "TYPI ΓKOYNTA450G/MI. GOUDA" -> "gouda
cheese", "AYTA ΦΡΕΣKA L 12T/FR.EGGS L" -> "eggs large").

``comparable_name`` is normally set at structuring time (from the model's
``name_en``), but many older / hard-doc extractions left it NULL -- which
collapses them into one useless bucket in the item-analytics surface
(``item_stars`` / ``lt items avg-price``). This pass batches the items lacking
a comparable_name, prompts the local structuring LLM with their
``name / name_normalized / brand / category_path``, and writes the result back.

Idempotent, re-runnable and convergent: only items still lacking a
``comparable_name`` that have not been marked processed are selected. Non-product
lines (tax, discounts, totals, OCR noise) keep a NULL ``comparable_name`` -- they
have no product to compare -- but once the model has answered for a row it is
flagged via ``comparable_name_enriched`` so it is not re-sent to the LLM on every
run. The cloud counterpart for stragglers is ``lt enrich normalize-names``
(Gemini).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from alibi.db.connection import DatabaseManager
from alibi.enrichment._batch import (
    apply_answered,
    call_enrichment_llm,
    run_vendor_batches,
)

logger = logging.getLogger(__name__)

# Max items per LLM call. Kept small: local models reliably return a value for
# every item in a short list but start dropping items from long ones. The pass
# is idempotent, so any dropped item is retried on the next run.
_BATCH_SIZE = 12

_ENRICHMENT_SOURCE = "llm_comparable_name"
_CONFIDENCE = 0.7

_PROMPT_TEMPLATE = """\
For each retail/receipt line item below, give a generic English "comparable \
name": the plain product type used to compare the SAME product across stores, \
languages and currencies.
Store: {vendor}

Items (idx. name | normalized | brand | category):
{items_block}

Return JSON only:
{{"items": [
  {{"idx": 1, "comparable_name": "gouda cheese"}},
  {{"idx": 2, "comparable_name": "eggs large"}},
  {{"idx": 3, "comparable_name": null}}
]}}

Rules:
- Return one entry for EVERY item (idx 1..N). Do not omit any.
- comparable_name: lowercase English, generic product type. STRIP the brand,
  the store name, pack/size numbers and units (e.g. "450g", "1L", "x12") and
  any OCR garble. Keep a distinguishing variant word only when it defines the
  product (e.g. "milk" vs "skimmed milk"; "eggs large").
- Use the SAME comparable_name for the same product seen differently (translate
  non-English names to their common English term).
- Non-product lines (tax, tip, service charge, discount, deposit, total,
  subtotal, change, payment/card info, receipt footer, pure OCR noise): return
  null.
- Only the JSON object, no explanation."""


@dataclass
class ComparableNameResult:
    """Outcome of comparable-name enrichment for a single fact item."""

    item_id: str
    comparable_name: str | None
    success: bool


def _build_items_block(items: list[dict[str, Any]]) -> str:
    """Render the indexed item lines for the prompt."""
    lines: list[str] = []
    for i, item in enumerate(items):
        idx = i + 1
        name = item.get("name") or ""
        norm = item.get("name_normalized") or ""
        brand = item.get("brand") or ""
        cat = item.get("category_path") or item.get("category") or ""
        lines.append(f"{idx}. {name} | {norm} | {brand} | {cat}")
    return "\n".join(lines)


# Size / pack tokens that must NOT live in a comparable_name: the size belongs
# in unit_quantity, the fat percentage in attributes.fat_pct, the pack count in
# unit_quantity. The prompt already asks the model to strip these, but it
# occasionally leaves them in ("olive oil 2l", "cottage cheese 9%",
# "eggs large x12"), which fragments the comparison bucket. This is the
# deterministic safety net. It deliberately strips ONLY size/pack/percentage
# tokens — never a brand prefix, because the brand column is unreliable (it
# sometimes holds the product type itself, e.g. "Parmigiano Reggiano").
_SIZE_TOKEN_RE = re.compile(
    r"""(?<![a-z0-9])               # left boundary: not mid-word/number
    (?:
        \d+(?:[.,]\d+)?\s*%                                   # 9%  0.5%
      | x\s*\d+                                               # x12 (pack count)
      | \d+\s*x\s*\d+(?:[.,]\d+)?\s*(?:l|ml|cl|kg|g|gr)?      # 4x70g
      | \d+(?:[.,]\d+)?\s*(?:l|ml|cl|kg|g|gr|pcs|pc)          # 2l 450g 1350g
    )
    (?![a-z])                       # right boundary: a unit, not a word start
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _tidy_comparable_name(name: str) -> str:
    """Strip leftover size / pack / percentage tokens from a comparable_name.

    Deterministic and idempotent (a fixpoint): ``_tidy(_tidy(x)) == _tidy(x)``.
    Applied to every value the LLM pass writes and re-applied to existing rows by
    :func:`retidy_comparable_names`. Only size/pack/percentage tokens are
    removed; word content (including brands) is preserved, so it can never turn a
    real name into a different product.
    """
    prev = None
    out = name
    while out != prev:
        prev = out
        out = _SIZE_TOKEN_RE.sub(" ", out)
    out = re.sub(r"\s+", " ", out).strip()
    # Trim separators the stripped token left dangling ("octopus p. " -> "octopus p").
    out = re.sub(r"^[\s./,\-]+|[\s./,\-]+$", "", out).strip()
    return out


def _clean_comparable_name(value: Any) -> str | None:
    """Validate and normalise a model-returned comparable_name.

    Returns a trimmed, lowercased, size-stripped name, or None for blank / null /
    non-string / implausibly long values (the model occasionally echoes a whole
    line).
    """
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned or cleaned in {"null", "none", "n/a"}:
        return None
    cleaned = _tidy_comparable_name(cleaned)
    if not cleaned:
        return None
    if len(cleaned) > 80:
        return None
    return cleaned


def infer_comparable_names(
    items: list[dict[str, Any]],
    vendor_name: str = "Unknown",
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: float | None = None,
) -> dict[int, str | None]:
    """Call the LLM to assign a comparable_name to each item.

    Args:
        items: List of dicts with name/name_normalized/brand/category fields.
        vendor_name: Store/vendor name for context.
        model: Ollama model override (default: config structuring model).
        ollama_url: Ollama URL override.
        timeout: LLM call timeout.

    Returns:
        Mapping of 1-based idx -> comparable_name, or ``None`` for an item the
        model explicitly returned as null (a non-product line) or whose value
        failed validation. Items the model dropped from its response are absent
        entirely — that distinction lets the caller mark answered rows processed
        while leaving dropped ones to be retried.
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
        label="Comparable-name enrichment",
    )

    out: dict[int, str | None] = {}
    for raw in inferred:
        if not isinstance(raw, dict):
            continue
        idx = raw.get("idx")
        if isinstance(idx, int):
            out[idx] = _clean_comparable_name(raw.get("comparable_name"))
    return out


def enrich_items(
    db: DatabaseManager,
    items: list[dict[str, Any]],
    vendor_name: str = "Unknown",
    model: str | None = None,
    ollama_url: str | None = None,
) -> list[ComparableNameResult]:
    """Infer comparable_names for a batch of fact_items and write them back.

    Args:
        db: Database manager.
        items: List of dicts with 'id', 'name' and optional context fields.
        vendor_name: Store name for context.
        model: Ollama model override.
        ollama_url: Ollama URL override.

    Returns:
        A ComparableNameResult per input item. ``success=False`` covers both
        LLM failures and intentional null returns for non-product lines.
    """
    if not items:
        return []

    names_by_idx = infer_comparable_names(
        items, vendor_name=vendor_name, model=model, ollama_url=ollama_url
    )

    from alibi.services.correction import update_fact_item

    def _write(item_id: str, item: dict[str, Any], name: str) -> ComparableNameResult:
        update_fact_item(
            db,
            item_id,
            {
                "comparable_name": name,
                "enrichment_source": _ENRICHMENT_SOURCE,
                "enrichment_confidence": _CONFIDENCE,
            },
        )
        return ComparableNameResult(item_id, name, success=True)

    results = apply_answered(
        db,
        items,
        names_by_idx,
        mark_column="comparable_name_enriched",
        on_value=_write,
        on_skip=lambda item_id, item: ComparableNameResult(
            item_id, None, success=False
        ),
    )

    enriched = sum(1 for r in results if r.success)
    if enriched:
        logger.info(
            "Comparable-named %d/%d items for vendor %s",
            enriched,
            len(items),
            vendor_name,
        )
    return results


def enrich_pending_comparable_names(
    db: DatabaseManager,
    limit: int = 200,
    model: str | None = None,
    ollama_url: str | None = None,
) -> list[ComparableNameResult]:
    """Find and comparable-name fact_items lacking a ``comparable_name``.

    Groups items by vendor for context-aware batching, then calls the LLM in
    sub-batches. Re-runnable and convergent: a row is selected only while it
    still lacks a ``comparable_name`` AND has not been marked processed
    (``comparable_name_enriched``). A non-product line the model returns null
    for is marked once and then skipped, so it is no longer re-sent to the LLM
    on every run.

    Args:
        db: Database manager.
        limit: Max items to process in this run.
        model: Ollama model override.
        ollama_url: Ollama URL override.

    Returns:
        A ComparableNameResult per processed item.
    """
    rows = db.fetchall(
        "SELECT fi.id, fi.name, fi.name_normalized, fi.brand, "
        "       fi.category, fi.category_path, f.vendor "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE (fi.comparable_name IS NULL OR fi.comparable_name = '') "
        "AND fi.comparable_name_enriched IS NULL "
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


@dataclass
class TidyResult:
    """A single comparable_name the deterministic tidy pass rewrote."""

    item_id: str
    before: str
    after: str


def retidy_comparable_names(
    db: DatabaseManager, limit: int | None = None
) -> list[TidyResult]:
    """Re-strip size/pack/percentage tokens from existing comparable_names.

    A deterministic, local, no-LLM backfill: it applies :func:`_tidy_comparable_name`
    to every stored ``comparable_name`` and rewrites only the rows where the tidy
    changes the value (e.g. "olive oil 2l" -> "olive oil", "cottage cheese 9%" ->
    "cottage cheese", "eggs large x12" -> "eggs large"). The stripped size / fat /
    count already live in ``unit_quantity`` / ``attributes``, so nothing is lost;
    the win is that variants of one product collapse into a single comparison
    bucket. Idempotent (the tidy is a fixpoint), so re-runs are no-ops. Run
    ``lt items rebuild`` afterwards to sync the analytics mirror.

    Args:
        db: Database manager.
        limit: Max rows to scan (default: all).

    Returns:
        A TidyResult per rewritten row.
    """
    sql = (
        "SELECT id, comparable_name FROM fact_items "
        "WHERE comparable_name IS NOT NULL AND comparable_name != ''"
    )
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    rows = db.fetchall(sql, params)

    changes: list[TidyResult] = []
    for row in rows:
        before = row["comparable_name"]
        after = _tidy_comparable_name(before.strip().lower())
        if after and after != before:
            changes.append(TidyResult(row["id"], before, after))

    for ch in changes:
        with db.transaction() as cur:
            cur.execute(
                "UPDATE fact_items SET comparable_name = ? WHERE id = ?",
                (ch.after, ch.item_id),
            )

    if changes:
        logger.info("Tidied %d comparable_name(s)", len(changes))
    return changes
