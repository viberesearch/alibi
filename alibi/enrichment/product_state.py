"""Controlled-vocabulary product-STATE enrichment for fact items.

A sibling of :mod:`alibi.enrichment.comparable_names` /
:mod:`alibi.enrichment.attributes`. It fills a single ``state`` facet inside the
flexible ``attributes`` JSON map: the preservation / preparation FORM that makes
the same food a different product for price comparison.

The motivating case: ``comparable_unit`` already separates volume (canned, €/L)
from weight (fresh, €/kg) from count (€/pcs), but it cannot express a within-unit
form difference -- raw vs roasted cashews (both €/kg), fresh vs smoked salmon,
fresh vs sun-dried tomatoes. ``state`` is that discriminator. Downstream it is an
ordinary attributes key, so it filters / groups for free
(``json_extract(attributes,'$.state') = 'canned'``) on the existing facet surface.

The vocabulary is deliberately small and closed (:data:`STATE_VOCAB`); the model's
phrasing is mapped onto it (:data:`_SYNONYMS`) and anything unresolved is dropped,
so the facet space stays clean. Items with no meaningful state (ambient dry
staples whose only form is dry -- pasta, rice, flour, sugar -- drinks, non-food)
get no ``state`` key.

Idempotent and convergent, mirroring comparable_names: a row is selected only
while it lacks a ``state`` AND has not been marked processed (``state_enriched``,
migration 044). A row the model answers "no state" for is marked once and then
skipped, so it is not re-sent to the LLM on every run; a row the model drops is
left unmarked and retried. Only real products (non-NULL ``comparable_name``) are
considered. Run ``lt items rebuild`` (or the command's own refresh) afterwards to
sync the analytics mirror.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from alibi.db.connection import DatabaseManager
from alibi.enrichment._batch import (
    apply_answered,
    call_enrichment_llm,
    run_vendor_batches,
)

logger = logging.getLogger(__name__)

# Max items per LLM call (see comparable_names for the rationale: small batches
# keep the local model from dropping items; the pass is idempotent regardless).
_BATCH_SIZE = 12

# The closed product-state vocabulary. Each value is the single form that makes
# the same food a distinct comparison item. Kept small on purpose: a larger set
# fragments the facet and burdens review for little analytical gain.
STATE_VOCAB: tuple[str, ...] = (
    "fresh",  # perishable, not preserved or cooked (fresh produce, raw meat/fish)
    "frozen",  # sold frozen
    "canned",  # canned/jarred/in brine/oil/syrup — shelf-stable in liquid
    "dried",  # dehydrated from fresh (raisins, dried herbs, sun-dried, pulses)
    "cured",  # salted/smoked/cured meat or fish (smoked salmon, bacon, ham)
    "pickled",  # pickled/brined/fermented vegetables (pickles, olives, kraut)
    "roasted",  # roasted/toasted (roasted nuts, roasted coffee, roasted peppers)
    "cooked",  # pre-cooked / ready-to-eat single foods or dishes
)

# Common model phrasings -> a canonical vocabulary value. Anything that resolves
# to neither a synonym nor a vocab member is dropped.
_SYNONYMS: dict[str, str] = {
    # fresh
    "fresh": "fresh",
    "raw": "fresh",
    "chilled": "fresh",
    "refrigerated": "fresh",
    "unprocessed": "fresh",
    # frozen
    "frozen": "frozen",
    "deep-frozen": "frozen",
    "deep frozen": "frozen",
    # canned
    "canned": "canned",
    "tinned": "canned",
    "tin": "canned",
    "jarred": "canned",
    "in brine": "canned",
    "in oil": "canned",
    "in water": "canned",
    "in syrup": "canned",
    "preserved": "canned",
    "conserved": "canned",
    # dried
    "dried": "dried",
    "dehydrated": "dried",
    "sun-dried": "dried",
    "sundried": "dried",
    "dry": "dried",
    # cured
    "cured": "cured",
    "smoked": "cured",
    "salted": "cured",
    "salt-cured": "cured",
    "cold-smoked": "cured",
    # pickled
    "pickled": "pickled",
    "brined": "pickled",
    "fermented": "pickled",
    # roasted
    "roasted": "roasted",
    "toasted": "roasted",
    "roast": "roasted",
    # cooked
    "cooked": "cooked",
    "pre-cooked": "cooked",
    "precooked": "cooked",
    "ready-to-eat": "cooked",
    "ready to eat": "cooked",
    "prepared": "cooked",
    "boiled": "cooked",
    "baked": "cooked",
    "grilled": "cooked",
    "fried": "cooked",
    "deli": "cooked",
}

_PROMPT_TEMPLATE = """\
For each retail/receipt FOOD item below, give its product STATE: the single \
preservation/preparation form that makes it a different product from the same \
food in another form (fresh tomato vs canned vs sun-dried; raw vs roasted \
cashews; fresh vs smoked salmon).
Store: {vendor}

Items (idx. name | comparable | category):
{items_block}

Return JSON only:
{{"items": [
  {{"idx": 1, "state": "fresh"}},
  {{"idx": 2, "state": "canned"}},
  {{"idx": 3, "state": null}}
]}}

Rules:
- Return one entry for EVERY item (idx 1..N). Do not omit any.
- state MUST be exactly one of: fresh, frozen, canned, dried, cured, pickled,
  roasted, cooked.
    fresh   = perishable, not preserved or cooked (fresh produce, raw meat/fish).
    frozen  = sold frozen.
    canned  = canned/jarred/in brine/oil/syrup (shelf-stable in liquid).
    dried   = dehydrated from a fresh form (raisins, dried herbs, sun-dried,
              dried pulses) — NOT staples whose only form is dry.
    cured   = salted/smoked/cured meat or fish (smoked salmon, bacon, ham).
    pickled = pickled/brined/fermented vegetables (pickles, olives, sauerkraut).
    roasted = roasted/toasted (roasted nuts, roasted coffee, roasted peppers).
    cooked  = pre-cooked / ready-to-eat single foods or dishes (rotisserie
              chicken, deli salads).
- Use the visible name first; use the category only as a hint.
- Return null (no state) when none applies: ambient dry staples whose only form
  is dry (pasta, rice, flour, sugar), drinks, household/non-food, or when unclear.
- Only the JSON object, no explanation."""

# JSON schema constraining the model's decoding to this pass's exact shape, so
# the local model cannot emit malformed JSON (the deterministic "Invalid JSON in
# response" failures on garbled OCR batches). Structural only — value cleanup /
# synonym mapping stays in _clean_state; gated by config.ollama_structured_output.
_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "state": {"type": ["string", "null"]},
                },
                "required": ["idx", "state"],
            },
        },
    },
    "required": ["items"],
}


@dataclass
class StateResult:
    """Outcome of product-state enrichment for a single fact item."""

    item_id: str
    state: str | None
    success: bool


def _build_items_block(items: list[dict[str, Any]]) -> str:
    """Render the indexed item lines for the prompt."""
    lines: list[str] = []
    for i, item in enumerate(items):
        idx = i + 1
        name = item.get("name") or ""
        comp = item.get("comparable_name") or ""
        cat = item.get("category_path") or item.get("category") or ""
        lines.append(f"{idx}. {name} | {comp} | {cat}")
    return "\n".join(lines)


def _clean_state(value: Any) -> str | None:
    """Resolve a model-returned state onto the closed vocabulary, or None.

    Lowercased and synonym-mapped; a value that is neither a synonym nor a vocab
    member returns None (dropped) so the facet space never accumulates one-off
    states.
    """
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    if not s or s in {"null", "none", "n/a", "na", "unknown"}:
        return None
    canon = _SYNONYMS.get(s)
    if canon is None and s in STATE_VOCAB:
        canon = s
    return canon


def _load_attributes(item: dict[str, Any]) -> dict[str, Any]:
    """Parse the stored attributes JSON for a row into a dict (or {})."""
    raw = item.get("attributes")
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def infer_states(
    items: list[dict[str, Any]],
    vendor_name: str = "Unknown",
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: float | None = None,
) -> dict[int, str | None]:
    """Call the LLM to assign a controlled product state to each item.

    Returns a mapping of 1-based idx -> canonical state, or ``None`` for an item
    the model returned null for / whose value failed to resolve. Items the model
    dropped from its response are absent entirely (that distinction lets the
    caller mark answered rows processed while leaving dropped ones to retry).
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
        label="Product-state enrichment",
        response_format=_RESPONSE_FORMAT,
    )

    out: dict[int, str | None] = {}
    for raw in inferred:
        if not isinstance(raw, dict):
            continue
        idx = raw.get("idx")
        if isinstance(idx, int):
            out[idx] = _clean_state(raw.get("state"))
    return out


def _write_state(
    db: DatabaseManager, item_id: str, item: dict[str, Any], state: str
) -> StateResult:
    """Merge a resolved state into the item's attributes JSON (preserving keys)."""
    attrs = _load_attributes(item)
    attrs["state"] = state
    with db.transaction() as cur:
        cur.execute(
            "UPDATE fact_items SET attributes = ? WHERE id = ?",
            (json.dumps(attrs), item_id),
        )
    return StateResult(item_id, state, success=True)


def enrich_items(
    db: DatabaseManager,
    items: list[dict[str, Any]],
    vendor_name: str = "Unknown",
    model: str | None = None,
    ollama_url: str | None = None,
) -> list[StateResult]:
    """Infer states for a batch of fact_items and merge them into attributes.

    ``success=False`` covers both LLM failures/drops and intentional null returns
    for items with no applicable state. Rows the model answered (value or explicit
    null) are stamped ``state_enriched`` so they are not re-sent.
    """
    if not items:
        return []

    states_by_idx = infer_states(
        items, vendor_name=vendor_name, model=model, ollama_url=ollama_url
    )

    results = apply_answered(
        db,
        items,
        states_by_idx,
        mark_column="state_enriched",
        on_value=lambda item_id, item, state: _write_state(db, item_id, item, state),
        on_skip=lambda item_id, item: StateResult(item_id, None, success=False),
    )

    enriched = sum(1 for r in results if r.success)
    if enriched:
        logger.info(
            "State-enriched %d/%d items for vendor %s",
            enriched,
            len(items),
            vendor_name,
        )
    return results


def enrich_pending_states(
    db: DatabaseManager,
    limit: int = 200,
    model: str | None = None,
    ollama_url: str | None = None,
) -> list[StateResult]:
    """Find and state-enrich real-product fact_items lacking a ``state`` facet.

    Groups items by vendor for context, then calls the LLM in sub-batches.
    Re-runnable and convergent: a row is selected only while it has not been
    marked processed (``state_enriched``). Only rows with a ``comparable_name``
    (real products) are considered — non-product lines have nothing to state.

    Returns a StateResult per processed item.
    """
    rows = db.fetchall(
        "SELECT fi.id, fi.name, fi.comparable_name, fi.category, "
        "       fi.category_path, fi.attributes, f.vendor "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE fi.state_enriched IS NULL "
        "AND fi.comparable_name IS NOT NULL AND fi.comparable_name != '' "
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
