"""LLM unit-extraction enrichment for fact items.

The structurer prompt now captures embedded sizes ("PASTA 450G" -> g/450) for
NEW documents, but ~700 already-processed fact_items were extracted before that
and have a NULL ``unit_quantity`` even though the size is sitting in the stored
``name``. Re-OCR does not recover them (shadow-tested: no gain); the size is in
the name we already have.

This is a decoupled, local-first LLM pass (sibling of
:mod:`alibi.enrichment.comparable_names`) that applies the same embedded-size /
multipack capability to existing data, with NO re-OCR and NO re-ingest. For
items missing ``unit_quantity`` it prompts the local model to read the size out
of the name, then writes back ``unit`` / ``unit_quantity`` and recomputes
``comparable_unit_price`` / ``comparable_unit`` via the canonical
:func:`alibi.atoms.parser._calculate_comparable_price`. Items with no size in
the name (genuine count items, non-product lines) are left untouched.

A plausibility ceiling refuses to write a comparable price derived from a
corrupt ``total_price``. Run ``lt items rebuild`` afterwards to sync item_stars.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from alibi.atoms.parser import _calculate_comparable_price
from alibi.db.connection import DatabaseManager
from alibi.db.models import UnitType
from alibi.enrichment._batch import (
    apply_answered,
    call_enrichment_llm,
    run_vendor_batches,
)
from alibi.normalizers.units import normalize_unit

logger = logging.getLogger(__name__)

_BATCH_SIZE = 12

# Unit families that carry a normalisable size worth writing.
_SIZED_UNITS = {
    UnitType.GRAM,
    UnitType.KILOGRAM,
    UnitType.POUND,
    UnitType.OUNCE,
    UnitType.MILLILITER,
    UnitType.LITER,
    UnitType.GALLON,
}

# Refuse a comparable price above this — signals a corrupt total_price.
_MAX_PLAUSIBLE_UNIT_PRICE = Decimal("1000")

_PROMPT_TEMPLATE = """\
For each retail/receipt line item below, read the PACKAGE SIZE out of the name \
and return its unit and numeric quantity.
Store: {vendor}

Items (idx. name | brand):
{items_block}

Return JSON only:
{{"items": [
  {{"idx": 1, "unit": "g", "unit_quantity": 450}},
  {{"idx": 2, "unit": "l", "unit_quantity": 2}},
  {{"idx": 3, "unit": null, "unit_quantity": null}}
]}}

Rules:
- Return one entry for EVERY item (idx 1..N). Do not omit any.
- unit: one of "g", "kg", "ml", "l" — the weight/volume unit of the size token
  in the name ("450G" -> "g", "2L" -> "l", "720ML" -> "ml", "1,5L" -> "l").
- unit_quantity: the numeric size (450, 2, 720). Use a dot decimal (1.5).
- MULTIPACKS "N x M<unit>" (e.g. "4X70G", "6X1,5L"): unit_quantity = N*M
  (4X70G -> 280; 6X1,5L -> 9).
- If the name has NO weight/volume size (plain count items like "BANANAS",
  "EGGS 12PK", non-product lines like tax/total/bag, or only a bare count
  "X12"): return unit=null, unit_quantity=null. Do NOT guess a size.
- Only the JSON object, no explanation."""


@dataclass
class UnitResult:
    """Outcome of unit enrichment for a single fact item."""

    item_id: str
    unit: str | None
    unit_quantity: str | None
    success: bool


def _build_items_block(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, item in enumerate(items):
        idx = i + 1
        name = item.get("name") or ""
        brand = item.get("brand") or ""
        lines.append(f"{idx}. {name} | {brand}")
    return "\n".join(lines)


def _clean_size(unit_raw: Any, qty_raw: Any) -> tuple[str, Decimal] | None:
    """Validate a model-returned (unit, unit_quantity) into a sized pair."""
    if unit_raw is None or qty_raw is None:
        return None
    unit_type = normalize_unit(str(unit_raw))
    if unit_type not in _SIZED_UNITS:
        return None
    try:
        qty = Decimal(str(qty_raw).replace(",", "."))
    except (InvalidOperation, ValueError):
        return None
    if qty <= 0 or qty >= Decimal("100000"):
        return None
    return unit_type.value, qty


def infer_units(
    items: list[dict[str, Any]],
    vendor_name: str = "Unknown",
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: float | None = None,
) -> dict[int, tuple[str, Decimal] | None]:
    """Call the LLM to read a (unit, unit_quantity) out of each item name.

    Returns a mapping of 1-based idx -> (unit_value, unit_quantity) for items
    with a parseable size, or ``None`` for an item the model explicitly
    answered with no usable size (a count item, a non-product line). Items the
    model dropped from its response are absent from the mapping entirely — that
    distinction lets the caller mark answered rows processed while leaving
    dropped ones to be retried.
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
        label="Unit enrichment",
    )

    out: dict[int, tuple[str, Decimal] | None] = {}
    for raw in inferred:
        if not isinstance(raw, dict):
            continue
        idx = raw.get("idx")
        if isinstance(idx, int):
            out[idx] = _clean_size(raw.get("unit"), raw.get("unit_quantity"))
    return out


def _dec(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def write_unit_size(
    db: DatabaseManager,
    item_id: str,
    item: dict[str, Any],
    sized: tuple[str, Decimal],
) -> tuple[str, str]:
    """Persist a sized ``(unit, unit_quantity)`` for one fact item.

    Writes ``unit`` + ``unit_quantity`` and, when a positive ``total_price`` is
    present and the recomputed price is plausible, ``comparable_unit`` +
    ``comparable_unit_price``. Shared by this pass's per-item write-back and by
    the combined enrichment pass so the comparable-price recompute lives in one
    place. Returns the written ``(unit, unit_quantity)`` as strings.
    """
    unit_value, unit_quantity = sized
    changes: dict[str, str] = {
        "unit": unit_value,
        "unit_quantity": str(unit_quantity),
    }

    total_price = _dec(item.get("total_price"))
    if total_price is not None and total_price > 0:
        data: dict[str, Any] = {
            "total_price": str(total_price),
            "quantity": str(_dec(item.get("quantity")) or Decimal("1")),
            "unit": unit_value,
            "unit_quantity": str(unit_quantity),
        }
        _calculate_comparable_price(data)
        cup = _dec(data.get("comparable_unit_price"))
        cu = data.get("comparable_unit")
        if cup is not None and cu and cup <= _MAX_PLAUSIBLE_UNIT_PRICE:
            changes["comparable_unit_price"] = str(cup)
            changes["comparable_unit"] = str(cu)

    set_clause = ", ".join(f"{col} = ?" for col in changes)
    with db.transaction() as cur:
        cur.execute(
            f"UPDATE fact_items SET {set_clause} WHERE id = ?",  # noqa: S608
            (*changes.values(), item_id),
        )
    return unit_value, str(unit_quantity)


def enrich_items(
    db: DatabaseManager,
    items: list[dict[str, Any]],
    vendor_name: str = "Unknown",
    model: str | None = None,
    ollama_url: str | None = None,
) -> list[UnitResult]:
    """Infer unit/unit_quantity for a batch and write them back.

    Writes ``unit`` + ``unit_quantity`` for every item with a parseable size,
    and additionally ``comparable_unit`` + ``comparable_unit_price`` when a
    total_price is present and the recomputed price is plausible.
    """
    if not items:
        return []

    sizes_by_idx = infer_units(
        items, vendor_name=vendor_name, model=model, ollama_url=ollama_url
    )

    def _write(
        item_id: str, item: dict[str, Any], sized: tuple[str, Decimal]
    ) -> UnitResult:
        unit_value, unit_quantity = write_unit_size(db, item_id, item, sized)
        return UnitResult(item_id, unit_value, unit_quantity, success=True)

    results = apply_answered(
        db,
        items,
        sizes_by_idx,
        mark_column="unit_enriched",
        on_value=_write,
        on_skip=lambda item_id, item: UnitResult(item_id, None, None, success=False),
    )

    enriched = sum(1 for r in results if r.success)
    if enriched:
        logger.info(
            "Unit-enriched %d/%d items for vendor %s",
            enriched,
            len(items),
            vendor_name,
        )
    return results


def enrich_pending_units(
    db: DatabaseManager,
    limit: int = 200,
    model: str | None = None,
    ollama_url: str | None = None,
) -> list[UnitResult]:
    """Find and unit-enrich fact_items lacking a ``unit_quantity``.

    Groups items by vendor for context, then calls the LLM in sub-batches.
    Re-runnable and convergent: a row is selected only while it still lacks a
    ``unit_quantity`` AND has not been marked processed (``unit_enriched``). A
    row the model answers with no usable size is marked once and then skipped,
    so unsolvable rows are no longer re-sent to the LLM on every run.

    Args:
        db: Database manager.
        limit: Max items to process in this run.
        model: Ollama model override.
        ollama_url: Ollama URL override.

    Returns:
        A UnitResult per processed item.
    """
    rows = db.fetchall(
        "SELECT fi.id, fi.name, fi.brand, fi.quantity, fi.total_price, f.vendor "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE fi.unit_quantity IS NULL AND fi.unit_enriched IS NULL "
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
