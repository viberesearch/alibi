"""Combined local-LLM enrichment pass: all four fields in one round-trip.

The ``units``, ``comparable_names``, ``categorize`` and ``attributes`` passes
each batch the *same* ``fact_items`` by vendor and round-trip the local
structuring model — four calls over the same rows. But the model already sees
each item's name; one prompt can answer all four field groups at once, cutting
the LLM round-trips ~4x (the bulk live run took ~30 min as four passes).

This module issues that single combined call and writes back every field, reusing
the *existing* validators and write-back helpers so its per-field semantics are
identical to the single passes:

* parse: :func:`units._clean_size`, :func:`comparable_names._clean_comparable_name`,
  :func:`taxonomy.normalize_path`, :func:`attributes._clean_attributes` /
  ``_clean_pack_count`` — no new parse logic.
* write-back: :func:`units.write_unit_size`, :func:`attributes.write_attributes`,
  and ``services.correction.update_fact_item`` for comparable_name / category.
* idempotency: the same per-field marker columns (``unit_enriched``,
  ``comparable_name_enriched``, ``category_taxonomy_version``, the ``{}`` sentinel
  in ``attributes``) via :func:`_batch.apply_answered` — so a field the combined
  model *drops* stays pending and the single-field pass picks it up as a fallback.

Per-row, each field is only written / marked when that row actually needed it
(never overwriting a populated ``comparable_name`` etc.). The units write-back
runs first and mutates the in-memory row, so the attributes count-pack precedence
(only when ``unit_quantity`` is still NULL) is preserved exactly as in the
sequential pipeline. Run ``lt items rebuild`` afterwards to sync item_stars.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from alibi.db.connection import DatabaseManager
from alibi.enrichment import taxonomy
from alibi.enrichment._batch import (
    apply_answered,
    call_enrichment_llm,
    run_vendor_batches,
)
from alibi.enrichment.attributes import (
    _clean_attributes,
    _clean_pack_count,
    write_attributes,
)
from alibi.enrichment.categorize import _CONFIDENCE as _CAT_CONFIDENCE
from alibi.enrichment.categorize import _ENRICHMENT_SOURCE as _CAT_SOURCE
from alibi.enrichment.comparable_names import _CONFIDENCE as _CN_CONFIDENCE
from alibi.enrichment.comparable_names import _ENRICHMENT_SOURCE as _CN_SOURCE
from alibi.enrichment.comparable_names import _clean_comparable_name
from alibi.enrichment.units import _clean_size, write_unit_size

logger = logging.getLogger(__name__)

# Smaller than the single passes (10–12): each item now carries more output, so a
# tighter batch keeps the local model from dropping items off the end of the list.
# Dropped items stay pending and are retried (or caught by the single-field
# fallbacks), so this only trades a few extra calls for a lower drop rate.
_BATCH_SIZE = 8

_PROMPT_TEMPLATE = """\
For each retail/receipt line item below, return ALL of the following in one \
object: the package size, a generic English comparable name, a taxonomy \
category, the product attributes, and (for counted packs) the piece count.
Store: {vendor}

Choose every category_path from THIS taxonomy (and nothing else):
{taxonomy}

Items (idx. name | normalized | brand):
{items_block}

Return JSON only:
{{"items": [
  {{"idx": 1, "unit": "g", "unit_quantity": 450, "comparable_name": "gouda cheese",
    "category_path": "food > dairy > cheese",
    "attributes": {{"type": "gouda", "fat_pct": 48}}, "pack_count": null}},
  {{"idx": 2, "unit": null, "unit_quantity": null, "comparable_name": "eggs large",
    "category_path": "food > dairy > eggs",
    "attributes": {{"size": "L", "free_range": true}}, "pack_count": 12}},
  {{"idx": 3, "unit": null, "unit_quantity": null, "comparable_name": null,
    "category_path": "adjustment > non_item", "attributes": {{}}, "pack_count": null}}
]}}

Rules:
- Return one entry for EVERY item (idx 1..N). Do not omit any. Always include all
  six keys (unit, unit_quantity, comparable_name, category_path, attributes,
  pack_count); use null / {{}} when a field does not apply.
- unit / unit_quantity: read the weight/volume size from the name. unit is one of
  "g", "kg", "ml", "l" ("450G" -> g/450, "2L" -> l/2, "720ML" -> ml/720). For a
  multipack "N x M<unit>" (e.g. "4X70G") unit_quantity = N*M (-> 280). If the name
  has NO weight/volume size (plain counts, non-product lines), use null/null.
- comparable_name: lowercase English generic product type for cross-store
  comparison. STRIP the brand, store, pack/size numbers, units and OCR garble
  (e.g. "450g", "1L", "x12"). Keep a distinguishing variant word only when it
  defines the product ("skimmed milk", "eggs large"). Translate non-English names.
  Non-product lines (tax, tip, discount, deposit, total, subtotal, change,
  payment/card info, receipt footer, OCR noise): null.
- category_path: a full path from the taxonomy above, lowercase, " > " separated;
  the deepest path that fits (a top-level-only path is allowed). Non-product lines
  go under "adjustment" (e.g. "adjustment > tax", "adjustment > non_item"). If
  genuinely undecidable, use "other".
- attributes: an object of ONLY the product facts stated in the name. Prefer these
  standard keys: size ("L"/"M"/"S"/"XL"), type (e.g. "gouda", "skimmed"), color,
  fat_pct (number), organic (bool), free_range (bool), lactose_free (bool),
  gluten_free (bool), light (bool), decaf (bool). You may add other clear facet
  keys. Use {{}} if the name states no facet. NEVER emit identifiers, codes,
  barcodes, prices, totals, tax/VAT, currency, dates, quantities, weights, units,
  the vendor or brand as attributes — those are not facets. Do NOT invent values.
- pack_count: pieces in a counted pack ("EGGS X12" -> 12, "30 EGGS" -> 30). null
  if not a counted pack.
- Only the JSON object, no explanation."""


@dataclass
class CombinedResult:
    """Per-item outcome of the combined pass (which fields it wrote)."""

    item_id: str
    unit_set: bool = False
    comparable_name_set: bool = False
    category_set: bool = False
    attributes_set: bool = False

    @property
    def changed(self) -> bool:
        return (
            self.unit_set
            or self.comparable_name_set
            or self.category_set
            or self.attributes_set
        )


def _needs_unit(item: dict[str, Any]) -> bool:
    return item.get("unit_quantity") is None and item.get("unit_enriched") is None


def _needs_comparable_name(item: dict[str, Any]) -> bool:
    return (
        not item.get("comparable_name") and item.get("comparable_name_enriched") is None
    )


def _needs_category(item: dict[str, Any]) -> bool:
    version = item.get("category_taxonomy_version")
    return not item.get("category_path") and (
        version is None or version < taxonomy.TAXONOMY_VERSION
    )


def _needs_attributes(item: dict[str, Any]) -> bool:
    return item.get("attributes") is None


def _build_items_block(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, item in enumerate(items):
        idx = i + 1
        name = item.get("name") or ""
        norm = item.get("name_normalized") or ""
        brand = item.get("brand") or ""
        lines.append(f"{idx}. {name} | {norm} | {brand}")
    return "\n".join(lines)


def infer_combined(
    items: list[dict[str, Any]],
    vendor_name: str = "Unknown",
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: float | None = None,
) -> tuple[
    dict[int, Any],
    dict[int, str | None],
    dict[int, str | None],
    dict[int, tuple[dict[str, Any], int | None]],
]:
    """Call the model once and split its reply into four per-field answer maps.

    Returns ``(units, comparable_names, categories, attributes)``, each keyed by
    1-based idx and following the single passes' convention: a key *present* in
    the model's item object (even as ``null``) is an answered value (possibly a
    validated ``None``); a key *absent* means the model dropped that field, so it
    is omitted from the map and stays pending for a later run / fallback pass.
    """
    empty: tuple[dict[int, Any], dict[int, Any], dict[int, Any], dict[int, Any]] = (
        {},
        {},
        {},
        {},
    )
    if not items:
        return empty  # type: ignore[return-value]

    prompt = _PROMPT_TEMPLATE.format(
        vendor=vendor_name,
        taxonomy=taxonomy.render_for_prompt(),
        items_block=_build_items_block(items),
    )

    inferred = call_enrichment_llm(
        prompt,
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
        label="Combined enrichment",
    )

    units: dict[int, Any] = {}
    names: dict[int, str | None] = {}
    cats: dict[int, str | None] = {}
    attrs: dict[int, tuple[dict[str, Any], int | None]] = {}
    for raw in inferred:
        if not isinstance(raw, dict):
            continue
        idx = raw.get("idx")
        if not isinstance(idx, int):
            continue
        if "unit" in raw or "unit_quantity" in raw:
            units[idx] = _clean_size(raw.get("unit"), raw.get("unit_quantity"))
        if "comparable_name" in raw:
            names[idx] = _clean_comparable_name(raw.get("comparable_name"))
        if "category_path" in raw:
            cats[idx] = taxonomy.normalize_path(raw.get("category_path"))
        if "attributes" in raw:
            attrs[idx] = (
                _clean_attributes(raw.get("attributes")),
                _clean_pack_count(raw.get("pack_count")),
            )
    return units, names, cats, attrs


def _apply_field(
    db: DatabaseManager,
    items: list[dict[str, Any]],
    answers_by_idx: dict[int, Any],
    needs: Callable[[dict[str, Any]], bool],
    *,
    mark_column: str,
    on_value: Callable[[str, dict[str, Any], Any], Any],
    mark_value: int = 1,
) -> None:
    """Apply one field's answers to just the rows that need it.

    Re-keys the combined-batch answers onto the sub-list of rows still missing
    this field, then defers to :func:`apply_answered` so the answered-vs-dropped
    marking invariant is shared with the single passes (rather than re-derived).
    """
    sub_items: list[dict[str, Any]] = []
    sub_answers: dict[int, Any] = {}
    for i, item in enumerate(items):
        if not needs(item):
            continue
        new_idx = len(sub_items) + 1
        sub_items.append(item)
        combined_idx = i + 1
        if combined_idx in answers_by_idx:
            sub_answers[new_idx] = answers_by_idx[combined_idx]
    if not sub_items:
        return
    apply_answered(
        db,
        sub_items,
        sub_answers,
        mark_column=mark_column,
        on_value=on_value,
        on_skip=lambda item_id, item: None,
        mark_value=mark_value,
    )


def enrich_items(
    db: DatabaseManager,
    items: list[dict[str, Any]],
    vendor_name: str = "Unknown",
    model: str | None = None,
    ollama_url: str | None = None,
) -> list[CombinedResult]:
    """Run the combined call for one batch and write every needed field back."""
    if not items:
        return []

    units, names, cats, attrs = infer_combined(
        items, vendor_name=vendor_name, model=model, ollama_url=ollama_url
    )

    results = {item["id"]: CombinedResult(item["id"]) for item in items}

    # Units first: write_unit_size mutates the in-memory row's unit_quantity so the
    # attributes count-pack step below keeps its "only when unit_quantity is NULL"
    # precedence, exactly as the sequential pipeline behaves.
    def _on_unit(item_id: str, item: dict[str, Any], sized: Any) -> None:
        # sized is a truthy (unit, Decimal) tuple — apply_answered routes the
        # falsy "no size" answers to on_skip, never here.
        _, unit_quantity = write_unit_size(db, item_id, item, sized)
        item["unit_quantity"] = unit_quantity
        results[item_id].unit_set = True

    _apply_field(
        db, items, units, _needs_unit, mark_column="unit_enriched", on_value=_on_unit
    )

    from alibi.services.correction import update_fact_item

    # comparable_name before category so enrichment_source ends as "llm_category",
    # matching the standing single-pass order.
    def _on_name(item_id: str, item: dict[str, Any], name: str) -> None:
        update_fact_item(
            db,
            item_id,
            {
                "comparable_name": name,
                "enrichment_source": _CN_SOURCE,
                "enrichment_confidence": _CN_CONFIDENCE,
            },
        )
        results[item_id].comparable_name_set = True

    _apply_field(
        db,
        items,
        names,
        _needs_comparable_name,
        mark_column="comparable_name_enriched",
        on_value=_on_name,
    )

    def _on_category(item_id: str, item: dict[str, Any], path: str) -> None:
        update_fact_item(
            db,
            item_id,
            {
                "category_path": path,
                "category": taxonomy.leaf_of(path),
                "enrichment_source": _CAT_SOURCE,
                "enrichment_confidence": _CAT_CONFIDENCE,
            },
        )
        results[item_id].category_set = True

    _apply_field(
        db,
        items,
        cats,
        _needs_category,
        mark_column="category_taxonomy_version",
        on_value=_on_category,
        mark_value=taxonomy.TAXONOMY_VERSION,
    )

    # Attributes keep their own {}-sentinel shape (the attributes column itself is
    # the marker), so they don't go through apply_answered.
    for i, item in enumerate(items):
        if not _needs_attributes(item):
            continue
        answer = attrs.get(i + 1)
        if answer is None:  # model dropped attributes for this row -> retry later
            continue
        attr_map, pack = answer
        write_attributes(db, item, attr_map, pack)
        results[item["id"]].attributes_set = True

    out = [results[item["id"]] for item in items]
    changed = sum(1 for r in out if r.changed)
    if changed:
        logger.info(
            "Combined-enriched %d/%d items for vendor %s",
            changed,
            len(items),
            vendor_name,
        )
    return out


def enrich_pending_combined(
    db: DatabaseManager,
    limit: int = 200,
    model: str | None = None,
    ollama_url: str | None = None,
) -> list[CombinedResult]:
    """Find fact_items missing ANY local-LLM field and enrich them in one call.

    Selects rows that still need at least one of unit / comparable_name /
    category / attributes (each guarded by its own marker so converged rows are
    skipped), groups them by vendor for context, and issues a single combined
    LLM call per sub-batch. Each field is written and marked only for the rows
    that needed it; a field the model drops stays pending for the next run or the
    single-field fallback pass. Run ``lt items rebuild`` afterwards.
    """
    rows = db.fetchall(
        "SELECT fi.id, fi.name, fi.name_normalized, fi.brand, fi.comparable_name, "
        "       fi.category, fi.category_path, fi.quantity, fi.total_price, "
        "       fi.unit_quantity, fi.product_variant, fi.attributes, "
        "       fi.unit_enriched, fi.comparable_name_enriched, "
        "       fi.category_taxonomy_version, f.vendor "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE fi.name IS NOT NULL AND fi.name != '' "
        "AND ("
        "  (fi.unit_quantity IS NULL AND fi.unit_enriched IS NULL) "
        "  OR ((fi.comparable_name IS NULL OR fi.comparable_name = '') "
        "      AND fi.comparable_name_enriched IS NULL) "
        "  OR ((fi.category_path IS NULL OR fi.category_path = '') "
        "      AND (fi.category_taxonomy_version IS NULL "
        "           OR fi.category_taxonomy_version < ?)) "
        "  OR fi.attributes IS NULL"
        ") "
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
