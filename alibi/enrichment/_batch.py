"""Shared scaffolding for the local-first, vendor-batched LLM enrichment passes.

``comparable_names``, ``units``, ``attributes`` and ``categorize`` all share the
same shape: call the local structuring model with an emphasis prompt over a
small batch of items, defend against a failed or malformed response, and iterate
the pending ``fact_items`` grouped by vendor in fixed-size sub-batches. This
module factors out that invariant scaffolding so each pass supplies only what is
genuinely pass-specific -- its prompt, its per-item parse, and its write-back.

These are deliberately plain functions (composition), not a base class: the four
passes diverge in their parse/validate and write-back steps (service
``update_fact_item`` vs raw UPDATE, ``{}``-sentinel vs NULL, count-pack recompute
...), so only the two truly-identical blocks are shared here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)


# Phrases the local model emits when it ignores the items already in the prompt
# and replies in prose asking for them ("Please provide the list of items..."),
# instead of returning JSON. A blind retry reproduces it; re-asserting that the
# items are already present (see ``_REINFORCE_SUFFIX``) breaks it out of the loop.
_ASKS_FOR_ITEMS_MARKERS = (
    "provide the list of items",
    "provide the items",
    "provide the line items",
    "list of items you would like",
    "haven't included",
    "you would like me to",
    "provide them",
)

_REINFORCE_SUFFIX = (
    "\n\nThe line items ARE already listed above in this same message. Do not ask "
    "for them and do not reply in prose. Return ONLY the JSON object described, "
    "for those exact items, now."
)


def _asks_for_items(exc: Exception) -> bool:
    """Whether a structuring failure is the model asking for items already given.

    The structurer raises with ``No JSON found in response: <prose>`` when the
    model replies in prose instead of JSON; we only treat the "give me the items"
    variant as retryable, not arbitrary prose.
    """
    msg = str(exc).lower()
    if "no json found in response" not in msg:
        return False
    return any(marker in msg for marker in _ASKS_FOR_ITEMS_MARKERS)


def call_enrichment_llm(
    prompt: str,
    *,
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: float | None = None,
    label: str = "enrichment",
    response_format: dict[str, Any] | None = None,
) -> list[Any]:
    """Run one emphasis-prompt structuring call and return its ``items`` list.

    Centralises the bits every LLM enrichment ``infer_*`` repeated verbatim: the
    timeout default, the ``structure_ocr_text`` call, the broad try/except, and
    the response-shape guard. Returns ``[]`` on any failure or a non-list
    response, so callers can iterate unconditionally.

    If the model replies in prose asking for the items that are already in the
    prompt (a recurring local-model quirk on small/odd batches), the call is
    retried ONCE with a reinforced prompt rather than abandoned, so those rows
    are not perpetually left pending.

    Args:
        prompt: The fully-rendered emphasis prompt for this batch.
        model: Ollama model override (default: config structuring model).
        ollama_url: Ollama URL override.
        timeout: LLM call timeout (default: config ``llm_enrichment_timeout``).
        label: Short pass name used in log messages.
        response_format: Optional JSON schema for this pass's item-list shape.
            When supplied, decoding is constrained to it so the local model
            cannot emit malformed JSON (the cause of the otherwise-deterministic
            "Invalid JSON in response" failures on garbled OCR batches).
    """
    if timeout is None:
        from alibi.config import get_config

        timeout = get_config().llm_enrichment_timeout

    inferred = _structure_items(
        prompt,
        model=model,
        ollama_url=ollama_url,
        timeout=timeout,
        label=label,
        response_format=response_format,
    )
    if inferred is _ASK_FOR_ITEMS:
        logger.warning(
            "%s: model replied asking for the items already in the prompt; "
            "retrying once with a reinforced prompt",
            label,
        )
        inferred = _structure_items(
            prompt + _REINFORCE_SUFFIX,
            model=model,
            ollama_url=ollama_url,
            timeout=timeout,
            label=label,
            retry_on_ask=False,
            response_format=response_format,
        )
    if inferred is None or inferred is _ASK_FOR_ITEMS:
        return []
    if not isinstance(inferred, list):
        logger.warning("%s: LLM returned non-list items: %s", label, type(inferred))
        return []
    return inferred


# Sentinel: the call failed specifically because the model asked for the items.
_ASK_FOR_ITEMS = object()


def _structure_items(
    prompt: str,
    *,
    model: str | None,
    ollama_url: str | None,
    timeout: float | None,
    label: str,
    retry_on_ask: bool = True,
    response_format: dict[str, Any] | None = None,
) -> Any:
    """One structuring call. Returns its ``items``, ``None`` on failure, or the
    ``_ASK_FOR_ITEMS`` sentinel when the model asked for items already supplied
    (only when ``retry_on_ask`` is set, so the reinforced retry can't recurse)."""
    try:
        from alibi.extraction.structurer import structure_ocr_text

        result = structure_ocr_text(
            raw_text="",
            emphasis_prompt=prompt,
            model=model,
            ollama_url=ollama_url,
            timeout=timeout,
            response_format=response_format,
        )
    except Exception as exc:
        if retry_on_ask and _asks_for_items(exc):
            return _ASK_FOR_ITEMS
        logger.exception("%s LLM call failed", label)
        return None

    return result.get("items", [])


def run_vendor_batches(
    rows: Iterable[Any],
    batch_size: int,
    enrich_batch: Callable[[str, list[dict[str, Any]]], list[Any]],
) -> list[Any]:
    """Group pending rows by vendor and enrich them in fixed-size sub-batches.

    Shared by every ``enrich_pending_*`` entry point: it builds the per-vendor
    groups (for context-aware prompting) and walks each group in ``batch_size``
    slices, delegating each slice to ``enrich_batch(vendor_name, items)`` and
    flattening the returned result lists.

    Each row is materialised to a ``dict`` and must carry a ``vendor`` key
    (falls back to ``"Unknown"``).
    """
    vendor_groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        item = dict(row)
        vendor = item.get("vendor") or "Unknown"
        vendor_groups.setdefault(vendor, []).append(item)

    results: list[Any] = []
    for vendor_name, items in vendor_groups.items():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            results.extend(enrich_batch(vendor_name, batch))
    return results


def mark_processed(
    db: "DatabaseManager", column: str, item_ids: Iterable[str], value: int = 1
) -> None:
    """Stamp an idempotency-sentinel column for the given fact_items.

    The ``units`` / ``comparable_names`` / ``categorize`` passes select rows by
    the absence of their result, so a row the model answers with "no result" (a
    count item with no size, a non-product line, an unmappable category) would be
    re-sent to the LLM every run. Calling this with the ids the model actually
    answered stamps them in one statement, mirroring the ``{}`` sentinel the
    attributes pass writes inline. Rows the model dropped or that errored are
    simply not passed here, so they stay unmarked and are retried next run.

    ``value`` is 1 for the boolean markers; ``categorize`` passes the taxonomy
    version it categorised under, so a later ``TAXONOMY_VERSION`` bump makes the
    row eligible again. ``column`` is a trusted internal literal (never user
    input), so the f-string interpolation is safe.
    """
    ids = list(item_ids)
    if not ids:
        return
    placeholders = ", ".join("?" for _ in ids)
    with db.transaction() as cur:
        cur.execute(
            f"UPDATE fact_items SET {column} = ? WHERE id IN ({placeholders})",  # noqa: S608,E501
            (value, *ids),
        )


# Sentinel distinguishing "model dropped this idx" from a present-but-falsy
# answer (None / "" — answered with no usable result): both are skips, but only
# the latter is marked processed.
_DROPPED = object()


def apply_answered(
    db: "DatabaseManager",
    items: list[dict[str, Any]],
    answers: dict[int, Any],
    *,
    mark_column: str,
    on_value: Callable[[str, dict[str, Any], Any], Any],
    on_skip: Callable[[str, dict[str, Any]], Any],
    mark_value: int = 1,
) -> list[Any]:
    """Apply a pass write-back to the model's per-item answers and stamp markers.

    Encodes the idempotency invariant shared by the ``units`` / ``comparable_names``
    / ``categorize`` passes in one place so the three cannot drift:

    * idx absent from ``answers`` -> the model dropped the row; ``on_skip`` and it
      is NOT marked, so a future run retries it.
    * ``answers[idx]`` falsy      -> the model answered "no result"; ``on_skip``
      and it IS marked, so it is not re-sent.
    * ``answers[idx]`` truthy     -> ``on_value(item_id, item, value)``; marked.

    ``answers`` is keyed by 1-based index into ``items`` (as the ``infer_*``
    functions return). ``on_value`` performs the pass-specific write-back and
    returns its result object; ``on_skip`` returns the result for a non-written
    row. ``mark_column`` is stamped with ``mark_value`` (1 for the boolean
    markers, the taxonomy version for categorize). Results are returned in input
    order.
    """
    results: list[Any] = []
    answered_ids: list[str] = []
    for i, item in enumerate(items):
        item_id = item["id"]
        value = answers.get(i + 1, _DROPPED)
        if value is _DROPPED:
            results.append(on_skip(item_id, item))
            continue
        answered_ids.append(item_id)
        if not value:
            results.append(on_skip(item_id, item))
            continue
        results.append(on_value(item_id, item, value))
    mark_processed(db, mark_column, answered_ids, mark_value)
    return results
