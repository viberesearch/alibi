"""Search-as-Code: compile NL -> validated search plan -> deterministic query.

Implements the "Search / Query as a Generated, Grounded Artifact" pattern
for alibi's fact search. Instead of treating a natural-language query as opaque
text (LIKE / embedding), the LLM compiles it into a typed, validated, reproducible
*search plan* grounded in the live SQLite facets (real vendors, fact_types,
categories, date/amount ranges). The plan then executes DETERMINISTICALLY through
the existing ``services.query`` layer.

PIPELINE:
    NL query
      -> build_search_context(db)            # live registry from the SQLite data
      -> compile_nl_to_plan(q, ctx, llm_fn)  # ONE LLM call (defaults to local Ollama)
      -> validate_plan(plan, ctx)            # structural raise + unknown-term flags
      -> plan_to_search_params(plan)         # deterministic -> list_facts filters
      -> query.list_facts(...) / search_facts(...)

The four disciplines (see the Search-as-Code pattern): ground the LLM in a
live registry; flag unknowns, never silently map; the plan is a typed, validated
artifact; plan -> execution is deterministic (the plan is the reproducibility key,
re-run with ZERO LLM calls).

ADDITIVE: the existing ``GET /api/v1/search`` (LIKE / semantic) is untouched. This
adds a grounded, structured path in front of the same ``services.query`` functions.

LOCAL-FIRST: the compile step takes an injectable ``llm_fn``. The default calls the
local Ollama structure model ($0); tests inject a stub so no live LLM runs in CI.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
from collections.abc import Callable
from typing import Any

from alibi.db.connection import DatabaseManager
from alibi.services import query as svc_query

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "0.1"
DEFAULT_LIMIT = 50
MAX_LIMIT = 200
MAX_FACET_VALUES = 50

# An llm_fn takes (system_prompt, user_prompt) and returns raw text (a JSON plan).
LLMFn = Callable[[str, str], str]

# Filter keys the plan may carry. These map 1:1 to services.query.list_facts, except
# `category` (applied as a deterministic post-filter over fact_items).
_LIST_FACTS_KEYS = (
    "vendor",
    "fact_type",
    "date_from",
    "date_to",
    "min_amount",
    "max_amount",
)


# ---------------------------------------------------------------------------
# build_search_context — live registry (the anti-hallucination surface)
# ---------------------------------------------------------------------------


def build_search_context(db: DatabaseManager) -> dict[str, Any]:
    """Build the live registry the LLM is grounded in, from the SQLite facts.

    Returns {schema_version, facets: {fact_types, vendors, categories,
    date_min, date_max, amount_min, amount_max, count}}.
    """

    def _distinct(sql: str) -> list[str]:
        try:
            rows = db.fetchall(sql)
        except Exception:  # pragma: no cover - empty/uninitialised db
            return []
        out = [str(r[0]) for r in rows if r[0] not in (None, "")]
        return sorted(set(out))[:MAX_FACET_VALUES]

    fact_types = _distinct("SELECT DISTINCT fact_type FROM facts")
    vendors = _distinct("SELECT DISTINCT vendor FROM facts")
    categories = _distinct("SELECT DISTINCT category FROM fact_items")

    rng = db.fetchone(
        "SELECT MIN(event_date), MAX(event_date), "
        "MIN(CAST(total_amount AS REAL)), MAX(CAST(total_amount AS REAL)) FROM facts"
    )
    date_min = str(rng[0]) if rng and rng[0] is not None else None
    date_max = str(rng[1]) if rng and rng[1] is not None else None
    amount_min = rng[2] if rng and rng[2] is not None else None
    amount_max = rng[3] if rng and rng[3] is not None else None

    count_row = db.fetchone("SELECT COUNT(*) FROM facts")
    count = count_row[0] if count_row else 0

    return {
        "schema_version": SCHEMA_VERSION,
        "facets": {
            "fact_types": fact_types,
            "vendors": vendors,
            "categories": categories,
            "date_min": date_min,
            "date_max": date_max,
            "amount_min": amount_min,
            "amount_max": amount_max,
            "count": count,
        },
    }


# ---------------------------------------------------------------------------
# Validation — structural raise + unknown-term flags (never silent)
# ---------------------------------------------------------------------------


def _is_iso_date(s: Any) -> bool:
    try:
        dt.date.fromisoformat(str(s)[:10])
        return True
    except (ValueError, TypeError):
        return False


def validate_plan(plan: dict[str, Any], context: dict[str, Any]) -> list[str]:
    """Validate a search plan against the DSL and the live facets.

    Returns flags for unknown facet values (never silently mapped). Raises
    ValueError on structural errors (bad schema version, malformed date,
    non-numeric amount, bad limit).
    """
    flags: list[str] = []

    if plan.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION!r}, "
            f"got {plan.get('schema_version')!r}"
        )

    facets = context.get("facets", {})
    filters = plan.get("filters") or {}
    if not isinstance(filters, dict):
        raise ValueError("plan.filters must be an object")

    known_keys = set(_LIST_FACTS_KEYS) | {"category"}
    for key in filters:
        if key not in known_keys:
            flags.append(f"filters.{key}: unknown filter key (ignored)")

    # fact_type: flag if not a live value
    fact_type = filters.get("fact_type")
    live_types = set(facets.get("fact_types", []) or [])
    if fact_type and live_types and str(fact_type) not in live_types:
        flags.append(
            f"filters.fact_type: {fact_type!r} not in live data "
            f"(known: {sorted(live_types)})"
        )

    # category: flag if not a live value
    category = filters.get("category")
    live_cats = set(facets.get("categories", []) or [])
    if category and live_cats and str(category) not in live_cats:
        flags.append(
            f"filters.category: {category!r} not in live data "
            f"(known: {sorted(live_cats)[:15]})"
        )

    # vendor: informational flag if no live vendor contains it
    vendor = filters.get("vendor")
    vendors = facets.get("vendors", []) or []
    if (
        vendor
        and vendors
        and not any(str(vendor).lower() in v.lower() for v in vendors)
    ):
        flags.append(f"filters.vendor: no known vendor contains {vendor!r}")

    # dates
    for key in ("date_from", "date_to"):
        v = filters.get(key)
        if v and not _is_iso_date(v):
            raise ValueError(f"filters.{key}={v!r} is not an ISO date")

    # amounts
    for key in ("min_amount", "max_amount"):
        v = filters.get(key)
        if v is not None:
            try:
                float(v)
            except (TypeError, ValueError):
                raise ValueError(f"filters.{key}={v!r} is not a number")

    limit = plan.get("limit", DEFAULT_LIMIT)
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError(f"plan.limit must be a positive integer, got {limit!r}")

    return flags


# ---------------------------------------------------------------------------
# compile_nl_to_plan — NL -> plan (one LLM call; defaults to local Ollama)
# ---------------------------------------------------------------------------


def _build_system_prompt(context: dict[str, Any]) -> str:
    f = context.get("facets", {})

    def show(key: str, n: int = 30) -> str:
        vals = f.get(key, []) or []
        return str(vals[:n]) if vals else "(none)"

    return (
        "You are a search-plan compiler for a personal finance/receipts database.\n"
        "Translate the user's question into a JSON search plan.\n\n"
        "OUTPUT: ONLY a JSON object. No prose, no markdown fences.\n\n"
        "PLAN SHAPE:\n"
        "{\n"
        f'  "schema_version": "{SCHEMA_VERSION}",\n'
        '  "query_text": "<free-text vendor/item terms, or empty>",\n'
        '  "filters": {\n'
        '    "vendor": "<substring of a known vendor>" | null,\n'
        '    "fact_type": "<one known fact_type>" | null,\n'
        '    "category": "<one known item category>" | null,\n'
        '    "date_from": "YYYY-MM-DD" | null,\n'
        '    "date_to": "YYYY-MM-DD" | null,\n'
        '    "min_amount": <number> | null,\n'
        '    "max_amount": <number> | null\n'
        "  },\n"
        '  "limit": <int>,\n'
        '  "flags": ["<note for any term you could NOT ground in the facets>"]\n'
        "}\n\n"
        "HARD RULES:\n"
        "- Use ONLY facet values that appear below for fact_type and category. Do\n"
        "  NOT invent them; if the question implies one not listed, leave it null\n"
        "  and add a note to 'flags'.\n"
        "- Put brand/product/free words in query_text; put a concrete store name in\n"
        "  filters.vendor.\n"
        '- Resolve relative dates ("last month", "Q1 2026") into date_from/date_to.\n'
        "- Default limit 50.\n\n"
        "LIVE FACETS (the only values that exist):\n"
        f"  fact_types: {show('fact_types')}\n"
        f"  categories: {show('categories')}\n"
        f"  example vendors: {show('vendors', 20)}\n"
        f"  date range: {f.get('date_min')} .. {f.get('date_max')}\n"
        f"  amount range: {f.get('amount_min')} .. {f.get('amount_max')}\n"
        f"  total facts: {f.get('count', 0)}\n"
    )


def _parse_plan_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            ln for ln in text.split("\n") if not ln.strip().startswith("```")
        ).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        plan = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM output is not valid JSON: {exc}\n---\n{raw}") from exc
    if not isinstance(plan, dict):
        raise ValueError(f"LLM output parsed as {type(plan).__name__}, expected object")
    return plan


def _default_llm_fn(system_prompt: str, user_prompt: str) -> str:
    """Local Ollama JSON-mode generate (prefer_local => $0)."""
    import httpx

    from alibi.config import get_config

    config = get_config()
    payload = {
        "model": config.ollama_structure_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
        "keep_alive": config.ollama_keep_alive,
    }
    url = config.ollama_url.rstrip("/") + "/api/chat"
    timeout = float(os.environ.get("ALIBI_STRUCTURE_TIMEOUT", "600"))
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        content: str = resp.json()["message"]["content"]
        return content


def compile_nl_to_plan(
    nl_text: str,
    context: dict[str, Any],
    llm_fn: LLMFn | None = None,
) -> dict[str, Any]:
    """Compile a natural-language query into a validated search plan.

    Exactly one LLM call. ``llm_fn`` defaults to a local Ollama generate (prefer_local, $0). Tests inject a stub (no live LLM in CI).
    """
    system = _build_system_prompt(context)
    fn = llm_fn or _default_llm_fn
    raw = fn(system, nl_text)
    plan = _parse_plan_json(raw)

    plan.setdefault("schema_version", SCHEMA_VERSION)
    plan.setdefault("query_text", nl_text)
    plan.setdefault("filters", {})
    plan.setdefault("limit", DEFAULT_LIMIT)
    plan.setdefault("flags", [])

    new_flags = validate_plan(plan, context)
    plan["flags"] = list(plan.get("flags") or []) + new_flags
    return plan


# ---------------------------------------------------------------------------
# plan_to_search_params — deterministic translation
# ---------------------------------------------------------------------------


def plan_to_search_params(plan: dict[str, Any]) -> dict[str, Any]:
    """Translate a validated plan into deterministic execution parameters.

    Returns {filters: {<list_facts keys>}, category: str|None, query_text: str,
    limit: int}. No LLM. Same plan -> identical params.
    """
    raw = plan.get("filters") or {}
    filters: dict[str, Any] = {}
    for key in _LIST_FACTS_KEYS:
        val = raw.get(key)
        if val is not None and val != "":
            if key in ("min_amount", "max_amount"):
                filters[key] = float(val)
            else:
                filters[key] = str(val) if key not in () else val
    limit = int(plan.get("limit", DEFAULT_LIMIT))
    limit = max(1, min(limit, MAX_LIMIT))
    return {
        "filters": filters,
        "category": (raw.get("category") or None),
        "query_text": str(plan.get("query_text") or "").strip(),
        "limit": limit,
    }


def _filter_by_category(
    db: DatabaseManager, facts: list[dict[str, Any]], category: str
) -> list[dict[str, Any]]:
    """Keep facts that have at least one item in `category` (deterministic)."""
    if not facts:
        return []
    ids = [f["id"] for f in facts]
    placeholders = ",".join("?" for _ in ids)
    rows = db.fetchall(
        f"SELECT DISTINCT fact_id FROM fact_items "  # noqa: S608 - placeholders are param-bound
        f"WHERE category = ? AND fact_id IN ({placeholders})",
        tuple([category, *ids]),
    )
    keep = {r[0] for r in rows}
    return [f for f in facts if f["id"] in keep]


# ---------------------------------------------------------------------------
# execute / compile_and_search
# ---------------------------------------------------------------------------


def execute_plan(
    plan: dict[str, Any], db: DatabaseManager, context: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Execute a validated plan deterministically. ZERO LLM calls.

    Returns {plan, flags, params, facts, total}.
    """
    if context is not None:
        extra = validate_plan(plan, context)
        plan = {**plan, "flags": list(plan.get("flags") or []) + extra}

    params = plan_to_search_params(plan)
    filters = params["filters"]
    query_text = params["query_text"]
    limit = params["limit"]
    category = params["category"]

    # Fetch a wider pool when we still need to post-filter by category.
    fetch_limit = limit if not category else min(MAX_LIMIT, limit * 4)

    if filters:
        result = svc_query.list_facts(db, filters=filters, offset=0, limit=fetch_limit)
        facts = result["facts"]
    elif query_text:
        result = svc_query.search_facts(db, query_text, offset=0, limit=fetch_limit)
        facts = result["facts"]
    else:
        result = svc_query.list_facts(db, filters={}, offset=0, limit=fetch_limit)
        facts = result["facts"]

    if category:
        facts = _filter_by_category(db, facts, category)

    facts = facts[:limit]
    return {
        "plan": plan,
        "flags": plan.get("flags", []),
        "params": params,
        "facts": facts,
        "total": len(facts),
    }


def compile_and_search(
    nl_text: str,
    db: DatabaseManager,
    llm_fn: LLMFn | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full pipeline: build registry -> compile (1 LLM call) -> execute."""
    context = context or build_search_context(db)
    plan = compile_nl_to_plan(nl_text, context, llm_fn)
    return execute_plan(plan, db, context=None)
