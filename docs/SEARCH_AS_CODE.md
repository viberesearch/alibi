# Search-as-Code: grounded NL -> validated search plan

Implements the fleet "Search / Query as a Generated, Grounded Artifact" pattern for
alibi's fact search. Instead of treating a natural-language query as opaque text
(SQL `LIKE` or an embedding), the LLM compiles it into a typed, validated,
reproducible **search plan** grounded in the live SQLite facets, which then executes
**deterministically** through the existing `services.query` layer.

Code: `alibi/services/search_compiler.py`
Endpoint: `POST /api/v1/search/compile`

## Pipeline

```
NL query
  -> build_search_context(db)            # live registry from the SQLite data
  -> compile_nl_to_plan(q, ctx, llm_fn)  # ONE LLM call; defaults to local Ollama ($0)
  -> validate_plan(plan, ctx)            # structural raise + unknown-term flags
  -> plan_to_search_params(plan)         # deterministic -> list_facts filters
  -> services.query.list_facts(...) / search_facts(...)
```

## Why this beats the opaque `GET /search`

The plain search treats the query as a string: "expensive milk from Alphamega in Q1"
becomes a `LIKE '%...%'` or an embedding lookup — it cannot enforce the schema, flag
an unmappable term, or be replayed deterministically. Search-as-Code adds the four
transferable disciplines:

1. **Grounded in a live registry.** `build_search_context()` reads the real
   `fact_types`, `vendors`, item `categories`, and date/amount ranges from SQLite, so
   the model cannot invent a fact_type or category that doesn't exist.
2. **Flag unknowns, never silently map.** A value outside the live facets lands in
   `plan["flags"]` (e.g. `fact_type "lease" not in live data`), never quietly dropped.
3. **Typed artifact, validated before execution.** `validate_plan()` raises on
   structural errors (bad date, non-numeric amount, bad limit) and flags ungrounded
   facet values.
4. **Deterministic translation + reproducibility.** `plan_to_search_params()` and
   `execute_plan()` have no LLM in the hot path; the plan is the cache/audit key and
   re-runs with **zero LLM calls**.

## Injection safety

The plan never builds SQL by string interpolation: `plan_to_search_params` produces a
filters dict handed to `services.query.list_facts`, which binds every value as a `?`
parameter. The category post-filter is likewise fully parameterised.

## Local-first ($0)

The compile step takes an injectable `llm_fn`; the default calls the local Ollama
structure model (`ollama_structure_model`) in JSON mode. No paid API is in the path.
Tests inject a stub `llm_fn` — no live LLM in CI.

## Usage

```bash
curl -s localhost:8000/api/v1/search/compile \
  -H 'content-type: application/json' \
  -d '{"query": "dairy purchases from Alphamega under 50 EUR in Q1 2026", "limit": 20}'
# -> { "plan": {...}, "flags": [...], "results": [...] }
```

The plain `GET /api/v1/search?q=...` (LIKE / semantic) is **unchanged and additive**.

## Tests

`tests/test_search_compiler.py` — deterministic translation, grounding flags a
hallucinated fact_type/category, structural raises, zero-LLM reproducibility, plus a
real-SQLite end-to-end test (faceted filtering) and an injection-safety test (a
malicious vendor value is bound as a parameter, matches nothing). All pure tests need
no live LLM.
