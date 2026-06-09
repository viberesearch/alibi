# Item-as-Star model — roadmap

The atomic decomposition (`Atom → Bundle → Cloud → Fact → FactItem`) treats each
**FactItem** as the unit of analysis — a "star" that can be filtered/grouped
along any axis independent of its source document, and overlaid against the
**payment** layer (POS/card slips). This roadmap tracks turning that model into
queryable surfaces.

## Done

### A — Multi-axis item filter
`alibi/services/query.py::list_fact_items_with_fact` now filters fact_items by
`name, category, brand, vendor, vendor_key, currency, country`, a date+time
range (`datetime_from/_to`, finer than the date-only bounds) and a price range
(`price_min/_max`); each row is self-describing (carries vendor_key, event_date,
event_time, currency, country). Exposed via `GET /export/line-items/csv`.
Tests: `tests/test_service_query.py::TestItemStarFilters`.

### B — Category enrichment layer
`category_path` (e.g. `food > dairy > milk`) is a hierarchical category on
`fact_items` (migration **v38**), with the flat `category` kept as the leaf.
`alibi/enrichment/categorize.py::enrich_pending_categories` is a decoupled,
local-first LLM pass (mirrors the brand/category inference pass): it batches
items lacking a `category_path`, prompts the structuring model with
`name / name_normalized / comparable_name / brand` against the **controlled
taxonomy** in `alibi/enrichment/taxonomy.py`, validates the returned path, and
writes back `category_path` + leaf via `update_fact_item` (with
`enrichment_source="llm_category"`). Idempotent/re-runnable: only NULL-path
items are selected, so re-running converges coverage without redoing work.
The A filter gained a `category_path` **prefix** axis ("everything under
food"). CLI: `lt enrich categorize`. Tests:
`tests/test_service_categorize.py`, `tests/test_service_query.py::TestCategoryPathFilter`.

The `adjustment > *` taxonomy bucket deliberately captures non-product receipt
lines (tax, tip, service charge, totals, footers) that slip past the pollution
filter, so analytics can exclude them via the same prefix filter.

### C — Item ↔ payment reconciliation view
`alibi/services/reconciliation.py::reconcile` overlays the item and payment
layers and classifies each transaction `matched / items_only / payment_only /
empty`. It reconciles **three amounts** — the fact total, the line-item sum,
and the **normalised payment amount** (summed from `facts.payments`, the
persisted payment-atom data) — surfacing `amount_mismatch` (items vs total) and
`payment_mismatch` (payment vs total). Filters reuse the **fact-level A axes**
(vendor, vendor_key, currency, country, date and date+time ranges); item-level
axes are intentionally excluded because an item predicate would drop every
`payment_only` row. A `coverage` filter narrows to one class, powering the
`payment_only` (card charges with no receipt) and `items_only` (receipts never
matched to a payment) worklists. Surfaced via `GET /api/v1/reconciliation` and
a "Reconcile" Web view (summary tiles + worklist quick-filters + table with
mismatch flags). Tests: `tests/test_service_reconciliation.py`,
`tests/test_api_reconciliation.py`.

## Next session

### D — Item analytics surface (larger)
Once item volume grows, make multi-axis aggregation fast and expose the
cross-vendor price-comparison payload the schema already carries
(`comparable_unit_price`, `comparable_unit`, `product_variant`).

- Storage: a denormalised/indexed item view (generated columns + indexes, or a
  materialised `item_stars` table refreshed on collapse) carrying every star
  axis so filters/aggregations don't re-join `facts` each time.
- Service/API: aggregation endpoints — e.g. avg `comparable_unit_price` for a
  `comparable_name` grouped by `country`/period; price trend for a product
  across vendors; basket composition by category over time.
- Web: an "item sky" view — filter chips (vendor/currency/country/category/
  date-time/price) over a scatter/grid of items, drilling into a FactItem.
- Acceptance: a query like "avg EUR/L of milk in CY in Q1 2026 across vendors"
  returns in one call; filters compose with A's axes.

> This is alibi's thesis made visible — *"the schema creates the capability."*
> The decomposition + self-describing items + comparable_unit_price is exactly
> what enables cross-vendor / -country / -currency comparison; A/C/B/D are the
> surfaces that expose it.
