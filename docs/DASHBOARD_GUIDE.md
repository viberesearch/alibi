# Dashboard & Analytics Guide

What every number on the web dashboard (`/web`) means, where it comes from, and
the data-quality rules that shape it. The analytics read the materialised
`item_stars` table (one row per line item, denormalised with its parent fact's
vendor / currency / date) and the `facts` table (one row per confirmed
transaction).

> **Currency:** all spend and price figures are normalised to **EUR** using the
> historical exchange rate at each receipt's date. See
> [Currency normalisation](#currency-normalisation). Run `lt fx backfill` after
> importing receipts in a new currency.

---

## Core concepts (the vocabulary)

| Term | Meaning |
|------|---------|
| **comparable_name** | The brand-stripped, English, generic product name (e.g. `milk`, `gouda cheese`). The key that lets the same product be compared across vendors, languages, currencies. Filled by `lt enrich comparable-names` / `lt enrich all`. |
| **comparable_unit** | The standard unit a product's price is measured in: `kg` (weight), `l` (volume), `pcs` (count). **Prices are never compared across different units** — €/kg is not €/L. |
| **comparable_unit_price** | Price per one standard unit (e.g. €4.50/kg). The unit of cross-vendor comparison. |
| **comparable_unit_price_eur** | The same, converted to EUR (what the dashboard shows). |
| **state** | The preservation/preparation form that makes the *same* food a different product: `fresh` / `frozen` / `canned` / `dried` / `cured` / `pickled` / `roasted` / `cooked`. Filled by `lt enrich states`. |
| **attributes** | A flexible JSON map of product facts (`organic`, `size`, `fat_pct`, `state`, …) parsed from the name. Powers the facet chips. |
| **category** | A broad bucket (Dairy, Seafood, Produce, …). **non-product** lines are categorised under the `adjustment` branch — see below. |

---

## Item Sky (`/web` → Item Sky)

The item-level analytics page. Everything here is filtered by the **Filters** card
at the top and the **Facet** chips.

### Filters
| Filter | Effect |
|--------|--------|
| Product | Substring match on `comparable_name`. |
| Category | Hierarchical prefix on `category_path` (e.g. `food > dairy` matches everything under dairy). |
| Vendor / Country / Currency | Exact/substring match on the parent fact. |
| From / To | Event-date window. |
| State (CLI: `--state`) | A product-state facet filter (`fresh`, `canned`, …). |
| Facet chips | Click a chip (e.g. `organic=yes`, `size=L`) to filter to items carrying that attribute. |

### Sky scatter
A scatter of every **priced, dated** line item:
- **x = event date**, **y = comparable_unit_price (log scale)**, **colour = comparable_unit**.
- The x-axis domain is the **2nd–98th percentile** of the dates, so a single
  corrupt date (e.g. an OCR-misread year) can't stretch the axis and crush the
  real data; out-of-range points are pinned at the edges (still hoverable).
- "N items, M plottable" — only items with both a price and a date are plotted.

### Avg comparable price (EUR)
Average **EUR** comparable_unit_price grouped along the chosen axis (`group-by`).
- `comparable_unit` is **always** part of the grouping key, so €/kg, €/L and
  €/pcs rows never blend into one average.
- Columns: **Avg / Unit / Items / Vendors** (distinct vendors contributing).
- Group by any facet with `attr:<key>` (e.g. `attr:state`, `attr:size`).
- CLI: `lt items avg-price --group-by comparable_name,attr:state`.
- Only items with a positive EUR price contribute (NULL / ≤0 excluded — a 0 is a
  missing/garbled price, not a real datum).

> **Why does `milk` sometimes show unit `pcs`, not `l`?** Because those milk
> items have **no parsed volume** (the receipt/name didn't carry "1L"), so they
> fall back to per-piece pricing and can't be compared to the €/L milk. It's a
> data-completeness gap, not a bug — the size simply wasn't extractable. Items
> with a parsed size (`MILK LTR 1L`) appear under `l`.

### Basket composition (spend in EUR)
Total **EUR** spend grouped by category (or any axis via `by`).
- **Non-product receipt lines are excluded** — tax, tip, fee, discount, deposit,
  totals, "non_item" (the taxonomy's `adjustment` branch, and broad-category
  `Non_Item`). A basket's spend is what you *bought*, not the receipt's
  adjustments. CLI: `lt items basket`.

### Price by state (EUR)
For each product sold in **more than one state**, its EUR price per state side by
side (fresh vs canned vs frozen; raw vs roasted). Scoped to real comparisons —
a product seen in only one state is omitted. Grouped within `comparable_unit`.
CLI: `lt items price-by-state --comparable-name salmon`.

---

## Dashboard & Analytics views (fact-level spend)

Spend by vendor / by month / by category, summed over **facts**:
- Each fact's `total_amount` is multiplied by its `eur_rate`, so currencies are
  never blended (a TRY 25,000 receipt is ~€470, not €25,000).
- A fact with no resolvable rate (foreign currency, missing date) is **excluded**
  from EUR totals rather than blended — `lt fx backfill` reports how many.

---

## Currency normalisation

Receipts are recorded in their own currency. Summing them directly would treat
1 CAD or 1 TRY as 1 EUR. Instead:

1. `lt fx backfill` fetches the **historical** EUR rate at each fact's
   `event_date` from [Frankfurter](https://frankfurter.dev) (free, ECB reference
   rates), caches it in `exchange_rates`, and stamps `facts.eur_rate`
   (1.0 for EUR; weekend/holiday → prior business-day rate). For currencies the
   ECB no longer publishes — chiefly **RUB** (dropped March 2022) — it falls
   back to the Central Bank of Russia (`cbr.ru`, also free/no-key) so Russian
   receipts still convert. A date neither source can serve stays NULL.
2. `item_stars` then materialises `total_price_eur` and
   `comparable_unit_price_eur` = the item's amount × the fact's `eur_rate`.
3. Every spend/price aggregation reads the `_eur` columns.

A foreign row not yet converted (NULL `_eur`) drops out of the EUR-only
aggregations — it is never summed back in at face value. `lt fx rates` shows the
cached rates.

---

## Data-quality rules baked into the analytics

These guard the dashboard from the OCR/extraction noise inherent in receipts:

| Rule | Why |
|------|-----|
| x-axis uses the 2nd–98th date percentile | one corrupt date can't blow out the time axis |
| prices ≤ 0 / NULL excluded from price stats | 0 is a garbled price, not a datum |
| non-product (`adjustment`) lines excluded from basket spend | tax/tip/fee aren't products |
| amounts summed in EUR via historical rate | never blend currencies |
| unconvertible foreign rows excluded, not blended | a missing rate must not inflate a total |
| `comparable_unit` always in the price grouping key | €/kg never averaged with €/L |

Genuinely corrupt source values (an €8,916 OCR-misread caviar price, an
impossible year-2616 date) are cleaned at the data level — null the bad field,
keep the line — rather than papered over in the query. Run
[`lt enrich coverage`](#) to see per-field fill / pending counts plus
**item coverage** — the share of purchase facts whose line-item prices sum to
(within 92% of) the fact total. Facts below that are listed worst-first as a
re-extraction queue: `partial` facts have items that fall short (likely missing
lines), `item-less` facts have a total but no items (card-only slips, or severe
under-extraction). Currency symbols/words (`€`, `ΕΥΡΩ`, `₽`, `руб`) and OCR junk
are normalized to ISO codes at ingestion, so no fact carries a junk currency.

---

## Keeping it current

After importing receipts or re-running extraction:

```bash
lt enrich all          # fill comparable_name / unit / category / attributes
lt enrich states       # fill the product-state facet
lt fx backfill         # resolve EUR rates + rebuild item_stars (EUR-normalised)
lt enrich coverage     # check what's filled vs pending
```
