# ADR: Product Enrichment Pipeline

## Status

Accepted and implemented.

## Context

Receipt line items come with minimal structure — a product name, quantity, and price. For meaningful analytics (cross-vendor price comparison, spending by category, nutritional tracking), each item needs standardized enrichment: brand, category, unit_quantity (comparable package size), and comparable_name (standardized English name for cross-language comparison).

No single data source covers all products. Open Food Facts has excellent barcode data but limited coverage for local/regional products. LLMs can infer brand and category but hallucinate on specific details. Historical data from the system's own database is the most reliable but only works for repeat purchases.

## Decision

Implement an 8-tier enrichment cascade where each tier has:
- A confidence score (0.0 to 1.0)
- Provenance tracking (which source enriched which field)
- A user feedback loop (confirm/reject enrichment decisions)

**Cascade order** (highest confidence first):

1. **Historical lookup** (0.95) — same product at same vendor from previous extractions
2. **Open Food Facts barcode** (0.95) — global product database, 3M+ products
3. **UPCitemdb barcode** (0.90) — 686M item database, 100 req/day free tier
4. **GS1 prefix** (0.80) — company prefix decoding from barcode, brand propagation
5. **FTS5 fuzzy name match** (varies) — full-text search against product_cache
6. **Local LLM inference** (0.70) — Ollama model infers brand/category from name
7. **Gemini mega-batch** (0.85) — single API call for all items (brand + category + unit_quantity + comparable_name)
8. **Anthropic cloud** (0.85) — refinement for remaining unknowns

Each tier only fills fields that previous tiers left empty. A field enriched at confidence 0.95 by Open Food Facts will not be overwritten by a 0.70 LLM inference.

## Key Fields

| Field | Purpose | Example |
|-------|---------|---------|
| `brand` | Product manufacturer | "Alphamega", "Barilla" |
| `category` | Spending category | "Dairy", "Pasta", "Fuel" |
| `unit_quantity` | Comparable package size | 1.0 (for "Milk 1L"), 0.5 (for "Cheese 500g") |
| `unit` | Unit type | kg, l, pcs, pack |
| `comparable_name` | Standardized English name | "whole milk" (from "ΓΑΛΑ ΠΛΗΡΕΣ") |
| `barcode` | EAN/UPC code | "5201054025123" |
| `enrichment_source` | Which tier enriched | "openfoodfacts", "gemini", "user_confirmed" |
| `enrichment_confidence` | How confident | 0.95, 0.70 |

## Consequences

**Positive:**
- Cross-vendor price comparison via `comparable_name` + `unit_quantity` (e.g., "milk costs EUR 1.35/L at Alphamega vs EUR 1.20/L at Papas")
- Spending analytics by category without manual tagging
- Nutritional tracking via Open Food Facts barcode data
- System improves over time: confirmed enrichments become historical data for tier 1

**Negative:**
- Complexity: 8 tiers with different APIs, rate limits, and failure modes
- External dependencies: OFF API, UPCitemdb, Gemini, Anthropic (all optional, gated by feature flags)
- Cold start: first few documents have no historical data; enrichment quality improves with volume
