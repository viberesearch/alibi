-- Migration 044: idempotency sentinel for the product-state enrichment pass
--
-- The `states` pass (lt enrich states) assigns a controlled-vocabulary product
-- STATE (fresh / frozen / canned / dried / cured / pickled / roasted / cooked)
-- into the flexible attributes JSON. State distinguishes the SAME comparable_name
-- across forms that are genuinely different products and must not share a price
-- bucket: fresh vs canned artichokes, raw vs roasted cashews, fresh vs smoked
-- salmon. (The comparable_unit already separates volume/weight/count; state is
-- the within-form discriminator the unit alone can't express.)
--
-- Like units / comparable_names (v41) and categorize (v42), the pass selects rows
-- by the absence of its result (no `state` key in attributes), so a row for which
-- no state applies (a generic item, an ambient dry staple, a drink) would be
-- re-sent to the LLM every run. This boolean marker records that a row was
-- processed so an answered-but-stateless row is not re-asked.
--
-- Bookkeeping only — not analytics (the state itself lives in attributes, which
-- IS mirrored to item_stars), so this column is NOT mirrored to item_stars.

-- UP
ALTER TABLE fact_items ADD COLUMN state_enriched INTEGER DEFAULT NULL;

INSERT OR IGNORE INTO schema_version (version) VALUES (44);
