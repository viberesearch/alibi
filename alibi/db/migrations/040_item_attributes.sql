-- Migration 040: flexible per-item attribute facets (JSON)
--
-- product_variant is a single scalar ("L", "3%") — too narrow for items that
-- carry several price-influencing parameters at once (eggs: size=L + organic +
-- free_range; milk: fat + lactose_free; cheese: fat + type). This adds an
-- `attributes` JSON map so any number of fact-based facets can be captured and
-- filtered independently (json_extract(attributes,'$.organic') = 1).
--
-- Keys are lightly normalised to a standard vocabulary by the extraction pass
-- (alibi/enrichment/attributes.py) so filtering is consistent; product_variant
-- stays as the primary/display facet for back-compat.

-- UP
ALTER TABLE fact_items ADD COLUMN attributes JSON DEFAULT NULL;
ALTER TABLE item_stars ADD COLUMN attributes JSON DEFAULT NULL;

INSERT OR IGNORE INTO schema_version (version) VALUES (40);
