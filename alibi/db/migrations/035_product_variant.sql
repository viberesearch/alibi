-- Migration 035: Add product_variant field to fact_items
-- Stores product subcategory info (e.g., "3%" for milk fat, "L" for egg size)
-- Enables within-variant and cross-variant price comparison

-- UP
ALTER TABLE fact_items ADD COLUMN product_variant TEXT DEFAULT NULL;

INSERT OR IGNORE INTO schema_version (version) VALUES (35);
