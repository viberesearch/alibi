-- Migration 030: Add comparable_name column to fact_items.
-- Stores the English product name for cross-language comparison.

ALTER TABLE fact_items ADD COLUMN comparable_name TEXT;

INSERT OR IGNORE INTO schema_version (version) VALUES (30);
