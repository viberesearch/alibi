-- Migration 038: Add hierarchical category_path to fact_items
-- `category` is a sparse, inconsistent per-item extraction field. The category
-- enrichment pass (alibi/enrichment/categorize.py) assigns a normalised,
-- hierarchical path from a controlled taxonomy (e.g. "food > dairy > milk").
-- The flat `category` keeps the leaf segment; `category_path` carries the full
-- path so items can be grouped/filtered at any depth ("everything under food").

-- UP
ALTER TABLE fact_items ADD COLUMN category_path TEXT DEFAULT NULL;

CREATE INDEX IF NOT EXISTS idx_fact_items_category_path
    ON fact_items(category_path);

INSERT OR IGNORE INTO schema_version (version) VALUES (38);
