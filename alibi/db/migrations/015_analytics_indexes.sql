-- Migration 015: Analytics indexes
-- Supports Metabase dashboards, analytics queries, and service layer performance.

INSERT OR IGNORE INTO schema_version (version) VALUES (15);

-- Facts: date-range queries (spending by month, reports)
CREATE INDEX IF NOT EXISTS idx_facts_event_date ON facts(event_date);

-- Facts: vendor filtering and grouping
CREATE INDEX IF NOT EXISTS idx_facts_vendor ON facts(vendor);

-- Facts: vendor_key lookups (identity resolution, deduplication)
CREATE INDEX IF NOT EXISTS idx_facts_vendor_key ON facts(vendor_key);

-- Facts: type + date composite for filtered spending queries
CREATE INDEX IF NOT EXISTS idx_facts_type_date ON facts(fact_type, event_date);

-- Fact items: category grouping per fact
CREATE INDEX IF NOT EXISTS idx_fact_items_fact_category ON fact_items(fact_id, category);

-- Annotations: lookup by target entity
CREATE INDEX IF NOT EXISTS idx_annotations_target ON annotations(target_type, target_id);

-- Cloud bundles: cloud membership lookups
CREATE INDEX IF NOT EXISTS idx_cloud_bundles_cloud ON cloud_bundles(cloud_id);

-- Bundles: document relationship
CREATE INDEX IF NOT EXISTS idx_bundles_document ON bundles(document_id);
