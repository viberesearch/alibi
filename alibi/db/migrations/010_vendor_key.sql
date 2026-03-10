-- Migration 010: Add vendor_key to facts table
-- Stable vendor identifier for analytics/grouping.
-- Registration-based (e.g. "CY12345678X") or name-hash ("noid_<sha256[:10]>").

-- Up migration
-- ============

ALTER TABLE facts ADD COLUMN vendor_key TEXT;
CREATE INDEX IF NOT EXISTS idx_facts_vendor_key ON facts(vendor_key);

INSERT OR IGNORE INTO schema_version (version) VALUES (10);
