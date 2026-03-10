-- Migration 017: Drop dormant tags, consumers, and related tables
-- These features are superseded by the annotations system (migration 014).

DROP TABLE IF EXISTS line_item_allocations;
DROP TABLE IF EXISTS consumers;
DROP TABLE IF EXISTS item_tags;
DROP TABLE IF EXISTS vendor_patterns;
DROP TABLE IF EXISTS tags;

INSERT OR IGNORE INTO schema_version (version) VALUES (17);
