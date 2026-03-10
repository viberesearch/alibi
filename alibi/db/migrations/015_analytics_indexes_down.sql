-- Revert migration 015: drop analytics indexes

DROP INDEX IF EXISTS idx_facts_event_date;
DROP INDEX IF EXISTS idx_facts_vendor;
DROP INDEX IF EXISTS idx_facts_vendor_key;
DROP INDEX IF EXISTS idx_facts_type_date;
DROP INDEX IF EXISTS idx_fact_items_fact_category;
DROP INDEX IF EXISTS idx_annotations_target;
DROP INDEX IF EXISTS idx_cloud_bundles_cloud;
DROP INDEX IF EXISTS idx_bundles_document;

DELETE FROM schema_version WHERE version = 15;
