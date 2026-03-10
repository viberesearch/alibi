-- Rollback enrichment provenance columns.
-- SQLite cannot DROP COLUMN before 3.35.0, but Python 3.12 ships 3.43+.
ALTER TABLE fact_items DROP COLUMN enrichment_source;
ALTER TABLE fact_items DROP COLUMN enrichment_confidence;

DELETE FROM schema_version WHERE version = 28;
