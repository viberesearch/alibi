-- Migration 042 DOWN: remove the categorize idempotency sentinel column

ALTER TABLE fact_items DROP COLUMN category_taxonomy_version;

DELETE FROM schema_version WHERE version = 42;
