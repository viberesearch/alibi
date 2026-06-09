-- Migration 038 DOWN: Remove category_path from fact_items

DROP INDEX IF EXISTS idx_fact_items_category_path;

ALTER TABLE fact_items DROP COLUMN category_path;

DELETE FROM schema_version WHERE version = 38;
