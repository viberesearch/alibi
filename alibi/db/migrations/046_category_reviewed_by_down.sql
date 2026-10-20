-- Migration 046 DOWN: remove the Jev category review marker column

ALTER TABLE fact_items DROP COLUMN category_reviewed_by;

DELETE FROM schema_version WHERE version = 46;
