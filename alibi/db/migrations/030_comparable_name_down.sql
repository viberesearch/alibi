-- Migration 030 down: Remove comparable_name from fact_items.

ALTER TABLE fact_items DROP COLUMN comparable_name;

DELETE FROM schema_version WHERE version = 30;
