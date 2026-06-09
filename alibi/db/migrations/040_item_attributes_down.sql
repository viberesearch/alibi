-- Migration 040 DOWN: remove the attributes facet column

ALTER TABLE item_stars DROP COLUMN attributes;
ALTER TABLE fact_items DROP COLUMN attributes;

DELETE FROM schema_version WHERE version = 40;
