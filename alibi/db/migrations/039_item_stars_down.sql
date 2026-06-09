-- Migration 039 DOWN: Remove the materialised item_stars table

DROP INDEX IF EXISTS idx_item_stars_event_date;
DROP INDEX IF EXISTS idx_item_stars_currency;
DROP INDEX IF EXISTS idx_item_stars_country;
DROP INDEX IF EXISTS idx_item_stars_vendor_key;
DROP INDEX IF EXISTS idx_item_stars_category_path;
DROP INDEX IF EXISTS idx_item_stars_comparable_name;
DROP INDEX IF EXISTS idx_item_stars_fact;

DROP TABLE IF EXISTS item_stars;

DELETE FROM schema_version WHERE version = 39;
