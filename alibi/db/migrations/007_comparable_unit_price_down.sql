-- Rollback migration 007
ALTER TABLE line_items DROP COLUMN comparable_unit_price;
ALTER TABLE line_items DROP COLUMN comparable_unit;

DELETE FROM schema_version WHERE version = 7;
