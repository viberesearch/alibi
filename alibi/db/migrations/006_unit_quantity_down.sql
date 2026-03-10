-- Rollback migration 006
ALTER TABLE line_items DROP COLUMN unit_quantity;

DELETE FROM schema_version WHERE version = 6;
