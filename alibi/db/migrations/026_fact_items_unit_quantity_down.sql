ALTER TABLE fact_items DROP COLUMN unit_quantity;

DELETE FROM schema_version WHERE version = 26;
