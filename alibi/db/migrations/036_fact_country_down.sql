-- Migration 036 DOWN: Remove country/jurisdiction from facts

DROP INDEX IF EXISTS idx_facts_country;

ALTER TABLE facts DROP COLUMN country;

DELETE FROM schema_version WHERE version = 36;
