-- Revert migration 031: Drop correction_events table.

DROP TABLE IF EXISTS correction_events;
DELETE FROM schema_version WHERE version = 31;
