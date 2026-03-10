-- Down migration 014: Remove annotations table.
DROP TABLE IF EXISTS annotations;
DELETE FROM schema_version WHERE version = 14;
