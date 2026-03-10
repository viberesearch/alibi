-- Down migration: drop the unique index. Deleted duplicate data cannot be recovered.
DROP INDEX IF EXISTS idx_facts_cloud_unique;
DELETE FROM schema_version WHERE version = 21;
