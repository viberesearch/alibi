-- Down migration 019: Remove system user seed.
-- Note: does not DELETE the user as it may have FK references.
-- Simply records the version rollback.

DELETE FROM schema_version WHERE version = 19;
