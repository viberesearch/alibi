-- Migration 023 (down): Revert provenance backfill
-- Restore NULL values for source and user_id on backfilled documents

UPDATE documents SET source = NULL, user_id = NULL
WHERE source = 'cli' AND user_id = 'system';

DELETE FROM schema_version WHERE version = 23;
