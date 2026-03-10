-- Migration 023: Backfill provenance on documents table
-- Pre-session-43 documents lack source and user_id provenance
-- Set source='cli' and user_id='system' for documents with NULL values

UPDATE documents SET source = 'cli', user_id = 'system'
WHERE source IS NULL OR user_id IS NULL;

INSERT OR IGNORE INTO schema_version (version) VALUES (23);
