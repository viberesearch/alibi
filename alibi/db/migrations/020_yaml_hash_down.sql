-- Remove yaml_hash column from documents.
-- SQLite doesn't support DROP COLUMN before 3.35.0; recreate table.
CREATE TABLE documents_backup AS SELECT
    id, file_path, file_hash, perceptual_hash, raw_extraction,
    source, user_id, ingested_at, created_at
FROM documents;
DROP TABLE documents;
ALTER TABLE documents_backup RENAME TO documents;

DELETE FROM schema_version WHERE version = 20;
