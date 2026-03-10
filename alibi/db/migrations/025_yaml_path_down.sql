-- Remove yaml_path column from documents.
ALTER TABLE documents DROP COLUMN yaml_path;

DELETE FROM schema_version WHERE version = 25;
