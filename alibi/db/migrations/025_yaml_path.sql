-- Add yaml_path column to documents for decoupled YAML store.
-- Stores the path to the .alibi.yaml file when using the yaml_store feature.
ALTER TABLE documents ADD COLUMN yaml_path TEXT DEFAULT NULL;

INSERT OR IGNORE INTO schema_version (version) VALUES (25);
