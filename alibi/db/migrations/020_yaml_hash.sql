-- Add yaml_hash column to documents for YAML-first pipeline change detection.
-- Stores SHA-256 of the .alibi.yaml file so we can detect admin corrections.
ALTER TABLE documents ADD COLUMN yaml_hash TEXT DEFAULT NULL;

INSERT OR IGNORE INTO schema_version (version) VALUES (20);
