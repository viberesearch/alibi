-- Migration 019: Ensure system user exists for existing databases.
-- Fresh installs seed this in DatabaseManager.initialize(), but DBs
-- upgraded from pre-018 may lack the row.

INSERT OR IGNORE INTO users (id, name) VALUES ('system', 'System');

INSERT OR IGNORE INTO schema_version (version) VALUES (19);
