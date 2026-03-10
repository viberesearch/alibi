-- Migration 034 down: Remove salt column and barcode index.

-- SQLite doesn't support DROP COLUMN before 3.35.0, so recreate the table.
CREATE TABLE api_keys_backup AS SELECT id, user_id, key_hash, key_prefix, label, created_at, last_used_at, is_active FROM api_keys;
DROP TABLE api_keys;
CREATE TABLE api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    label TEXT DEFAULT 'default',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_used_at DATETIME DEFAULT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);
INSERT INTO api_keys SELECT * FROM api_keys_backup;
DROP TABLE api_keys_backup;
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);

DROP INDEX IF EXISTS idx_fact_items_barcode;

DELETE FROM schema_version WHERE version = 34;
