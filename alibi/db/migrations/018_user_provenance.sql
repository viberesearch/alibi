-- Migration 018: User identity and document provenance
-- Extends users table, adds API keys, adds provenance columns to documents.

-- Extend users table
ALTER TABLE users ADD COLUMN email TEXT DEFAULT NULL;
ALTER TABLE users ADD COLUMN telegram_user_id TEXT DEFAULT NULL;
ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1;

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_telegram ON users(telegram_user_id);

-- API keys: hashed 6-word mnemonic passphrases
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    label TEXT DEFAULT 'default',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_used_at DATETIME DEFAULT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);

-- Document provenance
ALTER TABLE documents ADD COLUMN source TEXT DEFAULT NULL;
ALTER TABLE documents ADD COLUMN user_id TEXT DEFAULT NULL;

INSERT OR IGNORE INTO schema_version (version) VALUES (18);
