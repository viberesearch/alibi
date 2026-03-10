-- Migration 018 rollback: Remove user provenance

-- Drop API keys table
DROP TABLE IF EXISTS api_keys;

-- Drop indexes before dropping columns
DROP INDEX IF EXISTS idx_users_email;
DROP INDEX IF EXISTS idx_users_telegram;
DROP INDEX IF EXISTS idx_api_keys_user;
DROP INDEX IF EXISTS idx_api_keys_hash;

-- Remove added columns (SQLite 3.35+ supports DROP COLUMN)
ALTER TABLE users DROP COLUMN email;
ALTER TABLE users DROP COLUMN telegram_user_id;
ALTER TABLE users DROP COLUMN is_active;

ALTER TABLE documents DROP COLUMN source;
ALTER TABLE documents DROP COLUMN user_id;

DELETE FROM schema_version WHERE version = 18;
