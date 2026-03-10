-- Migration 024 (down): Revert user_contacts back to columns on users

PRAGMA foreign_keys = OFF;

-- 1. Recreate users table with email/telegram columns
CREATE TABLE users_new (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL DEFAULT 'user',
    email TEXT DEFAULT NULL,
    telegram_user_id TEXT DEFAULT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 2. Migrate data back (take first contact of each type per user)
INSERT INTO users_new (id, name, is_active, created_at)
SELECT id, COALESCE(name, 'user'), is_active, created_at FROM users;

UPDATE users_new SET telegram_user_id = (
    SELECT value FROM user_contacts
    WHERE user_contacts.user_id = users_new.id AND contact_type = 'telegram'
    LIMIT 1
);

UPDATE users_new SET email = (
    SELECT value FROM user_contacts
    WHERE user_contacts.user_id = users_new.id AND contact_type = 'email'
    LIMIT 1
);

DROP TABLE users;
ALTER TABLE users_new RENAME TO users;

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_telegram ON users(telegram_user_id);

-- 3. Drop user_contacts table
DROP TABLE IF EXISTS user_contacts;

PRAGMA foreign_keys = ON;

DELETE FROM schema_version WHERE version = 24;
