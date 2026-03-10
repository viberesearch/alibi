-- Migration 024: User contacts junction table
-- Replaces single email/telegram_user_id columns on users with a 1:N contacts table.
-- Also makes name nullable (no PII by default).

-- 1. Create user_contacts junction table
CREATE TABLE IF NOT EXISTS user_contacts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    contact_type TEXT NOT NULL CHECK(contact_type IN ('telegram', 'email')),
    value TEXT NOT NULL,
    label TEXT DEFAULT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(contact_type, value)
);

CREATE INDEX IF NOT EXISTS idx_user_contacts_user ON user_contacts(user_id);
CREATE INDEX IF NOT EXISTS idx_user_contacts_lookup ON user_contacts(contact_type, value);

-- 2. Migrate existing telegram links
INSERT OR IGNORE INTO user_contacts (id, user_id, contact_type, value)
SELECT lower(hex(randomblob(16))), id, 'telegram', telegram_user_id
FROM users WHERE telegram_user_id IS NOT NULL;

-- 3. Migrate existing emails
INSERT OR IGNORE INTO user_contacts (id, user_id, contact_type, value)
SELECT lower(hex(randomblob(16))), id, 'email', email
FROM users WHERE email IS NOT NULL;

-- 4. Recreate users table without email/telegram, name nullable
PRAGMA foreign_keys = OFF;

CREATE TABLE users_new (
    id TEXT PRIMARY KEY,
    name TEXT DEFAULT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO users_new (id, name, is_active, created_at)
SELECT id, name, is_active, created_at FROM users;

DROP TABLE users;
ALTER TABLE users_new RENAME TO users;

PRAGMA foreign_keys = ON;

-- 5. Drop old unique indexes (table was recreated, they're gone)
-- New indexes are on user_contacts table above

INSERT OR IGNORE INTO schema_version (version) VALUES (24);
