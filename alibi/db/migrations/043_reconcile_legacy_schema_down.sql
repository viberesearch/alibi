-- Down migration 043: restore the pre-reconciliation legacy drift.
--
-- Rebuilds `items` in its looser v1 shape and recreates the empty
-- `space_members` table, so reverting to version 42 reproduces the schema that
-- migration 043 reconciled.

PRAGMA foreign_keys = OFF;

CREATE TABLE items_old (
    id TEXT PRIMARY KEY,
    space_id TEXT,
    name TEXT,
    category TEXT,
    model TEXT,
    serial_number TEXT,
    purchase_date DATE,
    purchase_price DECIMAL,
    current_value DECIMAL,
    currency TEXT DEFAULT 'EUR',
    status TEXT,
    warranty_expires DATE,
    warranty_type TEXT,
    insurance_covered BOOLEAN,
    note_path TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    modified_at DATETIME,
    created_by TEXT
);

INSERT INTO items_old
    SELECT id, space_id, name, category, model, serial_number, purchase_date,
           purchase_price, current_value, currency, status, warranty_expires,
           warranty_type, insurance_covered, note_path, created_at, modified_at,
           created_by
    FROM items;

DROP TABLE items;
ALTER TABLE items_old RENAME TO items;

CREATE TABLE IF NOT EXISTS space_members (
    space_id TEXT REFERENCES spaces(id),
    user_id TEXT REFERENCES users(id),
    role TEXT CHECK(role IN ('owner', 'admin', 'member', 'viewer')),
    PRIMARY KEY (space_id, user_id)
);

PRAGMA foreign_keys = ON;

DELETE FROM schema_version WHERE version = 43;
