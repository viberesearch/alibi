-- Migration 043: Reconcile the two long-standing schema-drift cases so the
-- schema.sql snapshot (now generated from this chain) and the migration chain
-- agree with no allow-list.
--
--   1. space_members — declared only in the old hand-maintained schema.sql,
--      created by no migration and read/written by no code. Drop it so fresh
--      installs and migrated databases converge (no-op where it never existed).
--   2. items — the v1 baseline left it looser than schema.sql intended
--      (purchase_price/current_value DECIMAL not DECIMAL(10,2); name nullable;
--      no status CHECK / FK refs / defaults). Rebuild it to the canonical tight
--      form. SQLite cannot ALTER column types in place, so recreate-and-copy
--      (the same pattern as migration 008), with foreign keys disabled for the
--      swap. `name` becomes NOT NULL: every real row already carries a name.

PRAGMA foreign_keys = OFF;

DROP TABLE IF EXISTS space_members;

CREATE TABLE items_new (
    id TEXT PRIMARY KEY,
    space_id TEXT REFERENCES spaces(id),
    name TEXT NOT NULL,
    category TEXT,                -- electronics, appliances, vehicle, etc.
    model TEXT,
    serial_number TEXT,
    purchase_date DATE,
    purchase_price DECIMAL(10,2),
    current_value DECIMAL(10,2),
    currency TEXT DEFAULT 'EUR',
    status TEXT CHECK(status IN ('active', 'sold', 'disposed', 'returned', 'lost')),
    warranty_expires DATE,
    warranty_type TEXT,           -- manufacturer, extended
    insurance_covered BOOLEAN DEFAULT FALSE,
    note_path TEXT,               -- Link to Obsidian item note
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    modified_at DATETIME,
    created_by TEXT REFERENCES users(id)
);

INSERT INTO items_new
    SELECT id, space_id, name, category, model, serial_number, purchase_date,
           purchase_price, current_value, currency, status, warranty_expires,
           warranty_type, insurance_covered, note_path, created_at, modified_at,
           created_by
    FROM items;

DROP TABLE items;
ALTER TABLE items_new RENAME TO items;

PRAGMA foreign_keys = ON;

INSERT OR IGNORE INTO schema_version (version) VALUES (43);
