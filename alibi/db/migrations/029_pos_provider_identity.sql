-- Migration 029: Add pos_provider entity type to identities table.
--
-- SQLite does not support ALTER TABLE ... DROP CONSTRAINT, so we
-- create a new table, copy data, drop old, and rename.

PRAGMA foreign_keys = OFF;

CREATE TABLE _identities_new (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('vendor', 'item', 'pos_provider')),
    canonical_name TEXT NOT NULL,
    metadata JSON,
    active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME
);

INSERT INTO _identities_new SELECT * FROM identities;
DROP TABLE identities;
ALTER TABLE _identities_new RENAME TO identities;

CREATE INDEX IF NOT EXISTS idx_identities_type ON identities(entity_type);
CREATE INDEX IF NOT EXISTS idx_identities_name ON identities(canonical_name);

PRAGMA foreign_keys = ON;

INSERT OR IGNORE INTO schema_version (version) VALUES (29);
