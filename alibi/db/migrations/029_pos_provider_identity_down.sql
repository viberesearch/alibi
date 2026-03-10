-- Migration 029 down: Remove pos_provider from entity_type CHECK.
-- Delete any pos_provider identities first, then recreate table.

PRAGMA foreign_keys = OFF;

DELETE FROM identity_members WHERE identity_id IN (
    SELECT id FROM identities WHERE entity_type = 'pos_provider'
);
DELETE FROM identities WHERE entity_type = 'pos_provider';

CREATE TABLE _identities_new (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('vendor', 'item')),
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

DELETE FROM schema_version WHERE version = 29;
