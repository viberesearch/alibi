-- Migration 012: Identity tables for manual entity grouping
-- Adds identities + identity_members tables for user-defined
-- vendor/item grouping (algorithmic collapse + manual merge).
-- Also adds barcode column to fact_items.

-- Identities: user-defined canonical entities
CREATE TABLE IF NOT EXISTS identities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('vendor', 'item')),
    canonical_name TEXT NOT NULL,
    metadata JSON,
    active BOOLEAN DEFAULT TRUE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_identities_type ON identities(entity_type);
CREATE INDEX IF NOT EXISTS idx_identities_name ON identities(canonical_name);

-- Identity members: recognized values that belong to an identity
CREATE TABLE IF NOT EXISTS identity_members (
    id TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    member_type TEXT NOT NULL CHECK(member_type IN (
        'name',
        'normalized_name',
        'registration',
        'vendor_key',
        'barcode'
    )),
    value TEXT NOT NULL,
    source TEXT DEFAULT 'user',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(identity_id, member_type, value)
);

CREATE INDEX IF NOT EXISTS idx_identity_members_identity ON identity_members(identity_id);
CREATE INDEX IF NOT EXISTS idx_identity_members_value ON identity_members(value);
CREATE INDEX IF NOT EXISTS idx_identity_members_type_value ON identity_members(member_type, value);

-- Add barcode column to fact_items
ALTER TABLE fact_items ADD COLUMN barcode TEXT;

-- Record migration
INSERT OR IGNORE INTO schema_version (version) VALUES (12);
