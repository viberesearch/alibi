-- Down migration 016: Re-add 'registration' member type.
-- Cannot undo data conversion (vat_number ← registration).

DELETE FROM schema_version WHERE version = 16;

CREATE TABLE identity_members_old (
    id TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    member_type TEXT NOT NULL CHECK(member_type IN (
        'name',
        'normalized_name',
        'registration',
        'vat_number',
        'tax_id',
        'vendor_key',
        'barcode'
    )),
    value TEXT NOT NULL,
    source TEXT DEFAULT 'user',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(identity_id, member_type, value)
);

INSERT INTO identity_members_old SELECT * FROM identity_members;
DROP TABLE identity_members;
ALTER TABLE identity_members_old RENAME TO identity_members;

CREATE INDEX IF NOT EXISTS idx_identity_members_identity ON identity_members(identity_id);
CREATE INDEX IF NOT EXISTS idx_identity_members_value ON identity_members(value);
CREATE INDEX IF NOT EXISTS idx_identity_members_type_value ON identity_members(member_type, value);
