-- Migration 016: Drop legacy 'registration' member type from identity_members.
-- The vendor_vat/vendor_tax_id fields replace the old vendor_registration field.
-- Any existing 'registration' members are converted to 'vat_number'.

INSERT OR IGNORE INTO schema_version (version) VALUES (16);

UPDATE identity_members
SET member_type = 'vat_number'
WHERE member_type = 'registration'
  AND value NOT IN (
      SELECT value FROM identity_members
      WHERE member_type = 'vat_number' AND identity_id = identity_members.identity_id
  );

DELETE FROM identity_members WHERE member_type = 'registration';

-- Recreate table without 'registration' in CHECK constraint.
-- SQLite requires table recreation to change CHECK constraints.
CREATE TABLE identity_members_new (
    id TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    member_type TEXT NOT NULL CHECK(member_type IN (
        'name',
        'normalized_name',
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

INSERT INTO identity_members_new SELECT * FROM identity_members;
DROP TABLE identity_members;
ALTER TABLE identity_members_new RENAME TO identity_members;

CREATE INDEX IF NOT EXISTS idx_identity_members_identity ON identity_members(identity_id);
CREATE INDEX IF NOT EXISTS idx_identity_members_value ON identity_members(value);
CREATE INDEX IF NOT EXISTS idx_identity_members_type_value ON identity_members(member_type, value);
