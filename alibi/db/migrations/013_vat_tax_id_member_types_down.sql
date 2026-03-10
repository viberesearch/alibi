-- Rollback migration 013: Remove vat_number and tax_id member types.

CREATE TABLE identity_members_old (
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
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    UNIQUE(identity_id, member_type, value)
);

INSERT INTO identity_members_old
    SELECT * FROM identity_members
    WHERE member_type NOT IN ('vat_number', 'tax_id');

DROP TABLE identity_members;

ALTER TABLE identity_members_old RENAME TO identity_members;

DELETE FROM schema_version WHERE version = 13;
