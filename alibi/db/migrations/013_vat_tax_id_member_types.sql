-- Migration 013: Add vat_number and tax_id as identity_members member types.
-- SQLite cannot ALTER CHECK constraints, so recreate the table.

CREATE TABLE identity_members_new (
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
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    UNIQUE(identity_id, member_type, value)
);

INSERT INTO identity_members_new
    SELECT * FROM identity_members;

DROP TABLE identity_members;

ALTER TABLE identity_members_new RENAME TO identity_members;

INSERT OR IGNORE INTO schema_version (version) VALUES (13);
