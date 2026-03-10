-- Migration 011 down: Restore v1 tables
-- This only recreates empty tables — data cannot be restored.

PRAGMA foreign_keys = OFF;

-- Drop v2 junction tables
DROP TABLE IF EXISTS item_documents;
DROP TABLE IF EXISTS item_facts;

-- Recreate v1 tables
CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    space_id TEXT REFERENCES spaces(id),
    type TEXT CHECK(type IN ('receipt', 'invoice', 'statement', 'warranty',
                              'policy', 'contract', 'payment_confirmation', 'other')),
    record_type TEXT,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    perceptual_hash TEXT,
    vendor TEXT,
    vendor_id TEXT,
    vendor_address TEXT,
    vendor_phone TEXT,
    vendor_website TEXT,
    vendor_registration TEXT,
    document_id TEXT,
    document_date DATE,
    amount DECIMAL(10,2),
    currency TEXT DEFAULT 'EUR',
    raw_text TEXT,
    extracted_data JSON,
    status TEXT CHECK(status IN ('pending', 'processed', 'verified', 'error')),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    modified_at DATETIME,
    created_by TEXT REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS transactions (
    id TEXT PRIMARY KEY,
    space_id TEXT REFERENCES spaces(id),
    type TEXT CHECK(type IN ('expense', 'income', 'transfer')),
    vendor TEXT,
    description TEXT,
    amount DECIMAL(10,2) NOT NULL,
    currency TEXT DEFAULT 'EUR',
    transaction_date DATE NOT NULL,
    transaction_time TEXT,
    payment_method TEXT,
    card_last4 TEXT,
    account_reference TEXT,
    status TEXT CHECK(status IN ('pending', 'matched', 'verified', 'disputed')),
    note_path TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    modified_at DATETIME,
    created_by TEXT REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS line_items (
    id TEXT PRIMARY KEY,
    artifact_id TEXT REFERENCES artifacts(id),
    transaction_id TEXT REFERENCES transactions(id),
    name TEXT NOT NULL,
    name_normalized TEXT,
    original_language TEXT,
    quantity DECIMAL(10,3) DEFAULT 1,
    unit TEXT DEFAULT 'pcs',
    unit_raw TEXT,
    unit_quantity DECIMAL(10,3),
    comparable_unit_price DECIMAL(10,4),
    comparable_unit TEXT,
    unit_price DECIMAL(10,2),
    total_price DECIMAL(10,2),
    currency TEXT DEFAULT 'EUR',
    tax_type TEXT DEFAULT 'none',
    tax_rate DECIMAL(5,2),
    tax_amount DECIMAL(10,2),
    discount_amount DECIMAL(10,2),
    discount_percentage DECIMAL(5,2),
    category TEXT,
    subcategory TEXT,
    item_id TEXT REFERENCES items(id),
    consumer_id TEXT,
    field_type TEXT DEFAULT 'count',
    barcode TEXT,
    brand TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transaction_artifacts (
    transaction_id TEXT REFERENCES transactions(id),
    artifact_id TEXT REFERENCES artifacts(id),
    link_type TEXT,
    match_confidence DECIMAL(3,2),
    matched_at DATETIME,
    matched_by TEXT,
    PRIMARY KEY (transaction_id, artifact_id)
);

CREATE TABLE IF NOT EXISTS transaction_tags (
    transaction_id TEXT REFERENCES transactions(id),
    tag_id TEXT REFERENCES tags(id),
    PRIMARY KEY (transaction_id, tag_id)
);

CREATE TABLE IF NOT EXISTS item_artifacts (
    item_id TEXT REFERENCES items(id),
    artifact_id TEXT REFERENCES artifacts(id),
    link_type TEXT,
    PRIMARY KEY (item_id, artifact_id)
);

CREATE TABLE IF NOT EXISTS item_transactions (
    item_id TEXT REFERENCES items(id),
    transaction_id TEXT REFERENCES transactions(id),
    link_type TEXT,
    PRIMARY KEY (item_id, transaction_id)
);

CREATE TABLE IF NOT EXISTS duplicate_log (
    id TEXT PRIMARY KEY,
    original_artifact_id TEXT REFERENCES artifacts(id),
    duplicate_file_path TEXT,
    duplicate_hash TEXT,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    action TEXT CHECK(action IN ('skipped', 'notified', 'merged', 'linked'))
);

CREATE TABLE IF NOT EXISTS provenance (
    id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    record_type TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_id TEXT,
    confidence DECIMAL(3,2),
    processor TEXT,
    parent_provenance_id TEXT REFERENCES provenance(id),
    metadata JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Recreate line_item_allocations with v1 FK
CREATE TABLE line_item_allocations_old (
    id TEXT PRIMARY KEY,
    line_item_id TEXT REFERENCES line_items(id),
    consumer_id TEXT REFERENCES consumers(id),
    share DECIMAL(5,4) DEFAULT 1.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO line_item_allocations_old
    SELECT id, line_item_id, consumer_id, share, created_at
    FROM line_item_allocations;

DROP TABLE line_item_allocations;
ALTER TABLE line_item_allocations_old RENAME TO line_item_allocations;

CREATE INDEX IF NOT EXISTS idx_allocations_line_item ON line_item_allocations(line_item_id);
CREATE INDEX IF NOT EXISTS idx_allocations_consumer ON line_item_allocations(consumer_id);

PRAGMA foreign_keys = ON;

DELETE FROM schema_version WHERE version = 11;
