-- Migration 002 Down: Revert V2 Models
-- SQLite doesn't support DROP COLUMN before 3.35.0, so we recreate tables.

-- Recreate line_items without v2 columns
CREATE TABLE line_items_backup AS SELECT
    id, artifact_id, transaction_id, name, quantity, unit_price,
    total_price, category, item_id, created_at
FROM line_items;

DROP TABLE line_items;

CREATE TABLE line_items (
    id TEXT PRIMARY KEY,
    artifact_id TEXT REFERENCES artifacts(id),
    transaction_id TEXT REFERENCES transactions(id),
    name TEXT NOT NULL,
    quantity DECIMAL(10,3) DEFAULT 1,
    unit_price DECIMAL(10,2),
    total_price DECIMAL(10,2),
    category TEXT,
    item_id TEXT REFERENCES items(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO line_items SELECT * FROM line_items_backup;
DROP TABLE line_items_backup;

-- Remove record_type from artifacts (recreate without it)
CREATE TABLE artifacts_backup AS SELECT
    id, space_id, type, file_path, file_hash, perceptual_hash,
    vendor, vendor_id, document_id, document_date, amount, currency,
    raw_text, extracted_data, status, created_at, modified_at, created_by
FROM artifacts;

DROP TABLE artifacts;

CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    space_id TEXT REFERENCES spaces(id),
    type TEXT CHECK(type IN ('receipt', 'invoice', 'statement', 'warranty',
                              'policy', 'contract', 'other')),
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    perceptual_hash TEXT,
    vendor TEXT,
    vendor_id TEXT,
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

INSERT INTO artifacts SELECT * FROM artifacts_backup;
DROP TABLE artifacts_backup;

CREATE INDEX IF NOT EXISTS idx_artifacts_hash ON artifacts(file_hash);
CREATE INDEX IF NOT EXISTS idx_artifacts_vendor ON artifacts(vendor);
CREATE INDEX IF NOT EXISTS idx_artifacts_date ON artifacts(document_date);

-- Drop new tables
DROP TABLE IF EXISTS masking_snapshots;
DROP TABLE IF EXISTS budgets;
DROP TABLE IF EXISTS provenance;

-- Revert schema version
DELETE FROM schema_version WHERE version = 2;
