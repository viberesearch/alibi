-- Migration 008: Add payment_confirmation artifact type and linked duplicate action
-- Supports proof-of-transaction model: different document types provide different
-- proof dimensions for the same purchase.
--
-- SQLite cannot ALTER CHECK constraints, so affected tables must be recreated.

PRAGMA foreign_keys = OFF;

-- 1. Recreate artifacts table with payment_confirmation in type CHECK
CREATE TABLE artifacts_new (
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

INSERT INTO artifacts_new (
    id, space_id, type, record_type, file_path, file_hash, perceptual_hash,
    vendor, vendor_id, vendor_address, vendor_phone, vendor_website,
    vendor_registration, document_id, document_date, amount, currency,
    raw_text, extracted_data, status, created_at, modified_at, created_by
)
SELECT
    id, space_id, type, record_type, file_path, file_hash, perceptual_hash,
    vendor, vendor_id, vendor_address, vendor_phone, vendor_website,
    vendor_registration, document_id, document_date, amount, currency,
    raw_text, extracted_data, status, created_at, modified_at, created_by
FROM artifacts;

DROP TABLE artifacts;
ALTER TABLE artifacts_new RENAME TO artifacts;

CREATE INDEX IF NOT EXISTS idx_artifacts_hash ON artifacts(file_hash);
CREATE INDEX IF NOT EXISTS idx_artifacts_vendor ON artifacts(vendor);
CREATE INDEX IF NOT EXISTS idx_artifacts_date ON artifacts(document_date);

-- 2. Ensure transaction_artifacts exists with updated CHECK constraint.
-- Drop old table if it exists and recreate with new constraint.
-- Data loss is acceptable here since this table is auto-populated by the pipeline.
DROP TABLE IF EXISTS transaction_artifacts;
CREATE TABLE transaction_artifacts (
    transaction_id TEXT REFERENCES transactions(id),
    artifact_id TEXT REFERENCES artifacts(id),
    link_type TEXT CHECK(link_type IN ('receipt', 'invoice', 'statement_line',
                                        'warranty', 'payment_confirmation', 'other')),
    match_confidence DECIMAL(3,2),
    matched_at DATETIME,
    matched_by TEXT,
    PRIMARY KEY (transaction_id, artifact_id)
);

-- 3. Ensure duplicate_log exists with updated CHECK constraint.
DROP TABLE IF EXISTS duplicate_log;
CREATE TABLE duplicate_log (
    id TEXT PRIMARY KEY,
    original_artifact_id TEXT REFERENCES artifacts(id),
    duplicate_file_path TEXT,
    duplicate_hash TEXT,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    action TEXT CHECK(action IN ('skipped', 'notified', 'merged', 'linked'))
);

PRAGMA foreign_keys = ON;

INSERT OR IGNORE INTO schema_version (version) VALUES (8);
