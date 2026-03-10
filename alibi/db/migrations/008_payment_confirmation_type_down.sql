-- Rollback migration 008: Remove payment_confirmation type and linked action

PRAGMA foreign_keys = OFF;

-- 1. Recreate artifacts without payment_confirmation
CREATE TABLE artifacts_new (
    id TEXT PRIMARY KEY,
    space_id TEXT REFERENCES spaces(id),
    type TEXT CHECK(type IN ('receipt', 'invoice', 'statement', 'warranty',
                              'policy', 'contract', 'other')),
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

INSERT INTO artifacts_new SELECT * FROM artifacts WHERE type != 'payment_confirmation';
DROP TABLE artifacts;
ALTER TABLE artifacts_new RENAME TO artifacts;

CREATE INDEX IF NOT EXISTS idx_artifacts_hash ON artifacts(file_hash);
CREATE INDEX IF NOT EXISTS idx_artifacts_vendor ON artifacts(vendor);
CREATE INDEX IF NOT EXISTS idx_artifacts_date ON artifacts(document_date);

-- 2. Recreate transaction_artifacts without payment_confirmation
CREATE TABLE transaction_artifacts_new (
    transaction_id TEXT REFERENCES transactions(id),
    artifact_id TEXT REFERENCES artifacts(id),
    link_type TEXT CHECK(link_type IN ('receipt', 'invoice', 'statement_line',
                                        'warranty', 'other')),
    match_confidence DECIMAL(3,2),
    matched_at DATETIME,
    matched_by TEXT,
    PRIMARY KEY (transaction_id, artifact_id)
);

INSERT INTO transaction_artifacts_new SELECT * FROM transaction_artifacts
    WHERE link_type != 'payment_confirmation';
DROP TABLE transaction_artifacts;
ALTER TABLE transaction_artifacts_new RENAME TO transaction_artifacts;

-- 3. Recreate duplicate_log without linked
CREATE TABLE duplicate_log_new (
    id TEXT PRIMARY KEY,
    original_artifact_id TEXT REFERENCES artifacts(id),
    duplicate_file_path TEXT,
    duplicate_hash TEXT,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    action TEXT CHECK(action IN ('skipped', 'notified', 'merged'))
);

INSERT INTO duplicate_log_new SELECT * FROM duplicate_log WHERE action != 'linked';
DROP TABLE duplicate_log;
ALTER TABLE duplicate_log_new RENAME TO duplicate_log;

PRAGMA foreign_keys = ON;

DELETE FROM schema_version WHERE version = 8;
