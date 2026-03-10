-- Migration 010 down: Remove vendor_key from facts
-- SQLite does not support DROP COLUMN before 3.35.0;
-- this migration recreates the table without vendor_key.

DROP INDEX IF EXISTS idx_facts_vendor_key;

CREATE TABLE facts_backup AS SELECT
    id, cloud_id, fact_type, vendor, total_amount, currency,
    event_date, payments, status, created_at
FROM facts;

DROP TABLE facts;

CREATE TABLE facts (
    id TEXT PRIMARY KEY,
    cloud_id TEXT NOT NULL REFERENCES clouds(id),
    fact_type TEXT NOT NULL CHECK(fact_type IN (
        'purchase', 'refund', 'subscription_payment'
    )),
    vendor TEXT,
    total_amount DECIMAL(10,2),
    currency TEXT DEFAULT 'EUR',
    event_date DATE,
    payments JSON,
    status TEXT NOT NULL DEFAULT 'confirmed' CHECK(status IN (
        'confirmed', 'partial', 'needs_review'
    )),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO facts SELECT * FROM facts_backup;
DROP TABLE facts_backup;

CREATE INDEX IF NOT EXISTS idx_facts_vendor ON facts(vendor);
CREATE INDEX IF NOT EXISTS idx_facts_date ON facts(event_date);
CREATE INDEX IF NOT EXISTS idx_facts_cloud ON facts(cloud_id);

DELETE FROM schema_version WHERE version = 10;
