-- Migration 011: Drop v1 tables
-- All application code now uses v2 tables (documents, facts, fact_items).
-- v1 tables are no longer read or written by any code path.

-- Disable foreign keys during migration
PRAGMA foreign_keys = OFF;

-- Drop v1 junction tables first (depend on v1 primary tables)
DROP TABLE IF EXISTS transaction_artifacts;
DROP TABLE IF EXISTS transaction_tags;
DROP TABLE IF EXISTS item_artifacts;
DROP TABLE IF EXISTS item_transactions;

-- Drop v1 primary tables
DROP TABLE IF EXISTS line_items;
DROP TABLE IF EXISTS artifacts;
DROP TABLE IF EXISTS transactions;

-- Drop v1 support tables
DROP TABLE IF EXISTS duplicate_log;
DROP TABLE IF EXISTS provenance;

-- Rename item junction tables to reference v2 entities
-- item_artifacts → item_documents (items linked to documents)
CREATE TABLE IF NOT EXISTS item_documents (
    item_id TEXT REFERENCES items(id),
    document_id TEXT REFERENCES documents(id),
    link_type TEXT CHECK(link_type IN ('receipt', 'invoice', 'warranty',
                                        'insurance', 'manual', 'photo')),
    PRIMARY KEY (item_id, document_id)
);

-- item_transactions → item_facts (items linked to facts)
CREATE TABLE IF NOT EXISTS item_facts (
    item_id TEXT REFERENCES items(id),
    fact_id TEXT REFERENCES facts(id),
    link_type TEXT CHECK(link_type IN ('purchase', 'maintenance', 'upgrade',
                                        'sale', 'insurance_claim')),
    PRIMARY KEY (item_id, fact_id)
);

-- Migrate line_item_allocations FK from line_items to fact_items
-- SQLite doesn't support ALTER TABLE ... DROP/ADD CONSTRAINT, so recreate
CREATE TABLE line_item_allocations_new (
    id TEXT PRIMARY KEY,
    line_item_id TEXT REFERENCES fact_items(id),
    consumer_id TEXT REFERENCES consumers(id),
    share DECIMAL(5,4) DEFAULT 1.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO line_item_allocations_new
    SELECT id, line_item_id, consumer_id, share, created_at
    FROM line_item_allocations;

DROP TABLE line_item_allocations;
ALTER TABLE line_item_allocations_new RENAME TO line_item_allocations;

CREATE INDEX IF NOT EXISTS idx_allocations_line_item ON line_item_allocations(line_item_id);
CREATE INDEX IF NOT EXISTS idx_allocations_consumer ON line_item_allocations(consumer_id);

-- Re-enable foreign keys
PRAGMA foreign_keys = ON;

-- Record migration
INSERT OR IGNORE INTO schema_version (version) VALUES (11);
