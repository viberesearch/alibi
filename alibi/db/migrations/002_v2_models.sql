-- Migration 002: V2 Models
-- Adds atomized line item fields, provenance tracking, and future table stubs.

-- Up migration
-- ============

-- Enhance line_items with atomized fields
ALTER TABLE line_items ADD COLUMN name_normalized TEXT;
ALTER TABLE line_items ADD COLUMN original_language TEXT;
ALTER TABLE line_items ADD COLUMN unit TEXT DEFAULT 'pcs';
ALTER TABLE line_items ADD COLUMN unit_raw TEXT;
ALTER TABLE line_items ADD COLUMN currency TEXT DEFAULT 'EUR';
ALTER TABLE line_items ADD COLUMN tax_type TEXT DEFAULT 'none';
ALTER TABLE line_items ADD COLUMN tax_rate DECIMAL(5,2);
ALTER TABLE line_items ADD COLUMN tax_amount DECIMAL(10,2);
ALTER TABLE line_items ADD COLUMN discount_amount DECIMAL(10,2);
ALTER TABLE line_items ADD COLUMN discount_percentage DECIMAL(5,2);
ALTER TABLE line_items ADD COLUMN subcategory TEXT;
ALTER TABLE line_items ADD COLUMN consumer_id TEXT;
ALTER TABLE line_items ADD COLUMN field_type TEXT DEFAULT 'count';
ALTER TABLE line_items ADD COLUMN barcode TEXT;
ALTER TABLE line_items ADD COLUMN brand TEXT;

-- Add record_type to artifacts for unified type system
ALTER TABLE artifacts ADD COLUMN record_type TEXT;

-- Provenance tracking table
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

CREATE INDEX IF NOT EXISTS idx_provenance_record
    ON provenance(record_id, record_type);
CREATE INDEX IF NOT EXISTS idx_provenance_source
    ON provenance(source_type, source_id);

-- Budget scenarios table (Phase 3 stub)
CREATE TABLE IF NOT EXISTS budgets (
    id TEXT PRIMARY KEY,
    space_id TEXT REFERENCES spaces(id),
    name TEXT NOT NULL,
    description TEXT,
    data_type TEXT CHECK(data_type IN ('actual', 'projected', 'target')),
    parent_id TEXT REFERENCES budgets(id),
    period_start DATE,
    period_end DATE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    modified_at DATETIME
);

-- Masking snapshots table (Phase 2C stub)
CREATE TABLE IF NOT EXISTS masking_snapshots (
    id TEXT PRIMARY KEY,
    tier TEXT NOT NULL,
    masking_map JSON NOT NULL,
    record_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Update schema version
INSERT OR IGNORE INTO schema_version (version) VALUES (2);
