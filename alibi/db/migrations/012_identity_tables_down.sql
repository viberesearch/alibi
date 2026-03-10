-- Migration 012 down: Remove identity tables and barcode column

DROP TABLE IF EXISTS identity_members;
DROP TABLE IF EXISTS identities;

-- SQLite doesn't support DROP COLUMN before 3.35.0.
-- Recreate fact_items without barcode column.
CREATE TABLE fact_items_backup AS SELECT
    id, fact_id, atom_id, name, name_normalized,
    quantity, unit, unit_price, total_price,
    brand, category, comparable_unit_price, comparable_unit,
    tax_rate, tax_type, created_at
FROM fact_items;

DROP TABLE fact_items;

CREATE TABLE fact_items (
    id TEXT PRIMARY KEY,
    fact_id TEXT NOT NULL REFERENCES facts(id),
    atom_id TEXT NOT NULL REFERENCES atoms(id),
    name TEXT NOT NULL,
    name_normalized TEXT,
    quantity DECIMAL(10,3) DEFAULT 1,
    unit TEXT DEFAULT 'pcs',
    unit_price DECIMAL(10,2),
    total_price DECIMAL(10,2),
    brand TEXT,
    category TEXT,
    comparable_unit_price DECIMAL(10,4),
    comparable_unit TEXT,
    tax_rate DECIMAL(5,2),
    tax_type TEXT DEFAULT 'none',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO fact_items SELECT * FROM fact_items_backup;
DROP TABLE fact_items_backup;

CREATE INDEX IF NOT EXISTS idx_fact_items_fact ON fact_items(fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_items_name ON fact_items(name_normalized);

DELETE FROM schema_version WHERE version = 12;
