-- Migration 035 DOWN: Remove product_variant from fact_items

CREATE TABLE fact_items_backup AS SELECT
    id, fact_id, atom_id, name, name_normalized, comparable_name,
    quantity, unit, unit_price, total_price,
    brand, category, comparable_unit_price, comparable_unit,
    tax_rate, tax_type, barcode, unit_quantity,
    enrichment_source, enrichment_confidence, created_at
FROM fact_items;

DROP TABLE fact_items;

CREATE TABLE fact_items (
    id TEXT PRIMARY KEY,
    fact_id TEXT NOT NULL REFERENCES facts(id),
    atom_id TEXT NOT NULL REFERENCES atoms(id),
    name TEXT NOT NULL,
    name_normalized TEXT,
    comparable_name TEXT,
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
    barcode TEXT,
    unit_quantity DECIMAL(10,3),
    enrichment_source TEXT,
    enrichment_confidence REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO fact_items SELECT * FROM fact_items_backup;
DROP TABLE fact_items_backup;

CREATE INDEX IF NOT EXISTS idx_fact_items_fact ON fact_items(fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_items_name ON fact_items(name_normalized);
CREATE INDEX IF NOT EXISTS idx_fact_items_barcode ON fact_items(barcode);

DELETE FROM schema_version WHERE version = 35;
