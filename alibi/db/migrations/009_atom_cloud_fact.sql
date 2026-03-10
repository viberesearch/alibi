-- Migration 009: Atom-Cloud-Fact tables (v2 schema)
-- Adds observation-centric data model for cross-document clustering.

-- Up migration
-- ============

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    perceptual_hash TEXT,
    raw_extraction JSON,
    ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_phash ON documents(perceptual_hash);

CREATE TABLE IF NOT EXISTS atoms (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    atom_type TEXT NOT NULL CHECK(atom_type IN (
        'item', 'payment', 'vendor', 'datetime', 'amount', 'tax'
    )),
    data JSON NOT NULL,
    confidence REAL DEFAULT 1.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_atoms_document ON atoms(document_id);
CREATE INDEX IF NOT EXISTS idx_atoms_type ON atoms(atom_type);

CREATE TABLE IF NOT EXISTS bundles (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    bundle_type TEXT NOT NULL CHECK(bundle_type IN (
        'basket', 'payment_record', 'invoice', 'statement_line'
    )),
    cloud_id TEXT REFERENCES clouds(id),  -- authoritative cloud assignment (NULL = unassigned)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_bundles_document ON bundles(document_id);
CREATE INDEX IF NOT EXISTS idx_bundles_cloud ON bundles(cloud_id);

CREATE TABLE IF NOT EXISTS bundle_atoms (
    bundle_id TEXT NOT NULL REFERENCES bundles(id),
    atom_id TEXT NOT NULL REFERENCES atoms(id),
    role TEXT NOT NULL CHECK(role IN (
        'basket_item', 'total', 'subtotal', 'vendor_info',
        'payment_info', 'event_time', 'tax_detail'
    )),
    PRIMARY KEY (bundle_id, atom_id)
);

CREATE TABLE IF NOT EXISTS clouds (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'forming' CHECK(status IN (
        'forming', 'collapsed', 'disputed'
    )),
    confidence REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cloud_bundles (
    cloud_id TEXT NOT NULL REFERENCES clouds(id),
    bundle_id TEXT NOT NULL REFERENCES bundles(id),
    match_type TEXT NOT NULL CHECK(match_type IN (
        'exact_amount', 'sum_of_parts', 'vendor+date', 'item_overlap', 'manual'
    )),
    match_confidence REAL DEFAULT 0.0,
    PRIMARY KEY (cloud_id, bundle_id)
);

CREATE TABLE IF NOT EXISTS facts (
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

CREATE INDEX IF NOT EXISTS idx_facts_vendor ON facts(vendor);
CREATE INDEX IF NOT EXISTS idx_facts_date ON facts(event_date);
CREATE INDEX IF NOT EXISTS idx_facts_cloud ON facts(cloud_id);

CREATE TABLE IF NOT EXISTS fact_items (
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

CREATE INDEX IF NOT EXISTS idx_fact_items_fact ON fact_items(fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_items_name ON fact_items(name_normalized);

INSERT OR IGNORE INTO schema_version (version) VALUES (9);
