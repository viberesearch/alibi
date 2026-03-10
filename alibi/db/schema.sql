-- Alibi: Life Tracker Database Schema
-- Version: 2.1.0
-- Created: 2025-12-27
-- Updated: 2026-02-19 (identity tables, barcode on fact_items)

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Users and access control
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,  -- uuid
    name TEXT DEFAULT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- User contacts: 1:N relationship for telegram accounts, emails, etc.
CREATE TABLE IF NOT EXISTS user_contacts (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    contact_type TEXT NOT NULL CHECK(contact_type IN ('telegram', 'email')),
    value TEXT NOT NULL,
    label TEXT DEFAULT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(contact_type, value)
);

CREATE INDEX IF NOT EXISTS idx_user_contacts_user ON user_contacts(user_id);
CREATE INDEX IF NOT EXISTS idx_user_contacts_lookup ON user_contacts(contact_type, value);

-- API keys: PBKDF2-hashed 6-word mnemonic passphrases
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    label TEXT DEFAULT 'default',
    salt BLOB DEFAULT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_used_at DATETIME DEFAULT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);

CREATE TABLE IF NOT EXISTS spaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT CHECK(type IN ('private', 'shared')),
    owner_id TEXT REFERENCES users(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS space_members (
    space_id TEXT REFERENCES spaces(id),
    user_id TEXT REFERENCES users(id),
    role TEXT CHECK(role IN ('owner', 'admin', 'member', 'viewer')),
    PRIMARY KEY (space_id, user_id)
);

-- Items (assets)
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY,
    space_id TEXT REFERENCES spaces(id),
    name TEXT NOT NULL,
    category TEXT,                -- electronics, appliances, vehicle, etc.
    model TEXT,
    serial_number TEXT,
    purchase_date DATE,
    purchase_price DECIMAL(10,2),
    current_value DECIMAL(10,2),
    currency TEXT DEFAULT 'EUR',
    status TEXT CHECK(status IN ('active', 'sold', 'disposed', 'returned', 'lost')),
    warranty_expires DATE,
    warranty_type TEXT,           -- manufacturer, extended
    insurance_covered BOOLEAN DEFAULT FALSE,
    note_path TEXT,               -- Link to Obsidian item note
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    modified_at DATETIME,
    created_by TEXT REFERENCES users(id)
);

CREATE INDEX IF NOT EXISTS idx_items_warranty ON items(warranty_expires);
CREATE INDEX IF NOT EXISTS idx_items_category ON items(category);

-- Budget scenarios
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

-- Budget entries (per-category amounts within a scenario)
CREATE TABLE IF NOT EXISTS budget_entries (
    id TEXT PRIMARY KEY,
    scenario_id TEXT REFERENCES budgets(id),
    category TEXT NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    currency TEXT DEFAULT 'EUR',
    period TEXT NOT NULL,  -- YYYY-MM format
    note TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_budget_entries_scenario ON budget_entries(scenario_id);
CREATE INDEX IF NOT EXISTS idx_budget_entries_period ON budget_entries(period);

-- Masking snapshots
CREATE TABLE IF NOT EXISTS masking_snapshots (
    id TEXT PRIMARY KEY,
    tier TEXT NOT NULL,
    masking_map JSON NOT NULL,
    record_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Product cache for external data lookups (Open Food Facts, etc.)
CREATE TABLE IF NOT EXISTS product_cache (
    barcode    TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'openfoodfacts',
    fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ===========================================================================
-- Atom-Cloud-Fact tables (v2 schema)
-- ===========================================================================

-- Documents: source files
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,          -- SHA-256
    perceptual_hash TEXT,             -- dHash for images
    raw_extraction JSON,              -- Full LLM output preserved
    source TEXT DEFAULT NULL,         -- telegram, api, cli, watcher, mcp
    user_id TEXT REFERENCES users(id) DEFAULT NULL,
    yaml_hash TEXT DEFAULT NULL,      -- SHA-256 of .alibi.yaml for correction detection
    yaml_path TEXT DEFAULT NULL,      -- Path to .alibi.yaml in yaml_store (NULL = sidecar)
    ingested_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_phash ON documents(perceptual_hash);

-- Atoms: individual extracted observations
CREATE TABLE IF NOT EXISTS atoms (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    atom_type TEXT NOT NULL CHECK(atom_type IN (
        'item', 'payment', 'vendor', 'datetime', 'amount', 'tax'
    )),
    data JSON NOT NULL,               -- Type-specific normalized payload
    confidence REAL DEFAULT 1.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_atoms_document ON atoms(document_id);
CREATE INDEX IF NOT EXISTS idx_atoms_type ON atoms(atom_type);

-- Bundles: structural atom groups from one document
CREATE TABLE IF NOT EXISTS bundles (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    bundle_type TEXT NOT NULL CHECK(bundle_type IN (
        'basket', 'payment_record', 'invoice', 'statement_line'
    )),
    cloud_id TEXT REFERENCES clouds(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_bundles_document ON bundles(document_id);
CREATE INDEX IF NOT EXISTS idx_bundles_cloud ON bundles(cloud_id);

-- Bundle-Atom links with role
CREATE TABLE IF NOT EXISTS bundle_atoms (
    bundle_id TEXT NOT NULL REFERENCES bundles(id),
    atom_id TEXT NOT NULL REFERENCES atoms(id),
    role TEXT NOT NULL CHECK(role IN (
        'basket_item', 'total', 'subtotal', 'vendor_info',
        'payment_info', 'event_time', 'tax_detail'
    )),
    PRIMARY KEY (bundle_id, atom_id)
);

-- Clouds: probabilistic clusters across documents
CREATE TABLE IF NOT EXISTS clouds (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'forming' CHECK(status IN (
        'forming', 'collapsed', 'disputed'
    )),
    confidence REAL DEFAULT 0.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Cloud-Bundle links with match metadata
CREATE TABLE IF NOT EXISTS cloud_bundles (
    cloud_id TEXT NOT NULL REFERENCES clouds(id),
    bundle_id TEXT NOT NULL REFERENCES bundles(id),
    match_type TEXT NOT NULL CHECK(match_type IN (
        'exact_amount', 'near_amount', 'sum_of_parts', 'vendor+date', 'item_overlap', 'manual'
    )),
    match_confidence REAL DEFAULT 0.0,
    PRIMARY KEY (cloud_id, bundle_id)
);

-- Facts: collapsed clouds = confirmed reality
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    cloud_id TEXT NOT NULL REFERENCES clouds(id),
    fact_type TEXT NOT NULL CHECK(fact_type IN (
        'purchase', 'refund', 'subscription_payment'
    )),
    vendor TEXT,
    vendor_key TEXT,                   -- Stable vendor ID (registration or noid_<hash>)
    total_amount DECIMAL(10,2),
    currency TEXT DEFAULT 'EUR',
    event_date DATE,
    payments JSON,                    -- [{method, card_last4, amount, date}, ...]
    status TEXT NOT NULL DEFAULT 'confirmed' CHECK(status IN (
        'confirmed', 'partial', 'needs_review'
    )),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_facts_vendor ON facts(vendor);
CREATE INDEX IF NOT EXISTS idx_facts_vendor_key ON facts(vendor_key);
CREATE INDEX IF NOT EXISTS idx_facts_date ON facts(event_date);
CREATE INDEX IF NOT EXISTS idx_facts_cloud ON facts(cloud_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_cloud_unique ON facts(cloud_id);

-- Fact items: denormalized from item atoms for fast queries
CREATE TABLE IF NOT EXISTS fact_items (
    id TEXT PRIMARY KEY,
    fact_id TEXT NOT NULL REFERENCES facts(id),
    atom_id TEXT NOT NULL REFERENCES atoms(id),
    name TEXT NOT NULL,
    name_normalized TEXT,
    comparable_name TEXT,
    quantity DECIMAL(10,3) DEFAULT 1,
    unit TEXT DEFAULT 'pcs',          -- UnitType enum
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
    product_variant TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_fact_items_fact ON fact_items(fact_id);
CREATE INDEX IF NOT EXISTS idx_fact_items_name ON fact_items(name_normalized);
CREATE INDEX IF NOT EXISTS idx_fact_items_barcode ON fact_items(barcode);

-- Item to Document links
CREATE TABLE IF NOT EXISTS item_documents (
    item_id TEXT REFERENCES items(id),
    document_id TEXT REFERENCES documents(id),
    link_type TEXT CHECK(link_type IN ('receipt', 'invoice', 'warranty',
                                        'insurance', 'manual', 'photo')),
    PRIMARY KEY (item_id, document_id)
);

-- Item to Fact links
CREATE TABLE IF NOT EXISTS item_facts (
    item_id TEXT REFERENCES items(id),
    fact_id TEXT REFERENCES facts(id),
    link_type TEXT CHECK(link_type IN ('purchase', 'maintenance', 'upgrade',
                                        'sale', 'insurance_claim')),
    PRIMARY KEY (item_id, fact_id)
);

-- Identities: user-defined canonical entities (manual merge layer)
CREATE TABLE IF NOT EXISTS identities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL CHECK(entity_type IN ('vendor', 'item', 'pos_provider')),
    canonical_name TEXT NOT NULL,
    metadata JSON,                    -- Type-specific: barcode, legal_name, category, etc.
    active BOOLEAN DEFAULT TRUE,      -- Toggle identity view on/off
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME
);

CREATE INDEX IF NOT EXISTS idx_identities_type ON identities(entity_type);
CREATE INDEX IF NOT EXISTS idx_identities_name ON identities(canonical_name);

-- Identity members: recognized values that belong to an identity
CREATE TABLE IF NOT EXISTS identity_members (
    id TEXT PRIMARY KEY,
    identity_id TEXT NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    member_type TEXT NOT NULL CHECK(member_type IN (
        'name',
        'normalized_name',
        'vat_number',
        'tax_id',
        'vendor_key',
        'barcode'
    )),
    value TEXT NOT NULL,
    source TEXT DEFAULT 'user',       -- 'user' | 'auto' (system-learned)
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(identity_id, member_type, value)
);

CREATE INDEX IF NOT EXISTS idx_identity_members_identity ON identity_members(identity_id);
CREATE INDEX IF NOT EXISTS idx_identity_members_value ON identity_members(value);
CREATE INDEX IF NOT EXISTS idx_identity_members_type_value ON identity_members(member_type, value);

-- Annotations: open-ended key-value metadata on entities
CREATE TABLE IF NOT EXISTS annotations (
    id TEXT PRIMARY KEY,
    annotation_type TEXT NOT NULL,
    target_type TEXT NOT NULL CHECK(target_type IN ('fact', 'fact_item', 'vendor', 'identity')),
    target_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    metadata JSON,
    source TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_annotations_target ON annotations(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_annotations_type ON annotations(annotation_type);
CREATE INDEX IF NOT EXISTS idx_annotations_key_value ON annotations(key, value);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_version (version) VALUES (1);
INSERT OR IGNORE INTO schema_version (version) VALUES (2);
INSERT OR IGNORE INTO schema_version (version) VALUES (3);
INSERT OR IGNORE INTO schema_version (version) VALUES (4);
INSERT OR IGNORE INTO schema_version (version) VALUES (5);
INSERT OR IGNORE INTO schema_version (version) VALUES (6);
INSERT OR IGNORE INTO schema_version (version) VALUES (7);
INSERT OR IGNORE INTO schema_version (version) VALUES (8);
INSERT OR IGNORE INTO schema_version (version) VALUES (9);
INSERT OR IGNORE INTO schema_version (version) VALUES (10);
INSERT OR IGNORE INTO schema_version (version) VALUES (11);
INSERT OR IGNORE INTO schema_version (version) VALUES (12);
INSERT OR IGNORE INTO schema_version (version) VALUES (13);
INSERT OR IGNORE INTO schema_version (version) VALUES (14);
INSERT OR IGNORE INTO schema_version (version) VALUES (15);
INSERT OR IGNORE INTO schema_version (version) VALUES (16);
INSERT OR IGNORE INTO schema_version (version) VALUES (17);
INSERT OR IGNORE INTO schema_version (version) VALUES (18);
INSERT OR IGNORE INTO schema_version (version) VALUES (19);
INSERT OR IGNORE INTO schema_version (version) VALUES (20);
INSERT OR IGNORE INTO schema_version (version) VALUES (21);
INSERT OR IGNORE INTO schema_version (version) VALUES (22);
INSERT OR IGNORE INTO schema_version (version) VALUES (23);
INSERT OR IGNORE INTO schema_version (version) VALUES (24);
INSERT OR IGNORE INTO schema_version (version) VALUES (25);
INSERT OR IGNORE INTO schema_version (version) VALUES (26);
INSERT OR IGNORE INTO schema_version (version) VALUES (27);
INSERT OR IGNORE INTO schema_version (version) VALUES (28);
INSERT OR IGNORE INTO schema_version (version) VALUES (29);
INSERT OR IGNORE INTO schema_version (version) VALUES (30);
INSERT OR IGNORE INTO schema_version (version) VALUES (31);
INSERT OR IGNORE INTO schema_version (version) VALUES (32);
INSERT OR IGNORE INTO schema_version (version) VALUES (33);
INSERT OR IGNORE INTO schema_version (version) VALUES (34);
INSERT OR IGNORE INTO schema_version (version) VALUES (35);

-- Migration 031: correction_events table for adaptive learning
CREATE TABLE IF NOT EXISTS correction_events (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    field TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    source TEXT NOT NULL,
    user_id TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_correction_events_entity
    ON correction_events (entity_type, entity_id);

CREATE INDEX IF NOT EXISTS idx_correction_events_created
    ON correction_events (created_at);

CREATE INDEX IF NOT EXISTS idx_correction_events_field
    ON correction_events (field);

-- Migration 032: cloud_correction_history for adaptive cloud formation
CREATE TABLE IF NOT EXISTS cloud_correction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor_key_a TEXT,
    vendor_key_b TEXT,
    vendor_similarity REAL NOT NULL DEFAULT 0.0,
    amount_diff REAL NOT NULL DEFAULT 0.0,
    date_diff_days INTEGER NOT NULL DEFAULT 0,
    location_distance REAL,
    was_false_positive INTEGER NOT NULL DEFAULT 0,
    source_bundle_type TEXT,
    target_bundle_type TEXT,
    item_overlap REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_cch_vendor_a
    ON cloud_correction_history (vendor_key_a);
CREATE INDEX IF NOT EXISTS idx_cch_vendor_b
    ON cloud_correction_history (vendor_key_b);
CREATE INDEX IF NOT EXISTS idx_cch_false_positive
    ON cloud_correction_history (was_false_positive);
CREATE INDEX IF NOT EXISTS idx_cch_created
    ON cloud_correction_history (created_at);

-- Analytics indexes (migration 015)
CREATE INDEX IF NOT EXISTS idx_facts_event_date ON facts(event_date);
CREATE INDEX IF NOT EXISTS idx_facts_vendor ON facts(vendor);
CREATE INDEX IF NOT EXISTS idx_facts_vendor_key ON facts(vendor_key);
CREATE INDEX IF NOT EXISTS idx_facts_type_date ON facts(fact_type, event_date);
CREATE INDEX IF NOT EXISTS idx_fact_items_fact_category ON fact_items(fact_id, category);
CREATE INDEX IF NOT EXISTS idx_annotations_target ON annotations(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_cloud_bundles_cloud ON cloud_bundles(cloud_id);
CREATE INDEX IF NOT EXISTS idx_bundles_document ON bundles(document_id);

-- Migration 033: FTS5 index for product name matching
CREATE VIRTUAL TABLE IF NOT EXISTS product_name_fts USING fts5(
    item_id UNINDEXED,
    name,
    name_normalized,
    brand,
    category,
    vendor_key UNINDEXED,
    unit_quantity UNINDEXED,
    unit UNINDEXED,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS trg_fts_insert AFTER INSERT ON fact_items
WHEN (NEW.brand IS NOT NULL AND NEW.brand != '') OR (NEW.category IS NOT NULL AND NEW.category != '')
BEGIN
    INSERT OR REPLACE INTO product_name_fts (item_id, name, name_normalized, brand, category, vendor_key, unit_quantity, unit)
    SELECT NEW.id, NEW.name, NEW.name_normalized, NEW.brand, NEW.category, f.vendor_key, NEW.unit_quantity, NEW.unit
    FROM facts f WHERE f.id = NEW.fact_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_fts_update AFTER UPDATE ON fact_items
WHEN (NEW.brand IS NOT NULL AND NEW.brand != '') OR (NEW.category IS NOT NULL AND NEW.category != '')
BEGIN
    DELETE FROM product_name_fts WHERE item_id = NEW.id;
    INSERT INTO product_name_fts (item_id, name, name_normalized, brand, category, vendor_key, unit_quantity, unit)
    SELECT NEW.id, NEW.name, NEW.name_normalized, NEW.brand, NEW.category, f.vendor_key, NEW.unit_quantity, NEW.unit
    FROM facts f WHERE f.id = NEW.fact_id;
END;

CREATE TRIGGER IF NOT EXISTS trg_fts_delete AFTER DELETE ON fact_items
BEGIN
    DELETE FROM product_name_fts WHERE item_id = OLD.id;
END;
