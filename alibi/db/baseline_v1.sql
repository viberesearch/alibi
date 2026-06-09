-- Alibi v1 baseline schema (schema_version = 1).
--
-- This is the canonical starting point of the migration chain: the pre-v2
-- database that migrations 002+ transform into the head schema. It is NOT a
-- migration (the chain floor is 002); it is the seed applied before the chain.
--
-- Two consumers read this file:
--   * scripts/generate_schema.py — builds alibi/db/schema.sql by applying this
--     baseline then the full migration chain and dumping the result.
--   * tests/test_schema_migration_consistency.py — builds the "migrated" database
--     this same way to assert it matches the generated schema.sql snapshot.
--
-- Most of these v1 tables are later dropped or reshaped by the chain (e.g.
-- migration 011 drops artifacts/transactions/line_items); only what survives to
-- head appears in schema.sql. Do not hand-edit schema.sql — regenerate it.
PRAGMA foreign_keys = ON;

CREATE TABLE users (id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE spaces (id TEXT PRIMARY KEY, name TEXT NOT NULL, type TEXT,
                     owner_id TEXT REFERENCES users(id),
                     created_at DATETIME DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE tags (id TEXT PRIMARY KEY, space_id TEXT, name TEXT NOT NULL,
                   path TEXT NOT NULL, type TEXT, color TEXT, parent_id TEXT,
                   created_at DATETIME DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE artifacts (id TEXT PRIMARY KEY, space_id TEXT, type TEXT,
                        file_path TEXT NOT NULL, file_hash TEXT NOT NULL,
                        perceptual_hash TEXT, vendor TEXT, vendor_id TEXT,
                        document_id TEXT, document_date DATE,
                        amount DECIMAL(10,2), currency TEXT DEFAULT 'EUR',
                        raw_text TEXT, extracted_data JSON, status TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        modified_at DATETIME,
                        created_by TEXT REFERENCES users(id));
CREATE TABLE transactions (id TEXT PRIMARY KEY, space_id TEXT, type TEXT,
                           vendor TEXT, description TEXT, amount DECIMAL NOT NULL,
                           currency TEXT DEFAULT 'EUR',
                           transaction_date DATE NOT NULL, payment_method TEXT,
                           card_last4 TEXT, account_reference TEXT, status TEXT,
                           note_path TEXT,
                           created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                           modified_at DATETIME, created_by TEXT);
CREATE TABLE items (id TEXT PRIMARY KEY, space_id TEXT, name TEXT,
                    category TEXT, model TEXT, serial_number TEXT,
                    purchase_date DATE, purchase_price DECIMAL,
                    current_value DECIMAL, currency TEXT DEFAULT 'EUR',
                    status TEXT, warranty_expires DATE, warranty_type TEXT,
                    insurance_covered BOOLEAN, note_path TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    modified_at DATETIME, created_by TEXT);
CREATE TABLE line_items (id TEXT PRIMARY KEY,
                         artifact_id TEXT REFERENCES artifacts(id),
                         transaction_id TEXT REFERENCES transactions(id),
                         name TEXT NOT NULL, quantity DECIMAL(10,3) DEFAULT 1,
                         unit_price DECIMAL(10,2), total_price DECIMAL(10,2),
                         category TEXT, item_id TEXT REFERENCES items(id),
                         created_at DATETIME DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE schema_version (version INTEGER PRIMARY KEY,
                             applied_at DATETIME DEFAULT CURRENT_TIMESTAMP);
INSERT INTO schema_version (version) VALUES (1);
