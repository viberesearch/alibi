-- Migration 034: Add salt column to api_keys for PBKDF2 hashing,
-- and barcode index on fact_items.

-- Per-key random salt for PBKDF2 hashing. NULL = legacy unsalted SHA-256.
ALTER TABLE api_keys ADD COLUMN salt BLOB DEFAULT NULL;

-- Barcode lookup index for enrichment cascade and cross-vendor matching.
CREATE INDEX IF NOT EXISTS idx_fact_items_barcode ON fact_items(barcode);

INSERT OR IGNORE INTO schema_version (version) VALUES (34);
