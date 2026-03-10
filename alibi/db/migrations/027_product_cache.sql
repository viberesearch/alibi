-- Product cache for external data lookups (Open Food Facts, etc.).
-- Keyed by barcode; stores full product JSON from external source.
CREATE TABLE IF NOT EXISTS product_cache (
    barcode    TEXT PRIMARY KEY,
    data       TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'openfoodfacts',
    fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_version (version) VALUES (27);
