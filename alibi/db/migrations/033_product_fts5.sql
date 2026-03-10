-- Migration 033: FTS5 index for product name matching.
-- Virtual table for fast text search on fact_item names.
-- Replaces O(N) linear scan in product_resolver.py with sub-linear lookup.

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

-- Populate from existing enriched items
INSERT INTO product_name_fts (item_id, name, name_normalized, brand, category, vendor_key, unit_quantity, unit)
SELECT fi.id, fi.name, fi.name_normalized, fi.brand, fi.category, f.vendor_key, fi.unit_quantity, fi.unit
FROM fact_items fi
JOIN facts f ON fi.fact_id = f.id
WHERE (fi.brand IS NOT NULL AND fi.brand != '')
   OR (fi.category IS NOT NULL AND fi.category != '');

-- Triggers to keep FTS in sync with fact_items
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

INSERT OR IGNORE INTO schema_version (version) VALUES (33);
