-- Add comparable unit price for cross-shop comparison
-- comparable_unit_price: normalized price (per kg, per liter, or per piece)
-- comparable_unit: the standard unit (kg, l, pcs)

ALTER TABLE line_items ADD COLUMN comparable_unit_price DECIMAL(10,4);
ALTER TABLE line_items ADD COLUMN comparable_unit TEXT;

INSERT OR IGNORE INTO schema_version (version) VALUES (7);
