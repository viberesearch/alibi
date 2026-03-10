-- Migration 006: Add unit_quantity to line_items
-- Stores the content/volume per item (e.g. 400 for "400g can")

ALTER TABLE line_items ADD COLUMN unit_quantity DECIMAL(10,3);

INSERT OR IGNORE INTO schema_version (version) VALUES (6);
