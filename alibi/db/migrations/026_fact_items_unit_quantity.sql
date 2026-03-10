-- Add unit_quantity column to fact_items for measured amounts (weight/volume).
-- quantity = purchase count (e.g. 4 cans), unit_quantity = per-unit measure (e.g. 0.355 l).
ALTER TABLE fact_items ADD COLUMN unit_quantity DECIMAL(10,3);

INSERT OR IGNORE INTO schema_version (version) VALUES (26);
