-- Down migration 045: drop EUR-normalisation storage.
--
-- SQLite supports DROP COLUMN (>= 3.35, which Alibi requires), so the added
-- columns can be removed directly; the rates cache table is dropped wholesale.

-- DOWN
ALTER TABLE item_stars DROP COLUMN comparable_unit_price_eur;
ALTER TABLE item_stars DROP COLUMN total_price_eur;
ALTER TABLE facts DROP COLUMN eur_rate;
DROP TABLE IF EXISTS exchange_rates;

DELETE FROM schema_version WHERE version = 45;
