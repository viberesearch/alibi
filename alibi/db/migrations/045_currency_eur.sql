-- Migration 045: EUR-normalised amounts for currency-comparable analytics
--
-- Facts are recorded in their receipt currency (EUR, CAD, TRY, ...). The spend
-- and price analytics summed those amounts together as if 1 CAD == 1 TRY == 1
-- EUR, which silently inflates/blends every cross-currency total. This migration
-- adds the storage to normalise everything to EUR using the historical rate at
-- each fact's event_date (fetched + cached by `lt fx backfill`):
--
--   exchange_rates  -- cached EUR-per-unit rate for a (currency, date)
--   facts.eur_rate  -- the resolved EUR-per-unit for this fact (1.0 for EUR)
--   item_stars.total_price_eur / comparable_unit_price_eur
--                   -- the item's amounts pre-multiplied by the fact's eur_rate,
--                      materialised on rebuild so the aggregations stay simple
--
-- A NULL eur_rate / *_eur means "not yet converted" (a non-EUR fact before
-- backfill, or a date with no available rate): such rows drop out of the
-- EUR-only aggregations rather than blending a foreign amount back in. EUR facts
-- are seeded to 1.0 here so they need no backfill and existing tests (which seed
-- EUR data) get correct *_eur values straight from the rebuild.

-- UP
CREATE TABLE IF NOT EXISTS exchange_rates (
    base TEXT NOT NULL,          -- ISO 4217 currency code (e.g. CAD, TRY)
    rate_date TEXT NOT NULL,     -- ISO date the rate applies to
    eur_per_unit REAL NOT NULL,  -- EUR for 1 unit of `base` on `rate_date`
    fetched_at TEXT,             -- when this rate was fetched/cached
    PRIMARY KEY (base, rate_date)
);

ALTER TABLE facts ADD COLUMN eur_rate REAL DEFAULT NULL;
ALTER TABLE item_stars ADD COLUMN total_price_eur REAL DEFAULT NULL;
ALTER TABLE item_stars ADD COLUMN comparable_unit_price_eur REAL DEFAULT NULL;

-- EUR (and currency-less) facts convert 1:1 — seed them so no backfill is needed.
UPDATE facts SET eur_rate = 1.0 WHERE eur_rate IS NULL AND COALESCE(currency, 'EUR') = 'EUR';

INSERT OR IGNORE INTO schema_version (version) VALUES (45);
