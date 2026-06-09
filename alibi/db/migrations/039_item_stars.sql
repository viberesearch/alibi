-- Migration 039: Materialised item_stars analytics table
--
-- "Item as star": each fact_item is an observation that can be filtered and
-- aggregated along every axis. The A-axis query (services.query.
-- list_fact_items_with_fact) joins fact_items to facts on every call. For the
-- analytics surface (avg comparable_unit_price by comparable_name across
-- vendors/countries/periods, price trends, basket composition) those joins and
-- GROUP BYs are hot, so this denormalises the item axes together with the
-- parent fact's vendor / vendor_key / currency / country / event_date /
-- event_time into one indexed table.
--
-- It is a MATERIALISED mirror, not a source of truth: fact_items + facts remain
-- canonical. Kept in sync by refresh_fact() on the collapse/store path and the
-- per-fact enrichment hook, and fully rebuildable via `lt items rebuild`
-- (services.item_stars.rebuild_item_stars) so it can never silently drift.

-- UP
CREATE TABLE IF NOT EXISTS item_stars (
    -- Item axes (mirrors the analytic columns of fact_items). ON DELETE CASCADE
    -- on both FKs: item_stars is a derived mirror, so removing a fact_item or a
    -- fact (delete_fact, recollapse cleanup, dedup) auto-prunes its star rows.
    item_id TEXT PRIMARY KEY REFERENCES fact_items(id) ON DELETE CASCADE,
    fact_id TEXT NOT NULL REFERENCES facts(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    name_normalized TEXT,
    comparable_name TEXT,
    quantity DECIMAL(10,3),
    unit TEXT,
    unit_price DECIMAL(10,2),
    total_price DECIMAL(10,2),
    brand TEXT,
    category TEXT,
    category_path TEXT,
    comparable_unit_price DECIMAL(10,4),
    comparable_unit TEXT,
    product_variant TEXT,
    -- Parent fact axes (denormalised so each star is self-describing)
    fact_type TEXT,
    vendor TEXT,
    vendor_key TEXT,
    currency TEXT,
    country TEXT,
    event_date DATE,
    event_time TEXT,
    refreshed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_item_stars_fact ON item_stars(fact_id);
CREATE INDEX IF NOT EXISTS idx_item_stars_comparable_name
    ON item_stars(comparable_name);
CREATE INDEX IF NOT EXISTS idx_item_stars_category_path
    ON item_stars(category_path);
CREATE INDEX IF NOT EXISTS idx_item_stars_vendor_key ON item_stars(vendor_key);
CREATE INDEX IF NOT EXISTS idx_item_stars_country ON item_stars(country);
CREATE INDEX IF NOT EXISTS idx_item_stars_currency ON item_stars(currency);
CREATE INDEX IF NOT EXISTS idx_item_stars_event_date ON item_stars(event_date);

-- Backfill from existing facts/fact_items so the table is populated immediately
-- on migration (fresh installs start empty and fill as facts collapse).
INSERT OR IGNORE INTO item_stars (
    item_id, fact_id, name, name_normalized, comparable_name, quantity, unit,
    unit_price, total_price, brand, category, category_path,
    comparable_unit_price, comparable_unit, product_variant,
    fact_type, vendor, vendor_key, currency, country, event_date, event_time
)
SELECT
    fi.id, fi.fact_id, fi.name, fi.name_normalized, fi.comparable_name,
    fi.quantity, fi.unit, fi.unit_price, fi.total_price, fi.brand, fi.category,
    fi.category_path, fi.comparable_unit_price, fi.comparable_unit,
    fi.product_variant,
    f.fact_type, f.vendor, f.vendor_key, f.currency, f.country,
    f.event_date, f.event_time
FROM fact_items fi
JOIN facts f ON fi.fact_id = f.id;

INSERT OR IGNORE INTO schema_version (version) VALUES (39);
