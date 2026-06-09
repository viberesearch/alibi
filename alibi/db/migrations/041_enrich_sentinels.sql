-- Migration 041: idempotency sentinels for the units / comparable_name passes
--
-- The `units` and `comparable_names` enrichment passes select rows by the
-- absence of their result (unit_quantity IS NULL / comparable_name IS NULL).
-- A row the model can never solve (a count item with no size, a non-product
-- line like a total/tax) stays NULL forever and was therefore re-sent to the
-- LLM on EVERY run — wasted local inference that never converges.
--
-- The attributes pass already avoids this by writing a `{}` sentinel into its
-- own column. unit_quantity / comparable_name can't hold a sentinel (they are
-- numeric / user-facing), so each pass gets a dedicated marker column instead:
-- set to 1 once the model has answered for that row (value found OR explicit
-- "no result"), which removes it from the pending set. Rows the model dropped
-- or that errored are left unmarked and still retried.
--
-- Bookkeeping only — not analytics, so NOT mirrored to item_stars.

-- UP
ALTER TABLE fact_items ADD COLUMN unit_enriched INTEGER DEFAULT NULL;
ALTER TABLE fact_items ADD COLUMN comparable_name_enriched INTEGER DEFAULT NULL;

INSERT OR IGNORE INTO schema_version (version) VALUES (41);
