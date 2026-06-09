-- Migration 041 DOWN: remove the enrichment idempotency sentinel columns

ALTER TABLE fact_items DROP COLUMN comparable_name_enriched;
ALTER TABLE fact_items DROP COLUMN unit_enriched;

DELETE FROM schema_version WHERE version = 41;
