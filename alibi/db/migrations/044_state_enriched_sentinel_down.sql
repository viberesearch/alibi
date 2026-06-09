-- Migration 044 DOWN: remove the product-state idempotency sentinel column

ALTER TABLE fact_items DROP COLUMN state_enriched;

DELETE FROM schema_version WHERE version = 44;
