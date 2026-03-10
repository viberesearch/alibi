-- Track enrichment provenance on fact_items.
-- enrichment_source: who set brand/category (openfoodfacts, product_resolver,
--   llm_inference, cloud_api, user_confirmed, manual).
-- enrichment_confidence: 0.0-1.0 how confident the assignment is.
ALTER TABLE fact_items ADD COLUMN enrichment_source TEXT;
ALTER TABLE fact_items ADD COLUMN enrichment_confidence REAL;

INSERT OR IGNORE INTO schema_version (version) VALUES (28);
