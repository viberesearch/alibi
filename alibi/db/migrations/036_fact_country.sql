-- Migration 036: Add country/jurisdiction field to facts
-- Stores the inferred transaction jurisdiction (ISO 3166-1 alpha-2, or the
-- sentinel 'CY-NORTH' for the Turkish Republic of Northern Cyprus) so
-- cross-country analysis can distinguish CY/AT/CA/TR and resolve the canonical
-- currency (e.g. Northern Cyprus -> TRY).

-- UP
ALTER TABLE facts ADD COLUMN country TEXT DEFAULT NULL;

CREATE INDEX IF NOT EXISTS idx_facts_country ON facts(country);

INSERT OR IGNORE INTO schema_version (version) VALUES (36);
