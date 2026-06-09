-- Migration 042: idempotency sentinel for the categorize pass
--
-- Like units / comparable_names (v41), the categorize pass selects rows by the
-- absence of its result (category_path IS NULL), so a line the model can never
-- map to a valid taxonomy path was re-sent to the LLM on every run.
--
-- A plain boolean marker would break re-categorisation: bumping
-- taxonomy.TAXONOMY_VERSION and clearing category_path is the documented signal
-- to recategorise, and a boolean marker would keep those rows excluded. So this
-- column stores the TAXONOMY_VERSION a row was categorised under instead. The
-- pending SELECT re-selects a row when it has no category_path AND was never
-- processed OR was processed under an OLDER taxonomy version — which both stops
-- re-LLM-ing unsolvable rows and gives TAXONOMY_VERSION the recategorise teeth
-- its docstring already promises.
--
-- Bookkeeping only — not analytics, so NOT mirrored to item_stars.

-- UP
ALTER TABLE fact_items ADD COLUMN category_taxonomy_version INTEGER DEFAULT NULL;

INSERT OR IGNORE INTO schema_version (version) VALUES (42);
