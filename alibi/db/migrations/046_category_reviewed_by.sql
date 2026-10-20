-- Migration 046: marker for the hosted Jev category review
--
-- `lt enrich review-jev` (alibi/enrichment/jev_review.py) re-reads items that
-- already carry a category_path. Once a row has been answered it is stamped
-- with '<model>@t<TAXONOMY_VERSION>' (e.g. 'jev-1.14.0@t1'), so later runs send
-- only new products, and a pinned-model or taxonomy change makes every row
-- eligible again.
--
-- Bookkeeping only, so it is NOT mirrored to item_stars.

-- UP
ALTER TABLE fact_items ADD COLUMN category_reviewed_by TEXT DEFAULT NULL;

INSERT OR IGNORE INTO schema_version (version) VALUES (46);
