-- Migration 022: Add 'near_amount' to cloud_bundles.match_type CHECK constraint
-- The CloudMatchType enum includes NEAR_AMOUNT but the CHECK constraint didn't.
-- INSERT OR IGNORE silently drops rows failing CHECK, causing silent data loss.

-- SQLite requires table recreation to alter CHECK constraints
CREATE TABLE cloud_bundles_new (
    cloud_id TEXT NOT NULL REFERENCES clouds(id),
    bundle_id TEXT NOT NULL REFERENCES bundles(id),
    match_type TEXT NOT NULL CHECK(match_type IN (
        'exact_amount', 'near_amount', 'sum_of_parts', 'vendor+date', 'item_overlap', 'manual'
    )),
    match_confidence REAL DEFAULT 0.0,
    PRIMARY KEY (cloud_id, bundle_id)
);

INSERT INTO cloud_bundles_new SELECT * FROM cloud_bundles;
DROP TABLE cloud_bundles;
ALTER TABLE cloud_bundles_new RENAME TO cloud_bundles;

CREATE INDEX IF NOT EXISTS idx_cloud_bundles_cloud ON cloud_bundles(cloud_id);

INSERT OR IGNORE INTO schema_version (version) VALUES (22);
