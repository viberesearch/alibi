-- Revert migration 022: restore original CHECK constraint (without near_amount)
-- WARNING: any rows with match_type='near_amount' will be lost

CREATE TABLE cloud_bundles_old (
    cloud_id TEXT NOT NULL REFERENCES clouds(id),
    bundle_id TEXT NOT NULL REFERENCES bundles(id),
    match_type TEXT NOT NULL CHECK(match_type IN (
        'exact_amount', 'sum_of_parts', 'vendor+date', 'item_overlap', 'manual'
    )),
    match_confidence REAL DEFAULT 0.0,
    PRIMARY KEY (cloud_id, bundle_id)
);

INSERT OR IGNORE INTO cloud_bundles_old SELECT * FROM cloud_bundles;
DROP TABLE cloud_bundles;
ALTER TABLE cloud_bundles_old RENAME TO cloud_bundles;

CREATE INDEX IF NOT EXISTS idx_cloud_bundles_cloud ON cloud_bundles(cloud_id);

DELETE FROM schema_version WHERE version = 22;
