-- Migration 032: Cloud formation correction history.
-- Records feature vectors from bundle moves (merge/split corrections)
-- for adaptive weight learning in cloud formation.

CREATE TABLE IF NOT EXISTS cloud_correction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vendor_key_a TEXT,
    vendor_key_b TEXT,
    vendor_similarity REAL NOT NULL DEFAULT 0.0,
    amount_diff REAL NOT NULL DEFAULT 0.0,
    date_diff_days INTEGER NOT NULL DEFAULT 0,
    location_distance REAL,
    was_false_positive INTEGER NOT NULL DEFAULT 0,
    source_bundle_type TEXT,
    target_bundle_type TEXT,
    item_overlap REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_cch_vendor_a
    ON cloud_correction_history (vendor_key_a);

CREATE INDEX IF NOT EXISTS idx_cch_vendor_b
    ON cloud_correction_history (vendor_key_b);

CREATE INDEX IF NOT EXISTS idx_cch_false_positive
    ON cloud_correction_history (was_false_positive);

CREATE INDEX IF NOT EXISTS idx_cch_created
    ON cloud_correction_history (created_at);

INSERT OR IGNORE INTO schema_version (version) VALUES (32);
