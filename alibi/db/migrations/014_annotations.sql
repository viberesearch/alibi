-- Migration 014: Annotations table for user-defined metadata on entities.
-- Supports person attribution, project tagging, purchase splitting, notes.
-- Open-ended key-value design: no migration needed for new annotation types.

CREATE TABLE IF NOT EXISTS annotations (
    id TEXT PRIMARY KEY,
    annotation_type TEXT NOT NULL,  -- "person", "project", "category", "split", "note", etc.
    target_type TEXT NOT NULL CHECK(target_type IN ('fact', 'fact_item', 'vendor', 'identity')),
    target_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    metadata JSON,  -- Extra structured data (e.g., split ratios, item lists)
    source TEXT NOT NULL DEFAULT 'user',  -- "user", "auto", "inference"
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_annotations_target ON annotations(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_annotations_type ON annotations(annotation_type);
CREATE INDEX IF NOT EXISTS idx_annotations_key_value ON annotations(key, value);

INSERT OR IGNORE INTO schema_version (version) VALUES (14);
