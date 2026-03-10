-- Migration 031: Create correction_events table.
-- Foundation for adaptive learning: logs every correction made to facts,
-- fact items, vendors, and bundles so feedback loops can improve extraction.

CREATE TABLE IF NOT EXISTS correction_events (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,  -- fact, fact_item, vendor, bundle
    entity_id TEXT NOT NULL,
    field TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    source TEXT NOT NULL,       -- api, cli, mcp, telegram, system
    user_id TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_correction_events_entity
    ON correction_events (entity_type, entity_id);

CREATE INDEX IF NOT EXISTS idx_correction_events_created
    ON correction_events (created_at);

CREATE INDEX IF NOT EXISTS idx_correction_events_field
    ON correction_events (field);

INSERT OR IGNORE INTO schema_version (version) VALUES (31);
