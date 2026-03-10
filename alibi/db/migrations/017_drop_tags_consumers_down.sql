-- Migration 017 down: Recreate tags, consumers, and related tables

CREATE TABLE IF NOT EXISTS tags (
    id TEXT PRIMARY KEY,
    space_id TEXT REFERENCES spaces(id),
    name TEXT NOT NULL,
    path TEXT NOT NULL,
    type TEXT,
    color TEXT,
    parent_id TEXT REFERENCES tags(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(space_id, path)
);

CREATE TABLE IF NOT EXISTS item_tags (
    item_id TEXT REFERENCES items(id),
    tag_id TEXT REFERENCES tags(id),
    PRIMARY KEY (item_id, tag_id)
);

CREATE TABLE IF NOT EXISTS vendor_patterns (
    id TEXT PRIMARY KEY,
    pattern TEXT NOT NULL,
    vendor_name TEXT,
    default_category TEXT,
    default_tags JSON,
    confidence DECIMAL(3,2),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME,
    usage_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS consumers (
    id TEXT PRIMARY KEY,
    space_id TEXT REFERENCES spaces(id),
    name TEXT NOT NULL,
    user_id TEXT REFERENCES users(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS line_item_allocations (
    id TEXT PRIMARY KEY,
    line_item_id TEXT REFERENCES fact_items(id),
    consumer_id TEXT REFERENCES consumers(id),
    share DECIMAL(5,4) DEFAULT 1.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_allocations_line_item ON line_item_allocations(line_item_id);
CREATE INDEX IF NOT EXISTS idx_allocations_consumer ON line_item_allocations(consumer_id);

DELETE FROM schema_version WHERE version = 17;
