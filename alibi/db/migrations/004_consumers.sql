-- Migration 004: Consumers and Line Item Allocations
-- Adds consumer tracking and per-line-item allocation for shared expenses.

-- Up migration
-- ============

CREATE TABLE IF NOT EXISTS consumers (
    id TEXT PRIMARY KEY,
    space_id TEXT REFERENCES spaces(id),
    name TEXT NOT NULL,
    user_id TEXT REFERENCES users(id),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS line_item_allocations (
    id TEXT PRIMARY KEY,
    line_item_id TEXT REFERENCES line_items(id),
    consumer_id TEXT REFERENCES consumers(id),
    share DECIMAL(5,4) DEFAULT 1.0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_allocations_line_item ON line_item_allocations(line_item_id);
CREATE INDEX IF NOT EXISTS idx_allocations_consumer ON line_item_allocations(consumer_id);

INSERT OR IGNORE INTO schema_version (version) VALUES (4);
