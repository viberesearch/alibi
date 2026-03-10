-- Migration 003: Budget Entries
-- Adds budget_entries table for storing individual budget line items per scenario.

-- Up migration
-- ============

CREATE TABLE IF NOT EXISTS budget_entries (
    id TEXT PRIMARY KEY,
    scenario_id TEXT REFERENCES budgets(id),
    category TEXT NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    currency TEXT DEFAULT 'EUR',
    period TEXT NOT NULL,  -- YYYY-MM format
    note TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_budget_entries_scenario ON budget_entries(scenario_id);
CREATE INDEX IF NOT EXISTS idx_budget_entries_period ON budget_entries(period);

INSERT OR IGNORE INTO schema_version (version) VALUES (3);
