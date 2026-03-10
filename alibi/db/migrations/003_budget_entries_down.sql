-- Migration 003 Down: Remove Budget Entries
-- Reverts budget_entries table creation.

DROP TABLE IF EXISTS budget_entries;
DELETE FROM schema_version WHERE version = 3;
