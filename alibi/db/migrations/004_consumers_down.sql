-- Migration 004 Down: Remove Consumers and Allocations
-- Reverts consumer tracking tables.

DROP TABLE IF EXISTS line_item_allocations;
DROP TABLE IF EXISTS consumers;
DELETE FROM schema_version WHERE version = 4;
