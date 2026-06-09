-- Migration 037: Add event_time to facts
-- The extraction already reads the transaction time, but collapse dropped it
-- (facts kept event_date only). Storing event_time (HH:MM:SS) retains the full
-- date+time so same-day transactions are distinguishable and analysable.

-- UP
ALTER TABLE facts ADD COLUMN event_time TEXT DEFAULT NULL;

INSERT OR IGNORE INTO schema_version (version) VALUES (37);
