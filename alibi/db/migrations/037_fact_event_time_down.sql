-- Migration 037 DOWN: Remove event_time from facts

ALTER TABLE facts DROP COLUMN event_time;

DELETE FROM schema_version WHERE version = 37;
