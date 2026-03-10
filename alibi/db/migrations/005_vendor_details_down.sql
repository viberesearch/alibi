-- Rollback migration 005: Remove vendor contact details and transaction time
-- SQLite doesn't support DROP COLUMN before 3.35.0, so we recreate tables

-- Note: This is a destructive rollback. Data in the dropped columns will be lost.
-- For SQLite >= 3.35.0:
ALTER TABLE artifacts DROP COLUMN vendor_address;
ALTER TABLE artifacts DROP COLUMN vendor_phone;
ALTER TABLE artifacts DROP COLUMN vendor_website;
ALTER TABLE artifacts DROP COLUMN vendor_registration;

ALTER TABLE transactions DROP COLUMN transaction_time;

DELETE FROM schema_version WHERE version = 5;
