-- Migration 005: Add vendor contact details and transaction time
-- Adds address, phone, website, registration to artifacts
-- Adds transaction_time to transactions

ALTER TABLE artifacts ADD COLUMN vendor_address TEXT;
ALTER TABLE artifacts ADD COLUMN vendor_phone TEXT;
ALTER TABLE artifacts ADD COLUMN vendor_website TEXT;
ALTER TABLE artifacts ADD COLUMN vendor_registration TEXT;

ALTER TABLE transactions ADD COLUMN transaction_time TEXT;

INSERT OR IGNORE INTO schema_version (version) VALUES (5);
