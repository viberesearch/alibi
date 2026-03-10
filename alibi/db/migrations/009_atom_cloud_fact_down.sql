-- Migration 009 Down: Remove Atom-Cloud-Fact tables
-- Reverts the v2 observation-centric schema.

DROP TABLE IF EXISTS fact_items;
DROP TABLE IF EXISTS facts;
DROP TABLE IF EXISTS cloud_bundles;
DROP TABLE IF EXISTS clouds;
DROP TABLE IF EXISTS bundle_atoms;
DROP TABLE IF EXISTS bundles;
DROP TABLE IF EXISTS atoms;
DROP TABLE IF EXISTS documents;
DELETE FROM schema_version WHERE version = 9;
