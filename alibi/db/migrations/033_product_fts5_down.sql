DROP TRIGGER IF EXISTS trg_fts_insert;
DROP TRIGGER IF EXISTS trg_fts_update;
DROP TRIGGER IF EXISTS trg_fts_delete;
DROP TABLE IF EXISTS product_name_fts;
DELETE FROM schema_version WHERE version = 33;
