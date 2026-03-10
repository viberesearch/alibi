-- Deduplicate facts: keep newest fact per cloud_id, delete older duplicates.
-- Then add UNIQUE index to prevent future duplicates.

DELETE FROM fact_items WHERE fact_id IN (
    SELECT f.id FROM facts f
    WHERE f.id NOT IN (
        SELECT id FROM (
            SELECT id, cloud_id, ROW_NUMBER() OVER (
                PARTITION BY cloud_id ORDER BY created_at DESC
            ) AS rn
            FROM facts
        ) ranked
        WHERE rn = 1
    )
);

DELETE FROM facts WHERE id NOT IN (
    SELECT id FROM (
        SELECT id, cloud_id, ROW_NUMBER() OVER (
            PARTITION BY cloud_id ORDER BY created_at DESC
        ) AS rn
        FROM facts
    ) ranked
    WHERE rn = 1
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_cloud_unique ON facts(cloud_id);

INSERT OR IGNORE INTO schema_version (version) VALUES (21);
