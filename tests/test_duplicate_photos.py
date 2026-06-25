"""Document-level duplicate-PHOTO detection via perceptual hash.

find_duplicate_photos surfaces near-duplicate receipt photos straight from
documents.perceptual_hash, independent of the amount+date coincidence fact
dedup relies on.
"""

from __future__ import annotations

import os

os.environ["ALIBI_TESTING"] = "1"

from alibi.clouds.dedup import find_duplicate_photos


def _seed_document(
    db,
    doc_id: str,
    phash: str,
    *,
    cloud_id: str | None = None,
    with_fact: bool = False,
) -> None:
    conn = db.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash, perceptual_hash) "
        "VALUES (?, ?, ?, ?)",
        (doc_id, f"/inbox/{doc_id}.jpg", f"fh-{doc_id}", phash),
    )
    if cloud_id:
        conn.execute(
            "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')",
            (cloud_id,),
        )
        conn.execute(
            "INSERT OR IGNORE INTO bundles "
            "(id, document_id, bundle_type, cloud_id) VALUES (?, ?, 'basket', ?)",
            (f"bundle-{doc_id}", doc_id, cloud_id),
        )
        if with_fact:
            conn.execute(
                "INSERT OR IGNORE INTO facts "
                "(id, cloud_id, fact_type, vendor, total_amount, currency) "
                "VALUES (?, ?, 'purchase', 'Shop', 10.0, 'EUR')",
                (f"fact-{cloud_id}", cloud_id),
            )
    conn.commit()


# Two 64-bit hex dHashes 2 bits apart (near-duplicate), plus a far one.
_PHASH_A = "ffffffffffffffff"
_PHASH_A_NEAR = "fffffffffffffff3"  # differs in 2 low bits
_PHASH_FAR = "0000000000000000"  # 64 bits away


class TestFindDuplicatePhotos:
    def test_no_duplicates_when_all_distinct(self, db):
        _seed_document(db, "d1", _PHASH_A)
        _seed_document(db, "d2", _PHASH_FAR)
        assert find_duplicate_photos(db) == []

    def test_near_match_grouped(self, db):
        _seed_document(db, "d1", _PHASH_A)
        _seed_document(db, "d2", _PHASH_A_NEAR)
        groups = find_duplicate_photos(db)
        assert len(groups) == 1
        assert {d.document_id for d in groups[0].documents} == {"d1", "d2"}

    def test_distance_threshold_respected(self, db):
        _seed_document(db, "d1", _PHASH_A)
        _seed_document(db, "d2", _PHASH_A_NEAR)  # 2 bits apart
        assert find_duplicate_photos(db, max_distance=1) == []
        assert len(find_duplicate_photos(db, max_distance=2)) == 1

    def test_multi_cloud_group_is_actionable(self, db):
        # Same photo, two separate clouds -> un-merged duplicate (actionable).
        _seed_document(db, "d1", _PHASH_A, cloud_id="c1", with_fact=True)
        _seed_document(db, "d2", _PHASH_A_NEAR, cloud_id="c2", with_fact=True)
        groups = find_duplicate_photos(db)
        assert len(groups) == 1
        grp = groups[0]
        assert not grp.collapsed_together
        assert grp.distinct_clouds == {"c1", "c2"}
        assert {d.fact_id for d in grp.documents} == {"fact-c1", "fact-c2"}

    def test_same_cloud_group_is_benign(self, db):
        # Both docs already in one cloud -> collapsed_together (benign).
        _seed_document(db, "d1", _PHASH_A, cloud_id="c1")
        _seed_document(db, "d2", _PHASH_A_NEAR, cloud_id="c1")
        groups = find_duplicate_photos(db)
        assert len(groups) == 1
        assert groups[0].collapsed_together

    def test_transitive_grouping(self, db):
        # A~B and B~C chain into one group even if A and C are >threshold apart.
        _seed_document(db, "a", "ffffffffffffffff")
        _seed_document(db, "b", "fffffffffffffff8")  # 3 bits from a
        _seed_document(db, "c", "ffffffffffffffc0")  # near b, further from a
        groups = find_duplicate_photos(db, max_distance=4)
        assert len(groups) == 1
        assert {d.document_id for d in groups[0].documents} == {"a", "b", "c"}

    def test_ignores_documents_without_phash(self, db):
        _seed_document(db, "d1", _PHASH_A)
        conn = db.get_connection()
        conn.execute(
            "INSERT INTO documents (id, file_path, file_hash, perceptual_hash) "
            "VALUES ('d2', '/x.jpg', 'fh2', NULL)",
        )
        conn.commit()
        assert find_duplicate_photos(db) == []

    def test_actionable_groups_sorted_first(self, db):
        # One benign (same cloud) + one actionable (multi cloud); actionable
        # must sort before benign.
        _seed_document(db, "b1", "aaaaaaaaaaaaaaaa", cloud_id="cb")
        _seed_document(db, "b2", "aaaaaaaaaaaaaaa8", cloud_id="cb")  # benign pair
        _seed_document(db, "a1", "ffffffffffffffff", cloud_id="ca")
        _seed_document(db, "a2", "fffffffffffffff8", cloud_id="cc")  # actionable
        groups = find_duplicate_photos(db, max_distance=4)
        assert len(groups) == 2
        assert not groups[0].collapsed_together  # actionable first
