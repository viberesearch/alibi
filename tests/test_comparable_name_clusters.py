"""Tests for embedding-based comparable_name canonicalization (F2)."""

from __future__ import annotations

import os
from pathlib import Path

os.environ["ALIBI_TESTING"] = "1"

import pytest

from alibi.enrichment.comparable_name_clusters import (
    MergeCluster,
    NameStat,
    _cosine,
    apply_name_merges,
    load_approved_clusters,
    propose_name_merges,
    write_proposal_yaml,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_fact_item(
    db,
    item_id: str,
    name: str,
    *,
    comparable_name: str | None = None,
    comparable_unit: str | None = None,
    category: str | None = None,
) -> None:
    """Insert a fact_item with its supporting chain (one fact per item)."""
    doc_id = f"doc-{item_id}"
    atom_id = f"atom-{item_id}"
    cloud_id = f"cloud-{item_id}"
    fact_id = f"fact-{item_id}"

    conn = db.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
        (doc_id, f"/tmp/{doc_id}.jpg", f"hash-{doc_id}"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO atoms (id, document_id, atom_type, data) "
        "VALUES (?, ?, 'item', '{}')",
        (atom_id, doc_id),
    )
    conn.execute(
        "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')",
        (cloud_id,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO facts "
        "(id, cloud_id, fact_type, vendor, total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', 'Test Store', 10.0, 'EUR', '2026-01-01')",
        (fact_id, cloud_id),
    )
    conn.execute(
        "INSERT OR IGNORE INTO fact_items "
        "(id, fact_id, atom_id, name, comparable_name, comparable_unit, category) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (item_id, fact_id, atom_id, name, comparable_name, comparable_unit, category),
    )
    conn.commit()


def _fake_embedder(groups: dict[str, list[str]]):
    """Build a deterministic embed_fn from synonym groups.

    Each group shares one one-hot axis, so names in the same group are cosine 1.0
    and names in different groups are cosine 0.0 -- a clean, controllable stand-in
    for the real model. Any name not listed gets its own unique axis (a singleton).
    """
    axis: dict[str, int] = {}
    for gi, names in enumerate(groups.values()):
        for n in names:
            axis[n] = gi
    next_axis = len(groups)
    dim = len(groups) + 64  # headroom for unlisted singletons

    def embed(text: str) -> list[float]:
        nonlocal next_axis
        if text not in axis:
            axis[text] = next_axis
            next_axis += 1
        vec = [0.0] * dim
        vec[axis[text] % dim] = 1.0
        return vec

    return embed


# ===========================================================================
# Vector math
# ===========================================================================


class TestCosine:
    def test_identical(self):
        assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_is_zero(self):
        # The embedding-failure fallback: a zero vector never merges with anything.
        assert _cosine([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_length_mismatch_is_zero(self):
        assert _cosine([1.0], [1.0, 0.0]) == 0.0


# ===========================================================================
# Propose
# ===========================================================================


class TestPropose:
    def test_clusters_synonyms_within_unit(self, db):
        _seed_fact_item(
            db, "a1", "ARTICHOKE", comparable_name="artichoke", comparable_unit="kg"
        )
        _seed_fact_item(
            db, "a2", "ARTICHOKES", comparable_name="artichokes", comparable_unit="kg"
        )
        _seed_fact_item(
            db, "o1", "OLIVE OIL", comparable_name="olive oil", comparable_unit="l"
        )

        embed = _fake_embedder({"arti": ["artichoke", "artichokes"]})
        clusters = propose_name_merges(db, embed_fn=embed, threshold=0.9)

        assert len(clusters) == 1
        c = clusters[0]
        assert c.comparable_unit == "kg"
        assert set(m.name for m in c.members) == {"artichoke", "artichokes"}

    def test_never_merges_across_units(self, db):
        # avocado appears as both kg and pcs; the group-by-unit invariant must
        # keep them apart even though the names are identical.
        _seed_fact_item(
            db, "v1", "AVOCADO", comparable_name="avocado", comparable_unit="kg"
        )
        _seed_fact_item(
            db, "v2", "AVOCADO", comparable_name="avocado", comparable_unit="pcs"
        )
        # Same string, identical embedding -- but different units -> no cluster.
        embed = _fake_embedder({})
        clusters = propose_name_merges(db, embed_fn=embed, threshold=0.9)
        assert clusters == []

    def test_distinct_products_not_clustered(self, db):
        _seed_fact_item(
            db, "p1", "APPLE", comparable_name="apple", comparable_unit="kg"
        )
        _seed_fact_item(
            db, "p2", "APRICOT", comparable_name="apricot", comparable_unit="kg"
        )
        embed = _fake_embedder({})  # each gets its own axis -> cosine 0
        clusters = propose_name_merges(db, embed_fn=embed, threshold=0.9)
        assert clusters == []

    def test_canonical_is_highest_count(self, db):
        # "goat milk" has 2 rows, "fr. goat milk" has 1 -> canonical is "goat milk".
        _seed_fact_item(
            db, "g1", "GOAT MILK", comparable_name="goat milk", comparable_unit="l"
        )
        _seed_fact_item(
            db, "g2", "GOAT MILK", comparable_name="goat milk", comparable_unit="l"
        )
        _seed_fact_item(
            db,
            "g3",
            "FR.GOAT MILK",
            comparable_name="fr. goat milk",
            comparable_unit="l",
        )
        embed = _fake_embedder({"goat": ["goat milk", "fr. goat milk"]})
        clusters = propose_name_merges(db, embed_fn=embed, threshold=0.9)
        assert len(clusters) == 1
        assert clusters[0].canonical == "goat milk"
        assert clusters[0].variant_names() == ["fr. goat milk"]

    def test_unitless_bucket_clusters(self, db):
        # NULL and '' units both fold into the unitless key and can cluster.
        _seed_fact_item(db, "u1", "BAG", comparable_name="bag", comparable_unit=None)
        _seed_fact_item(db, "u2", "BAGS", comparable_name="bags", comparable_unit="")
        embed = _fake_embedder({"bag": ["bag", "bags"]})
        clusters = propose_name_merges(db, embed_fn=embed, threshold=0.9)
        assert len(clusters) == 1
        assert clusters[0].comparable_unit == ""

    def test_empty_db_returns_nothing(self, db):
        assert propose_name_merges(db, embed_fn=_fake_embedder({})) == []

    def test_threshold_gates_merges(self, db):
        _seed_fact_item(db, "a1", "X", comparable_name="alpha", comparable_unit="kg")
        _seed_fact_item(db, "a2", "Y", comparable_name="beta", comparable_unit="kg")

        # Two axes 45 degrees apart -> cosine ~0.707.
        def embed(text: str) -> list[float]:
            return {"alpha": [1.0, 0.0], "beta": [1.0, 1.0]}[text]

        assert propose_name_merges(db, embed_fn=embed, threshold=0.9) == []
        clustered = propose_name_merges(db, embed_fn=embed, threshold=0.6)
        assert len(clustered) == 1


# ===========================================================================
# Proposal file round-trip
# ===========================================================================


class TestProposalFile:
    def _sample(self) -> list[MergeCluster]:
        return [
            MergeCluster(
                canonical="artichokes",
                comparable_unit="kg",
                members=[
                    NameStat("artichokes", "kg", 2, ["ARTICHOKES"], ["produce"]),
                    NameStat("artichoke", "kg", 1, ["ΑΓΚΙΝΑΡΕΣ"], ["produce"]),
                ],
            )
        ]

    def test_write_is_deterministic(self, tmp_path: Path):
        path = tmp_path / "p.yaml"
        write_proposal_yaml(
            self._sample(), path, threshold=0.92, generated="2026-06-09"
        )
        text = path.read_text(encoding="utf-8")
        assert "approved: false" in text
        assert "artichokes" in text
        assert "ΑΓΚΙΝΑΡΕΣ" in text  # unicode preserved
        assert "generated: 2026-06-09" in text

    def test_unapproved_file_yields_nothing(self, tmp_path: Path):
        path = tmp_path / "p.yaml"
        write_proposal_yaml(
            self._sample(), path, threshold=0.92, generated="2026-06-09"
        )
        # Nothing flipped to approved -> safe default of applying nothing.
        assert load_approved_clusters(path) == []

    def test_approved_cluster_loads(self, tmp_path: Path):
        path = tmp_path / "p.yaml"
        write_proposal_yaml(
            self._sample(), path, threshold=0.92, generated="2026-06-09"
        )
        text = path.read_text(encoding="utf-8").replace(
            "approved: false\n", "approved: true\n", 1
        )
        path.write_text(text, encoding="utf-8")
        approved = load_approved_clusters(path)
        assert len(approved) == 1
        assert approved[0].canonical == "artichokes"
        assert {m.name for m in approved[0].members} == {"artichoke", "artichokes"}

    def test_empty_file_returns_empty(self, tmp_path: Path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_approved_clusters(path) == []

    def test_malformed_raises(self, tmp_path: Path):
        path = tmp_path / "bad.yaml"
        path.write_text("not_clusters: 1\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_approved_clusters(path)

    def test_approved_without_canonical_raises(self, tmp_path: Path):
        path = tmp_path / "bad.yaml"
        path.write_text(
            "clusters:\n"
            "  - comparable_unit: kg\n"
            "    approved: true\n"
            "    members:\n"
            "      - name: artichoke\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError):
            load_approved_clusters(path)


# ===========================================================================
# Apply
# ===========================================================================


class TestApply:
    def test_rewrites_members_to_canonical(self, db):
        _seed_fact_item(
            db, "a1", "ARTICHOKE", comparable_name="artichoke", comparable_unit="kg"
        )
        _seed_fact_item(
            db, "a2", "ARTICHOKES", comparable_name="artichokes", comparable_unit="kg"
        )

        cluster = MergeCluster(
            canonical="artichokes",
            comparable_unit="kg",
            members=[
                NameStat("artichokes", "kg", 1),
                NameStat("artichoke", "kg", 1),
            ],
        )
        result = apply_name_merges(db, [cluster])

        assert result.rewritten_rows == 1
        names = {
            r["comparable_name"]
            for r in db.fetchall("SELECT comparable_name FROM fact_items")
        }
        assert names == {"artichokes"}

    def test_apply_respects_unit_boundary(self, db):
        # Same name in a different unit must NOT be rewritten by a kg cluster.
        _seed_fact_item(db, "k1", "X", comparable_name="herb", comparable_unit="kg")
        _seed_fact_item(db, "p1", "Y", comparable_name="herb", comparable_unit="pcs")
        cluster = MergeCluster(
            canonical="herbs",
            comparable_unit="kg",
            members=[NameStat("herbs", "kg", 0), NameStat("herb", "kg", 1)],
        )
        apply_name_merges(db, [cluster])
        rows = {
            (r["comparable_name"], r["comparable_unit"])
            for r in db.fetchall(
                "SELECT comparable_name, comparable_unit FROM fact_items"
            )
        }
        assert rows == {("herbs", "kg"), ("herb", "pcs")}

    def test_apply_rebuilds_item_stars(self, db):
        _seed_fact_item(
            db, "a1", "ARTICHOKE", comparable_name="artichoke", comparable_unit="kg"
        )
        _seed_fact_item(
            db, "a2", "ARTICHOKES", comparable_name="artichokes", comparable_unit="kg"
        )
        cluster = MergeCluster(
            canonical="artichokes",
            comparable_unit="kg",
            members=[NameStat("artichokes", "kg", 1), NameStat("artichoke", "kg", 1)],
        )
        apply_name_merges(db, [cluster])
        stars = {
            r["comparable_name"]
            for r in db.fetchall("SELECT comparable_name FROM item_stars")
        }
        assert stars == {"artichokes"}

    def test_propose_then_apply_end_to_end(self, db, tmp_path: Path):
        _seed_fact_item(
            db, "a1", "ARTICHOKE", comparable_name="artichoke", comparable_unit="kg"
        )
        _seed_fact_item(
            db, "a2", "ARTICHOKES", comparable_name="artichokes", comparable_unit="kg"
        )
        _seed_fact_item(
            db, "a3", "ARTICHOKES", comparable_name="artichokes", comparable_unit="kg"
        )

        embed = _fake_embedder({"arti": ["artichoke", "artichokes"]})
        clusters = propose_name_merges(db, embed_fn=embed, threshold=0.9)
        path = tmp_path / "p.yaml"
        write_proposal_yaml(clusters, path, threshold=0.9, generated="2026-06-09")

        # Human approves.
        text = path.read_text(encoding="utf-8").replace(
            "approved: false\n", "approved: true\n", 1
        )
        path.write_text(text, encoding="utf-8")

        approved = load_approved_clusters(path)
        result = apply_name_merges(db, approved)
        assert result.rewritten_rows == 1
        names = {
            r["comparable_name"]
            for r in db.fetchall("SELECT comparable_name FROM fact_items")
        }
        assert names == {"artichokes"}

    def test_apply_is_idempotent(self, db):
        _seed_fact_item(
            db, "a1", "ARTICHOKE", comparable_name="artichoke", comparable_unit="kg"
        )
        _seed_fact_item(
            db, "a2", "ARTICHOKES", comparable_name="artichokes", comparable_unit="kg"
        )
        cluster = MergeCluster(
            canonical="artichokes",
            comparable_unit="kg",
            members=[NameStat("artichokes", "kg", 1), NameStat("artichoke", "kg", 1)],
        )
        apply_name_merges(db, [cluster])
        second = apply_name_merges(db, [cluster])
        # Nothing left matching "artichoke" -> no rows rewritten the second time.
        assert second.rewritten_rows == 0
