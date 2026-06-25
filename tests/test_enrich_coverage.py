"""Tests for the read-only enrichment coverage report."""

from __future__ import annotations

import os

os.environ["ALIBI_TESTING"] = "1"

from alibi.enrichment.coverage import coverage_report


def _seed(db, item_id: str, name: str = "ITEM", **cols) -> None:
    """Insert a fact_item with its supporting chain (mirrors other enrich tests)."""
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
        "VALUES (?, ?, 'purchase', 'Store', 10.0, 'EUR', '2026-01-01')",
        (fact_id, cloud_id),
    )
    keys = ["id", "fact_id", "atom_id", "name"]
    vals = [item_id, fact_id, atom_id, name]
    for k, v in cols.items():
        keys.append(k)
        vals.append(v)
    ph = ", ".join("?" for _ in keys)
    conn.execute(
        f"INSERT OR IGNORE INTO fact_items ({', '.join(keys)}) VALUES ({ph})",  # noqa: S608
        tuple(vals),
    )
    conn.commit()


def _by_field(db):
    return {fc.field: fc for fc in coverage_report(db)}


class TestComparableNameCoverage:
    def test_filled_answered_pending(self, db):
        # filled: has a comparable_name
        _seed(db, "f", comparable_name="milk")
        # answered-null: marked but no value (a non-product line the model nulled)
        _seed(db, "a", comparable_name=None, comparable_name_enriched=1)
        # pending: empty and unmarked
        _seed(db, "p", comparable_name=None)
        cn = _by_field(db)["comparable_name"]
        assert cn.filled == 1
        assert cn.answered_null == 1
        assert cn.pending == 1
        assert cn.eligible == 3
        assert cn.stragglers == ["ITEM"]  # the one pending row's name


class TestUnitCoverage:
    def test_filled_answered_pending(self, db):
        _seed(db, "f", unit_quantity=450.0)
        _seed(db, "a", unit_quantity=None, unit_enriched=1)  # answered no-size
        _seed(db, "p", unit_quantity=None)  # pending
        u = _by_field(db)["unit_quantity"]
        assert (u.filled, u.answered_null, u.pending) == (1, 1, 1)


class TestCategoryCoverage:
    def test_version_gated(self, db):
        _seed(db, "f", category_path="food > dairy", category_taxonomy_version=1)
        # answered-null: categorised under current version but no path resolved
        _seed(db, "a", category_path=None, category_taxonomy_version=1)
        # pending: never categorised
        _seed(db, "p", category_path=None)
        c = _by_field(db)["category"]
        assert (c.filled, c.answered_null, c.pending) == (1, 1, 1)


class TestAttributesCoverage:
    def test_brace_sentinel_is_answered_null(self, db):
        _seed(db, "f", attributes='{"organic": true}')  # filled
        _seed(db, "a", attributes="{}")  # answered-null (no facet)
        _seed(db, "p", attributes=None)  # pending
        a = _by_field(db)["attributes"]
        assert (a.filled, a.answered_null, a.pending) == (1, 1, 1)


class TestStateCoverage:
    def test_scoped_to_real_products(self, db):
        # state is only counted for real products (non-empty comparable_name)
        _seed(db, "f", comparable_name="salmon", attributes='{"state": "fresh"}')
        _seed(
            db,
            "a",
            comparable_name="sugar",
            attributes="{}",
            state_enriched=1,
        )  # answered-null
        _seed(db, "p", comparable_name="tuna")  # pending
        _seed(db, "nonprod", comparable_name=None)  # not a real product -> excluded
        s = _by_field(db)["state"]
        assert s.eligible == 3  # the non-product row is excluded
        assert (s.filled, s.answered_null, s.pending) == (1, 1, 1)


class TestStragglerLimit:
    def test_limit_caps_names(self, db):
        for i in range(5):
            _seed(db, f"p{i}", name=f"PENDING {i}")
        cn = coverage_report(db, straggler_limit=2)
        pending_field = next(fc for fc in cn if fc.field == "comparable_name")
        assert pending_field.pending == 5
        assert len(pending_field.stragglers) == 2


class TestEmptyDb:
    def test_zero_counts(self, db):
        report = coverage_report(db)
        assert {fc.field for fc in report} == {
            "comparable_name",
            "unit_quantity",
            "category",
            "attributes",
            "state",
        }
        for fc in report:
            assert (fc.filled, fc.answered_null, fc.pending, fc.eligible) == (
                0,
                0,
                0,
                0,
            )
            assert fc.stragglers == []


def _seed_fact(db, fid, total, prices, fact_type="purchase"):
    """Seed a fact with `total` and one item per entry in `prices` (None = no items)."""
    conn = db.get_connection()
    cloud_id = f"cloud-{fid}"
    doc_id = f"doc-{fid}"
    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) VALUES (?,?,?)",
        (doc_id, f"/tmp/{doc_id}.jpg", f"hash-{doc_id}"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')", (cloud_id,)
    )
    conn.execute(
        "INSERT OR IGNORE INTO facts "
        "(id, cloud_id, fact_type, vendor, total_amount, currency, event_date) "
        "VALUES (?,?,?,?,?,'EUR','2026-01-01')",
        (fid, cloud_id, fact_type, "Store", total),
    )
    for i, p in enumerate(prices or []):
        aid = f"atom-{fid}-{i}"
        conn.execute(
            "INSERT OR IGNORE INTO atoms (id, document_id, atom_type, data) "
            "VALUES (?,?, 'item', '{}')",
            (aid, doc_id),
        )
        conn.execute(
            "INSERT OR IGNORE INTO fact_items (id, fact_id, atom_id, name, total_price) "
            "VALUES (?,?,?,?,?)",
            (f"it-{fid}-{i}", fid, aid, f"ITEM{i}", p),
        )
    conn.commit()


class TestItemCoverageReport:
    def test_full_partial_and_itemless(self, db):
        from alibi.enrichment.coverage import item_coverage_report

        _seed_fact(db, "full", 10.0, [6.0, 4.0])  # 100% -> OK
        _seed_fact(db, "partial", 100.0, [50.0])  # 50% -> below, partial
        _seed_fact(db, "itemless", 30.0, [])  # 0% -> below, no_items
        rep = item_coverage_report(db, threshold_pct=92.0)
        assert rep.eligible == 3
        assert rep.below == 2
        assert rep.partial == 1
        assert rep.no_items == 1
        # worst-first: the 0% item-less fact precedes the 50% partial
        assert [w.fact_id for w in rep.worst] == ["itemless", "partial"]
        assert rep.worst[1].coverage_pct == 50.0

    def test_threshold_boundary_and_positive_total_only(self, db):
        from alibi.enrichment.coverage import item_coverage_report

        _seed_fact(db, "ok93", 100.0, [93.0])  # 93% -> OK at 92 threshold
        _seed_fact(db, "zerototal", 0.0, [5.0])  # total 0 -> excluded
        rep = item_coverage_report(db, threshold_pct=92.0)
        assert rep.eligible == 1  # zero-total fact excluded
        assert rep.below == 0
