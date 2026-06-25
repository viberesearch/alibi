"""Tests for the materialised item_stars analytics surface.

Covers the sync hooks (store_fact refresh, per-fact/document refresh, full
rebuild) and the read aggregations (list, avg comparable price, price trend,
basket composition) in :mod:`alibi.services.item_stars`, plus the
``/api/v1/item-stars`` endpoints.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from alibi.api.app import create_app
from alibi.api.deps import get_database
from alibi.db import v2_store
from alibi.db.connection import DatabaseManager
from alibi.db.models import (
    CloudStatus,
    Fact,
    FactItem,
    FactStatus,
    FactType,
    UnitType,
)
from alibi.services import item_stars as svc

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _make_cloud(db: DatabaseManager) -> str:
    cloud_id = str(uuid.uuid4())
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO clouds (id, status, confidence) VALUES (?, ?, ?)",
            (cloud_id, CloudStatus.COLLAPSED.value, 1.0),
        )
    return cloud_id


def _make_doc_and_atom(db: DatabaseManager) -> tuple[str, str]:
    doc_id = str(uuid.uuid4())
    atom_id = str(uuid.uuid4())
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
            (doc_id, f"/tmp/test-{doc_id[:8]}.jpg", doc_id),
        )
        cursor.execute(
            "INSERT INTO atoms (id, document_id, atom_type, data) VALUES (?, ?, ?, ?)",
            (atom_id, doc_id, "item", "{}"),
        )
    return doc_id, atom_id


def _store_fact_with_items(
    db: DatabaseManager,
    *,
    vendor: str = "LIDL",
    vendor_key: str = "vk-1",
    country: str = "CY",
    currency: str = "EUR",
    event_date: date = date(2026, 1, 15),
    items: list[dict] | None = None,
) -> tuple[str, str]:
    """Create cloud+doc+atom+fact with items via store_fact. Returns (fact_id, doc_id)."""
    cloud_id = _make_cloud(db)
    doc_id, atom_id = _make_doc_and_atom(db)
    # Link the document's atom into a bundle under the cloud so the
    # document -> fact join (used by refresh_item_stars_for_document) resolves.
    bundle_id = str(uuid.uuid4())
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT INTO bundles (id, document_id, bundle_type, cloud_id) "
            "VALUES (?, ?, 'basket', ?)",
            (bundle_id, doc_id, cloud_id),
        )
        cursor.execute(
            "INSERT INTO bundle_atoms (bundle_id, atom_id, role) "
            "VALUES (?, ?, 'basket_item')",
            (bundle_id, atom_id),
        )
        cursor.execute(
            "INSERT INTO cloud_bundles (cloud_id, bundle_id, match_type) "
            "VALUES (?, ?, 'manual')",
            (cloud_id, bundle_id),
        )
    fact_id = str(uuid.uuid4())
    fact = Fact(
        id=fact_id,
        cloud_id=cloud_id,
        fact_type=FactType.PURCHASE,
        vendor=vendor,
        vendor_key=vendor_key,
        total_amount=Decimal("10.00"),
        currency=currency,
        country=country,
        event_date=event_date,
        status=FactStatus.CONFIRMED,
    )
    fact_items = []
    for spec in items or []:
        fact_items.append(
            FactItem(
                id=str(uuid.uuid4()),
                fact_id=fact_id,
                atom_id=atom_id,
                name=spec["name"],
                comparable_name=spec.get("comparable_name"),
                quantity=Decimal(str(spec.get("quantity", 1))),
                unit=UnitType.PIECE,
                total_price=(
                    Decimal(str(spec["total_price"]))
                    if spec.get("total_price") is not None
                    else None
                ),
                category=spec.get("category"),
                comparable_unit_price=(
                    Decimal(str(spec["comparable_unit_price"]))
                    if spec.get("comparable_unit_price") is not None
                    else None
                ),
                comparable_unit=spec.get("comparable_unit"),
                product_variant=spec.get("product_variant"),
            )
        )
    v2_store.store_fact(db, fact, fact_items)

    # store_fact does NOT persist comparable_name / category_path -- those are
    # written later by the enrichment passes. Simulate that enrichment write
    # (straight to fact_items) plus the item_stars refresh that follows it, so
    # the mirror reflects the post-enrichment state the analytics rely on.
    enrich = [
        (item, spec)
        for item, spec in zip(fact_items, items or [])
        if spec.get("comparable_name")
        or spec.get("category_path")
        or spec.get("attributes") is not None
    ]
    if enrich:
        import json as _json

        for item, spec in enrich:
            attrs = spec.get("attributes")
            db.execute(
                "UPDATE fact_items SET comparable_name = ?, category_path = ?, "
                "attributes = ? WHERE id = ?",
                (
                    spec.get("comparable_name"),
                    spec.get("category_path"),
                    _json.dumps(attrs) if attrs is not None else None,
                    item.id,
                ),
            )
        db.get_connection().commit()
        svc.refresh_item_stars_for_fact(db, fact_id)
    return fact_id, doc_id


def _count_stars(db: DatabaseManager, fact_id: str | None = None) -> int:
    if fact_id:
        row = db.fetchone(
            "SELECT COUNT(*) AS n FROM item_stars WHERE fact_id = ?", (fact_id,)
        )
    else:
        row = db.fetchone("SELECT COUNT(*) AS n FROM item_stars")
    return int(row["n"])


# ---------------------------------------------------------------------------
# Sync hooks
# ---------------------------------------------------------------------------


class TestStoreRefresh:
    def test_store_fact_populates_item_stars(self, db: DatabaseManager) -> None:
        fact_id, _ = _store_fact_with_items(
            db,
            items=[
                {
                    "name": "Milk 1L",
                    "comparable_name": "milk",
                    "total_price": 1.20,
                    "comparable_unit_price": 1.20,
                    "comparable_unit": UnitType.LITER,
                },
                {"name": "Bread", "comparable_name": "bread", "total_price": 2.0},
            ],
        )
        assert _count_stars(db, fact_id) == 2
        row = db.fetchone("SELECT * FROM item_stars WHERE comparable_name = 'milk'")
        # Parent fact axes are denormalised onto the star.
        assert row["vendor"] == "LIDL"
        assert row["country"] == "CY"
        assert row["currency"] == "EUR"
        assert float(row["comparable_unit_price"]) == 1.20
        assert row["comparable_unit"] == "l"

    def test_store_fact_no_items_no_stars(self, db: DatabaseManager) -> None:
        fact_id, _ = _store_fact_with_items(db, items=[])
        assert _count_stars(db, fact_id) == 0


class TestRefreshHooks:
    def test_refresh_for_fact_picks_up_direct_edit(self, db: DatabaseManager) -> None:
        fact_id, _ = _store_fact_with_items(
            db,
            items=[{"name": "Milk", "comparable_name": "milk", "total_price": 1.0}],
        )
        # Simulate an enrichment write straight to fact_items (bypasses store).
        db.execute(
            "UPDATE fact_items SET category_path = ?, comparable_unit_price = ? "
            "WHERE fact_id = ?",
            ("food > dairy > milk", 1.5, fact_id),
        )
        db.get_connection().commit()
        # Mirror is stale until refreshed.
        stale = db.fetchone(
            "SELECT category_path FROM item_stars WHERE fact_id = ?", (fact_id,)
        )
        assert stale["category_path"] is None

        svc.refresh_item_stars_for_fact(db, fact_id)
        fresh = db.fetchone(
            "SELECT category_path, comparable_unit_price FROM item_stars "
            "WHERE fact_id = ?",
            (fact_id,),
        )
        assert fresh["category_path"] == "food > dairy > milk"
        assert float(fresh["comparable_unit_price"]) == 1.5

    def test_refresh_for_document(self, db: DatabaseManager) -> None:
        fact_id, doc_id = _store_fact_with_items(
            db, items=[{"name": "Milk", "total_price": 1.0}]
        )
        db.execute("DELETE FROM item_stars WHERE fact_id = ?", (fact_id,))
        db.get_connection().commit()
        assert _count_stars(db, fact_id) == 0
        n = svc.refresh_item_stars_for_document(db, doc_id)
        assert n == 1
        assert _count_stars(db, fact_id) == 1

    def test_refresh_for_items(self, db: DatabaseManager) -> None:
        fact_id, _ = _store_fact_with_items(
            db, items=[{"name": "Milk", "total_price": 1.0}]
        )
        item_id = db.fetchone(
            "SELECT id FROM fact_items WHERE fact_id = ?", (fact_id,)
        )["id"]
        db.execute("DELETE FROM item_stars WHERE fact_id = ?", (fact_id,))
        db.get_connection().commit()
        n = svc.refresh_item_stars_for_items(db, [item_id])
        assert n == 1
        assert _count_stars(db, fact_id) == 1

    def test_delete_fact_item_cascades_to_star(self, db: DatabaseManager) -> None:
        fact_id, _ = _store_fact_with_items(
            db,
            items=[
                {"name": "Milk", "total_price": 1.0},
                {"name": "Bread", "total_price": 2.0},
            ],
        )
        assert _count_stars(db, fact_id) == 2
        # Deleting a fact_item auto-prunes its star (ON DELETE CASCADE) -- no
        # FK violation, no manual refresh needed.
        db.execute("DELETE FROM fact_items WHERE name = 'Bread'")
        db.get_connection().commit()
        assert _count_stars(db, fact_id) == 1

    def test_delete_fact_cascades_to_stars(self, db: DatabaseManager) -> None:
        fact_id, _ = _store_fact_with_items(
            db,
            items=[{"name": "Milk", "total_price": 1.0}, {"name": "Bread"}],
        )
        assert _count_stars(db, fact_id) == 2
        # delete_fact removes fact + fact_items; stars cascade away with them.
        v2_store.delete_fact(db, fact_id)
        assert _count_stars(db, fact_id) == 0

    def test_refresh_clears_removed_items(self, db: DatabaseManager) -> None:
        fact_id, _ = _store_fact_with_items(
            db,
            items=[
                {"name": "Milk", "total_price": 1.0},
                {"name": "Bread", "total_price": 2.0},
            ],
        )
        assert _count_stars(db, fact_id) == 2
        db.execute("DELETE FROM fact_items WHERE name = 'Bread'")
        db.get_connection().commit()
        svc.refresh_item_stars_for_fact(db, fact_id)
        assert _count_stars(db, fact_id) == 1


class TestRebuild:
    def test_rebuild_is_idempotent(self, db: DatabaseManager) -> None:
        _store_fact_with_items(
            db, items=[{"name": "A", "total_price": 1.0}, {"name": "B"}]
        )
        first = svc.rebuild_item_stars(db)
        second = svc.rebuild_item_stars(db)
        assert first == second == 2

    def test_rebuild_recovers_from_drift(self, db: DatabaseManager) -> None:
        fact_id, _ = _store_fact_with_items(
            db, items=[{"name": "Milk", "total_price": 1.0}]
        )
        # Corrupt the mirror, then rebuild reconciles it.
        db.execute("DELETE FROM item_stars")
        db.get_connection().commit()
        assert _count_stars(db) == 0
        count = svc.rebuild_item_stars(db)
        assert count == 1
        assert _count_stars(db, fact_id) == 1


# ---------------------------------------------------------------------------
# Read aggregations
# ---------------------------------------------------------------------------


class TestAggregations:
    def _seed_milk(self, db: DatabaseManager) -> None:
        # Two CY vendors selling milk at different EUR/L prices in Q1 2026.
        _store_fact_with_items(
            db,
            vendor="LIDL",
            vendor_key="vk-lidl",
            country="CY",
            event_date=date(2026, 1, 10),
            items=[
                {
                    "name": "Milk 1L",
                    "comparable_name": "milk",
                    "category": "Dairy",
                    "total_price": 1.2,
                    "comparable_unit_price": 1.20,
                    "comparable_unit": UnitType.LITER,
                }
            ],
        )
        _store_fact_with_items(
            db,
            vendor="ALPHAMEGA",
            vendor_key="vk-am",
            country="CY",
            event_date=date(2026, 2, 20),
            items=[
                {
                    "name": "Fresh Milk",
                    "comparable_name": "milk",
                    "category": "Dairy",
                    "total_price": 1.4,
                    "comparable_unit_price": 1.40,
                    "comparable_unit": UnitType.LITER,
                }
            ],
        )
        # A non-CY milk that must be excluded by the country filter.
        _store_fact_with_items(
            db,
            vendor="IGA",
            vendor_key="vk-iga",
            country="CA",
            event_date=date(2026, 1, 15),
            items=[
                {
                    "name": "Milk",
                    "comparable_name": "milk",
                    "category": "Dairy",
                    "total_price": 2.0,
                    "comparable_unit_price": 5.0,
                    "comparable_unit": UnitType.LITER,
                }
            ],
        )

    def test_avg_comparable_price_headline(self, db: DatabaseManager) -> None:
        """avg EUR/L of milk in CY in Q1 2026 across vendors — one call."""
        self._seed_milk(db)
        rows = svc.avg_comparable_price(
            db,
            filters={
                "comparable_name": "milk",
                "country": "CY",
                "date_from": "2026-01-01",
                "date_to": "2026-03-31",
            },
            group_by=["comparable_name"],
        )
        assert len(rows) == 1
        r = rows[0]
        assert r["comparable_name"] == "milk"
        assert r["comparable_unit"] == "l"
        assert r["avg_comparable_unit_price"] == pytest.approx(1.30)
        assert r["item_count"] == 2
        assert r["vendor_count"] == 2  # across vendors

    def test_avg_excludes_other_countries(self, db: DatabaseManager) -> None:
        self._seed_milk(db)
        rows = svc.avg_comparable_price(
            db, filters={"comparable_name": "milk"}, group_by=["country"]
        )
        by_country = {r["country"]: r for r in rows}
        assert by_country["CY"]["avg_comparable_unit_price"] == pytest.approx(1.30)
        assert by_country["CA"]["avg_comparable_unit_price"] == pytest.approx(5.0)

    def test_avg_unknown_group_dim_raises(self, db: DatabaseManager) -> None:
        with pytest.raises(ValueError):
            svc.avg_comparable_price(db, group_by=["not_a_dimension"])

    def test_no_cross_unit_blending(self, db: DatabaseManager) -> None:
        """EUR/L and EUR/piece for the same product must not be averaged."""
        _store_fact_with_items(
            db,
            items=[
                {
                    "name": "Milk 1L",
                    "comparable_name": "milk",
                    "total_price": 1.2,
                    "comparable_unit_price": 1.20,
                    "comparable_unit": UnitType.LITER,
                },
                {
                    "name": "Milk carton",
                    "comparable_name": "milk",
                    "total_price": 3.0,
                    "comparable_unit_price": 3.00,
                    "comparable_unit": UnitType.PIECE,
                },
            ],
        )
        rows = svc.avg_comparable_price(
            db, filters={"comparable_name": "milk"}, group_by=["comparable_name"]
        )
        # Two rows: one per comparable_unit, never a single blended 2.10 average.
        by_unit = {r["comparable_unit"]: r["avg_comparable_unit_price"] for r in rows}
        assert by_unit == {"l": pytest.approx(1.20), "pcs": pytest.approx(3.00)}

    def test_zero_and_null_price_excluded(self, db: DatabaseManager) -> None:
        _store_fact_with_items(
            db,
            items=[
                {
                    "name": "Eggs good",
                    "comparable_name": "eggs",
                    "total_price": 3.0,
                    "comparable_unit_price": 0.30,
                    "comparable_unit": UnitType.PIECE,
                },
                {
                    "name": "Eggs zero",
                    "comparable_name": "eggs",
                    "total_price": 3.0,
                    "comparable_unit_price": 0,
                    "comparable_unit": UnitType.PIECE,
                },
            ],
        )
        rows = svc.avg_comparable_price(
            db, filters={"comparable_name": "eggs"}, group_by=["comparable_name"]
        )
        assert len(rows) == 1
        # The 0-price row is excluded, so avg is 0.30 (not 0.15).
        assert rows[0]["avg_comparable_unit_price"] == pytest.approx(0.30)
        assert rows[0]["item_count"] == 1

    def test_filter_and_group_by_attribute_facets(self, db: DatabaseManager) -> None:
        """Filter by any facet (organic/size) and group by attr:<key>."""
        _store_fact_with_items(
            db,
            items=[
                {
                    "name": "Eggs L organic",
                    "comparable_name": "eggs",
                    "total_price": 3.6,
                    "comparable_unit_price": 0.30,
                    "comparable_unit": UnitType.PIECE,
                    "attributes": {"size": "l", "organic": True},
                },
                {
                    "name": "Eggs M",
                    "comparable_name": "eggs",
                    "total_price": 2.4,
                    "comparable_unit_price": 0.20,
                    "comparable_unit": UnitType.PIECE,
                    "attributes": {"size": "m", "organic": False},
                },
            ],
        )
        # Filter: organic only -> 1 star.
        organic = svc.list_item_stars(db, {"attributes": {"organic": True}})
        assert len(organic) == 1
        # Filter: facet present (size) -> both.
        sized = svc.list_item_stars(db, {"attributes": {"size": None}})
        assert len(sized) == 2
        # Group by attr:size -> separate L/M rows.
        rows = svc.avg_comparable_price(
            db, filters={"comparable_name": "eggs"}, group_by=["attr:size"]
        )
        by_size = {r["size"]: r["avg_comparable_unit_price"] for r in rows}
        assert by_size == {"l": pytest.approx(0.30), "m": pytest.approx(0.20)}

    def test_price_by_state(self, db: DatabaseManager) -> None:
        """Multi-state products surface per-state prices; single-state omitted."""
        _store_fact_with_items(
            db,
            items=[
                # salmon: TWO states within kg -> a real comparison, included.
                {
                    "name": "Fresh salmon",
                    "comparable_name": "salmon",
                    "total_price": 27.0,
                    "comparable_unit_price": 27.0,
                    "comparable_unit": UnitType.KILOGRAM,
                    "attributes": {"state": "fresh"},
                },
                {
                    "name": "Smoked salmon",
                    "comparable_name": "salmon",
                    "total_price": 58.0,
                    "comparable_unit_price": 58.0,
                    "comparable_unit": UnitType.KILOGRAM,
                    "attributes": {"state": "cured"},
                },
                # sugar: only ONE state -> nothing to compare, omitted.
                {
                    "name": "White sugar",
                    "comparable_name": "sugar",
                    "total_price": 1.0,
                    "comparable_unit_price": 1.0,
                    "comparable_unit": UnitType.KILOGRAM,
                    "attributes": {"state": "dried"},
                },
                # stateless item -> never considered.
                {
                    "name": "Mystery",
                    "comparable_name": "mystery",
                    "total_price": 5.0,
                    "comparable_unit_price": 5.0,
                    "comparable_unit": UnitType.KILOGRAM,
                    "attributes": {},
                },
            ],
        )
        rows = svc.price_by_state(db)
        names = {r["comparable_name"] for r in rows}
        assert names == {"salmon"}  # sugar (1 state) + mystery (no state) excluded
        by_state = {r["state"]: r["avg_comparable_unit_price"] for r in rows}
        assert by_state == {"fresh": pytest.approx(27.0), "cured": pytest.approx(58.0)}
        # ordered cheapest-first within the product
        assert [r["state"] for r in rows] == ["fresh", "cured"]

    def test_price_by_state_min_states_3(self, db: DatabaseManager) -> None:
        """min_states raises the bar: a 2-state product drops out at min_states=3."""
        _store_fact_with_items(
            db,
            items=[
                {
                    "name": "Fresh salmon",
                    "comparable_name": "salmon",
                    "total_price": 27.0,
                    "comparable_unit_price": 27.0,
                    "comparable_unit": UnitType.KILOGRAM,
                    "attributes": {"state": "fresh"},
                },
                {
                    "name": "Smoked salmon",
                    "comparable_name": "salmon",
                    "total_price": 58.0,
                    "comparable_unit_price": 58.0,
                    "comparable_unit": UnitType.KILOGRAM,
                    "attributes": {"state": "cured"},
                },
            ],
        )
        assert svc.price_by_state(db, min_states=3) == []
        assert len(svc.price_by_state(db, min_states=2)) == 2

    def test_price_by_state_units_not_blended(self, db: DatabaseManager) -> None:
        """A product split across units is two separate comparison groups, so a
        single state in each is NOT a multi-state comparison."""
        _store_fact_with_items(
            db,
            items=[
                {
                    "name": "Fresh thing",
                    "comparable_name": "thing",
                    "total_price": 2.0,
                    "comparable_unit_price": 2.0,
                    "comparable_unit": UnitType.KILOGRAM,
                    "attributes": {"state": "fresh"},
                },
                {
                    "name": "Canned thing",
                    "comparable_name": "thing",
                    "total_price": 3.0,
                    "comparable_unit_price": 3.0,
                    "comparable_unit": UnitType.LITER,
                    "attributes": {"state": "canned"},
                },
            ],
        )
        # fresh is the only kg state, canned the only l state -> neither group has
        # >= 2 states, so nothing is returned.
        assert svc.price_by_state(db) == []

    def test_list_attribute_facets(self, db: DatabaseManager) -> None:
        _store_fact_with_items(
            db,
            items=[
                {
                    "name": "Eggs L organic",
                    "comparable_name": "eggs",
                    "total_price": 3.6,
                    "attributes": {"size": "l", "organic": True},
                },
                {
                    "name": "Eggs L",
                    "comparable_name": "eggs",
                    "total_price": 3.0,
                    "attributes": {"size": "l", "organic": False},
                },
                {
                    "name": "Plain",
                    "comparable_name": "plain",
                    "total_price": 1.0,
                    "attributes": {},
                },
            ],
        )
        # min_count=2 (default) keeps size=l (2 items) ...
        facets = svc.list_attribute_facets(db)
        assert {f["value"]: f["item_count"] for f in facets.get("size", [])} == {"l": 2}
        # ... and drops the single-count organic values as noise.
        assert "organic" not in facets
        # min_count=1 surfaces them (JSON true/false as 1/0 via json_each).
        all_facets = svc.list_attribute_facets(db, min_count=1)
        organic = {f["value"]: f["item_count"] for f in all_facets.get("organic", [])}
        assert organic == {1: 1, 0: 1}

    def test_facets_drops_denylisted_keys(self, db: DatabaseManager) -> None:
        _store_fact_with_items(
            db,
            items=[
                {
                    "name": "X",
                    "comparable_name": "x",
                    "total_price": 1.0,
                    "attributes": {"organic": True, "vat_rate": 19, "code": "abc"},
                },
                {
                    "name": "Y",
                    "comparable_name": "y",
                    "total_price": 1.0,
                    "attributes": {"organic": True, "vat_rate": 19},
                },
            ],
        )
        facets = svc.list_attribute_facets(db, min_count=1)
        assert "organic" in facets
        assert "vat_rate" not in facets  # denylisted structural key
        assert "code" not in facets

    def test_invalid_attr_key_rejected(self, db: DatabaseManager) -> None:
        with pytest.raises(ValueError):
            svc.list_item_stars(db, {"attributes": {"bad key!": "x"}})
        with pytest.raises(ValueError):
            svc.avg_comparable_price(db, group_by=["attr:bad-key"])

    def test_group_by_product_variant(self, db: DatabaseManager) -> None:
        """Large vs medium eggs must compare separately, not blended."""
        _store_fact_with_items(
            db,
            items=[
                {
                    "name": "Eggs L",
                    "comparable_name": "eggs",
                    "total_price": 3.6,
                    "comparable_unit_price": 0.30,
                    "comparable_unit": UnitType.PIECE,
                    "product_variant": "L",
                },
                {
                    "name": "Eggs M",
                    "comparable_name": "eggs",
                    "total_price": 2.4,
                    "comparable_unit_price": 0.20,
                    "comparable_unit": UnitType.PIECE,
                    "product_variant": "M",
                },
            ],
        )
        rows = svc.avg_comparable_price(
            db,
            filters={"comparable_name": "eggs"},
            group_by=["comparable_name", "product_variant"],
        )
        by_variant = {
            r["product_variant"]: r["avg_comparable_unit_price"] for r in rows
        }
        assert by_variant == {"L": pytest.approx(0.30), "M": pytest.approx(0.20)}

    def test_period_grouping(self, db: DatabaseManager) -> None:
        self._seed_milk(db)
        rows = svc.avg_comparable_price(
            db,
            filters={"comparable_name": "milk", "country": "CY"},
            group_by=["month"],
        )
        months = {r["month"] for r in rows}
        assert months == {"2026-01", "2026-02"}

    def test_price_trend(self, db: DatabaseManager) -> None:
        self._seed_milk(db)
        rows = svc.price_trend(db, "milk", filters={"country": "CY"}, period="month")
        assert len(rows) == 2
        assert {r["period"] for r in rows} == {"2026-01", "2026-02"}
        assert all("vendor" in r for r in rows)

    def test_price_trend_unknown_period_raises(self, db: DatabaseManager) -> None:
        with pytest.raises(ValueError):
            svc.price_trend(db, "milk", period="decade")

    def test_basket_composition(self, db: DatabaseManager) -> None:
        self._seed_milk(db)
        rows = svc.basket_composition(db, filters={"country": "CY"}, by="category")
        by_cat = {r["category"]: r for r in rows}
        assert by_cat["Dairy"]["item_count"] == 2
        assert by_cat["Dairy"]["total_spent"] == pytest.approx(2.6)

    def test_basket_excludes_non_product_lines(self, db: DatabaseManager) -> None:
        """Tax / tip / "non_item" adjustment lines are not basket spend."""
        fact_id, _ = _store_fact_with_items(
            db,
            items=[
                {
                    "name": "Cheese",
                    "category": "Dairy",
                    "total_price": 5.0,
                    "category_path": "food > dairy > cheese",
                },
                # A tip line the taxonomy filed under the adjustment branch.
                {
                    "name": "TIP",
                    "category": "Tip",
                    "total_price": 2.0,
                    "category_path": "adjustment > tip",
                },
                # A broad-category Non_Item with no adjustment path.
                {"name": "Custom amount", "category": "Non_Item", "total_price": 9.0},
            ],
        )
        # category_path isn't persisted by store_fact, so set it like the pass.
        db.execute(
            "UPDATE fact_items SET category_path = 'adjustment > tip' WHERE name='TIP'"
        )
        db.execute(
            "UPDATE fact_items SET category = 'Tip', category_path='adjustment > tip' "
            "WHERE name='TIP'"
        )
        db.execute(
            "UPDATE fact_items SET category = 'Non_Item' WHERE name='Custom amount'"
        )
        db.execute(
            "UPDATE fact_items SET category='Dairy', category_path='food > dairy > cheese' "
            "WHERE name='Cheese'"
        )
        db.get_connection().commit()
        svc.refresh_item_stars_for_fact(db, fact_id)

        rows = svc.basket_composition(db, by="category")
        cats = {r["category"] for r in rows}
        assert "Dairy" in cats
        assert "Tip" not in cats  # adjustment branch excluded
        assert "Non_Item" not in cats  # broad non-product marker excluded
        dairy = next(r for r in rows if r["category"] == "Dairy")
        assert dairy["total_spent"] == pytest.approx(5.0)

    def test_list_item_stars_category_path_prefix(self, db: DatabaseManager) -> None:
        fact_id, _ = _store_fact_with_items(
            db, items=[{"name": "Milk", "total_price": 1.0}]
        )
        db.execute(
            "UPDATE fact_items SET category_path = ? WHERE fact_id = ?",
            ("food > dairy > milk", fact_id),
        )
        db.get_connection().commit()
        svc.refresh_item_stars_for_fact(db, fact_id)
        # Hierarchical prefix matches at and under "food".
        assert len(svc.list_item_stars(db, {"category_path": "food"})) == 1
        assert len(svc.list_item_stars(db, {"category_path": "food > dairy"})) == 1
        assert len(svc.list_item_stars(db, {"category_path": "household"})) == 0


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@pytest.fixture
def client(db_manager: DatabaseManager) -> Generator[TestClient, None, None]:
    app = create_app()

    def override_get_database() -> DatabaseManager:
        return db_manager

    app.dependency_overrides[get_database] = override_get_database
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


class TestApi:
    def test_avg_price_endpoint(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        _store_fact_with_items(
            db_manager,
            vendor="LIDL",
            country="CY",
            event_date=date(2026, 1, 10),
            items=[
                {
                    "name": "Milk",
                    "comparable_name": "milk",
                    "total_price": 1.2,
                    "comparable_unit_price": 1.2,
                    "comparable_unit": UnitType.LITER,
                }
            ],
        )
        resp = client.get(
            "/api/v1/item-stars/avg-price",
            params={
                "comparable_name": "milk",
                "country": "CY",
                "group_by": "comparable_name",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["comparable_name"] == "milk"

    def test_avg_price_bad_group_by(self, client: TestClient) -> None:
        resp = client.get("/api/v1/item-stars/avg-price", params={"group_by": "bogus"})
        assert resp.status_code == 400

    def test_price_by_state_endpoint(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        _store_fact_with_items(
            db_manager,
            items=[
                {
                    "name": "Fresh salmon",
                    "comparable_name": "salmon",
                    "total_price": 27.0,
                    "comparable_unit_price": 27.0,
                    "comparable_unit": UnitType.KILOGRAM,
                    "attributes": {"state": "fresh"},
                },
                {
                    "name": "Smoked salmon",
                    "comparable_name": "salmon",
                    "total_price": 58.0,
                    "comparable_unit_price": 58.0,
                    "comparable_unit": UnitType.KILOGRAM,
                    "attributes": {"state": "cured"},
                },
            ],
        )
        resp = client.get("/api/v1/item-stars/price-by-state")
        assert resp.status_code == 200
        data = resp.json()
        assert {r["state"] for r in data} == {"fresh", "cured"}
        # min_states below 2 is rejected by the query param bound.
        assert (
            client.get(
                "/api/v1/item-stars/price-by-state", params={"min_states": 1}
            ).status_code
            == 422
        )

    def test_rebuild_endpoint(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        _store_fact_with_items(db_manager, items=[{"name": "A", "total_price": 1.0}])
        db_manager.execute("DELETE FROM item_stars")
        db_manager.get_connection().commit()
        resp = client.post("/api/v1/item-stars/rebuild")
        assert resp.status_code == 200
        assert resp.json()["rebuilt"] == 1

    def test_list_and_basket_endpoints(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        _store_fact_with_items(
            db_manager,
            country="CY",
            items=[{"name": "Milk", "category": "Dairy", "total_price": 1.0}],
        )
        assert len(client.get("/api/v1/item-stars").json()) == 1
        basket = client.get("/api/v1/item-stars/basket").json()
        assert any(r["category"] == "Dairy" for r in basket)

    def test_facets_and_trend_endpoints(
        self, client: TestClient, db_manager: DatabaseManager
    ) -> None:
        _store_fact_with_items(
            db_manager,
            country="CY",
            items=[
                {
                    "name": "Milk",
                    "comparable_name": "milk",
                    "total_price": 1.2,
                    "comparable_unit_price": 1.2,
                    "comparable_unit": UnitType.LITER,
                }
            ],
        )
        facets = client.get("/api/v1/item-stars/facets")
        assert facets.status_code == 200
        assert isinstance(facets.json(), dict)
        trend = client.get(
            "/api/v1/item-stars/trend", params={"comparable_name": "milk"}
        )
        assert trend.status_code == 200

    def test_list_bad_attr_returns_400(self, client: TestClient) -> None:
        # An invalid facet key (space/semicolon) must be a 400, not a 500.
        resp = client.get("/api/v1/item-stars", params={"attr": "bad key:1"})
        assert resp.status_code == 400

    def test_facets_bad_attr_returns_400(self, client: TestClient) -> None:
        resp = client.get("/api/v1/item-stars/facets", params={"attr": "bad key:1"})
        assert resp.status_code == 400
