"""Tests for nutritional analytics — alibi/analytics/nutrition.py."""

from __future__ import annotations

import json
import os

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.analytics.nutrition import (
    item_nutrition,
    nutrition_by_category,
    nutrition_summary,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

NUTELLA_BARCODE = "3017624010701"
UNKNOWN_BARCODE = "9999999999999"

FULL_PRODUCT = {
    "product_name": "Nutella",
    "brands": "Ferrero",
    "categories": "Breakfasts, Spreads, Hazelnut spreads",
    "categories_tags": ["en:breakfasts", "en:spreads", "en:hazelnut-spreads"],
    "quantity": "400 g",
    "nutriscore_grade": "e",
    "nutriments": {
        "energy-kcal_100g": 539.0,
        "fat_100g": 30.9,
        "saturated-fat_100g": 10.6,
        "carbohydrates_100g": 57.5,
        "sugars_100g": 56.3,
        "fiber_100g": 0.0,
        "proteins_100g": 6.3,
        "salt_100g": 0.107,
    },
}

MINIMAL_PRODUCT = {
    "product_name": "Plain Biscuits",
    "brands": "Generic",
}

NEGATIVE_CACHE = {"_not_found": True}


def _seed_chain(
    db,
    *,
    doc_id: str = "doc-1",
    atom_id: str = "atom-1",
    cloud_id: str = "cloud-1",
    fact_id: str = "fact-1",
    item_id: str = "item-1",
    barcode: str | None = NUTELLA_BARCODE,
    category: str | None = None,
    quantity: float | None = None,
    unit_quantity: float | None = None,
    event_date: str = "2026-01-15",
) -> None:
    """Insert the minimal document -> atom -> cloud -> fact -> fact_item chain."""
    conn = db.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
        (doc_id, f"/tmp/{doc_id}.jpg", f"hash-{doc_id}"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO atoms (id, document_id, atom_type, data) VALUES (?, ?, 'item', '{}')",
        (atom_id, doc_id),
    )
    conn.execute(
        "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')",
        (cloud_id,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO facts "
        "(id, cloud_id, fact_type, vendor, total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', 'TestShop', 5.0, 'EUR', ?)",
        (fact_id, cloud_id, event_date),
    )
    conn.execute(
        "INSERT OR IGNORE INTO fact_items "
        "(id, fact_id, atom_id, name, barcode, category, quantity, unit_quantity) "
        "VALUES (?, ?, ?, 'Test Item', ?, ?, ?, ?)",
        (item_id, fact_id, atom_id, barcode, category, quantity, unit_quantity),
    )
    conn.commit()


def _store_product_cache(db, barcode: str, product: dict) -> None:
    """Insert a product_cache row."""
    db.execute(
        "INSERT OR REPLACE INTO product_cache (barcode, data, source) VALUES (?, ?, 'openfoodfacts')",
        (barcode, json.dumps(product)),
    )
    db.get_connection().commit()


# ===========================================================================
# TestNutritionSummaryEmptyDb
# ===========================================================================


class TestNutritionSummaryEmptyDb:
    """Empty DB should return empty periods list."""

    def test_returns_empty_periods(self, db) -> None:
        result = nutrition_summary(db)
        assert result["periods"] == []
        assert result["top_sugar_items"] == []
        assert result["top_calorie_items"] == []

    def test_with_date_filters_still_empty(self, db) -> None:
        result = nutrition_summary(db, start_date="2026-01-01", end_date="2026-01-31")
        assert result["periods"] == []

    def test_returns_required_keys(self, db) -> None:
        result = nutrition_summary(db)
        assert "periods" in result
        assert "top_sugar_items" in result
        assert "top_calorie_items" in result


# ===========================================================================
# TestNutritionSummaryWithCachedProduct
# ===========================================================================


class TestNutritionSummaryWithCachedProduct:
    """Verify aggregation when product cache has full nutriment data."""

    def test_single_item_with_unit_quantity(self, db) -> None:
        """Item with unit_quantity should produce non-zero totals."""
        _seed_chain(
            db,
            item_id="item-1",
            barcode=NUTELLA_BARCODE,
            quantity=1.0,
            unit_quantity=400.0,
            event_date="2026-01-15",
        )
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = nutrition_summary(db, period="month")
        periods = result["periods"]

        assert len(periods) == 1
        p = periods[0]
        assert p["period"] == "2026-01"
        assert p["items_with_nutrition"] == 1
        assert p["items_total"] >= 1
        assert p["coverage_pct"] > 0.0

        totals = p["totals"]
        # energy for 400g Nutella: 539 * 400/100 = 2156 kcal
        assert abs(totals["energy_kcal"] - 2156.0) < 1.0
        assert abs(totals["sugars_g"] - (56.3 * 4.0)) < 0.1

    def test_nutriscore_distribution_populated(self, db) -> None:
        _seed_chain(
            db,
            item_id="item-1",
            barcode=NUTELLA_BARCODE,
            quantity=1.0,
            unit_quantity=400.0,
            event_date="2026-02-01",
        )
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = nutrition_summary(db, period="month")
        p = result["periods"][0]
        assert p["nutriscore_distribution"].get("e", 0) == 1

    def test_items_without_unit_quantity_counted_but_no_totals(self, db) -> None:
        """Items counted as having nutrition even without unit_quantity."""
        _seed_chain(
            db,
            item_id="item-nw",
            barcode=NUTELLA_BARCODE,
            quantity=None,
            unit_quantity=None,
            event_date="2026-01-10",
        )
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = nutrition_summary(db, period="month")
        p = result["periods"][0]
        assert p["items_with_nutrition"] == 1
        # Totals remain zero because no unit_quantity
        assert p["totals"]["energy_kcal"] == 0.0

    def test_multiple_periods(self, db) -> None:
        for i, month in enumerate(["2026-01-01", "2026-02-01", "2026-03-01"], start=1):
            _seed_chain(
                db,
                doc_id=f"doc-{i}",
                atom_id=f"atom-{i}",
                cloud_id=f"cloud-{i}",
                fact_id=f"fact-{i}",
                item_id=f"item-{i}",
                barcode=NUTELLA_BARCODE,
                quantity=1.0,
                unit_quantity=400.0,
                event_date=month,
            )
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = nutrition_summary(db, period="month")
        assert len(result["periods"]) == 3

    def test_date_filter_start(self, db) -> None:
        _seed_chain(
            db,
            doc_id="doc-jan",
            atom_id="atom-jan",
            cloud_id="cloud-jan",
            fact_id="fact-jan",
            item_id="item-jan",
            barcode=NUTELLA_BARCODE,
            event_date="2026-01-01",
        )
        _seed_chain(
            db,
            doc_id="doc-feb",
            atom_id="atom-feb",
            cloud_id="cloud-feb",
            fact_id="fact-feb",
            item_id="item-feb",
            barcode=NUTELLA_BARCODE,
            event_date="2026-02-01",
        )
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = nutrition_summary(db, start_date="2026-02-01", period="month")
        periods = result["periods"]
        assert all(p["period"] >= "2026-02" for p in periods)

    def test_top_calorie_items_populated(self, db) -> None:
        _seed_chain(
            db,
            item_id="item-cal",
            barcode=NUTELLA_BARCODE,
            quantity=1.0,
            unit_quantity=400.0,
            event_date="2026-01-01",
        )
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = nutrition_summary(db, period="month")
        assert len(result["top_calorie_items"]) >= 1
        assert result["top_calorie_items"][0]["energy_kcal"] > 0

    def test_top_sugar_items_sorted_descending(self, db) -> None:
        # Seed two items with different unit quantities -> different sugar totals
        for i, uq in enumerate([400.0, 200.0], start=1):
            _seed_chain(
                db,
                doc_id=f"doc-s{i}",
                atom_id=f"atom-s{i}",
                cloud_id=f"cloud-s{i}",
                fact_id=f"fact-s{i}",
                item_id=f"item-s{i}",
                barcode=NUTELLA_BARCODE,
                quantity=1.0,
                unit_quantity=uq,
                event_date="2026-01-01",
            )
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = nutrition_summary(db)
        sugar_items = result["top_sugar_items"]
        for a, b in zip(sugar_items, sugar_items[1:]):
            assert a["sugars_g"] >= b["sugars_g"]

    def test_period_day_grouping(self, db) -> None:
        for i, d in enumerate(["2026-01-01", "2026-01-02"], start=1):
            _seed_chain(
                db,
                doc_id=f"doc-d{i}",
                atom_id=f"atom-d{i}",
                cloud_id=f"cloud-d{i}",
                fact_id=f"fact-d{i}",
                item_id=f"item-d{i}",
                barcode=NUTELLA_BARCODE,
                event_date=d,
            )
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = nutrition_summary(db, period="day")
        period_keys = [p["period"] for p in result["periods"]]
        assert "2026-01-01" in period_keys
        assert "2026-01-02" in period_keys


# ===========================================================================
# TestItemNutrition
# ===========================================================================


class TestItemNutritionFound:
    """Single item lookup returns correct structure."""

    def test_found_returns_dict(self, db) -> None:
        _seed_chain(db, item_id="item-A", barcode=NUTELLA_BARCODE)
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = item_nutrition(db, "item-A")
        assert result is not None
        assert result["item_id"] == "item-A"
        assert result["barcode"] == NUTELLA_BARCODE

    def test_nutriments_per_100g_present(self, db) -> None:
        _seed_chain(db, item_id="item-B", barcode=NUTELLA_BARCODE)
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = item_nutrition(db, "item-B")
        assert result is not None
        nmap = result["nutriments_per_100g"]
        assert abs(nmap["energy_kcal"] - 539.0) < 0.01
        assert abs(nmap["sugars_g"] - 56.3) < 0.01
        assert abs(nmap["proteins_g"] - 6.3) < 0.01

    def test_totals_computed_when_unit_quantity_present(self, db) -> None:
        _seed_chain(
            db,
            item_id="item-C",
            barcode=NUTELLA_BARCODE,
            quantity=1.0,
            unit_quantity=400.0,
        )
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = item_nutrition(db, "item-C")
        assert result is not None
        totals = result["totals"]
        assert totals is not None
        assert abs(totals["energy_kcal"] - 2156.0) < 1.0

    def test_totals_none_when_no_unit_quantity(self, db) -> None:
        _seed_chain(
            db,
            item_id="item-D",
            barcode=NUTELLA_BARCODE,
            quantity=None,
            unit_quantity=None,
        )
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = item_nutrition(db, "item-D")
        assert result is not None
        assert result["totals"] is None

    def test_nutriscore_grade_returned(self, db) -> None:
        _seed_chain(db, item_id="item-E", barcode=NUTELLA_BARCODE)
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = item_nutrition(db, "item-E")
        assert result is not None
        assert result["nutriscore_grade"] == "e"

    def test_product_name_returned(self, db) -> None:
        _seed_chain(db, item_id="item-F", barcode=NUTELLA_BARCODE)
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = item_nutrition(db, "item-F")
        assert result is not None
        assert result["product_name"] == "Nutella"


class TestItemNutritionNotFound:
    """Various not-found scenarios."""

    def test_nonexistent_fact_item_returns_none(self, db) -> None:
        result = item_nutrition(db, "does-not-exist")
        assert result is None

    def test_no_barcode_returns_error(self, db) -> None:
        _seed_chain(db, item_id="item-nobar", barcode=None)

        result = item_nutrition(db, "item-nobar")
        assert result is not None
        assert result["error"] == "no_barcode"

    def test_barcode_not_in_cache_returns_error(self, db) -> None:
        _seed_chain(db, item_id="item-nocache", barcode=NUTELLA_BARCODE)
        # Do NOT store product cache entry

        result = item_nutrition(db, "item-nocache")
        assert result is not None
        assert result["error"] == "not_cached"

    def test_minimal_product_no_nutriments(self, db) -> None:
        """Products with no nutriments key still return a result (empty map)."""
        _seed_chain(db, item_id="item-min", barcode="1111111111111")
        _store_product_cache(db, "1111111111111", MINIMAL_PRODUCT)

        result = item_nutrition(db, "item-min")
        assert result is not None
        assert result.get("error") is None
        assert result["nutriments_per_100g"] == {}
        assert result["totals"] is None


# ===========================================================================
# TestNegativeCacheExcluded
# ===========================================================================


class TestNegativeCacheExcluded:
    """Items with _not_found cache entries must be excluded from aggregations."""

    def test_summary_excludes_negative_cache(self, db) -> None:
        _seed_chain(
            db,
            item_id="item-neg",
            barcode=UNKNOWN_BARCODE,
            event_date="2026-01-05",
        )
        _store_product_cache(db, UNKNOWN_BARCODE, NEGATIVE_CACHE)

        result = nutrition_summary(db, period="month")
        # Period may appear with items_total > 0 but items_with_nutrition == 0
        for p in result["periods"]:
            assert p["items_with_nutrition"] == 0

    def test_item_nutrition_negative_cache_returns_error(self, db) -> None:
        _seed_chain(db, item_id="item-nf", barcode=UNKNOWN_BARCODE)
        _store_product_cache(db, UNKNOWN_BARCODE, NEGATIVE_CACHE)

        result = item_nutrition(db, "item-nf")
        assert result is not None
        assert result["error"] == "not_found_in_off"

    def test_by_category_excludes_negative_cache(self, db) -> None:
        _seed_chain(
            db,
            item_id="item-neg2",
            barcode=UNKNOWN_BARCODE,
            event_date="2026-01-06",
        )
        _store_product_cache(db, UNKNOWN_BARCODE, NEGATIVE_CACHE)

        result = nutrition_by_category(db)
        assert result == []


# ===========================================================================
# TestNutritionByCategory
# ===========================================================================


class TestNutritionByCategory:
    """Aggregation by product category."""

    def test_empty_returns_empty_list(self, db) -> None:
        result = nutrition_by_category(db)
        assert result == []

    def test_single_category(self, db) -> None:
        _seed_chain(
            db,
            item_id="item-cat1",
            barcode=NUTELLA_BARCODE,
            category="Spreads",
            quantity=1.0,
            unit_quantity=400.0,
        )
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = nutrition_by_category(db)
        assert len(result) == 1
        assert result[0]["category"] == "Spreads"
        assert result[0]["item_count"] == 1
        assert result[0]["items_with_totals"] == 1

    def test_multiple_categories_sorted_by_count(self, db) -> None:
        # 2 items in "Spreads", 1 item in "Biscuits"
        for i in range(2):
            _seed_chain(
                db,
                doc_id=f"doc-sp{i}",
                atom_id=f"atom-sp{i}",
                cloud_id=f"cloud-sp{i}",
                fact_id=f"fact-sp{i}",
                item_id=f"item-sp{i}",
                barcode=NUTELLA_BARCODE,
                category="Spreads",
            )

        _seed_chain(
            db,
            doc_id="doc-bis",
            atom_id="atom-bis",
            cloud_id="cloud-bis",
            fact_id="fact-bis",
            item_id="item-bis",
            barcode="1111111111111",
            category="Biscuits",
        )
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)
        _store_product_cache(db, "1111111111111", MINIMAL_PRODUCT)

        result = nutrition_by_category(db)
        assert result[0]["category"] == "Spreads"
        assert result[0]["item_count"] == 2
        assert result[1]["category"] == "Biscuits"

    def test_nutriscore_distribution_in_category(self, db) -> None:
        _seed_chain(
            db,
            item_id="item-ns",
            barcode=NUTELLA_BARCODE,
            category="Spreads",
        )
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = nutrition_by_category(db)
        assert result[0]["nutriscore_distribution"].get("e", 0) == 1

    def test_date_filter_applied(self, db) -> None:
        _seed_chain(
            db,
            doc_id="doc-jan",
            atom_id="atom-jan",
            cloud_id="cloud-jan",
            fact_id="fact-jan",
            item_id="item-jan",
            barcode=NUTELLA_BARCODE,
            category="Spreads",
            event_date="2026-01-01",
        )
        _seed_chain(
            db,
            doc_id="doc-mar",
            atom_id="atom-mar",
            cloud_id="cloud-mar",
            fact_id="fact-mar",
            item_id="item-mar",
            barcode=NUTELLA_BARCODE,
            category="Spreads",
            event_date="2026-03-01",
        )
        _store_product_cache(db, NUTELLA_BARCODE, FULL_PRODUCT)

        result = nutrition_by_category(db, start_date="2026-02-01")
        assert len(result) == 1
        assert result[0]["item_count"] == 1
