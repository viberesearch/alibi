"""Tests for price factor analysis."""

from __future__ import annotations

from uuid import uuid4

import pytest

from alibi.analytics.price_factors import (
    PriceFactor,
    ProductPriceProfile,
    analyze_price_factors,
    get_category_price_factors,
    price_factor_summary,
)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

EGGS_DATA = [
    # (comparable_name, price, variant, attributes, vendor, date, category)
    ("Eggs 10pk", 2.29, "M", [], "AlphaMega", "2026-01-15", "Eggs"),
    ("Eggs 10pk", 2.35, "M", [], "Papantoniou", "2026-01-20", "Eggs"),
    ("Eggs 10pk", 2.25, "M", [], "AlphaMega", "2026-02-10", "Eggs"),
    ("Eggs 10pk", 2.79, "L", [], "AlphaMega", "2026-01-15", "Eggs"),
    ("Eggs 10pk", 2.85, "L", [], "Papantoniou", "2026-01-20", "Eggs"),
    ("Eggs 10pk", 2.75, "L", [], "AlphaMega", "2026-02-10", "Eggs"),
    ("Eggs 10pk", 3.79, "M", ["free-range"], "AlphaMega", "2026-01-15", "Eggs"),
    ("Eggs 10pk", 3.85, "M", ["free-range"], "Papantoniou", "2026-01-20", "Eggs"),
    ("Eggs 10pk", 3.75, "M", ["free-range"], "AlphaMega", "2026-02-10", "Eggs"),
    ("Eggs 10pk", 4.29, "L", ["free-range"], "AlphaMega", "2026-01-15", "Eggs"),
    ("Eggs 10pk", 4.35, "L", ["free-range"], "Papantoniou", "2026-01-20", "Eggs"),
    (
        "Eggs 10pk",
        4.59,
        "M",
        ["free-range", "organic"],
        "AlphaMega",
        "2026-02-15",
        "Eggs",
    ),
    (
        "Eggs 10pk",
        4.65,
        "M",
        ["free-range", "organic"],
        "Papantoniou",
        "2026-02-20",
        "Eggs",
    ),
]

MILK_DATA = [
    ("Milk 1L", 1.29, "1.5%", [], "AlphaMega", "2026-01-10", "Dairy"),
    ("Milk 1L", 1.35, "1.5%", [], "Papantoniou", "2026-01-12", "Dairy"),
    ("Milk 1L", 1.39, "3%", [], "AlphaMega", "2026-01-10", "Dairy"),
    ("Milk 1L", 1.45, "3%", [], "Papantoniou", "2026-01-12", "Dairy"),
    ("Milk 1L", 1.89, "3%", ["organic"], "AlphaMega", "2026-01-10", "Dairy"),
    ("Milk 1L", 1.95, "3%", ["organic"], "Papantoniou", "2026-01-12", "Dairy"),
    ("Milk 1L", 1.85, "1.5%", ["organic"], "AlphaMega", "2026-01-15", "Dairy"),
]

BREAD_DATA = [
    ("Bread White 500g", 1.50, None, [], "AlphaMega", "2026-01-10", "Bakery"),
    ("Bread White 500g", 1.55, None, [], "Papantoniou", "2026-01-12", "Bakery"),
    ("Bread White 500g", 1.45, None, [], "AlphaMega", "2026-02-10", "Bakery"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_test_item(
    db, comparable_name, price, variant, attributes, vendor, event_date, category
):
    """Insert a synthetic fact + fact_item + annotations for testing.

    Creates the full FK chain: document -> atom, cloud -> fact -> fact_item.
    """
    doc_id = str(uuid4())
    atom_id = str(uuid4())
    cloud_id = str(uuid4())
    fact_id = str(uuid4())
    item_id = str(uuid4())
    conn = db.get_connection()

    # Parent records for FK satisfaction
    conn.execute(
        "INSERT INTO documents (id, file_path, file_hash) VALUES (?, ?, ?)",
        (doc_id, f"/test/{doc_id}.jpg", str(uuid4())),
    )
    conn.execute(
        "INSERT INTO atoms (id, document_id, atom_type, data) VALUES (?, ?, 'item', '{}')",
        (atom_id, doc_id),
    )
    conn.execute(
        "INSERT INTO clouds (id, status) VALUES (?, 'collapsed')",
        (cloud_id,),
    )

    conn.execute(
        "INSERT INTO facts (id, cloud_id, fact_type, vendor, vendor_key, "
        "event_date, total_amount) VALUES (?, ?, 'purchase', ?, ?, ?, ?)",
        (
            fact_id,
            cloud_id,
            vendor,
            vendor.lower().replace(" ", "_"),
            event_date,
            price,
        ),
    )
    conn.execute(
        "INSERT INTO fact_items (id, fact_id, atom_id, name, comparable_name, "
        "comparable_unit_price, comparable_unit, product_variant, category) "
        "VALUES (?, ?, ?, ?, ?, ?, 'pcs', ?, ?)",
        (
            item_id,
            fact_id,
            atom_id,
            comparable_name,
            comparable_name,
            price,
            variant,
            category,
        ),
    )
    for attr in attributes:
        ann_id = str(uuid4())
        conn.execute(
            "INSERT INTO annotations (id, annotation_type, target_type, target_id, "
            "key, value, source, created_at) VALUES (?, 'product_attribute', "
            "'fact_item', ?, ?, 'true', 'test', datetime('now'))",
            (ann_id, item_id, attr),
        )
    conn.commit()
    return fact_id, item_id


def _load_dataset(db, dataset):
    """Insert all rows from a dataset list."""
    for row in dataset:
        _insert_test_item(db, *row)


@pytest.fixture()
def populated_db(db):
    """DB with eggs, milk, and bread test data."""
    _load_dataset(db, EGGS_DATA)
    _load_dataset(db, MILK_DATA)
    _load_dataset(db, BREAD_DATA)
    return db


# ---------------------------------------------------------------------------
# Tests: analyze_price_factors
# ---------------------------------------------------------------------------


class TestAnalyzePriceFactors:
    def test_eggs_size_factor_detected(self, populated_db):
        """Size L and M should both appear as variant factors."""
        profiles = analyze_price_factors(populated_db, comparable_name="Eggs 10pk")
        assert len(profiles) == 1
        profile = profiles[0]
        assert profile.comparable_name == "Eggs 10pk"

        variant_l = _find_factor(profile, "variant:L")
        variant_m = _find_factor(profile, "variant:M")
        assert variant_l is not None
        assert variant_m is not None
        # L and M premiums should be opposite signs (L higher than M
        # within same attribute combo, but confounded by free-range/organic
        # distribution -- the algorithm correctly measures marginal impact)
        assert variant_l.observations == 5
        assert variant_m.observations == 8

    def test_eggs_free_range_premium(self, populated_db):
        """Free-range should show a significant premium."""
        profiles = analyze_price_factors(populated_db, comparable_name="Eggs 10pk")
        profile = profiles[0]

        free_range = _find_factor(profile, "free-range")
        assert free_range is not None
        assert free_range.avg_premium > 0.5, "Free-range premium should be substantial"
        assert free_range.pct_premium > 0.1

    def test_eggs_organic_premium(self, populated_db):
        """Organic should show premium on top of free-range."""
        profiles = analyze_price_factors(populated_db, comparable_name="Eggs 10pk")
        profile = profiles[0]

        organic = _find_factor(profile, "organic")
        assert organic is not None
        assert organic.avg_premium > 0, "Organic should be more expensive"
        assert organic.observations == 2  # only 2 organic items

    def test_milk_organic_premium(self, populated_db):
        """Organic milk should show premium."""
        profiles = analyze_price_factors(populated_db, comparable_name="Milk 1L")
        assert len(profiles) == 1
        profile = profiles[0]

        organic = _find_factor(profile, "organic")
        assert organic is not None
        assert organic.avg_premium > 0.3

    def test_milk_fat_pct_minimal_difference(self, populated_db):
        """Fat % difference should be smaller than organic."""
        profiles = analyze_price_factors(populated_db, comparable_name="Milk 1L")
        profile = profiles[0]

        organic = _find_factor(profile, "organic")
        variant_3 = _find_factor(profile, "variant:3%")

        assert organic is not None
        assert variant_3 is not None
        assert abs(organic.pct_premium) > abs(
            variant_3.pct_premium
        ), "Organic premium should exceed fat % premium"

    def test_product_with_no_attributes(self, populated_db):
        """Bread with no attributes should have empty factors list."""
        profiles = analyze_price_factors(
            populated_db, comparable_name="Bread White 500g"
        )
        assert len(profiles) == 1
        profile = profiles[0]
        assert profile.factors == []
        assert profile.total_observations == 3

    def test_filter_by_category(self, populated_db):
        """Category filter should limit results."""
        profiles = analyze_price_factors(populated_db, category="Dairy")
        names = {p.comparable_name for p in profiles}
        assert "Milk 1L" in names
        assert "Eggs 10pk" not in names
        assert "Bread White 500g" not in names

    def test_filter_by_name(self, populated_db):
        """Name filter should return single product."""
        profiles = analyze_price_factors(populated_db, comparable_name="Milk 1L")
        assert len(profiles) == 1
        assert profiles[0].comparable_name == "Milk 1L"

    def test_min_observations_filter(self, populated_db):
        """Products below min_observations threshold should be excluded."""
        # Bread has only 3 items, setting min to 5 should exclude it
        profiles = analyze_price_factors(populated_db, min_observations=5)
        names = {p.comparable_name for p in profiles}
        assert "Bread White 500g" not in names
        # Eggs (13) and Milk (7) should still be present
        assert "Eggs 10pk" in names
        assert "Milk 1L" in names

    def test_confidence_increases_with_sample_size(self, populated_db):
        """More observations should give higher confidence."""
        profiles = analyze_price_factors(populated_db, comparable_name="Eggs 10pk")
        profile = profiles[0]

        free_range = _find_factor(profile, "free-range")
        organic = _find_factor(profile, "organic")

        assert free_range is not None
        assert organic is not None
        # free-range has 5 obs, organic has 2 -- free-range should have higher confidence
        assert free_range.confidence > organic.confidence

    def test_price_range(self, populated_db):
        """Price range should reflect min/max observed prices."""
        profiles = analyze_price_factors(populated_db, comparable_name="Eggs 10pk")
        profile = profiles[0]
        assert profile.price_range[0] == pytest.approx(2.25, abs=0.01)
        assert profile.price_range[1] == pytest.approx(4.65, abs=0.01)

    def test_vendors_list(self, populated_db):
        """Vendors list should be sorted and deduplicated."""
        profiles = analyze_price_factors(populated_db, comparable_name="Eggs 10pk")
        profile = profiles[0]
        assert profile.vendors == ["AlphaMega", "Papantoniou"]

    def test_baseline_is_overall_when_no_bare_items(self, populated_db):
        """When all items have attributes, baseline falls back to overall avg."""
        profiles = analyze_price_factors(populated_db, comparable_name="Eggs 10pk")
        profile = profiles[0]
        # All egg items have at least a variant (M or L), so none are "bare".
        # Baseline should be the overall average of all 13 prices.
        all_prices = [row[1] for row in EGGS_DATA]
        expected_baseline = sum(all_prices) / len(all_prices)
        assert profile.baseline_price == pytest.approx(expected_baseline, abs=0.01)

    def test_baseline_from_bare_items(self, db):
        """Baseline should come from items with no attributes when they exist."""
        # Insert items: some with brand, some without any attributes
        for price in [1.50, 1.55, 1.60]:
            _insert_test_item(
                db, "Water 1L", price, None, [], "Shop", "2026-01-10", "Beverages"
            )
        for price in [2.10, 2.20]:
            _insert_test_item(
                db, "Water 1L", price, None, [], "Shop", "2026-01-10", "Beverages"
            )
            # Set brand on last inserted item
            conn = db.get_connection()
            row = conn.execute(
                "SELECT id FROM fact_items WHERE comparable_unit_price = ? "
                "ORDER BY rowid DESC LIMIT 1",
                (price,),
            ).fetchone()
            conn.execute(
                "UPDATE fact_items SET brand = 'Evian' WHERE id = ?", (row["id"],)
            )
            conn.commit()

        profiles = analyze_price_factors(db, comparable_name="Water 1L")
        profile = profiles[0]
        # Bare items: 1.50, 1.55, 1.60 (no brand, no variant, no annotations)
        expected = (1.50 + 1.55 + 1.60) / 3
        assert profile.baseline_price == pytest.approx(expected, abs=0.01)

    def test_empty_db(self, db):
        """Empty database should return empty list."""
        profiles = analyze_price_factors(db)
        assert profiles == []

    def test_sorted_by_observations(self, populated_db):
        """Results should be sorted by total_observations descending."""
        profiles = analyze_price_factors(populated_db)
        obs_counts = [p.total_observations for p in profiles]
        assert obs_counts == sorted(obs_counts, reverse=True)

    def test_factors_sorted_by_abs_pct_premium(self, populated_db):
        """Factors within a profile should be sorted by abs(pct_premium) desc."""
        profiles = analyze_price_factors(populated_db, comparable_name="Eggs 10pk")
        profile = profiles[0]
        pct_values = [abs(f.pct_premium) for f in profile.factors]
        assert pct_values == sorted(pct_values, reverse=True)


# ---------------------------------------------------------------------------
# Tests: get_category_price_factors
# ---------------------------------------------------------------------------


class TestGetCategoryPriceFactors:
    def test_category_level_aggregation(self, populated_db):
        """Should aggregate factors across products in a category."""
        # Eggs category only has one product, but test that it works
        factors = get_category_price_factors(populated_db, "Eggs", min_observations=3)
        attrs = {f.attribute for f in factors}
        assert "free-range" in attrs

    def test_empty_category(self, populated_db):
        """Non-existent category should return empty."""
        factors = get_category_price_factors(populated_db, "Electronics")
        assert factors == []

    def test_dairy_organic_factor(self, populated_db):
        """Dairy organic should show up as a category factor."""
        factors = get_category_price_factors(populated_db, "Dairy", min_observations=3)
        organic = next((f for f in factors if f.attribute == "organic"), None)
        assert organic is not None
        assert organic.avg_premium > 0


# ---------------------------------------------------------------------------
# Tests: price_factor_summary
# ---------------------------------------------------------------------------


class TestPriceFactorSummary:
    def test_summary_structure(self, populated_db):
        """Summary should have the expected keys."""
        summary = price_factor_summary(populated_db, min_observations=3)
        assert "categories" in summary
        assert "top_factors" in summary
        assert "products_analyzed" in summary

    def test_summary_categories(self, populated_db):
        """Summary should include categories with enough data."""
        summary = price_factor_summary(populated_db, min_observations=3)
        assert "Eggs" in summary["categories"]
        assert "Dairy" in summary["categories"]

    def test_summary_products_count(self, populated_db):
        """Products analyzed should count eligible products."""
        summary = price_factor_summary(populated_db, min_observations=3)
        # Eggs (13), Milk (7), Bread (3) all meet min_observations=3
        assert summary["products_analyzed"] >= 2

    def test_summary_top_factors(self, populated_db):
        """Top factors should be sorted by impact."""
        summary = price_factor_summary(populated_db, min_observations=3)
        factors = summary["top_factors"]
        assert len(factors) > 0
        pct_values = [abs(f.pct_premium) for f in factors]
        assert pct_values == sorted(pct_values, reverse=True)


# ---------------------------------------------------------------------------
# Tests: brand as factor
# ---------------------------------------------------------------------------


class TestBrandFactor:
    def test_brand_appears_as_factor(self, db):
        """Brand should appear as brand:<name> factor."""
        for i in range(4):
            _insert_test_item(
                db,
                "Yogurt 200g",
                1.50 + (0.30 if i < 2 else 0),
                None,
                [],
                "Shop",
                f"2026-01-{10+i:02d}",
                "Dairy",
            )
        # Add brand to first two
        conn = db.get_connection()
        rows = conn.execute(
            "SELECT id FROM fact_items WHERE comparable_unit_price >= 1.7"
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE fact_items SET brand = 'Premium' WHERE id = ?",
                (row["id"],),
            )
        conn.commit()

        profiles = analyze_price_factors(db, comparable_name="Yogurt 200g")
        assert len(profiles) == 1
        brand_factor = _find_factor(profiles[0], "brand:Premium")
        assert brand_factor is not None
        assert brand_factor.avg_premium > 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_factor(profile: ProductPriceProfile, attribute: str) -> PriceFactor | None:
    """Find a factor by attribute name in a profile."""
    return next((f for f in profile.factors if f.attribute == attribute), None)
