"""Tests for product resolution service — fuzzy name matching."""

from __future__ import annotations

import os

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.enrichment.product_resolver import (
    ProductMatch,
    _fts5_available,
    _normalize,
    rebuild_product_fts,
    resolve_product,
    resolve_products_batch,
)
from alibi.enrichment.service import (
    enrich_item_by_name,
    enrich_pending_by_name,
)


# ---------------------------------------------------------------------------
# Helpers — seed enriched items into DB
# ---------------------------------------------------------------------------


def _seed_enriched_item(
    db,
    item_id: str,
    name: str,
    brand: str | None = None,
    category: str | None = None,
    barcode: str | None = None,
    vendor_key: str | None = None,
    fact_id: str | None = None,
    doc_id: str | None = None,
    atom_id: str | None = None,
    cloud_id: str | None = None,
) -> None:
    """Insert a fact_item with supporting chain for resolver tests."""
    fact_id = fact_id or f"fact-{item_id}"
    doc_id = doc_id or f"doc-{item_id}"
    atom_id = atom_id or f"atom-{item_id}"
    cloud_id = cloud_id or f"cloud-{item_id}"

    conn = db.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) "
        "VALUES (?, ?, ?)",
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
        "(id, cloud_id, fact_type, vendor, vendor_key, "
        "total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', 'Test Vendor', ?, 10.0, 'EUR', '2026-01-01')",
        (fact_id, cloud_id, vendor_key),
    )
    conn.execute(
        "INSERT OR IGNORE INTO bundles (id, document_id, bundle_type, cloud_id) "
        "VALUES (?, ?, 'basket', ?)",
        (f"bundle-{item_id}", doc_id, cloud_id),
    )
    conn.execute(
        "INSERT OR IGNORE INTO bundle_atoms (bundle_id, atom_id, role) "
        "VALUES (?, ?, 'basket_item')",
        (f"bundle-{item_id}", atom_id),
    )
    conn.execute(
        "INSERT OR IGNORE INTO cloud_bundles (cloud_id, bundle_id, match_type) "
        "VALUES (?, ?, 'manual')",
        (cloud_id, f"bundle-{item_id}"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO fact_items "
        "(id, fact_id, atom_id, name, barcode, brand, category) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (item_id, fact_id, atom_id, name, barcode, brand, category),
    )
    conn.commit()


# ===========================================================================
# TestNormalize
# ===========================================================================


class TestNormalize:
    """Tests for _normalize() helper."""

    def test_lowercase(self):
        assert _normalize("NUTELLA") == "nutella"

    def test_strip_trailing_weight(self):
        assert _normalize("Nutella 400g") == "nutella"

    def test_strip_trailing_volume(self):
        assert _normalize("Milk 1l") == "milk"

    def test_strip_trailing_decimal_weight(self):
        assert _normalize("Salmon 0.339kg") == "salmon"

    def test_strip_trailing_ml(self):
        assert _normalize("Juice 500ml") == "juice"

    def test_strip_trailing_cl(self):
        assert _normalize("Wine 75cl") == "wine"

    def test_collapse_whitespace(self):
        assert _normalize("Greek  Strained   Yogurt") == "greek strained yogurt"

    def test_strip_leading_trailing_spaces(self):
        assert _normalize("  Bread  ") == "bread"

    def test_empty_string(self):
        assert _normalize("") == ""

    def test_only_unit(self):
        # "500g" alone — the regex requires a leading space before digits
        assert _normalize("500g") == "500g"

    def test_name_with_embedded_number(self):
        # "3.5% fat" should not be stripped (no unit suffix)
        assert _normalize("Milk 3.5% fat") == "milk 3.5% fat"


# ===========================================================================
# TestResolveProduct
# ===========================================================================


class TestResolveProduct:
    """Tests for resolve_product() — main resolution function."""

    def test_empty_db_returns_none(self, db):
        assert resolve_product(db, "Nutella") is None

    def test_empty_name_returns_none(self, db):
        assert resolve_product(db, "") is None
        assert resolve_product(db, "   ") is None

    def test_exact_match(self, db):
        _seed_enriched_item(
            db, "i1", "Nutella", brand="Ferrero", category="Hazelnut Spreads"
        )
        result = resolve_product(db, "Nutella")

        assert result is not None
        assert result.brand == "Ferrero"
        assert result.category == "Hazelnut Spreads"
        assert result.similarity == 1.0
        assert result.source == "exact_match"

    def test_exact_match_case_insensitive(self, db):
        _seed_enriched_item(
            db, "i1", "NUTELLA 400G", brand="Ferrero", category="Spreads"
        )
        result = resolve_product(db, "nutella 400g")

        assert result is not None
        assert result.brand == "Ferrero"
        assert result.source == "exact_match"

    def test_exact_match_with_unit_stripping(self, db):
        _seed_enriched_item(
            db, "i1", "Nutella 400g", brand="Ferrero", category="Spreads"
        )
        # Query with different unit format
        result = resolve_product(db, "Nutella")

        # "nutella 400g" normalizes to "nutella" (unit stripped)
        # "Nutella" normalizes to "nutella"
        assert result is not None
        assert result.brand == "Ferrero"

    def test_fuzzy_match_above_threshold(self, db):
        _seed_enriched_item(
            db,
            "i1",
            "Charalambides Christis Full Fat Milk",
            brand="Charalambides Christis",
            category="Dairy",
        )
        # Slightly different name (truncated on receipt)
        result = resolve_product(db, "Charalambides Christis Full Fat")

        assert result is not None
        assert result.brand == "Charalambides Christis"
        assert result.source == "fuzzy_match"
        assert result.similarity >= 0.80

    def test_fuzzy_match_below_threshold_returns_none(self, db):
        _seed_enriched_item(
            db,
            "i1",
            "Organic Whole Wheat Bread",
            brand="Bakery",
            category="Bread",
        )
        result = resolve_product(db, "White Rice")

        assert result is None

    def test_vendor_key_preference(self, db):
        """Same-vendor matches should be preferred via bonus."""
        _seed_enriched_item(
            db,
            "i1",
            "Fresh Milk 1l",
            brand="Brand A",
            category="Dairy",
            vendor_key="vat-vendor-a",
        )
        _seed_enriched_item(
            db,
            "i2",
            "Fresh Milk 1l",
            brand="Brand B",
            category="Dairy",
            vendor_key="vat-vendor-b",
        )
        result = resolve_product(db, "Fresh Milk 1l", vendor_key="vat-vendor-a")

        assert result is not None
        assert result.brand == "Brand A"
        assert result.same_vendor is True

    def test_no_vendor_key_returns_any_match(self, db):
        _seed_enriched_item(db, "i1", "Butter 250g", brand="Lurpak", category="Dairy")
        result = resolve_product(db, "Butter 250g", vendor_key=None)

        assert result is not None
        assert result.brand == "Lurpak"

    def test_only_brand_populated(self, db):
        """Items with only brand (no category) are still valid candidates."""
        _seed_enriched_item(db, "i1", "Mystery Product", brand="ACME", category=None)
        result = resolve_product(db, "Mystery Product")

        assert result is not None
        assert result.brand == "ACME"
        assert result.category is None

    def test_only_category_populated(self, db):
        """Items with only category (no brand) are still valid candidates."""
        _seed_enriched_item(db, "i1", "Generic Bread", brand=None, category="Bakery")
        result = resolve_product(db, "Generic Bread")

        assert result is not None
        assert result.brand is None
        assert result.category == "Bakery"

    def test_deduplication_returns_valid_match(self, db):
        """When same name appears twice, a valid match is returned."""
        _seed_enriched_item(
            db,
            "i1",
            "Milk",
            brand="Old Brand",
            category="Dairy",
        )
        _seed_enriched_item(
            db,
            "i2",
            "Milk",
            brand="New Brand",
            category="Dairy",
        )
        result = resolve_product(db, "Milk")

        assert result is not None
        # Either brand is valid — FTS5 may return in rank order, full-scan by rowid
        assert result.brand in ("Old Brand", "New Brand")

    def test_items_without_brand_or_category_excluded(self, db):
        """Items with neither brand nor category should not be candidates."""
        _seed_enriched_item(db, "i1", "Plain Item", brand=None, category=None)
        result = resolve_product(db, "Plain Item")
        assert result is None

    def test_custom_threshold(self, db):
        _seed_enriched_item(
            db,
            "i1",
            "Organic Greek Yogurt",
            brand="Fage",
            category="Dairy",
        )
        # With very high threshold, fuzzy match should fail
        result = resolve_product(db, "Organic Greek Yogur", threshold=0.99)
        assert result is None

        # With lower threshold, should match
        result = resolve_product(db, "Organic Greek Yogur", threshold=0.80)
        assert result is not None


# ===========================================================================
# TestResolveProductsBatch
# ===========================================================================


class TestResolveProductsBatch:
    """Tests for resolve_products_batch() — multiple items at once."""

    def test_empty_items_returns_empty(self, db):
        _seed_enriched_item(db, "i1", "Milk", brand="Brand", category="Dairy")
        result = resolve_products_batch(db, [])
        assert result == {}

    def test_empty_db_returns_empty(self, db):
        items = [{"id": "x1", "name": "Milk"}]
        result = resolve_products_batch(db, items)
        assert result == {}

    def test_batch_with_matches(self, db):
        _seed_enriched_item(db, "i1", "Milk", brand="Farm", category="Dairy")
        _seed_enriched_item(db, "i2", "Bread", brand="Baker", category="Bakery")

        items = [
            {"id": "q1", "name": "Milk"},
            {"id": "q2", "name": "Bread"},
            {"id": "q3", "name": "Unknown Mystery Thing"},
        ]
        result = resolve_products_batch(db, items)

        assert "q1" in result
        assert result["q1"].brand == "Farm"
        assert "q2" in result
        assert result["q2"].brand == "Baker"
        assert "q3" not in result

    def test_batch_skips_empty_names(self, db):
        _seed_enriched_item(db, "i1", "Milk", brand="Farm", category="Dairy")
        items = [
            {"id": "q1", "name": ""},
            {"id": "q2", "name": "   "},
        ]
        result = resolve_products_batch(db, items)
        assert result == {}


# ===========================================================================
# TestEnrichItemByName
# ===========================================================================


class TestEnrichItemByName:
    """Tests for enrich_item_by_name() — service-level name enrichment."""

    def test_success_updates_db(self, db):
        # Seed a known enriched product
        _seed_enriched_item(
            db,
            "source-1",
            "Nutella 400g",
            brand="Ferrero",
            category="Hazelnut Spreads",
        )
        # Seed a target item without brand/category
        _seed_enriched_item(
            db,
            "target-1",
            "Nutella",
            brand=None,
            category=None,
        )

        result = enrich_item_by_name(db, "target-1", "Nutella")

        assert result.success is True
        assert result.brand == "Ferrero"
        assert result.category == "Hazelnut Spreads"
        assert result.source == "product_resolver"

        # Verify DB was updated
        row = db.fetchone(
            "SELECT brand, category FROM fact_items WHERE id = 'target-1'"
        )
        assert row["brand"] == "Ferrero"
        assert row["category"] == "Hazelnut Spreads"

    def test_no_match_returns_failure(self, db):
        _seed_enriched_item(
            db,
            "target-1",
            "Unknown Product XYZ",
            brand=None,
            category=None,
        )

        result = enrich_item_by_name(db, "target-1", "Unknown Product XYZ")

        assert result.success is False
        assert result.source == "product_resolver"

    def test_empty_name_returns_failure(self, db):
        result = enrich_item_by_name(db, "item-1", "")
        assert result.success is False


# ===========================================================================
# TestEnrichPendingByName
# ===========================================================================


class TestEnrichPendingByName:
    """Tests for enrich_pending_by_name() — batch name resolution."""

    def test_enriches_items_without_barcode(self, db):
        # Known enriched product
        _seed_enriched_item(
            db,
            "known-1",
            "Milk 1L",
            brand="Farm Fresh",
            category="Dairy",
            barcode="1111111111111",
        )
        # Target without barcode
        _seed_enriched_item(
            db,
            "target-1",
            "Milk 1L",
            brand=None,
            category=None,
            barcode=None,
        )

        results = enrich_pending_by_name(db, limit=10)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].brand == "Farm Fresh"

    def test_skips_items_with_barcode(self, db):
        """Items that have barcodes should not be picked up by name resolver."""
        _seed_enriched_item(
            db,
            "known-1",
            "Milk",
            brand="Farm",
            category="Dairy",
        )
        _seed_enriched_item(
            db,
            "target-1",
            "Milk",
            brand=None,
            category=None,
            barcode="1234567890123",
        )

        results = enrich_pending_by_name(db, limit=10)
        assert results == []

    def test_skips_already_enriched(self, db):
        _seed_enriched_item(
            db,
            "target-1",
            "Milk",
            brand="Already Set",
            category="Dairy",
            barcode=None,
        )

        results = enrich_pending_by_name(db, limit=10)
        assert results == []

    def test_empty_db_returns_empty(self, db):
        results = enrich_pending_by_name(db)
        assert results == []

    def test_respects_limit(self, db):
        _seed_enriched_item(
            db,
            "known-1",
            "Product A",
            brand="Brand",
            category="Category",
            barcode="1111111111111",
        )
        for i in range(5):
            _seed_enriched_item(
                db,
                f"target-{i}",
                "Product A",
                brand=None,
                category=None,
                barcode=None,
            )

        results = enrich_pending_by_name(db, limit=2)
        assert len(results) == 2


# ===========================================================================
# TestFTS5
# ===========================================================================


class TestFTS5:
    """Tests for FTS5-accelerated product resolution."""

    def test_fts5_table_created(self, db):
        """Verify FTS5 virtual table exists after schema init."""
        assert _fts5_available(db) is True

    def test_fts5_exact_match(self, db):
        """Exact match works via FTS5 path."""
        _seed_enriched_item(
            db, "fts-1", "Nutella 400g", brand="Ferrero", category="Spreads"
        )
        result = resolve_product(db, "Nutella 400g")
        assert result is not None
        assert result.brand == "Ferrero"
        assert result.source == "exact_match"

    def test_fts5_prefix_match(self, db):
        """FTS5 prefix query retrieves candidates for fuzzy match."""
        _seed_enriched_item(
            db,
            "fts-2",
            "Charalambides Christis Full Fat Milk",
            brand="Charalambides",
            category="Dairy",
        )
        # Truncated name — shares prefix tokens
        result = resolve_product(db, "Charalambides Christis Full Fat")
        assert result is not None
        assert result.brand == "Charalambides"
        assert result.source == "fuzzy_match"
        assert result.similarity >= 0.80

    def test_fts5_vendor_preference(self, db):
        """Same-vendor preference works with FTS5."""
        _seed_enriched_item(
            db,
            "fts-3a",
            "Fresh Milk",
            brand="Brand A",
            category="Dairy",
            vendor_key="vendor-a",
        )
        _seed_enriched_item(
            db,
            "fts-3b",
            "Fresh Milk",
            brand="Brand B",
            category="Dairy",
            vendor_key="vendor-b",
        )
        result = resolve_product(db, "Fresh Milk", vendor_key="vendor-a")
        assert result is not None
        assert result.brand == "Brand A"
        assert result.same_vendor is True

    def test_fts5_trigger_insert(self, db):
        """New enriched items are auto-indexed by trigger."""
        _seed_enriched_item(
            db, "fts-4", "Trigger Test Product", brand="TriggerBrand", category="Test"
        )
        # Should be findable via FTS5 immediately
        result = resolve_product(db, "Trigger Test Product")
        assert result is not None
        assert result.brand == "TriggerBrand"

    def test_fts5_trigger_update(self, db):
        """Updated items are reflected in FTS index."""
        _seed_enriched_item(
            db, "fts-5", "Update Me", brand="OldBrand", category="Dairy"
        )
        # Update the brand
        conn = db.get_connection()
        conn.execute("UPDATE fact_items SET brand = 'NewBrand' WHERE id = 'fts-5'")
        conn.commit()

        result = resolve_product(db, "Update Me")
        assert result is not None
        assert result.brand == "NewBrand"

    def test_fts5_trigger_delete(self, db):
        """Deleted items are removed from FTS index."""
        _seed_enriched_item(
            db, "fts-6", "Delete Me", brand="GoneBrand", category="Gone"
        )
        # Verify it's findable
        assert resolve_product(db, "Delete Me") is not None

        conn = db.get_connection()
        conn.execute("DELETE FROM fact_items WHERE id = 'fts-6'")
        conn.commit()

        # Should not be findable after delete
        assert resolve_product(db, "Delete Me") is None

    def test_rebuild_product_fts(self, db):
        """rebuild_product_fts clears and repopulates index."""
        _seed_enriched_item(
            db, "rebuild-1", "Rebuild Product", brand="RBrand", category="RCat"
        )
        count = rebuild_product_fts(db)
        assert count >= 1

        # Still findable after rebuild
        result = resolve_product(db, "Rebuild Product")
        assert result is not None
        assert result.brand == "RBrand"

    def test_fts5_broad_search_or_fallback(self, db):
        """OR-based broad search finds items when AND is too restrictive."""
        _seed_enriched_item(
            db,
            "fts-7",
            "Greek Strained Yogurt Natural",
            brand="Fage",
            category="Dairy",
        )
        # Query with only some matching tokens
        result = resolve_product(db, "Greek Yogurt Natural", threshold=0.70)
        assert result is not None
        assert result.brand == "Fage"

    def test_fts5_batch_resolution(self, db):
        """Batch resolution works via FTS5 for small batches."""
        _seed_enriched_item(db, "batch-1", "Milk 1L", brand="Farm", category="Dairy")
        _seed_enriched_item(db, "batch-2", "Bread", brand="Baker", category="Bakery")

        items = [
            {"id": "q1", "name": "Milk 1L"},
            {"id": "q2", "name": "Bread"},
            {"id": "q3", "name": "Unknown Mystery Thing"},
        ]
        result = resolve_products_batch(db, items)
        assert "q1" in result
        assert result["q1"].brand == "Farm"
        assert "q2" in result
        assert result["q2"].brand == "Baker"
        assert "q3" not in result

    def test_fts5_no_match_returns_none(self, db):
        """FTS5 path returns None when no match found."""
        _seed_enriched_item(
            db, "fts-8", "Specific Product", brand="SBrand", category="SCat"
        )
        result = resolve_product(db, "Completely Different Thing")
        assert result is None
