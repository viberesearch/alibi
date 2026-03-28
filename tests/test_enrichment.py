"""Tests for product enrichment — OFF client, service, and subscriber."""

from __future__ import annotations

import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.enrichment import off_client
from alibi.enrichment.off_client import (
    cached_lookup,
    get_cached,
    lookup_barcode,
    store_cache,
)
from alibi.enrichment.service import (
    EnrichmentResult,
    _extract_primary_category,
    enrich_by_barcode,
    enrich_item,
    enrich_pending_items,
)
from alibi.services.events import EventBus, EventType
from alibi.services.subscribers.enrichment import EnrichmentSubscriber

# ---------------------------------------------------------------------------
# Sample OFF product data
# ---------------------------------------------------------------------------

SAMPLE_PRODUCT = {
    "product_name": "Nutella",
    "brands": "Ferrero",
    "categories": "Breakfasts, Spreads, Sweet spreads, Hazelnut spreads",
    "categories_tags": [
        "en:breakfasts",
        "en:spreads",
        "en:sweet-spreads",
        "en:hazelnut-spreads",
    ],
    "quantity": "400 g",
    "nutriscore_grade": "e",
    "nutriments": {
        "energy-kcal_100g": 539,
        "sugars_100g": 56.3,
        "proteins_100g": 6.3,
    },
}

SAMPLE_PRODUCT_MINIMAL = {
    "product_name": "Plain Biscuits",
    "brands": "Generic",
}

NUTELLA_BARCODE = "3017624010701"
UNKNOWN_BARCODE = "9999999999999"


# ---------------------------------------------------------------------------
# Helpers — seed DB with minimal supporting data
# ---------------------------------------------------------------------------


def _seed_fact_item(
    db,
    item_id: str = "item-1",
    barcode: str | None = NUTELLA_BARCODE,
    brand: str | None = None,
    category: str | None = None,
    doc_id: str = "doc-1",
    fact_id: str = "fact-1",
    atom_id: str = "atom-1",
    cloud_id: str = "cloud-1",
) -> None:
    """Insert the minimal chain required for a fact_item row."""
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
        "(id, cloud_id, fact_type, vendor, total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', 'Test Vendor', 10.0, 'EUR', '2026-01-01')",
        (fact_id, cloud_id),
    )
    conn.execute(
        "INSERT OR IGNORE INTO fact_items "
        "(id, fact_id, atom_id, name, barcode, brand, category) "
        "VALUES (?, ?, ?, 'Test Item', ?, ?, ?)",
        (item_id, fact_id, atom_id, barcode, brand, category),
    )
    conn.commit()


def _seed_llm_inferred_item(
    db,
    item_id: str,
    name: str,
    category: str,
    brand: str | None = None,
    barcode: str | None = None,
    doc_id: str | None = None,
    fact_id: str | None = None,
    atom_id: str | None = None,
    cloud_id: str | None = None,
) -> None:
    """Seed a fact_item with enrichment_source='llm_inference' and a given category."""
    _doc_id = doc_id or f"doc-{item_id}"
    _fact_id = fact_id or f"fact-{item_id}"
    _atom_id = atom_id or f"atom-{item_id}"
    _cloud_id = cloud_id or f"cloud-{item_id}"

    conn = db.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO documents (id, file_path, file_hash) "
        "VALUES (?, ?, ?)",
        (_doc_id, f"/tmp/{_doc_id}.jpg", f"hash-{_doc_id}"),
    )
    conn.execute(
        "INSERT OR IGNORE INTO atoms (id, document_id, atom_type, data) "
        "VALUES (?, ?, 'item', '{}')",
        (_atom_id, _doc_id),
    )
    conn.execute(
        "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')",
        (_cloud_id,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO facts "
        "(id, cloud_id, fact_type, vendor, total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', 'Test Vendor', 10.0, 'EUR', '2026-01-01')",
        (_fact_id, _cloud_id),
    )
    conn.execute(
        "INSERT OR IGNORE INTO fact_items "
        "(id, fact_id, atom_id, name, barcode, brand, category, "
        "enrichment_source, enrichment_confidence) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'llm_inference', 0.7)",
        (item_id, _fact_id, _atom_id, name, barcode, brand, category),
    )
    conn.commit()


# ===========================================================================
# TestOffClientLookup
# ===========================================================================


class TestOffClientLookup:
    """Tests for lookup_barcode() — live HTTP path (mocked)."""

    def setup_method(self):
        # Reset rate-limit state so tests are not artificially delayed
        off_client._last_request_time = 0.0

    def _make_mock_client(self, status_code: int, json_payload: dict):
        mock_resp = MagicMock()
        mock_resp.status_code = status_code
        mock_resp.json.return_value = json_payload
        mock_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        return mock_client

    @patch("alibi.enrichment.off_client.httpx.Client")
    def test_found_returns_product(self, mock_client_cls):
        mock_client_cls.return_value = self._make_mock_client(
            200, {"status": 1, "product": SAMPLE_PRODUCT}
        )
        result = lookup_barcode(NUTELLA_BARCODE)
        assert result is not None
        assert result["product_name"] == "Nutella"
        assert result["brands"] == "Ferrero"

    @patch("alibi.enrichment.off_client.httpx.Client")
    def test_not_found_returns_none(self, mock_client_cls):
        mock_client_cls.return_value = self._make_mock_client(
            200, {"status": 0, "status_verbose": "product not found"}
        )
        result = lookup_barcode(UNKNOWN_BARCODE)
        assert result is None

    @patch("alibi.enrichment.off_client.httpx.Client")
    def test_404_returns_none(self, mock_client_cls):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = lookup_barcode(UNKNOWN_BARCODE)
        assert result is None

    @patch("alibi.enrichment.off_client.httpx.Client")
    def test_missing_product_key_returns_none(self, mock_client_cls):
        # status==1 but no "product" key
        mock_client_cls.return_value = self._make_mock_client(200, {"status": 1})
        result = lookup_barcode(NUTELLA_BARCODE)
        assert result is None

    @patch("alibi.enrichment.off_client.httpx.Client")
    def test_http_status_error_returns_none(self, mock_client_cls):
        import httpx

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=mock_resp
        )
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = lookup_barcode(NUTELLA_BARCODE)
        assert result is None

    @patch("alibi.enrichment.off_client.httpx.Client")
    def test_request_error_returns_none(self, mock_client_cls):
        import httpx

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.RequestError("Connection refused")
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = mock_client

        result = lookup_barcode(NUTELLA_BARCODE)
        assert result is None

    @patch("alibi.enrichment.off_client.httpx.Client")
    def test_rate_limit_respected(self, mock_client_cls):
        """Two consecutive calls should enforce minimum interval."""
        mock_client_cls.return_value = self._make_mock_client(
            200, {"status": 1, "product": SAMPLE_PRODUCT_MINIMAL}
        )
        # Pre-set last request to "just now"
        off_client._last_request_time = time.monotonic()

        t_start = time.monotonic()
        lookup_barcode("0000000000001")
        elapsed = time.monotonic() - t_start

        # Should have slept at least _MIN_REQUEST_INTERVAL - epsilon
        assert elapsed >= off_client._MIN_REQUEST_INTERVAL * 0.9

    @patch("alibi.enrichment.off_client.httpx.Client")
    def test_fields_in_request_url(self, mock_client_cls):
        """The request URL should include the fields parameter."""
        mock_client_cls.return_value = self._make_mock_client(
            200, {"status": 1, "product": SAMPLE_PRODUCT}
        )
        lookup_barcode(NUTELLA_BARCODE)

        mock_client = mock_client_cls.return_value
        call_args = mock_client.get.call_args
        url = call_args[0][0]
        assert "fields=" in url
        assert NUTELLA_BARCODE in url


# ===========================================================================
# TestProductCache
# ===========================================================================


class TestProductCache:
    """Tests for get_cached() / store_cache() — uses real DB."""

    def test_miss_returns_none(self, db):
        assert get_cached(db, "1234567890123") is None

    def test_store_and_retrieve(self, db):
        store_cache(db, "1234567890123", SAMPLE_PRODUCT)
        cached = get_cached(db, "1234567890123")
        assert cached is not None
        assert cached["product_name"] == "Nutella"
        assert cached["brands"] == "Ferrero"

    def test_replace_updates_value(self, db):
        store_cache(db, "1234567890123", {"product_name": "Old Name"})
        store_cache(db, "1234567890123", {"product_name": "New Name"})
        cached = get_cached(db, "1234567890123")
        assert cached["product_name"] == "New Name"

    def test_store_custom_source(self, db):
        store_cache(db, "1234567890123", SAMPLE_PRODUCT, source="test_source")
        row = db.fetchone(
            "SELECT source FROM product_cache WHERE barcode = ?",
            ("1234567890123",),
        )
        assert row["source"] == "test_source"

    def test_default_source_is_openfoodfacts(self, db):
        store_cache(db, "1234567890123", SAMPLE_PRODUCT)
        row = db.fetchone(
            "SELECT source FROM product_cache WHERE barcode = ?",
            ("1234567890123",),
        )
        assert row["source"] == "openfoodfacts"

    def test_different_barcodes_stored_independently(self, db):
        store_cache(db, "1111111111111", {"product_name": "Alpha"})
        store_cache(db, "2222222222222", {"product_name": "Beta"})
        assert get_cached(db, "1111111111111")["product_name"] == "Alpha"
        assert get_cached(db, "2222222222222")["product_name"] == "Beta"

    def test_complex_json_roundtrip(self, db):
        store_cache(db, "1234567890123", SAMPLE_PRODUCT)
        cached = get_cached(db, "1234567890123")
        assert cached["nutriments"]["energy-kcal_100g"] == 539
        assert cached["categories_tags"][-1] == "en:hazelnut-spreads"


# ===========================================================================
# TestCachedLookup
# ===========================================================================


class TestCachedLookup:
    """Tests for cached_lookup() — cache-first behaviour."""

    @patch("alibi.enrichment.off_client.lookup_barcode")
    def test_cache_hit_skips_api(self, mock_lookup, db):
        store_cache(db, NUTELLA_BARCODE, SAMPLE_PRODUCT)
        result = cached_lookup(db, NUTELLA_BARCODE)
        assert result is not None
        assert result["product_name"] == "Nutella"
        mock_lookup.assert_not_called()

    @patch("alibi.enrichment.off_client.lookup_barcode")
    def test_cache_miss_calls_api(self, mock_lookup, db):
        mock_lookup.return_value = SAMPLE_PRODUCT
        result = cached_lookup(db, NUTELLA_BARCODE)
        assert result is not None
        mock_lookup.assert_called_once_with(NUTELLA_BARCODE)

    @patch("alibi.enrichment.off_client.lookup_barcode")
    def test_api_result_stored_in_cache(self, mock_lookup, db):
        mock_lookup.return_value = SAMPLE_PRODUCT
        cached_lookup(db, NUTELLA_BARCODE)
        # Second call should not trigger API again
        mock_lookup.reset_mock()
        result = cached_lookup(db, NUTELLA_BARCODE)
        assert result is not None
        mock_lookup.assert_not_called()

    @patch("alibi.enrichment.off_client.lookup_barcode")
    def test_api_not_found_returns_none(self, mock_lookup, db):
        mock_lookup.return_value = None
        result = cached_lookup(db, UNKNOWN_BARCODE)
        assert result is None

    @patch("alibi.enrichment.off_client.lookup_barcode")
    def test_api_not_found_caches_negative(self, mock_lookup, db):
        """Negative lookups are cached to avoid repeated API calls."""
        mock_lookup.return_value = None
        cached_lookup(db, UNKNOWN_BARCODE)
        # Negative sentinel should be in cache
        cached = get_cached(db, UNKNOWN_BARCODE)
        assert cached is not None
        assert cached.get("_not_found") is True

    @patch("alibi.enrichment.off_client.lookup_barcode")
    def test_negative_cache_prevents_repeat_api_call(self, mock_lookup, db):
        """Second lookup of unknown barcode should not hit API."""
        mock_lookup.return_value = None
        cached_lookup(db, UNKNOWN_BARCODE)
        mock_lookup.reset_mock()

        result = cached_lookup(db, UNKNOWN_BARCODE)
        assert result is None
        mock_lookup.assert_not_called()


# ===========================================================================
# TestExtractPrimaryCategory
# ===========================================================================


class TestExtractPrimaryCategory:
    """Tests for the pure-function category extractor."""

    def test_from_categories_tags(self):
        product = {
            "categories_tags": ["en:breakfasts", "en:spreads", "en:hazelnut-spreads"]
        }
        assert _extract_primary_category(product) == "Hazelnut Spreads"

    def test_tags_title_case_and_hyphen_replacement(self):
        product = {"categories_tags": ["en:sweet-spreads"]}
        assert _extract_primary_category(product) == "Sweet Spreads"

    def test_tag_without_language_prefix(self):
        product = {"categories_tags": ["hazelnut-spreads"]}
        assert _extract_primary_category(product) == "Hazelnut Spreads"

    def test_from_categories_string_fallback(self):
        product = {"categories": "Breakfasts, Spreads, Hazelnut spreads"}
        assert _extract_primary_category(product) == "Hazelnut spreads"

    def test_categories_string_takes_last_part(self):
        product = {"categories": "A, B, C, D"}
        assert _extract_primary_category(product) == "D"

    def test_categories_tags_preferred_over_string(self):
        product = {
            "categories_tags": ["en:dairy"],
            "categories": "Something else, Different",
        }
        assert _extract_primary_category(product) == "Dairy"

    def test_empty_product(self):
        assert _extract_primary_category({}) is None

    def test_empty_tags_list_falls_back_to_string(self):
        product = {
            "categories_tags": [],
            "categories": "Dairy products",
        }
        assert _extract_primary_category(product) == "Dairy products"

    def test_single_tag(self):
        product = {"categories_tags": ["en:beverages"]}
        assert _extract_primary_category(product) == "Beverages"

    def test_full_sample_product(self):
        result = _extract_primary_category(SAMPLE_PRODUCT)
        assert result == "Hazelnut Spreads"


# ===========================================================================
# TestEnrichItem
# ===========================================================================


class TestEnrichItem:
    """Tests for enrich_item() — uses real DB."""

    @patch("alibi.enrichment.service.cached_lookup")
    def test_success_returns_enrichment_result(self, mock_lookup, db):
        mock_lookup.return_value = SAMPLE_PRODUCT
        _seed_fact_item(db, item_id="item-1", barcode=NUTELLA_BARCODE)

        result = enrich_item(db, "item-1", NUTELLA_BARCODE)

        assert isinstance(result, EnrichmentResult)
        assert result.success is True
        assert result.item_id == "item-1"
        assert result.barcode == NUTELLA_BARCODE
        assert result.brand == "Ferrero"
        assert result.category == "Hazelnut Spreads"
        assert result.product_name == "Nutella"

    @patch("alibi.enrichment.service.cached_lookup")
    def test_success_persists_brand_and_category(self, mock_lookup, db):
        mock_lookup.return_value = SAMPLE_PRODUCT
        _seed_fact_item(db, item_id="item-1", barcode=NUTELLA_BARCODE)

        enrich_item(db, "item-1", NUTELLA_BARCODE)

        row = db.fetchone("SELECT brand, category FROM fact_items WHERE id = 'item-1'")
        assert row["brand"] == "Ferrero"
        assert row["category"] == "Hazelnut Spreads"

    @patch("alibi.enrichment.service.cached_lookup")
    def test_not_found_returns_failure(self, mock_lookup, db):
        mock_lookup.return_value = None

        result = enrich_item(db, "item-1", UNKNOWN_BARCODE)

        assert result.success is False
        assert result.item_id == "item-1"
        assert result.barcode == UNKNOWN_BARCODE
        assert result.brand is None
        assert result.category is None

    @patch("alibi.enrichment.service.cached_lookup")
    def test_product_with_no_brand_skips_brand_update(self, mock_lookup, db):
        """If OFF returns no brands field, brand column stays NULL."""
        mock_lookup.return_value = {
            "product_name": "Mystery Item",
            "categories_tags": ["en:food"],
        }
        _seed_fact_item(db, item_id="item-2", barcode="1111111111111")

        result = enrich_item(db, "item-2", "1111111111111")

        assert result.success is True
        assert result.brand is None
        row = db.fetchone("SELECT brand FROM fact_items WHERE id = 'item-2'")
        assert row["brand"] is None

    @patch("alibi.enrichment.service.cached_lookup")
    def test_product_with_only_brand(self, mock_lookup, db):
        """Category stays NULL when OFF has no categories."""
        mock_lookup.return_value = {"product_name": "Brand Only", "brands": "ACME"}
        _seed_fact_item(db, item_id="item-3", barcode="2222222222222")

        result = enrich_item(db, "item-3", "2222222222222")

        assert result.success is True
        assert result.brand == "ACME"
        assert result.category is None
        row = db.fetchone("SELECT brand, category FROM fact_items WHERE id = 'item-3'")
        assert row["brand"] == "ACME"
        assert row["category"] is None

    @patch("alibi.enrichment.service.cached_lookup")
    def test_source_field_defaults_to_openfoodfacts(self, mock_lookup, db):
        mock_lookup.return_value = SAMPLE_PRODUCT
        _seed_fact_item(db, item_id="item-1", barcode=NUTELLA_BARCODE)

        result = enrich_item(db, "item-1", NUTELLA_BARCODE)

        assert result.source == "openfoodfacts"


# ===========================================================================
# TestEnrichPendingItems
# ===========================================================================


class TestEnrichPendingItems:
    """Tests for enrich_pending_items() — batch enrichment."""

    @patch("alibi.enrichment.service.cached_lookup")
    def test_enriches_items_without_brand(self, mock_lookup, db):
        mock_lookup.return_value = SAMPLE_PRODUCT
        _seed_fact_item(
            db,
            item_id="item-1",
            barcode=NUTELLA_BARCODE,
            brand=None,
            category=None,
            doc_id="doc-1",
            fact_id="fact-1",
            atom_id="atom-1",
            cloud_id="cloud-1",
        )

        results = enrich_pending_items(db)

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].brand == "Ferrero"

    @patch("alibi.enrichment.service.cached_lookup")
    def test_skips_items_without_barcode(self, mock_lookup, db):
        _seed_fact_item(
            db,
            item_id="item-nobar",
            barcode=None,
            brand=None,
            category=None,
            doc_id="doc-2",
            fact_id="fact-2",
            atom_id="atom-2",
            cloud_id="cloud-2",
        )

        results = enrich_pending_items(db)

        assert results == []
        mock_lookup.assert_not_called()

    @patch("alibi.enrichment.service.cached_lookup")
    def test_skips_items_already_enriched(self, mock_lookup, db):
        _seed_fact_item(
            db,
            item_id="item-done",
            barcode=NUTELLA_BARCODE,
            brand="Ferrero",
            category="Hazelnut Spreads",
            doc_id="doc-3",
            fact_id="fact-3",
            atom_id="atom-3",
            cloud_id="cloud-3",
        )

        results = enrich_pending_items(db)

        assert results == []
        mock_lookup.assert_not_called()

    @patch("alibi.enrichment.service.cached_lookup")
    def test_respects_limit_parameter(self, mock_lookup, db):
        mock_lookup.return_value = SAMPLE_PRODUCT
        for i in range(5):
            _seed_fact_item(
                db,
                item_id=f"item-{i}",
                barcode=f"100000000{i:04d}",
                brand=None,
                category=None,
                doc_id=f"doc-{i}",
                fact_id=f"fact-{i}",
                atom_id=f"atom-{i}",
                cloud_id=f"cloud-{i}",
            )

        results = enrich_pending_items(db, limit=2)

        assert len(results) == 2

    @patch("alibi.enrichment.service.cached_lookup")
    def test_empty_db_returns_empty_list(self, mock_lookup, db):
        results = enrich_pending_items(db)
        assert results == []
        mock_lookup.assert_not_called()


# ===========================================================================
# TestEnrichByBarcode
# ===========================================================================


class TestEnrichByBarcode:
    """Tests for enrich_by_barcode() — enrich all items sharing a barcode."""

    @patch("alibi.enrichment.service.cached_lookup")
    def test_single_matching_item(self, mock_lookup, db):
        mock_lookup.return_value = SAMPLE_PRODUCT
        _seed_fact_item(db, item_id="item-1", barcode=NUTELLA_BARCODE)

        results = enrich_by_barcode(db, NUTELLA_BARCODE)

        assert len(results) == 1
        assert results[0].success is True

    @patch("alibi.enrichment.service.cached_lookup")
    def test_no_matching_items_returns_empty(self, mock_lookup, db):
        results = enrich_by_barcode(db, NUTELLA_BARCODE)
        assert results == []

    @patch("alibi.enrichment.service.cached_lookup")
    def test_multiple_items_same_barcode(self, mock_lookup, db):
        mock_lookup.return_value = SAMPLE_PRODUCT
        for i in range(3):
            _seed_fact_item(
                db,
                item_id=f"item-{i}",
                barcode=NUTELLA_BARCODE,
                doc_id=f"doc-{i}",
                fact_id=f"fact-{i}",
                atom_id=f"atom-{i}",
                cloud_id=f"cloud-{i}",
            )

        results = enrich_by_barcode(db, NUTELLA_BARCODE)

        assert len(results) == 3
        assert all(r.success for r in results)


# ===========================================================================
# TestEnrichmentSubscriber
# ===========================================================================


class TestEnrichmentSubscriber:
    """Tests for the EventBus subscriber."""

    def test_start_registers_handler(self):
        bus = EventBus()
        sub = EnrichmentSubscriber(db_factory=MagicMock(), bus=bus)

        sub.start()

        assert len(bus._subscribers[EventType.FACT_CREATED]) == 1

    def test_stop_unregisters_handler(self):
        bus = EventBus()
        sub = EnrichmentSubscriber(db_factory=MagicMock(), bus=bus)
        sub.start()
        assert len(bus._subscribers[EventType.FACT_CREATED]) == 1

        sub.stop()

        assert len(bus._subscribers[EventType.FACT_CREATED]) == 0

    def test_stop_without_start_is_noop(self):
        bus = EventBus()
        sub = EnrichmentSubscriber(db_factory=MagicMock(), bus=bus)
        sub.stop()  # Should not raise

    def test_stop_with_no_bus_is_noop(self):
        sub = EnrichmentSubscriber(db_factory=MagicMock(), bus=None)
        sub.stop()  # Should not raise

    def test_initial_enrichment_count_is_zero(self):
        bus = EventBus()
        sub = EnrichmentSubscriber(db_factory=MagicMock(), bus=bus)
        assert sub.enrichment_count == 0

    def test_handle_spawns_daemon_thread(self):
        """Verify _handle() spawns a background thread (non-blocking)."""
        bus = EventBus()
        threads_started = []

        original_thread_start = threading.Thread.start

        def tracking_start(self_thread, *args, **kwargs):
            threads_started.append(self_thread)
            original_thread_start(self_thread, *args, **kwargs)

        db_mock = MagicMock()
        db_mock.fetchall.return_value = []

        sub = EnrichmentSubscriber(db_factory=lambda: db_mock, bus=bus)
        sub.start()

        with patch.object(threading.Thread, "start", tracking_start):
            bus.emit(EventType.FACT_CREATED, {"document_id": "doc-x"})

        # Give the thread a moment to start
        time.sleep(0.05)
        assert len(threads_started) >= 1

    def test_event_without_document_id_is_ignored(self):
        """Events missing document_id should not crash or call enrichment."""
        bus = EventBus()
        db_mock = MagicMock()
        sub = EnrichmentSubscriber(db_factory=lambda: db_mock, bus=bus)
        sub.start()

        bus.emit(EventType.FACT_CREATED, {})

        # Give the thread time to run
        time.sleep(0.05)
        # db_factory should not have been called (no document_id to work with)
        db_mock.fetchall.assert_not_called()

    @patch("alibi.enrichment.off_client.cached_lookup")
    def test_integration_enriches_items_on_fact_created(self, mock_lookup, db):
        """End-to-end: subscriber wires up, event triggers enrichment via DB."""
        mock_lookup.return_value = SAMPLE_PRODUCT

        # Build the full document → atom → bundle → cloud → fact → fact_item chain
        conn = db.get_connection()
        conn.executescript(
            """
INSERT INTO documents (id, file_path, file_hash)
    VALUES ('doc-sub', '/tmp/sub.jpg', 'hash-sub');
INSERT INTO atoms (id, document_id, atom_type, data)
    VALUES ('atom-sub', 'doc-sub', 'item', '{}');
INSERT INTO clouds (id, status) VALUES ('cloud-sub', 'collapsed');
INSERT INTO facts
    (id, cloud_id, fact_type, vendor, total_amount, currency, event_date)
    VALUES ('fact-sub', 'cloud-sub', 'purchase', 'Shop', 5.0, 'EUR', '2026-01-01');
INSERT INTO fact_items (id, fact_id, atom_id, name, barcode)
    VALUES ('fi-sub', 'fact-sub', 'atom-sub', 'Nutella 400g', '3017624010701');
INSERT INTO bundles (id, document_id, bundle_type, cloud_id)
    VALUES ('bundle-sub', 'doc-sub', 'basket', 'cloud-sub');
INSERT INTO bundle_atoms (bundle_id, atom_id, role)
    VALUES ('bundle-sub', 'atom-sub', 'basket_item');
INSERT INTO cloud_bundles (cloud_id, bundle_id, match_type)
    VALUES ('cloud-sub', 'bundle-sub', 'manual');
        """
        )
        conn.commit()

        bus = EventBus()
        sub = EnrichmentSubscriber(db_factory=lambda: db, bus=bus)
        sub.start()

        bus.emit(EventType.FACT_CREATED, {"document_id": "doc-sub"})

        # Wait for background thread to complete enrichment
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            row = db.fetchone("SELECT brand FROM fact_items WHERE id = 'fi-sub'")
            if row and row["brand"] == "Ferrero" and sub.enrichment_count >= 1:
                break
            time.sleep(0.05)

        row = db.fetchone("SELECT brand, category FROM fact_items WHERE id = 'fi-sub'")
        assert row["brand"] == "Ferrero"
        assert row["category"] == "Hazelnut Spreads"
        assert sub.enrichment_count == 1

        sub.stop()

    def test_uses_global_event_bus_when_none_provided(self):
        """When bus=None, start() should wire to the singleton event_bus."""
        from alibi.services.events import event_bus

        sub = EnrichmentSubscriber(db_factory=MagicMock(), bus=None)
        sub.start()

        try:
            assert sub._handle in event_bus._subscribers[EventType.FACT_CREATED]
        finally:
            sub.stop()
            # Ensure cleanup regardless of assertion result
            event_bus._subscribers[EventType.FACT_CREATED] = [
                h
                for h in event_bus._subscribers[EventType.FACT_CREATED]
                if h is not sub._handle
            ]


# ===========================================================================
# TestCloudEnrichment
# ===========================================================================


class TestCloudEnrichment:
    """Tests for cloud_enrichment module."""

    # ------------------------------------------------------------------
    # test_cloud_enrichment_disabled_by_default
    # ------------------------------------------------------------------

    def test_cloud_enrichment_disabled_by_default(self, db, monkeypatch):
        """enrich_pending_by_cloud returns [] when feature flag is off."""
        from alibi.enrichment.cloud_enrichment import enrich_pending_by_cloud

        # Remove the env var if present so the default (off) applies
        monkeypatch.delenv("ALIBI_CLOUD_ENRICHMENT_ENABLED", raising=False)

        # Seed an item that would otherwise be eligible
        _seed_fact_item(
            db,
            item_id="cloud-item-1",
            barcode=None,
            brand=None,
            category=None,
            doc_id="cloud-doc-1",
            fact_id="cloud-fact-1",
            atom_id="cloud-atom-1",
            cloud_id="cloud-cloud-1",
        )

        results = enrich_pending_by_cloud(db)

        assert results == []

    # ------------------------------------------------------------------
    # test_infer_cloud_brand_category_parses_response
    # ------------------------------------------------------------------

    def test_infer_cloud_brand_category_parses_response(self):
        """infer_cloud_brand_category correctly parses a mocked API response."""
        import json as _json

        import httpx
        from unittest.mock import MagicMock, patch

        from alibi.enrichment.cloud_enrichment import infer_cloud_brand_category

        api_payload = {
            "content": [
                {
                    "type": "text",
                    "text": _json.dumps(
                        {
                            "items": [
                                {"idx": 1, "brand": "Ferrero", "category": "Sweets"},
                                {"idx": 2, "brand": None, "category": "Beverages"},
                            ]
                        }
                    ),
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = api_payload
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            result = infer_cloud_brand_category(
                [
                    {"idx": 1, "name": "Nutella 400g", "barcode": "3017624010701"},
                    {"idx": 2, "name": "Still Water 1L"},
                ],
                api_key="test-key",
            )

        # Verify the API was called with the expected structure
        assert mock_post.called
        call_kwargs = mock_post.call_args
        sent_json = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert sent_json["model"] == "claude-haiku-4-5-20251001"
        assert len(sent_json["messages"]) == 1
        assert "Nutella 400g" in sent_json["messages"][0]["content"]
        assert "3017624010701" in sent_json["messages"][0]["content"]

        # Verify parsed output
        assert len(result) == 2
        assert result[0]["idx"] == 1
        assert result[0]["brand"] == "Ferrero"
        assert result[0]["category"] == "Sweets"
        assert result[1]["idx"] == 2
        assert result[1]["brand"] is None
        assert result[1]["category"] == "Beverages"

    # ------------------------------------------------------------------
    # test_cloud_enrichment_sets_provenance
    # ------------------------------------------------------------------

    def test_cloud_enrichment_sets_provenance(self, db, monkeypatch):
        """enrich_items_by_cloud writes enrichment_source=cloud_api and confidence=0.85."""
        import json as _json
        from unittest.mock import MagicMock, patch

        from alibi.enrichment.cloud_enrichment import enrich_items_by_cloud

        # Seed a fact_item with no brand/category
        _seed_fact_item(
            db,
            item_id="prov-item-1",
            barcode="1234567890123",
            brand=None,
            category=None,
            doc_id="prov-doc-1",
            fact_id="prov-fact-1",
            atom_id="prov-atom-1",
            cloud_id="prov-cloud-1",
        )

        api_payload = {
            "content": [
                {
                    "type": "text",
                    "text": _json.dumps(
                        {
                            "items": [
                                {"idx": 1, "brand": "Acme", "category": "Snacks"},
                            ]
                        }
                    ),
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = api_payload
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            results = enrich_items_by_cloud(
                db,
                [
                    {
                        "id": "prov-item-1",
                        "name": "Acme Crisps",
                        "barcode": "1234567890123",
                    }
                ],
                api_key="test-key",
            )

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].brand == "Acme"
        assert results[0].category == "Snacks"

        # Verify DB was updated with correct provenance fields
        row = db.fetchone(
            "SELECT brand, category, enrichment_source, enrichment_confidence "
            "FROM fact_items WHERE id = 'prov-item-1'"
        )
        assert row["brand"] == "Acme"
        assert row["category"] == "Snacks"
        assert row["enrichment_source"] == "cloud_api"
        assert abs(row["enrichment_confidence"] - 0.85) < 1e-6

    # ------------------------------------------------------------------
    # test_cloud_enrichment_no_api_key_returns_empty
    # ------------------------------------------------------------------

    def test_cloud_enrichment_no_api_key_returns_empty(self, db, monkeypatch):
        """enrich_pending_by_cloud returns [] when API key is missing."""
        from alibi.enrichment.cloud_enrichment import enrich_pending_by_cloud

        monkeypatch.setenv("ALIBI_CLOUD_ENRICHMENT_ENABLED", "1")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        _seed_fact_item(
            db,
            item_id="nokey-item-1",
            barcode=None,
            brand=None,
            category=None,
            doc_id="nokey-doc-1",
            fact_id="nokey-fact-1",
            atom_id="nokey-atom-1",
            cloud_id="nokey-cloud-1",
        )

        results = enrich_pending_by_cloud(db, api_key=None)

        assert results == []

    # ------------------------------------------------------------------
    # test_cloud_enrichment_deduplicates_by_name_and_barcode
    # ------------------------------------------------------------------

    def test_cloud_enrichment_deduplicates_by_name_and_barcode(self, db, monkeypatch):
        """Two items with identical name+barcode result in one API call entry."""
        import json as _json
        from unittest.mock import MagicMock, patch

        from alibi.enrichment.cloud_enrichment import enrich_items_by_cloud

        # Seed two identical items (same name+barcode, different IDs)
        for i in range(2):
            _seed_fact_item(
                db,
                item_id=f"dedup-item-{i}",
                barcode="9999999999001",
                brand=None,
                category=None,
                doc_id=f"dedup-doc-{i}",
                fact_id=f"dedup-fact-{i}",
                atom_id=f"dedup-atom-{i}",
                cloud_id=f"dedup-cloud-{i}",
            )

        # API returns a single result for idx=1 (the deduplicated entry)
        api_payload = {
            "content": [
                {
                    "type": "text",
                    "text": _json.dumps(
                        {
                            "items": [
                                {"idx": 1, "brand": "SharedBrand", "category": "Other"}
                            ]
                        }
                    ),
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.json.return_value = api_payload
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            results = enrich_items_by_cloud(
                db,
                [
                    {
                        "id": "dedup-item-0",
                        "name": "Test Item",
                        "barcode": "9999999999001",
                    },
                    {
                        "id": "dedup-item-1",
                        "name": "Test Item",
                        "barcode": "9999999999001",
                    },
                ],
                api_key="test-key",
            )

        # Only one unique item sent to the API
        call_kwargs = mock_post.call_args
        sent_json = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        prompt_content = sent_json["messages"][0]["content"]
        # Exactly one numbered entry in the prompt
        assert prompt_content.count("1. ") == 1
        assert "2. " not in prompt_content

        # Both items get the result
        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.brand == "SharedBrand" for r in results)
        assert all(r.category == "Other" for r in results)


# ===========================================================================
# TestRefineCategories
# ===========================================================================


class TestRefineCategories:
    """Tests for refine_categories_by_cloud()."""

    # ------------------------------------------------------------------
    # test_refine_corrects_wrong_category
    # ------------------------------------------------------------------

    def test_refine_corrects_wrong_category(self, db):
        """refine_categories_by_cloud updates category when cloud disagrees."""
        import json as _json
        from unittest.mock import MagicMock, patch

        from alibi.enrichment.cloud_enrichment import refine_categories_by_cloud

        _seed_llm_inferred_item(
            db,
            item_id="refine-item-1",
            name="Atlantic Salmon Fillet",
            category="Meat",  # wrong — local LLM misclassified it
            brand=None,
        )

        api_payload = {
            "content": [
                {
                    "type": "text",
                    "text": _json.dumps(
                        {
                            "items": [
                                {
                                    "idx": 1,
                                    "corrected_category": "Fish",
                                    "reason": "salmon is fish not meat",
                                }
                            ]
                        }
                    ),
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = api_payload
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            results = refine_categories_by_cloud(db, api_key="test-key")

        assert len(results) == 1
        assert results[0].item_id == "refine-item-1"
        assert results[0].category == "Fish"
        assert results[0].success is True

        # Verify DB was updated with cloud_refined provenance
        row = db.fetchone(
            "SELECT category, enrichment_source, enrichment_confidence "
            "FROM fact_items WHERE id = 'refine-item-1'"
        )
        assert row["category"] == "Fish"
        assert row["enrichment_source"] == "cloud_refined"
        assert abs(row["enrichment_confidence"] - 0.9) < 1e-6

    # ------------------------------------------------------------------
    # test_refine_skips_when_cloud_agrees
    # ------------------------------------------------------------------

    def test_refine_skips_when_cloud_agrees(self, db):
        """refine_categories_by_cloud returns empty list when cloud agrees with LLM."""
        import json as _json
        from unittest.mock import MagicMock, patch

        from alibi.enrichment.cloud_enrichment import refine_categories_by_cloud

        _seed_llm_inferred_item(
            db,
            item_id="refine-item-2",
            name="Greek Yoghurt 500g",
            category="Dairy",  # correct — cloud should agree
        )

        api_payload = {
            "content": [
                {
                    "type": "text",
                    "text": _json.dumps(
                        {
                            "items": [
                                {
                                    "idx": 1,
                                    "corrected_category": None,
                                }
                            ]
                        }
                    ),
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = api_payload
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response):
            results = refine_categories_by_cloud(db, api_key="test-key")

        # No correction was needed
        assert results == []

        # DB must not have changed the enrichment_source
        row = db.fetchone(
            "SELECT enrichment_source FROM fact_items WHERE id = 'refine-item-2'"
        )
        assert row["enrichment_source"] == "llm_inference"

    # ------------------------------------------------------------------
    # test_refine_no_api_key_returns_empty
    # ------------------------------------------------------------------

    def test_refine_no_api_key_returns_empty(self, db, monkeypatch):
        """refine_categories_by_cloud returns [] when API key is missing."""
        from alibi.enrichment.cloud_enrichment import refine_categories_by_cloud

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        _seed_llm_inferred_item(
            db,
            item_id="refine-item-3",
            name="Tuna Steak",
            category="Meat",
        )

        results = refine_categories_by_cloud(db, api_key=None)
        assert results == []

    # ------------------------------------------------------------------
    # test_refine_excludes_non_llm_items
    # ------------------------------------------------------------------

    def test_refine_excludes_non_llm_items(self, db):
        """refine_categories_by_cloud only queries llm_inference items."""
        import json as _json
        from unittest.mock import MagicMock, patch

        from alibi.enrichment.cloud_enrichment import refine_categories_by_cloud

        # Seed an item with enrichment_source='cloud_api' (not llm_inference)
        _seed_fact_item(
            db,
            item_id="cloud-api-item",
            barcode=None,
            brand="Acme",
            category="Snacks",
            doc_id="doc-cloud-api",
            fact_id="fact-cloud-api",
            atom_id="atom-cloud-api",
            cloud_id="cloud-cloud-api",
        )
        # Set cloud_api source
        conn = db.get_connection()
        conn.execute(
            "UPDATE fact_items SET enrichment_source = 'cloud_api', "
            "enrichment_confidence = 0.85 WHERE id = 'cloud-api-item'"
        )
        conn.commit()

        api_payload = {
            "content": [
                {
                    "type": "text",
                    "text": _json.dumps({"items": []}),
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = api_payload
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.post", return_value=mock_response) as mock_post:
            results = refine_categories_by_cloud(db, api_key="test-key")

        # No items matched the llm_inference filter → API should not have been called
        assert mock_post.call_count == 0
        assert results == []

    # ------------------------------------------------------------------
    # test_refine_uses_config_model
    # ------------------------------------------------------------------

    def test_refine_uses_config_model(self, db):
        """refine_categories_by_cloud resolves model from Config when not specified."""
        import json as _json
        from unittest.mock import MagicMock, patch

        from alibi.enrichment.cloud_enrichment import refine_categories_by_cloud

        _seed_llm_inferred_item(
            db,
            item_id="refine-item-4",
            name="Red Pepper",
            category="Dairy",  # wrong
        )

        api_payload = {
            "content": [
                {
                    "type": "text",
                    "text": _json.dumps(
                        {
                            "items": [
                                {
                                    "idx": 1,
                                    "corrected_category": "Vegetables",
                                    "reason": "pepper is a vegetable",
                                }
                            ]
                        }
                    ),
                }
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = api_payload
        mock_response.raise_for_status = MagicMock()

        mock_config = MagicMock()
        mock_config.cloud_enrichment_model = "claude-sonnet-4-6"
        mock_config.anthropic_api_key = None  # force api_key from argument

        with patch("alibi.config.get_config", return_value=mock_config):
            with patch("httpx.post", return_value=mock_response) as mock_post:
                refine_categories_by_cloud(db, api_key="test-key")

        sent_json = mock_post.call_args.kwargs.get("json") or mock_post.call_args[
            1
        ].get("json")
        assert sent_json["model"] == "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Product variant & attribute extraction
# ---------------------------------------------------------------------------


class TestExtractProductVariant:
    """Tests for _extract_product_variant (OFF data)."""

    def test_dairy_fat_percentage(self) -> None:
        from alibi.enrichment.service import _extract_product_variant

        product = {
            "nutriments": {"fat_100g": 3.5},
            "categories_tags": ["en:milk", "en:dairy"],
        }
        assert _extract_product_variant(product) == "3.5%"

    def test_no_variant_for_non_dairy(self) -> None:
        from alibi.enrichment.service import _extract_product_variant

        product = {
            "nutriments": {"fat_100g": 15.0},
            "categories_tags": ["en:snacks"],
        }
        assert _extract_product_variant(product) is None

    def test_no_nutriments(self) -> None:
        from alibi.enrichment.service import _extract_product_variant

        assert _extract_product_variant({}) is None


class TestExtractProductAttributes:
    """Tests for extract_product_attributes (secondary attributes)."""

    def test_off_organic_label(self) -> None:
        from alibi.enrichment.service import extract_product_attributes

        product = {"labels_tags": ["en:organic", "en:eu-organic"]}
        attrs = extract_product_attributes(product=product)
        assert "organic" in attrs
        assert attrs.count("organic") == 1  # no duplicates

    def test_name_based_free_range(self) -> None:
        from alibi.enrichment.service import extract_product_attributes

        attrs = extract_product_attributes(item_name="ALPHAMEGA FREE RANGE EGGS")
        assert "free-range" in attrs

    def test_name_based_lactose_free(self) -> None:
        from alibi.enrichment.service import extract_product_attributes

        attrs = extract_product_attributes(item_name="CHRISTIS FRESH DELACT MILK")
        assert "lactose-free" in attrs

    def test_combined_off_and_name(self) -> None:
        from alibi.enrichment.service import extract_product_attributes

        product = {"labels_tags": ["en:organic"]}
        attrs = extract_product_attributes(
            product=product, item_name="ORGANIC WHOLEGRAIN BREAD"
        )
        assert "organic" in attrs
        assert "wholegrain" in attrs
        assert attrs.count("organic") == 1

    def test_no_attributes(self) -> None:
        from alibi.enrichment.service import extract_product_attributes

        attrs = extract_product_attributes(item_name="BANANA")
        assert attrs == []


class TestExtractVariantFromName:
    """Tests for extract_variant_from_name (primary variant heuristic)."""

    def test_dairy_fat_percent(self) -> None:
        from alibi.enrichment.service import extract_variant_from_name

        assert extract_variant_from_name("MILK 3.6%", "Dairy") == "3.6%"
        assert extract_variant_from_name("TVOROG 9%", "Dairy") == "9%"
        assert extract_variant_from_name("SMETANA 20%", "Dairy") == "20%"

    def test_egg_size_word(self) -> None:
        from alibi.enrichment.service import extract_variant_from_name

        assert extract_variant_from_name("FRESH EGGS LARGE X12", "Eggs") == "L"
        assert extract_variant_from_name("EGGS MEDIUM", "Eggs") == "M"

    def test_egg_size_letter(self) -> None:
        from alibi.enrichment.service import extract_variant_from_name

        assert extract_variant_from_name("TOUKAIDES L X12", "Eggs") == "L"

    def test_light_dairy(self) -> None:
        from alibi.enrichment.service import extract_variant_from_name

        assert extract_variant_from_name("YOGHURT LIGHT", "Dairy") == "light"

    def test_strained_yoghurt(self) -> None:
        from alibi.enrichment.service import extract_variant_from_name

        assert extract_variant_from_name("CHRISTIS ST.YOGHURT", "Dairy") == "strained"

    def test_unsalted_butter(self) -> None:
        from alibi.enrichment.service import extract_variant_from_name

        assert (
            extract_variant_from_name("KERRYGOLD UNSALTED BUTTER", "Dairy")
            == "unsalted"
        )

    def test_prune_size(self) -> None:
        from alibi.enrichment.service import extract_variant_from_name

        assert extract_variant_from_name("PRUNES 20/30", "Fruit") == "20/30"

    def test_no_variant_wrong_category(self) -> None:
        from alibi.enrichment.service import extract_variant_from_name

        # % in non-dairy category should not match
        assert extract_variant_from_name("SAUCE 20%", "Condiments") is None

    def test_no_variant_plain_name(self) -> None:
        from alibi.enrichment.service import extract_variant_from_name

        assert extract_variant_from_name("BANANA", "Fruit") is None
