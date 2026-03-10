"""Tests for barcode cascade enrichment — UPCitemdb, GS1, OFF contribution, cascade."""

from __future__ import annotations

import os
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

os.environ["ALIBI_TESTING"] = "1"

import alibi.enrichment.upcitemdb_client as upcitemdb_client
from alibi.enrichment import off_client
from alibi.enrichment.gs1_client import decode_gs1_prefix, lookup_brand_by_prefix
from alibi.enrichment.off_client import get_cached, store_cache
from alibi.enrichment.service import EnrichmentResult, enrich_item_cascade
from alibi.enrichment.upcitemdb_client import (
    cached_lookup as upc_cached_lookup,
    lookup_barcode as upc_lookup_barcode,
)

# ---------------------------------------------------------------------------
# Test barcodes
# ---------------------------------------------------------------------------

NUTELLA_BARCODE = "3017624010701"  # French prefix 301
GREEK_BARCODE = "5200000000003"  # Greek prefix 520
CYPRIOT_BARCODE = "5290000000002"  # Cypriot prefix 529
GERMAN_BARCODE = "4000000000008"  # German prefix 400
UNKNOWN_BARCODE = "9999999999999"  # No matching prefix
SHORT_BARCODE = "123"  # Too short

SAMPLE_OFF_PRODUCT = {
    "product_name": "Nutella",
    "brands": "Ferrero",
    "categories": "Breakfasts, Spreads, Hazelnut spreads",
    "categories_tags": ["en:breakfasts", "en:spreads", "en:hazelnut-spreads"],
    "quantity": "400 g",
}

SAMPLE_UPC_ITEM = {
    "title": "Pringles Original",
    "brand": "Pringles",
    "category": "Snacks",
    "description": "Original flavour crisps",
    "ean": "5053990107803",
}

SAMPLE_UPC_PRODUCT = {
    "product_name": "Pringles Original",
    "brands": "Pringles",
    "categories": "Snacks",
    "description": "Original flavour crisps",
    "ean": "5053990107803",
}

UPC_TEST_BARCODE = "5053990107803"


# ---------------------------------------------------------------------------
# DB seeding helper — mirrors _seed_fact_item from test_enrichment.py
# ---------------------------------------------------------------------------


def _seed_fact_item(
    db,
    item_id: str = "item-1",
    barcode: str | None = None,
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


# ---------------------------------------------------------------------------
# UPCitemdb helpers
# ---------------------------------------------------------------------------


def _make_upc_mock_client(status_code: int, json_payload: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_payload
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    return mock_client


def _reset_upc_rate_limit() -> None:
    """Reset UPCitemdb module-level rate limit state."""
    upcitemdb_client._daily_count = 0
    upcitemdb_client._daily_date = date.today().isoformat()


# ===========================================================================
# UPCitemdb client tests
# ===========================================================================


@patch("alibi.enrichment.upcitemdb_client.httpx.Client")
def test_upcitemdb_lookup_success(mock_client_cls):
    """Successful API response returns normalized product dict."""
    _reset_upc_rate_limit()
    mock_client_cls.return_value = _make_upc_mock_client(
        200, {"items": [SAMPLE_UPC_ITEM], "code": "OK"}
    )

    result = upc_lookup_barcode(UPC_TEST_BARCODE)

    assert result is not None
    assert result["product_name"] == "Pringles Original"
    assert result["brands"] == "Pringles"
    assert result["categories"] == "Snacks"


@patch("alibi.enrichment.upcitemdb_client.httpx.Client")
def test_upcitemdb_lookup_not_found(mock_client_cls):
    """404 response returns None."""
    _reset_upc_rate_limit()
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    mock_client = MagicMock()
    mock_client.get.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client_cls.return_value = mock_client

    result = upc_lookup_barcode(UNKNOWN_BARCODE)

    assert result is None


@patch("alibi.enrichment.upcitemdb_client.httpx.Client")
def test_upcitemdb_lookup_api_error(mock_client_cls):
    """500 server error returns None."""
    import httpx

    _reset_upc_rate_limit()
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

    result = upc_lookup_barcode(UPC_TEST_BARCODE)

    assert result is None


@patch("alibi.enrichment.upcitemdb_client.httpx.Client")
def test_upcitemdb_rate_limit_blocks(mock_client_cls):
    """When daily counter is at limit, returns None without making an HTTP call."""
    upcitemdb_client._daily_count = upcitemdb_client._DAILY_LIMIT
    upcitemdb_client._daily_date = date.today().isoformat()

    result = upc_lookup_barcode(UPC_TEST_BARCODE)

    assert result is None
    mock_client_cls.assert_not_called()


@patch("alibi.enrichment.upcitemdb_client.httpx.Client")
def test_upcitemdb_rate_limit_resets_daily(mock_client_cls):
    """Counter from a previous day is reset when a new day is detected."""
    _reset_upc_rate_limit()
    mock_client_cls.return_value = _make_upc_mock_client(
        200, {"items": [SAMPLE_UPC_ITEM], "code": "OK"}
    )
    # Simulate counter pinned at limit from yesterday
    upcitemdb_client._daily_count = upcitemdb_client._DAILY_LIMIT
    upcitemdb_client._daily_date = "2000-01-01"

    # The new call should detect the date change and reset
    result = upc_lookup_barcode(UPC_TEST_BARCODE)

    assert result is not None
    assert upcitemdb_client._daily_count == 1
    assert upcitemdb_client._daily_date == date.today().isoformat()


@patch("alibi.enrichment.upcitemdb_client.httpx.Client")
def test_upcitemdb_cached_lookup_cache_hit(mock_client_cls, db):
    """If product_cache already has real data, no HTTP call is made."""
    _reset_upc_rate_limit()
    store_cache(db, UPC_TEST_BARCODE, SAMPLE_UPC_PRODUCT, source="upcitemdb")

    result = upc_cached_lookup(db, UPC_TEST_BARCODE)

    assert result is not None
    assert result["product_name"] == "Pringles Original"
    mock_client_cls.assert_not_called()


@patch("alibi.enrichment.upcitemdb_client.httpx.Client")
def test_upcitemdb_cached_lookup_off_negative_still_tries(mock_client_cls, db):
    """A _not_found entry from source='negative' (OFF) does NOT block UPCitemdb."""
    _reset_upc_rate_limit()
    # OFF cached a negative result
    store_cache(db, UPC_TEST_BARCODE, {"_not_found": True}, source="negative")

    mock_client_cls.return_value = _make_upc_mock_client(
        200, {"items": [SAMPLE_UPC_ITEM], "code": "OK"}
    )

    result = upc_cached_lookup(db, UPC_TEST_BARCODE)

    assert result is not None
    assert result["product_name"] == "Pringles Original"
    mock_client_cls.assert_called_once()


@patch("alibi.enrichment.upcitemdb_client.httpx.Client")
def test_upcitemdb_cached_lookup_own_negative_skips(mock_client_cls, db):
    """A _not_found entry from source='upcitemdb' prevents another API call."""
    _reset_upc_rate_limit()
    store_cache(db, UPC_TEST_BARCODE, {"_not_found": True}, source="upcitemdb")

    result = upc_cached_lookup(db, UPC_TEST_BARCODE)

    assert result is None
    mock_client_cls.assert_not_called()


# ===========================================================================
# GS1 client tests
# ===========================================================================


def test_gs1_decode_prefix_greece():
    """Barcode starting with '520' maps to Greece."""
    result = decode_gs1_prefix(GREEK_BARCODE)

    assert result is not None
    assert result["country"] == "Greece"
    assert result["is_local"] is False


def test_gs1_decode_prefix_cyprus():
    """Barcode starting with '529' maps to Cyprus and is marked local."""
    result = decode_gs1_prefix(CYPRIOT_BARCODE)

    assert result is not None
    assert result["country"] == "Cyprus"
    assert result["is_local"] is True


def test_gs1_decode_prefix_germany():
    """Barcode starting with '400' maps to Germany."""
    result = decode_gs1_prefix(GERMAN_BARCODE)

    assert result is not None
    assert result["country"] == "Germany"
    assert result["is_local"] is False


def test_gs1_decode_prefix_unknown():
    """A prefix with no known country mapping returns None."""
    result = decode_gs1_prefix(UNKNOWN_BARCODE)

    assert result is None


def test_gs1_decode_short_barcode():
    """Barcode shorter than 8 characters returns None."""
    result = decode_gs1_prefix(SHORT_BARCODE)

    assert result is None


def test_gs1_brand_propagation(db):
    """Existing fact_item with same company prefix propagates its brand."""
    # Seed an item with same prefix (5290000) as CYPRIOT_BARCODE
    _seed_fact_item(
        db,
        item_id="item-known",
        barcode="5290000111112",
        brand="Halloumi Co",
        doc_id="doc-known",
        fact_id="fact-known",
        atom_id="atom-known",
        cloud_id="cloud-known",
    )

    # Look up a different barcode with the same company prefix
    result = lookup_brand_by_prefix(db, "5290000999991")

    assert result is not None
    assert result["brands"] == "Halloumi Co"
    assert result["country"] == "Cyprus"
    assert result["is_local"] is True


def test_gs1_brand_propagation_no_match(db):
    """No items with the same prefix returns None."""
    # No items in DB share the prefix 5290001
    result = lookup_brand_by_prefix(db, "5290001000001")

    assert result is None


# ===========================================================================
# OFF contribution tests
# ===========================================================================


def _make_off_mock_client(status_code: int, json_payload: dict) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_payload
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.post.return_value = mock_resp
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    return mock_client


@patch("alibi.enrichment.off_client.httpx.Client")
def test_off_contribute_product_success(mock_client_cls):
    """200 + status=1 returns True."""
    off_client._last_request_time = 0.0
    mock_client_cls.return_value = _make_off_mock_client(200, {"status": 1})

    result = off_client.contribute_product(
        NUTELLA_BARCODE, "Ferrero", "Hazelnut spreads", "Nutella"
    )

    assert result is True


@patch("alibi.enrichment.off_client.httpx.Client")
def test_off_contribute_product_failure(mock_client_cls):
    """200 + status=0 (rejection) returns False."""
    off_client._last_request_time = 0.0
    mock_client_cls.return_value = _make_off_mock_client(
        200, {"status": 0, "status_verbose": "fields saved"}
    )

    result = off_client.contribute_product(
        NUTELLA_BARCODE, "Ferrero", "Hazelnut spreads"
    )

    assert result is False


@patch("alibi.enrichment.off_client.contribute_product")
def test_off_contribute_if_enabled_gate(mock_contribute, db):
    """When off_contribution_enabled is False the gate returns False immediately."""
    from alibi.config import get_config

    cfg = get_config()
    # off_contribution_enabled should be False in test env
    assert not getattr(cfg, "off_contribution_enabled", False)

    result = off_client.contribute_if_enabled(
        db, NUTELLA_BARCODE, "Ferrero", "Hazelnut spreads", "Nutella"
    )

    assert result is False
    mock_contribute.assert_not_called()


@patch("alibi.enrichment.off_client.contribute_product")
def test_off_contribute_if_enabled_already_known(mock_contribute, db):
    """If product_cache already has a real entry, contribution is skipped."""
    # Patch config to enable contributions — get_config is imported locally inside
    # contribute_if_enabled so we patch it at the source module level.
    with patch("alibi.config.get_config") as mock_cfg:
        cfg_obj = MagicMock()
        cfg_obj.off_contribution_enabled = True
        mock_cfg.return_value = cfg_obj

        # A real product exists — not a negative entry
        store_cache(db, NUTELLA_BARCODE, SAMPLE_OFF_PRODUCT, source="openfoodfacts")

        result = off_client.contribute_if_enabled(
            db, NUTELLA_BARCODE, "Ferrero", "Hazelnut spreads", "Nutella"
        )

    assert result is False
    mock_contribute.assert_not_called()


# ===========================================================================
# Cascade integration tests
# ===========================================================================


@patch("alibi.enrichment.off_client.cached_lookup")
def test_enrich_item_cascade_off_hit(mock_off_lookup, db):
    """OFF finds product — uses OFF, confidence 0.95."""
    _seed_fact_item(
        db,
        item_id="item-c1",
        barcode=NUTELLA_BARCODE,
        doc_id="doc-c1",
        fact_id="fact-c1",
        atom_id="atom-c1",
        cloud_id="cloud-c1",
    )
    mock_off_lookup.return_value = SAMPLE_OFF_PRODUCT

    result = enrich_item_cascade(db, "item-c1", NUTELLA_BARCODE)

    assert isinstance(result, EnrichmentResult)
    assert result.success is True
    assert result.source == "openfoodfacts"
    assert result.brand == "Ferrero"

    row = db.fetchone(
        "SELECT enrichment_source, enrichment_confidence FROM fact_items WHERE id = 'item-c1'"
    )
    assert row["enrichment_source"] == "openfoodfacts"
    assert abs(row["enrichment_confidence"] - 0.95) < 0.01


@patch("alibi.enrichment.upcitemdb_client.cached_lookup")
@patch("alibi.enrichment.off_client.cached_lookup")
def test_enrich_item_cascade_off_miss_upc_hit(mock_off_lookup, mock_upc_lookup, db):
    """OFF misses, UPCitemdb finds product — uses UPCitemdb, confidence 0.90."""
    _seed_fact_item(
        db,
        item_id="item-c2",
        barcode=UPC_TEST_BARCODE,
        doc_id="doc-c2",
        fact_id="fact-c2",
        atom_id="atom-c2",
        cloud_id="cloud-c2",
    )
    mock_off_lookup.return_value = None
    mock_upc_lookup.return_value = SAMPLE_UPC_PRODUCT

    result = enrich_item_cascade(db, "item-c2", UPC_TEST_BARCODE)

    assert isinstance(result, EnrichmentResult)
    assert result.success is True
    assert result.source == "upcitemdb"
    assert result.brand == "Pringles"

    row = db.fetchone(
        "SELECT enrichment_source, enrichment_confidence FROM fact_items WHERE id = 'item-c2'"
    )
    assert row["enrichment_source"] == "upcitemdb"
    assert abs(row["enrichment_confidence"] - 0.90) < 0.01


@patch("alibi.enrichment.upcitemdb_client.cached_lookup")
@patch("alibi.enrichment.off_client.cached_lookup")
def test_enrich_item_cascade_both_miss_gs1_hit(mock_off_lookup, mock_upc_lookup, db):
    """Both OFF and UPCitemdb miss; GS1 prefix propagates brand — confidence 0.80."""
    # Seed a known item with brand for the same company prefix
    _seed_fact_item(
        db,
        item_id="item-known-gs1",
        barcode="5290000111112",
        brand="Halloumi Co",
        doc_id="doc-gs1-known",
        fact_id="fact-gs1-known",
        atom_id="atom-gs1-known",
        cloud_id="cloud-gs1-known",
    )

    target_barcode = "5290000999111"
    _seed_fact_item(
        db,
        item_id="item-c3",
        barcode=target_barcode,
        doc_id="doc-c3",
        fact_id="fact-c3",
        atom_id="atom-c3",
        cloud_id="cloud-c3",
    )
    mock_off_lookup.return_value = None
    mock_upc_lookup.return_value = None

    result = enrich_item_cascade(db, "item-c3", target_barcode)

    assert isinstance(result, EnrichmentResult)
    assert result.success is True
    assert result.source == "gs1"
    assert result.brand == "Halloumi Co"

    row = db.fetchone(
        "SELECT enrichment_source, enrichment_confidence FROM fact_items WHERE id = 'item-c3'"
    )
    assert row["enrichment_source"] == "gs1"
    assert abs(row["enrichment_confidence"] - 0.80) < 0.01


@patch("alibi.enrichment.upcitemdb_client.cached_lookup")
@patch("alibi.enrichment.off_client.cached_lookup")
def test_enrich_item_cascade_all_miss(mock_off_lookup, mock_upc_lookup, db):
    """All three stages miss — returns EnrichmentResult with success=False."""
    _seed_fact_item(
        db,
        item_id="item-c4",
        barcode=UNKNOWN_BARCODE,
        doc_id="doc-c4",
        fact_id="fact-c4",
        atom_id="atom-c4",
        cloud_id="cloud-c4",
    )
    mock_off_lookup.return_value = None
    mock_upc_lookup.return_value = None
    # No items in DB share a prefix with UNKNOWN_BARCODE (prefix 999)

    result = enrich_item_cascade(db, "item-c4", UNKNOWN_BARCODE)

    assert isinstance(result, EnrichmentResult)
    assert result.success is False
    assert result.brand is None
