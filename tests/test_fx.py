"""Tests for historical currency -> EUR conversion (alibi.services.fx)."""

from __future__ import annotations

import os
from unittest.mock import patch

os.environ["ALIBI_TESTING"] = "1"

from alibi.services.fx import backfill_fact_rates, get_eur_per_unit, stamp_fact_rate


def _seed_fact(db, fact_id, *, currency="EUR", event_date="2025-06-02", total=10.0):
    cloud_id = f"cloud-{fact_id}"
    conn = db.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')",
        (cloud_id,),
    )
    conn.execute(
        "INSERT OR IGNORE INTO facts "
        "(id, cloud_id, fact_type, vendor, total_amount, currency, event_date) "
        "VALUES (?, ?, 'purchase', 'V', ?, ?, ?)",
        (fact_id, cloud_id, total, currency, event_date),
    )
    conn.commit()


def _cache_rate(db, base, date_str, rate):
    with db.transaction() as cur:
        cur.execute(
            "INSERT OR REPLACE INTO exchange_rates "
            "(base, rate_date, eur_per_unit, fetched_at) VALUES (?, ?, ?, '')",
            (base, date_str, rate),
        )


class TestGetEurPerUnit:
    def test_eur_is_identity(self, db):
        assert get_eur_per_unit(db, "EUR", "2025-06-02") == 1.0
        assert get_eur_per_unit(db, None, "2025-06-02") == 1.0

    def test_non_eur_without_date_is_none(self, db):
        assert get_eur_per_unit(db, "CAD", None) is None

    def test_cache_hit_no_fetch(self, db):
        _cache_rate(db, "CAD", "2025-06-02", 0.68)
        # fetch=False must still find the cached rate
        assert get_eur_per_unit(db, "CAD", "2025-06-02", fetch=False) == 0.68

    def test_uncached_no_fetch_is_none(self, db):
        assert get_eur_per_unit(db, "TRY", "2025-06-02", fetch=False) is None

    @patch("alibi.services.fx._fetch_frankfurter", return_value=0.028)
    def test_fetch_caches(self, mock_fetch, db):
        rate = get_eur_per_unit(db, "TRY", "2025-05-13")
        assert rate == 0.028
        mock_fetch.assert_called_once_with("TRY", "2025-05-13")
        # second call is served from cache (no second fetch)
        assert get_eur_per_unit(db, "TRY", "2025-05-13") == 0.028
        assert mock_fetch.call_count == 1

    @patch("alibi.services.fx._fetch_cbr", return_value=None)
    @patch("alibi.services.fx._fetch_frankfurter", return_value=None)
    def test_fetch_failure_is_none(self, mock_fetch, mock_cbr, db):
        assert get_eur_per_unit(db, "TRY", "2025-05-13") is None

    @patch("alibi.services.fx._fetch_cbr", return_value=0.0109)
    @patch("alibi.services.fx._fetch_frankfurter", return_value=None)
    def test_falls_back_to_cbr_when_frankfurter_missing(self, mock_fr, mock_cbr, db):
        """RUB: ECB dropped it, so the CBR fallback resolves and caches it."""
        rate = get_eur_per_unit(db, "RUB", "2025-11-27")
        assert rate == 0.0109
        mock_fr.assert_called_once_with("RUB", "2025-11-27")
        mock_cbr.assert_called_once_with("RUB", "2025-11-27")
        # cached -> neither source consulted again
        assert get_eur_per_unit(db, "RUB", "2025-11-27") == 0.0109
        assert mock_cbr.call_count == 1

    @patch("alibi.services.fx._fetch_cbr")
    @patch("alibi.services.fx._fetch_frankfurter", return_value=0.68)
    def test_cbr_not_consulted_when_frankfurter_succeeds(self, mock_fr, mock_cbr, db):
        assert get_eur_per_unit(db, "CAD", "2025-06-02") == 0.68
        mock_cbr.assert_not_called()


# windows-1251 CBR sample (ASCII-only content, comma decimals, Nominal!=1 case)
_CBR_XML = (
    b"<?xml version='1.0' encoding='windows-1251'?>"
    b'<ValCurs Date="27.11.2025" name="Foreign Currency Market">'
    b'<Valute ID="R01239"><CharCode>EUR</CharCode><Nominal>1</Nominal>'
    b"<Value>91,5047</Value></Valute>"
    b'<Valute ID="R01235"><CharCode>USD</CharCode><Nominal>1</Nominal>'
    b"<Value>78,5941</Value></Valute>"
    b'<Valute ID="R01239X"><CharCode>JPY</CharCode><Nominal>100</Nominal>'
    b"<Value>50,2000</Value></Valute>"
    b"</ValCurs>"
)


class _FakeResp:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self) -> None:
        return None


class _FakeClient:
    def __init__(self, content: bytes):
        self._content = content

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, *a, **k):
        return _FakeResp(self._content)


class TestFetchCbr:
    def _patch(self, content: bytes):
        return patch(
            "alibi.services.fx.httpx.Client", return_value=_FakeClient(content)
        )

    def test_rub_self_rate(self):
        from alibi.services.fx import _fetch_cbr

        with self._patch(_CBR_XML):
            rate = _fetch_cbr("RUB", "2025-11-27")
        # EUR per 1 RUB = 1 / 91.5047
        assert rate is not None
        assert abs(rate - 1 / 91.5047) < 1e-9

    def test_usd_cross_rate(self):
        from alibi.services.fx import _fetch_cbr

        with self._patch(_CBR_XML):
            rate = _fetch_cbr("USD", "2025-11-27")
        # EUR per 1 USD = (RUB/USD) / (RUB/EUR) = 78.5941 / 91.5047
        assert abs(rate - 78.5941 / 91.5047) < 1e-9

    def test_nominal_scaling(self):
        from alibi.services.fx import _fetch_cbr

        with self._patch(_CBR_XML):
            rate = _fetch_cbr("JPY", "2025-11-27")
        # JPY Nominal is 100, so RUB per 1 JPY = 50.2/100, EUR per JPY divides EUR
        assert abs(rate - (50.2000 / 100) / 91.5047) < 1e-9

    def test_unknown_currency_is_none(self):
        from alibi.services.fx import _fetch_cbr

        with self._patch(_CBR_XML):
            assert _fetch_cbr("ZWL", "2025-11-27") is None

    def test_dtd_is_refused(self):
        from alibi.services.fx import _fetch_cbr

        evil = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]>' + _CBR_XML
        with self._patch(evil):
            assert _fetch_cbr("RUB", "2025-11-27") is None


class TestBackfillFactRates:
    def test_eur_facts_seeded_to_one(self, db):
        _seed_fact(db, "e", currency="EUR")
        stats = backfill_fact_rates(db, fetch=False)
        row = db.fetchone("SELECT eur_rate FROM facts WHERE id='e'")
        assert row["eur_rate"] == 1.0
        assert stats["pairs_resolved"] == 0  # no non-EUR pairs

    def test_non_eur_resolved_from_cache(self, db):
        _seed_fact(db, "c", currency="CAD", event_date="2025-06-02")
        _cache_rate(db, "CAD", "2025-06-02", 0.68)
        stats = backfill_fact_rates(db, fetch=False)
        row = db.fetchone("SELECT eur_rate FROM facts WHERE id='c'")
        assert abs(row["eur_rate"] - 0.68) < 1e-9
        assert stats["pairs_resolved"] == 1
        assert stats["currencies"] == ["CAD"]

    def test_unresolved_stays_null(self, db):
        _seed_fact(db, "t", currency="TRY", event_date="2025-05-13")  # no cached rate
        stats = backfill_fact_rates(db, fetch=False)
        row = db.fetchone("SELECT eur_rate FROM facts WHERE id='t'")
        assert row["eur_rate"] is None
        assert stats["pairs_unresolved"] == 1
        assert stats["facts_unconverted"] == 1


class TestStampFactRate:
    """Per-fact eur_rate stamping used by the ingestion finalizer."""

    def test_eur_fact_stamped_one_without_network(self, db):
        _seed_fact(db, "e", currency="EUR")
        # fetch=False proves EUR needs no network to resolve.
        rate = stamp_fact_rate(db, "e", fetch=False)
        assert rate == 1.0
        row = db.fetchone("SELECT eur_rate FROM facts WHERE id='e'")
        assert row["eur_rate"] == 1.0

    def test_foreign_fact_stamped_from_cache(self, db):
        _seed_fact(db, "c", currency="CAD", event_date="2025-06-02")
        _cache_rate(db, "CAD", "2025-06-02", 0.68)
        rate = stamp_fact_rate(db, "c", fetch=False)
        assert abs(rate - 0.68) < 1e-9
        row = db.fetchone("SELECT eur_rate FROM facts WHERE id='c'")
        assert abs(row["eur_rate"] - 0.68) < 1e-9

    def test_unresolved_foreign_left_null(self, db):
        _seed_fact(db, "t", currency="TRY", event_date="2025-05-13")
        rate = stamp_fact_rate(db, "t", fetch=False)
        assert rate is None
        row = db.fetchone("SELECT eur_rate FROM facts WHERE id='t'")
        assert row["eur_rate"] is None

    def test_missing_fact_is_none(self, db):
        assert stamp_fact_rate(db, "does-not-exist", fetch=False) is None


class TestItemStarsEurMaterialisation:
    """The rebuild multiplies item amounts by the fact's eur_rate (1.0 for EUR)."""

    def _seed_item(self, db, fact_id, *, currency, total_price, comparable=2.0):
        _seed_fact(db, fact_id, currency=currency)
        doc_id = f"doc-{fact_id}"
        atom_id = f"atom-{fact_id}"
        conn = db.get_connection()
        conn.execute(
            "INSERT OR IGNORE INTO documents (id, file_path, file_hash) "
            "VALUES (?, ?, ?)",
            (doc_id, f"/tmp/{doc_id}", f"h{fact_id}"),
        )
        conn.execute(
            "INSERT OR IGNORE INTO atoms (id, document_id, atom_type, data) "
            "VALUES (?, ?, 'item', '{}')",
            (atom_id, doc_id),
        )
        conn.execute(
            "INSERT OR IGNORE INTO fact_items "
            "(id, fact_id, atom_id, name, total_price, comparable_unit_price, "
            " comparable_unit) VALUES (?, ?, ?, 'X', ?, ?, 'kg')",
            (f"item-{fact_id}", fact_id, atom_id, total_price, comparable),
        )
        conn.commit()

    def test_eur_item_is_identity(self, db):
        from alibi.services import rebuild_item_stars

        self._seed_item(db, "e", currency="EUR", total_price=10.0, comparable=2.0)
        rebuild_item_stars(db)
        row = db.fetchone(
            "SELECT total_price_eur, comparable_unit_price_eur FROM item_stars "
            "WHERE item_id='item-e'"
        )
        assert abs(row["total_price_eur"] - 10.0) < 1e-9
        assert abs(row["comparable_unit_price_eur"] - 2.0) < 1e-9

    def test_cad_item_converted_after_backfill(self, db):
        from alibi.services import rebuild_item_stars

        self._seed_item(db, "c", currency="CAD", total_price=10.0, comparable=2.0)
        _cache_rate(db, "CAD", "2025-06-02", 0.5)
        backfill_fact_rates(db, fetch=False)  # stamps facts.eur_rate = 0.5
        rebuild_item_stars(db)
        row = db.fetchone(
            "SELECT total_price_eur, comparable_unit_price_eur FROM item_stars "
            "WHERE item_id='item-c'"
        )
        assert abs(row["total_price_eur"] - 5.0) < 1e-9  # 10 * 0.5
        assert abs(row["comparable_unit_price_eur"] - 1.0) < 1e-9  # 2 * 0.5

    def test_cad_item_unconverted_is_null(self, db):
        from alibi.services import rebuild_item_stars

        self._seed_item(db, "u", currency="CAD", total_price=10.0)
        rebuild_item_stars(db)  # no backfill -> eur_rate NULL -> _eur NULL
        row = db.fetchone(
            "SELECT total_price_eur FROM item_stars WHERE item_id='item-u'"
        )
        assert row["total_price_eur"] is None
