"""EUR-normalisation of the monthly_report service aggregation.

monthly_report previously summed raw ``total_amount`` across currencies, so a
month mixing EUR with RUB/CAD/TRY facts blended currencies. These tests lock in
the eur_rate-weighted totals (the same convention as
``alibi.analytics.spending.eur_amount``).
"""

from __future__ import annotations

import os

os.environ["ALIBI_TESTING"] = "1"

from alibi.services import analytics


def _seed_fact(
    db,
    fact_id: str,
    *,
    vendor: str,
    total: float,
    currency: str = "EUR",
    eur_rate: float | None = None,
    fact_type: str = "purchase",
    event_date: str = "2026-03-15",
) -> None:
    cloud_id = f"cloud-{fact_id}"
    conn = db.get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO clouds (id, status) VALUES (?, 'collapsed')",
        (cloud_id,),
    )
    conn.execute(
        "INSERT INTO facts "
        "(id, cloud_id, fact_type, vendor, total_amount, currency, eur_rate, "
        " event_date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (fact_id, cloud_id, fact_type, vendor, total, currency, eur_rate, event_date),
    )
    conn.commit()


class TestMonthlyReportEur:
    def test_mixed_currency_total_is_eur_weighted(self, db):
        # EUR 100 + RUB 1000 @ 0.01 = EUR 10  ->  EUR 110, not raw 1100.
        _seed_fact(db, "f-eur", vendor="Shop A", total=100.0)
        _seed_fact(
            db, "f-rub", vendor="Shop B", total=1000.0, currency="RUB", eur_rate=0.01
        )
        report = analytics.monthly_report(db, 2026, 3)
        assert report["expenses"]["count"] == 2
        assert report["expenses"]["total"] == 110.0

    def test_eur_fact_without_rate_counts_one_to_one(self, db):
        # No eur_rate, currency EUR -> 1:1 (no backfill needed for euros).
        _seed_fact(db, "f1", vendor="Shop", total=42.5, eur_rate=None)
        report = analytics.monthly_report(db, 2026, 3)
        assert report["expenses"]["total"] == 42.5

    def test_foreign_fact_without_rate_dropped_from_total(self, db):
        # A foreign fact with no rate yet must not be summed as if it were euros.
        _seed_fact(db, "f-eur", vendor="Shop A", total=50.0)
        _seed_fact(
            db, "f-try", vendor="Shop B", total=900.0, currency="TRY", eur_rate=None
        )
        report = analytics.monthly_report(db, 2026, 3)
        # Total reflects only the convertible euro fact; count stays a tx count.
        assert report["expenses"]["total"] == 50.0
        assert report["expenses"]["count"] == 2

    def test_top_vendors_total_is_eur_weighted(self, db):
        _seed_fact(
            db,
            "f-rub",
            vendor="Foreign Mart",
            total=2000.0,
            currency="RUB",
            eur_rate=0.01,
        )
        report = analytics.monthly_report(db, 2026, 3)
        vendors = {v["vendor"]: v for v in report["top_vendors"]}
        assert vendors["Foreign Mart"]["total"] == 20.0

    def test_refund_income_is_eur_weighted(self, db):
        _seed_fact(
            db,
            "r1",
            vendor="Shop",
            total=500.0,
            currency="RUB",
            eur_rate=0.01,
            fact_type="refund",
        )
        report = analytics.monthly_report(db, 2026, 3)
        assert report["income"]["total"] == 5.0
