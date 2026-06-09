"""Item ↔ payment reconciliation view.

The pipeline decomposes every document to atoms: a receipt yields an *item*
layer (line items) and a POS/card slip yields a *payment* layer. Cloud
formation already overlays the two when they describe one transaction, so a
fact may be backed by a receipt bundle, a payment bundle, or both.

This service makes that overlap queryable. It classifies each transaction by
which layers cover it (``matched`` / ``items_only`` / ``payment_only`` /
``empty``) and reconciles three amounts — the fact total, the sum of line
items, and the **normalised payment amount** (the persisted payment-atom
``amount`` from ``facts.payments``) — flagging where they disagree.

Two coverage classes are actionable worklists:
- ``payment_only``: a card charge with no itemised receipt captured.
- ``items_only``: a receipt never matched to a payment (often cash).

Filters reuse the fact-level axes of the A item filter (vendor, vendor_key,
currency, country, date and date+time ranges). Item-level axes (name, brand,
category, price) are intentionally NOT applied: reconciliation operates on
transactions, and an item predicate would silently drop every ``payment_only``
row (which has no items). Pass ``coverage`` to return a single class.
"""

from __future__ import annotations

import json
from typing import Any

from alibi.db.connection import DatabaseManager

# Coverage classes for a transaction (fact).
MATCHED = "matched"  # has line items AND a payment record
ITEMS_ONLY = "items_only"  # itemized receipt, no payment captured (cash?)
PAYMENT_ONLY = "payment_only"  # POS/card slip, no itemized receipt captured
EMPTY = "empty"  # neither layer — usually a bad extraction

_COVERAGE_CLASSES = (MATCHED, ITEMS_ONLY, PAYMENT_ONLY, EMPTY)


def _has_payment(payments_json: str | None, bundle_types: str | None) -> bool:
    """A fact carries the payment layer if it has payment atoms or a
    payment-record bundle in its cloud."""
    if bundle_types and "payment" in bundle_types.lower():
        return True
    if payments_json:
        try:
            data = json.loads(payments_json)
            return bool(data)
        except (json.JSONDecodeError, TypeError):
            return False
    return False


def _sum_payments(payments_json: str | None) -> float | None:
    """Sum the normalised ``amount`` across a fact's payment records.

    ``facts.payments`` is the persisted payment-atom data; each entry's
    ``amount`` was normalised to a numeric value at parse time. Returns None
    when there is no parseable payment amount (so callers can distinguish
    "no payment data" from "payment totals zero").
    """
    if not payments_json:
        return None
    try:
        data = json.loads(payments_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, list):
        return None
    total = 0.0
    seen = False
    for entry in data:
        if not isinstance(entry, dict) or entry.get("amount") is None:
            continue
        try:
            total += float(entry["amount"])
            seen = True
        except (TypeError, ValueError):
            continue
    return total if seen else None


def _mismatch(a: float | None, b: float | None) -> bool:
    """True if two amounts differ beyond tolerance (2c or 5% of the larger)."""
    if a is None or b is None:
        return False
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return abs(fa - fb) > max(0.02, 0.05 * max(abs(fa), abs(fb)))


def _build_filters(filters: dict[str, Any]) -> tuple[str, list[Any]]:
    """Build the WHERE clause for the fact-level A filter axes."""
    conditions: list[str] = ["1=1"]
    params: list[Any] = []

    if filters.get("vendor"):
        conditions.append("LOWER(f.vendor) LIKE ?")
        params.append(f"%{str(filters['vendor']).lower()}%")

    for key, column in (
        ("vendor_key", "f.vendor_key"),
        ("currency", "f.currency"),
        ("country", "f.country"),
    ):
        if filters.get(key):
            conditions.append(f"{column} = ?")
            params.append(filters[key])

    if filters.get("date_from"):
        conditions.append("f.event_date >= ?")
        params.append(str(filters["date_from"]))
    if filters.get("date_to"):
        conditions.append("f.event_date <= ?")
        params.append(str(filters["date_to"]))
    if filters.get("datetime_from") is not None:
        conditions.append(
            "(f.event_date || ' ' || COALESCE(f.event_time, '00:00:00')) >= ?"
        )
        params.append(str(filters["datetime_from"]))
    if filters.get("datetime_to") is not None:
        conditions.append(
            "(f.event_date || ' ' || COALESCE(f.event_time, '23:59:59')) <= ?"
        )
        params.append(str(filters["datetime_to"]))

    return " AND ".join(conditions), params


def reconcile(
    db: DatabaseManager, filters: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Overlay the item and payment layers and classify each transaction.

    Args:
        db: Database manager.
        filters: Optional dict. Fact-level A axes: ``vendor`` (substring),
            ``vendor_key``/``currency``/``country`` (exact), ``date_from``/
            ``date_to`` (inclusive on event_date), ``datetime_from``/
            ``datetime_to`` (finer, on event_date+event_time). Plus
            ``coverage`` to restrict the result to one coverage class.

    Returns:
        {
          "summary": {matched, items_only, payment_only, empty, total},
          "transactions": [ {id, vendor, total_amount, items_sum,
                             payment_amount, currency, country, event_date,
                             event_time, n_items, has_payment, coverage,
                             amount_mismatch, payment_mismatch}, ... ],
        }
        The summary counts the returned (filtered) set.
    """
    filters = filters or {}
    where, params = _build_filters(filters)
    coverage_filter = filters.get("coverage")

    rows = db.fetchall(
        f"""
        SELECT f.id, f.vendor, f.total_amount, f.currency, f.country,
               f.event_date, f.event_time, f.payments,
               (SELECT COUNT(*) FROM fact_items fi WHERE fi.fact_id = f.id)
                   AS n_items,
               (SELECT GROUP_CONCAT(DISTINCT b.bundle_type)
                  FROM bundles b WHERE b.cloud_id = f.cloud_id) AS bundle_types,
               (SELECT COALESCE(SUM(CAST(fi.total_price AS REAL)), 0)
                  FROM fact_items fi WHERE fi.fact_id = f.id) AS items_sum
        FROM facts f
        WHERE {where}
        ORDER BY f.event_date DESC, f.event_time DESC
        """,  # noqa: S608 — columns fixed, filters parameterised
        tuple(params),
    )

    summary = {cls: 0 for cls in _COVERAGE_CLASSES}
    summary["total"] = 0
    transactions: list[dict[str, Any]] = []
    for r in rows:
        has_items = (r["n_items"] or 0) > 0
        has_payment = _has_payment(r["payments"], r["bundle_types"])
        if has_items and has_payment:
            coverage = MATCHED
        elif has_items:
            coverage = ITEMS_ONLY
        elif has_payment:
            coverage = PAYMENT_ONLY
        else:
            coverage = EMPTY

        if coverage_filter and coverage != coverage_filter:
            continue

        total = r["total_amount"]
        items_sum = r["items_sum"] if has_items else None
        payment_amount = _sum_payments(r["payments"])

        summary[coverage] += 1
        summary["total"] += 1

        transactions.append(
            {
                "id": r["id"],
                "vendor": r["vendor"],
                "total_amount": total,
                "items_sum": items_sum,
                "payment_amount": payment_amount,
                "currency": r["currency"],
                "country": r["country"],
                "event_date": r["event_date"],
                "event_time": r["event_time"],
                "n_items": r["n_items"] or 0,
                "has_payment": has_payment,
                "coverage": coverage,
                # Receipt total vs sum of line items.
                "amount_mismatch": _mismatch(total, items_sum),
                # Receipt total vs normalised payment amount.
                "payment_mismatch": _mismatch(total, payment_amount),
            }
        )

    return {"summary": summary, "transactions": transactions}
