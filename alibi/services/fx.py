"""Historical currency -> EUR conversion for currency-comparable analytics.

Facts are recorded in their receipt currency; the spend / price analytics need a
single comparable currency (EUR). This service resolves the **historical** rate
at each fact's ``event_date`` (so a 2025 CAD purchase uses the 2025 rate, not
today's) and caches it, then stamps each fact's ``eur_rate`` so the materialised
``item_stars`` rows and the fact-level aggregations can convert by simple
multiplication.

Rates come from Frankfurter (``api.frankfurter.dev``) -- free, no key, ECB
reference rates. For a weekend / holiday date the API returns the most recent
prior business-day rate; we cache it under the requested date so the lookup is
exact next time.

Frankfurter follows the ECB, which dropped some currencies (notably RUB, after
March 2022), so it cannot convert e.g. a Russian receipt. For any currency it
can't serve we fall back to the Central Bank of Russia (``cbr.ru``) -- also
free / no key, and authoritative for RUB cross-rates -- which lists EUR among
~40 currencies. A currency/date that *neither* source can serve resolves to
``None`` and is left unconverted (its ``eur_rate`` / ``*_eur`` stay NULL and it
drops out of the EUR-only aggregations rather than blending a foreign amount
back in).
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

_FRANKFURTER_BASE = "https://api.frankfurter.dev/v1"
_CBR_DAILY = "https://www.cbr.ru/scripts/XML_daily.asp"
_TIMEOUT = 10.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fetch_frankfurter(currency: str, date_str: str) -> float | None:
    """Fetch EUR-per-1-unit of ``currency`` on ``date_str`` from Frankfurter.

    Returns ``None`` on any network / parse error or an unsupported currency.
    """
    url = f"{_FRANKFURTER_BASE}/{date_str}"
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(url, params={"base": currency, "symbols": "EUR"})
            resp.raise_for_status()
            data = resp.json()
        rate = data.get("rates", {}).get("EUR")
        if rate is None:
            logger.warning(
                "Frankfurter returned no EUR rate for %s %s", currency, date_str
            )
            return None
        return float(rate)
    except Exception:  # network, HTTP, JSON, value — all non-fatal
        logger.warning(
            "FX rate fetch failed for %s %s", currency, date_str, exc_info=True
        )
        return None


def _cbr_decimal(value: str) -> float:
    """Parse a CBR numeric string (comma decimal separator) to float."""
    return float(value.strip().replace("\xa0", "").replace(" ", "").replace(",", "."))


def _fetch_cbr(currency: str, date_str: str) -> float | None:
    """Fetch EUR-per-1-unit of ``currency`` on ``date_str`` from the CBR.

    The Bank of Russia publishes an authoritative daily table of RUB per foreign
    unit (``Value`` per ``Nominal``) for ~40 currencies including EUR. We derive
    EUR-per-unit of the requested currency as its RUB price over EUR's RUB price.
    Used only as a fallback for currencies the ECB (Frankfurter) no longer
    serves -- chiefly RUB. Returns ``None`` on any error or missing currency.
    """
    try:
        req_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%d/%m/%Y")
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(_CBR_DAILY, params={"date_req": req_date})
            resp.raise_for_status()
            payload = resp.content
        # Defence-in-depth against entity-expansion ("billion laughs"): stdlib
        # ElementTree never resolves *external* entities, and CBR never sends a
        # DTD, so reject any inline DOCTYPE/ENTITY before parsing.
        if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
            logger.warning("CBR response contained a DTD; refusing to parse")
            return None
        # CBR serves windows-1251; let ElementTree honour the XML declaration.
        root = ET.fromstring(payload)

        rub_per: dict[str, float] = {}
        for valute in root.findall("Valute"):
            code = (valute.findtext("CharCode") or "").upper()
            nominal = valute.findtext("Nominal")
            value = valute.findtext("Value")
            if not code or not nominal or not value:
                continue
            nom = _cbr_decimal(nominal)
            if nom == 0:
                continue
            rub_per[code] = _cbr_decimal(value) / nom  # RUB per 1 unit

        eur_rub = rub_per.get("EUR")
        cur = currency.upper()
        cur_rub = 1.0 if cur == "RUB" else rub_per.get(cur)
        if not eur_rub or cur_rub is None:
            logger.warning("CBR returned no rate for %s %s", currency, date_str)
            return None
        # EUR per 1 unit of currency = (RUB per unit) / (RUB per EUR)
        return cur_rub / eur_rub
    except Exception:  # network, HTTP, XML, value — all non-fatal
        logger.warning(
            "CBR FX fetch failed for %s %s", currency, date_str, exc_info=True
        )
        return None


def get_eur_per_unit(
    db: "DatabaseManager",
    currency: str | None,
    date_str: str | None,
    *,
    fetch: bool = True,
) -> float | None:
    """Resolve EUR per 1 unit of ``currency`` on ``date_str`` (cache then fetch).

    EUR (or a missing currency) is always 1.0. A missing date can't be dated to a
    historical rate, so a non-EUR currency with no date resolves to ``None``.
    With ``fetch=False`` only the cache is consulted (no network) -- used by
    tests and offline runs.
    """
    if not currency or currency.upper() == "EUR":
        return 1.0
    if not date_str:
        return None
    cur = currency.upper()
    row = db.fetchone(
        "SELECT eur_per_unit FROM exchange_rates WHERE base = ? AND rate_date = ?",
        (cur, date_str),
    )
    if row is not None:
        return float(row["eur_per_unit"])
    if not fetch:
        return None
    # ECB reference rates first; fall back to the CBR for currencies the ECB
    # dropped (chiefly RUB) so Russian receipts still convert to EUR.
    rate = _fetch_frankfurter(cur, date_str)
    if rate is None:
        rate = _fetch_cbr(cur, date_str)
    if rate is None:
        return None
    with db.transaction() as txn:
        txn.execute(
            "INSERT OR REPLACE INTO exchange_rates "
            "(base, rate_date, eur_per_unit, fetched_at) VALUES (?, ?, ?, ?)",
            (cur, date_str, rate, _now_iso()),
        )
    return rate


def stamp_fact_rate(
    db: "DatabaseManager", fact_id: str, *, fetch: bool = True
) -> float | None:
    """Resolve and stamp one fact's ``eur_rate`` at its own currency/date.

    Called by the ingestion finalizer so a freshly-created fact is
    EUR-convertible immediately, instead of waiting for a global
    ``backfill_fact_rates`` sweep (which, in the API-only deployment, nothing
    runs automatically). An EUR / currency-less fact gets ``1.0`` with no
    network call; a foreign fact resolves its historical rate (cache, then
    optional fetch) and is left NULL only if unresolved. Returns the stamped
    rate, or ``None`` if it could not be resolved (rate left unchanged).
    """
    row = db.fetchone("SELECT currency, event_date FROM facts WHERE id = ?", (fact_id,))
    if row is None:
        return None
    currency = row["currency"]
    date_str = str(row["event_date"]) if row["event_date"] else None
    rate = get_eur_per_unit(db, currency, date_str, fetch=fetch)
    if rate is None:
        return None
    with db.transaction() as txn:
        txn.execute(
            "UPDATE facts SET eur_rate = ? WHERE id = ?",
            (rate, fact_id),
        )
    return rate


def backfill_fact_rates(db: "DatabaseManager", *, fetch: bool = True) -> dict[str, Any]:
    """Stamp ``facts.eur_rate`` for every fact, fetching rates as needed.

    EUR / currency-less facts get 1.0; each distinct non-EUR (currency, date) is
    resolved once (cache, then optional fetch) and applied to all its facts. A
    fact whose rate can't be resolved is left with ``eur_rate`` NULL (it stays
    out of the EUR-only analytics). Returns a stats dict. Run ``lt items rebuild``
    afterwards to propagate into ``item_stars``'s ``*_eur`` columns.
    """
    pairs = db.fetchall(
        "SELECT DISTINCT currency, event_date FROM facts "
        "WHERE COALESCE(currency, 'EUR') != 'EUR' AND event_date IS NOT NULL"
    )
    # Resolve every (currency, date) rate first (each caches in its own txn),
    # then apply all the fact UPDATEs in one transaction — never nest writes.
    resolved_rates: list[tuple[float, str, str]] = []
    unresolved = 0
    currencies: set[str] = set()
    for pair in pairs:
        currency = pair["currency"]
        date_str = str(pair["event_date"])
        currencies.add(currency)
        rate = get_eur_per_unit(db, currency, date_str, fetch=fetch)
        if rate is None:
            unresolved += 1
            continue
        resolved_rates.append((rate, currency, date_str))

    with db.transaction() as txn:
        # EUR / currency-less always convert 1:1.
        txn.execute(
            "UPDATE facts SET eur_rate = 1.0 "
            "WHERE COALESCE(currency, 'EUR') = 'EUR' "
            "AND (eur_rate IS NULL OR eur_rate != 1.0)"
        )
        for rate, currency, date_str in resolved_rates:
            txn.execute(
                "UPDATE facts SET eur_rate = ? "
                "WHERE currency = ? AND event_date = ?",
                (rate, currency, date_str),
            )
    resolved = len(resolved_rates)

    # Count facts that remain unconverted (foreign currency, no rate / no date).
    still_null = db.fetchone("SELECT COUNT(*) AS n FROM facts WHERE eur_rate IS NULL")
    return {
        "pairs_resolved": resolved,
        "pairs_unresolved": unresolved,
        "currencies": sorted(currencies),
        "facts_unconverted": int(still_null["n"]) if still_null else 0,
    }
