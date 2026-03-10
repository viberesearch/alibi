"""UPCitemdb API client.

Barcode-based product lookup with local caching. Uses httpx for HTTP
requests and the product_cache table for persistent caching.

API docs: https://www.upcitemdb.com/api/explorer#!/lookup/get_trial_lookup
Rate limit: 100 requests/day for trial tier.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import httpx

from alibi.db.connection import DatabaseManager
from alibi.enrichment.off_client import get_cached, store_cache

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.upcitemdb.com/prod/trial/lookup"
_USER_AGENT = "Alibi/1.0 (https://github.com/viberesearch/alibi)"
_TIMEOUT = 10.0
_SOURCE = "upcitemdb"

# Daily rate limit tracking
_DAILY_LIMIT = 100
_daily_count: int = 0
_daily_date: str = ""


def _check_rate_limit() -> bool:
    """Return True if we can make a request today."""
    global _daily_count, _daily_date

    today = date.today().isoformat()
    if _daily_date != today:
        _daily_date = today
        _daily_count = 0

    if _daily_count >= _DAILY_LIMIT:
        logger.warning(
            "UPCitemdb daily limit reached (%d/%d) for %s",
            _daily_count,
            _DAILY_LIMIT,
            today,
        )
        return False

    return True


def _increment_counter() -> None:
    """Increment the daily request counter."""
    global _daily_count
    _daily_count += 1


def _normalize(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a UPCitemdb item to the shared product dict shape."""
    return {
        "product_name": item.get("title") or "",
        "brands": item.get("brand") or "",
        "categories": item.get("category") or "",
        "description": item.get("description") or "",
        "ean": item.get("ean") or "",
    }


def lookup_barcode(barcode: str) -> dict[str, Any] | None:
    """Look up a product by barcode from UPCitemdb.

    Returns a normalized product dict, or None if not found / error.
    Does NOT check cache — use cached_lookup() for the cached version.
    """
    if not _check_rate_limit():
        return None

    url = f"{_BASE_URL}?upc={barcode}"

    try:
        _increment_counter()
        with httpx.Client(
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT,
        ) as client:
            resp = client.get(url)

        if resp.status_code == 404:
            logger.debug("UPCitemdb: barcode %s not found (404)", barcode)
            return None

        resp.raise_for_status()

        data = resp.json()
        items = data.get("items")
        if not items:
            logger.debug("UPCitemdb: barcode %s returned no items", barcode)
            return None

        product = _normalize(items[0])
        logger.debug(
            "UPCitemdb: barcode %s -> %s (%s)",
            barcode,
            product.get("product_name", "?"),
            product.get("brands", "?"),
        )
        return product

    except httpx.HTTPStatusError as e:
        logger.warning("UPCitemdb API error for %s: %s", barcode, e)
        return None
    except httpx.RequestError as e:
        logger.warning("UPCitemdb request failed for %s: %s", barcode, e)
        return None


def cached_lookup(
    db: DatabaseManager,
    barcode: str,
) -> dict[str, Any] | None:
    """Look up a barcode, checking product_cache first.

    Cache-hit logic:
    - Real product in cache (no _not_found flag) -> return it.
    - _not_found AND source is "upcitemdb" -> return None (already checked).
    - _not_found from another source (e.g. OFF) -> still try UPCitemdb.
    - No cache entry -> call the API.
    """
    cached = get_cached(db, barcode)

    if cached is not None:
        if not cached.get("_not_found"):
            return cached
        row = db.fetchone(
            "SELECT source FROM product_cache WHERE barcode = ?",
            (barcode,),
        )
        if row and row["source"] == _SOURCE:
            return None

    product = lookup_barcode(barcode)
    if product:
        store_cache(db, barcode, product, source=_SOURCE)
    else:
        store_cache(db, barcode, {"_not_found": True}, source=_SOURCE)

    return product
