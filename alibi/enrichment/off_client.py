"""Open Food Facts API client.

Barcode-based product lookup with local caching. Uses httpx for HTTP
requests and the product_cache table for persistent caching.

API docs: https://openfoodfacts.github.io/openfoodfacts-server/api/
Rate limit: 100 req/min for product lookups.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

_BASE_URL = "https://world.openfoodfacts.org/api/v2/product"
_USER_AGENT = "Alibi/1.0 (https://github.com/viberesearch/alibi)"
_TIMEOUT = 10.0

# Fields we request from OFF — keeps response small
_FIELDS = [
    "product_name",
    "brands",
    "categories",
    "categories_tags",
    "quantity",
    "nutriments",
    "nutriscore_grade",
    "nova_group",
    "image_front_small_url",
]

# Minimum seconds between API requests (rate limit: 100/min ≈ 0.6s)
_MIN_REQUEST_INTERVAL = 0.7
_last_request_time: float = 0.0


def lookup_barcode(barcode: str) -> dict[str, Any] | None:
    """Look up a product by barcode from Open Food Facts.

    Returns the product dict, or None if not found / error.
    Does NOT check cache — use cached_lookup() for cached version.
    """
    global _last_request_time

    # Rate limiting
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

    # Build URL with fields (no URL-encoding for comma — OFF quirk)
    fields_str = ",".join(_FIELDS)
    url = f"{_BASE_URL}/{barcode}?fields={fields_str}"

    try:
        _last_request_time = time.monotonic()
        with httpx.Client(
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT,
        ) as client:
            resp = client.get(url)

        if resp.status_code == 404:
            return None
        resp.raise_for_status()

        data = resp.json()
        if data.get("status") == 0:
            logger.debug("OFF: barcode %s not found", barcode)
            return None

        product = data.get("product")
        if not product:
            return None

        logger.debug(
            "OFF: barcode %s → %s (%s)",
            barcode,
            product.get("product_name", "?"),
            product.get("brands", "?"),
        )
        return dict(product)

    except httpx.HTTPStatusError as e:
        logger.warning("OFF API error for %s: %s", barcode, e)
        return None
    except httpx.RequestError as e:
        logger.warning("OFF request failed for %s: %s", barcode, e)
        return None


def get_cached(db: DatabaseManager, barcode: str) -> dict[str, Any] | None:
    """Get product data from local cache."""
    row = db.fetchone(
        "SELECT data FROM product_cache WHERE barcode = ?",
        (barcode,),
    )
    if row:
        result: dict[str, Any] = json.loads(row["data"])
        return result
    return None


def store_cache(
    db: DatabaseManager,
    barcode: str,
    product: dict[str, Any],
    source: str = "openfoodfacts",
) -> None:
    """Store product data in local cache."""
    with db.transaction() as cursor:
        cursor.execute(
            "INSERT OR REPLACE INTO product_cache (barcode, data, source) "
            "VALUES (?, ?, ?)",
            (barcode, json.dumps(product), source),
        )


# Sentinel stored in product_cache.data for barcodes not found in OFF
_NOT_FOUND_SENTINEL = '{"_not_found": true}'


def cached_lookup(
    db: DatabaseManager,
    barcode: str,
) -> dict[str, Any] | None:
    """Look up a barcode, checking cache first.

    Returns product dict from cache or fresh API call. Stores result
    in cache on successful API lookup. Caches negative results (404s)
    to avoid repeated API calls for unknown barcodes.
    Returns None if not found.
    """
    # Check cache first
    cached = get_cached(db, barcode)
    if cached is not None:
        # Check for negative cache sentinel
        if cached.get("_not_found"):
            return None
        return cached

    # Fetch from API
    product = lookup_barcode(barcode)
    if product:
        store_cache(db, barcode, product)
    else:
        # Cache negative result to avoid repeated API calls
        store_cache(db, barcode, {"_not_found": True}, source="negative")

    return product


def contribute_product(
    barcode: str,
    brand: str,
    category: str,
    product_name: str | None = None,
) -> bool:
    """Submit product data to Open Food Facts.

    Called after Gemini enriches a barcode product to contribute back
    to the open database. Only submits when barcode + brand + category
    are all present.

    Returns True if submission succeeded, False otherwise.
    """
    global _last_request_time

    if not barcode or not brand or not category:
        return False

    elapsed = time.monotonic() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

    url = "https://world.openfoodfacts.org/cgi/product_jqm2.pl"
    payload = {
        "code": barcode,
        "brands": brand,
        "categories": category,
        "comment": "Contributed by Alibi document tracker",
    }
    if product_name:
        payload["product_name"] = product_name

    try:
        _last_request_time = time.monotonic()
        with httpx.Client(
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT,
        ) as client:
            resp = client.post(url, data=payload)

        if resp.status_code == 200:
            data = resp.json()
            if data.get("status") == 1:
                logger.info(
                    "OFF contribution: barcode %s (%s / %s)",
                    barcode,
                    brand,
                    category,
                )
                return True
            logger.debug("OFF contribution rejected for %s: %s", barcode, data)
            return False

        logger.warning("OFF contribution HTTP %d for %s", resp.status_code, barcode)
        return False

    except httpx.HTTPStatusError as e:
        logger.warning("OFF contribution error for %s: %s", barcode, e)
        return False
    except httpx.RequestError as e:
        logger.warning("OFF contribution request failed for %s: %s", barcode, e)
        return False


def contribute_if_enabled(
    db: DatabaseManager,
    barcode: str,
    brand: str,
    category: str,
    product_name: str | None = None,
) -> bool:
    """Submit to OFF only if contribution is enabled and barcode was unknown.

    Checks:
    1. ALIBI_OFF_CONTRIBUTION_ENABLED config flag
    2. product_cache has negative entry for this barcode (was unknown to OFF)
    3. All required fields present

    Returns True if contributed, False otherwise.
    """
    from alibi.config import get_config

    cfg = get_config()
    if not getattr(cfg, "off_contribution_enabled", False):
        return False

    cached = get_cached(db, barcode)
    if cached is not None and not cached.get("_not_found"):
        return False

    result = contribute_product(barcode, brand, category, product_name)
    if result:
        store_cache(
            db,
            barcode,
            {
                "product_name": product_name or "",
                "brands": brand,
                "categories": category,
                "_contributed": True,
            },
            source="openfoodfacts",
        )
    return result
