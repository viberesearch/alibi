"""Google Maps URL parser and geolocation utilities.

Parses various Google Maps URL formats to extract latitude/longitude
coordinates, and provides haversine distance calculation.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any
from urllib.parse import ParseResult, parse_qs, urlparse

logger = logging.getLogger(__name__)

# Regex to extract @lat,lng from Google Maps place URLs
_AT_COORDS_RE = re.compile(r"@(-?\d+\.\d+),(-?\d+\.\d+)")

# Regex to extract lat,lng from q= parameter
_Q_COORDS_RE = re.compile(r"^(-?\d+\.\d+),\s*(-?\d+\.\d+)$")

# Regex to extract place name from /place/Name/ URLs
_PLACE_NAME_RE = re.compile(r"/place/([^/@]+)")

# Tracking params to strip from cleaned URLs
_STRIP_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
    "entry",
    "ftid",
    "gs_lcp",
    "hl",
}

# Known Google Maps hostnames
_MAPS_HOSTS = {
    "maps.google.com",
    "www.google.com",
    "google.com",
    "maps.app.goo.gl",
    "goo.gl",
}

# Earth radius in meters for haversine
_EARTH_RADIUS_M = 6_371_000


def parse_map_url(url: str) -> dict[str, Any] | None:
    """Parse a Google Maps URL and extract coordinates.

    Handles:
    - https://maps.google.com/maps?q=34.123,33.456
    - https://www.google.com/maps/place/.../@34.123,33.456,17z/...
    - https://www.google.com/maps/@34.123,33.456,17z
    - https://maps.app.goo.gl/xxxxx (short link — requires HTTP redirect)
    - https://goo.gl/maps/xxxxx (legacy short link)
    - Direct coordinate in q= or query= or ll= parameters

    Args:
        url: A Google Maps URL string.

    Returns:
        Dict with keys: lat, lng, clean_url, place_name (or None).
        Returns None if URL cannot be parsed or is not a maps URL.
    """
    if not url or not isinstance(url, str):
        return None

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    host = parsed.hostname or ""

    # Validate it's a Google Maps URL
    if not _is_maps_host(host):
        return None

    # Short links need redirect resolution
    if host in ("maps.app.goo.gl", "goo.gl"):
        return _resolve_short_link(url)

    lat, lng = _extract_coords(parsed)
    if lat is None or lng is None:
        return None

    # Validate coordinate ranges
    if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
        return None

    place_name = _extract_place_name(parsed.path)
    clean_url = _clean_url(parsed)

    return {
        "lat": lat,
        "lng": lng,
        "clean_url": clean_url,
        "place_name": place_name,
    }


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate great-circle distance between two points in meters.

    Uses the haversine formula.

    Args:
        lat1, lng1: First point (decimal degrees).
        lat2, lng2: Second point (decimal degrees).

    Returns:
        Distance in meters.
    """
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return _EARTH_RADIUS_M * c


def is_map_url(text: str) -> bool:
    """Check if a text string looks like a Google Maps URL.

    Quick check without full parsing — useful for message filtering.
    """
    if not text:
        return False
    text = text.strip()
    return any(
        marker in text
        for marker in (
            "maps.google.com",
            "google.com/maps",
            "maps.app.goo.gl",
            "goo.gl/maps",
        )
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_maps_host(host: str) -> bool:
    """Check if hostname is a known Google Maps domain."""
    if host in _MAPS_HOSTS:
        return True
    # Handle country-specific subdomains (maps.google.gr, etc.)
    if host.startswith("maps.google.") or host.endswith(".google.com"):
        return True
    return False


def _extract_coords(parsed: ParseResult) -> tuple[float | None, float | None]:
    """Extract lat/lng from parsed URL components."""
    # Try @lat,lng in path (place URLs, direct map URLs)
    m = _AT_COORDS_RE.search(parsed.path)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Try q= parameter
    params = parse_qs(parsed.query)
    for key in ("q", "query", "ll", "center"):
        values = params.get(key)
        if values:
            m = _Q_COORDS_RE.match(values[0])
            if m:
                return float(m.group(1)), float(m.group(2))

    # Try sll= (source lat/lng)
    sll = params.get("sll")
    if sll:
        m = _Q_COORDS_RE.match(sll[0])
        if m:
            return float(m.group(1)), float(m.group(2))

    return None, None


def _extract_place_name(path: str) -> str | None:
    """Extract place name from /place/Name/ URL pattern."""
    m = _PLACE_NAME_RE.search(path)
    if m:
        # URL-decode plus signs and percent encoding
        name = m.group(1).replace("+", " ")
        # Basic percent-decode
        try:
            from urllib.parse import unquote

            name = unquote(name)
        except Exception:
            pass
        return name if name else None
    return None


def _clean_url(parsed: ParseResult) -> str:
    """Rebuild URL without tracking parameters."""
    params = parse_qs(parsed.query, keep_blank_values=True)
    clean_params = {k: v for k, v in params.items() if k.lower() not in _STRIP_PARAMS}

    if clean_params:
        from urllib.parse import urlencode

        query = urlencode(clean_params, doseq=True)
    else:
        query = ""

    # Rebuild
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    path = parsed.path

    if query:
        return f"{scheme}://{netloc}{path}?{query}"
    return f"{scheme}://{netloc}{path}"


def _resolve_short_link(url: str) -> dict[str, Any] | None:
    """Resolve a Google Maps short link by following redirects.

    Uses urllib to follow the redirect and then parses the final URL.
    """
    try:
        import urllib.request

        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "alibi/1.0")
        # Follow redirects manually to capture the final URL
        opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
        response = opener.open(req, timeout=10)
        final_url = response.url
        if final_url and final_url != url:
            # Parse the resolved URL (recursion is safe — resolved URLs
            # are full google.com/maps URLs, not short links)
            return parse_map_url(final_url)
    except Exception as e:
        logger.debug(f"Short link resolution failed for {url}: {e}")

    return None
