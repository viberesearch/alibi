"""GS1 company prefix decoder and brand propagation.

Extracts manufacturer/country information from barcode structure and
uses historical identity data to propagate brands for barcodes sharing
the same GS1 company prefix.

Note: The "Verified by GS1" API requires paid registration. This module
decodes the GS1 prefix structure directly and cross-references existing
enriched products to infer brand ownership — no external API calls needed.
"""

from __future__ import annotations

import logging
from typing import Any

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)

_SOURCE = "gs1"

# GS1 country prefixes — prefix range (string) to country name.
# Ranges are expressed as "NNN" (exact) or "NNN-MMM" (inclusive range).
_GS1_PREFIXES: list[tuple[str, str, str]] = [
    ("000", "019", "USA/Canada"),
    ("020", "029", "Store internal"),
    ("030", "039", "USA/Canada"),
    ("300", "379", "France"),
    ("380", "380", "Bulgaria"),
    ("383", "383", "Slovenia"),
    ("385", "385", "Croatia"),
    ("387", "387", "Bosnia"),
    ("389", "389", "Montenegro"),
    ("400", "440", "Germany"),
    ("450", "459", "Japan"),
    ("460", "469", "Russia"),
    ("470", "470", "Kyrgyzstan"),
    ("471", "471", "Taiwan"),
    ("474", "474", "Estonia"),
    ("475", "475", "Latvia"),
    ("476", "476", "Azerbaijan"),
    ("477", "477", "Lithuania"),
    ("478", "478", "Uzbekistan"),
    ("479", "479", "Sri Lanka"),
    ("480", "480", "Philippines"),
    ("481", "481", "Belarus"),
    ("482", "482", "Ukraine"),
    ("484", "484", "Moldova"),
    ("485", "485", "Armenia"),
    ("486", "486", "Georgia"),
    ("487", "487", "Kazakhstan"),
    ("489", "489", "Hong Kong"),
    ("490", "499", "Japan"),
    ("500", "509", "UK"),
    ("520", "520", "Greece"),
    ("528", "528", "Lebanon"),
    ("529", "529", "Cyprus"),
    ("530", "530", "Albania"),
    ("531", "531", "North Macedonia"),
    ("535", "535", "Malta"),
    ("539", "539", "Ireland"),
    ("540", "549", "Belgium/Luxembourg"),
    ("560", "560", "Portugal"),
    ("569", "569", "Iceland"),
    ("570", "579", "Denmark"),
    ("590", "590", "Poland"),
    ("594", "594", "Romania"),
    ("599", "599", "Hungary"),
    ("600", "601", "South Africa"),
    ("609", "609", "Mauritius"),
    ("611", "611", "Morocco"),
    ("613", "613", "Algeria"),
    ("616", "616", "Kenya"),
    ("618", "618", "Ivory Coast"),
    ("619", "619", "Tunisia"),
    ("621", "621", "Syria"),
    ("622", "622", "Egypt"),
    ("624", "624", "Libya"),
    ("625", "625", "Jordan"),
    ("626", "626", "Iran"),
    ("628", "628", "Saudi Arabia"),
    ("629", "629", "UAE"),
    ("640", "649", "Finland"),
    ("690", "699", "China"),
    ("700", "709", "Norway"),
    ("729", "729", "Israel"),
    ("730", "739", "Sweden"),
    ("740", "741", "Guatemala"),
    ("742", "742", "Honduras"),
    ("743", "743", "Nicaragua"),
    ("744", "744", "Costa Rica"),
    ("745", "745", "Panama"),
    ("746", "746", "Dominican Republic"),
    ("750", "750", "Mexico"),
    ("754", "755", "Canada"),
    ("759", "759", "Venezuela"),
    ("760", "769", "Switzerland"),
    ("770", "770", "Colombia"),
    ("773", "773", "Uruguay"),
    ("775", "775", "Peru"),
    ("777", "777", "Bolivia"),
    ("778", "779", "Argentina"),
    ("780", "780", "Chile"),
    ("784", "784", "Paraguay"),
    ("786", "786", "Ecuador"),
    ("789", "790", "Brazil"),
    ("800", "839", "Italy"),
    ("840", "849", "Spain"),
    ("858", "858", "Slovakia"),
    ("859", "859", "Czech Republic"),
    ("860", "860", "Serbia"),
    ("865", "865", "Mongolia"),
    ("867", "867", "North Korea"),
    ("868", "869", "Turkey"),
    ("870", "879", "Netherlands"),
    ("880", "880", "South Korea"),
    ("884", "884", "Cambodia"),
    ("885", "885", "Thailand"),
    ("888", "888", "Singapore"),
    ("890", "890", "India"),
    ("893", "893", "Vietnam"),
    ("896", "896", "Pakistan"),
    ("899", "899", "Indonesia"),
    ("900", "919", "Austria"),
    ("930", "939", "Australia"),
    ("940", "949", "New Zealand"),
    ("950", "950", "GS1 Global Office"),
    ("955", "955", "Malaysia"),
    ("958", "958", "Macau"),
]

# Cypriot and local prefixes likely absent from global barcode DBs
_LOCAL_PREFIXES = {"529"}


def _match_prefix(barcode: str) -> str | None:
    """Match barcode prefix to a country name.

    Checks the first 3 digits of the barcode against the GS1 prefix table.
    Returns the country name or None if no match.
    """
    if not barcode or len(barcode) < 3:
        return None

    prefix3 = barcode[:3]
    for low, high, country in _GS1_PREFIXES:
        if low <= prefix3 <= high:
            return country

    return None


def decode_gs1_prefix(barcode: str) -> dict[str, Any] | None:
    """Decode GS1 company prefix from barcode.

    For a standard 13-digit EAN barcode:
    - Digits 0-2: GS1 country prefix
    - Digits 0-6 to 0-8: Company prefix (variable length)
    - Remaining: Product code + check digit

    Returns dict with country, prefix_range, company_prefix, is_local,
    or None if the barcode is too short or unrecognized.
    """
    if not barcode or len(barcode) < 8:
        return None

    # Strip non-digit characters
    clean = "".join(c for c in barcode if c.isdigit())
    if len(clean) < 8:
        return None

    country = _match_prefix(clean)
    if not country:
        return None

    prefix3 = clean[:3]
    # Company prefix is typically 7-9 digits for a 13-digit EAN.
    # Use 7 as default (most common assignment length).
    company_prefix = clean[:7]

    return {
        "country": country,
        "prefix_range": prefix3,
        "company_prefix": company_prefix,
        "is_local": prefix3 in _LOCAL_PREFIXES,
    }


def _find_brand_by_company_prefix(
    db: DatabaseManager,
    company_prefix: str,
) -> str | None:
    """Search enriched fact_items for barcodes sharing the same company prefix.

    If another item with the same company prefix already has a brand,
    return that brand. This leverages the observation that all products
    from the same manufacturer share a GS1 company prefix.
    """
    row = db.fetchone(
        "SELECT brand FROM fact_items "
        "WHERE barcode LIKE ? "
        "AND brand IS NOT NULL AND brand != '' "
        "LIMIT 1",
        (company_prefix + "%",),
    )
    if row:
        return str(row["brand"])
    return None


def lookup_brand_by_prefix(
    db: DatabaseManager,
    barcode: str,
) -> dict[str, Any] | None:
    """Look up brand information using GS1 prefix + historical data.

    Strategy:
    1. Decode GS1 prefix for country/manufacturer info
    2. Search existing enriched items for barcodes with same company prefix
    3. If found, return the brand from the matching item

    This is brand-only enrichment — category comes from other sources.
    Returns dict with brand info or None.
    """
    prefix_info = decode_gs1_prefix(barcode)
    if not prefix_info:
        return None

    brand = _find_brand_by_company_prefix(db, prefix_info["company_prefix"])
    if not brand:
        logger.debug(
            "GS1: no brand found for prefix %s (%s)",
            prefix_info["company_prefix"],
            prefix_info["country"],
        )
        return None

    logger.info(
        "GS1: barcode %s -> brand=%s (prefix %s, %s)",
        barcode,
        brand,
        prefix_info["company_prefix"],
        prefix_info["country"],
    )
    return {
        "brands": brand,
        "country": prefix_info["country"],
        "is_local": prefix_info["is_local"],
        "company_prefix": prefix_info["company_prefix"],
    }
