"""Identity matching and resolution.

Finds identities for extracted vendor names, item names, registration IDs,
and barcodes. Used during:
1. Cloud formation (vendor identity as scoring signal)
2. Display (resolve fact vendor/item to canonical name)
3. New document processing (identity-aware extraction)
"""

from __future__ import annotations

import difflib
import json
import logging
from typing import Any

from alibi.db.connection import DatabaseManager

logger = logging.getLogger(__name__)


def find_vendor_identity(
    db: DatabaseManager,
    vendor_name: str | None = None,
    vendor_key: str | None = None,
    registration: str | None = None,
) -> dict[str, Any] | None:
    """Find a vendor identity by any matching signal.

    Resolution priority (strongest first):
    1. vendor_key match
    2. registration match
    3. exact name match
    4. normalized_name match

    Returns:
        Identity dict with members, or None.
    """
    conn = db.get_connection()

    # Try vendor_key (strongest)
    if vendor_key:
        result = _find_by_member(conn, "vendor", "vendor_key", vendor_key)
        if result:
            return result

    # Try registration (vat_number > tax_id)
    if registration:
        result = _find_by_member(conn, "vendor", "vat_number", registration)
        if not result:
            result = _find_by_member(conn, "vendor", "tax_id", registration)
        if result:
            return result

    if not vendor_name:
        return None

    # Try exact name
    result = _find_by_member(conn, "vendor", "name", vendor_name)
    if result:
        return result

    # Try normalized name
    from alibi.normalizers.vendors import normalize_vendor_slug

    normalized = normalize_vendor_slug(vendor_name)
    if normalized:
        result = _find_by_member(conn, "vendor", "normalized_name", normalized)
        if result:
            return result

    return None


def suggest_vendor_correction(
    db: DatabaseManager,
    vendor_name: str,
    min_similarity: float = 0.82,
) -> str | None:
    """Suggest a corrected vendor name from the identity database.

    Compares the extracted vendor_name against all known vendor names
    (canonical_name + name members) using SequenceMatcher. Returns
    the best match if similarity >= min_similarity and the name is
    not already an exact match.

    Returns:
        Corrected vendor name (canonical_name), or None if no good match.
    """
    if not vendor_name or len(vendor_name) < 3:
        return None

    # If already an exact identity match, no correction needed
    existing = find_vendor_identity(db, vendor_name=vendor_name)
    if existing:
        return None

    conn = db.get_connection()

    # Load all active vendor canonical names
    canonical_rows = conn.execute(
        "SELECT id, canonical_name FROM identities "
        "WHERE entity_type = 'vendor' AND active = 1"
    ).fetchall()

    if not canonical_rows:
        return None

    # Build candidate map: candidate_value → canonical_name
    candidates: dict[str, str] = {}
    for row in canonical_rows:
        identity_id = row["id"]
        canonical = row["canonical_name"]
        if canonical:
            candidates[canonical] = canonical

        member_rows = conn.execute(
            "SELECT value FROM identity_members "
            "WHERE identity_id = ? AND member_type = 'name'",
            (identity_id,),
        ).fetchall()
        for mrow in member_rows:
            candidates[mrow["value"]] = canonical

    # Find best fuzzy match
    vendor_lower = vendor_name.lower()
    best_canonical: str | None = None
    best_ratio = 0.0
    best_candidate: str | None = None

    for candidate, canonical in candidates.items():
        candidate_lower = candidate.lower()

        # Skip exact match (already handled above)
        if vendor_lower == candidate_lower:
            return None

        ratio = difflib.SequenceMatcher(None, vendor_lower, candidate_lower).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_canonical = canonical
            best_candidate = candidate

    if best_ratio < min_similarity or best_canonical is None:
        return None

    # Guard: must differ by at most 3 characters
    if best_candidate and abs(len(vendor_name) - len(best_candidate)) > 3:
        return None

    logger.info(
        "Vendor name corrected via identity: '%s' -> '%s' (similarity: %.2f)",
        vendor_name,
        best_canonical,
        best_ratio,
    )
    return best_canonical


def suggest_item_correction(
    db: DatabaseManager,
    item_name: str,
    min_similarity: float = 0.82,
) -> str | None:
    """Suggest a corrected item name from the identity database.

    Compares the extracted item_name against all known item names
    (canonical_name + name members) using SequenceMatcher. Returns
    the best match if similarity >= min_similarity and the name is
    not already an exact match.

    Returns:
        Corrected item name (canonical_name), or None if no good match.
    """
    if not item_name or len(item_name) < 3:
        return None

    # If already an exact identity match, no correction needed
    existing = find_item_identity(db, item_name=item_name)
    if existing:
        return None

    conn = db.get_connection()

    # Load all active item canonical names
    canonical_rows = conn.execute(
        "SELECT id, canonical_name FROM identities "
        "WHERE entity_type = 'item' AND active = 1"
    ).fetchall()

    # Build candidate map: candidate_value -> canonical_name
    candidates: dict[str, str] = {}
    for row in canonical_rows or []:
        identity_id = row["id"]
        canonical = row["canonical_name"]
        if canonical:
            candidates[canonical] = canonical

        member_rows = conn.execute(
            "SELECT value FROM identity_members "
            "WHERE identity_id = ? AND member_type = 'name'",
            (identity_id,),
        ).fetchall()
        for mrow in member_rows:
            candidates[mrow["value"]] = canonical

    # Supplement with product_cache product names (OFF/UPCitemdb)
    try:
        cache_rows = conn.execute(
            "SELECT data FROM product_cache WHERE source != 'negative'"
        ).fetchall()
        for crow in cache_rows:
            try:
                product = json.loads(crow["data"])
                pname = (product.get("product_name") or "").strip()
                if (
                    pname
                    and len(pname) >= 3
                    and pname.lower() not in {k.lower() for k in candidates}
                ):
                    candidates[pname] = pname
            except (json.JSONDecodeError, TypeError):
                continue
    except Exception:
        pass  # product_cache may not exist in all DBs

    if not candidates:
        return None

    # Find best fuzzy match
    item_lower = item_name.lower()
    best_canonical: str | None = None
    best_ratio = 0.0
    best_candidate: str | None = None

    for candidate, canonical in candidates.items():
        candidate_lower = candidate.lower()

        # Skip exact match (already handled above)
        if item_lower == candidate_lower:
            return None

        ratio = difflib.SequenceMatcher(None, item_lower, candidate_lower).ratio()

        if ratio > best_ratio:
            best_ratio = ratio
            best_canonical = canonical
            best_candidate = candidate

    if best_ratio < min_similarity or best_canonical is None:
        return None

    # Guard: must differ by at most 3 characters
    if best_candidate and abs(len(item_name) - len(best_candidate)) > 3:
        return None

    logger.info(
        "Item name corrected via identity: '%s' -> '%s' (similarity: %.2f)",
        item_name,
        best_canonical,
        best_ratio,
    )
    return best_canonical


def find_item_identity(
    db: DatabaseManager,
    item_name: str | None = None,
    barcode: str | None = None,
) -> dict[str, Any] | None:
    """Find an item identity by barcode or name.

    Resolution priority:
    1. barcode match (strongest)
    2. exact name match
    3. normalized_name match

    Returns:
        Identity dict with members, or None.
    """
    conn = db.get_connection()

    # Try barcode (strongest)
    if barcode:
        result = _find_by_member(conn, "item", "barcode", barcode)
        if result:
            return result

    if not item_name:
        return None

    # Try exact name
    result = _find_by_member(conn, "item", "name", item_name)
    if result:
        return result

    # Try normalized name
    normalized = _normalize_item_name(item_name)
    if normalized:
        result = _find_by_member(conn, "item", "normalized_name", normalized)
        if result:
            return result

    return None


def resolve_vendor(
    db: DatabaseManager,
    vendor_name: str | None = None,
    vendor_key: str | None = None,
    registration: str | None = None,
) -> str | None:
    """Resolve a vendor to its canonical identity name.

    Returns:
        Canonical name if identity found and active, else None.
    """
    identity = find_vendor_identity(db, vendor_name, vendor_key, registration)
    if identity and identity.get("active"):
        return str(identity["canonical_name"])
    return None


def resolve_item(
    db: DatabaseManager,
    item_name: str | None = None,
    barcode: str | None = None,
) -> str | None:
    """Resolve an item to its canonical identity name.

    Returns:
        Canonical name if identity found and active, else None.
    """
    identity = find_item_identity(db, item_name, barcode)
    if identity and identity.get("active"):
        return str(identity["canonical_name"])
    return None


def find_identities_for_fact(
    db: DatabaseManager,
    vendor: str | None,
    vendor_key: str | None,
    item_names: list[str] | None = None,
) -> dict[str, Any]:
    """Find all relevant identities for a fact (vendor + items).

    Returns:
        Dict with 'vendor_identity' and 'item_identities' keys.
    """
    result: dict[str, Any] = {
        "vendor_identity": None,
        "item_identities": {},
    }

    # Vendor identity
    if vendor or vendor_key:
        result["vendor_identity"] = find_vendor_identity(
            db, vendor_name=vendor, vendor_key=vendor_key
        )

    # Item identities
    if item_names:
        for name in item_names:
            identity = find_item_identity(db, item_name=name)
            if identity:
                result["item_identities"][name] = identity

    return result


def _find_by_member(
    conn: Any,
    entity_type: str,
    member_type: str,
    value: str,
) -> dict[str, Any] | None:
    """Find an active identity by member type + value."""
    row = conn.execute(
        "SELECT i.* FROM identities i "
        "JOIN identity_members m ON i.id = m.identity_id "
        "WHERE i.entity_type = ? AND i.active = 1 "
        "AND m.member_type = ? AND m.value = ? "
        "LIMIT 1",
        (entity_type, member_type, value),
    ).fetchone()

    if not row and member_type in ("vat_number", "vendor_key"):
        bare = _strip_country_prefix(value)
        rows = conn.execute(
            "SELECT i.*, m.value AS _member_value FROM identities i "
            "JOIN identity_members m ON i.id = m.identity_id "
            "WHERE i.entity_type = ? AND i.active = 1 "
            "AND m.member_type = ?",
            (entity_type, member_type),
        ).fetchall()
        for candidate in rows:
            if _strip_country_prefix(candidate["_member_value"]) == bare:
                row = candidate
                break

    if not row:
        return None

    identity = dict(row)
    identity.pop("_member_value", None)
    if identity.get("metadata") and isinstance(identity["metadata"], str):
        identity["metadata"] = json.loads(identity["metadata"])

    members = conn.execute(
        "SELECT * FROM identity_members WHERE identity_id = ?",
        (identity["id"],),
    ).fetchall()
    identity["members"] = [dict(m) for m in members]

    return identity


def ensure_vendor_identity(
    db: DatabaseManager,
    vendor_name: str | None = None,
    vendor_key: str | None = None,
    vat_number: str | None = None,
    tax_id: str | None = None,
    source: str = "extraction",
) -> str | None:
    """Find or create a vendor identity from extraction data.

    Called after extraction to auto-register vendors in the identity system.
    If an identity already exists (by vendor_key, vat_number, or name), adds
    any new members. Otherwise creates a new identity.

    Returns:
        The identity_id, or None if no usable vendor signal.
    """
    from alibi.identities.store import add_member, create_identity
    from alibi.normalizers.vendors import normalize_vendor, normalize_vendor_slug

    # Try to find existing identity
    existing = find_vendor_identity(
        db,
        vendor_name=vendor_name,
        vendor_key=vendor_key,
        registration=vat_number or tax_id,
    )

    if existing:
        identity_id = existing["id"]

        # When correcting, update canonical_name to the corrected value
        if source == "correction" and vendor_name:
            from alibi.identities.store import update_identity

            display = normalize_vendor(vendor_name)
            if display != existing.get("canonical_name"):
                update_identity(db, identity_id, canonical_name=display)

        existing_values = {
            (m["member_type"], m["value"]) for m in existing.get("members", [])
        }

        # Add any new members not already present
        if vendor_name:
            if ("name", vendor_name) not in existing_values:
                add_member(db, identity_id, "name", vendor_name, source=source)
            slug = normalize_vendor_slug(vendor_name)
            if slug and ("normalized_name", slug) not in existing_values:
                add_member(db, identity_id, "normalized_name", slug, source=source)
        if vendor_key and ("vendor_key", vendor_key) not in existing_values:
            add_member(db, identity_id, "vendor_key", vendor_key, source=source)
        if vat_number and ("vat_number", vat_number) not in existing_values:
            add_member(db, identity_id, "vat_number", vat_number, source=source)
        if tax_id and ("tax_id", tax_id) not in existing_values:
            add_member(db, identity_id, "tax_id", tax_id, source=source)

        return str(identity_id)

    # No existing identity — create one if we have enough signal
    if not vendor_name and not vendor_key:
        return None

    display_name = normalize_vendor(vendor_name) if vendor_name else vendor_key
    new_id = str(create_identity(db, "vendor", display_name or ""))

    # Add all available members
    if vendor_name:
        add_member(db, new_id, "name", vendor_name, source=source)
        slug = normalize_vendor_slug(vendor_name)
        if slug:
            add_member(db, new_id, "normalized_name", slug, source=source)
    if vendor_key:
        add_member(db, new_id, "vendor_key", vendor_key, source=source)
    if vat_number:
        add_member(db, new_id, "vat_number", vat_number, source=source)
    if tax_id:
        add_member(db, new_id, "tax_id", tax_id, source=source)

    return new_id


def ensure_item_identity(
    db: DatabaseManager,
    item_name: str | None = None,
    barcode: str | None = None,
    source: str = "extraction",
) -> str | None:
    """Find or create an item identity from extraction data.

    Called after fact creation to auto-register items in the identity system.
    If an identity already exists (by barcode or name), adds any new members.
    Otherwise creates a new identity.

    Returns:
        The identity_id, or None if no usable item signal.
    """
    from alibi.identities.store import add_member, create_identity

    # Try to find existing identity
    existing = find_item_identity(db, item_name=item_name, barcode=barcode)

    if existing:
        identity_id = existing["id"]

        existing_values = {
            (m["member_type"], m["value"]) for m in existing.get("members", [])
        }

        # Add any new members not already present
        if item_name:
            if ("name", item_name) not in existing_values:
                add_member(db, identity_id, "name", item_name, source=source)
            normalized = _normalize_item_name(item_name)
            if normalized and ("normalized_name", normalized) not in existing_values:
                add_member(
                    db, identity_id, "normalized_name", normalized, source=source
                )
        if barcode and ("barcode", barcode) not in existing_values:
            add_member(db, identity_id, "barcode", barcode, source=source)

        return str(identity_id)

    # No existing identity — create one if we have enough signal
    if not item_name and not barcode:
        return None

    display_name = item_name or barcode or ""
    new_id = str(create_identity(db, "item", display_name))

    # Add all available members
    if item_name:
        add_member(db, new_id, "name", item_name, source=source)
        normalized = _normalize_item_name(item_name)
        if normalized:
            add_member(db, new_id, "normalized_name", normalized, source=source)
    if barcode:
        add_member(db, new_id, "barcode", barcode, source=source)

    return new_id


def get_canonical_vendor_key(
    db: DatabaseManager,
    identity_id: str,
) -> str | None:
    """Get the canonical vendor_key for an identity.

    Picks the best VAT number from identity members and builds a
    vendor_key from it. Falls back to existing vendor_key members.

    Returns:
        Canonical vendor_key string, or None.
    """
    from alibi.identities.store import get_members_by_type
    from alibi.extraction.historical import make_vendor_key

    # Prefer vat_number members (strongest signal)
    vat_members = get_members_by_type(db, identity_id, "vat_number")
    if vat_members:
        values = [m["value"] for m in vat_members]
        canonical_vat = _pick_canonical_vat(values)
        return make_vendor_key(canonical_vat, None)

    # Fall back to existing vendor_key members
    vk_members = get_members_by_type(db, identity_id, "vendor_key")
    if vk_members:
        return str(vk_members[0]["value"])

    return None


def _pick_canonical_vat(values: list[str]) -> str:
    """Pick the canonical VAT number from multiple variants.

    Handles EU country-prefixed variants (e.g., CY10180201N == 10180201N).
    Groups by bare number (prefix stripped), then picks best representative.

    Preference within a group: prefixed form (official EU VIES format).
    Preference across groups:
    1. Bare number ending with a letter (Cyprus VAT format)
    2. Shorter bare number (less likely OCR artifacts)
    3. Alphabetical (deterministic tie-break)
    """
    if len(values) == 1:
        return values[0]

    # Group by bare number (strip country prefix for comparison)
    groups: dict[str, list[str]] = {}
    for v in values:
        bare = _strip_country_prefix(v)
        groups.setdefault(bare, []).append(v)

    # Pick best representative from each group (prefer prefixed = longer)
    representatives = []
    for variants in groups.values():
        best = sorted(variants, key=lambda v: (-len(v), v))[0]
        representatives.append(best)

    if len(representatives) == 1:
        return representatives[0]

    # Among different base numbers, prefer letter-ending bare form
    letter_ending = [
        v for v in representatives if _strip_country_prefix(v)[-1:].isalpha()
    ]
    if letter_ending:
        return sorted(letter_ending, key=lambda v: (len(_strip_country_prefix(v)), v))[
            0
        ]

    return sorted(representatives, key=lambda v: (len(_strip_country_prefix(v)), v))[0]


# EU country codes for VAT prefix detection (ISO 3166-1 alpha-2)
_EU_COUNTRY_CODES = frozenset(
    {
        "AT",
        "BE",
        "BG",
        "HR",
        "CY",
        "CZ",
        "DK",
        "EE",
        "FI",
        "FR",
        "DE",
        "EL",
        "GR",
        "HU",
        "IE",
        "IT",
        "LV",
        "LT",
        "LU",
        "MT",
        "NL",
        "PL",
        "PT",
        "RO",
        "SK",
        "SI",
        "ES",
        "SE",
    }
)


def _strip_country_prefix(vat: str) -> str:
    """Strip 2-letter EU country code prefix from VAT number if present."""
    if len(vat) > 2 and vat[:2].upper() in _EU_COUNTRY_CODES:
        return vat[2:]
    return vat


def _normalize_item_name(name: str) -> str:
    """Normalize item name for matching.

    Lowercases, strips extra whitespace, removes common suffixes
    like unit quantities.
    """
    import re

    normalized = name.lower().strip()
    # Remove trailing unit quantities: "milk 1l", "bread 500g"
    normalized = re.sub(r"\s+\d+\s*(ml|l|g|kg|cl|oz|lb)$", "", normalized)
    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized
