"""Historical verification: cross-reference extractions against existing DB data.

Uses prior extractions to improve consistency of new extractions:
- Vendor identity via registration ID (VAT number) — strongest identifier
- Fallback: vendor name matching when no registration is available
- Product name consistency for known vendors
- Vendor detail enrichment (address, phone, website)

A vendor chain shares the same registration across all stores, even if
addresses differ. Registration is therefore the primary lookup key.

When no registration is present, we fall back to normalized vendor name
matching against the facts table.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from alibi.db.connection import DatabaseManager
from alibi.db import v2_store

logger = logging.getLogger(__name__)

# Minimum similarity ratio for fuzzy product name matching
_PRODUCT_NAME_SIMILARITY = 0.75

# Minimum similarity ratio for vendor name suggestion
_VENDOR_NAME_SIMILARITY = 0.5


def make_vendor_key(
    registration: str | None,
    vendor_name: str | None,
) -> str | None:
    """Build a stable vendor key for historical lookups.

    Priority:
    1. Registration ID (VAT number) — strongest, same across all stores
    2. Fallback: "noid_<hash>" from normalized vendor name

    Returns None if neither registration nor name is available.
    """
    from alibi.normalizers.vendors import normalize_vendor_slug

    if registration:
        return registration.strip().upper().replace(" ", "")

    if vendor_name:
        normalized = normalize_vendor_slug(vendor_name)
        if not normalized:
            return None
        # Hash for stability (avoids issues with OCR variants)
        name_hash = hashlib.sha256(normalized.encode()).hexdigest()[:10]
        return f"noid_{name_hash}"

    return None


@dataclass
class HistoricalCorrection:
    """A single correction suggested by historical data."""

    field: str  # "vendor", "line_items[2].name", etc.
    original: str
    suggested: str
    reason: str  # "registration_match", "product_name_consistency", etc.
    confidence: float  # 0-1


@dataclass
class HistoricalResult:
    """Result of historical verification."""

    corrections: list[HistoricalCorrection] = field(default_factory=list)
    vendor_identified: bool = False
    known_vendor_name: str | None = None
    registration_id: str | None = None
    products_matched: int = 0
    products_total: int = 0

    @property
    def applied_count(self) -> int:
        return len(self.corrections)


def _best_product_match(
    name: str,
    known_names: list[str],
) -> tuple[str | None, float]:
    """Find the best fuzzy match for a product name in known names.

    Returns (best_match, similarity_ratio) or (None, 0.0) if no match.
    """
    if not name or not known_names:
        return None, 0.0

    name_lower = name.lower().strip()
    best_match: str | None = None
    best_ratio = 0.0

    for known in known_names:
        known_lower = known.lower().strip()

        # Exact match (normalized)
        if name_lower == known_lower:
            return known, 1.0

        ratio = SequenceMatcher(None, name_lower, known_lower).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = known

    return best_match, best_ratio


def check_vendor_identity(
    db: DatabaseManager,
    extracted: dict[str, Any],
) -> HistoricalResult:
    """Check if the vendor can be identified via registration ID or name.

    Lookup priority:
    1. Registration ID (VAT) — strongest, same across all stores in a chain
    2. Fallback: fuzzy vendor name match against facts table

    If the extracted vendor name differs from the canonical, suggests a
    correction to the most common historical name.

    Args:
        db: Database manager.
        extracted: Extraction dict with vendor, vendor_vat, etc.

    Returns:
        HistoricalResult with vendor identity info and corrections.
    """
    result = HistoricalResult()

    registration = (
        extracted.get("vendor_vat") or extracted.get("vendor_tax_id") or ""
    ).strip()
    extracted_vendor = (extracted.get("vendor") or "").strip()

    # Path 1: Registration-based lookup (strongest)
    if registration:
        result.registration_id = registration
        known_names = v2_store.get_known_vendor_names(db, registration)
        if known_names:
            result.vendor_identified = True
            result.known_vendor_name = known_names[0]
            return _suggest_vendor_correction(
                result, extracted_vendor, known_names, "registration_match"
            )

    # Path 2: Fallback — name-based lookup against facts table
    if extracted_vendor:
        known_facts_vendors = v2_store.find_matching_fact_vendors(db, extracted_vendor)
        if known_facts_vendors:
            result.vendor_identified = True
            result.known_vendor_name = known_facts_vendors[0]
            return _suggest_vendor_correction(
                result, extracted_vendor, known_facts_vendors, "name_match"
            )

    return result


def _suggest_vendor_correction(
    result: HistoricalResult,
    extracted_vendor: str,
    known_names: list[str],
    reason: str,
) -> HistoricalResult:
    """Suggest a vendor name correction if extracted name differs from canonical."""
    canonical = known_names[0]

    if not extracted_vendor:
        # No vendor extracted but we know who it is
        confidence = 0.95 if reason == "registration_match" else 0.7
        result.corrections.append(
            HistoricalCorrection(
                field="vendor",
                original="",
                suggested=canonical,
                reason=reason,
                confidence=confidence,
            )
        )
        return result

    # Check if already correct
    if extracted_vendor.lower().strip() == canonical.lower().strip():
        return result

    # Registration match is definitive — always trust it
    if reason == "registration_match":
        result.corrections.append(
            HistoricalCorrection(
                field="vendor",
                original=extracted_vendor,
                suggested=canonical,
                reason=reason,
                confidence=0.95,
            )
        )
        return result

    # Name-based fallback: only suggest if similarity is high enough
    _, similarity = _best_product_match(extracted_vendor, known_names)
    if similarity >= _VENDOR_NAME_SIMILARITY:
        result.corrections.append(
            HistoricalCorrection(
                field="vendor",
                original=extracted_vendor,
                suggested=canonical,
                reason=reason,
                confidence=min(0.7, similarity + 0.1),
            )
        )

    return result


def check_product_names(
    db: DatabaseManager,
    extracted: dict[str, Any],
    vendor_name: str,
) -> list[HistoricalCorrection]:
    """Check line item names against historical product names for this vendor.

    If a product name is a fuzzy match for a known product, suggests the
    known name for consistency.

    Args:
        db: Database manager.
        extracted: Extraction dict with line_items.
        vendor_name: Vendor name to look up products for.

    Returns:
        List of corrections for product names.
    """
    line_items = extracted.get("line_items") or []
    if not line_items or not vendor_name:
        return []

    known_products = v2_store.get_known_product_names_for_vendor(db, vendor_name)
    if not known_products:
        return []

    corrections: list[HistoricalCorrection] = []

    for i, item in enumerate(line_items):
        name = (item.get("name") or "").strip()
        if not name:
            continue

        best_match, ratio = _best_product_match(name, known_products)
        if best_match is None:
            continue

        # Exact match — already consistent
        if ratio >= 0.99:
            continue

        # Close match — suggest correction
        if ratio >= _PRODUCT_NAME_SIMILARITY:
            corrections.append(
                HistoricalCorrection(
                    field=f"line_items[{i}].name",
                    original=name,
                    suggested=best_match,
                    reason="product_name_consistency",
                    confidence=ratio,
                )
            )

    return corrections


def check_vendor_details(
    db: DatabaseManager,
    extracted: dict[str, Any],
    vendor_name: str,
) -> list[HistoricalCorrection]:
    """Enrich vendor details from historical records.

    If vendor_phone or vendor_website are missing, fills from history.
    Does NOT override address (varies per store in a chain).

    Args:
        db: Database manager.
        extracted: Extraction dict.
        vendor_name: Vendor name to look up.

    Returns:
        List of enrichment corrections.
    """
    if not vendor_name:
        return []

    history = v2_store.get_vendor_details_history(db, vendor_name)
    corrections: list[HistoricalCorrection] = []

    # Enrich missing fields (not address — that's per-store)
    for field_name, extraction_key in [
        ("phone", "vendor_phone"),
        ("website", "vendor_website"),
    ]:
        known_values = history.get(field_name, [])
        if not known_values:
            continue

        current = (extracted.get(extraction_key) or "").strip()
        if not current:
            # Missing — fill from history
            corrections.append(
                HistoricalCorrection(
                    field=extraction_key,
                    original="",
                    suggested=known_values[0],
                    reason="vendor_detail_enrichment",
                    confidence=0.8,
                )
            )

    # VAT enrichment — if missing in extraction but known in history
    known_regs = history.get("vat_number", [])
    current_reg = (extracted.get("vendor_vat") or "").strip()
    if not current_reg and known_regs:
        corrections.append(
            HistoricalCorrection(
                field="vendor_vat",
                original="",
                suggested=known_regs[0],
                reason="vendor_detail_enrichment",
                confidence=0.85,
            )
        )

    return corrections


def backfill_vendor_key(
    db: DatabaseManager,
    registration: str,
    vendor_name: str,
) -> int:
    """Backfill vendor_key on old facts when a registration is first discovered.

    Computes the old name-hash key and the new registration-based key,
    then updates all facts that had the old key.

    Args:
        db: Database manager.
        registration: Newly discovered registration (VAT number).
        vendor_name: Vendor name used in prior facts.

    Returns:
        Number of facts updated.
    """
    old_key = make_vendor_key(None, vendor_name)
    new_key = make_vendor_key(registration, None)
    if not old_key or not new_key or old_key == new_key:
        return 0

    count = v2_store.backfill_vendor_keys(db, old_key, new_key)
    if count > 0:
        logger.info(
            f"Backfilled vendor_key: {old_key} → {new_key} " f"({count} facts updated)"
        )
    return count


def apply_historical_corrections(
    db: DatabaseManager,
    extracted: dict[str, Any],
) -> HistoricalResult:
    """Run all historical checks and apply corrections to the extraction.

    Modifies extracted dict in-place with corrections. Returns a result
    describing what was found and changed.

    Correction priority:
    1. Vendor identity via registration ID (highest confidence)
    2. Vendor details enrichment
    3. Product name consistency

    Args:
        db: Database manager with existing v2 data.
        extracted: Extraction dict to verify and correct (modified in-place).

    Returns:
        HistoricalResult with all corrections made.
    """
    result = check_vendor_identity(db, extracted)

    # OCR spell correction via identity database (before vendor name is used)
    if not result.vendor_identified:
        raw_vendor = (extracted.get("vendor") or "").strip()
        if raw_vendor:
            from alibi.identities.matching import suggest_vendor_correction

            corrected = suggest_vendor_correction(db, raw_vendor)
            if corrected:
                extracted["vendor"] = corrected
                result.corrections.append(
                    HistoricalCorrection(
                        field="vendor",
                        original=raw_vendor,
                        suggested=corrected,
                        reason="vendor_name_correction",
                        confidence=0.82,
                    )
                )
                # Re-run identity check with corrected name
                result = check_vendor_identity(db, extracted)

    # Determine the vendor name for further lookups
    vendor_name = result.known_vendor_name or (extracted.get("vendor") or "").strip()
    if not vendor_name:
        return result

    # Apply vendor name correction
    for correction in result.corrections:
        if correction.field == "vendor":
            extracted["vendor"] = correction.suggested
            logger.info(
                f"Historical: vendor corrected "
                f"'{correction.original}' → '{correction.suggested}' "
                f"(reason={correction.reason}, confidence={correction.confidence:.2f})"
            )

    # Backfill vendor_key when a registration is first discovered
    registration = (
        extracted.get("vendor_vat") or extracted.get("vendor_tax_id") or ""
    ).strip()
    if registration and result.vendor_identified:
        # Check if this registration is new (not yet in any vendor atom)
        known_by_reg = v2_store.get_known_vendor_names(db, registration)
        if not known_by_reg:
            backfill_vendor_key(db, registration, vendor_name)

    # Check and enrich vendor details
    detail_corrections = check_vendor_details(db, extracted, vendor_name)
    for correction in detail_corrections:
        extracted[correction.field] = correction.suggested
        logger.info(
            f"Historical: {correction.field} enriched → '{correction.suggested}'"
        )
    result.corrections.extend(detail_corrections)

    # Check product name consistency
    line_items = extracted.get("line_items") or []
    product_corrections = check_product_names(db, extracted, vendor_name)
    result.products_total = len(line_items)
    products_matched = 0

    for correction in product_corrections:
        # Parse "line_items[2].name" → index 2
        try:
            idx = int(correction.field.split("[")[1].split("]")[0])
            if 0 <= idx < len(line_items):
                line_items[idx]["name"] = correction.suggested
                products_matched += 1
                logger.debug(
                    f"Historical: product name corrected "
                    f"'{correction.original}' → '{correction.suggested}'"
                )
        except (IndexError, ValueError):
            pass

    result.products_matched = products_matched
    result.corrections.extend(product_corrections)

    # OCR spell correction for item names via identity + product_cache
    for i, item in enumerate(line_items):
        item_name = (item.get("name") or "").strip()
        if not item_name or len(item_name) < 3:
            continue

        from alibi.identities.matching import suggest_item_correction

        corrected = suggest_item_correction(db, item_name)
        if corrected and corrected != item_name:
            line_items[i]["name"] = corrected
            result.corrections.append(
                HistoricalCorrection(
                    field=f"line_items[{i}].name",
                    original=item_name,
                    suggested=corrected,
                    reason="item_name_correction",
                    confidence=0.82,
                )
            )
            logger.info(
                "Historical: item name corrected '%s' → '%s'",
                item_name,
                corrected,
            )

    # Feed discoveries back to identity system
    try:
        from alibi.identities.matching import ensure_vendor_identity

        ensure_vendor_identity(
            db,
            vendor_name=vendor_name,
            vendor_key=make_vendor_key(registration, None) if registration else None,
            vat_number=extracted.get("vendor_vat"),
            tax_id=extracted.get("vendor_tax_id"),
            source="historical",
        )
    except Exception as e:
        logger.debug(f"Identity feedback from historical skipped: {e}")

    return result
