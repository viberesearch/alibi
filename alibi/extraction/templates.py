"""Vendor and POS provider template learning.

Learns document format fingerprints from successful extractions and applies
them as parser hints to future documents from the same vendor or POS system.

Template lifecycle:
1. After successful extraction (confidence >= threshold), extract fingerprint
2. Store fingerprint on vendor identity metadata (JSON blob)
3. Before extraction of new document from known vendor, load template
4. Convert template to ParserHints for text_parser.py

No migration needed — uses existing identities.metadata JSON field.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Minimum parser confidence to learn a template from an extraction
_LEARN_THRESHOLD = 0.8

# Minimum successful extractions before template is considered reliable
_RELIABLE_COUNT = 2

# Adaptive threshold tuning constants
_MAX_CONFIDENCE_HISTORY = 20
_ADAPTIVE_THRESHOLD_HIGH = 0.95  # for vendors with avg confidence > 0.95
_ADAPTIVE_THRESHOLD_LOW = 0.7  # for vendors with avg confidence < 0.7

# Number of recent entries used for rolling average
_ROLLING_WINDOW = 10

# Fix count threshold for flagging unreliable fields in hints
_UNRELIABLE_FIX_COUNT = 5

# Staleness detection: confidence drop threshold
_STALENESS_DROP = 0.15  # flag stale if recent avg < historical avg - this
_STALENESS_MIN_HISTORY = 5  # need at least this many observations

# Known POS provider signatures (lowercase text → provider name)
_POS_SIGNATURES: dict[str, str] = {
    "sap customer checkout": "SAP",
    "e.f visionsoft": "VISIONSOFT",
    "jcc payment systems": "JCC",
    "jcc payment": "JCC",
    "micros": "MICROS",
    "oracle fiscal": "ORACLE_FISCAL",
    "sterna": "STERNA",
    "epilogi": "EPILOGI",
    "poseidon": "POSEIDON",
    "softone": "SOFTONE",
    "entersoft": "ENTERSOFT",
    "retail pro": "RETAIL_PRO",
    "lightspeed": "LIGHTSPEED",
    "square pos": "SQUARE_POS",
    "clover station": "CLOVER",
    "toast pos": "TOAST",
}


@dataclass
class ParserHints:
    """Hints passed to text_parser to guide extraction.

    Pre-seeds known values and selects parser strategies based on
    prior successful extractions from the same vendor/POS.
    """

    vendor_name: str | None = None
    currency: str | None = None
    layout_type: str | None = None  # columnar, nqa, markdown_table, standard
    pos_provider: str | None = None
    barcode_position: str | None = None  # before_item, after_item, inline
    unreliable_fields: list[str] | None = None
    date_format: str | None = None
    total_marker: str | None = None
    expected_header_lines: int | None = None


@dataclass
class VendorTemplate:
    """Learned document format fingerprint for a vendor.

    Stored as JSON in identities.metadata under the "template" key.
    """

    layout_type: str = "standard"  # columnar, nqa, markdown_table, standard
    currency: str | None = None
    pos_provider: str | None = None
    success_count: int = 0
    # Gemini-learned schema insights
    gemini_bootstrapped: bool = False
    language: str | None = None
    has_barcodes: bool | None = None
    barcode_position: str | None = None  # before_item, after_item, inline
    has_unit_quantities: bool | None = None
    typical_item_count: int | None = None
    # Adaptive learning metrics
    confidence_history: list[float] = field(default_factory=list)
    adaptive_skip_threshold: float | None = None
    preferred_ocr_tier: int | None = (
        None  # 0=normal, 1=enhanced, 2=rotation, 3=fallback
    )
    needs_rotation: bool = False
    common_fixes: dict[str, int] = field(default_factory=dict)
    last_updated: str | None = None  # ISO datetime of last observation
    stale: bool = False  # True when recent confidence drops significantly
    default_category: str | None = None  # auto-derived when 5+ items share a category
    date_format: str | None = (
        None  # "dmy", "mdy", "ymd", "dmy_time", "mdy_en", "dmy_en"
    )
    date_format_confidence: int = 0  # count of observations confirming this format
    total_marker: str | None = None  # e.g., "TOTAL EUR", "SYNOLO", "GESAMT"
    typical_header_lines: int | None = None
    typical_footer_ratio: float | None = None  # footer_start / total_lines

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        d: dict[str, Any] = {
            "layout_type": self.layout_type,
            "success_count": self.success_count,
        }
        if self.currency:
            d["currency"] = self.currency
        if self.pos_provider:
            d["pos_provider"] = self.pos_provider
        if self.gemini_bootstrapped:
            d["gemini_bootstrapped"] = True
        if self.language:
            d["language"] = self.language
        if self.has_barcodes is not None:
            d["has_barcodes"] = self.has_barcodes
        if self.barcode_position:
            d["barcode_position"] = self.barcode_position
        if self.has_unit_quantities is not None:
            d["has_unit_quantities"] = self.has_unit_quantities
        if self.typical_item_count is not None:
            d["typical_item_count"] = self.typical_item_count
        if self.confidence_history:
            d["confidence_history"] = self.confidence_history
        if self.adaptive_skip_threshold is not None:
            d["adaptive_skip_threshold"] = self.adaptive_skip_threshold
        if self.preferred_ocr_tier is not None:
            d["preferred_ocr_tier"] = self.preferred_ocr_tier
        if self.needs_rotation:
            d["needs_rotation"] = True
        if self.common_fixes:
            d["common_fixes"] = self.common_fixes
        if self.last_updated:
            d["last_updated"] = self.last_updated
        if self.stale:
            d["stale"] = True
        if self.default_category:
            d["default_category"] = self.default_category
        if self.date_format:
            d["date_format"] = self.date_format
            d["date_format_confidence"] = self.date_format_confidence
        if self.total_marker:
            d["total_marker"] = self.total_marker
        if self.typical_header_lines is not None:
            d["typical_header_lines"] = self.typical_header_lines
        if self.typical_footer_ratio is not None:
            d["typical_footer_ratio"] = self.typical_footer_ratio
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> VendorTemplate:
        """Deserialize from dict."""
        return cls(
            layout_type=d.get("layout_type", "standard"),
            currency=d.get("currency"),
            pos_provider=d.get("pos_provider"),
            success_count=d.get("success_count", 0),
            gemini_bootstrapped=d.get("gemini_bootstrapped", False),
            language=d.get("language"),
            has_barcodes=d.get("has_barcodes"),
            barcode_position=d.get("barcode_position"),
            has_unit_quantities=d.get("has_unit_quantities"),
            typical_item_count=d.get("typical_item_count"),
            confidence_history=d.get("confidence_history", []),
            adaptive_skip_threshold=d.get("adaptive_skip_threshold"),
            preferred_ocr_tier=d.get("preferred_ocr_tier"),
            needs_rotation=d.get("needs_rotation", False),
            common_fixes=d.get("common_fixes", {}),
            last_updated=d.get("last_updated"),
            stale=d.get("stale", False),
            default_category=d.get("default_category"),
            date_format=d.get("date_format"),
            date_format_confidence=d.get("date_format_confidence", 0),
            total_marker=d.get("total_marker"),
            typical_header_lines=d.get("typical_header_lines"),
            typical_footer_ratio=d.get("typical_footer_ratio"),
        )

    @property
    def is_reliable(self) -> bool:
        """Template has enough observations to be trusted."""
        return self.success_count >= _RELIABLE_COUNT


def detect_pos_provider(ocr_text: str) -> str | None:
    """Detect POS provider from OCR text.

    Scans for known POS signatures in the text footer/header area.
    Returns provider name string or None.
    """
    if not ocr_text:
        return None
    text_lower = ocr_text.lower()
    for signature, provider in _POS_SIGNATURES.items():
        if signature in text_lower:
            return provider
    return None


def detect_layout_type(parse_data: dict[str, Any]) -> str:
    """Detect layout type from successful parse result data.

    Checks _layout_type metadata set by text_parser, falls back to
    inferring from line_items structure.
    """
    # Check parser-reported layout
    layout = parse_data.get("_layout_type")
    if layout:
        return str(layout)

    # Infer from items: columnar items tend to have quantity > 0
    items = parse_data.get("line_items", [])
    if not items:
        return "standard"

    has_qty = sum(1 for i in items if i.get("quantity") and i["quantity"] != "1")
    if has_qty > len(items) * 0.5:
        return "columnar"
    return "standard"


def detect_barcode_position(parse_data: dict[str, Any]) -> str | None:
    """Detect barcode position pattern from extraction data.

    Analyzes the ``_barcode_position`` metadata set by text_parser, or
    infers from line_items structure when barcodes are present.

    Returns ``"before_item"``, ``"after_item"``, ``"inline"``, or None.
    """
    # Prefer explicit metadata from text_parser
    position = parse_data.get("_barcode_position")
    if position in ("before_item", "after_item", "inline"):
        return str(position)

    # Need at least 2 barcoded items to learn a pattern
    items = parse_data.get("line_items", [])
    barcode_items = [i for i in items if i.get("barcode")]
    if len(barcode_items) < 2:
        return None

    return None


def extract_template_fingerprint(
    parse_data: dict[str, Any],
    ocr_text: str,
    confidence: float,
) -> VendorTemplate | None:
    """Extract template fingerprint from a successful extraction.

    Returns VendorTemplate if confidence is high enough, None otherwise.
    """
    if confidence < _LEARN_THRESHOLD:
        return None

    layout = detect_layout_type(parse_data)
    currency = parse_data.get("currency")
    pos = detect_pos_provider(ocr_text)
    bc_pos = detect_barcode_position(parse_data)
    date_fmt = parse_data.get("_date_format")
    total_mkr = parse_data.get("_total_marker")
    detected_lang = parse_data.get("_detected_language")
    header_lines = parse_data.get("_header_lines")
    footer_ratio = parse_data.get("_footer_ratio")

    return VendorTemplate(
        layout_type=layout,
        currency=currency,
        pos_provider=pos,
        barcode_position=bc_pos,
        success_count=1,
        date_format=date_fmt if isinstance(date_fmt, str) else None,
        total_marker=total_mkr if isinstance(total_mkr, str) else None,
        language=detected_lang or None,
        typical_header_lines=header_lines if isinstance(header_lines, int) else None,
        typical_footer_ratio=(
            footer_ratio if isinstance(footer_ratio, (int, float)) else None
        ),
    )


def merge_template(
    existing: VendorTemplate,
    new: VendorTemplate,
) -> VendorTemplate:
    """Merge a new template observation into an existing template.

    Increments success_count if layout matches. If layout changed,
    marks template as stale and resets counter.
    Preserves adaptive learning fields from the existing template.
    """
    # Resolve date_format across observations
    merged_date_fmt: str | None = None
    merged_date_conf: int = 0
    if new.date_format and new.date_format == existing.date_format:
        merged_date_fmt = existing.date_format
        merged_date_conf = existing.date_format_confidence + 1
    elif (
        new.date_format
        and existing.date_format
        and new.date_format != existing.date_format
    ):
        if existing.date_format_confidence > 1:
            merged_date_fmt = existing.date_format
            merged_date_conf = existing.date_format_confidence
        else:
            merged_date_fmt = new.date_format
            merged_date_conf = 1
    elif new.date_format:
        merged_date_fmt = new.date_format
        merged_date_conf = 1
    else:
        merged_date_fmt = existing.date_format
        merged_date_conf = existing.date_format_confidence

    # Resolve total_marker
    merged_total_marker = new.total_marker or existing.total_marker

    # Resolve typical_header_lines (rolling average)
    if (
        new.typical_header_lines is not None
        and existing.typical_header_lines is not None
    ):
        merged_header_lines: int | None = round(
            (existing.typical_header_lines + new.typical_header_lines) / 2
        )
    elif new.typical_header_lines is not None:
        merged_header_lines = new.typical_header_lines
    else:
        merged_header_lines = existing.typical_header_lines

    # Resolve typical_footer_ratio (rolling average)
    if (
        new.typical_footer_ratio is not None
        and existing.typical_footer_ratio is not None
    ):
        merged_footer_ratio: float | None = round(
            (existing.typical_footer_ratio + new.typical_footer_ratio) / 2, 2
        )
    elif new.typical_footer_ratio is not None:
        merged_footer_ratio = new.typical_footer_ratio
    else:
        merged_footer_ratio = existing.typical_footer_ratio

    def _layout_family(lt: str) -> str:
        return "columnar" if lt.startswith("columnar") else lt

    if _layout_family(existing.layout_type) == _layout_family(new.layout_type):
        # Prefer the more specific variant (columnar_4 > columnar)
        merged_layout = (
            new.layout_type
            if len(new.layout_type) >= len(existing.layout_type)
            else existing.layout_type
        )
        return VendorTemplate(
            layout_type=merged_layout,
            currency=new.currency or existing.currency,
            pos_provider=new.pos_provider or existing.pos_provider,
            success_count=existing.success_count + 1,
            # Preserve adaptive fields
            confidence_history=existing.confidence_history,
            adaptive_skip_threshold=existing.adaptive_skip_threshold,
            preferred_ocr_tier=existing.preferred_ocr_tier,
            needs_rotation=existing.needs_rotation,
            common_fixes=existing.common_fixes,
            last_updated=existing.last_updated,
            stale=existing.stale,
            # Preserve Gemini fields
            gemini_bootstrapped=existing.gemini_bootstrapped,
            language=existing.language,
            has_barcodes=existing.has_barcodes,
            barcode_position=existing.barcode_position or new.barcode_position,
            has_unit_quantities=existing.has_unit_quantities,
            typical_item_count=existing.typical_item_count,
            # Learned fields
            date_format=merged_date_fmt,
            date_format_confidence=merged_date_conf,
            total_marker=merged_total_marker,
            typical_header_lines=merged_header_lines,
            typical_footer_ratio=merged_footer_ratio,
        )
    else:
        # Layout changed — reset counter, adopt new layout, mark stale
        logger.info(
            f"Template layout changed: {existing.layout_type} -> {new.layout_type}"
        )
        return VendorTemplate(
            layout_type=new.layout_type,
            currency=new.currency or existing.currency,
            pos_provider=new.pos_provider or existing.pos_provider,
            success_count=1,
            # Preserve adaptive fields but mark stale
            confidence_history=existing.confidence_history,
            adaptive_skip_threshold=None,  # reset — layout changed
            preferred_ocr_tier=existing.preferred_ocr_tier,
            needs_rotation=existing.needs_rotation,
            common_fixes=existing.common_fixes,
            last_updated=existing.last_updated,
            stale=True,  # layout change triggers staleness
            # Preserve Gemini fields
            gemini_bootstrapped=existing.gemini_bootstrapped,
            language=existing.language,
            barcode_position=existing.barcode_position or new.barcode_position,
            # Learned fields
            date_format=merged_date_fmt,
            date_format_confidence=merged_date_conf,
            total_marker=merged_total_marker,
            typical_header_lines=merged_header_lines,
            typical_footer_ratio=merged_footer_ratio,
        )


def record_extraction_observation(
    template: VendorTemplate,
    confidence: float,
    ocr_tier: int | None = None,
    was_rotated: bool = False,
    fixes_applied: list[str] | None = None,
) -> VendorTemplate:
    """Record an extraction observation and update adaptive metrics.

    Returns a new VendorTemplate with updated metrics (immutable pattern).
    """
    from datetime import datetime, timezone

    # Append confidence, keep last _MAX_CONFIDENCE_HISTORY entries
    history = list(template.confidence_history) + [confidence]
    if len(history) > _MAX_CONFIDENCE_HISTORY:
        history = history[-_MAX_CONFIDENCE_HISTORY:]

    # Compute rolling average over last _ROLLING_WINDOW entries
    window = history[-_ROLLING_WINDOW:]
    rolling_avg = sum(window) / len(window)

    # Derive adaptive skip threshold from rolling average
    if rolling_avg > _ADAPTIVE_THRESHOLD_HIGH:
        adaptive_skip_threshold: float | None = _ADAPTIVE_THRESHOLD_HIGH
    elif rolling_avg < _ADAPTIVE_THRESHOLD_LOW:
        adaptive_skip_threshold = _ADAPTIVE_THRESHOLD_LOW
    else:
        adaptive_skip_threshold = None

    # Staleness detection: compare recent vs historical average
    stale = template.stale
    if len(history) >= _STALENESS_MIN_HISTORY:
        recent = history[-3:]  # last 3 observations
        historical = history[:-3] if len(history) > 3 else history
        recent_avg = sum(recent) / len(recent)
        historical_avg = sum(historical) / len(historical)
        if recent_avg < historical_avg - _STALENESS_DROP:
            if not stale:
                logger.warning(
                    f"Template staleness detected: recent avg "
                    f"{recent_avg:.2f} < historical {historical_avg:.2f} "
                    f"- {_STALENESS_DROP}"
                )
            stale = True
        elif stale and recent_avg >= historical_avg - (_STALENESS_DROP / 2):
            # Recovery: recent performance improved, clear stale flag
            logger.info("Template staleness cleared: confidence recovered")
            stale = False

    # Update preferred OCR tier: track the highest (worst) tier seen
    preferred_ocr_tier = template.preferred_ocr_tier
    if ocr_tier is not None and ocr_tier > 0:
        if preferred_ocr_tier is None:
            preferred_ocr_tier = ocr_tier
        else:
            preferred_ocr_tier = max(preferred_ocr_tier, ocr_tier)

    # Latch needs_rotation: once set, stays set
    needs_rotation = template.needs_rotation or was_rotated

    # Increment counters for any applied fix types
    common_fixes = dict(template.common_fixes)
    if fixes_applied:
        for fix_type in fixes_applied:
            common_fixes[fix_type] = common_fixes.get(fix_type, 0) + 1

    return VendorTemplate(
        layout_type=template.layout_type,
        currency=template.currency,
        pos_provider=template.pos_provider,
        success_count=template.success_count,
        gemini_bootstrapped=template.gemini_bootstrapped,
        language=template.language,
        has_barcodes=template.has_barcodes,
        barcode_position=template.barcode_position,
        has_unit_quantities=template.has_unit_quantities,
        typical_item_count=template.typical_item_count,
        confidence_history=history,
        adaptive_skip_threshold=adaptive_skip_threshold,
        preferred_ocr_tier=preferred_ocr_tier,
        needs_rotation=needs_rotation,
        common_fixes=common_fixes,
        last_updated=datetime.now(timezone.utc).isoformat(),
        stale=stale,
        date_format=template.date_format,
        date_format_confidence=template.date_format_confidence,
        total_marker=template.total_marker,
        typical_header_lines=template.typical_header_lines,
        typical_footer_ratio=template.typical_footer_ratio,
    )


def template_to_hints(
    template: VendorTemplate,
    vendor_name: str | None = None,
) -> ParserHints:
    """Convert a vendor template to parser hints.

    Only produces hints from reliable templates (success_count >= threshold).
    Stale templates provide degraded hints (vendor_name only, no layout/currency).
    """
    if not template.is_reliable:
        return ParserHints(vendor_name=vendor_name)

    # Collect fields that have been repeatedly corrected (likely unreliable)
    unreliable: list[str] = [
        fix_type
        for fix_type, count in template.common_fixes.items()
        if count >= _UNRELIABLE_FIX_COUNT
    ]

    # Stale templates: provide degraded hints (don't trust layout/currency)
    if template.stale:
        return ParserHints(
            vendor_name=vendor_name,
            unreliable_fields=unreliable or None,
        )

    return ParserHints(
        vendor_name=vendor_name,
        currency=template.currency,
        layout_type=template.layout_type,
        pos_provider=template.pos_provider,
        barcode_position=template.barcode_position,
        unreliable_fields=unreliable or None,
        date_format=template.date_format,
        total_marker=template.total_marker,
        expected_header_lines=template.typical_header_lines,
    )


# Minimum items sharing a category to auto-derive vendor default
_DEFAULT_CATEGORY_MIN_ITEMS = 5


def derive_vendor_default_category(
    db: Any,
    vendor_key: str,
) -> str | None:
    """Derive a default category for a vendor from its items.

    If 5+ items from this vendor all share the same category,
    returns that category. Otherwise returns None.
    """
    rows = db.fetchall(
        "SELECT fi.category, COUNT(*) AS cnt "
        "FROM fact_items fi "
        "JOIN facts f ON fi.fact_id = f.id "
        "WHERE f.vendor_key = ? AND fi.category IS NOT NULL "
        "AND fi.category != '' "
        "GROUP BY fi.category "
        "ORDER BY cnt DESC",
        (vendor_key,),
    )
    if not rows:
        return None
    top = rows[0]
    if top["cnt"] >= _DEFAULT_CATEGORY_MIN_ITEMS:
        return str(top["category"])
    return None


# -------------------------------------------------------------------
# Identity metadata CRUD for templates
# -------------------------------------------------------------------

_TEMPLATE_KEY = "template"
_VENDOR_DETAILS_KEY = "vendor_details"

# Fields that are stable per-vendor (don't change between documents)
_VENDOR_DETAIL_FIELDS = {
    "address",
    "phone",
    "website",
    "legal_name",
    "vat",
    "tax_id",
}


def load_vendor_template(
    db: Any,
    identity_id: str,
) -> VendorTemplate | None:
    """Load vendor template from identity metadata.

    Returns VendorTemplate or None if no template stored.
    """
    from alibi.identities.store import get_identity

    identity = get_identity(db, identity_id)
    if not identity:
        return None

    metadata = identity.get("metadata") or {}
    template_data = metadata.get(_TEMPLATE_KEY)
    if not template_data:
        return None

    try:
        return VendorTemplate.from_dict(template_data)
    except Exception as e:
        logger.debug(f"Failed to load template for {identity_id[:8]}: {e}")
        return None


def save_vendor_template(
    db: Any,
    identity_id: str,
    template: VendorTemplate,
) -> None:
    """Save vendor template to identity metadata.

    Read-modify-write to preserve other metadata fields.
    """
    from alibi.identities.store import get_identity, update_identity

    identity = get_identity(db, identity_id)
    if not identity:
        logger.debug(f"Cannot save template: identity {identity_id[:8]} not found")
        return

    metadata = identity.get("metadata") or {}
    metadata[_TEMPLATE_KEY] = template.to_dict()
    update_identity(db, identity_id, metadata=metadata)
    logger.info(
        f"Saved template for {identity.get('canonical_name', '?')}: "
        f"layout={template.layout_type}, count={template.success_count}"
    )


def load_vendor_details(
    db: Any,
    identity_id: str,
) -> dict[str, str]:
    """Load cached vendor details from identity metadata.

    Returns dict of vendor details (address, phone, website, etc.)
    or empty dict if none stored.
    """
    from alibi.identities.store import get_identity

    identity = get_identity(db, identity_id)
    if not identity:
        return {}

    metadata = identity.get("metadata") or {}
    details: dict[str, str] = metadata.get(_VENDOR_DETAILS_KEY, {})
    return details


def save_vendor_details(
    db: Any,
    identity_id: str,
    details: dict[str, str],
) -> None:
    """Save vendor contact details to identity metadata.

    Read-modify-write to preserve other metadata fields.
    Only stores non-empty values from the allowed field set.
    """
    from alibi.identities.store import get_identity, update_identity

    identity = get_identity(db, identity_id)
    if not identity:
        return

    # Filter to known fields, skip empty values
    clean = {k: v for k, v in details.items() if k in _VENDOR_DETAIL_FIELDS and v}
    if not clean:
        return

    metadata = identity.get("metadata") or {}
    existing = metadata.get(_VENDOR_DETAILS_KEY, {})

    # Merge: new values fill gaps, don't overwrite existing
    merged = dict(existing)
    for k, v in clean.items():
        if k not in merged or not merged[k]:
            merged[k] = v

    metadata[_VENDOR_DETAILS_KEY] = merged
    update_identity(db, identity_id, metadata=metadata)
    logger.info(
        f"Saved vendor details for {identity.get('canonical_name', '?')}: "
        f"{list(merged.keys())}"
    )


def _infer_date_format_from_correction(
    old_iso: str,
    new_iso: str,
) -> str | None:
    """Infer date format from a DD/MM vs MM/DD swap correction.

    Both values must be ISO dates (YYYY-MM-DD). If swapping day and month
    in the old date produces the new date, the user is correcting a
    DD/MM vs MM/DD misinterpretation.

    Returns "dmy" or "mdy" indicating the format the user intends,
    or None if the correction is not a day/month swap.
    """
    from datetime import date

    try:
        old_date = date.fromisoformat(old_iso)
        new_date = date.fromisoformat(new_iso)
    except (ValueError, TypeError):
        return None

    if old_date == new_date:
        return None

    # Check if swapping day/month converts old to new
    if (
        old_date.year == new_date.year
        and old_date.month == new_date.day
        and old_date.day == new_date.month
    ):
        # The parser produced old_date by misreading the raw text.
        # The user says new_date is correct.
        # old_date.month == new_date.day: parser put the real day into
        # the month slot. If new_date.month < old_date.month, the real
        # month is smaller (was in the day slot) -> raw format is dmy.
        if new_date.month < old_date.month:
            return "dmy"
        else:
            return "mdy"

    return None


def apply_correction_to_template(
    db: Any,
    vendor_key: str,
    field: str,
    old_value: str,
    new_value: str,
) -> None:
    """Update vendor template based on a user correction.

    When the user corrects a date, infer the date format from the new value
    and update the template. This teaches the system to use the correct
    format for future documents from this vendor.
    """
    if field != "date":
        return

    fmt = _infer_date_format_from_correction(old_value, new_value)
    if not fmt:
        return

    template, identity_id, canonical = find_template_for_vendor(
        db, vendor_key=vendor_key
    )

    if template and identity_id:
        template.date_format = fmt
        template.date_format_confidence = max(template.date_format_confidence, 1)
        save_vendor_template(db, identity_id, template)
        logger.info(
            "Correction-driven template update: vendor_key=%s date_format=%s",
            vendor_key,
            fmt,
        )
    elif identity_id:
        # Identity exists but no template yet -- create a minimal one
        new_tpl = VendorTemplate(
            date_format=fmt,
            date_format_confidence=1,
            success_count=1,
        )
        save_vendor_template(db, identity_id, new_tpl)
        logger.info(
            "Correction-driven template created: vendor_key=%s date_format=%s",
            vendor_key,
            fmt,
        )
    else:
        logger.debug(
            "No identity found for vendor_key=%s; skipping template correction",
            vendor_key,
        )


def find_template_for_vendor(
    db: Any,
    vendor_name: str | None = None,
    vendor_vat: str | None = None,
    vendor_key: str | None = None,
) -> tuple[VendorTemplate | None, str | None, str | None]:
    """Find template for a vendor by any matching signal.

    Returns (template, identity_id, canonical_name) or (None, None, None).
    """
    from alibi.identities.matching import find_vendor_identity

    identity = find_vendor_identity(
        db,
        vendor_name=vendor_name,
        vendor_key=vendor_key,
        registration=vendor_vat,
    )
    if not identity:
        return None, None, None

    identity_id = identity["id"]
    canonical = identity.get("canonical_name")
    template = load_vendor_template(db, identity_id)
    return template, identity_id, canonical


# -------------------------------------------------------------------
# POS provider identity
# -------------------------------------------------------------------


def ensure_pos_identity(
    db: Any,
    pos_provider: str,
    template: VendorTemplate | None = None,
) -> str | None:
    """Find or create a POS provider identity.

    POS identities use entity_type='pos_provider' and store
    a base template that applies to all vendors using this POS.
    """
    from alibi.identities.store import (
        create_identity,
        get_identity,
        update_identity,
    )

    conn = db.get_connection()
    # Look up existing POS identity by name
    row = conn.execute(
        "SELECT id FROM identities WHERE entity_type = 'pos_provider' "
        "AND canonical_name = ?",
        (pos_provider,),
    ).fetchone()

    if row:
        identity_id: str = str(row["id"])
        # Merge template if provided
        if template:
            existing = load_vendor_template(db, identity_id)
            if existing:
                merged = merge_template(existing, template)
                save_vendor_template(db, identity_id, merged)
            else:
                save_vendor_template(db, identity_id, template)
        return identity_id

    # Create new POS identity
    metadata = {}
    if template:
        metadata[_TEMPLATE_KEY] = template.to_dict()

    identity_id = create_identity(
        db,
        entity_type="pos_provider",
        canonical_name=pos_provider,
        metadata=metadata or None,
    )
    logger.info(f"Created POS provider identity: {pos_provider}")
    return identity_id


def load_pos_template(
    db: Any,
    pos_provider: str,
) -> VendorTemplate | None:
    """Load template for a POS provider.

    Returns VendorTemplate or None if no POS identity/template exists.
    """
    conn = db.get_connection()
    row = conn.execute(
        "SELECT id FROM identities WHERE entity_type = 'pos_provider' "
        "AND canonical_name = ?",
        (pos_provider,),
    ).fetchone()
    if not row:
        return None
    return load_vendor_template(db, row["id"])


def find_template_by_location(
    db: Any,
    lat: float,
    lng: float,
    radius_m: float = 100.0,
) -> tuple[VendorTemplate | None, str | None, str | None]:
    """Find vendor template by proximity to a known location.

    Looks up location annotations to find vendors at the same coordinates.
    If a vendor with a template is found within radius, returns it.

    Returns (template, identity_id, canonical_name) or (None, None, None).
    """
    from alibi.utils.map_url import haversine_distance

    conn = db.get_connection()
    rows = conn.execute(
        """
        SELECT DISTINCT
            f.vendor_key,
            f.vendor AS vendor_name,
            a.metadata
        FROM annotations a
        JOIN facts f ON f.id = a.target_id
        WHERE a.annotation_type = 'location'
          AND a.target_type = 'fact'
          AND a.metadata IS NOT NULL
          AND f.vendor_key IS NOT NULL
        ORDER BY a.created_at DESC
        LIMIT 50
        """,
    ).fetchall()

    for row in rows:
        meta = (
            json.loads(row["metadata"])
            if isinstance(row["metadata"], str)
            else row["metadata"]
        )
        ann_lat = meta.get("lat")
        ann_lng = meta.get("lng")
        if ann_lat is None or ann_lng is None:
            continue
        dist = haversine_distance(lat, lng, ann_lat, ann_lng)
        if dist <= radius_m:
            # Found a vendor at this location — load their template
            tpl, identity_id, canonical = find_template_for_vendor(
                db,
                vendor_key=row["vendor_key"],
            )
            if tpl:
                logger.info(
                    f"Location match: {row['vendor_name']} at {dist:.0f}m "
                    f"(template: {tpl.layout_type})"
                )
                return tpl, identity_id, canonical
            elif identity_id:
                return None, identity_id, canonical

    return None, None, None


@dataclass
class _TemplateCandidate:
    """A template candidate from one signal source."""

    template: VendorTemplate
    identity_id: str | None
    canonical_name: str | None
    source: str  # "vendor", "location", "pos"
    priority: int  # lower = higher priority

    @property
    def score(self) -> float:
        """Rank: priority first, then success_count (reliability)."""
        return -self.priority * 1000 + self.template.success_count


def resolve_hints(
    db: Any,
    vendor_name: str | None = None,
    vendor_vat: str | None = None,
    vendor_key: str | None = None,
    ocr_text: str | None = None,
    lat: float | None = None,
    lng: float | None = None,
) -> tuple[ParserHints | None, str | None]:
    """Resolve parser hints from multiple independent signals.

    Each signal (vendor identity, location, POS provider) is resolved
    independently. If one fails, others are still tried. The best
    reliable template wins by priority: vendor > location > POS.

    This is a fail-safe multi-signal pattern: no single signal failure
    blocks the entire template resolution process.

    Returns (hints, identity_id) or (None, None) if no template found.
    """
    candidates: list[_TemplateCandidate] = []
    identity_id: str | None = None
    canonical: str | None = None

    # Signal 1: Vendor identity (highest priority)
    try:
        vendor_tpl, vid, vname = find_template_for_vendor(
            db,
            vendor_name=vendor_name,
            vendor_vat=vendor_vat,
            vendor_key=vendor_key,
        )
        if vid:
            identity_id = vid
            canonical = vname
        if vendor_tpl:
            candidates.append(_TemplateCandidate(vendor_tpl, vid, vname, "vendor", 0))
    except Exception as e:
        logger.debug(f"Vendor template signal failed: {e}")

    # Signal 2: Location proximity (medium priority)
    if lat is not None and lng is not None:
        try:
            loc_tpl, loc_id, loc_name = find_template_by_location(db, lat, lng)
            if loc_tpl:
                candidates.append(
                    _TemplateCandidate(loc_tpl, loc_id, loc_name, "location", 1)
                )
                if not identity_id and loc_id:
                    identity_id = loc_id
                    canonical = loc_name
        except Exception as e:
            logger.debug(f"Location template signal failed: {e}")

    # Signal 3: POS provider (lowest priority)
    if ocr_text:
        try:
            pos = detect_pos_provider(ocr_text)
            if pos:
                pos_tpl = load_pos_template(db, pos)
                if pos_tpl:
                    candidates.append(
                        _TemplateCandidate(pos_tpl, identity_id, canonical, "pos", 2)
                    )
        except Exception as e:
            logger.debug(f"POS template signal failed: {e}")

    # Pick best reliable candidate
    reliable = [c for c in candidates if c.template.is_reliable]
    if reliable:
        best = max(reliable, key=lambda c: c.score)
        hints = template_to_hints(best.template, vendor_name=best.canonical_name)
        logger.info(
            f"Template resolved via {best.source}: "
            f"layout={best.template.layout_type}, "
            f"count={best.template.success_count}"
        )
        return hints, best.identity_id or identity_id

    # No reliable template — return vendor name hint if known
    if canonical:
        return ParserHints(vendor_name=canonical), identity_id

    return None, identity_id
