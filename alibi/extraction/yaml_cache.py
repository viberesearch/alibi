"""YAML intermediary file support for extraction caching.

YAML caches live in a dedicated store tree, decoupled from source documents:

    ``<yaml_store>/<user_id>/<doc_type>/<stem>.alibi.yaml``

Default store: ``data/yaml_store/`` (configurable via ``ALIBI_YAML_STORE``).
Source files can be purged without affecting YAML caches.

On re-processing, reads from YAML if present (skips LLM). Users can edit
the YAML to correct OCR errors, add barcodes, fill missing fields before
re-ingestion.

Outputs all expected fields per document type with empty values for
unextracted fields, making the file a self-documenting template.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

YAML_SUFFIX = ".alibi.yaml"
YAML_VERSION = 5

# ---------------------------------------------------------------------------
# YAML store root
# ---------------------------------------------------------------------------
_yaml_store_root: Path | None = None
_yaml_store_initialized = False


def _get_yaml_store_root() -> Path | None:
    """Get the yaml store root directory, lazily initialized from config.

    Returns None only in testing when no explicit root is set.
    """
    global _yaml_store_root, _yaml_store_initialized
    if not _yaml_store_initialized:
        try:
            from alibi.config import get_config

            _yaml_store_root = get_config().get_yaml_store_path()
        except Exception:
            _yaml_store_root = None
        _yaml_store_initialized = True
    return _yaml_store_root


def set_yaml_store_root(root: Path | None) -> None:
    """Override the yaml store root (for testing)."""
    global _yaml_store_root, _yaml_store_initialized
    _yaml_store_root = root
    _yaml_store_initialized = True


def reset_yaml_store() -> None:
    """Reset to config-derived value on next access."""
    global _yaml_store_initialized
    _yaml_store_initialized = False


# ---------------------------------------------------------------------------
# Document-type field templates
# All expected fields per document type. Missing values are written as empty
# strings or None so users can fill them in.
# ---------------------------------------------------------------------------

_RECEIPT_FIELDS: dict[str, Any] = {
    "vendor": "",
    "vendor_legal_name": "",
    "vendor_address": "",
    "vendor_phone": "",
    "vendor_website": "",
    "vendor_vat": "",
    "vendor_tax_id": "",
    "date": "",
    "time": "",
    "currency": "",
    "language": "",
    "total": None,
    "subtotal": None,
    "tax_total": None,
    "discount": None,
    "payment_method": "",
    "amount_tendered": None,
    "change_due": None,
    "card_type": "",
    "card_last4": "",
    "authorization_code": "",
    "terminal_id": "",
    "merchant_id": "",
    "vat_analysis": {},
}

_RECEIPT_ITEM_FIELDS: dict[str, Any] = {
    "name": "",
    "quantity": 1,
    "unit": "pcs",
    "unit_raw": "",
    "unit_quantity": None,
    "unit_price": None,
    "total_price": None,
    "tax_code": "",
    "tax_rate": None,
    "discount": None,
    "brand": "",
    "category": "",
    "barcode": "",
}

_PAYMENT_FIELDS: dict[str, Any] = {
    "vendor": "",
    "vendor_legal_name": "",
    "vendor_vat": "",
    "vendor_tax_id": "",
    "date": "",
    "time": "",
    "currency": "",
    "total": None,
    "payment_method": "",
    "amount_tendered": None,
    "change_due": None,
    "card_type": "",
    "card_last4": "",
    "authorization_code": "",
    "terminal_id": "",
    "merchant_id": "",
}

_INVOICE_FIELDS: dict[str, Any] = {
    "issuer": "",
    "issuer_legal_name": "",
    "issuer_address": "",
    "issuer_phone": "",
    "issuer_website": "",
    "issuer_vat": "",
    "issuer_tax_id": "",
    "invoice_number": "",
    "issue_date": "",
    "due_date": "",
    "customer": "",
    "billing_address": "",
    "payment_terms": "",
    "po_number": "",
    "currency": "",
    "language": "",
    "amount": None,
    "subtotal": None,
    "tax_total": None,
    "discount": None,
    "vat_analysis": {},
}

_INVOICE_ITEM_FIELDS: dict[str, Any] = {
    "name": "",
    "quantity": 1,
    "unit": "pcs",
    "unit_price": None,
    "total_price": None,
    "tax_code": "",
    "tax_rate": None,
    "discount": None,
    "barcode": "",
}

_STATEMENT_FIELDS: dict[str, Any] = {
    "bank": "",
    "account_number": "",
    "iban": "",
    "statement_period_start": "",
    "statement_period_end": "",
    "opening_balance": None,
    "closing_balance": None,
    "currency": "",
}

_STATEMENT_TXN_FIELDS: dict[str, Any] = {
    "date": "",
    "description": "",
    "amount": None,
    "balance": None,
    "type": "",
}

_CONTRACT_FIELDS: dict[str, Any] = {
    "vendor": "",
    "vendor_legal_name": "",
    "vendor_vat": "",
    "vendor_tax_id": "",
    "date": "",
    "effective_date": "",
    "expiration_date": "",
    "payment_terms": "",
    "renewal_terms": "",
    "termination_terms": "",
    "total_value": None,
    "currency": "",
}

_WARRANTY_FIELDS: dict[str, Any] = {
    "vendor": "",
    "vendor_legal_name": "",
    "product": "",
    "serial_number": "",
    "purchase_date": "",
    "warranty_start": "",
    "warranty_end": "",
    "warranty_type": "",
    "coverage": "",
}

_TYPE_FIELDS: dict[str, dict[str, Any]] = {
    "receipt": _RECEIPT_FIELDS,
    "payment_confirmation": _PAYMENT_FIELDS,
    "invoice": _INVOICE_FIELDS,
    "statement": _STATEMENT_FIELDS,
    "contract": _CONTRACT_FIELDS,
    "warranty": _WARRANTY_FIELDS,
}

_TYPE_ITEM_FIELDS: dict[str, dict[str, Any]] = {
    "receipt": _RECEIPT_ITEM_FIELDS,
    "invoice": _INVOICE_ITEM_FIELDS,
    "statement": _STATEMENT_TXN_FIELDS,
}


SUPPORTED_DOCUMENT_TYPES = list(_TYPE_FIELDS.keys())


def generate_blank_template(
    document_type: str, include_meta: bool = True
) -> dict[str, Any]:
    """Generate a blank YAML template for a document type.

    Returns dict ready for yaml.dump() with all expected fields as empty defaults.
    Useful for manual YAML creation without running alibi extraction.

    Args:
        document_type: One of SUPPORTED_DOCUMENT_TYPES.
        include_meta: Whether to include the _meta block.

    Returns:
        Template dict.

    Raises:
        ValueError: If document_type is not supported.
    """
    if document_type not in _TYPE_FIELDS:
        raise ValueError(
            f"Unknown document type '{document_type}'. "
            f"Supported: {', '.join(SUPPORTED_DOCUMENT_TYPES)}"
        )

    output: dict[str, Any] = {}
    if include_meta:
        output["_meta"] = {
            "version": YAML_VERSION,
            "extracted_at": "",
            "source": "",
        }
    output["document_type"] = document_type

    for key, default in _TYPE_FIELDS[document_type].items():
        output[key] = default

    item_template = _TYPE_ITEM_FIELDS.get(document_type)
    if item_template is not None:
        items_key = "transactions" if document_type == "statement" else "line_items"
        output[items_key] = [dict(item_template)]

    return output


def get_yaml_path(
    source_path: Path,
    is_group: bool = False,
    *,
    user_id: str = "system",
    doc_type: str = "unsorted",
) -> Path:
    """Get the .alibi.yaml path for a source document.

    Returns ``<yaml_store>/<user_id>/<doc_type>/<stem>.alibi.yaml``.

    Falls back to sidecar placement (next to source) only when no store is
    configured (testing without explicit store setup).

    Args:
        source_path: Path to the source file or folder
        is_group: True if source_path is a folder (document group)
        user_id: Owner user ID (default "system")
        doc_type: Document type for store organization (default "unsorted")

    Returns:
        Path to the corresponding .alibi.yaml file
    """
    store = _get_yaml_store_root()
    if store is not None:
        stem = source_path.name if is_group else source_path.stem
        return store / user_id / doc_type / f"{stem}{YAML_SUFFIX}"
    # Test-only fallback: sidecar next to source
    if is_group:
        return source_path / YAML_SUFFIX
    return source_path.parent / f"{source_path.stem}{YAML_SUFFIX}"


def read_yaml_cache(
    source_path: Path,
    is_group: bool = False,
    *,
    user_id: str = "system",
    doc_type: str = "unsorted",
) -> dict[str, Any] | None:
    """Read cached extraction data from .alibi.yaml if it exists.

    Requires YAML_VERSION match. Stale caches are rejected — re-extract.

    Args:
        source_path: Path to the source file or folder
        is_group: True if source_path is a folder (document group)
        user_id: Owner user ID for store lookup
        doc_type: Document type for store lookup

    Returns:
        Extracted data dict if YAML exists and is valid, None otherwise
    """
    yaml_path = get_yaml_path(source_path, is_group, user_id=user_id, doc_type=doc_type)

    if not yaml_path.exists():
        return None

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            logger.warning(f"Invalid YAML structure in {yaml_path}")
            return None

        version = data.get("_meta", {}).get("version")
        if version != YAML_VERSION:
            logger.warning(
                f"YAML version {version} != required {YAML_VERSION} "
                f"in {yaml_path}, re-extract needed"
            )
            return None

        # Strip metadata, return the extraction data.
        extracted = {k: v for k, v in data.items() if k != "_meta"}

        logger.info(f"Loaded extraction from YAML cache: {yaml_path}")
        return extracted

    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML {yaml_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to read YAML cache {yaml_path}: {e}")
        return None


def write_yaml_cache(
    source_path: Path,
    extracted_data: dict[str, Any],
    document_type: str,
    is_group: bool = False,
    ocr_model: str | None = None,
    structure_model: str | None = None,
    confidence: float | None = None,
    ocr_text: str | None = None,
    pipeline: str | None = None,
    parser_confidence: float | None = None,
    parser_gaps: list[str] | None = None,
    file_hash: str | None = None,
    perceptual_hash: str | None = None,
    needs_review: bool = False,
    user_id: str = "system",
) -> Path | None:
    """Write extraction data as .alibi.yaml.

    When yaml_store is configured, writes to the store tree.
    Otherwise writes as a sidecar alongside the source document.

    Outputs all expected fields per document type with empty/null values
    for unextracted fields. Users can fill in missing values (barcodes,
    vendor details, etc.) and re-process to update the database.

    Args:
        source_path: Path to the source file or folder
        extracted_data: Raw LLM extraction output
        document_type: Detected document type string
        is_group: True if source_path is a folder (document group)
        ocr_model: OCR model name
        structure_model: Structure model name
        confidence: Verification confidence score (0-1)
        ocr_text: Raw OCR text
        pipeline: Pipeline identifier
        parser_confidence: Heuristic parser confidence
        parser_gaps: Fields the parser couldn't fill
        needs_review: If True, mark _meta.needs_review=true for human review
        file_hash: SHA-256 of source file (stored in _meta for source-less re-ingestion)
        perceptual_hash: dHash of source image (stored in _meta)
        user_id: Owner user ID for store path

    Returns:
        Path to the written YAML file, or None on failure
    """
    yaml_path = get_yaml_path(
        source_path, is_group, user_id=user_id, doc_type=document_type
    )

    # Build YAML-friendly output
    output: dict[str, Any] = {
        "_meta": {
            "version": YAML_VERSION,
            "extracted_at": datetime.now().isoformat(),
            "source": str(source_path),
            "source_path": str(source_path.resolve()),
            "is_group": is_group,
        },
        "document_type": document_type,
    }

    if file_hash:
        output["_meta"]["file_hash"] = file_hash
    if perceptual_hash:
        output["_meta"]["perceptual_hash"] = perceptual_hash
    if ocr_model:
        output["_meta"]["ocr_model"] = ocr_model
    if structure_model:
        output["_meta"]["structure_model"] = structure_model
    if confidence is not None:
        output["_meta"]["confidence"] = round(confidence, 3)
    if ocr_text:
        output["_meta"]["ocr_text"] = ocr_text
    if pipeline:
        output["_meta"]["pipeline"] = pipeline
    if parser_confidence is not None:
        output["_meta"]["parser_confidence"] = round(parser_confidence, 3)
    if parser_gaps:
        output["_meta"]["parser_gaps"] = parser_gaps
    if needs_review:
        output["_meta"]["needs_review"] = True

    # Start with the template fields for this document type,
    # then overlay with extracted data.
    template = _TYPE_FIELDS.get(document_type, {})
    item_template = _TYPE_ITEM_FIELDS.get(document_type)

    # Write template fields first (provides structure for user editing)
    for key, default in template.items():
        value = extracted_data.get(key)
        if value is not None and value != "":
            output[key] = _yaml_safe_value(value)
        else:
            output[key] = default

    # Write non-template fields from extraction (preserves any extra data)
    for key, value in extracted_data.items():
        if key == "document_type":
            continue
        if key.startswith("_"):
            continue
        if key in template:
            continue  # Already written above
        if key in ("line_items", "items", "transactions"):
            continue  # Handled below
        output[key] = _yaml_safe_value(value)

    # Write line items / transactions with complete field templates
    items_key = "transactions" if document_type == "statement" else "line_items"
    raw_items = extracted_data.get(items_key) or extracted_data.get("line_items") or []

    if item_template is not None:
        complete_items = []
        for raw_item in raw_items:
            item: dict[str, Any] = {}
            # Template fields first
            for field, default in item_template.items():
                value = raw_item.get(field) if isinstance(raw_item, dict) else None
                if value is not None and value != "":
                    item[field] = _yaml_safe_value(value)
                else:
                    item[field] = default
            # Extra fields from extraction
            if isinstance(raw_item, dict):
                for field, value in raw_item.items():
                    if field not in item_template:
                        item[field] = _yaml_safe_value(value)
            complete_items.append(item)
        output[items_key] = complete_items
    elif raw_items:
        output[items_key] = [_yaml_safe_value(item) for item in raw_items]

    try:
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(
                output,
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=120,
            )

        logger.info(f"Wrote extraction cache: {yaml_path}")

        # Track for git versioning (best-effort)
        try:
            from alibi.mycelium.yaml_versioning import get_yaml_versioner

            get_yaml_versioner().track(yaml_path)
        except Exception as ve:
            logger.debug(f"YAML git tracking skipped: {ve}")

        return yaml_path

    except Exception as e:
        logger.warning(f"Failed to write YAML cache {yaml_path}: {e}")
        return None


def read_yaml_with_meta(
    source_path: Path,
    is_group: bool = False,
    *,
    user_id: str = "system",
    doc_type: str = "unsorted",
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Read cached extraction data and _meta from .alibi.yaml.

    Like read_yaml_cache but returns (extracted_data, meta_dict) instead
    of stripping meta. Returns None on missing, invalid, or version mismatch.
    """
    yaml_path = get_yaml_path(source_path, is_group, user_id=user_id, doc_type=doc_type)

    if not yaml_path.exists():
        return None

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            logger.warning(f"Invalid YAML structure in {yaml_path}")
            return None

        meta = data.get("_meta", {})
        version = meta.get("version")
        if version != YAML_VERSION:
            logger.warning(
                f"YAML version {version} != required {YAML_VERSION} "
                f"in {yaml_path}, re-extract needed"
            )
            return None

        extracted = {k: v for k, v in data.items() if k != "_meta"}
        logger.info(f"Loaded extraction+meta from YAML: {yaml_path}")
        return extracted, meta

    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML {yaml_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Failed to read YAML {yaml_path}: {e}")
        return None


def compute_yaml_hash(
    source_path: Path,
    is_group: bool = False,
    *,
    user_id: str = "system",
    doc_type: str = "unsorted",
) -> str | None:
    """Compute SHA-256 of the .alibi.yaml file bytes.

    Returns None if the YAML file doesn't exist.
    """
    import hashlib

    yaml_path = get_yaml_path(source_path, is_group, user_id=user_id, doc_type=doc_type)
    if not yaml_path.exists():
        return None

    try:
        return hashlib.sha256(yaml_path.read_bytes()).hexdigest()
    except OSError as e:
        logger.warning(f"Failed to hash YAML {yaml_path}: {e}")
        return None


def resolve_source_from_yaml(yaml_path: Path) -> tuple[Path, bool] | None:
    """Reverse of get_yaml_path: find the source document for a YAML file.

    Reads ``_meta.source_path`` and ``_meta.is_group`` from the YAML.
    Returns ``(source_path, is_group)`` or None if the YAML is missing/invalid.
    """
    if not yaml_path.exists():
        return None

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None
        meta = data.get("_meta", {})
        source_path_str = meta.get("source_path")
        if not source_path_str:
            logger.warning(f"YAML missing _meta.source_path: {yaml_path}")
            return None
        return Path(source_path_str), bool(meta.get("is_group", False))
    except Exception as e:
        logger.warning(f"Failed to resolve source from {yaml_path}: {e}")
        return None


def read_yaml_direct(
    yaml_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Read extracted data and _meta from a known .alibi.yaml path.

    Like read_yaml_with_meta but takes the YAML path directly instead of
    resolving it from a source path. Returns (extracted_data, meta) or None.
    """
    if not yaml_path.exists():
        return None
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return None
        meta = data.get("_meta", {})
        if meta.get("version") != YAML_VERSION:
            return None
        extracted = {k: v for k, v in data.items() if k != "_meta"}
        return extracted, meta
    except Exception as e:
        logger.warning(f"Failed to read YAML {yaml_path}: {e}")
        return None


def find_yaml_in_store(
    source_path: Path,
    is_group: bool = False,
    *,
    user_id: str = "system",
) -> Path | None:
    """Search the yaml_store for a YAML matching source_path stem.

    Searches across all doc_type subdirectories under the user_id.
    Use when the doc_type is unknown (e.g., before extraction determines it).

    Returns the first match, or None.
    """
    store = _get_yaml_store_root()
    if store is None:
        return None
    stem = source_path.name if is_group else source_path.stem
    user_dir = store / user_id
    if not user_dir.exists():
        return None
    target = f"{stem}{YAML_SUFFIX}"
    for match in user_dir.rglob(target):
        return match
    return None


def scan_yaml_store() -> list[Path]:
    """Find all .alibi.yaml files in the yaml_store.

    Returns:
        Sorted list of paths to .alibi.yaml files.
    """
    store = _get_yaml_store_root()
    if store is None or not store.exists():
        return []
    return sorted(store.rglob(f"*{YAML_SUFFIX}"))


def _yaml_safe_value(value: Any) -> Any:
    """Convert value to YAML-safe type.

    Handles Decimal, date, datetime, and nested structures.
    """
    from datetime import date, datetime
    from decimal import Decimal

    if isinstance(value, Decimal):
        return float(value)
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, date):
        return value.isoformat()
    elif isinstance(value, dict):
        return {k: _yaml_safe_value(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_yaml_safe_value(item) for item in value]
    elif hasattr(value, "value"):
        # Enum-like objects
        return value.value
    return value
