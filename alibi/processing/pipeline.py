"""Document processing pipeline."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from alibi.config import get_config
from alibi.atoms.parser import parse_extraction
from alibi.clouds.collapse import try_collapse
from alibi.clouds.formation import (
    add_bundle_to_cloud,
    create_cloud_for_bundle,
    extract_bundle_summary,
    find_cloud_for_bundle,
)
from alibi.db.connection import DatabaseManager
from alibi.db import v2_store
from alibi.normalizers.units import init_unit_mappings
from alibi.db.models import (
    BundleType,
    Document,
    DocumentType,
    RecordType,
)
from alibi.extraction.historical import apply_historical_corrections
from alibi.extraction.pdf import PDFExtractionError, extract_from_pdf
from alibi.extraction.prompts import classify_from_extraction
from alibi.extraction.verification import verify_extraction
from alibi.extraction.vision import (
    VisionExtractionError,
    detect_document_type as vision_detect_document_type,
    extract_from_image,
    extract_from_images,
)
from alibi.extraction.yaml_cache import (
    YAML_VERSION,
    compute_yaml_hash,
    find_yaml_in_store,
    get_yaml_path,
    read_yaml_cache,
    read_yaml_with_meta,
    write_yaml_cache,
)
from alibi.matching.duplicates import (
    canonicalize_vendor,
    compute_file_hash,
    compute_perceptual_hash,
    init_vendor_mappings,
    is_image_file,
)
from alibi.extraction.templates import (
    ParserHints,
    extract_template_fingerprint,
    detect_pos_provider,
    ensure_pos_identity,
    find_template_for_vendor,
    load_vendor_details,
    load_vendor_template,
    merge_template,
    record_extraction_observation,
    resolve_hints,
    save_vendor_details,
    save_vendor_template,
)
from alibi.processing.folder_router import FolderContext
from alibi.processing.watcher import is_supported_file
from alibi.refiners.registry import get_refiner

logger = logging.getLogger(__name__)

# Per-type skip-LLM thresholds when folder-routed (parser runs correct type
# from the start, so its confidence is more reliable).
_FOLDER_ROUTED_SKIP_THRESHOLD: dict[DocumentType, float] = {
    DocumentType.RECEIPT: 0.8,
    DocumentType.PAYMENT_CONFIRMATION: 0.8,
    DocumentType.STATEMENT: 0.85,
    DocumentType.INVOICE: 0.85,
}

# Single source-of-truth: string label → DocumentType.
# Used by vision detection, YAML cache type override, and LLM type override.
STR_TO_ARTIFACT_TYPE: dict[str, DocumentType] = {
    "receipt": DocumentType.RECEIPT,
    "invoice": DocumentType.INVOICE,
    "payment_confirmation": DocumentType.PAYMENT_CONFIRMATION,
    "statement": DocumentType.STATEMENT,
    "warranty": DocumentType.WARRANTY,
    "contract": DocumentType.CONTRACT,
}

# Types where LLM extraction document_type can override vision detection.
# Excludes receipt (would be a no-op).
_LLM_OVERRIDABLE_TYPES = {
    "invoice",
    "payment_confirmation",
    "contract",
    "warranty",
    "statement",
}

# Mapping from DocumentType to RecordType for refiner routing
ARTIFACT_TO_RECORD_TYPE: dict[DocumentType, RecordType] = {
    DocumentType.RECEIPT: RecordType.PURCHASE,
    DocumentType.INVOICE: RecordType.INVOICE,
    DocumentType.STATEMENT: RecordType.STATEMENT,
    DocumentType.WARRANTY: RecordType.WARRANTY,
    DocumentType.POLICY: RecordType.INSURANCE,
    DocumentType.PAYMENT_CONFIRMATION: RecordType.PAYMENT,
    DocumentType.CONTRACT: RecordType.CONTRACT,
}


@dataclass
class ProcessingResult:
    """Result of processing a document."""

    success: bool
    file_path: Path
    document_id: str | None = None  # v2 document ID
    is_duplicate: bool = False
    duplicate_of: str | None = None
    error: str | None = None
    extracted_data: dict[str, Any] | None = None
    refined_data: dict[str, Any] | None = None
    line_items: list[dict[str, Any]] = field(default_factory=list)
    record_type: RecordType | None = None
    source: str | None = None  # Entry point provenance
    user_id: str | None = None  # User who submitted the document


@dataclass
class _ExtractionMeta:
    """Pipeline metadata stripped from extraction results."""

    confidence: float | None = None
    ocr_text: str | None = None
    pipeline_type: str = "two_stage"
    parser_confidence: float | None = None
    parser_gaps: list[str] | None = None


class ProcessingPipeline:
    """Document processing pipeline."""

    def __init__(
        self,
        db: DatabaseManager | None = None,
        space_id: str = "default",
        user_id: str = "system",
    ) -> None:
        """Initialize the pipeline.

        Args:
            db: Database manager (creates one if not provided)
            space_id: Space to store artifacts in
            user_id: User performing the processing
        """
        self.config = get_config()
        self.db = db
        self.space_id = space_id
        self.user_id = user_id
        self._owns_db = db is None

        # Load user unit alias overrides
        init_unit_mappings(self.config.get_unit_aliases_path())

        # Load user vendor alias overrides
        init_vendor_mappings(self.config.get_vendor_aliases_path())

    def _get_db(self) -> DatabaseManager:
        """Get or create database manager."""
        if self.db is None:
            self.db = DatabaseManager(self.config)
            if not self.db.is_initialized():
                self.db.initialize()
        return self.db

    # Alias for backward-compat; prefer module-level STR_TO_ARTIFACT_TYPE.
    _VISION_TYPE_MAP = STR_TO_ARTIFACT_TYPE

    def _detect_document_type(self, file_path: Path) -> DocumentType:
        """Detect document type from file.

        For images, uses the LLM vision model to classify the document.
        For PDFs, renders the first page to an image and uses vision detection.
        Falls back to extension-based defaults if vision detection fails.
        """
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            return self._detect_pdf_type(file_path)

        if suffix in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}:
            return self._detect_image_type(file_path)

        return DocumentType.OTHER

    def _detect_image_type(self, file_path: Path) -> DocumentType:
        """Detect document type from an image file via vision model."""
        try:
            detected = vision_detect_document_type(file_path)
            mapped = STR_TO_ARTIFACT_TYPE.get(detected)
            if mapped is not None:
                logger.info(f"Vision detected {file_path.name} as {detected}")
                return mapped
            logger.debug(
                f"Vision returned '{detected}' for {file_path.name}, "
                f"defaulting to receipt"
            )
        except Exception as e:
            logger.warning(
                f"Vision type detection failed for {file_path.name}: {e}, "
                f"defaulting to receipt"
            )
        return DocumentType.RECEIPT

    def _detect_pdf_type(self, file_path: Path) -> DocumentType:
        """Detect document type from a PDF by rendering first page to image.

        Falls back to INVOICE if vision detection fails or pdf2image
        is unavailable.
        """
        import tempfile

        try:
            from pdf2image import convert_from_path
        except ImportError:
            logger.debug("pdf2image not available, defaulting PDF to invoice")
            return DocumentType.INVOICE

        try:
            images = convert_from_path(file_path, dpi=150, first_page=1, last_page=1)
            if not images:
                return DocumentType.INVOICE

            with tempfile.NamedTemporaryFile(
                suffix=".png", delete=False, prefix="alibi_pdf_type_"
            ) as tmp:
                images[0].save(tmp.name, "PNG")
                tmp_path = Path(tmp.name)

            try:
                detected = vision_detect_document_type(tmp_path)
                mapped = STR_TO_ARTIFACT_TYPE.get(detected)
                if mapped is not None:
                    logger.info(f"Vision detected PDF {file_path.name} as {detected}")
                    return mapped
                logger.debug(
                    f"Vision returned '{detected}' for PDF {file_path.name}, "
                    f"defaulting to invoice"
                )
            finally:
                tmp_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(
                f"PDF type detection failed for {file_path.name}: {e}, "
                f"defaulting to invoice"
            )

        return DocumentType.INVOICE

    # Phantom item names that card terminal slips produce
    _PHANTOM_ITEM_NAMES = {"PURCHASE", "PAYMENT", "SALE", "TOTAL"}

    @staticmethod
    def _looks_like_payment_confirmation(extracted_data: dict[str, Any]) -> bool:
        """Check if extraction data looks like a payment confirmation.

        Heuristic safety net: if the vision type detection classified a card
        terminal slip as "receipt", the extraction will have no real line
        items but will contain payment-specific fields (card_last4,
        authorization_code, terminal_id).

        Returns True if the extraction appears to be a payment confirmation.
        """
        if not extracted_data:
            return False

        line_items = extracted_data.get("line_items") or []
        phantom_names = ProcessingPipeline._PHANTOM_ITEM_NAMES
        real_items = [
            item
            for item in line_items
            if item.get("name", "").upper() not in phantom_names
        ]

        has_payment_details = bool(
            extracted_data.get("card_last4")
            or extracted_data.get("authorization_code")
            or extracted_data.get("terminal_id")
        )

        return len(real_items) == 0 and has_payment_details

    @staticmethod
    def _looks_like_statement(extracted_data: dict[str, Any]) -> bool:
        """Check if extraction data looks like a bank/card statement.

        Heuristic safety net: if vision classified a bank statement as
        "invoice", the raw text will contain account/IBAN markers and
        debit/credit patterns typical of statements.
        """
        if not extracted_data:
            return False
        import re

        raw_text = (extracted_data.get("raw_text") or "").lower()
        if not raw_text:
            return False
        has_account = bool(re.search(r"account\s*(?:no|number|activity)", raw_text))
        has_iban = "iban" in raw_text
        if not has_account and not has_iban:
            return False
        has_period = bool(re.search(r"period\s*:?\s*\d", raw_text))
        has_debit_credit = "debit" in raw_text and "credit" in raw_text
        has_statement = any(
            w in raw_text
            for w in ["statement", "account activity", "kontoauszug", "выписка"]
        )
        return has_period or has_debit_credit or has_statement

    @staticmethod
    def _looks_like_transaction_confirmation(
        extracted_data: dict[str, Any],
    ) -> bool:
        """Check if extraction data looks like a bank transaction confirmation.

        Heuristic safety net: if vision classified a bank payment confirmation
        as "invoice", the raw text will contain confirmation header and
        single-transaction markers (beneficiary, reference, debit account).
        """
        if not extracted_data:
            return False
        import re

        raw_text = (extracted_data.get("raw_text") or "").lower()
        if not raw_text:
            return False
        has_header = any(
            phrase in raw_text
            for phrase in [
                "transaction confirmation",
                "confirmation of transaction",
                "payment confirmation",
                "confirmation of payment",
            ]
        )
        if not has_header:
            return False
        # Exclude statements
        if (
            "debit" in raw_text
            and "credit" in raw_text
            and bool(re.search(r"period\s*:?\s*\d", raw_text))
        ):
            return False
        return any(
            phrase in raw_text
            for phrase in [
                "beneficiary",
                "transaction reference",
                "reference number",
                "debit account",
                "transaction amount",
                "amount debited",
                "value date",
                "transaction date",
            ]
        )

    def _extract_document(
        self,
        file_path: Path,
        doc_type: DocumentType | None,
        skip_llm_threshold: float | None = None,
        country: str | None = None,
        hints: ParserHints | None = None,
    ) -> dict[str, Any]:
        """Extract data from document.

        When doc_type is None, extraction uses post-OCR text classification
        to determine the document type (eliminates separate vision call).

        Args:
            hints: Optional parser hints from vendor/POS template learning.
        """
        extraction_type = self._type_to_str(doc_type) if doc_type else None

        if file_path.suffix.lower() == ".pdf":
            return extract_from_pdf(
                file_path,
                doc_type=extraction_type,
                skip_llm_threshold=skip_llm_threshold,
                country=country,
                hints=hints,
            )
        else:
            return extract_from_image(
                file_path,
                doc_type=extraction_type,
                skip_llm_threshold=skip_llm_threshold,
                country=country,
                hints=hints,
            )

    @staticmethod
    def _extract_dates_from_text(raw_text: str) -> list[date]:
        """Extract all plausible date values from raw text.

        Finds date-like patterns (DD/MM/YYYY, DD.MM.YYYY, YYYY-MM-DD)
        and returns parsed date objects.
        """
        import re

        patterns = [
            (r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", "dmy"),
            (r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", "dmy"),
            (r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", "ymd"),
        ]

        dates: list[date] = []
        for pattern, fmt in patterns:
            for match in re.finditer(pattern, raw_text):
                try:
                    if fmt == "dmy":
                        d, m, y = (
                            int(match.group(1)),
                            int(match.group(2)),
                            int(match.group(3)),
                        )
                    else:  # ymd
                        y, m, d = (
                            int(match.group(1)),
                            int(match.group(2)),
                            int(match.group(3)),
                        )
                    if 1 <= d <= 31 and 1 <= m <= 12 and 2000 <= y <= 2099:
                        dates.append(date(y, m, d))
                except (ValueError, OverflowError):
                    continue
        return dates

    def _validate_date(
        self, parsed_date: date | None, raw_text: str | None
    ) -> date | None:
        """Validate and correct an LLM-extracted date.

        Cross-references against dates found in raw_text. If the LLM
        date doesn't appear in the raw text but other dates do, use the
        most recent raw text date instead.

        Also applies a sanity range: reject dates more than 2 years in
        the past or future.
        """
        if parsed_date is None:
            return None

        today = date.today()
        max_age_days = 365 * 2

        # Sanity check: date within reasonable range?
        date_sane = abs((today - parsed_date).days) <= max_age_days

        if not raw_text:
            return parsed_date if date_sane else None

        raw_dates = self._extract_dates_from_text(raw_text)
        if not raw_dates:
            return parsed_date if date_sane else None

        # If the LLM date matches one found in raw text, trust it
        if parsed_date in raw_dates:
            return parsed_date

        # LLM date not in raw text — pick the most plausible raw text date
        # (most recent date within sanity range)
        valid_raw = [d for d in raw_dates if abs((today - d).days) <= max_age_days]
        if valid_raw:
            # Prefer the most recent date (likely the transaction date)
            best = max(valid_raw)
            logger.info(
                f"Date correction: LLM gave {parsed_date}, "
                f"raw text has {valid_raw}, using {best}"
            )
            return best

        # No valid raw text dates either — use LLM date if sane
        return parsed_date if date_sane else None

    # ------------------------------------------------------------------
    # Shared post-extraction pipeline steps
    # ------------------------------------------------------------------

    def _apply_type_overrides(
        self, doc_type: DocumentType, extracted_data: dict[str, Any]
    ) -> DocumentType:
        """Apply document type overrides and heuristic reclassification.

        Steps:
        1. Universal mode: classify from atom composition
        2. Specialized mode: LLM document_type field override
        3. Heuristic: receipt with no items + card details → payment_confirmation
        4. Heuristic: invoice with account/IBAN + statement markers → statement
        5. Heuristic: invoice with confirmation header → payment_confirmation
        6. Write final type back to extracted_data
        """
        if not extracted_data:
            return doc_type

        if self.config.prompt_mode == "universal":
            inferred_type = classify_from_extraction(extracted_data)
            if (
                inferred_type in STR_TO_ARTIFACT_TYPE
                and doc_type != STR_TO_ARTIFACT_TYPE[inferred_type]
            ):
                logger.info(
                    f"Universal extraction classified as "
                    f"{inferred_type}, overriding {doc_type.value}"
                )
                doc_type = STR_TO_ARTIFACT_TYPE[inferred_type]
        else:
            llm_type = extracted_data.get("document_type", "").lower()
            if (
                llm_type in _LLM_OVERRIDABLE_TYPES
                and llm_type in STR_TO_ARTIFACT_TYPE
                and doc_type != STR_TO_ARTIFACT_TYPE[llm_type]
            ):
                logger.info(f"LLM detected as {llm_type}, overriding {doc_type.value}")
                doc_type = STR_TO_ARTIFACT_TYPE[llm_type]

        if doc_type == DocumentType.RECEIPT and self._looks_like_payment_confirmation(
            extracted_data
        ):
            logger.info(
                "Heuristic reclassification: no real items + payment "
                "details → payment_confirmation"
            )
            doc_type = DocumentType.PAYMENT_CONFIRMATION

        if doc_type == DocumentType.INVOICE and self._looks_like_statement(
            extracted_data
        ):
            logger.info(
                "Heuristic reclassification: account/IBAN + statement "
                "markers → statement"
            )
            doc_type = DocumentType.STATEMENT

        if (
            doc_type == DocumentType.INVOICE
            and self._looks_like_transaction_confirmation(extracted_data)
        ):
            logger.info(
                "Heuristic reclassification: confirmation header + "
                "transaction markers → payment_confirmation"
            )
            doc_type = DocumentType.PAYMENT_CONFIRMATION

        extracted_data["document_type"] = self._type_to_str(doc_type)
        return doc_type

    @staticmethod
    def _fill_vendor_gaps(
        extracted_data: dict[str, Any], folder_context: FolderContext | None
    ) -> None:
        """Fill missing vendor fields from folder context (_vendor.yaml).

        Document extraction is primary; vendor config only fills fields
        the document didn't provide (via setdefault).
        """
        if not folder_context or not folder_context.vendor_config or not extracted_data:
            return
        vc = folder_context.vendor_config
        if vc.trade_name:
            extracted_data.setdefault("vendor", vc.trade_name)
        if vc.legal_name:
            extracted_data.setdefault("vendor_legal_name", vc.legal_name)
        if vc.vat_number:
            extracted_data.setdefault("vendor_vat", vc.vat_number)
        if vc.tax_id:
            extracted_data.setdefault("vendor_tax_id", vc.tax_id)
        if vc.phone:
            extracted_data.setdefault("vendor_phone", vc.phone)
        if vc.website:
            extracted_data.setdefault("vendor_website", vc.website)
        if vc.locations:
            extracted_data.setdefault("vendor_address", vc.locations[0].address)
            extracted_data.setdefault("vendor_map_url", vc.locations[0].map_url)

    @staticmethod
    def _fill_locale_gaps(
        extracted_data: dict[str, Any], folder_context: FolderContext | None
    ) -> None:
        """Fill missing currency/language from folder context config."""
        if not folder_context or not extracted_data:
            return
        ic = folder_context.inbox_config
        if ic and ic.default_currency:
            extracted_data.setdefault("currency", ic.default_currency)

    @staticmethod
    def _disambiguate_date(extracted_data: dict[str, Any], file_path: Path) -> None:
        """Resolve ambiguous DD/MM vs MM/DD dates using file metadata.

        Uses file modification date as a proximity reference.  A document
        photographed/uploaded in March with a date reading "03/05/2026"
        is more likely March 5 than May 3 (two months in the future).
        """
        raw_date = extracted_data.get("date")
        if not raw_date:
            return

        from alibi.normalizers.dates import disambiguate_date, parse_date

        parsed = parse_date(raw_date)
        if not parsed:
            return

        # Get file modification date as reference
        file_date = None
        try:
            stat = file_path.stat()
            file_date = date.fromtimestamp(stat.st_mtime)
        except (OSError, ValueError):
            pass

        fixed = disambiguate_date(
            parsed,
            str(raw_date),
            reference_date=date.today(),
            file_date=file_date,
        )

        if fixed != parsed:
            logger.info(
                "Date disambiguated: %s -> %s (file_date=%s)",
                parsed.isoformat(),
                fixed.isoformat(),
                file_date,
            )
            extracted_data["date"] = fixed.isoformat()

    def _strip_pipeline_meta(self, extracted_data: dict[str, Any]) -> _ExtractionMeta:
        """Pop pipeline metadata from extraction and run verification."""
        ocr_text = extracted_data.get("raw_text")
        confidence = extracted_data.pop("_two_stage_confidence", None)
        pipeline_type = extracted_data.pop("_pipeline", "two_stage")
        p_confidence = extracted_data.pop("_parser_confidence", None)

        if confidence is None:
            v_result = verify_extraction(extracted_data, ocr_text=ocr_text)
            confidence = v_result.confidence
            if not v_result.passed:
                logger.warning(
                    f"Low confidence extraction: {confidence:.2f}, "
                    f"flags={v_result.flags}"
                )

        return _ExtractionMeta(
            confidence=confidence,
            ocr_text=ocr_text,
            pipeline_type=pipeline_type,
            parser_confidence=p_confidence,
        )

    def _write_cache(
        self,
        file_path: Path,
        extracted_data: dict[str, Any],
        doc_type: DocumentType,
        meta: _ExtractionMeta,
        is_group: bool = False,
        file_hash: str | None = None,
        perceptual_hash: str | None = None,
        user_id: str = "system",
    ) -> Path | None:
        """Write .alibi.yaml cache with standard parameters.

        Returns the path to the written YAML file, or None on failure.
        """
        _CRITICAL_GAPS = {"vendor", "date", "total"}
        low_confidence = meta.confidence is not None and meta.confidence < 0.5
        critical_gaps = bool(
            meta.parser_gaps and _CRITICAL_GAPS.intersection(meta.parser_gaps)
        )
        needs_review = low_confidence or critical_gaps

        return write_yaml_cache(
            file_path,
            extracted_data,
            document_type=doc_type.value,
            is_group=is_group,
            ocr_model=self.config.ollama_ocr_model,
            structure_model=self.config.ollama_structure_model,
            confidence=meta.confidence,
            ocr_text=meta.ocr_text,
            pipeline=meta.pipeline_type,
            parser_confidence=meta.parser_confidence,
            parser_gaps=meta.parser_gaps,
            file_hash=file_hash,
            perceptual_hash=perceptual_hash,
            needs_review=needs_review,
            user_id=user_id,
        )

    def _extract_to_yaml(
        self,
        db: DatabaseManager,
        file_path: Path,
        file_hash: str,
        perceptual_hash: str | None,
        doc_type: DocumentType,
        extracted_data: dict[str, Any],
        folder_context: FolderContext | None = None,
        is_group: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Phase A: extract document and write YAML.

        Applies type overrides, fills vendor gaps, strips pipeline meta,
        and writes .alibi.yaml with file_hash + perceptual_hash in _meta.

        Returns (extracted_data, meta_dict) or None on failure.
        """
        if not extracted_data:
            return None

        # 1. Type overrides + heuristic reclassification
        doc_type = self._apply_type_overrides(doc_type, extracted_data)

        # 2. Fill vendor gaps from folder context
        self._fill_vendor_gaps(extracted_data, folder_context)
        self._fill_locale_gaps(extracted_data, folder_context)

        # 2a. Swap payment processor with actual merchant name
        from alibi.extraction.text_parser import _swap_processor_vendor

        _swap_processor_vendor(extracted_data)

        # 2b. Disambiguate date using file metadata and context
        self._disambiguate_date(extracted_data, file_path)

        # 2.5. Item verification (Layer 1 + optional Layer 2)
        if extracted_data.get("line_items"):
            from alibi.extraction.item_verifier import verify_items, verify_items_llm

            iv_result = verify_items(extracted_data)
            if iv_result.flags:
                ocr_text = extracted_data.get("_ocr_text", "")
                if ocr_text:
                    verify_items_llm(extracted_data, iv_result.flags, ocr_text)

        # 3. Verify extraction + extract pipeline metadata
        meta = self._strip_pipeline_meta(extracted_data)

        # 4. Write .alibi.yaml cache
        user_id = (folder_context.user_id if folder_context else None) or "system"
        written_yaml = self._write_cache(
            file_path,
            extracted_data,
            doc_type,
            meta,
            is_group,
            file_hash=file_hash,
            perceptual_hash=perceptual_hash,
            user_id=user_id,
        )

        # Build meta dict matching what read_yaml_with_meta returns
        meta_dict: dict[str, Any] = {"version": YAML_VERSION}
        if file_hash:
            meta_dict["file_hash"] = file_hash
        if perceptual_hash:
            meta_dict["perceptual_hash"] = perceptual_hash
        if written_yaml is not None:
            meta_dict["yaml_path"] = str(written_yaml)

        return extracted_data, meta_dict

    @staticmethod
    def _commit_yaml_versioning() -> None:
        """Commit any pending YAML files to git (best-effort)."""
        try:
            from alibi.mycelium.yaml_versioning import get_yaml_versioner

            get_yaml_versioner().commit_pending()
        except Exception as e:
            logger.debug(f"YAML git commit skipped: {e}")

    def _migrate_saved_annotations(
        self,
        db: DatabaseManager,
        saved_annotations: list[dict[str, Any]],
        document_id: str,
    ) -> int:
        """Migrate saved annotations to the new fact created for a document.

        Finds the fact(s) created for the given document and re-attaches
        the saved annotations to them.
        """
        from alibi.annotations.store import migrate_annotations_to_fact

        # Find fact(s) for the newly created document
        facts = db.fetchall(
            "SELECT f.id FROM facts f "
            "JOIN clouds c ON f.cloud_id = c.id "
            "JOIN bundles b ON b.cloud_id = c.id "
            "WHERE b.document_id = ? "
            "GROUP BY f.id",
            (document_id,),
        )
        if not facts:
            return 0

        total_migrated = 0
        for fact_row in facts:
            fact_id = fact_row["id"]
            items = db.fetchall(
                "SELECT id, name FROM fact_items WHERE fact_id = ?",
                (fact_id,),
            )
            item_dicts = [dict(i) for i in items]
            total_migrated += migrate_annotations_to_fact(
                db, saved_annotations, fact_id, item_dicts
            )
        return total_migrated

    def _ingest_from_yaml(
        self,
        db: DatabaseManager,
        file_path: Path,
        file_hash: str,
        perceptual_hash: str | None,
        extracted_data: dict[str, Any],
        doc_type: DocumentType,
        folder_context: FolderContext | None = None,
        is_group: bool = False,
        yaml_path: str | None = None,
    ) -> ProcessingResult:
        """Phase B: ingest extracted data into DB from YAML.

        Applies historical corrections (without rewriting YAML), refines,
        canonicalizes vendor, runs v2 pipeline. Shared by fresh ingest
        and corrections.
        """
        # 1. Historical verification (cross-reference against DB)
        # NOTE: do NOT rewrite YAML — admin edits are preserved
        if extracted_data:
            try:
                hist_result = apply_historical_corrections(db, extracted_data)
                if hist_result.applied_count > 0:
                    logger.info(
                        f"Historical verification for {file_path.name}: "
                        f"{hist_result.applied_count} corrections applied"
                    )
            except Exception as e:
                logger.warning(f"Historical verification skipped: {e}")

        # 2. Route through refiner
        refined_data: dict[str, Any] | None = None
        refined_line_items: list[dict[str, Any]] = []
        record_type = ARTIFACT_TO_RECORD_TYPE.get(doc_type)

        if extracted_data and record_type is not None:
            try:
                refined_data = self._refine_extraction(
                    extracted_data, record_type, artifact_id=None
                )
                refined_line_items = refined_data.get("line_items", [])
                logger.info(
                    f"Refined {file_path}: record_type={record_type.value}, "
                    f"line_items={len(refined_line_items)}"
                )
            except Exception as e:
                logger.warning(f"Refiner failed for {file_path}: {e}")

        # 3. Vendor canonicalization
        raw_vendor = extracted_data.get("vendor")
        if raw_vendor:
            canonical = canonicalize_vendor(raw_vendor)
            if canonical and canonical != raw_vendor:
                extracted_data["vendor"] = canonical
                logger.debug(f"Vendor canonicalized: {raw_vendor} -> {canonical}")

        # 4. Compute yaml_hash for change detection
        _uid = (folder_context.user_id if folder_context else None) or "system"
        _doc_type_str = (
            folder_context.doc_type.value
            if folder_context and folder_context.doc_type
            else doc_type.value
        )
        yaml_hash = compute_yaml_hash(
            file_path, is_group, user_id=_uid, doc_type=_doc_type_str
        )

        # 4.5. Cross-validate item totals vs receipt total (Layer 3)
        if extracted_data.get("line_items"):
            from alibi.extraction.item_verifier import (
                cross_validate_receipt,
                validate_barcode_items,
            )

            cv_result = cross_validate_receipt(extracted_data)
            for w in cv_result.warnings:
                logger.warning("Cross-validation: %s (%s)", w, file_path.name)
            if cv_result.needs_review:
                extracted_data["_cv_needs_review"] = True
            if cv_result.item_count_mismatch > 0:
                extracted_data["_items_missing"] = cv_result.item_count_mismatch

            # 4.6. Barcode-item cross-validation (auto-correct from product_cache)
            bc_flags = validate_barcode_items(extracted_data, db)
            for flag in bc_flags:
                logger.info("Barcode validation: %s (%s)", flag.issue, file_path.name)

        # 5. V2 atom-cloud-fact pipeline (sole storage path)
        _source = folder_context.source if folder_context else "cli"
        _user_id = folder_context.user_id if folder_context else "system"
        document_id: str | None = None
        if extracted_data:
            document_id = self._run_v2_pipeline(
                db,
                file_path,
                file_hash,
                perceptual_hash,
                extracted_data,
                source=_source,
                user_id=_user_id,
                yaml_hash=yaml_hash,
                folder_context=folder_context,
                yaml_path=yaml_path,
            )

        logger.info(f"Processed: {file_path} -> {document_id}")

        return ProcessingResult(
            success=True,
            file_path=file_path,
            document_id=document_id,
            extracted_data=extracted_data,
            refined_data=refined_data,
            line_items=refined_line_items,
            record_type=record_type,
            source=_source,
            user_id=_user_id,
        )

    def _post_extraction_pipeline(
        self,
        db: DatabaseManager,
        file_path: Path,
        file_hash: str,
        perceptual_hash: str | None,
        doc_type: DocumentType,
        extracted_data: dict[str, Any],
        folder_context: FolderContext | None = None,
        was_cached: bool = False,
        is_group: bool = False,
    ) -> ProcessingResult:
        """Shared post-extraction pipeline (backward compat wrapper).

        Delegates to Phase A (extract_to_yaml) if not cached, then Phase B
        (ingest_from_yaml).
        """
        if not was_cached and extracted_data:
            self._extract_to_yaml(
                db,
                file_path,
                file_hash,
                perceptual_hash,
                doc_type,
                extracted_data,
                folder_context=folder_context,
                is_group=is_group,
            )
        else:
            # Cached data: still apply type overrides + vendor/locale fill
            doc_type = self._apply_type_overrides(doc_type, extracted_data)
            self._fill_vendor_gaps(extracted_data, folder_context)
            self._fill_locale_gaps(extracted_data, folder_context)
            if extracted_data:
                self._strip_pipeline_meta(extracted_data)

        return self._ingest_from_yaml(
            db,
            file_path,
            file_hash,
            perceptual_hash,
            extracted_data,
            doc_type,
            folder_context=folder_context,
            is_group=is_group,
        )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def process_file(
        self, file_path: Path, folder_context: FolderContext | None = None
    ) -> ProcessingResult:
        """Process a single document file (YAML-first pipeline).

        Flow: validate → hash → Phase A (ensure YAML) → duplicate/correction
        check → Phase B (YAML → DB).

        Args:
            file_path: Path to the document
            folder_context: Optional folder routing context (type, vendor, config)

        Returns:
            ProcessingResult with outcome
        """
        if not file_path.exists():
            return ProcessingResult(
                success=False,
                file_path=file_path,
                error=f"File not found: {file_path}",
            )

        if not is_supported_file(file_path):
            return ProcessingResult(
                success=False,
                file_path=file_path,
                error=f"Unsupported file type: {file_path.suffix}",
            )

        db = self._get_db()

        try:
            # 0. Image optimization (strip EXIF, resize, compress)
            if is_image_file(file_path) and self.config.optimize_images:
                from alibi.processing.image_optimizer import optimize_image

                opt_result = optimize_image(
                    file_path,
                    max_dim=self.config.image_max_dim,
                    quality=self.config.image_quality,
                )
                if opt_result.get("optimized"):
                    new_path = opt_result.get("new_path")
                    if new_path and new_path != file_path:
                        file_path = new_path

            # 1. Compute hashes
            file_hash = compute_file_hash(file_path)
            perceptual_hash = None
            if is_image_file(file_path):
                try:
                    perceptual_hash = compute_perceptual_hash(file_path)
                except Exception as e:
                    logger.warning(f"Failed to compute perceptual hash: {e}")

            # 2. Detect document type from folder context
            doc_type: DocumentType | None = None
            if folder_context and folder_context.doc_type is not None:
                doc_type = folder_context.doc_type
                logger.info(f"Folder-routed {file_path.name} as {doc_type.value}")

            # Derive user_id and doc_type_str for yaml_cache calls
            _user_id = (folder_context.user_id if folder_context else None) or "system"
            _doc_type_str = (
                folder_context.doc_type.value
                if folder_context and folder_context.doc_type
                else "unsorted"
            )

            # 3. Phase A: ensure YAML exists
            yaml_result = read_yaml_with_meta(
                file_path, user_id=_user_id, doc_type=_doc_type_str
            )
            # Fallback: YAML may live under a different doc_type
            # (written as "receipt" on first run, queried with "unsorted" now)
            _found_yaml: Path | None = None
            if yaml_result is None:
                _found_yaml = find_yaml_in_store(file_path, user_id=_user_id)
                if _found_yaml is not None:
                    from alibi.extraction.yaml_cache import read_yaml_direct

                    yaml_result = read_yaml_direct(_found_yaml)

            # yaml_path to store on Document (resolved after type is known)
            _yaml_path: str | None = None
            if yaml_result is not None:
                extracted_data, meta = yaml_result
                logger.info(f"Using YAML cache for {file_path}")
                cached_type = extracted_data.get("document_type", "").lower()
                if cached_type and cached_type in STR_TO_ARTIFACT_TYPE:
                    doc_type = STR_TO_ARTIFACT_TYPE[cached_type]
                    _doc_type_str = cached_type
                elif doc_type is None:
                    doc_type = DocumentType.RECEIPT
                # Apply type overrides + vendor/locale fill for cached data
                doc_type = self._apply_type_overrides(doc_type, extracted_data)
                self._fill_vendor_gaps(extracted_data, folder_context)
                self._fill_locale_gaps(extracted_data, folder_context)
                if extracted_data:
                    self._strip_pipeline_meta(extracted_data)
                # Resolve yaml_path for the cached document
                if _found_yaml is not None:
                    _yaml_path = str(_found_yaml)
                else:
                    _cached_yaml = get_yaml_path(
                        file_path, user_id=_user_id, doc_type=_doc_type_str
                    )
                    if _cached_yaml.exists():
                        _yaml_path = str(_cached_yaml)
            else:
                # Fresh extraction needed
                skip_threshold: float | None = None
                if folder_context and folder_context.doc_type is not None and doc_type:
                    skip_threshold = _FOLDER_ROUTED_SKIP_THRESHOLD.get(doc_type)

                extracted_data = {}
                country = folder_context.country if folder_context else None

                # Template learning: load hints from vendor/POS/location
                _hints: ParserHints | None = None
                _tpl_identity_id: str | None = None
                try:
                    _fc_vendor = folder_context.vendor_hint if folder_context else None
                    _fc_vat: str | None = None
                    if folder_context and folder_context.vendor_config:
                        _fc_vat = folder_context.vendor_config.vat_number or None
                        if not _fc_vendor:
                            _fc_vendor = folder_context.vendor_config.trade_name or None
                    _fc_lat = folder_context.lat if folder_context else None
                    _fc_lng = folder_context.lng if folder_context else None
                    _hints, _tpl_identity_id = resolve_hints(
                        db,
                        vendor_name=_fc_vendor,
                        vendor_vat=_fc_vat,
                        lat=_fc_lat,
                        lng=_fc_lng,
                    )
                    if _hints and (_hints.layout_type or _hints.currency):
                        logger.info(
                            f"Template hints for {file_path.name}: "
                            f"layout={_hints.layout_type}, "
                            f"currency={_hints.currency}, "
                            f"vendor={_hints.vendor_name}"
                        )
                except Exception as e:
                    logger.debug(f"Template hint loading skipped: {e}")

                try:
                    extracted_data = self._extract_document(
                        file_path,
                        doc_type,
                        skip_llm_threshold=skip_threshold,
                        country=country,
                        hints=_hints,
                    )
                    if doc_type is None:
                        detected_type = extracted_data.get("document_type", "").lower()
                        doc_type = STR_TO_ARTIFACT_TYPE.get(
                            detected_type, DocumentType.RECEIPT
                        )
                        logger.info(
                            f"Post-OCR classified {file_path.name} as "
                            f"{doc_type.value}"
                        )
                except (VisionExtractionError, PDFExtractionError) as e:
                    logger.warning(f"Extraction failed for {file_path}: {e}")
                    if doc_type is None:
                        doc_type = DocumentType.RECEIPT

                # Gemini template bootstrapping for new vendors
                if extracted_data:
                    _doc_type_for_bootstrap = doc_type.value if doc_type else "receipt"
                    try:
                        from alibi.extraction.template_bootstrapper import (
                            apply_vendor_details,
                            bootstrap_with_gemini,
                            merge_extraction,
                            needs_bootstrapping,
                        )

                        # Check if bootstrapping needed
                        _existing_tpl = None
                        if _tpl_identity_id:
                            _existing_tpl = load_vendor_template(db, _tpl_identity_id)
                        _is_parser_only = (
                            extracted_data.get("_pipeline") == "parser_only"
                        )
                        if _is_parser_only and needs_bootstrapping(
                            _existing_tpl,
                            self.config.gemini_extraction_enabled,
                        ):
                            logger.info(
                                f"Bootstrapping template for "
                                f"{file_path.name} via Gemini"
                            )
                            _ocr = extracted_data.get("raw_text", "")
                            _gemini = bootstrap_with_gemini(
                                _ocr,
                                _doc_type_for_bootstrap,
                            )
                            if _gemini:
                                extracted_data = merge_extraction(
                                    extracted_data,
                                    _gemini,
                                    _doc_type_for_bootstrap,
                                )

                        # Apply cached vendor details for known vendors
                        elif _tpl_identity_id:
                            _cached = load_vendor_details(db, _tpl_identity_id)
                            if _cached:
                                extracted_data = apply_vendor_details(
                                    extracted_data,
                                    _cached,
                                    _doc_type_for_bootstrap,
                                )
                    except Exception as e:
                        logger.debug(f"Template bootstrapping skipped: {e}")

                # Write YAML (Phase A) — may update doc_type via overrides
                if extracted_data:
                    _extract_result = self._extract_to_yaml(
                        db,
                        file_path,
                        file_hash,
                        perceptual_hash,
                        doc_type,
                        extracted_data,
                        folder_context=folder_context,
                    )
                    if _extract_result is not None:
                        _yaml_path = _extract_result[1].get("yaml_path")
                    # Re-resolve doc_type (overrides may have changed it)
                    final_type = extracted_data.get("document_type", "").lower()
                    if final_type in STR_TO_ARTIFACT_TYPE:
                        doc_type = STR_TO_ARTIFACT_TYPE[final_type]
                        # Update doc_type_str after override
                        _doc_type_str = final_type

            # 4. Duplicate / correction check
            saved_annotations: list[dict[str, Any]] = []
            yaml_hash = compute_yaml_hash(
                file_path, user_id=_user_id, doc_type=_doc_type_str
            )
            existing_doc = v2_store.get_document_by_hash(db, file_hash)
            if existing_doc:
                stored_yaml_hash = existing_doc.get("yaml_hash")
                if stored_yaml_hash == yaml_hash:
                    logger.info(
                        f"Duplicate detected: {file_path} matches "
                        f"document {existing_doc['id']}"
                    )
                    return ProcessingResult(
                        success=True,
                        file_path=file_path,
                        document_id=existing_doc["id"],
                        is_duplicate=True,
                        duplicate_of=existing_doc["id"],
                    )
                # YAML changed (correction) — cleanup old records
                logger.info(
                    f"YAML correction detected for {file_path.name}, " f"re-ingesting"
                )
                cleanup_result = v2_store.cleanup_document(db, existing_doc["id"])
                saved_annotations = cleanup_result.get("saved_annotations", [])

            # 5. Phase B: ingest from YAML
            result = self._ingest_from_yaml(
                db,
                file_path,
                file_hash,
                perceptual_hash,
                extracted_data,
                doc_type,
                folder_context=folder_context,
                yaml_path=_yaml_path,
            )

            # 6. Migrate annotations from old fact to new
            if saved_annotations and result.success and result.document_id:
                self._migrate_saved_annotations(
                    db, saved_annotations, result.document_id
                )

            # 7. Commit pending YAML git changes
            self._commit_yaml_versioning()

            return result

        except Exception as e:
            logger.error(f"Processing failed for {file_path}: {e}")
            self._commit_yaml_versioning()
            return ProcessingResult(
                success=False,
                file_path=file_path,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # V2 atom-cloud-fact pipeline
    # ------------------------------------------------------------------

    def _run_v2_pipeline(
        self,
        db: DatabaseManager,
        file_path: Path,
        file_hash: str,
        perceptual_hash: str | None,
        extracted_data: dict[str, Any],
        source: str | None = None,
        user_id: str | None = None,
        yaml_hash: str | None = None,
        yaml_path: str | None = None,
        folder_context: FolderContext | None = None,
    ) -> str | None:
        """Run v2 atom-cloud-fact pipeline.

        1. Create v2 Document record
        2. Parse atoms + form bundle
        3. Store document, atoms, bundle
        4. Cloud formation: match against existing bundles
        5. Create/join cloud
        6. Try collapse → store fact + items

        Returns:
            The v2 document ID, or None on failure.
        """
        created_doc_id: str | None = None
        try:
            # 8.1. Check if document already exists (dedup by hash)
            existing_doc = v2_store.get_document_by_hash(db, file_hash)
            if existing_doc:
                logger.debug(f"V2: document already exists for {file_path.name}")
                doc_id: str = existing_doc["id"]
                return doc_id

            # 8.2. Create and store Document
            doc = Document(
                id=str(uuid.uuid4()),
                file_path=str(file_path),
                file_hash=file_hash,
                perceptual_hash=perceptual_hash,
                raw_extraction=extracted_data,
                source=source,
                user_id=user_id,
                yaml_hash=yaml_hash,
                yaml_path=yaml_path,
            )
            v2_store.store_document(db, doc)
            created_doc_id = doc.id

            # 8.3. Parse atoms + bundles from extraction
            parse_result = parse_extraction(doc.id, extracted_data)
            if not parse_result.atoms:
                logger.debug(f"V2: no atoms parsed from {file_path.name}")
                return doc.id

            # 8.4. Store all atoms
            v2_store.store_atoms(db, parse_result.atoms)

            # 8.5. Process each bundle (one for receipts/invoices, many for statements)
            for bundle_result in parse_result.bundles:
                self._process_v2_bundle(
                    db,
                    file_path,
                    bundle_result.bundle,
                    bundle_result.bundle_atoms,
                    parse_result.atoms,
                    extracted_data=extracted_data,
                    folder_context=folder_context,
                )

            return doc.id

        except Exception as e:
            logger.warning(f"V2 pipeline failed for {file_path.name}: {e}")
            if created_doc_id:
                try:
                    v2_store.cleanup_document(db, created_doc_id)
                except Exception as cleanup_err:
                    logger.warning(f"V2 cleanup also failed: {cleanup_err}")
            return None

    def _process_v2_bundle(
        self,
        db: DatabaseManager,
        file_path: Path,
        bundle: Any,  # Bundle model
        bundle_atom_links: list[Any],  # BundleAtom models
        all_atoms: list[Any],  # Atom models
        extracted_data: dict[str, Any] | None = None,
        folder_context: FolderContext | None = None,
    ) -> None:
        """Process a single bundle: store, match cloud, try collapse."""
        from alibi.db.models import Cloud, CloudStatus

        # Store bundle + links
        v2_store.store_bundle(db, bundle, bundle_atom_links)

        bundle_id = bundle.id
        bundle_type = bundle.bundle_type

        # Build summary using only this bundle's atoms
        linked_atom_ids = {ba.atom_id for ba in bundle_atom_links}
        bundle_atoms = [a for a in all_atoms if a.id in linked_atom_ids]
        atom_dicts = [
            {"atom_type": a.atom_type.value, "data": a.data} for a in bundle_atoms
        ]
        new_summary = extract_bundle_summary(bundle_id, bundle_type, atom_dicts)

        # Populate lat/lng from folder context (user-provided map URL)
        if folder_context and folder_context.lat is not None:
            new_summary.lat = folder_context.lat
            new_summary.lng = folder_context.lng

        # Resolve vendor identity (auto-register if new)
        identity_id = None
        try:
            from alibi.identities.matching import ensure_vendor_identity

            vat = extracted_data.get("vendor_vat") if extracted_data else None
            tid = extracted_data.get("vendor_tax_id") if extracted_data else None
            identity_id = ensure_vendor_identity(
                db,
                vendor_name=new_summary.vendor,
                vendor_key=new_summary.vendor_key,
                vat_number=vat,
                tax_id=tid,
                source="extraction",
            )
            new_summary.identity_id = identity_id
        except Exception as e:
            logger.debug(f"Identity resolution skipped: {e}")

        # Template learning: save fingerprint after successful extraction
        if identity_id and extracted_data:
            try:
                confidence = extracted_data.get("_parser_confidence", 0.0)
                ocr_text = extracted_data.get("raw_text", "")
                _was_bootstrapped = extracted_data.get("_gemini_bootstrapped", False)

                _doc_type_str_tpl = (
                    extracted_data.get("document_type", "receipt") or "receipt"
                )

                if _was_bootstrapped:
                    # Gemini-bootstrapped: use enhanced template
                    from alibi.extraction.template_bootstrapper import (
                        build_enhanced_template,
                        extract_vendor_details,
                    )

                    tpl = build_enhanced_template(
                        extracted_data, ocr_text, _doc_type_str_tpl
                    )
                    existing_tpl = load_vendor_template(db, identity_id)
                    if existing_tpl:
                        merged = merge_template(existing_tpl, tpl)
                        # Preserve Gemini schema insights
                        merged.gemini_bootstrapped = True
                        merged.language = tpl.language or merged.language
                        merged.has_barcodes = (
                            tpl.has_barcodes
                            if tpl.has_barcodes is not None
                            else merged.has_barcodes
                        )
                        merged.has_unit_quantities = (
                            tpl.has_unit_quantities
                            if tpl.has_unit_quantities is not None
                            else merged.has_unit_quantities
                        )
                        merged.typical_item_count = (
                            tpl.typical_item_count or merged.typical_item_count
                        )
                        save_vendor_template(db, identity_id, merged)
                    else:
                        save_vendor_template(db, identity_id, tpl)

                    # Store vendor details on identity
                    details = extract_vendor_details(extracted_data, _doc_type_str_tpl)
                    if details:
                        save_vendor_details(db, identity_id, details)

                    # Also learn POS provider template
                    if tpl.pos_provider:
                        ensure_pos_identity(db, tpl.pos_provider, tpl)
                else:
                    # Standard template learning (non-bootstrapped)
                    tpl = extract_template_fingerprint(
                        extracted_data, ocr_text, confidence
                    )
                    if tpl:
                        existing_tpl = load_vendor_template(db, identity_id)
                        if existing_tpl:
                            merged = merge_template(existing_tpl, tpl)
                            save_vendor_template(db, identity_id, merged)
                        else:
                            save_vendor_template(db, identity_id, tpl)

                        # Also learn POS provider template
                        if tpl.pos_provider:
                            ensure_pos_identity(db, tpl.pos_provider, tpl)

                    # Store vendor details even without bootstrapping
                    # (Gemini Stage 3 or parser may have extracted them)
                    from alibi.extraction.template_bootstrapper import (
                        extract_vendor_details,
                    )

                    details = extract_vendor_details(extracted_data, _doc_type_str_tpl)
                    if details:
                        save_vendor_details(db, identity_id, details)

                # Record extraction observation for adaptive learning
                saved_tpl = load_vendor_template(db, identity_id)
                if saved_tpl:
                    ocr_tier = 1 if extracted_data.get("_ocr_enhanced") else 0
                    # Feed correction history into template
                    correction_fields: list[str] = []
                    if new_summary.vendor_key:
                        try:
                            from alibi.services.correction_log import (
                                get_vendor_unreliable_fields,
                            )

                            correction_fields = get_vendor_unreliable_fields(
                                db, new_summary.vendor_key
                            )
                        except Exception:
                            pass
                    updated_tpl = record_extraction_observation(
                        saved_tpl,
                        confidence=confidence,
                        ocr_tier=ocr_tier,
                        fixes_applied=correction_fields or None,
                    )
                    # Derive vendor default category from item history
                    if new_summary.vendor_key:
                        from alibi.extraction.templates import (
                            derive_vendor_default_category,
                        )

                        default_cat = derive_vendor_default_category(
                            db, new_summary.vendor_key
                        )
                        if default_cat != updated_tpl.default_category:
                            updated_tpl.default_category = default_cat
                    save_vendor_template(db, identity_id, updated_tpl)
            except Exception as e:
                logger.debug(f"Template learning skipped: {e}")

        # Load existing bundles from DB for matching — pre-filtered by vendor
        existing_raw = v2_store.get_bundle_summaries_for_vendor(
            db,
            vendor_key=new_summary.vendor_key,
            vendor_name=new_summary.vendor_normalized,
        )
        existing_summaries = [
            extract_bundle_summary(
                b["bundle_id"],
                BundleType(b["bundle_type"]),
                b["atoms"],
                cloud_id=b["cloud_id"],
            )
            for b in existing_raw
            if b["bundle_id"] != bundle_id
        ]

        # Enrich existing summaries with location data from annotations
        cloud_ids_with_summaries = {
            s.cloud_id for s in existing_summaries if s.cloud_id
        }
        if cloud_ids_with_summaries:
            cloud_locs = v2_store.get_cloud_locations(db, cloud_ids_with_summaries)
            for s in existing_summaries:
                if s.cloud_id and s.cloud_id in cloud_locs:
                    s.lat, s.lng = cloud_locs[s.cloud_id]

        match_result = find_cloud_for_bundle(new_summary, existing_summaries)

        if match_result.is_new_cloud:
            cloud, cloud_bundle = create_cloud_for_bundle(bundle_id)
            v2_store.store_cloud(db, cloud, cloud_bundle)
            cloud_id = cloud.id
            logger.debug(f"V2: new cloud {cloud_id[:8]} for {file_path.name}")
        else:
            cloud_id = match_result.cloud_id  # type: ignore[assignment]
            cloud_bundle = add_bundle_to_cloud(
                cloud_id,
                bundle_id,
                match_result.match_type,  # type: ignore[arg-type]
                match_result.confidence,
            )
            v2_store.add_cloud_bundle(db, cloud_bundle)
            logger.debug(
                f"V2: bundle joined cloud {cloud_id[:8]} "
                f"(confidence={match_result.confidence})"
            )

        # Delete existing fact before re-collapse (prevents duplicates
        # when a second document joins an already-collapsed cloud)
        existing_fact = v2_store.get_fact_for_cloud(db, cloud_id)
        if existing_fact:
            v2_store.delete_fact(db, existing_fact["id"])

        # Try collapse
        cloud_data = v2_store.get_cloud_bundle_data(db, cloud_id)
        cloud_obj = Cloud(id=cloud_id, status=CloudStatus.FORMING)
        collapse_result = try_collapse(cloud_obj, cloud_data)

        if collapse_result.collapsed and collapse_result.fact:
            # Use canonical vendor_key from identity if available
            if identity_id:
                from alibi.identities.matching import get_canonical_vendor_key

                canonical_key = get_canonical_vendor_key(db, identity_id)
                if canonical_key:
                    collapse_result.fact.vendor_key = canonical_key
            # Flag fact as needs_review when cross-validation found >50% mismatch
            if extracted_data and extracted_data.get("_cv_needs_review"):
                from alibi.db.models import FactStatus

                collapse_result.fact.status = FactStatus.NEEDS_REVIEW
            for item in collapse_result.items:
                item.fact_id = collapse_result.fact.id
            v2_store.store_fact(db, collapse_result.fact, collapse_result.items)

            # Resolve item identities for each fact item
            try:
                from alibi.identities.matching import ensure_item_identity

                for item in collapse_result.items:
                    ensure_item_identity(
                        db,
                        item_name=item.name,
                        barcode=item.barcode,
                        source="extraction",
                    )
            except Exception as e:
                logger.debug(f"Item identity resolution skipped: {e}")

            # Store location annotation if map_url provided
            if folder_context and folder_context.map_url:
                try:
                    from alibi.services.correction import set_fact_location

                    set_fact_location(
                        db,
                        collapse_result.fact.id,
                        folder_context.map_url,
                    )
                except Exception as e:
                    logger.debug(f"Location annotation skipped: {e}")

            logger.info(
                f"V2: cloud {cloud_id[:8]} collapsed → "
                f"fact {collapse_result.fact.id[:8]} "
                f"({collapse_result.fact.vendor}, "
                f"{collapse_result.fact.total_amount} "
                f"{collapse_result.fact.currency})"
            )
        elif not match_result.is_new_cloud:
            v2_store.update_cloud_status(db, cloud_id, collapse_result.cloud_status)

    def _refine_extraction(
        self,
        extracted_data: dict[str, Any],
        record_type: RecordType,
        artifact_id: str | None = None,
    ) -> dict[str, Any]:
        """Route extraction through the appropriate refiner.

        Args:
            extracted_data: Raw extracted data dict
            record_type: RecordType for refiner lookup
            artifact_id: Source artifact ID for provenance

        Returns:
            Refined data dict with normalized fields and line items
        """
        refiner = get_refiner(record_type)
        return refiner.refine(extracted_data, artifact_id=artifact_id)

    def process_document_group(
        self,
        folder_path: Path,
        files: list[Path],
        folder_context: FolderContext | None = None,
    ) -> ProcessingResult:
        """Process multiple files as pages of a single document (YAML-first).

        Flow: combined hash → Phase A (ensure YAML) → duplicate/correction
        check → Phase B (YAML → DB).

        Args:
            folder_path: Path to the folder containing the pages
            files: Sorted list of page files
            folder_context: Optional folder routing context (type, vendor,
                source, user_id). When provided, doc_type and provenance
                are derived from context instead of vision detection.

        Returns:
            Single ProcessingResult for the grouped document
        """
        if not files:
            return ProcessingResult(
                success=False,
                file_path=folder_path,
                error="No supported files in folder",
            )

        db = self._get_db()

        try:
            # 1. Compute combined hash (hash of all file hashes)
            import hashlib

            hasher = hashlib.sha256()
            for f in files:
                hasher.update(compute_file_hash(f).encode())
            combined_hash = hasher.hexdigest()

            # 2. Detect document type from first page (or folder context)
            if folder_context and folder_context.doc_type:
                doc_type = folder_context.doc_type
            else:
                doc_type = self._detect_document_type(files[0])

            # Derive user_id and doc_type_str for yaml_cache calls
            _grp_user_id = (
                folder_context.user_id if folder_context else None
            ) or "system"
            _grp_doc_type_str = (
                folder_context.doc_type.value
                if folder_context and folder_context.doc_type
                else "unsorted"
            )

            # 3. Phase A: ensure YAML exists
            yaml_result = read_yaml_with_meta(
                folder_path,
                is_group=True,
                user_id=_grp_user_id,
                doc_type=_grp_doc_type_str,
            )
            _found_grp_yaml: Path | None = None
            if yaml_result is None:
                _found_grp_yaml = find_yaml_in_store(
                    folder_path, is_group=True, user_id=_grp_user_id
                )
                if _found_grp_yaml is not None:
                    from alibi.extraction.yaml_cache import read_yaml_direct

                    yaml_result = read_yaml_direct(_found_grp_yaml)

            _grp_yaml_path: str | None = None
            if yaml_result is not None:
                logger.info(f"Using YAML cache for document group {folder_path}")
                extracted_data, meta = yaml_result
                cached_type = extracted_data.get("document_type", "").lower()
                if cached_type and cached_type in STR_TO_ARTIFACT_TYPE:
                    doc_type = STR_TO_ARTIFACT_TYPE[cached_type]
                    _grp_doc_type_str = cached_type
                # Apply type overrides + vendor/locale fill for cached data
                doc_type = self._apply_type_overrides(doc_type, extracted_data)
                self._fill_vendor_gaps(extracted_data, folder_context)
                self._fill_locale_gaps(extracted_data, folder_context)
                if extracted_data:
                    self._strip_pipeline_meta(extracted_data)
                # Resolve yaml_path for the cached document group
                if _found_grp_yaml is not None:
                    _grp_yaml_path = str(_found_grp_yaml)
                else:
                    _cached_grp_yaml = get_yaml_path(
                        folder_path,
                        is_group=True,
                        user_id=_grp_user_id,
                        doc_type=_grp_doc_type_str,
                    )
                    if _cached_grp_yaml.exists():
                        _grp_yaml_path = str(_cached_grp_yaml)
            else:
                extracted_data = {}
                image_files = [
                    f
                    for f in files
                    if f.suffix.lower()
                    in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
                ]
                pdf_files = [f for f in files if f.suffix.lower() == ".pdf"]

                # Template hints for document groups
                _grp_hints: ParserHints | None = None
                try:
                    _gfc_vendor = folder_context.vendor_hint if folder_context else None
                    _gfc_vat: str | None = None
                    if folder_context and folder_context.vendor_config:
                        _gfc_vat = folder_context.vendor_config.vat_number or None
                        if not _gfc_vendor:
                            _gfc_vendor = (
                                folder_context.vendor_config.trade_name or None
                            )
                    _gfc_lat = folder_context.lat if folder_context else None
                    _gfc_lng = folder_context.lng if folder_context else None
                    _grp_hints, _ = resolve_hints(
                        db,
                        vendor_name=_gfc_vendor,
                        vendor_vat=_gfc_vat,
                        lat=_gfc_lat,
                        lng=_gfc_lng,
                    )
                except Exception as e:
                    logger.debug(f"Template hint loading (group) skipped: {e}")

                try:
                    if image_files and not pdf_files:
                        extracted_data = extract_from_images(
                            image_files,
                            doc_type=self._type_to_str(doc_type),
                            hints=_grp_hints,
                        )
                    elif pdf_files and not image_files:
                        extracted_data = extract_from_pdf(
                            pdf_files[0], hints=_grp_hints
                        )
                        for pdf in pdf_files[1:]:
                            extra = extract_from_pdf(pdf, hints=_grp_hints)
                            self._merge_extraction(extracted_data, extra)
                    else:
                        extracted_data = extract_from_images(
                            image_files,
                            doc_type=self._type_to_str(doc_type),
                            hints=_grp_hints,
                        )
                        for pdf in pdf_files:
                            extra = extract_from_pdf(pdf, hints=_grp_hints)
                            self._merge_extraction(extracted_data, extra)
                except (VisionExtractionError, PDFExtractionError) as e:
                    logger.warning(f"Extraction failed for {folder_path}: {e}")

                # Write YAML (Phase A) — may update doc_type via overrides
                if extracted_data:
                    _grp_extract_result = self._extract_to_yaml(
                        db,
                        folder_path,
                        combined_hash,
                        None,
                        doc_type,
                        extracted_data,
                        folder_context=folder_context,
                        is_group=True,
                    )
                    if _grp_extract_result is not None:
                        _grp_yaml_path = _grp_extract_result[1].get("yaml_path")
                    # Re-resolve doc_type (overrides may have changed it)
                    final_type = extracted_data.get("document_type", "").lower()
                    if final_type in STR_TO_ARTIFACT_TYPE:
                        doc_type = STR_TO_ARTIFACT_TYPE[final_type]
                        # Update doc_type_str after override
                        _grp_doc_type_str = final_type

            # 4. Duplicate / correction check
            saved_annotations_group: list[dict[str, Any]] = []
            yaml_hash = compute_yaml_hash(
                folder_path,
                is_group=True,
                user_id=_grp_user_id,
                doc_type=_grp_doc_type_str,
            )
            existing_doc = v2_store.get_document_by_hash(db, combined_hash)
            if existing_doc:
                stored_yaml_hash = existing_doc.get("yaml_hash")
                if stored_yaml_hash == yaml_hash:
                    logger.info(f"Duplicate document group: {folder_path}")
                    return ProcessingResult(
                        success=True,
                        file_path=folder_path,
                        document_id=existing_doc["id"],
                        is_duplicate=True,
                        duplicate_of=existing_doc["id"],
                    )
                logger.info(
                    f"YAML correction detected for group {folder_path}, "
                    f"re-ingesting"
                )
                cleanup_result = v2_store.cleanup_document(db, existing_doc["id"])
                saved_annotations_group = cleanup_result.get("saved_annotations", [])

            # 5. Phase B: ingest from YAML
            result = self._ingest_from_yaml(
                db,
                folder_path,
                combined_hash,
                None,
                extracted_data,
                doc_type,
                folder_context=folder_context,
                is_group=True,
                yaml_path=_grp_yaml_path,
            )

            # 6. Migrate annotations from old fact to new
            if saved_annotations_group and result.success and result.document_id:
                self._migrate_saved_annotations(
                    db, saved_annotations_group, result.document_id
                )

            # 7. Commit pending YAML git changes
            self._commit_yaml_versioning()

            return result

        except Exception as e:
            logger.error(f"Processing failed for document group {folder_path}: {e}")
            self._commit_yaml_versioning()
            return ProcessingResult(
                success=False,
                file_path=folder_path,
                error=str(e),
            )

    def ingest_from_yaml(
        self,
        db: DatabaseManager,
        source_path: Path,
        is_group: bool = False,
    ) -> ProcessingResult:
        """Public entry: re-ingest a document from its edited .alibi.yaml.

        Reads YAML, cleans up existing DB records, runs Phase B.
        Source file is optional (file_hash read from YAML _meta).

        Args:
            db: Database manager
            source_path: Path to the source file or folder
            is_group: True if source_path is a folder (document group)

        Returns:
            ProcessingResult with outcome
        """
        yaml_result = read_yaml_with_meta(source_path, is_group)
        if yaml_result is None:
            _found = find_yaml_in_store(source_path, is_group=is_group)
            if _found is not None:
                from alibi.extraction.yaml_cache import read_yaml_direct

                yaml_result = read_yaml_direct(_found)
        if yaml_result is None:
            return ProcessingResult(
                success=False,
                file_path=source_path,
                error="No valid .alibi.yaml found (missing or version mismatch)",
            )

        extracted_data, meta = yaml_result

        # Resolve file_hash: from source file if it exists, else from YAML _meta
        file_hash = meta.get("file_hash")
        perceptual_hash = meta.get("perceptual_hash")

        if not is_group and source_path.exists() and source_path.is_file():
            file_hash = compute_file_hash(source_path)
            if is_image_file(source_path):
                try:
                    perceptual_hash = compute_perceptual_hash(source_path)
                except Exception:
                    pass

        if not file_hash:
            return ProcessingResult(
                success=False,
                file_path=source_path,
                error="Cannot determine file_hash: source missing and no "
                "file_hash in YAML _meta",
            )

        # Resolve doc_type from extracted data
        cached_type = extracted_data.get("document_type", "").lower()
        doc_type = STR_TO_ARTIFACT_TYPE.get(cached_type, DocumentType.RECEIPT)

        # Clean up existing DB records
        saved_annotations_reingest: list[dict[str, Any]] = []
        existing_doc = v2_store.get_document_by_hash(db, file_hash)
        if existing_doc:
            logger.info(f"Re-ingestion: cleaning up document {existing_doc['id'][:8]}")
            cleanup_result = v2_store.cleanup_document(db, existing_doc["id"])
            saved_annotations_reingest = cleanup_result.get("saved_annotations", [])

        # Strip pipeline meta keys from extracted data
        for key in ("_two_stage_confidence", "_pipeline", "_parser_confidence"):
            extracted_data.pop(key, None)

        result = self._ingest_from_yaml(
            db,
            source_path,
            file_hash,
            perceptual_hash,
            extracted_data,
            doc_type,
            is_group=is_group,
        )

        # Migrate annotations from old fact to new
        if saved_annotations_reingest and result.success and result.document_id:
            self._migrate_saved_annotations(
                db, saved_annotations_reingest, result.document_id
            )

        return result

    @staticmethod
    def _type_to_str(doc_type: DocumentType) -> str:
        """Convert DocumentType to extraction type string."""
        return (
            doc_type.value if doc_type in STR_TO_ARTIFACT_TYPE.values() else "receipt"
        )

    @staticmethod
    def _merge_extraction(base: dict[str, Any], extra: dict[str, Any]) -> None:
        """Merge extra extraction data into base (in-place).

        Appends line items and raw_text, keeps first non-null values for
        scalar fields.
        """
        # Merge line items
        base_items = base.get("line_items", [])
        extra_items = extra.get("line_items", [])
        base["line_items"] = base_items + extra_items

        # Append raw text
        base_text = base.get("raw_text", "")
        extra_text = extra.get("raw_text", "")
        if extra_text:
            base["raw_text"] = (
                f"{base_text}\n---\n{extra_text}" if base_text else extra_text
            )

        # Fill in missing scalar fields from extra
        for key in (
            "vendor",
            "document_date",
            "date",
            "total",
            "amount",
            "currency",
            "document_id",
            "vendor_id",
        ):
            if not base.get(key) and extra.get(key):
                base[key] = extra[key]

    def process_batch(
        self,
        files: list[Path],
        folder_contexts: list[FolderContext] | None = None,
    ) -> list[ProcessingResult]:
        """Process multiple files.

        Args:
            files: List of file paths
            folder_contexts: Optional per-file folder contexts (same length as files)

        Returns:
            List of ProcessingResult for each file
        """
        results = []
        for i, file_path in enumerate(files):
            ctx = folder_contexts[i] if folder_contexts else None
            result = self.process_file(file_path, folder_context=ctx)
            results.append(result)

        return results

    def close(self) -> None:
        """Close database connection if we own it."""
        if self._owns_db and self.db is not None:
            self.db.close()
