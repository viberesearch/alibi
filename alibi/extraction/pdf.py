"""PDF text extraction using pdfplumber.

Supports two extraction paths:
1. Text-layer PDFs: pdfplumber text -> heuristic parser -> skip LLM if high confidence
2. Image-only PDFs: render to images -> OCR -> heuristic parser -> LLM correction
"""

import io
import logging
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import pdfplumber

from alibi.config import get_config
from alibi.extraction.prompts import get_text_structure_prompt
from alibi.extraction.templates import ParserHints
from alibi.extraction.vision import VisionExtractionError, extract_json_from_response

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Error during PDF extraction."""

    pass


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, list[list[list[str | None]]]]:
    """Extract text and tables from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Tuple of (full_text, tables)

    Raises:
        PDFExtractionError: If extraction fails
    """
    if not pdf_path.exists():
        raise PDFExtractionError(f"PDF file not found: {pdf_path}")

    text_content = []
    tables = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text
                page_text = page.extract_text() or ""
                text_content.append(page_text)

                # Extract tables
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
    except Exception as e:
        raise PDFExtractionError(f"Failed to extract PDF: {e}") from e

    full_text = "\n".join(text_content)
    return full_text, tables


def structure_text_with_llm(
    text: str,
    tables: list[list[list[str | None]]] | None = None,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Use LLM to structure extracted text.

    Uses the text-only structure model (config.ollama_structure_model)
    instead of the vision model, since no images are involved.

    Args:
        text: Extracted text content
        tables: Extracted tables (list of rows, each row is list of cells)
        timeout: Request timeout in seconds (default 180s for large documents)

    Returns:
        Structured data as dictionary
    """
    import httpx

    config = get_config()
    prompt = get_text_structure_prompt(text, tables)

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{config.ollama_url}/api/generate",
                json={
                    "model": config.ollama_structure_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
            response.raise_for_status()
    except httpx.RequestError as e:
        raise PDFExtractionError(f"LLM request failed: {e}") from e

    result = response.json()

    if "error" in result:
        raise PDFExtractionError(f"Ollama error: {result['error']}")

    if "response" not in result:
        raise PDFExtractionError(f"Unexpected response format: {result}")

    return extract_json_from_response(result["response"])


# Max characters of extracted text to send in one LLM call.
# Beyond this, switch to per-page extraction to avoid timeouts.
_MAX_TEXT_PER_LLM_CALL = 8000


def extract_from_pdf(
    pdf_path: Path,
    doc_type: str | None = "invoice",
    use_vision_fallback: bool = True,
    skip_llm_threshold: float | None = None,
    country: str | None = None,
    hints: ParserHints | None = None,
) -> dict[str, Any]:
    """Extract structured data from a PDF file.

    Uses a 3-stage approach matching the image pipeline:
    1. Text acquisition (pdfplumber text extraction)
    1b. If doc_type is None, classify from extracted text (post-OCR)
    2. Heuristic parser (instant, no LLM)
    3. LLM correction/structuring (only if parser confidence is low)

    Falls back to vision-based OCR for image-only PDFs.

    Args:
        pdf_path: Path to the PDF file
        doc_type: Document type string, or None to auto-classify from text
        use_vision_fallback: Whether to use vision model if text extraction fails

    Returns:
        Extracted data as dictionary

    Raises:
        PDFExtractionError: If extraction fails
    """
    t_start = time.monotonic()
    text, tables = extract_text_from_pdf(pdf_path)

    # Post-text classification when type not pre-determined
    if doc_type is None and text.strip() and len(text) >= 100:
        from alibi.extraction.text_parser import classify_ocr_text

        doc_type = classify_ocr_text(text)
        logger.info(f"Post-text classification for PDF {pdf_path.name}: {doc_type}")

    # Default if still None (image-only PDF or very short text)
    if doc_type is None:
        doc_type = "invoice"

    # If we have reasonable text content, try parser-first path
    if text.strip() and len(text) >= 100:
        return _extract_pdf_with_parser(
            pdf_path,
            text,
            tables,
            doc_type,
            use_vision_fallback,
            skip_llm_threshold=skip_llm_threshold,
            country=country,
            hints=hints,
        )

    # Fall back to vision-based extraction (image-only PDFs)
    if use_vision_fallback:
        logger.debug(f"Falling back to vision extraction for {pdf_path}")
        return extract_pdf_via_vision(
            pdf_path,
            doc_type=doc_type,
            skip_llm_threshold=skip_llm_threshold,
            country=country,
            hints=hints,
        )

    raise PDFExtractionError(
        f"PDF has insufficient text ({len(text)} chars) and vision fallback disabled"
    )


def _extract_pdf_with_parser(
    pdf_path: Path,
    text: str,
    tables: list[list[list[str | None]]],
    doc_type: str,
    use_vision_fallback: bool,
    skip_llm_threshold: float | None = None,
    country: str | None = None,
    hints: ParserHints | None = None,
) -> dict[str, Any]:
    """Extract from text-layer PDF using heuristic parser first.

    Mirrors the 3-stage image pipeline:
    - Parser confidence >= skip_threshold: skip LLM entirely
    - Parser confidence >= 0.3: use LLM correction prompt
    - Parser confidence < 0.3: full LLM structuring
    """
    from alibi.extraction.text_parser import parse_ocr_text

    config = get_config()
    skip_threshold = (
        skip_llm_threshold
        if skip_llm_threshold is not None
        else config.skip_llm_threshold
    )
    t_start = time.monotonic()

    # For large PDFs (statements), concatenate all text for parsing
    if len(text) > _MAX_TEXT_PER_LLM_CALL:
        logger.debug(
            f"Large PDF ({len(text)} chars) for {pdf_path.name}, "
            f"attempting parser-first on full text"
        )

    # Pre-parser heuristic: detect statement content misclassified as invoice
    if doc_type != "statement" and _looks_like_statement_text(text):
        logger.info(
            f"Text heuristic reclassified {pdf_path.name} "
            f"from {doc_type} to statement"
        )
        doc_type = "statement"

    # Pre-parser heuristic: detect bank transaction confirmation
    if doc_type not in (
        "payment_confirmation",
        "statement",
    ) and _looks_like_transaction_confirmation_text(text):
        logger.info(
            f"Text heuristic reclassified {pdf_path.name} "
            f"from {doc_type} to payment_confirmation"
        )
        doc_type = "payment_confirmation"

    # Stage 2: Heuristic parse (same as image pipeline)
    parse_result = None
    try:
        parse_result = parse_ocr_text(text, doc_type=doc_type, hints=hints)
        t_parse = time.monotonic()
        logger.info(
            f"PDF parser for {pdf_path.name}: "
            f"confidence={parse_result.confidence:.2f}, "
            f"items={parse_result.line_item_count}, "
            f"gaps={parse_result.gaps[:5]}"
            f" [{(t_parse - t_start) * 1000:.1f}ms]"
        )
    except Exception as e:
        logger.warning(
            f"PDF parser failed for {pdf_path.name}: {e}, "
            f"falling through to LLM structuring"
        )

    # Stage 3: LLM structuring (correction, full, or skip)
    skippable_types = ("receipt", "payment_confirmation", "statement", "invoice")

    if (
        parse_result
        and parse_result.confidence >= skip_threshold
        and doc_type in skippable_types
    ):
        # High-confidence parser output — skip LLM entirely
        extracted = dict(parse_result.data)
        extracted["_parser_confidence"] = parse_result.confidence
        extracted["_pipeline"] = "pdf_parser_only"
        extracted["raw_text"] = text
        extracted["_text_source"] = "pdfplumber"
        logger.info(
            f"Skipping LLM for PDF {pdf_path.name}: "
            f"parser confidence {parse_result.confidence:.2f} "
            f">= threshold {skip_threshold}"
        )
        return extracted

    if parse_result and parse_result.confidence >= 0.3:
        from alibi.extraction.structurer import structure_ocr_text
        from alibi.extraction.vision import DEFAULT_LLM_TIMEOUT

        timeout = DEFAULT_LLM_TIMEOUT
        config = get_config()

        if config.gemini_extraction_enabled:
            # Gemini does comprehensive extraction; skip micro-prompts
            # to avoid loading Ollama models (VRAM contention with OCR)
            try:
                extracted = structure_ocr_text(text, doc_type=doc_type, timeout=timeout)
                extracted["_parser_confidence"] = parse_result.confidence
                extracted["_pipeline"] = "pdf_three_stage_gemini"
                extracted["_text_source"] = "pdfplumber"
                if not extracted.get("raw_text"):
                    extracted["raw_text"] = text
                return extracted
            except Exception as e:
                logger.warning(
                    f"PDF Gemini extraction failed for {pdf_path.name}: {e}, "
                    f"using parser result"
                )
                extracted = dict(parse_result.data)
                extracted["_parser_confidence"] = parse_result.confidence
                extracted["_pipeline"] = "pdf_parser_fallback"
                extracted["raw_text"] = text
                extracted["_text_source"] = "pdfplumber"
                return extracted
        else:
            # Ollama path: try micro-prompts first, then correction
            from alibi.extraction.micro_prompts import run_micro_prompts

            micro_result = run_micro_prompts(
                parse_result, text, doc_type, timeout=timeout
            )
            if micro_result is not None:
                extracted = micro_result
                extracted["_text_source"] = "pdfplumber"
                if not extracted.get("raw_text"):
                    extracted["raw_text"] = text
                return extracted

            from alibi.extraction.prompts import get_correction_prompt

            correction_prompt = get_correction_prompt(
                parse_result.data,
                text,
                doc_type,
                field_confidence=parse_result.field_confidence,
                regions=parse_result.regions,
            )
            try:
                extracted = structure_ocr_text(
                    text,
                    doc_type=doc_type,
                    timeout=timeout,
                    emphasis_prompt=correction_prompt,
                )
                extracted["_parser_confidence"] = parse_result.confidence
                extracted["_pipeline"] = "pdf_three_stage"
                extracted["_text_source"] = "pdfplumber"
                if not extracted.get("raw_text"):
                    extracted["raw_text"] = text
                return extracted
            except Exception as e:
                logger.warning(
                    f"PDF LLM correction failed for {pdf_path.name}: {e}, "
                    f"using parser result"
                )
                extracted = dict(parse_result.data)
                extracted["_parser_confidence"] = parse_result.confidence
                extracted["_pipeline"] = "pdf_parser_fallback"
                extracted["raw_text"] = text
                extracted["_text_source"] = "pdfplumber"
                return extracted

    # Parser couldn't extract much — full LLM structuring
    if len(text) <= _MAX_TEXT_PER_LLM_CALL:
        logger.debug(f"Using LLM text structuring for PDF {pdf_path.name}")
        try:
            extracted = structure_text_with_llm(text, tables)
            extracted["_pipeline"] = "pdf_llm_full"
            extracted["_text_source"] = "pdfplumber"
            if not extracted.get("raw_text"):
                extracted["raw_text"] = text
            return extracted
        except PDFExtractionError:
            if use_vision_fallback:
                logger.warning(
                    f"LLM structuring failed for {pdf_path.name}, "
                    f"falling back to vision"
                )
                return extract_pdf_via_vision(
                    pdf_path,
                    doc_type=doc_type,
                    skip_llm_threshold=skip_llm_threshold,
                    country=country,
                    hints=hints,
                )
            raise
    else:
        logger.debug(f"Large PDF ({len(text)} chars), per-page LLM for {pdf_path.name}")
        return _extract_pdf_per_page(pdf_path)


def _extract_pdf_per_page(pdf_path: Path) -> dict[str, Any]:
    """Extract data from a large PDF by processing each page separately.

    Sends each page's text to the LLM individually, then merges results.
    This avoids timeouts on multi-page documents like bank statements.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Merged extraction from all pages
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = list(pdf.pages)
    except Exception as e:
        raise PDFExtractionError(f"Failed to open PDF: {e}") from e

    if not pages:
        raise PDFExtractionError(f"PDF has no pages: {pdf_path}")

    result: dict[str, Any] | None = None

    for i, page in enumerate(pages):
        page_text = page.extract_text() or ""
        page_tables = page.extract_tables() or []

        if not page_text.strip():
            logger.debug(f"Skipping empty page {i + 1} of {pdf_path.name}")
            continue

        logger.debug(
            f"Processing page {i + 1}/{len(pages)} of {pdf_path.name} "
            f"({len(page_text)} chars)"
        )

        try:
            page_result = structure_text_with_llm(
                page_text, page_tables if page_tables else None
            )
        except PDFExtractionError as e:
            logger.warning(f"Page {i + 1} extraction failed: {e}")
            continue

        if result is None:
            result = page_result
        else:
            _merge_extractions(result, page_result)

    if result is None:
        raise PDFExtractionError(f"No pages could be extracted from {pdf_path}")

    return result


def extract_pdf_via_vision(
    pdf_path: Path,
    doc_type: str | None = "invoice",
    skip_llm_threshold: float | None = None,
    country: str | None = None,
    hints: ParserHints | None = None,
) -> dict[str, Any]:
    """Extract data from PDF by converting pages to images, OCRing, and structuring.

    Uses the three-stage pipeline matching the image extraction path:
    1. OCR each page image separately (with retry/fallback)
    2. Heuristic parser on combined text (deterministic, instant)
    3. LLM correction only if parser confidence is low

    Args:
        pdf_path: Path to the PDF file
        doc_type: Document type string, or None to auto-classify from OCR text
        skip_llm_threshold: Override for LLM skip threshold (defaults to config)
        country: ISO country code for OCR model selection

    Returns:
        Extracted data from all pages as a single unified result

    Raises:
        PDFExtractionError: If extraction fails
    """
    try:
        from pdf2image import convert_from_path  # type: ignore[import-not-found]
    except ImportError:
        raise PDFExtractionError(
            "pdf2image is required for vision-based PDF extraction. "
            "Install with: uv sync --extra pdf"
        )

    from alibi.extraction.ocr import ocr_image_with_retry
    from alibi.extraction.structurer import structure_ocr_text

    try:
        images = convert_from_path(pdf_path, dpi=150)
        if not images:
            raise PDFExtractionError("Failed to convert PDF to images")

        logger.debug(f"Converted {pdf_path} to {len(images)} page images")

        tmp_paths: list[Path] = []
        try:
            for i, img in enumerate(images):
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False, prefix=f"alibi_pdf_p{i+1}_"
                )
                img.save(tmp.name, "PNG")
                tmp_paths.append(Path(tmp.name))
                tmp.close()

            # Stage 1: OCR each page separately with retry/fallback
            page_texts: list[str] = []
            for i, tmp_path in enumerate(tmp_paths):
                try:
                    from alibi.config import get_config

                    text, was_enhanced = ocr_image_with_retry(
                        tmp_path, timeout=get_config().ocr_timeout, country=country
                    )
                    page_texts.append(f"--- PAGE {i + 1} ---\n{text}")
                    logger.debug(
                        f"PDF OCR page {i + 1}: {len(text)} chars"
                        f"{' (enhanced)' if was_enhanced else ''}"
                    )
                except VisionExtractionError as e:
                    logger.warning(f"PDF OCR page {i + 1} failed: {e}")

            if not page_texts:
                raise PDFExtractionError("All page OCR attempts failed")

            combined_text = "\n\n".join(page_texts)

            # Post-OCR classification when type not pre-determined
            if doc_type is None:
                from alibi.extraction.text_parser import classify_ocr_text

                doc_type = classify_ocr_text(combined_text)
                logger.info(
                    f"Post-OCR classification for PDF {pdf_path.name}: {doc_type}"
                )
            if doc_type is None:
                doc_type = "invoice"

            # Stage 2: Heuristic parse
            from alibi.extraction.text_parser import parse_ocr_text

            config = get_config()
            skip_threshold = (
                skip_llm_threshold
                if skip_llm_threshold is not None
                else config.skip_llm_threshold
            )

            parse_result = None
            try:
                parse_result = parse_ocr_text(
                    combined_text, doc_type=doc_type, hints=hints
                )
                logger.info(
                    f"PDF vision parser for {pdf_path.name}: "
                    f"confidence={parse_result.confidence:.2f}, "
                    f"items={parse_result.line_item_count}, "
                    f"gaps={parse_result.gaps[:5]}"
                )
            except Exception as e:
                logger.warning(
                    f"PDF vision parser failed for {pdf_path.name}: {e}, "
                    f"falling through to LLM"
                )

            # Stage 3: LLM structuring based on parser confidence
            skippable_types = (
                "receipt",
                "payment_confirmation",
                "statement",
                "invoice",
            )

            if (
                parse_result
                and parse_result.confidence >= skip_threshold
                and doc_type in skippable_types
            ):
                # High confidence — skip LLM entirely
                extracted = dict(parse_result.data)
                extracted["_parser_confidence"] = parse_result.confidence
                extracted["_pipeline"] = "pdf_vision_parser_only"
                if not extracted.get("raw_text"):
                    extracted["raw_text"] = combined_text
                logger.info(
                    f"Skipping LLM for PDF {pdf_path.name}: "
                    f"parser confidence {parse_result.confidence:.2f} "
                    f">= threshold {skip_threshold}"
                )
                return extracted

            if parse_result and parse_result.confidence >= 0.3:
                from alibi.extraction.vision import DEFAULT_LLM_TIMEOUT

                llm_timeout = DEFAULT_LLM_TIMEOUT
                pdf_config = get_config()

                if pdf_config.gemini_extraction_enabled:
                    # Gemini path: skip micro-prompts (VRAM contention)
                    try:
                        extracted = structure_ocr_text(
                            combined_text,
                            doc_type=doc_type,
                            timeout=llm_timeout,
                        )
                        extracted["_parser_confidence"] = parse_result.confidence
                        extracted["_pipeline"] = "pdf_vision_three_stage_gemini"
                        if not extracted.get("raw_text"):
                            extracted["raw_text"] = combined_text
                        return extracted
                    except Exception as e:
                        logger.warning(
                            f"PDF vision Gemini extraction failed for "
                            f"{pdf_path.name}: {e}, using parser result"
                        )
                        extracted = dict(parse_result.data)
                        extracted["_parser_confidence"] = parse_result.confidence
                        extracted["_pipeline"] = "pdf_vision_parser_fallback"
                        extracted["raw_text"] = combined_text
                        return extracted
                else:
                    # Ollama path: micro-prompts then correction
                    from alibi.extraction.micro_prompts import run_micro_prompts

                    micro_result = run_micro_prompts(
                        parse_result,
                        combined_text,
                        doc_type,
                        timeout=llm_timeout,
                    )
                    if micro_result is not None:
                        if not micro_result.get("raw_text"):
                            micro_result["raw_text"] = combined_text
                        micro_result["_pipeline"] = "pdf_vision_three_stage"
                        return micro_result

                    from alibi.extraction.prompts import get_correction_prompt

                    correction_prompt = get_correction_prompt(
                        parse_result.data,
                        combined_text,
                        doc_type,
                        field_confidence=parse_result.field_confidence,
                        regions=parse_result.regions,
                    )
                    try:
                        extracted = structure_ocr_text(
                            combined_text,
                            doc_type=doc_type,
                            timeout=llm_timeout,
                            emphasis_prompt=correction_prompt,
                        )
                        extracted["_parser_confidence"] = parse_result.confidence
                        extracted["_pipeline"] = "pdf_vision_three_stage"
                        if not extracted.get("raw_text"):
                            extracted["raw_text"] = combined_text
                        return extracted
                    except Exception as e:
                        logger.warning(
                            f"PDF vision LLM correction failed for "
                            f"{pdf_path.name}: {e}, using parser result"
                        )
                        extracted = dict(parse_result.data)
                        extracted["_parser_confidence"] = parse_result.confidence
                        extracted["_pipeline"] = "pdf_vision_parser_fallback"
                        extracted["raw_text"] = combined_text
                        return extracted

            # Low confidence or parser failed — full LLM structuring
            result = structure_ocr_text(
                combined_text,
                doc_type=doc_type,
                timeout=180.0,
            )
            result["_pipeline"] = "pdf_vision_llm_full"
            if not result.get("raw_text"):
                result["raw_text"] = combined_text
            return result
        finally:
            for p in tmp_paths:
                p.unlink(missing_ok=True)

    except Exception as e:
        if isinstance(e, (PDFExtractionError, VisionExtractionError)):
            raise
        raise PDFExtractionError(f"Vision extraction failed: {e}") from e


def _merge_extractions(base: dict[str, Any], extra: dict[str, Any]) -> None:
    """Merge extraction from an additional page into the base result.

    Appends line_items/items lists and fills in missing scalar fields.
    """
    # Merge line items
    for key in ("line_items", "items"):
        if key in extra and extra[key]:
            base.setdefault(key, [])
            base[key].extend(extra[key])

    # Fill missing scalar fields from extra page
    for key, value in extra.items():
        if key in ("line_items", "items"):
            continue
        if value and not base.get(key):
            base[key] = value


def get_pdf_page_count(pdf_path: Path) -> int:
    """Get the number of pages in a PDF file."""
    if not pdf_path.exists():
        raise PDFExtractionError(f"PDF file not found: {pdf_path}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            return len(pdf.pages)
    except Exception as e:
        raise PDFExtractionError(f"Failed to read PDF: {e}") from e


def _looks_like_statement_text(text: str) -> bool:
    """Detect if PDF text content appears to be a bank/card statement.

    Checks for account-related markers combined with statement-specific
    patterns (period, debit/credit columns, statement keywords).
    """
    low = text.lower()
    has_account = bool(re.search(r"account\s*(?:no|number|activity)", low))
    has_iban = "iban" in low
    if not has_account and not has_iban:
        return False
    has_period = bool(re.search(r"period\s*:?\s*\d", low))
    has_debit_credit = "debit" in low and "credit" in low
    has_statement = any(
        w in low for w in ["statement", "account activity", "kontoauszug", "выписка"]
    )
    return has_period or has_debit_credit or has_statement


def _looks_like_transaction_confirmation_text(text: str) -> bool:
    """Detect if PDF text is a bank-issued single transaction confirmation.

    Distinct from card terminal slips (which have terminal_id, auth_code)
    and from statements (which have period, debit/credit columns).
    """
    low = text.lower()
    has_confirmation_header = any(
        phrase in low
        for phrase in [
            "transaction confirmation",
            "confirmation of transaction",
            "payment confirmation",
            "confirmation of payment",
            # Multilingual bank transfer titles
            "αιτηση μεταφορας",  # EL: request for transfer
            "εντολη πληρωμης",  # EL: payment order
            "платежное поручение",  # RU: payment order
            "zahlungsauftrag",  # DE: payment order
            "payment order",
            "wire transfer",
        ]
    )
    if not has_confirmation_header:
        return False
    # Exclude statements (have both debit and credit columns)
    if "debit" in low and "credit" in low and bool(re.search(r"period\s*:?\s*\d", low)):
        return False
    has_transaction_markers = any(
        phrase in low
        for phrase in [
            "beneficiary",
            "transaction reference",
            "reference number",
            "debit account",
            "transaction amount",
            "amount debited",
            "value date",
            "transaction date",
            # Multilingual beneficiary labels
            "δικαιούχος",  # EL: beneficiary (accented)
            "δικαιουχος",  # EL: beneficiary (unaccented)
            "получатель",  # RU: recipient
            "empfänger",  # DE: recipient
        ]
    )
    return has_transaction_markers


def extract_pdf_metadata(pdf_path: Path) -> dict[str, Any]:
    """Extract metadata from a PDF file."""
    if not pdf_path.exists():
        raise PDFExtractionError(f"PDF file not found: {pdf_path}")

    try:
        with pdfplumber.open(pdf_path) as pdf:
            return {
                "page_count": len(pdf.pages),
                "metadata": pdf.metadata or {},
            }
    except Exception as e:
        raise PDFExtractionError(f"Failed to read PDF metadata: {e}") from e
