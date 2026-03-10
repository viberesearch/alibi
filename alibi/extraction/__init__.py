"""Document extraction modules for Alibi."""

from alibi.extraction.pdf import (
    PDFExtractionError,
    extract_from_pdf,
    extract_pdf_metadata,
    extract_text_from_pdf,
    get_pdf_page_count,
)
from alibi.extraction.prompts import (
    INVOICE_PROMPT,
    INVOICE_PROMPT_V2,
    PURCHASE_ATOMIZATION_PROMPT,
    RECEIPT_PROMPT,
    RECEIPT_PROMPT_V2,
    STATEMENT_PROMPT,
    WARRANTY_PROMPT,
    get_prompt_for_type,
    get_purchase_atomization_prompt,
    get_text_structure_prompt,
)
from alibi.extraction.schemas import (
    INVOICE_SCHEMA,
    LINE_ITEM_SCHEMA,
    RECEIPT_SCHEMA,
    get_schema,
    validate_extraction,
)
from alibi.extraction.vision import (
    VisionExtractionError,
    detect_document_type,
    extract_from_image,
    extract_json_from_response,
)

__all__ = [
    # Vision extraction
    "VisionExtractionError",
    "extract_from_image",
    "detect_document_type",
    "extract_json_from_response",
    # PDF extraction
    "PDFExtractionError",
    "extract_from_pdf",
    "extract_text_from_pdf",
    "extract_pdf_metadata",
    "get_pdf_page_count",
    # Prompts (v1)
    "RECEIPT_PROMPT",
    "INVOICE_PROMPT",
    "STATEMENT_PROMPT",
    "WARRANTY_PROMPT",
    "get_prompt_for_type",
    "get_text_structure_prompt",
    # Prompts (v2)
    "RECEIPT_PROMPT_V2",
    "INVOICE_PROMPT_V2",
    "PURCHASE_ATOMIZATION_PROMPT",
    "get_purchase_atomization_prompt",
    # Schemas
    "RECEIPT_SCHEMA",
    "INVOICE_SCHEMA",
    "LINE_ITEM_SCHEMA",
    "get_schema",
    "validate_extraction",
]
