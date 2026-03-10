"""JSON schemas for structured extraction per record type.

These schemas define the expected output format for LLM extraction.
They serve as:
1. Validation targets for extraction output
2. Documentation for prompt engineering
3. Contract between extraction and refiner layers
"""

from __future__ import annotations

from typing import Any

# Line item schema shared across receipt and invoice types
LINE_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Item name in original language",
        },
        "name_en": {
            "type": ["string", "null"],
            "description": "Item name translated to English (if not already English)",
        },
        "quantity": {
            "type": "number",
            "description": "Quantity purchased",
            "default": 1,
        },
        "unit_raw": {
            "type": ["string", "null"],
            "description": "Unit as printed on receipt (e.g. 'kg', 'шт', 'Stk')",
        },
        "unit_price": {
            "type": ["number", "null"],
            "description": "Price per unit including VAT (gross price as printed on receipt). Do NOT back-calculate net prices from VAT rate.",
        },
        "total_price": {
            "type": ["number", "null"],
            "description": "Total price for this line (quantity * unit_price)",
        },
        "tax_rate": {
            "type": ["number", "null"],
            "description": "Tax rate as percentage (e.g. 19 for 19%)",
        },
        "tax_type": {
            "type": ["string", "null"],
            "description": "Tax type: vat, sales_tax, gst, exempt, included, none",
        },
        "discount": {
            "type": ["number", "null"],
            "description": "Discount amount applied to this item",
        },
        "brand": {
            "type": ["string", "null"],
            "description": "Brand name if identifiable",
        },
        "barcode": {
            "type": ["string", "null"],
            "description": "Barcode/EAN/PLU if visible on receipt",
        },
        "category": {
            "type": ["string", "null"],
            "description": "Product category (e.g. 'dairy', 'electronics')",
        },
    },
    "required": ["name"],
}

RECEIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "vendor": {
            "type": ["string", "null"],
            "description": "Store/Business name",
        },
        "vendor_address": {
            "type": ["string", "null"],
            "description": "Store address if visible",
        },
        "vendor_phone": {
            "type": ["string", "null"],
            "description": "Vendor phone number",
        },
        "vendor_website": {
            "type": ["string", "null"],
            "description": "Vendor website URL",
        },
        "vendor_vat": {
            "type": ["string", "null"],
            "description": "VAT registration number",
        },
        "vendor_tax_id": {
            "type": ["string", "null"],
            "description": "TIC/TIN tax identification number",
        },
        "date": {
            "type": ["string", "null"],
            "description": "Transaction date in YYYY-MM-DD format",
        },
        "time": {
            "type": ["string", "null"],
            "description": "Transaction time in HH:MM or HH:MM:SS format",
        },
        "document_id": {
            "type": ["string", "null"],
            "description": "Receipt number if present",
        },
        "subtotal": {
            "type": ["number", "null"],
            "description": "Subtotal before tax",
        },
        "tax": {
            "type": ["number", "null"],
            "description": "Total tax amount",
        },
        "total": {
            "type": "number",
            "description": "Total amount paid",
        },
        "currency": {
            "type": "string",
            "description": "ISO 4217 currency code (EUR, USD, etc.)",
        },
        "payment_method": {
            "type": ["string", "null"],
            "description": "Payment method: card, cash, mobile, etc.",
        },
        "card_last4": {
            "type": ["string", "null"],
            "description": "Last 4 digits of card if visible",
        },
        "language": {
            "type": "string",
            "description": "Document language as ISO 639-1 code (e.g. 'de', 'en', 'ru')",
        },
        "line_items": {
            "type": "array",
            "items": LINE_ITEM_SCHEMA,
            "description": "Individual items on the receipt",
        },
        "raw_text": {
            "type": ["string", "null"],
            "description": "All visible text on the document",
        },
    },
    "required": ["total", "currency", "line_items"],
}

INVOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "issuer": {
            "type": ["string", "null"],
            "description": "Company/Issuer name",
        },
        "issuer_address": {
            "type": ["string", "null"],
            "description": "Issuer address",
        },
        "issuer_phone": {
            "type": ["string", "null"],
            "description": "Issuer phone number",
        },
        "issuer_website": {
            "type": ["string", "null"],
            "description": "Issuer website URL",
        },
        "issuer_vat": {
            "type": ["string", "null"],
            "description": "Issuer VAT registration number",
        },
        "issuer_tax_id": {
            "type": ["string", "null"],
            "description": "Issuer TIC/TIN tax identification number",
        },
        "customer": {
            "type": ["string", "null"],
            "description": "Customer/recipient name if visible",
        },
        "invoice_number": {
            "type": ["string", "null"],
            "description": "Invoice number/ID",
        },
        "issue_date": {
            "type": ["string", "null"],
            "description": "Issue date in YYYY-MM-DD format",
        },
        "due_date": {
            "type": ["string", "null"],
            "description": "Payment due date in YYYY-MM-DD format",
        },
        "subtotal": {
            "type": ["number", "null"],
            "description": "Subtotal before tax",
        },
        "tax": {
            "type": ["number", "null"],
            "description": "Total tax amount",
        },
        "amount": {
            "type": "number",
            "description": "Total invoice amount",
        },
        "currency": {
            "type": "string",
            "description": "ISO 4217 currency code (EUR, USD, etc.)",
        },
        "payment_terms": {
            "type": ["string", "null"],
            "description": "Payment terms (Net 30, immediate, etc.)",
        },
        "language": {
            "type": "string",
            "description": "Document language as ISO 639-1 code (e.g. 'de', 'en', 'ru')",
        },
        "line_items": {
            "type": "array",
            "items": LINE_ITEM_SCHEMA,
            "description": "Individual line items on the invoice",
        },
        "raw_text": {
            "type": ["string", "null"],
            "description": "All visible text on the document",
        },
    },
    "required": ["amount", "currency", "line_items"],
}

WARRANTY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "vendor": {
            "type": ["string", "null"],
            "description": "Manufacturer/Seller name",
        },
        "vendor_address": {
            "type": ["string", "null"],
            "description": "Vendor address if visible",
        },
        "product_name": {
            "type": ["string", "null"],
            "description": "Product name",
        },
        "product_model": {
            "type": ["string", "null"],
            "description": "Model number if present",
        },
        "serial_number": {
            "type": ["string", "null"],
            "description": "Serial number if present",
        },
        "purchase_date": {
            "type": ["string", "null"],
            "description": "Purchase date in YYYY-MM-DD format",
        },
        "date": {
            "type": ["string", "null"],
            "description": "Warranty start date in YYYY-MM-DD format",
        },
        "warranty_end": {
            "type": ["string", "null"],
            "description": "Warranty end date in YYYY-MM-DD format",
        },
        "warranty_type": {
            "type": ["string", "null"],
            "description": "Warranty type: manufacturer, extended, seller, lifetime, limited",
        },
        "coverage": {
            "type": ["string", "null"],
            "description": "What is covered by the warranty",
        },
        "document_id": {
            "type": ["string", "null"],
            "description": "Warranty certificate number if present",
        },
        "total": {
            "type": ["number", "null"],
            "description": "Product purchase price if visible",
        },
        "currency": {
            "type": ["string", "null"],
            "description": "ISO 4217 currency code",
        },
        "language": {
            "type": "string",
            "description": "Document language as ISO 639-1 code",
        },
        "raw_text": {
            "type": ["string", "null"],
            "description": "All visible text on the document",
        },
    },
    "required": ["vendor"],
}

CONTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "vendor": {
            "type": ["string", "null"],
            "description": "Company/Counterparty/Service provider name",
        },
        "vendor_address": {
            "type": ["string", "null"],
            "description": "Company address if visible",
        },
        "customer": {
            "type": ["string", "null"],
            "description": "Other party (customer) name if visible",
        },
        "document_id": {
            "type": ["string", "null"],
            "description": "Contract number/reference",
        },
        "date": {
            "type": ["string", "null"],
            "description": "Signing or effective date in YYYY-MM-DD format",
        },
        "start_date": {
            "type": ["string", "null"],
            "description": "Contract start date in YYYY-MM-DD format",
        },
        "end_date": {
            "type": ["string", "null"],
            "description": "Contract end date in YYYY-MM-DD format or null if open-ended",
        },
        "total": {
            "type": ["number", "null"],
            "description": "Total contract value or periodic amount",
        },
        "currency": {
            "type": ["string", "null"],
            "description": "ISO 4217 currency code",
        },
        "payment_terms": {
            "type": ["string", "null"],
            "description": "Payment terms: monthly, annual, one-time, etc.",
        },
        "renewal": {
            "type": ["string", "null"],
            "description": "Renewal type: auto, manual, none",
        },
        "summary": {
            "type": ["string", "null"],
            "description": "Brief description of what the contract covers",
        },
        "language": {
            "type": "string",
            "description": "Document language as ISO 639-1 code",
        },
        "line_items": {
            "type": "array",
            "items": LINE_ITEM_SCHEMA,
            "description": "Contracted services/products if identifiable",
        },
        "raw_text": {
            "type": ["string", "null"],
            "description": "All visible text on the document",
        },
    },
    "required": ["vendor"],
}

STATEMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "institution": {
            "type": ["string", "null"],
            "description": "Bank/Card company name",
        },
        "account_type": {
            "type": ["string", "null"],
            "description": "Account type: checking, savings, credit, debit",
        },
        "account_last4": {
            "type": ["string", "null"],
            "description": "Last 4 digits of account/card",
        },
        "statement_period": {
            "type": ["object", "null"],
            "description": "Statement period with start and end dates",
        },
        "opening_balance": {
            "type": ["number", "null"],
            "description": "Opening balance",
        },
        "closing_balance": {
            "type": ["number", "null"],
            "description": "Closing balance",
        },
        "currency": {
            "type": "string",
            "description": "ISO 4217 currency code",
        },
        "date": {
            "type": ["string", "null"],
            "description": "Statement date in YYYY-MM-DD format",
        },
        "language": {
            "type": "string",
            "description": "Document language as ISO 639-1 code",
        },
        "transactions": {
            "type": "array",
            "description": "Transaction lines on the statement",
        },
        "raw_text": {
            "type": ["string", "null"],
            "description": "All visible text on the document",
        },
    },
    "required": ["currency", "transactions"],
}

UNIVERSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "description": "Document type classification",
        },
        "vendor": {
            "type": ["string", "null"],
            "description": "Store/Company/Issuer/Merchant name",
        },
        "date": {
            "type": ["string", "null"],
            "description": "Primary document date in YYYY-MM-DD format",
        },
        "total": {
            "type": ["number", "null"],
            "description": "Total amount",
        },
        "currency": {
            "type": ["string", "null"],
            "description": "ISO 4217 currency code",
        },
        "language": {
            "type": "string",
            "description": "Document language as ISO 639-1 code",
        },
        "line_items": {
            "type": "array",
            "items": LINE_ITEM_SCHEMA,
            "description": "Individual items/services",
        },
        "transactions": {
            "type": "array",
            "description": "Statement transaction lines",
        },
        "raw_text": {
            "type": ["string", "null"],
            "description": "All visible text on the document",
        },
    },
    "required": ["document_type"],
}

# Map of schema name to schema dict for programmatic access
SCHEMAS: dict[str, dict[str, Any]] = {
    "receipt": RECEIPT_SCHEMA,
    "invoice": INVOICE_SCHEMA,
    "warranty": WARRANTY_SCHEMA,
    "contract": CONTRACT_SCHEMA,
    "statement": STATEMENT_SCHEMA,
    "universal": UNIVERSAL_SCHEMA,
}


def get_schema(doc_type: str) -> dict[str, Any] | None:
    """Get the JSON schema for a document type.

    Args:
        doc_type: Document type string (receipt, invoice)

    Returns:
        Schema dict or None if no schema defined for this type
    """
    return SCHEMAS.get(doc_type)


def validate_extraction(data: dict[str, Any], doc_type: str) -> list[str]:
    """Validate extracted data against the schema for a document type.

    Lightweight validation without jsonschema dependency.
    Checks required fields and basic type constraints.

    Args:
        data: Extracted data dict
        doc_type: Document type string

    Returns:
        List of validation error strings (empty if valid)
    """
    schema = get_schema(doc_type)
    if schema is None:
        return []  # No schema = no validation errors

    errors: list[str] = []

    # Check required fields
    required = schema.get("required", [])
    properties = schema.get("properties", {})

    for field in required:
        if field not in data or data[field] is None:
            errors.append(f"Missing required field: {field}")

    # Check line_items is a list if present
    if "line_items" in data:
        if not isinstance(data["line_items"], list):
            errors.append("line_items must be a list")
        else:
            # Validate each line item
            item_schema = properties.get("line_items", {}).get("items", {})
            item_required = item_schema.get("required", [])
            for i, item in enumerate(data["line_items"]):
                if not isinstance(item, dict):
                    errors.append(f"line_items[{i}] must be an object")
                    continue
                for field in item_required:
                    if field not in item or item[field] is None:
                        errors.append(
                            f"line_items[{i}]: missing required field: {field}"
                        )

    return errors
