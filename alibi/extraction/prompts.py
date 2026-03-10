"""LLM prompts for document extraction.

V1 prompts: Basic extraction (backward compatible).
V2 prompts: Enhanced extraction with per-item detail, language detection,
            and atomized line items.
"""

from typing import Any

# ---------------------------------------------------------------------------
# V1 prompts (original, kept for backward compatibility)
# ---------------------------------------------------------------------------

RECEIPT_PROMPT = """Analyze this receipt image and extract all information.
Return a JSON object with:
{
    "vendor": "Store/Business name",
    "vendor_address": "Address if visible",
    "document_date": "YYYY-MM-DD",
    "document_id": "Receipt number if present",
    "subtotal": 0.00,
    "tax": 0.00,
    "total": 0.00,
    "currency": "EUR/USD/etc",
    "payment_method": "card/cash/etc",
    "card_last4": "1234 if visible",
    "line_items": [
        {"name": "Item name", "quantity": 1, "unit_price": 0.00, "total": 0.00}
    ],
    "raw_text": "All visible text"
}

Be precise with numbers. If something is unclear, use null."""

INVOICE_PROMPT = """Analyze this invoice and extract all information.
Return a JSON object with:
{
    "vendor": "Company name",
    "vendor_id": "Company tax/VAT ID",
    "vendor_address": "Address",
    "customer": "Customer name if visible",
    "document_id": "Invoice number",
    "document_date": "YYYY-MM-DD",
    "due_date": "YYYY-MM-DD if present",
    "subtotal": 0.00,
    "tax": 0.00,
    "total": 0.00,
    "currency": "EUR/USD/etc",
    "line_items": [
        {"name": "Item/Service", "quantity": 1, "unit_price": 0.00, "total": 0.00}
    ],
    "payment_terms": "Net 30 / immediate / etc",
    "raw_text": "All visible text"
}

Be precise with numbers. If something is unclear, use null."""

# ---------------------------------------------------------------------------
# V2 prompts (enhanced extraction)
# ---------------------------------------------------------------------------

RECEIPT_PROMPT_V2 = """Analyze this receipt image and extract ALL information with maximum detail.

IMPORTANT INSTRUCTIONS:
1. Identify the document language and return its ISO 639-1 code (e.g. "de", "en", "ru", "es").
2. Preserve item names in their ORIGINAL language.
3. For non-English items, also provide an English translation in "name_en".
4. For each line item, extract unit, tax, discount, brand, and barcode if visible.

Return a JSON object with:
{
    "vendor": "Store/Business name",
    "vendor_address": "Full address if visible",
    "vendor_phone": "Phone number if visible",
    "vendor_website": "Website URL if visible",
    "vendor_vat": "VAT registration number if visible",
    "vendor_tax_id": "TIC/TIN tax identification number if visible (separate from VAT)",
    "date": "YYYY-MM-DD",
    "time": "HH:MM or HH:MM:SS if visible, null otherwise",
    "document_id": "Receipt number if present",
    "subtotal": 0.00,
    "tax": 0.00,
    "total": 0.00,
    "currency": "EUR",
    "payment_method": "card/cash/mobile/etc",
    "card_last4": "1234 or null",
    "language": "de",
    "line_items": [
        {
            "name": "Original-language item name",
            "name_en": "English translation or null if already English",
            "quantity": 1,
            "unit_raw": "kg/ml/Stk/pcs/null",
            "unit_quantity": 500,
            "unit_price": 0.00,
            "total_price": 0.00,
            "tax_rate": 19,
            "tax_type": "vat/sales_tax/gst/exempt/included/none",
            "discount": 0.00,
            "brand": "Brand name or null",
            "barcode": "EAN/PLU code or null",
            "category": "dairy/produce/electronics/etc or null"
        }
    ],
    "raw_text": "All visible text on the receipt"
}

Rules:
- Be precise with all numbers (prices, quantities, tax rates).
- Use null for any field that is unclear or not visible.
- tax_rate should be the actual percentage rate (e.g. 19 for 19%, 5 for 5%, not 0.19).
- IMPORTANT: If the receipt has a "VAT Analysis" or tax summary table with category codes (like 100, 103, 106) next to percentage rates, extract the PERCENTAGE RATE (e.g. 5.00, 19.00), NOT the category code number.
- unit_raw should be the unit exactly as printed on the receipt.
- quantity is the count of items purchased (e.g. 2 cans, 1 bag). unit_quantity is the measurement per item (e.g. 400 for a "400g can", 0.76 for "0.76 kg lemons").
- WEIGHED ITEMS: For items sold by weight, quantity is 1 (one purchase) and unit_quantity is the measured weight. Example: "0,250Kg x 5.56 = 1.39" → quantity=1, unit_raw="Kg", unit_quantity=0.250, unit_price=1.39, total_price=1.39. Another example: "1.29 kg PATATES 2.89" → name="PATATES", quantity=1, unit_raw="kg", unit_quantity=1.29, total_price=2.89. Do NOT put the weight in the item name. Do NOT set unit_quantity to conversion factors (e.g. 1000 for kg→g).
- PACKAGED ITEMS WITH WEIGHT IN NAME: If a product name ends in a bare number like "Frozen Blueberries 500" or "Basmati Rice 1000", this typically means grams. Extract unit_raw as "g" and unit_quantity as the number (500, 1000). Same for "ml" volumes (e.g. "Orange Juice 330" means 330ml).
- Ensure total_price = quantity * unit_price where possible.
- unit_price MUST be the VAT-inclusive (gross) price as printed on the receipt. Never back-calculate net (ex-VAT) prices from the VAT rate.
- Include ALL line items, even if some fields are missing."""

INVOICE_PROMPT_V2 = """Analyze this invoice and extract ALL information with maximum detail.

IMPORTANT INSTRUCTIONS:
1. Identify the document language and return its ISO 639-1 code (e.g. "de", "en", "ru", "es").
2. Preserve item/service names in their ORIGINAL language.
3. For non-English items, also provide an English translation in "name_en".
4. For each line item, extract unit, tax, discount, brand, and barcode if visible.

Return a JSON object with:
{
    "issuer": "Company/Issuer name",
    "issuer_address": "Company address",
    "issuer_phone": "Phone number if visible",
    "issuer_website": "Website URL if visible",
    "issuer_vat": "VAT registration number if visible",
    "issuer_tax_id": "TIC/TIN tax identification number if visible (separate from VAT)",
    "customer": "Customer name if visible",
    "invoice_number": "Invoice number/ID",
    "issue_date": "YYYY-MM-DD",
    "due_date": "YYYY-MM-DD or null",
    "subtotal": 0.00,
    "tax": 0.00,
    "amount": 0.00,
    "currency": "EUR",
    "payment_terms": "Net 30 / immediate / etc",
    "language": "de",
    "line_items": [
        {
            "name": "Original-language item/service name",
            "name_en": "English translation or null if already English",
            "quantity": 1,
            "unit_raw": "hr/pcs/m/null",
            "unit_quantity": "content per item or null",
            "unit_price": 0.00,
            "total_price": 0.00,
            "tax_rate": 19,
            "tax_type": "vat/sales_tax/gst/exempt/included/none",
            "discount": 0.00,
            "brand": "Brand name or null",
            "barcode": "Code or null",
            "category": "service/product category or null"
        }
    ],
    "raw_text": "All visible text on the invoice"
}

Rules:
- Be precise with all numbers (prices, quantities, tax rates).
- Use null for any field that is unclear or not visible.
- tax_rate should be a percentage number (e.g. 19 for 19%, not 0.19).
- unit_raw should be the unit exactly as printed on the invoice.
- Include ALL line items, even if some fields are missing."""

PAYMENT_CONFIRMATION_PROMPT_V2 = """Analyze this payment confirmation document and extract ALL payment details.

This document is a PAYMENT CONFIRMATION — it proves that a payment was made.
It may be a card terminal slip, bank transfer confirmation, contactless payment receipt,
or similar document. Focus on payment details, NOT item lists.

IMPORTANT INSTRUCTIONS:
1. Identify the document language and return its ISO 639-1 code.
2. Extract all payment method details visible.

Return a JSON object with:
{
    "vendor": "Merchant/Payee name",
    "vendor_address": "Address if visible",
    "vendor_phone": "Phone if visible",
    "vendor_website": "Website if visible",
    "vendor_vat": "VAT registration number if visible",
    "vendor_tax_id": "TIC/TIN tax identification number if visible (separate from VAT)",
    "date": "YYYY-MM-DD",
    "time": "HH:MM or HH:MM:SS if visible, null otherwise",
    "document_id": "Transaction/reference number if present",
    "total": 0.00,
    "currency": "EUR",
    "payment_method": "card/cash/contactless/bank_transfer/mobile/etc",
    "card_type": "visa/mastercard/amex/etc or null",
    "card_last4": "1234 or null",
    "authorization_code": "Auth code if visible or null",
    "terminal_id": "Terminal ID if visible or null",
    "merchant_id": "Merchant ID / MID if visible or null",
    "reference_number": "Bank reference if visible or null",
    "iban": "IBAN if visible (bank transfer) or null",
    "bic": "BIC/SWIFT if visible or null",
    "language": "en",
    "document_type": "payment_confirmation",
    "line_items": [],
    "raw_text": "All visible text on the document"
}

Rules:
- Be precise with all numbers.
- Use null for any field that is unclear or not visible.
- This is a payment confirmation — line_items should be empty unless items are clearly listed.
- Focus on extracting payment method details (card type, last 4, auth code, terminal).
- If this looks like a receipt with items, still extract it but note document_type as "payment_confirmation" only if it lacks item details."""

PURCHASE_ATOMIZATION_PROMPT = """You are a receipt line item extraction specialist.

Given the following extracted receipt data, perform DEEP atomization of every
line item. Your goal is to extract maximum detail for each purchased item.

Receipt data:
{receipt_json}

For EACH line item, extract or infer:
1. **name**: The item name exactly as printed (preserve original language).
2. **name_en**: English translation if the name is not in English.
3. **quantity**: Numeric quantity (default 1 if not clear).
4. **unit_raw**: Measurement unit as printed (kg, ml, Stk, pcs, etc.).
5. **unit_price**: Price per single unit.
6. **total_price**: Total for this line (quantity * unit_price).
7. **tax_rate**: Tax rate as percentage (e.g. 19 for 19%).
8. **tax_type**: One of: vat, sales_tax, gst, exempt, included, none.
9. **discount**: Discount amount on this item (0 if none).
10. **brand**: Brand name if identifiable from the item name or context.
11. **barcode**: EAN, PLU, or other product code if printed.
12. **category**: Product category (e.g. dairy, produce, bakery, household, beverages).

Return a JSON object:
{{
    "language": "ISO 639-1 code of receipt language",
    "line_items": [
        {{
            "name": "...",
            "name_en": "...",
            "quantity": 1,
            "unit_raw": "...",
            "unit_price": 0.00,
            "total_price": 0.00,
            "tax_rate": 19,
            "tax_type": "vat",
            "discount": 0.00,
            "brand": "...",
            "barcode": "...",
            "category": "..."
        }}
    ]
}}

Rules:
- Preserve original language for item names.
- Use null for any field you cannot determine.
- Do NOT skip any line items.
- If a line appears to be a subtotal, tax, or total line, do NOT include it as a line item.
- tax_rate is a percentage (19 not 0.19).
- Be as specific as possible with categories."""

STATEMENT_PROMPT = """Analyze this bank/credit card statement and extract:
{
    "institution": "Bank/Card company name",
    "account_type": "checking/savings/credit",
    "account_last4": "Last 4 digits",
    "statement_period": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
    "opening_balance": 0.00,
    "closing_balance": 0.00,
    "currency": "EUR/USD/etc",
    "transactions": [
        {
            "date": "YYYY-MM-DD",
            "description": "Transaction description",
            "amount": 0.00,
            "type": "debit/credit"
        }
    ]
}

Be precise with numbers. If something is unclear, use null."""

WARRANTY_PROMPT = """Analyze this warranty document and extract:
{
    "vendor": "Manufacturer/Seller name",
    "product_name": "Product name",
    "product_model": "Model number if present",
    "serial_number": "Serial number if present",
    "purchase_date": "YYYY-MM-DD if visible",
    "warranty_start": "YYYY-MM-DD",
    "warranty_end": "YYYY-MM-DD",
    "warranty_type": "manufacturer/extended/seller",
    "coverage": "What is covered",
    "contact_info": "Support contact if present",
    "raw_text": "All visible text"
}

Be precise with dates. If something is unclear, use null."""

WARRANTY_PROMPT_V2 = """Analyze this warranty document and extract ALL information with maximum detail.

IMPORTANT INSTRUCTIONS:
1. Identify the document language and return its ISO 639-1 code (e.g. "de", "en", "ru", "es").
2. Extract all product and warranty coverage details.

Return a JSON object with:
{
    "vendor": "Manufacturer/Seller name",
    "vendor_address": "Address if visible",
    "vendor_phone": "Phone number if visible",
    "vendor_website": "Website URL if visible",
    "vendor_vat": "VAT registration number if visible",
    "vendor_tax_id": "TIC/TIN tax identification number if visible (separate from VAT)",
    "product_name": "Product name",
    "product_model": "Model number if present",
    "serial_number": "Serial number if present",
    "purchase_date": "YYYY-MM-DD if visible",
    "date": "YYYY-MM-DD (warranty start date)",
    "warranty_end": "YYYY-MM-DD",
    "warranty_type": "manufacturer/extended/seller/lifetime/limited",
    "coverage": "What is covered (brief description)",
    "exclusions": "What is excluded or null",
    "contact_info": "Support contact info if present",
    "document_id": "Warranty certificate number if present",
    "total": 0.00,
    "currency": "EUR",
    "language": "en",
    "document_type": "warranty",
    "line_items": [],
    "raw_text": "All visible text on the document"
}

Rules:
- Be precise with all dates.
- Use null for any field that is unclear or not visible.
- date should be the warranty START date (or purchase date if start not explicit).
- total is the product purchase price if visible, 0 otherwise.
- line_items should be empty for warranty documents."""

CONTRACT_PROMPT = """Analyze this contract/agreement and extract:
{
    "vendor": "Company/Counterparty name",
    "customer": "Other party name if visible",
    "document_id": "Contract number",
    "document_date": "YYYY-MM-DD (signing date)",
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD or null if open-ended",
    "total": 0.00,
    "currency": "EUR/USD/etc",
    "payment_terms": "Monthly/Annual/One-time/etc",
    "summary": "Brief contract description",
    "raw_text": "All visible text"
}

Be precise with dates and amounts. If something is unclear, use null."""

CONTRACT_PROMPT_V2 = """Analyze this contract or agreement and extract ALL information with maximum detail.

IMPORTANT INSTRUCTIONS:
1. Identify the document language and return its ISO 639-1 code (e.g. "de", "en", "ru", "es").
2. Extract all parties, dates, financial terms, and key obligations.

Return a JSON object with:
{
    "vendor": "Company/Counterparty/Service provider name",
    "vendor_address": "Company address if visible",
    "vendor_phone": "Phone number if visible",
    "vendor_website": "Website URL if visible",
    "vendor_vat": "VAT registration number if visible",
    "vendor_tax_id": "TIC/TIN tax identification number if visible (separate from VAT)",
    "customer": "Other party (you/customer) name if visible",
    "document_id": "Contract number/reference",
    "date": "YYYY-MM-DD (signing or effective date)",
    "start_date": "YYYY-MM-DD (contract start)",
    "end_date": "YYYY-MM-DD or null if open-ended",
    "total": 0.00,
    "currency": "EUR",
    "payment_terms": "Monthly/Annual/One-time/etc",
    "renewal": "auto/manual/none or null",
    "notice_period": "Notice period for cancellation or null",
    "summary": "Brief one-line description of what the contract covers",
    "language": "en",
    "document_type": "contract",
    "line_items": [
        {
            "name": "Service/product covered by contract",
            "name_en": "English translation or null if already English",
            "quantity": 1,
            "unit_raw": "month/year/pcs/null",
            "unit_quantity": null,
            "unit_price": 0.00,
            "total_price": 0.00,
            "tax_rate": null,
            "tax_type": "vat/none/null",
            "discount": 0.00,
            "brand": null,
            "barcode": null,
            "category": "service/subscription/lease/etc or null"
        }
    ],
    "raw_text": "All visible text on the contract"
}

Rules:
- Be precise with all dates and amounts.
- Use null for any field that is unclear or not visible.
- date should be the signing/effective date of the contract.
- total is the total contract value (e.g. monthly fee * duration) or the periodic amount if ongoing.
- line_items should list contracted services/products if identifiable. Leave empty if the contract is too general.
- Include ALL parties mentioned in the contract."""

STATEMENT_PROMPT_V2 = """Analyze this bank or credit card statement and extract ALL information with maximum detail.

IMPORTANT INSTRUCTIONS:
1. Identify the document language and return its ISO 639-1 code (e.g. "de", "en", "ru", "es").
2. Extract ALL transactions listed on the statement.
3. For each transaction, determine whether it is a debit (expense) or credit (income).

Return a JSON object with:
{
    "institution": "Bank/Card company name",
    "institution_address": "Bank address if visible",
    "institution_phone": "Phone number if visible",
    "institution_website": "Website URL if visible",
    "institution_registration": "Bank registration/license number if visible",
    "account_type": "checking/savings/credit/debit",
    "account_last4": "Last 4 digits of account/card",
    "account_holder": "Account holder name if visible",
    "statement_period": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
    "opening_balance": 0.00,
    "closing_balance": 0.00,
    "currency": "EUR",
    "document_id": "Statement number/reference if present",
    "date": "YYYY-MM-DD (statement date or period end date)",
    "language": "en",
    "document_type": "statement",
    "transactions": [
        {
            "date": "YYYY-MM-DD",
            "value_date": "YYYY-MM-DD or null (settlement date if different)",
            "description": "Transaction description as printed",
            "vendor": "Merchant/payee name extracted from description",
            "amount": 0.00,
            "type": "debit/credit",
            "reference": "Transaction reference number or null",
            "category": "groceries/utilities/transfer/salary/etc or null"
        }
    ],
    "raw_text": "All visible text on the statement"
}

Rules:
- Be precise with all numbers and dates.
- Use null for any field that is unclear or not visible.
- For debit transactions, amount should be positive (the absolute value spent).
- For credit transactions, amount should be positive (the absolute value received).
- The "type" field distinguishes debits from credits.
- Extract the merchant/vendor name from each transaction description where possible.
- Include ALL transactions, even if some fields are missing."""

UNIVERSAL_PROMPT_V2 = """Analyze this document and extract ALL observable information with maximum detail.

Do NOT assume a document type in advance. Extract EVERY piece of information you can see,
regardless of whether it is a receipt, invoice, statement, payment confirmation, warranty,
contract, or any other document type.

IMPORTANT INSTRUCTIONS:
1. Identify the document language and return its ISO 639-1 code (e.g. "de", "en", "ru", "es").
2. Preserve all names (items, vendors, people) in their ORIGINAL language.
3. For non-English text, also provide English translations where applicable.
4. Extract ALL visible numbers, dates, amounts, and identifiers.

Return a JSON object with ALL of the following sections. Use null for fields not present
in the document. Include empty arrays [] for list fields with no data.

{
    "document_type": "receipt/invoice/statement/payment_confirmation/warranty/contract/other",

    "vendor": "Store/Company/Issuer/Merchant name",
    "vendor_address": "Full address if visible",
    "vendor_phone": "Phone number if visible",
    "vendor_website": "Website URL if visible",
    "vendor_vat": "VAT registration number if visible",
    "vendor_tax_id": "TIC/TIN tax identification number if visible (separate from VAT)",
    "customer": "Customer/recipient name if visible",

    "date": "YYYY-MM-DD (primary document date)",
    "time": "HH:MM or HH:MM:SS if visible, null otherwise",
    "document_id": "Document number/reference if present",

    "subtotal": 0.00,
    "tax": 0.00,
    "total": 0.00,
    "currency": "EUR",
    "language": "en",

    "payment_method": "card/cash/contactless/bank_transfer/mobile/etc or null",
    "card_type": "visa/mastercard/amex/etc or null",
    "card_last4": "1234 or null",
    "authorization_code": "Auth code if visible or null",
    "terminal_id": "Terminal ID if visible or null",
    "merchant_id": "Merchant ID / MID if visible or null",
    "reference_number": "Reference number if visible or null",
    "iban": "IBAN if visible or null",
    "bic": "BIC/SWIFT if visible or null",

    "line_items": [
        {
            "name": "Original-language item/service name",
            "name_en": "English translation or null if already English",
            "quantity": 1,
            "unit_raw": "kg/ml/Stk/pcs/hr/null",
            "unit_quantity": 500,
            "unit_price": 0.00,
            "total_price": 0.00,
            "tax_rate": 19,
            "tax_type": "vat/sales_tax/gst/exempt/included/none",
            "discount": 0.00,
            "brand": "Brand name or null",
            "barcode": "EAN/PLU code or null",
            "category": "dairy/produce/electronics/service/etc or null"
        }
    ],

    "transactions": [
        {
            "date": "YYYY-MM-DD",
            "value_date": "YYYY-MM-DD or null",
            "description": "Transaction description",
            "vendor": "Merchant/payee name",
            "amount": 0.00,
            "type": "debit/credit",
            "reference": "Reference number or null",
            "category": "Category or null"
        }
    ],

    "invoice_number": "Invoice number if this is an invoice, null otherwise",
    "issue_date": "YYYY-MM-DD (invoice issue date if applicable)",
    "due_date": "YYYY-MM-DD or null",
    "payment_terms": "Net 30 / Monthly / Annual / immediate / etc or null",

    "institution": "Bank/Card company name (for statements)",
    "account_type": "checking/savings/credit or null",
    "account_last4": "Last 4 digits of account (for statements)",
    "statement_period": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
    "opening_balance": 0.00,
    "closing_balance": 0.00,

    "product_name": "Product name (for warranties)",
    "product_model": "Model number or null",
    "serial_number": "Serial number or null",
    "warranty_end": "YYYY-MM-DD or null",
    "warranty_type": "manufacturer/extended/seller/lifetime/limited or null",
    "coverage": "What is covered or null",

    "start_date": "YYYY-MM-DD (contract/warranty start)",
    "end_date": "YYYY-MM-DD or null (contract end)",
    "renewal": "auto/manual/none or null",
    "notice_period": "Notice period or null",
    "summary": "Brief one-line description of document content or null",

    "raw_text": "All visible text on the document"
}

Rules:
- Be precise with all numbers (prices, quantities, tax rates, balances).
- Use null for any field that is not present or unclear. Use [] for empty lists.
- tax_rate should be the actual percentage rate (e.g. 19 for 19%, not 0.19).
- IMPORTANT: If the document has a "VAT Analysis" or tax summary table with category codes
  (like 100, 103, 106) next to percentage rates, extract the PERCENTAGE RATE, NOT the code.
- unit_raw should be the unit exactly as printed on the document.
- quantity is the count of items purchased. unit_quantity is the measurement per item
  (e.g. 400 for a "400g can", 0.76 for "0.76 kg lemons").
- WEIGHED ITEMS: For items sold by weight, quantity is 1 and unit_quantity is the measured weight.
- PACKAGED ITEMS WITH WEIGHT IN NAME: If a product name ends in a bare number like
  "Frozen Blueberries 500", this typically means grams.
- For statements: transactions[] should list each line. line_items[] should be empty.
- For receipts/invoices: line_items[] should list items. transactions[] should be empty.
- For payment confirmations: focus on payment fields. line_items[] empty unless items shown.
- Ensure total_price = quantity * unit_price where possible.
- unit_price MUST be the VAT-inclusive (gross) price as printed on the receipt. Never back-calculate net (ex-VAT) prices from the VAT rate.
- Include ALL items/transactions, even if some fields are missing.
- The document_type field is your best classification of what this document is."""

OCR_PROMPT = (
    "Read ALL text visible in this document image. "
    "Return the complete text exactly as written, preserving layout and line breaks. "
    "Include every number, price, date, and symbol you can see."
)


def get_text_extraction_prompt(
    raw_text: str,
    doc_type: str,
    *,
    version: int = 2,
    mode: str = "specialized",
) -> str:
    """Wrap OCR text with the appropriate extraction prompt.

    Replaces the vision-oriented preamble with an OCR text preamble so the
    text-only model can structure the raw OCR output.

    Args:
        raw_text: Raw OCR text from Stage 1.
        doc_type: Document type string (receipt, invoice, …).
        version: Prompt version (1 or 2).
        mode: 'specialized' or 'universal'.

    Returns:
        Full prompt string ready to send to a text-only model.
    """
    base_prompt = get_prompt_for_type(doc_type, version=version, mode=mode)

    # Replace vision-oriented opening line with OCR-text preamble
    preamble = (
        f"The following is OCR text from a {doc_type}. "
        "It may contain minor OCR errors — use your best judgment to correct "
        "obvious mistakes (misread digits, garbled characters).\n"
        "--- BEGIN OCR TEXT ---\n"
        f"{raw_text}\n"
        "--- END OCR TEXT ---\n\n"
    )

    # Strip the vision-oriented opening sentence from the base prompt
    # (lines like "Analyze this receipt image and extract…")
    lines = base_prompt.split("\n")
    # Skip lines until we hit the first instruction block or JSON template
    skip_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("IMPORTANT", "Return a JSON", "{", "1.", "Rules:")):
            skip_idx = i
            break
    # If we found a good start point, use it; otherwise keep the whole prompt
    if skip_idx > 0:
        base_prompt = "\n".join(lines[skip_idx:])

    return preamble + "Extract structured data from the text above.\n\n" + base_prompt


TEXT_STRUCTURE_PROMPT = """Analyze the following extracted text from a document and structure it.
Determine the document type and extract relevant fields.

Text:
{text}

Tables (if any):
{tables}

Return a JSON object with:
{{
    "document_type": "receipt/invoice/statement/warranty/contract/other",
    "vendor": "Company/Store name",
    "document_date": "YYYY-MM-DD",
    "document_id": "Document number if present",
    "total": 0.00,
    "currency": "EUR/USD/etc",
    "summary": "Brief description of document content",
    "extracted_fields": {{...}}  // Type-specific fields
}}

Be precise. If something is unclear, use null."""


def get_prompt_for_type(
    doc_type: str, *, version: int = 1, mode: str = "specialized"
) -> str:
    """Get the appropriate prompt for a document type.

    Args:
        doc_type: Document type string (receipt, invoice, statement, warranty)
        version: Prompt version (1=original, 2=enhanced). Default 1 for
                 backward compatibility.
        mode: Prompt mode — 'specialized' uses type-specific prompts,
              'universal' uses UNIVERSAL_PROMPT_V2 for all types.

    Returns:
        Prompt string for the given document type and version.
    """
    if mode == "universal":
        return UNIVERSAL_PROMPT_V2
    if version == 2:
        return _get_prompt_v2(doc_type)
    prompts = {
        "receipt": RECEIPT_PROMPT,
        "invoice": INVOICE_PROMPT,
        "statement": STATEMENT_PROMPT,
        "warranty": WARRANTY_PROMPT,
        "contract": CONTRACT_PROMPT,
    }
    return prompts.get(doc_type, RECEIPT_PROMPT)


def _get_prompt_v2(doc_type: str) -> str:
    """Get v2 prompt for a document type.

    Args:
        doc_type: Document type string

    Returns:
        V2 prompt string
    """
    v2_prompts: dict[str, str] = {
        "receipt": RECEIPT_PROMPT_V2,
        "invoice": INVOICE_PROMPT_V2,
        "payment_confirmation": PAYMENT_CONFIRMATION_PROMPT_V2,
        "statement": STATEMENT_PROMPT_V2,
        "warranty": WARRANTY_PROMPT_V2,
        "contract": CONTRACT_PROMPT_V2,
    }
    if doc_type in v2_prompts:
        return v2_prompts[doc_type]
    return RECEIPT_PROMPT_V2


def classify_from_extraction(data: dict[str, Any]) -> str:
    """Infer document type from universal extraction output.

    When using UNIVERSAL_PROMPT_V2, the LLM self-classifies via
    document_type. This function validates that classification and
    provides a fallback heuristic based on atom composition.

    Args:
        data: Extraction result dict from UNIVERSAL_PROMPT_V2

    Returns:
        Document type string (receipt, invoice, statement, etc.)
    """
    _KNOWN_TYPES = {
        "receipt",
        "invoice",
        "statement",
        "payment_confirmation",
        "warranty",
        "contract",
    }

    # Trust LLM classification if it's a known type
    llm_type = str(data.get("document_type", "")).lower().strip()
    if llm_type in _KNOWN_TYPES:
        return llm_type

    # Heuristic fallback: infer from which fields are populated
    transactions = data.get("transactions") or []
    line_items = data.get("line_items") or []
    has_institution = bool(data.get("institution"))
    has_warranty_end = bool(data.get("warranty_end"))
    has_payment_only = bool(
        data.get("card_last4")
        or data.get("authorization_code")
        or data.get("terminal_id")
    )

    # Statement: has institution + transactions list
    if has_institution and len(transactions) > 0:
        return "statement"

    # Warranty: has warranty_end or warranty_type
    if has_warranty_end or data.get("warranty_type"):
        return "warranty"

    # Contract: has start_date + end_date or renewal
    if data.get("start_date") and (data.get("end_date") or data.get("renewal")):
        return "contract"

    # Invoice: has invoice_number or due_date
    if data.get("invoice_number") or data.get("due_date"):
        return "invoice"

    # Payment confirmation: payment details but no real line items
    phantom_names = {"PURCHASE", "PAYMENT", "SALE", "TOTAL"}
    real_items = [
        item for item in line_items if item.get("name", "").upper() not in phantom_names
    ]
    if has_payment_only and len(real_items) == 0:
        return "payment_confirmation"

    # Default: receipt (most common document type)
    return "receipt"


# ---------------------------------------------------------------------------
# Correction prompts for three-stage pipeline
# ---------------------------------------------------------------------------

RECEIPT_CORRECTION_PROMPT = """The following data was pre-parsed from OCR text by a deterministic parser.
Some fields may have OCR errors, and some fields (marked null) could not be filled automatically.

--- PRE-PARSED DATA ---
{parsed_yaml}
--- END PRE-PARSED DATA ---

--- ORIGINAL OCR TEXT ---
{ocr_text}
--- END ORIGINAL OCR TEXT ---

Your tasks:
1. Fix any OCR misread values (wrong digits, garbled characters) by checking against the original text.
2. Fill null fields where the OCR text contains the information.
3. Verify line item math: quantity * unit_price should equal total_price.
4. Add name_en (English translation) for each non-English line item.
5. Add category (dairy/produce/bakery/beverages/meat/household/etc) for each line item.
6. Add brand if identifiable from the item name.
7. Identify the document language and set language (ISO 639-1 code).
8. Map tax codes to actual tax rates using the VAT summary if present.
9. For weighed items: quantity=1, unit_quantity=measured weight, unit_raw=unit.
10. For packaged items with bare numbers in name (e.g. "Blueberries 500"): extract unit_quantity and unit_raw.

Return ONLY the corrected JSON object matching the standard extraction schema.
Do NOT include any explanation or markdown formatting — just the raw JSON."""

PAYMENT_CORRECTION_PROMPT = """The following payment confirmation data was pre-parsed from OCR text.
Some fields may have OCR errors.

--- PRE-PARSED DATA ---
{parsed_yaml}
--- END PRE-PARSED DATA ---

--- ORIGINAL OCR TEXT ---
{ocr_text}
--- END ORIGINAL OCR TEXT ---

Your tasks:
1. Fix any OCR misread values by checking against the original text.
2. Fill null fields where the OCR text contains the information.
3. Identify the document language (ISO 639-1 code).
4. Ensure payment details (card_type, card_last4, authorization_code) are correct.

Return ONLY the corrected JSON object matching the payment_confirmation schema.
Do NOT include any explanation or markdown formatting — just the raw JSON."""


INVOICE_CORRECTION_PROMPT = """The following invoice data was pre-parsed from OCR text.
Some fields may have OCR errors, and some fields (marked null) could not be filled automatically.

--- PRE-PARSED DATA ---
{parsed_yaml}
--- END PRE-PARSED DATA ---

--- ORIGINAL OCR TEXT ---
{ocr_text}
--- END ORIGINAL OCR TEXT ---

Your tasks:
1. Fix any OCR misread values by checking against the original text.
2. Fill null fields where the OCR text contains the information.
3. Verify line item math: quantity * unit_price should equal total_price.
4. Add name_en (English translation) for each non-English line item.
5. Add category for each line item.
6. Identify the document language (ISO 639-1 code).
7. Ensure issuer, customer, invoice_number, issue_date, due_date, payment_terms are correct.

Return ONLY the corrected JSON object matching the invoice schema.
Do NOT include any explanation or markdown formatting — just the raw JSON."""


def _build_confidence_hint(field_confidence: dict[str, float]) -> str:
    """Build a confidence hint for the LLM from per-field confidence scores.

    Only mentions fields that are missing (0.0) or uncertain (<1.0).
    Semantic-only fields (name_en, category, brand, language) are excluded
    since the LLM always fills those.
    """
    semantic_fields = {"name_en", "category", "brand", "language"}
    uncertain = {
        k: v
        for k, v in field_confidence.items()
        if v < 1.0 and k not in semantic_fields
    }
    if not uncertain:
        return ""

    missing = [k for k, v in uncertain.items() if v == 0.0]
    low = [k for k, v in uncertain.items() if 0.0 < v < 0.7]

    parts = []
    if missing:
        parts.append(f"MISSING fields (could not be parsed): {', '.join(missing)}")
    if low:
        parts.append(f"LOW-CONFIDENCE fields (may contain errors): {', '.join(low)}")
    return "\n".join(parts)


def _scope_ocr_text(
    ocr_text: str,
    field_confidence: dict[str, float],
    regions: Any | None,
) -> str:
    """Narrow OCR text to regions relevant to low-confidence fields.

    When regions are available and only specific areas need correction,
    returns only the relevant region text (reducing LLM input by ~60%).
    Falls back to full text when regions aren't available or when fields
    span multiple regions.
    """
    if regions is None:
        return ocr_text

    semantic_fields = {"name_en", "category", "brand", "language"}
    uncertain = {
        k for k, v in field_confidence.items() if v < 1.0 and k not in semantic_fields
    }
    if not uncertain:
        return ocr_text

    # Map fields to regions
    header_fields = {
        "vendor",
        "issuer",
        "vendor_vat",
        "vendor_tax_id",
        "issuer_vat",
        "issuer_tax_id",
        "date",
        "issue_date",
        "invoice_number",
        "institution",
        "document_date",
    }
    body_fields = {"line_items", "transactions"}
    footer_fields = {"total", "amount", "subtotal", "tax", "discount"}

    needs_header = bool(uncertain & header_fields)
    needs_body = bool(uncertain & body_fields)
    needs_footer = bool(uncertain & footer_fields)

    # If all regions needed, return full text
    if (needs_header and needs_body and needs_footer) or not any(
        [needs_header, needs_body, needs_footer]
    ):
        return ocr_text

    parts = []
    if needs_header and regions.header:
        parts.append("--- HEADER REGION ---\n" + regions.header)
    if needs_body and regions.body:
        parts.append("--- BODY REGION ---\n" + regions.body)
    if needs_footer and regions.footer:
        parts.append("--- FOOTER REGION ---\n" + regions.footer)

    return "\n\n".join(parts) if parts else ocr_text


def get_correction_prompt(
    parsed_data: dict[str, Any],
    ocr_text: str,
    doc_type: str = "receipt",
    field_confidence: dict[str, float] | None = None,
    regions: Any | None = None,
) -> str:
    """Build a correction prompt with pre-parsed data embedded.

    Args:
        parsed_data: Structured data from the heuristic parser.
        ocr_text: Raw OCR text from Stage 1.
        doc_type: Document type string.
        field_confidence: Per-field confidence scores from parser.
        regions: TextRegions from parser for scoped OCR text.

    Returns:
        Full prompt string for the correction/enrichment LLM call.
    """
    import yaml

    # Compact YAML representation of parsed data
    parsed_yaml = yaml.dump(
        parsed_data,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
        width=120,
    )

    # Scope OCR text to relevant regions when possible
    scoped_text = _scope_ocr_text(ocr_text, field_confidence or {}, regions)

    if doc_type == "payment_confirmation":
        template = PAYMENT_CORRECTION_PROMPT
    elif doc_type == "invoice":
        template = INVOICE_CORRECTION_PROMPT
    else:
        template = RECEIPT_CORRECTION_PROMPT

    prompt = template.format(parsed_yaml=parsed_yaml, ocr_text=scoped_text)

    # Add confidence hints if available
    confidence_hint = ""
    if field_confidence:
        confidence_hint = _build_confidence_hint(field_confidence)
        if confidence_hint:
            confidence_hint = (
                "\n\n⚠️ PARSER CONFIDENCE REPORT:\n" + confidence_hint + "\n"
                "Focus your corrections on these fields first.\n"
            )

    # Wrap with OCR-text preamble for consistency
    preamble = (
        f"The following is OCR text from a {doc_type}, with pre-parsed data. "
        "It may contain minor OCR errors — use your best judgment to correct "
        "obvious mistakes (misread digits, garbled characters).\n\n"
    )

    return preamble + prompt + confidence_hint


def get_purchase_atomization_prompt(receipt_json: str) -> str:
    """Get the purchase atomization prompt with receipt data inserted.

    Args:
        receipt_json: JSON string of the receipt extraction data

    Returns:
        Formatted atomization prompt
    """
    return PURCHASE_ATOMIZATION_PROMPT.format(receipt_json=receipt_json)


def get_text_structure_prompt(
    text: str, tables: list[list[list[str | None]]] | None = None
) -> str:
    """Get prompt for structuring extracted text."""
    tables_str = ""
    if tables:
        for i, table in enumerate(tables):
            tables_str += f"\nTable {i + 1}:\n"
            for row in table:
                tables_str += " | ".join(str(cell) for cell in row) + "\n"
    return TEXT_STRUCTURE_PROMPT.format(text=text, tables=tables_str or "None")
