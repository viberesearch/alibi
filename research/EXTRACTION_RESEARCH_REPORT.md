# Alibi Extraction Prompts & Pipeline Research Report

**Date**: 2026-02-18
**Scope**: Receipt+card slip combos, sales invoices (VAT invoices), document type detection gaps

---

## 1. Receipt + Card Slip Combo Documents

### Current Behavior

When a single image contains BOTH an itemized receipt AND a card payment terminal slip:

#### 1.1 Vision Detection
**Function**: `detect_document_type()` in `alibi/extraction/vision.py:404-484`

The vision prompt asks the LLM to classify the document as ONE type:
```python
prompt = """Look at this document image and determine its type.
Return ONLY one of these words: receipt, invoice, statement, warranty, contract, payment_confirmation, other
...
Use "payment_confirmation" for card terminal slips, bank transfer confirmations,
or similar payment-only documents that show payment details but no item list.
Use "receipt" for documents with an itemized list of purchased goods/services.
...
Do not include any other text."""
```

**Gap**: The vision model must classify a combo document as either "receipt" OR "payment_confirmation". If the card slip is visually prominent, it may classify as `payment_confirmation` (losing item data). If the receipt is prominent, it classifies as `receipt` (losing payment details).

**Status**: No explicit combo handling — the LLM makes a binary choice.

#### 1.2 Receipt V2 Extraction Prompt
**Function**: `RECEIPT_PROMPT_V2` in `alibi/extraction/prompts.py:60-115`

The receipt prompt extracts:
- Vendor, vendor_address, vendor_phone, vendor_website, vendor_registration
- date, time, document_id, subtotal, tax, total, currency
- **payment_method**, **card_last4**  ← *Includes payment fields*
- language, line_items (with unit, tax, discount, brand, barcode, category)
- raw_text

**Fields Captured**: Yes, the receipt prompt DOES ask for `payment_method` and `card_last4`. So if a combo image is classified as a receipt, the prompt will attempt to extract payment details from the card slip portion.

**Reality Check**: The prompt instruction says:
```
"payment_method": "card/cash/mobile/etc",
"card_last4": "1234 or null",
```

So the receipt prompt CAN extract card payment data. However, if the card slip is a separate visual element within the same image, the LLM may not associate it with the receipt line items — it would need to understand that "this card slip is for this receipt."

#### 1.3 Payment Confirmation V2 Extraction Prompt
**Function**: `PAYMENT_CONFIRMATION_PROMPT_V2` in `alibi/extraction/prompts.py:169-210`

```python
PAYMENT_CONFIRMATION_PROMPT_V2 = """Analyze this payment confirmation document and extract ALL payment details.

This document is a PAYMENT CONFIRMATION — it proves that a payment was made.
It may be a card terminal slip, bank transfer confirmation, contactless payment receipt,
or similar document. Focus on payment details, NOT item lists.
...
Return a JSON object with:
{
    "vendor": "Merchant/Payee name",
    "vendor_address": "Address if visible",
    ...
    "payment_method": "card/cash/contactless/bank_transfer/mobile/etc",
    "card_type": "visa/mastercard/amex/etc or null",
    "card_last4": "1234 or null",
    "authorization_code": "Auth code if visible or null",
    "terminal_id": "Terminal ID if visible or null",
    "reference_number": "Bank reference if visible or null",
    ...
    "line_items": [],     ← Explicitly empty
    "raw_text": "All visible text on the document"
}
...
Rules:
- This is a payment confirmation — line_items should be empty unless items are clearly listed.
- Focus on extracting payment method details (card type, last 4, auth code, terminal).
```

**Gap**: If a combo document is classified as `payment_confirmation`, the prompt explicitly discourages item extraction. Line items will be empty unless "clearly listed." For a combo doc, the receipt items ARE clearly listed but the prompt is optimized for payment-only slips.

### 1.4 Pipeline Handling of Combo Documents

**File**: `alibi/processing/pipeline.py:395-713`

**Step 1: Type Detection** (line 449)
```python
doc_type = self._detect_document_type(file_path)
```
Vision classifies as either `receipt` or `payment_confirmation`.

**Step 2: YAML Cache Check** (line 452-459)
If `.alibi.yaml` exists, use cached data and optionally override doc_type.

**Step 3: Extraction** (line 464)
```python
extracted_data = self._extract_document(file_path, doc_type)
```
Sends the image to Ollama with the appropriate V2 prompt (receipt or payment_confirmation).

**Step 3.5: Heuristic Reclassification** (line 482-494)
```python
if (
    doc_type == ArtifactType.RECEIPT
    and self._looks_like_payment_confirmation(extracted_data)
):
    logger.info(f"Heuristic reclassification: {file_path.name} has no real "
                f"line items + payment details, reclassifying as "
                f"payment_confirmation")
    doc_type = ArtifactType.PAYMENT_CONFIRMATION
```

This checks if a document classified as `receipt` actually has:
- NO real line items (only phantom names like "PURCHASE", "PAYMENT", "SALE", "TOTAL")
- BUT has payment details (card_last4, authorization_code, terminal_id)

**If both are true** → reclassify as payment_confirmation.

**Step 3.6: Content Duplicate Check** (line 558-587)
After extraction, checks if vendor+date+amount+type already exists. If yes, enrich the existing transaction with payment metadata (line 577-581).

### 1.5 The Combo Document Problem

**Scenario**: A single image with receipt on top, card slip on bottom.

**Path A**: Vision classifies as "receipt"
- RECEIPT_PROMPT_V2 extracts: line_items + payment_method/card_last4 ✓
- Heuristic check: Has real line items? Yes → Stays as receipt ✓
- Outcome: Single artifact with both item list and payment details ✓

**Path B**: Vision classifies as "payment_confirmation"
- PAYMENT_CONFIRMATION_PROMPT_V2 extracts: payment details only, line_items intentionally empty
- Pipeline creates payment_confirmation artifact with NO line items ✗
- Outcome: Item data is lost

**Root Cause**: The PAYMENT_CONFIRMATION prompt explicitly avoids item extraction. If the vision model misfires on a combo document, the receipt portion is lost.

### 1.6 Known Limitation (from CONTINUATION_PROMPT.md)

No tests exist for combo documents (single image with receipt + card slip). The proof-of-transaction model handles:
- Receipt processed first, then payment_confirmation found as complementary ✓ (via `find_complementary_match()`)
- Payment_confirmation processed first, then receipt found as complementary ✓ (both arrival orders work)

But this assumes TWO SEPARATE DOCUMENTS. A single combo image has no such separation.

---

## 2. Sales Invoices (Tax Invoices)

### Current Support

**Invoice V2 Prompt**: `INVOICE_PROMPT_V2` in `alibi/extraction/prompts.py:117-167`

```python
INVOICE_PROMPT_V2 = """Analyze this invoice and extract ALL information with maximum detail.
...
Return a JSON object with:
{
    "issuer": "Company/Issuer name",
    "issuer_address": "Company address",
    "issuer_phone": "Phone number if visible",
    "issuer_website": "Website URL if visible",
    "issuer_registration": "VAT/TIC/registration number if visible",
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
```

**Fields Captured**:
- ✓ Issuer registration (VAT/TIC number)
- ✓ Invoice number
- ✓ Issue date, due date
- ✓ Subtotal, tax, amount
- ✓ Line items with tax_rate, tax_type
- ✓ Payment terms
- ✓ Language

**VAT-Specific Handling**: The prompt says:
```
- tax_rate should be a percentage number (e.g. 19 for 19%, not 0.19).
```

And in RECEIPT_PROMPT_V2 there's explicit guidance:
```
- IMPORTANT: If the receipt has a "VAT Analysis" or tax summary table with category codes (like 100, 103, 106) next to percentage rates, extract the PERCENTAGE RATE (e.g. 5.00, 19.00), NOT the category code number.
```

### 2.2 InvoiceRefiner

**File**: `alibi/refiners/invoice.py:12-104`

The refiner processes invoice extraction:

```python
class InvoiceRefiner(BaseRefiner):
    def _refine_specific(self, raw: dict[str, Any], artifact_id: str | None) -> dict[str, Any]:
        """Apply invoice-specific refinement logic."""
        enriched = raw.copy()

        # Set record type
        enriched["record_type"] = RecordType.INVOICE

        # Normalize issuer/vendor
        if "issuer" in enriched:
            enriched["issuer"] = self._normalize_issuer(enriched.get("issuer"))
        elif "vendor" in enriched:
            enriched["issuer"] = self._normalize_issuer(enriched.get("vendor"))

        # Map issuer_* fields to vendor_* for unified storage
        _issuer_map = {
            "issuer_address": "vendor_address",
            "issuer_phone": "vendor_phone",
            "issuer_website": "vendor_website",
            "issuer_registration": "vendor_registration",
            "issuer_id": "vendor_registration",
        }
        for src, dst in _issuer_map.items():
            if src in enriched and dst not in enriched:
                enriched[dst] = enriched[src]

        # Normalize issuer name
        def _normalize_issuer(self, issuer: Any) -> str | None:
            if not issuer:
                return None
            issuer_str = str(issuer).strip()
            # Remove common suffixes
            suffixes = [" LLC", " Inc", " Ltd", " GmbH", " S.L.", " S.A."]
            for suffix in suffixes:
                if issuer_str.endswith(suffix):
                    issuer_str = issuer_str[: -len(suffix)].strip()
            return issuer_str if issuer_str else None
```

**Coverage**:
- ✓ Normalizes issuer name (removes legal suffixes)
- ✓ Maps issuer_* fields to vendor_* (unified schema)
- ✓ Sets record_type = INVOICE
- ✓ Handles invoice_number normalization (removes INV-, INVOICE-, # prefixes)
- ✓ Normalizes currency
- ✓ Parses line items (delegates to PurchaseRefiner)

**Gap**: No special handling for:
- Multiple VAT rates (e.g., 19% and 7% on different items)
- Reverse charge clauses (for B2B invoices)
- Intrastat or customs declarations (for exports)
- Gross vs. net pricing displays (common in some countries)

**However**: These are edge cases. The schema supports them (tax_rate per line item, tax_type enum).

### 2.3 DB Schema Support

**Artifact fields** (from `alibi/db/models.py:261-288`):
```python
class Artifact(TimestampedModel):
    ...
    vendor_registration: Optional[str] = None  # ✓ VAT/TIC number
    document_id: Optional[str] = None           # ✓ Invoice number
    document_date: Optional[date] = None        # ✓ Issue date
    amount: Optional[Decimal] = None            # ✓ Total amount
    ...
```

**LineItem fields** (from `alibi/db/models.py:374-401`):
```python
class LineItem(TimestampedModel):
    ...
    tax_type: TaxType = TaxType.NONE
    tax_rate: Optional[Decimal] = None
    tax_amount: Optional[Decimal] = None
    discount_amount: Optional[Decimal] = None
    discount_percentage: Optional[Decimal] = None
    ...
```

**Schema fully supports sales invoices** (VAT, line-item taxes, discounts).

---

## 3. Document Type Detection Gaps

### 3.1 The Three-Layer Detection System

**Layer 1: Vision Pre-classification** (`_detect_image_type()`, line 147-164)
- Calls `detect_document_type(file_path)` (vision model)
- Maps string result to ArtifactType enum

**Layer 2: LLM Extraction Override** (line 470-480)
```python
llm_type = extracted_data.get("document_type", "").lower()
if (
    llm_type in _LLM_OVERRIDABLE_TYPES      # {"invoice", "payment_confirmation", "contract", "warranty"}
    and llm_type in STR_TO_ARTIFACT_TYPE
    and doc_type != STR_TO_ARTIFACT_TYPE[llm_type]
):
    logger.info(f"LLM detected {file_path.name} as {llm_type}, ...")
    doc_type = STR_TO_ARTIFACT_TYPE[llm_type]
```

**Layer 3: Heuristic Post-Classification** (line 482-494)
- If classified as RECEIPT but extraction has no real line items + has payment details → reclassify as PAYMENT_CONFIRMATION

### 3.2 Gaps in Detection

**Combo Documents (Receipt + Card Slip)**
- No prompt instruction to detect or handle
- Vision prompt forces binary choice
- No splitting logic

**PDF Type Detection**
- PDFs default to INVOICE unless vision detection overrides (line 166-209)
- First page only is rendered for type detection (line 181)
- Multi-page PDFs may have type clues on later pages (e.g., invoice #/date on page 2)

**Payment Confirmation Nuances**
- Bank statement lines are classified as STATEMENT, not PAYMENT_CONFIRMATION
- Individual bank transfers/wire confirmations: no explicit handling
- Contactless payment receipts: handled (mentioned in prompt)

**Edge Cases**
- Receipt + invoice hybrid (POS receipt with invoice number printed) → currently would be classified as receipt
- Statement excerpt with single transaction highlighted → classified as statement (correct)
- Warranty with purchase receipt attached → two documents, would need separate processing

---

## 4. Schema and Database Coverage

### 4.1 Artifact Type Enum

From `alibi/db/models.py:30-40`:
```python
class ArtifactType(str, Enum):
    RECEIPT = "receipt"
    INVOICE = "invoice"
    STATEMENT = "statement"
    WARRANTY = "warranty"
    POLICY = "policy"
    CONTRACT = "contract"
    PAYMENT_CONFIRMATION = "payment_confirmation"
    OTHER = "other"
```

**DB constraint**: All types explicitly listed in schema v8.

### 4.2 Transaction & LineItem Fields

**Transaction**: Captures payment_method, card_last4, account_reference, transaction_time
**LineItem**: Captures tax_rate, tax_type, discount_amount, discount_percentage, category, subcategory, brand, barcode
**ProvenanceRecord**: Tracks source_type (ocr, csv_import, ofx_import, manual, ai_refinement), confidence, processor

**Fully adequate for sales invoices, tax analysis, and payment enrichment.**

---

## 5. Summary: What's Extracted vs. What's Missing

### Receipt V2 Prompt
| Field | Extracted | Notes |
|-------|-----------|-------|
| Vendor, address, phone, website, registration | ✓ | Yes |
| Payment method, card_last4 | ✓ | Yes, included in receipt prompt |
| Line items (name, quantity, unit, price, tax, discount, brand, barcode, category) | ✓ | Yes, detailed |
| Language detection | ✓ | Yes, ISO 639-1 |
| Time | ✓ | Yes, HH:MM:SS if visible |
| **Gap**: Combo document awareness | ✗ | No instruction for handling receipt+card slip in one image |

### Invoice V2 Prompt
| Field | Extracted | Notes |
|-------|-----------|-------|
| Issuer (vendor), address, phone, website, VAT/registration | ✓ | Yes, explicitly VAT number |
| Invoice number | ✓ | Yes |
| Issue date, due date | ✓ | Yes |
| Subtotal, tax, amount | ✓ | Yes |
| Payment terms | ✓ | Yes |
| Line items (with per-item tax rates) | ✓ | Yes, detailed |
| **Gap**: Multiple VAT rates (common in EU) | ~ | Schema supports it, prompt doesn't warn about it |
| **Gap**: Reverse charge, intrastat, customs | ✗ | No specific handling |

### Payment Confirmation V2 Prompt
| Field | Extracted | Notes |
|-------|-----------|-------|
| Vendor, merchant name | ✓ | Yes |
| Payment method, card type, card_last4 | ✓ | Yes |
| Authorization code, terminal ID, reference number | ✓ | Yes, detailed payment fields |
| IBAN, BIC | ✓ | Yes, for bank transfers |
| Line items | ✗ | Explicitly empty; prompt discourages extraction |
| **Gap**: Combo document + items | ✗ | If classified as payment_confirmation, items won't be extracted |

---

## 6. Recommendations

### For Combo Documents (Receipt + Card Slip)
1. **Option A**: Enhance PAYMENT_CONFIRMATION_PROMPT_V2 to extract items if present:
   ```
   "line_items": [// Extract if present, otherwise empty],
   ...
   Rules:
   - If this is ALSO a receipt with itemized purchases, extract line items.
   - If this is payment-only, line_items may be empty.
   ```

2. **Option B**: Add a new combo-aware extraction prompt that handles both simultaneously.

3. **Option C**: Improve vision type detection with a "receipt_with_payment_slip" type (requires schema change).

4. **Option D**: Document current behavior and recommend users separate combo images or use `.alibi.yaml` override.

### For Sales Invoices
1. Current system is adequate — V2 prompt captures all standard invoice fields.
2. Consider adding guidance for VAT Analysis tables (already done in RECEIPT_PROMPT_V2, could repeat in INVOICE_PROMPT_V2 for clarity).
3. Schema already supports multi-rate VAT (per-line tax_rate), no changes needed.

### For Type Detection
1. Document the three-layer detection and heuristic reclassification in user-facing docs.
2. Consider adding logging output to show which layer made the final classification decision.
3. For PDFs, consider rendering multiple pages for type detection (currently only first page).
4. For ambiguous cases (receipt + invoice number), document that receipt takes precedence.

---

## 7. Code Locations Quick Reference

| Component | File | Lines |
|-----------|------|-------|
| Extraction Prompts | `alibi/extraction/prompts.py` | 60-210 |
| Vision Type Detection | `alibi/extraction/vision.py` | 404-484 |
| Pipeline Type Detection (3-layer) | `alibi/processing/pipeline.py` | 130-209, 449, 470-494 |
| InvoiceRefiner | `alibi/refiners/invoice.py` | All |
| PaymentRefiner | `alibi/refiners/payment.py` | All |
| DB Models (enums, schemas) | `alibi/db/models.py` | 30-40, 261-288, 374-401 |
| Complementary Proof Matching | `alibi/matching/duplicates.py` | (find_complementary_match) |
| Pipeline Refiner Routing | `alibi/processing/pipeline.py` | 505-522 |

---

## Conclusion

The alibi extraction system is **well-designed for individual document types** (receipt, invoice, payment_confirmation) with comprehensive field coverage. **Schema and DB fully support** sales invoices with VAT numbers, line-item tax rates, and discount tracking.

**Main gap**: **Combo documents** (receipt + card slip in a single image) are not explicitly handled. Vision detection forces a binary choice, and the PAYMENT_CONFIRMATION prompt explicitly discourages item extraction. This is a **known limitation** documented in the continuation prompt with a workaround: users can use separate images or `.alibi.yaml` overrides.

All extracted data maps correctly to the DB schema. No missing fields or gaps in the refiner logic for supported document types.
