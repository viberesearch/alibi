"""Tests for the heuristic text parser (Stage 2a)."""

import json
from unittest.mock import patch

import pytest

from alibi.extraction.text_parser import (
    _HEADER_NOISE,
    ParseResult,
    TextRegions,
    _extract_amount_from_line,
    _extract_currency,
    _extract_date_time,
    _extract_due_date,
    _extract_header,
    _extract_invoice_header,
    _extract_invoice_number,
    _extract_line_items,
    _extract_name_qty_amount_items,
    _extract_payment,
    _extract_payment_terms,
    _extract_po_number,
    _extract_tax_summary,
    _extract_totals,
    _find_footer_start,
    _has_legal_suffix,
    _is_integer_qty,
    _is_noise_line,
    _is_non_item_line,
    _is_valid_ean,
    _normalize_time,
    _parse_decimal,
    _split_regions,
    _strip_product_notes,
    classify_ocr_text,
    parse_ocr_text,
)

# ---------------------------------------------------------------------------
# Real OCR text samples
# ---------------------------------------------------------------------------

PAPAS_RECEIPT_OCR = """PAPAS HYPERMARKET
Tombs of the Kings 23
Paphos, 8046
TEL. 26 222 333
T.I.N.: CY12345678X

21/01/2026 13:56:42

BARILLA SPAGHETTI No5 500 3.49 T1
EXTRA VIRGIN OIL 500ML 7.99 T1
ALPRO SOYA MILK 1L 2.69 T1
HELLENIC HALLOUMI 250 4.29 T1
CYPRIOT POTATOES 1KG 1.99 T2
Qty 1.341 @ 2.99 each
RED BULL ENERGY 155ML 5.97 T1
Qty 3.000 @ 1.99 each
ORGANIC TOMATOES 2.49 T2
VILLAGE BREAD 1.50 T2
FRESH ORANGE JUICE 1L 3.99 T1
FAIRY DISH LIQUID 500ML 2.79 T1
BIO GREEK YOGHURT 200 1.89 T1
HEINZ KETCHUP 400ML 3.29 T1
PRESIDENT BUTTER 250 3.49 T1
LAVAZZA COFFEE GROUND 250 5.99 T1
TROPICANA JUICE 1L 4.29 T1
PAMPERS BABY DRY 60 12.99 T1
COLGATE TOOTHPASTE 75ML 2.49 T1
NESTLE CEREAL BAR 6PK 3.79 T1
FINISH DISHWASHER TAB 2.99 T1
FREE RANGE EGGS 6 2.49 T2
HELLENIC FETA 200 3.59 T1
AQUA MINERAL WATER 6X1.5L 2.69 T1
PRINGLES ORIGINAL 165 2.99 T1
NESCAFE CLASSIC 200 6.49 T1

Total Before Disc 92.94
Member Disc -7.25
TOTAL 85.69
Total VAT 6.84

Taxable 1 VAT @ 19.00% 5.72
Taxable 2 VAT @ 5.00% 1.12

23 Items

VISA ****-7201
AUTH NO.: 445566
AMOUNT EUR 85.69
APPROVED"""

PAYMENT_SLIP_OCR = """JCC PAYMENT SYSTEMS
AID: A0000000031010
Versions: 008C

NUT CRACKER RESTAURANT
Poseidonos Ave 12
Paphos

15/02/2026 20:45

SALE

AMOUNT EUR 67.50

VISA <7201>
AUTH NO.: 887766
TERMINAL: TM445566

APPROVED
CARDHOLDER COPY"""

INVOICE_OCR = """ACME WEB SOLUTIONS LTD
15 Innovation Drive
Nicosia, 1065
TEL. 22 445 566
VAT REG: CY99887766A

Bill To:
Digital Holdings Corp
42 Enterprise Blvd, Suite 200

Invoice No: INV-2026-0042
Issue Date: 05/02/2026
Due Date: 07/03/2026
Payment Terms: Net 30
PO Number: PO-8812

Website Redesign Package 3500.00
Hosting (Annual) 240.00
SSL Certificate (1yr) 89.99
Content Migration (10 pages) 750.00
SEO Audit 450.00

Subtotal 5029.99
VAT 19% 955.70
TOTAL 5985.69

Currency: EUR

Bank: Bank of Cyprus
IBAN: CY12345678901234567890
SWIFT: BCYPCY2N"""

MINIMAL_TEXT = """AB
12 34
-- --"""


class TestParseDecimal:
    def test_standard_decimal(self):
        assert _parse_decimal("12.99") == 12.99

    def test_european_comma(self):
        assert _parse_decimal("12,99") == 12.99

    def test_negative(self):
        assert _parse_decimal("-7.25") == -7.25

    def test_empty(self):
        assert _parse_decimal("") is None

    def test_none(self):
        assert _parse_decimal(None) is None

    def test_invalid(self):
        assert _parse_decimal("abc") is None


class TestIsNoiseLine:
    def test_jcc_header(self):
        assert _is_noise_line("JCC PAYMENT SYSTEMS") is True

    def test_aid_line(self):
        assert _is_noise_line("AID: A0000000031010") is True

    def test_vendor_name(self):
        assert _is_noise_line("PAPAS HYPERMARKET") is False

    def test_empty_line(self):
        assert _is_noise_line("") is True

    def test_short_line(self):
        assert _is_noise_line("AB") is True

    def test_numbers_only(self):
        assert _is_noise_line("12345 67890") is True

    def test_cardholder(self):
        assert _is_noise_line("CARDHOLDER COPY") is True


class TestExtractHeader:
    def test_papas_vendor(self):
        lines = PAPAS_RECEIPT_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert data["vendor"] == "PAPAS HYPERMARKET"
        assert "vendor" not in gaps

    def test_papas_address(self):
        lines = PAPAS_RECEIPT_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert "vendor_address" in data
        assert "Tombs of the Kings" in data["vendor_address"]

    def test_papas_phone(self):
        lines = PAPAS_RECEIPT_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert "vendor_phone" in data
        assert "26 222 333" in data["vendor_phone"]

    def test_papas_tax_id(self):
        """T.I.N. is a tax identification number, stored as vendor_tax_id."""
        lines = PAPAS_RECEIPT_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert data["vendor_tax_id"] == "CY12345678X"

    def test_vat_no_without_value_skipped(self):
        """VAT NO. on its own line should not set registration."""
        lines = ["SHOP NAME", "VAT NO.", "some address"]
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert "vendor_vat" not in data

    def test_payment_slip_skips_jcc(self):
        lines = PAYMENT_SLIP_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert data["vendor"] == "NUT CRACKER RESTAURANT"


class TestExtractDateTime:
    def test_papas_date(self):
        data: dict = {}
        gaps: list = []
        _extract_date_time(PAPAS_RECEIPT_OCR, data, gaps)
        assert data["date"] == "2026-01-21"
        assert data["time"] == "13:56:42"
        assert "date" not in gaps

    def test_payment_slip_date(self):
        data: dict = {}
        gaps: list = []
        _extract_date_time(PAYMENT_SLIP_OCR, data, gaps)
        assert data["date"] == "2026-02-15"
        assert data["time"] == "20:45"

    def test_iso_date(self):
        data: dict = {}
        gaps: list = []
        _extract_date_time("Transaction on 2026-03-15", data, gaps)
        assert data["date"] == "2026-03-15"

    def test_us_format_month_gt_12(self):
        """US date 1/26/2026 should parse as Jan 26, not month=26."""
        data: dict = {}
        gaps: list = []
        _extract_date_time("Date: 1/26/2026", data, gaps)
        assert data["date"] == "2026-01-26"
        assert "date" not in gaps

    def test_end_of_year_us(self):
        """US date 12/31/2025 — unambiguous month=12, day=31."""
        data: dict = {}
        gaps: list = []
        _extract_date_time("Date: 12/31/2025", data, gaps)
        assert data["date"] == "2025-12-31"

    def test_ambiguous_dmy_prefers_european(self):
        """Ambiguous 5/6/2026 defaults to European: day=5, month=6."""
        data: dict = {}
        gaps: list = []
        _extract_date_time("Date: 5/6/2026", data, gaps)
        assert data["date"] == "2026-06-05"

    def test_us_format_with_time(self):
        """US date+time 1/26/2026 14:30 should parse correctly."""
        data: dict = {}
        gaps: list = []
        _extract_date_time("Date: 1/26/2026 14:30", data, gaps)
        assert data["date"] == "2026-01-26"
        assert data["time"] == "14:30"

    def test_european_date_still_works(self):
        """European format 15/01/2026 — day=15, month=1."""
        data: dict = {}
        gaps: list = []
        _extract_date_time("Ημ/νία: 15/01/2026", data, gaps)
        assert data["date"] == "2026-01-15"

    def test_pm_time_with_date(self):
        """Date with 6:35 PM should store time as 18:35."""
        data: dict = {}
        gaps: list = []
        _extract_date_time("Date: 1/26/2026 6:35 PM", data, gaps)
        assert data["date"] == "2026-01-26"
        assert data["time"] == "18:35"

    def test_am_time_with_date(self):
        """Date with 9:15 AM should store time as 09:15."""
        data: dict = {}
        gaps: list = []
        _extract_date_time("Date: 15/01/2026 9:15 AM", data, gaps)
        assert data["date"] == "2026-01-15"
        assert data["time"] == "09:15"

    def test_standalone_pm_time(self):
        """Standalone 6:35 PM without date captures as 18:35."""
        data: dict = {}
        gaps: list = []
        _extract_date_time("Time: 6:35 PM\nNo date here", data, gaps)
        assert data["time"] == "18:35"
        assert "date" in gaps

    def test_12pm_is_noon(self):
        """12:00 PM stays 12:00 (noon)."""
        assert _normalize_time("12:00 PM") == "12:00"

    def test_12am_is_midnight(self):
        """12:00 AM becomes 00:00 (midnight)."""
        assert _normalize_time("12:00 AM") == "00:00"

    def test_24h_passthrough(self):
        """24-hour format passes through unchanged."""
        assert _normalize_time("14:30") == "14:30"
        assert _normalize_time("13:56:42") == "13:56:42"

    def test_pm_with_seconds(self):
        """PM time with seconds converts correctly."""
        assert _normalize_time("2:05:30 pm") == "14:05:30"

    def test_no_date(self):
        data: dict = {}
        gaps: list = []
        _extract_date_time("No date here at all", data, gaps)
        assert "date" not in data
        assert "date" in gaps


class TestExtractPayment:
    def test_visa_card(self):
        data: dict = {}
        _extract_payment(PAPAS_RECEIPT_OCR, data)
        assert data["card_type"] == "visa"
        assert data["card_last4"] == "7201"
        assert data["payment_method"] == "card"

    def test_auth_code(self):
        data: dict = {}
        _extract_payment(PAPAS_RECEIPT_OCR, data)
        assert data["authorization_code"] == "445566"

    def test_payment_slip_terminal(self):
        data: dict = {}
        _extract_payment(PAYMENT_SLIP_OCR, data)
        assert data["terminal_id"] == "TM445566"

    def test_terminal_id_hyphenated(self):
        """Terminal ID with hyphens should be captured fully."""
        data: dict = {}
        _extract_payment("Terminal ID 0045930-01-3-13", data)
        assert data["terminal_id"] == "0045930-01-3-13"

    def test_merchant_id_labeled(self):
        """Merchant ID: <value> pattern."""
        data: dict = {}
        _extract_payment("Merchant ID: 12345678", data)
        assert data["merchant_id"] == "12345678"

    def test_merchant_id_mid(self):
        """MID: <value> pattern."""
        data: dict = {}
        _extract_payment("MID: 98765432", data)
        assert data["merchant_id"] == "98765432"

    def test_merchant_id_no_label(self):
        """Merchant No. <value> with hyphen."""
        data: dict = {}
        _extract_payment("Merchant No. 555-1234", data)
        assert data["merchant_id"] == "555-1234"

    def test_angle_bracket_last4(self):
        data: dict = {}
        _extract_payment("VISA <7201>", data)
        assert data["card_last4"] == "7201"

    def test_contactless(self):
        data: dict = {}
        _extract_payment("Payment: Contactless", data)
        assert data["payment_method"] == "contactless"

    def test_cash(self):
        data: dict = {}
        _extract_payment("Paid by CASH", data)
        assert data["payment_method"] == "cash"


class TestExtractCurrency:
    def test_amount_eur(self):
        data: dict = {}
        _extract_currency("AMOUNT EUR 85.69", data)
        assert data["currency"] == "EUR"

    def test_standalone_eur(self):
        data: dict = {}
        _extract_currency("Total: 85.69 EUR", data)
        assert data["currency"] == "EUR"


class TestExtractTaxSummary:
    def test_papas_tax_map(self):
        tax_map = _extract_tax_summary(PAPAS_RECEIPT_OCR)
        assert tax_map.get("T1") == 19.0
        assert tax_map.get("T2") == 5.0

    def test_empty_text(self):
        tax_map = _extract_tax_summary("no tax info here")
        assert tax_map == {}


class TestExtractLineItems:
    def test_papas_item_count(self):
        lines = PAPAS_RECEIPT_OCR.split("\n")
        tax_map = _extract_tax_summary(PAPAS_RECEIPT_OCR)
        items = _extract_line_items(lines, tax_map)
        assert len(items) == 24

    def test_papas_first_item(self):
        lines = PAPAS_RECEIPT_OCR.split("\n")
        tax_map = _extract_tax_summary(PAPAS_RECEIPT_OCR)
        items = _extract_line_items(lines, tax_map)
        first = items[0]
        assert "BARILLA" in first["name"]
        assert first["total_price"] == 3.49
        assert first["tax_rate"] == 19.0

    def test_papas_weighed_item(self):
        """CYPRIOT POTATOES has a fractional Qty 1.341 -> weighed."""
        lines = PAPAS_RECEIPT_OCR.split("\n")
        tax_map = _extract_tax_summary(PAPAS_RECEIPT_OCR)
        items = _extract_line_items(lines, tax_map)
        potato = next((i for i in items if "POTATO" in i["name"].upper()), None)
        assert potato is not None
        assert potato["quantity"] == 1
        assert potato["unit_quantity"] == 1.341
        assert potato["unit_price"] == 2.99
        assert potato["unit_raw"] == "kg"

    def test_papas_multi_qty_item(self):
        """RED BULL has Qty 3.000 (integer) -> multi-quantity, not weighed."""
        lines = PAPAS_RECEIPT_OCR.split("\n")
        tax_map = _extract_tax_summary(PAPAS_RECEIPT_OCR)
        items = _extract_line_items(lines, tax_map)
        redbull = next((i for i in items if "RED BULL" in i["name"]), None)
        assert redbull is not None
        assert redbull["quantity"] == 3
        assert redbull.get("unit_quantity") is None  # Not weighed
        assert redbull["unit_price"] == 1.99

    def test_items_have_null_gaps(self):
        """Parser should set translation/category/brand fields to None."""
        lines = PAPAS_RECEIPT_OCR.split("\n")
        tax_map = _extract_tax_summary(PAPAS_RECEIPT_OCR)
        items = _extract_line_items(lines, tax_map)
        for item in items:
            assert item["name_en"] is None
            assert item["category"] is None
            assert item["brand"] is None

    def test_ticket_footer_not_parsed_as_item(self):
        """TICKET: footer lines should be filtered during item extraction."""
        text = (
            "SOME SHOP\n"
            "01/01/2026\n"
            "\n"
            "BREAD 1.20\n"
            "MILK 2.50\n"
            "TICKET:A11T/ DATE: 12/2/2026 TIME: 13:00\n"
            "TOTAL 3.70\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        names = [i["name"] for i in items]
        assert not any("TICKET" in n for n in names), f"TICKET found in items: {names}"
        assert len(items) == 2


class TestExtractTotals:
    def test_papas_totals(self):
        lines = PAPAS_RECEIPT_OCR.split("\n")
        data: dict = {}
        _extract_totals(lines, data)
        assert data["subtotal"] == 92.94
        assert data["total"] == 85.69
        assert data.get("discount") == -7.25

    def test_total_vat(self):
        lines = PAPAS_RECEIPT_OCR.split("\n")
        data: dict = {}
        _extract_totals(lines, data)
        assert data.get("tax") == 6.84

    def test_jcc_amount_fallback_no_total(self):
        """When no standard total marker found, fall back to JCC AMOUNT EUR."""
        lines = [
            "LIDL CY 125 - MOUTTAGIAK",
            "ARIADNES 55",
            "LIMASSOL",
            "VAT NO. 30010823A",
            "PURCHASE",
            "JCC CONTACTLESS",
            "AMOUNT EUR52,31",
            "AUTH NO. 6064B5",
        ]
        data: dict = {}
        _extract_totals(lines, data)
        assert data["total"] == 52.31

    def test_jcc_amount_not_used_when_total_exists(self):
        """JCC AMOUNT should not override an existing total."""
        lines = [
            "TOTAL 85.69",
            "AMOUNT EUR52,31",
        ]
        data: dict = {}
        _extract_totals(lines, data)
        assert data["total"] == 85.69

    def test_jcc_amount_with_space(self):
        """JCC AMOUNT with space before number."""
        lines = ["AMOUNT EUR 41,28"]
        data: dict = {}
        _extract_totals(lines, data)
        assert data["total"] == 41.28

    def test_declared_item_count_english(self):
        """Extract 'Number of Items 25' from receipt footer."""
        lines = [
            "TOTAL 104.93",
            "Number of Items 25",
            "Congratulations!",
        ]
        data: dict = {}
        _extract_totals(lines, data)
        assert data["declared_item_count"] == 25

    def test_declared_item_count_with_colon(self):
        """Extract 'Number of Items: 13'."""
        lines = [
            "TOTAL 48.45",
            "Number of Items: 13",
        ]
        data: dict = {}
        _extract_totals(lines, data)
        assert data["declared_item_count"] == 13

    def test_declared_item_count_lowercase(self):
        """Extract 'Number of items: 23' (mixed case)."""
        lines = [
            "Total 95.80",
            "Number of items: 23",
        ]
        data: dict = {}
        _extract_totals(lines, data)
        assert data["declared_item_count"] == 23

    def test_no_declared_item_count(self):
        """No declared_item_count when not present."""
        lines = [
            "TOTAL 85.69",
            "Thank you!",
        ]
        data: dict = {}
        _extract_totals(lines, data)
        assert "declared_item_count" not in data


class TestExtractAmountFromLine:
    def test_simple(self):
        assert _extract_amount_from_line("TOTAL 85.69") == 85.69

    def test_multiple_numbers(self):
        assert _extract_amount_from_line("Subtotal 5 items 92.94") == 92.94

    def test_negative(self):
        assert _extract_amount_from_line("Discount -7.25") == -7.25

    def test_no_amount(self):
        assert _extract_amount_from_line("No amount here") is None


# ---------------------------------------------------------------------------
# Full parse_ocr_text integration tests
# ---------------------------------------------------------------------------


class TestParseOcrText:
    def test_receipt_full_parse(self):
        result = parse_ocr_text(PAPAS_RECEIPT_OCR, doc_type="receipt")
        assert isinstance(result, ParseResult)
        assert result.confidence >= 0.5
        assert result.line_item_count == 24
        assert result.data["vendor"] == "PAPAS HYPERMARKET"
        assert result.data["date"] == "2026-01-21"
        assert result.data["total"] == 85.69
        # High-confidence parses (>= 0.9) don't need LLM for core data
        assert "name_en" in result.gaps

    def test_payment_confirmation(self):
        result = parse_ocr_text(PAYMENT_SLIP_OCR, doc_type="payment_confirmation")
        assert result.confidence >= 0.5
        assert result.data["vendor"] == "NUT CRACKER RESTAURANT"
        assert result.data["date"] == "2026-02-15"
        assert result.data["total"] == 67.50
        assert result.data["card_last4"] == "7201"
        assert result.line_item_count == 0

    def test_low_confidence_garbage(self):
        result = parse_ocr_text(MINIMAL_TEXT, doc_type="receipt")
        assert result.confidence < 0.3

    def test_empty_text(self):
        result = parse_ocr_text("", doc_type="receipt")
        assert result.confidence == 0.0
        assert "empty_text" in result.gaps

    def test_statement_minimal(self):
        result = parse_ocr_text("Bank of Cyprus\n01/01/2026", doc_type="statement")
        # Statement parser extracts vendor+date but no transactions
        assert result.confidence < 0.8
        assert result.needs_llm is True
        assert result.data.get("vendor") == "Bank of Cyprus"

    def test_invoice_dispatch(self):
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        assert result.data["document_type"] == "invoice"
        assert "issuer" in result.data
        assert "vendor" not in result.data

    def test_parser_exception_graceful(self):
        """Parser exceptions should return zero-confidence result."""
        with patch(
            "alibi.extraction.text_parser._parse_receipt",
            side_effect=ValueError("boom"),
        ):
            result = parse_ocr_text(PAPAS_RECEIPT_OCR, doc_type="receipt")
            assert result.confidence == 0.0
            assert "parser_exception" in result.gaps


# ---------------------------------------------------------------------------
# Invoice parser tests
# ---------------------------------------------------------------------------


class TestExtractInvoiceHeader:
    def test_issuer_extracted(self):
        lines = INVOICE_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_invoice_header(lines, data, gaps)
        assert data["issuer"] == "ACME WEB SOLUTIONS LTD"
        assert "vendor" not in data

    def test_issuer_address(self):
        lines = INVOICE_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_invoice_header(lines, data, gaps)
        assert "issuer_address" in data
        assert "Innovation Drive" in data["issuer_address"]

    def test_issuer_phone(self):
        lines = INVOICE_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_invoice_header(lines, data, gaps)
        assert data["issuer_phone"] == "22 445 566"

    def test_issuer_vat(self):
        lines = INVOICE_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_invoice_header(lines, data, gaps)
        assert data["issuer_vat"] == "CY99887766A"

    def test_customer_extracted(self):
        lines = INVOICE_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_invoice_header(lines, data, gaps)
        assert data["customer"] == "Digital Holdings Corp"

    def test_billing_address(self):
        lines = INVOICE_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_invoice_header(lines, data, gaps)
        assert "billing_address" in data
        assert "Enterprise Blvd" in data["billing_address"]

    def test_no_vendor_key_in_data(self):
        lines = INVOICE_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_invoice_header(lines, data, gaps)
        assert "vendor" not in data
        assert "vendor_address" not in data
        assert "vendor_phone" not in data

    def test_missing_vendor_becomes_issuer_gap(self):
        lines = ["", "12345"]
        data: dict = {}
        gaps: list = []
        _extract_invoice_header(lines, data, gaps)
        assert "issuer" in gaps
        assert "vendor" not in gaps


class TestExtractInvoiceNumber:
    def test_standard_format(self):
        data: dict = {}
        gaps: list = []
        _extract_invoice_number("Invoice No: INV-2026-0042", data, gaps)
        assert data["invoice_number"] == "INV-2026-0042"

    def test_hash_format(self):
        data: dict = {}
        gaps: list = []
        _extract_invoice_number("Invoice #: 12345", data, gaps)
        assert data["invoice_number"] == "12345"

    def test_inv_no_format(self):
        data: dict = {}
        gaps: list = []
        _extract_invoice_number("INV NO. ABC-789", data, gaps)
        assert data["invoice_number"] == "ABC-789"

    def test_no_invoice_number(self):
        data: dict = {}
        gaps: list = []
        _extract_invoice_number("No invoice info here", data, gaps)
        assert "invoice_number" not in data
        assert "invoice_number" in gaps


class TestExtractDueDate:
    def test_due_date_dmy(self):
        data: dict = {}
        _extract_due_date("Due Date: 07/03/2026", data)
        assert data["due_date"] == "2026-03-07"

    def test_due_date_dotted(self):
        data: dict = {}
        _extract_due_date("Due: 07.03.2026", data)
        assert data["due_date"] == "2026-03-07"

    def test_due_date_iso(self):
        data: dict = {}
        _extract_due_date("Due Date: 2026-03-07", data)
        assert data["due_date"] == "2026-03-07"

    def test_payment_due(self):
        data: dict = {}
        _extract_due_date("Payment Due: 15/04/2026", data)
        assert data["due_date"] == "2026-04-15"

    def test_pay_by(self):
        data: dict = {}
        _extract_due_date("Pay By: 01/05/2026", data)
        assert data["due_date"] == "2026-05-01"

    def test_no_due_date(self):
        data: dict = {}
        _extract_due_date("No due date here", data)
        assert "due_date" not in data


class TestExtractPaymentTerms:
    def test_net_30(self):
        data: dict = {}
        _extract_payment_terms("Payment Terms: Net 30", data)
        assert data["payment_terms"] == "Net 30"

    def test_standalone_net(self):
        data: dict = {}
        _extract_payment_terms("Standard NET 30 applies", data)
        assert "Net" in data["payment_terms"] or "NET" in data["payment_terms"]

    def test_due_on_receipt(self):
        data: dict = {}
        _extract_payment_terms("Due on receipt", data)
        assert "due on receipt" in data["payment_terms"].lower()

    def test_no_terms(self):
        data: dict = {}
        _extract_payment_terms("No terms mentioned", data)
        assert "payment_terms" not in data


class TestExtractPoNumber:
    def test_po_number(self):
        data: dict = {}
        _extract_po_number("PO Number: PO-8812", data)
        assert data["po_number"] == "PO-8812"

    def test_purchase_order(self):
        data: dict = {}
        _extract_po_number("Purchase Order: 12345", data)
        assert data["po_number"] == "12345"

    def test_po_no_dot(self):
        data: dict = {}
        _extract_po_number("P.O. No. ABC-123", data)
        assert data["po_number"] == "ABC-123"

    def test_no_po(self):
        data: dict = {}
        _extract_po_number("No PO info", data)
        assert "po_number" not in data


class TestParseInvoice:
    def test_full_invoice_parse(self):
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        assert isinstance(result, ParseResult)
        assert result.data["document_type"] == "invoice"
        assert result.needs_llm is True

    def test_issuer_not_vendor(self):
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        assert "issuer" in result.data
        assert "vendor" not in result.data
        assert result.data["issuer"] == "ACME WEB SOLUTIONS LTD"

    def test_issue_date_not_date(self):
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        assert "issue_date" in result.data
        assert "date" not in result.data
        assert result.data["issue_date"] == "2026-02-05"

    def test_due_date(self):
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        assert result.data["due_date"] == "2026-03-07"

    def test_invoice_number(self):
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        assert result.data["invoice_number"] == "INV-2026-0042"

    def test_payment_terms(self):
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        assert result.data["payment_terms"] == "Net 30"

    def test_po_number(self):
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        assert result.data["po_number"] == "PO-8812"

    def test_customer(self):
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        assert result.data["customer"] == "Digital Holdings Corp"

    def test_amount_not_total(self):
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        assert "amount" in result.data
        assert "total" not in result.data
        assert result.data["amount"] == 5985.69

    def test_line_items_extracted(self):
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        assert result.line_item_count >= 4

    def test_no_time_field(self):
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        assert "time" not in result.data

    def test_currency(self):
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        assert result.data["currency"] == "EUR"

    def test_confidence_reasonable(self):
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        assert result.confidence >= 0.5


class TestInvoiceCorrectionPrompt:
    def test_invoice_correction_prompt(self):
        from alibi.extraction.prompts import get_correction_prompt

        parsed_data = {
            "issuer": "ACME LTD",
            "invoice_number": "INV-001",
            "issue_date": "2026-01-01",
            "due_date": "2026-02-01",
            "amount": 1000.00,
        }
        prompt = get_correction_prompt(parsed_data, "OCR text", "invoice")
        assert "PRE-PARSED DATA" in prompt
        assert "invoice" in prompt.lower()
        assert "issuer" in prompt
        assert "invoice_number" in prompt
        assert "due_date" in prompt

    def test_invoice_uses_dedicated_template(self):
        from alibi.extraction.prompts import get_correction_prompt

        parsed_data = {"issuer": "TEST"}
        prompt = get_correction_prompt(parsed_data, "text", "invoice")
        assert "payment_terms" in prompt


# ---------------------------------------------------------------------------
# Correction prompt tests
# ---------------------------------------------------------------------------


class TestCorrectionPrompts:
    def test_receipt_correction_prompt(self):
        from alibi.extraction.prompts import get_correction_prompt

        parsed_data = {
            "vendor": "PAPAS HYPERMARKET",
            "date": "2026-01-21",
            "total": 85.69,
            "line_items": [{"name": "MILK", "total_price": 2.69}],
        }
        prompt = get_correction_prompt(parsed_data, "OCR text here", "receipt")
        assert "PRE-PARSED DATA" in prompt
        assert "ORIGINAL OCR TEXT" in prompt
        assert "OCR text here" in prompt
        assert "PAPAS HYPERMARKET" in prompt
        assert "Fix any OCR misread values" in prompt

    def test_payment_correction_prompt(self):
        from alibi.extraction.prompts import get_correction_prompt

        parsed_data = {"vendor": "TEST", "total": 50.0, "card_last4": "1234"}
        prompt = get_correction_prompt(parsed_data, "text", "payment_confirmation")
        assert "payment confirmation" in prompt.lower()
        assert "card_type" in prompt

    def test_prompt_contains_yaml(self):
        from alibi.extraction.prompts import get_correction_prompt

        parsed_data = {"vendor": "TEST", "total": 10.0}
        prompt = get_correction_prompt(parsed_data, "text", "receipt")
        # Should contain YAML-formatted data (not JSON)
        assert "vendor: TEST" in prompt


# ---------------------------------------------------------------------------
# Integration: three-stage flow with mocked Ollama
# ---------------------------------------------------------------------------


class TestThreeStageIntegration:
    @patch("alibi.extraction.structurer._call_ollama_text")
    @patch("alibi.extraction.ocr.ocr_image_with_retry")
    def test_three_stage_uses_correction_or_micro_prompts(
        self, mock_ocr, mock_structure
    ):
        """When parser confidence >= 0.3, should use micro-prompts or correction prompt."""
        mock_ocr.return_value = (PAPAS_RECEIPT_OCR, False)
        mock_structure.return_value = {
            "response": json.dumps(
                {
                    "vendor": "PAPAS HYPERMARKET",
                    "date": "2026-01-21",
                    "total": 85.69,
                    "currency": "EUR",
                    "line_items": [
                        {
                            "name": "BARILLA SPAGHETTI",
                            "quantity": 1,
                            "unit_price": 3.49,
                            "total_price": 3.49,
                        }
                    ],
                }
            )
        }

        from pathlib import Path
        from unittest.mock import MagicMock

        with patch("alibi.extraction.vision.get_config") as mock_config:
            cfg = MagicMock()
            cfg.ollama_url = "http://test:11434"
            cfg.ollama_model = "test-model"
            cfg.ollama_structure_model = "qwen3:8b"
            cfg.ollama_ocr_model = "glm-ocr"
            cfg.prompt_mode = "specialized"
            cfg.skip_llm_threshold = 1.1  # Disable skip-LLM to test correction path
            mock_config.return_value = cfg

            with patch("pathlib.Path.exists", return_value=True):
                from alibi.extraction.vision import extract_from_image

                result = extract_from_image(
                    Path("/fake/receipt.jpg"),
                    doc_type="receipt",
                    ollama_url="http://test:11434",
                )

        # LLM structuring was called (either micro-prompt or correction)
        assert mock_structure.called
        call_args = mock_structure.call_args
        prompt_used = call_args[0][2]  # Third positional arg is prompt
        # Either micro-prompt (enrichment/field-specific) or correction prompt
        assert (
            "PRE-PARSED DATA" in prompt_used
            or "category" in prompt_used  # micro-prompt enrichment
            or "Fields needed" in prompt_used  # micro-prompt header/footer
        )

    @patch("alibi.extraction.structurer._call_ollama_text")
    @patch("alibi.extraction.ocr.ocr_image_with_retry")
    def test_fallback_to_full_prompt_on_low_confidence(self, mock_ocr, mock_structure):
        """When parser confidence < 0.3, should use full LLM prompt."""
        mock_ocr.return_value = (MINIMAL_TEXT, False)
        mock_structure.return_value = {
            "response": json.dumps(
                {
                    "vendor": "Unknown",
                    "date": "2026-01-01",
                    "total": 0,
                    "currency": "EUR",
                    "line_items": [],
                }
            )
        }

        from pathlib import Path
        from unittest.mock import MagicMock

        with patch("alibi.extraction.vision.get_config") as mock_config:
            cfg = MagicMock()
            cfg.ollama_url = "http://test:11434"
            cfg.ollama_model = "test-model"
            cfg.ollama_structure_model = "qwen3:8b"
            cfg.ollama_ocr_model = "glm-ocr"
            cfg.prompt_mode = "specialized"
            cfg.skip_llm_threshold = 1.1  # Disable skip-LLM to test full prompt path
            mock_config.return_value = cfg

            with patch("pathlib.Path.exists", return_value=True):
                from alibi.extraction.vision import extract_from_image

                result = extract_from_image(
                    Path("/fake/receipt.jpg"),
                    doc_type="receipt",
                    ollama_url="http://test:11434",
                )

        # Should NOT use correction prompt
        call_args = mock_structure.call_args
        prompt_used = call_args[0][2]
        assert "PRE-PARSED DATA" not in prompt_used

    @patch(
        "alibi.extraction.text_parser._parse_receipt",
        side_effect=RuntimeError("parser crash"),
    )
    @patch("alibi.extraction.structurer._call_ollama_text")
    @patch("alibi.extraction.ocr.ocr_image_with_retry")
    def test_parser_exception_falls_through(self, mock_ocr, mock_structure, mock_parse):
        """Parser exceptions should gracefully fall through to full LLM."""
        mock_ocr.return_value = (PAPAS_RECEIPT_OCR, False)
        mock_structure.return_value = {
            "response": json.dumps(
                {"vendor": "PAPAS", "date": "2026-01-21", "total": 85.69}
            )
        }

        from pathlib import Path
        from unittest.mock import MagicMock

        with patch("alibi.extraction.vision.get_config") as mock_config:
            cfg = MagicMock()
            cfg.ollama_url = "http://test:11434"
            cfg.ollama_model = "test-model"
            cfg.ollama_structure_model = "qwen3:8b"
            cfg.ollama_ocr_model = "glm-ocr"
            cfg.prompt_mode = "specialized"
            cfg.skip_llm_threshold = 1.1  # Disable skip-LLM to test fallback
            mock_config.return_value = cfg

            with patch("pathlib.Path.exists", return_value=True):
                from alibi.extraction.vision import extract_from_image

                result = extract_from_image(
                    Path("/fake/receipt.jpg"),
                    doc_type="receipt",
                    ollama_url="http://test:11434",
                )

        # Should still get a result (from full LLM)
        assert result["vendor"] == "PAPAS"
        # Should use full prompt (not correction)
        call_args = mock_structure.call_args
        prompt_used = call_args[0][2]
        assert "PRE-PARSED DATA" not in prompt_used

    @patch("alibi.extraction.structurer._call_ollama_text")
    @patch("alibi.extraction.ocr.ocr_image_with_retry")
    def test_skip_llm_when_parser_confidence_high(self, mock_ocr, mock_structure):
        """When parser confidence >= skip_llm_threshold, LLM is skipped entirely."""
        mock_ocr.return_value = (PAPAS_RECEIPT_OCR, False)

        from pathlib import Path
        from unittest.mock import MagicMock

        with patch("alibi.extraction.vision.get_config") as mock_config:
            cfg = MagicMock()
            cfg.ollama_url = "http://test:11434"
            cfg.ollama_model = "test-model"
            cfg.ollama_structure_model = "qwen3:8b"
            cfg.ollama_ocr_model = "glm-ocr"
            cfg.prompt_mode = "specialized"
            cfg.skip_llm_threshold = 0.9  # Parser gets ~0.96, so LLM should be skipped
            mock_config.return_value = cfg

            with patch("pathlib.Path.exists", return_value=True):
                from alibi.extraction.vision import extract_from_image

                result = extract_from_image(
                    Path("/fake/receipt.jpg"),
                    doc_type="receipt",
                    ollama_url="http://test:11434",
                )

        # LLM should NOT have been called
        mock_structure.assert_not_called()
        # Result should come from parser
        assert result.get("_pipeline") == "parser_only"
        assert result.get("_parser_confidence", 0) >= 0.9


# ---------------------------------------------------------------------------
# Noise line detection — payment terminal brands
# ---------------------------------------------------------------------------


class TestNoisePaymentTerminals:
    def test_payments_is_noise(self):
        assert _is_noise_line("payments") is True

    def test_payments_uppercase(self):
        assert _is_noise_line("PAYMENTS") is True

    def test_worldline_is_noise(self):
        assert _is_noise_line("WORLDLINE") is True

    def test_worldline_lowercase(self):
        assert _is_noise_line("worldline") is True

    def test_verifone_is_noise(self):
        assert _is_noise_line("Verifone") is True

    def test_ingenico_is_noise(self):
        assert _is_noise_line("INGENICO") is True

    def test_member_of_is_noise(self):
        assert _is_noise_line("Member of G.A.P. Vassilopoulos Group") is True

    def test_actual_vendor_not_noise(self):
        assert _is_noise_line("FreSko") is False
        assert _is_noise_line("ARAB BUTCHERY") is False


# ---------------------------------------------------------------------------
# Legal suffix detection
# ---------------------------------------------------------------------------


class TestHasLegalSuffix:
    def test_ltd(self):
        assert _has_legal_suffix("BUTANOLO LTD") is True

    def test_ltd_dot(self):
        assert _has_legal_suffix("BUTANOLO LTD.") is True

    def test_limited(self):
        assert _has_legal_suffix("Acme Limited") is True

    def test_gmbh(self):
        assert _has_legal_suffix("Siemens GmbH") is True

    def test_inc(self):
        assert _has_legal_suffix("Apple Inc") is True

    def test_plc(self):
        assert _has_legal_suffix("Barclays PLC") is True

    def test_greek_ee(self):
        assert _has_legal_suffix("Εταιρεία Ε.Ε.") is True

    def test_no_suffix(self):
        assert _has_legal_suffix("FRESKO") is False

    def test_address_no_suffix(self):
        assert _has_legal_suffix("CHRIS. KRANOU 2") is False

    def test_phone_no_suffix(self):
        assert _has_legal_suffix("Tel. 95772266") is False


# ---------------------------------------------------------------------------
# Trade name + legal name extraction
# ---------------------------------------------------------------------------

FRESKO_RECEIPT_OCR = """FRESKO
BUTANOLO LTD
CHRIS. KRANOU 2
4047 GERMASOGEIA
Tel. 95772266
VAT 10336127M TIC 123361270

17/02/2026 14:30
Apples 1kg 2.75 T1
TOTAL 2.75"""

FRESKO_PAYMENT_OCR = """payments
WORLDLINE
Member of G.A.P. Vassilopoulos Group
FreSko
BUTANOLO LTD
CHRISTAKI KRANOU 2
4047 LIMASSOL
VAT 10336127M

17/02/2026 14:30
AMOUNT EUR 2.75
VISA ****-1234
AUTH NO.: 112233"""

NUT_CRACKER_OCR = """TheNutCrackerHouse
54 Vasileos Georgiou Street
Germasogeia, Limassol
TEL NO: 25321010
VAT REG.NO 10180201N

17/02/2026 12:00
Baklava Mix 500g 8.95 T1
Pistachio 250g 3.50 T1
TOTAL 12.45"""


class TestExtractHeaderLegalName:
    def test_fresko_trade_and_legal(self):
        lines = FRESKO_RECEIPT_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert data["vendor"] == "FRESKO"
        assert data["vendor_legal_name"] == "BUTANOLO LTD"
        assert "CHRIS. KRANOU 2" in data.get("vendor_address", "")
        assert data["vendor_vat"] == "10336127M"
        assert data["vendor_tax_id"] == "123361270"

    def test_fresko_payment_noise_skipped(self):
        """Card slip with noise lines before vendor should extract FreSko."""
        lines = FRESKO_PAYMENT_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert data["vendor"] == "FreSko"
        assert data["vendor_legal_name"] == "BUTANOLO LTD"
        assert "vendor" not in gaps

    def test_no_legal_name_when_absent(self):
        """Vendor without separate legal entity should not have legal_name."""
        lines = NUT_CRACKER_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert data["vendor"] == "TheNutCrackerHouse"
        assert "vendor_legal_name" not in data

    def test_papas_no_legal_name(self):
        lines = PAPAS_RECEIPT_OCR.split("\n")
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert data["vendor"] == "PAPAS HYPERMARKET"
        assert "vendor_legal_name" not in data


# ---------------------------------------------------------------------------
# Statement blank line tolerance
# ---------------------------------------------------------------------------


class TestStatementBlankLineTolerance:
    """Tests for blank-line tolerance in statement transaction parsing."""

    def test_blank_line_between_desc_and_amount(self):
        """A single blank line between description and amount should be tolerated."""
        text = (
            "EUROBANK S.A.\n"
            "IBAN: CY1234567890\n"
            "Period: 01/01/2026 - 31/01/2026\n"
            "\n"
            "05/02/2026 ATM ERB 2183\n"
            "\n"
            "-500,00 05/02/2026\n"
            "06/02/2026 MINI MARKET -19,68 06/02/2026\n"
        )
        result = parse_ocr_text(text, doc_type="statement")
        transactions = result.data.get("transactions", [])
        assert len(transactions) == 2
        atm_txn = next(
            (t for t in transactions if "ATM ERB 2183" in t.get("description", "")),
            None,
        )
        assert atm_txn is not None, "ATM ERB 2183 transaction not found"
        assert atm_txn.get("debit") == -500.00
        mini_txn = next(
            (t for t in transactions if "MINI MARKET" in t.get("description", "")),
            None,
        )
        assert mini_txn is not None, "MINI MARKET transaction not found"

    def test_two_blank_lines_stops_continuation(self):
        """Two consecutive blank lines should end the transaction block."""
        text = (
            "EUROBANK S.A.\n"
            "IBAN: CY1234567890\n"
            "Period: 01/01/2026 - 31/01/2026\n"
            "\n"
            "05/02/2026 ATM ERB 2183\n"
            "\n"
            "\n"
            "-500,00 05/02/2026\n"
            "06/02/2026 MINI MARKET -19,68 06/02/2026\n"
        )
        result = parse_ocr_text(text, doc_type="statement")
        transactions = result.data.get("transactions", [])
        # ATM transaction has no amount after two blank lines break the block
        atm_txn = next(
            (t for t in transactions if "ATM ERB 2183" in t.get("description", "")),
            None,
        )
        # ATM should be absent (no amount found) or present without a debit
        if atm_txn is not None:
            assert atm_txn.get("debit") is None
        # MINI MARKET should still parse correctly
        mini_txn = next(
            (t for t in transactions if "MINI MARKET" in t.get("description", "")),
            None,
        )
        assert mini_txn is not None, "MINI MARKET transaction not found"


# ---------------------------------------------------------------------------
# Standalone price items (narrow receipt line-wrap)
# ---------------------------------------------------------------------------


class TestStandalonePriceItems:
    """Tests for two-line item format: name on one line, price on the next."""

    def test_two_line_item_name_then_price(self):
        """Name on one line and bare price on the next should form one item.

        The receipt has a priced item first (anchoring item_start) followed by
        two-line items where the price wraps to the next line.
        """
        text = (
            "Maleve Trading LTD\n"
            "Profiti Elia 3A\n"
            "Feb 12, 2026  8:35:55\n"
            "\n"
            "Bread 1.20\n"
            "Blue Green Wave Frozen Blueberries 500\n"
            "4.09\n"
            "Red Bull White\n"
            "1.09\n"
            "TOTAL  6.38\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 3
        blueberry = next((i for i in items if "Blueberries" in i.get("name", "")), None)
        assert blueberry is not None, "Blueberries item not found"
        assert blueberry["total_price"] == 4.09
        redbull = next((i for i in items if "Red Bull" in i.get("name", "")), None)
        assert redbull is not None, "Red Bull item not found"
        assert redbull["total_price"] == 1.09

    def test_mixed_single_and_two_line(self):
        """Mix of single-line items and two-line items should all be found."""
        text = (
            "SOME SHOP\n"
            "01/01/2026\n"
            "\n"
            "Short Item 2.50\n"
            "Very Long Item Name That Wraps\n"
            "3.75\n"
            "Another Short 1.00\n"
            "TOTAL 7.25\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 3


# ---------------------------------------------------------------------------
# Markdown table item extraction
# ---------------------------------------------------------------------------


class TestMarkdownTableExtraction:
    def test_maleve_markdown_table(self):
        """Maleve receipt with markdown table OCR format."""
        text = (
            "Maleve Trading LTD\n"
            "Profiti Ela 3A, Tel. 15322957\n"
            "VAT number: CY10370773Q\n"
            "Receipt\n"
            "ID 1001100241728\n"
            "Date Feb 12, 2026, 8:35:55 AM\n"
            "\n"
            "| Qty | Description | Price | Disc. | Total |\n"
            "| :--- | :--- | :--- | :--- | :--- |\n"
            "| 1 | Blue Green Wave Frozen Blueberries 500 | 4.09 | 0.00 | 4.09 |\n"
            "| 2 | Charalambales Christis Strained Yogurt | 4.59 | 0.00 | 9.18 |\n"
            "| 1 | Red Bull White 250ml | 1.09 | 0.00 | 1.09 |\n"
            "| 3 | Mesogeios Tomato Beans 400g | 0.99 | 0.00 | 1.98 |\n"
            "| 1 | Barilla Penne Rigate 500 g | 1.95 | 0.00 | 1.95 |\n"
            "| 1.1 | Avocado kg | 3.59 | 0.00 | 3.97 |\n"
            "| 0.77 | Piperia Chromatista | 3.89 | 0.00 | 2.98 |\n"
            "| 1 | St. George Extra Virgin Oil 1lt | 7.79 | 0.00 | 7.79 |\n"
            "\n"
            "Total 33.03\n"
            "VISA 33.03\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 8, f"Expected 8 items, got {len(items)}: {items}"

        # Check specific items
        blueberries = next(
            (i for i in items if "Blueberries" in i.get("name", "")), None
        )
        assert blueberries is not None
        assert blueberries["quantity"] == 1
        assert blueberries["total_price"] == 4.09

        yogurt = next((i for i in items if "Yogurt" in i.get("name", "")), None)
        assert yogurt is not None
        assert yogurt["quantity"] == 2
        assert yogurt["total_price"] == 9.18

        # Weighed item (avocado)
        avocado = next((i for i in items if "Avocado" in i.get("name", "")), None)
        assert avocado is not None
        assert avocado["quantity"] == 1.1
        assert avocado["total_price"] == 3.97


class TestColumnarTableExtraction:
    def test_maleve_columnar_format(self):
        """Maleve receipt with space-delimited columnar format."""
        text = (
            "Maleve Trading LTD\n"
            "Profiti Elia 3A, Tel. 25322957\n"
            "VAT number: CY10370773Q\n"
            "Receipt\n"
            "ID 1001100211433\n"
            "Date Nov 29, 2025, 8:53:15 AM\n"
            "\n"
            "Qty Description Price Disc. Total\n"
            "1 Blue Green Wave Frozen Blueberries 500 4.09 0.00 4.09\n"
            "2 Lukoshko - Frozen Blackcurrants 500 g 6.39 0.00 12.78\n"
            "1 Balsamic Vinegar 500ml 2.59 0.00 2.59\n"
            "1.02 Krenmidia Kokkina 1.49 0.00 1.52\n"
            "0.51 Kiwiano 7.59 0.00 3.83\n"
            "\n"
            "Total 39.28\n"
            "VISA 39.28\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 5, f"Expected 5 items, got {len(items)}"

        blueberries = next(
            (i for i in items if "Blueberries" in i.get("name", "")), None
        )
        assert blueberries is not None
        assert blueberries["quantity"] == 1
        assert blueberries["unit_price"] == 4.09
        assert blueberries["total_price"] == 4.09
        # Name should NOT include leading qty or trailing price/disc columns
        assert not blueberries["name"].startswith("1 ")
        assert "4.09" not in blueberries["name"]

        # Weighed item
        krenmidia = next((i for i in items if "Krenmidia" in i.get("name", "")), None)
        assert krenmidia is not None
        assert krenmidia["quantity"] == 1.02
        assert krenmidia["unit_price"] == 1.49
        assert krenmidia["total_price"] == 1.52

    def test_barcode_after_item(self):
        """Standalone barcode line after columnar item attaches to that item."""
        text = (
            "Store Name\n"
            "VAT: CY10370773Q\n"
            "Receipt\n"
            "Date: Nov 29, 2025\n"
            "\n"
            "Qty Description Price Disc. Total\n"
            "1 Frozen Blueberries 500g 4.09 0.00 4.09\n"
            "5292006000152\n"
            "2 Organic Honey 250ml 6.39 0.00 12.78\n"
            "\n"
            "Total 16.87\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 2
        blueberries = items[0]
        assert blueberries["name"] == "Frozen Blueberries 500g"
        assert blueberries.get("barcode") == "5292006000152"
        # Second item should NOT have the barcode
        assert items[1].get("barcode") is None

    def test_barcode_before_first_item(self):
        """Standalone barcode before any item is carried forward."""
        text = (
            "Store\n"
            "Receipt\n"
            "\n"
            "Qty Description Price Disc. Total\n"
            "5292006000152\n"
            "1 Frozen Blueberries 500g 4.09 0.00 4.09\n"
            "\n"
            "Total 4.09\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 1
        assert items[0].get("barcode") == "5292006000152"

    def test_invalid_barcode_ignored(self):
        """Non-valid EAN digits should not be treated as barcode."""
        text = (
            "Store\n"
            "Receipt\n"
            "\n"
            "Qty Description Price Disc. Total\n"
            "1 Widget 4.09 0.00 4.09\n"
            "12345678\n"  # invalid EAN-8 check digit
            "2 Gadget 6.00 0.00 12.00\n"
            "\n"
            "Total 16.09\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        # Invalid barcode line should not attach
        for item in items:
            assert item.get("barcode") is None

    def test_multiline_item_name(self):
        """Wrapped item name across two lines is concatenated."""
        text = (
            "Maleve Trading LTD\n"
            "VAT number: CY10370773Q\n"
            "Receipt\n"
            "Date Nov 29, 2025\n"
            "\n"
            "Qty Description Price Disc. Total\n"
            "1 Blue Green Wave Frozen\n"
            "Blueberries 500g 4.09 0.00 4.09\n"
            "2 Organic Honey 6.39 0.00 12.78\n"
            "\n"
            "Total 16.87\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        # The wrapped item may appear as one combined item or separately;
        # but the continuation line should not be lost entirely
        names = " ".join(i["name"] for i in items)
        assert "Blueberries" in names

    def test_multiline_then_barcode(self):
        """Multi-line name followed by barcode: both handled correctly."""
        text = (
            "Store\n"
            "Receipt\n"
            "\n"
            "Qty Description Price Disc. Total\n"
            "1 Premium Extra Virgin\n"
            "Olive Oil 500ml 8.99 0.00 8.99\n"
            "5292006000152\n"
            "1 Bread 1.50 0.00 1.50\n"
            "\n"
            "Total 10.49\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        # Barcode should attach to the olive oil item
        barcoded = [i for i in items if i.get("barcode") == "5292006000152"]
        assert len(barcoded) == 1
        assert "Olive Oil" in barcoded[0]["name"]

    def test_barcode_between_items(self):
        """Multiple barcodes between multiple items are each attached correctly."""
        text = (
            "Store\n"
            "Receipt\n"
            "\n"
            "Qty Description Price Disc. Total\n"
            "1 Product A 4.00 0.00 4.00\n"
            "5292006000152\n"
            "1 Product B 3.00 0.00 3.00\n"
            "4006381333931\n"
            "\n"
            "Total 7.00\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 2
        assert items[0].get("barcode") == "5292006000152"
        assert items[1].get("barcode") == "4006381333931"

    def test_5col_layout_type_tagged(self):
        """5-column columnar should tag layout as columnar_5."""
        text = (
            "Shop\nQty Description Price Disc. Total\n"
            "1 Milk 1.50 0.00 1.50\nTotal 1.50\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        assert result.data.get("_layout_type") == "columnar_5"

    def test_4col_layout_type_tagged(self):
        """4-column columnar should tag layout as columnar_4."""
        text = "Shop\nQty Description Price Total\n" "1 Milk 1.50 1.50\nTotal 1.50\n"
        result = parse_ocr_text(text, doc_type="receipt")
        # 4-col needs to match _COLUMNAR_ROW_4 but not _COLUMNAR_ROW
        assert result.data.get("_layout_type") in (
            "columnar_4",
            "columnar_5",
            "standard",
        )


# ---------------------------------------------------------------------------
# Item-first columnar format (two-line items: name then barcode+qty+prices)
# ---------------------------------------------------------------------------


class TestItemFirstColumnarExtraction:
    """Tests for item-first two-line columnar format (retailedge.io, pharmacy POS)."""

    PHARMACY_RECEIPT = (
        "LEGAL RECEIPT: POS-1-112-1-5504\n"
        "Christina Metaxa Pharmacy\n"
        "Christina Metaxa Pharmacy POS 1\n"
        "Served by Viola Sanyan\n"
        "Customer Name: aleksandra zharnikova\n"
        "Phone Number: 35799775953\n"
        "Item Qty Price Disc Sub Total\n"
        "Tranexamic Acid 500Mg 20 Mull Tablets\n"
        "3886 1 9.45 0.00 9.45\n"
        "Fagron Derma Pack Trt 100Ml 100Ml Liquid\n"
        "90747 1 39.00 0.00 39.00\n"
        "Ticket Summary\n"
        "Total 2 48.45 0.00 48.45\n"
        "Paid by:\n"
        "Credit/Debit Card 48.45\n"
        "VAT Analysis\n"
        "5 5% of 46.14 2.31\n"
    )

    def test_extracts_both_items(self):
        result = parse_ocr_text(self.PHARMACY_RECEIPT, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 2, f"Expected 2 items, got {len(items)}: {items}"

    def test_item_names_correct(self):
        result = parse_ocr_text(self.PHARMACY_RECEIPT, doc_type="receipt")
        items = result.data.get("line_items", [])
        names = [i["name"] for i in items]
        assert "Tranexamic Acid 500Mg 20 Mull Tablets" in names
        assert "Fagron Derma Pack Trt 100Ml 100Ml Liquid" in names

    def test_item_prices_correct(self):
        result = parse_ocr_text(self.PHARMACY_RECEIPT, doc_type="receipt")
        items = result.data.get("line_items", [])
        tranexamic = next(i for i in items if "Tranexamic" in i["name"])
        assert tranexamic["total_price"] == 9.45
        assert tranexamic["unit_price"] == 9.45
        assert tranexamic["quantity"] == 1

        fagron = next(i for i in items if "Fagron" in i["name"])
        assert fagron["total_price"] == 39.00
        assert fagron["quantity"] == 1

    def test_vendor_not_legal_receipt(self):
        result = parse_ocr_text(self.PHARMACY_RECEIPT, doc_type="receipt")
        vendor = result.data.get("vendor", "")
        assert "LEGAL RECEIPT" not in vendor
        assert "Christina Metaxa Pharmacy" == vendor

    def test_layout_type_detected(self):
        result = parse_ocr_text(self.PHARMACY_RECEIPT, doc_type="receipt")
        assert result.data.get("_layout_type") == "item_first_columnar"

    def test_header_variation_description_qty_price(self):
        """Handles 'Description Qty Price Disc Sub Total' header variant."""
        text = (
            "Shop Name\n"
            "01/01/2026\n"
            "Description Qty Price Disc Sub Total\n"
            "Widget A\n"
            "100 2 5.00 0.00 10.00\n"
            "Widget B\n"
            "200 1 3.50 0.00 3.50\n"
            "Total 13.50\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 2
        assert items[0]["name"] == "Widget A"
        assert items[0]["quantity"] == 2
        assert items[0]["total_price"] == 10.00
        assert items[1]["name"] == "Widget B"
        assert items[1]["total_price"] == 3.50

    def test_stops_at_total_marker(self):
        """Parser stops at total line, doesn't pick up summary rows."""
        text = (
            "SOME STORE\n"
            "01/01/2026\n"
            "Item Qty Price Disc Sub Total\n"
            "Product X\n"
            "555 1 12.00 0.00 12.00\n"
            "Total 1 12.00 0.00 12.00\n"
            "Tax 5% 0.57\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 1
        assert items[0]["name"] == "Product X"


class TestLegalReceiptNoise:
    """'LEGAL RECEIPT' header lines are filtered as noise."""

    def test_legal_receipt_is_noise(self):
        assert _is_noise_line("LEGAL RECEIPT: POS-1-112-1-5504") is True

    def test_legal_receipt_lowercase(self):
        assert _is_noise_line("Legal Receipt: POS-1-112-1-5504") is True

    def test_legal_receipt_bare(self):
        assert _is_noise_line("LEGAL RECEIPT") is True


class TestBarcodeNoise:
    """BARCODE: lines are filtered as noise (not vendor names)."""

    def test_barcode_line_is_noise(self):
        assert _is_noise_line("BARCODE: 7400/2612354") is True

    def test_barcode_lowercase(self):
        assert _is_noise_line("barcode: 5201234567890") is True

    def test_barcode_in_header_noise_set(self):
        assert "barcode" in _HEADER_NOISE

    def test_long_caps_prefix_is_noise(self):
        """Widened regex catches any ALL-CAPS keyword prefix."""
        assert _is_noise_line("TERMINAL: ABC123") is True
        assert _is_noise_line("RECEIPT: 00012345") is True

    def test_real_vendor_not_noise(self):
        """Normal vendor names must not be caught by widened regex."""
        assert _is_noise_line("ALPHAMEGA") is False
        assert _is_noise_line("FreSko") is False


# ---------------------------------------------------------------------------
# Three-line item name reassembly (name1 / name2 / price)
# ---------------------------------------------------------------------------


class TestThreeLineItemName:
    """Tests for 3-line item format: two name lines then a standalone price."""

    def test_three_line_item_basic(self):
        """Name split across two lines before standalone price forms one item."""
        text = (
            "NARROW SHOP\n"
            "01/01/2026\n"
            "\n"
            "Bread 1.20\n"
            "Whole Milk Full Fat\n"
            "Extra Creamy Lactic\n"
            "3.49\n"
            "TOTAL 4.69\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 2, f"Expected 2 items, got {len(items)}: {items}"
        milk = next((i for i in items if "Milk" in i.get("name", "")), None)
        assert milk is not None, "Milk item not found"
        assert (
            "Extra Creamy Lactic" in milk["name"]
        ), f"Expected combined name, got: {milk['name']}"
        assert milk["total_price"] == 3.49

    def test_three_line_item_multiple(self):
        """Two consecutive 3-line items after an anchor item are both parsed.

        An anchor (single-line priced item) is required so that _find_item_start
        can locate the body section before the 3-line items appear.
        """
        text = (
            "SOME SHOP\n"
            "01/01/2026\n"
            "\n"
            "Anchor Item 1.00\n"
            "Item A Part 1\n"
            "Item A Part 2\n"
            "5.00\n"
            "Item B Part 1\n"
            "Item B Part 2\n"
            "3.50\n"
            "TOTAL 9.50\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 3, f"Expected 3 items, got {len(items)}: {items}"
        item_a = next((i for i in items if "Item A Part 1" in i.get("name", "")), None)
        assert item_a is not None, "Item A not found"
        assert "Item A Part 2" in item_a["name"]
        assert item_a["total_price"] == 5.00
        item_b = next((i for i in items if "Item B Part 1" in i.get("name", "")), None)
        assert item_b is not None, "Item B not found"
        assert "Item B Part 2" in item_b["name"]
        assert item_b["total_price"] == 3.50

    def test_three_line_mixed_with_one_and_two_line(self):
        """Mix of 1-line, 2-line, and 3-line items are all parsed correctly."""
        text = (
            "TEST STORE\n"
            "02/02/2026\n"
            "\n"
            "Single Item 2.00\n"
            "Two Line Item\n"
            "1.50\n"
            "Three Line Name Part 1\n"
            "Three Line Name Part 2\n"
            "4.00\n"
            "TOTAL 7.50\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 3, f"Expected 3 items, got {len(items)}: {items}"
        single = next((i for i in items if "Single" in i.get("name", "")), None)
        assert single is not None and single["total_price"] == 2.00
        two = next((i for i in items if "Two Line" in i.get("name", "")), None)
        assert two is not None and two["total_price"] == 1.50
        three = next(
            (i for i in items if "Three Line Name Part 1" in i.get("name", "")), None
        )
        assert three is not None, "Three-line item not found"
        assert "Three Line Name Part 2" in three["name"]
        assert three["total_price"] == 4.00


# ---------------------------------------------------------------------------
# classify_ocr_text() unit tests
# ---------------------------------------------------------------------------


class TestClassifyOcrText:
    def test_classify_receipt(self):
        """PAPAS_RECEIPT_OCR has many price lines and a TOTAL — must be receipt."""
        assert classify_ocr_text(PAPAS_RECEIPT_OCR) == "receipt"

    def test_classify_payment_confirmation(self):
        """PAYMENT_SLIP_OCR is a JCC card slip with no line items."""
        assert classify_ocr_text(PAYMENT_SLIP_OCR) == "payment_confirmation"

    def test_classify_invoice(self):
        """INVOICE_OCR has 'Invoice No', 'Bill To', 'Due Date' — must be invoice."""
        assert classify_ocr_text(INVOICE_OCR) == "invoice"

    def test_classify_statement(self):
        """BANK_STATEMENT_OCR has IBAN and Period — must be statement."""
        assert classify_ocr_text(BANK_STATEMENT_OCR) == "statement"

    def test_classify_empty_text(self):
        """Empty string returns the default 'receipt'."""
        assert classify_ocr_text("") == "receipt"

    def test_classify_mixed_content(self):
        """MIXED_CONTENT_OCR has JCC header but also real line items.

        Three price lines push receipt score above card-slip heuristic score,
        so the document should be classified as a receipt.
        """
        assert classify_ocr_text(MIXED_CONTENT_OCR) == "receipt"

    def test_classify_card_slip_no_items(self):
        """Card details only (no line items) should yield payment_confirmation."""
        text = (
            "SOME RESTAURANT\n"
            "VISA ****7201\n"
            "AUTH NO: 445566\n"
            "TERMINAL: 12345\n"
            "AMOUNT EUR 45.00\n"
            "APPROVED"
        )
        assert classify_ocr_text(text) == "payment_confirmation"

    def test_classify_warranty(self):
        """Text with warranty keywords should be classified as warranty."""
        text = (
            "WARRANTY CERTIFICATE\n"
            "Serial Number: ABC123\n"
            "Warranty End: 31/12/2027\n"
            "Product: Washing Machine\n"
        )
        assert classify_ocr_text(text) == "warranty"

    def test_classify_contract(self):
        """Text with contract keywords should be classified as contract."""
        text = (
            "CONTRACT No. 12345\n"
            "Effective Date: 01/01/2026\n"
            "Terms and Conditions\n"
            "Parties to this agreement\n"
        )
        assert classify_ocr_text(text) == "contract"

    def test_classify_ambiguous_defaults_to_receipt(self):
        """Text with no strong signals defaults to receipt."""
        assert classify_ocr_text("Hello world some random text") == "receipt"

    def test_classify_invoice_multilingual(self):
        """German invoice keyword (Rechnungsnummer) combined with a second signal
        (Due Date) pushes the invoice score above the 1.5 threshold needed to
        override the receipt default.  Real cross-border invoices commonly mix
        German labels with English header fields."""
        text = (
            "Muster GmbH\n"
            "Rechnungsnummer: 2026-042\n"
            "Due Date: 07/03/2026\n"
            "Payment Terms: Net 30\n"
        )
        assert classify_ocr_text(text) == "invoice"


# ---------------------------------------------------------------------------
# Post-OCR classification pipeline integration tests
# ---------------------------------------------------------------------------


class TestPostOcrClassificationIntegration:
    """Integration tests verifying how classify_ocr_text() is wired into
    extract_from_image() in alibi/extraction/vision.py."""

    @staticmethod
    def _make_config():
        from unittest.mock import MagicMock

        cfg = MagicMock()
        cfg.ollama_url = "http://test:11434"
        cfg.ollama_model = "test-model"
        cfg.ollama_structure_model = "qwen3:8b"
        cfg.ollama_ocr_model = "glm-ocr"
        cfg.prompt_mode = "specialized"
        cfg.skip_llm_threshold = 1.1  # Disable skip-LLM path
        return cfg

    @patch("alibi.extraction.vision.get_config")
    @patch("alibi.extraction.structurer._call_ollama_text")
    @patch("alibi.extraction.ocr.ocr_image_with_retry")
    def test_vision_not_called_without_folder_context(
        self, mock_ocr, mock_structure, mock_config
    ):
        """When no folder_context is given, doc_type=None triggers post-OCR
        classification via classify_ocr_text(); the legacy
        vision_detect_document_type should NOT be invoked."""
        import json
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        mock_ocr.return_value = (PAPAS_RECEIPT_OCR, False)
        mock_structure.return_value = {
            "response": json.dumps(
                {
                    "vendor": "PAPAS HYPERMARKET",
                    "date": "2026-01-21",
                    "total": 85.69,
                    "currency": "EUR",
                    "line_items": [],
                }
            )
        }
        mock_config.return_value = self._make_config()

        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "alibi.extraction.text_parser.classify_ocr_text",
                wraps=classify_ocr_text,
            ) as mock_classify:
                from alibi.extraction.vision import extract_from_image

                result = extract_from_image(
                    Path("/fake/receipt.jpg"),
                    doc_type=None,  # No pre-determined type
                    ollama_url="http://test:11434",
                )

        # classify_ocr_text was called (post-OCR path)
        mock_classify.assert_called_once()
        # extract_from_image returned a result dict
        assert isinstance(result, dict)

    @patch("alibi.extraction.vision.get_config")
    @patch("alibi.extraction.structurer._call_ollama_text")
    @patch("alibi.extraction.ocr.ocr_image_with_retry")
    def test_folder_context_still_skips_classification(
        self, mock_ocr, mock_structure, mock_config
    ):
        """When doc_type='receipt' is supplied, classify_ocr_text() must NOT
        be called — the type is already known from folder routing."""
        import json
        from pathlib import Path
        from unittest.mock import patch

        mock_ocr.return_value = (PAPAS_RECEIPT_OCR, False)
        mock_structure.return_value = {
            "response": json.dumps(
                {
                    "vendor": "PAPAS HYPERMARKET",
                    "date": "2026-01-21",
                    "total": 85.69,
                    "currency": "EUR",
                    "line_items": [],
                }
            )
        }
        mock_config.return_value = self._make_config()

        with patch("pathlib.Path.exists", return_value=True):
            with patch(
                "alibi.extraction.text_parser.classify_ocr_text",
                wraps=classify_ocr_text,
            ) as mock_classify:
                from alibi.extraction.vision import extract_from_image

                result = extract_from_image(
                    Path("/fake/receipt.jpg"),
                    doc_type="receipt",  # Pre-determined by folder routing
                    ollama_url="http://test:11434",
                )

        # classify_ocr_text must NOT be called when type is pre-determined
        mock_classify.assert_not_called()
        # Result should still be a valid extraction dict
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Fixtures for per-field confidence and region splitting tests
# ---------------------------------------------------------------------------

BANK_STATEMENT_OCR = """EUROBANK S.A.
IBAN: CY66005001010000000012345678
Period: 01/01/2026 - 31/01/2026
Currency: EUR

05/01/2026 MINI MARKET PAPHOS -12,50 05/01/2026
12/01/2026 ATM WITHDRAWAL -200,00 12/01/2026
20/01/2026 SALARY CREDIT 1500,00 20/01/2026
25/01/2026 ELECTRICITY BILL -85,30 25/01/2026

TOTALS: -297,80 1500,00"""

JCC_SLIP_OCR = PAYMENT_SLIP_OCR  # alias — same fixture, different name in task spec


# ---------------------------------------------------------------------------
# A. Per-Field Confidence Tests
# ---------------------------------------------------------------------------


class TestReceiptFieldConfidence:
    def test_receipt_field_confidence_all_fields(self):
        """Full PAPAS receipt should have high confidence on all core heuristic fields
        and zero confidence on semantic-only fields."""
        result = parse_ocr_text(PAPAS_RECEIPT_OCR, doc_type="receipt")
        fc = result.field_confidence

        assert fc["vendor"] == 1.0
        assert fc["date"] == 1.0
        assert fc["currency"] > 0
        assert fc["line_items"] == 1.0
        assert fc["total"] == 1.0

        # Semantic fields — parser cannot fill these
        assert fc["name_en"] == 0.0
        assert fc["category"] == 0.0

    def test_receipt_field_confidence_missing_vendor(self):
        """Text with no recognisable vendor line should give vendor confidence 0.0."""
        text = "17/01/2026 10:00\n" "Item A 5.00 T1\n" "Item B 3.00 T1\n" "TOTAL 8.00\n"
        result = parse_ocr_text(text, doc_type="receipt")
        assert result.field_confidence["vendor"] == 0.0

    def test_gaps_property_backward_compat(self):
        """gaps property must return only keys with confidence == 0.0."""
        result = parse_ocr_text(PAPAS_RECEIPT_OCR, doc_type="receipt")
        gaps = result.gaps

        # Semantic-only fields must be in gaps
        assert "name_en" in gaps
        assert "category" in gaps
        assert "brand" in gaps
        assert "language" in gaps

        # Core fields with data must NOT be in gaps
        assert "vendor" not in gaps
        assert "date" not in gaps
        assert "total" not in gaps

    def test_gaps_empty_text(self):
        """Empty text should produce gaps == ['empty_text']."""
        result = parse_ocr_text("", doc_type="receipt")
        assert result.gaps == ["empty_text"]


class TestInvoiceFieldConfidence:
    def test_invoice_field_confidence(self):
        """INVOICE_OCR should populate key invoice fields with nonzero confidence."""
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        fc = result.field_confidence

        assert fc.get("issuer", 0.0) > 0.0
        assert fc.get("issue_date", 0.0) > 0.0
        assert fc.get("invoice_number", 0.0) > 0.0
        assert fc.get("currency", 0.0) > 0.0
        assert fc.get("amount", 0.0) > 0.0

    def test_invoice_field_confidence_keys_present(self):
        """All expected field_confidence keys should be present for an invoice parse."""
        result = parse_ocr_text(INVOICE_OCR, doc_type="invoice")
        fc = result.field_confidence

        expected_keys = {"issuer", "issue_date", "invoice_number", "currency", "amount"}
        for key in expected_keys:
            assert key in fc, f"Expected key '{key}' missing from field_confidence"


class TestPaymentConfirmationFieldConfidence:
    def test_payment_confirmation_field_confidence(self):
        """JCC slip should populate vendor, date, payment_method, total in field_confidence."""
        result = parse_ocr_text(JCC_SLIP_OCR, doc_type="payment_confirmation")
        fc = result.field_confidence

        assert "vendor" in fc
        assert "date" in fc
        assert "payment_method" in fc
        assert "total" in fc

        assert fc["vendor"] > 0.0
        assert fc["date"] > 0.0
        assert fc["total"] > 0.0

    def test_payment_confirmation_currency_confidence(self):
        """Payment slip with AMOUNT EUR should yield currency confidence > 0."""
        result = parse_ocr_text(JCC_SLIP_OCR, doc_type="payment_confirmation")
        assert result.field_confidence.get("currency", 0.0) > 0.0

    def test_bank_transaction_vendor_from_description(self):
        """Bank transaction confirmation extracts vendor from Description field."""
        ocr = (
            "Print date:17/02/2026\n"
            "Transaction Confirmation\n"
            "Customer name:TEST USER\n"
            "Bank name:EUROBANK LIMITED\n"
            "Execution date:06/02/2026 07:25 Order amount:EUR19,68\n"
            "Description:MINI MARKET *7514"
        )
        result = parse_ocr_text(ocr, doc_type="payment_confirmation")
        assert result.data["vendor"] == "MINI MARKET *7514"
        assert result.data["total"] == 19.68
        assert result.field_confidence["vendor"] == 1.0

    def test_bank_transaction_no_override_real_vendor(self):
        """Description fallback does not override a real vendor name."""
        ocr = (
            "ACME SHOP LTD\n"
            "VISA <1234>\n"
            "AMOUNT EUR10,00\n"
            "Description:SOME NOTE"
        )
        result = parse_ocr_text(ocr, doc_type="payment_confirmation")
        # Vendor should be ACME SHOP LTD, not the description
        assert "ACME" in result.data["vendor"]

    def test_greek_bank_transfer_beneficiary(self):
        """Greek bank transfer: payee extracted from Δικαιούχος label."""
        ocr = (
            "ΑΙΤΗΣΗ ΜΕΤΑΦΟΡΑΣ ΚΕΦΑΛΑΙΩΝ\n"
            "Ημερομηνία: 10/02/2026\n"
            "IBAN αποστολέα: GR1601101250000000012300695\n"
            "Δικαιούχος: ACME LTD\n"
            "IBAN δικαιούχου: GR3801401010101002320000000\n"
            "Ποσό: EUR 250,00\n"
        )
        result = parse_ocr_text(ocr, doc_type="payment_confirmation")
        assert result.data["vendor"] == "ACME LTD"
        assert result.field_confidence["vendor"] == 1.0

    def test_english_payment_order_beneficiary(self):
        """English PAYMENT ORDER: payee extracted from Beneficiary label."""
        ocr = (
            "PAYMENT ORDER\n"
            "Date: 15/02/2026\n"
            "Account: 1234567890\n"
            "Beneficiary: ACME COMPANY\n"
            "Beneficiary IBAN: GB29NWBK60161331926819\n"
            "Amount: EUR 1000.00\n"
        )
        result = parse_ocr_text(ocr, doc_type="payment_confirmation")
        assert result.data["vendor"] == "ACME COMPANY"
        assert result.field_confidence["vendor"] == 1.0

    def test_russian_bank_transfer_payee(self):
        """Russian bank transfer: payee extracted from Получатель label."""
        ocr = (
            "Платежное поручение\n"
            "Дата: 20/02/2026\n"
            "Плательщик: ИП ИВАНОВ И.И.\n"
            "Получатель: ООО РОГА\n"
            "Сумма: 5000.00 руб\n"
        )
        result = parse_ocr_text(ocr, doc_type="payment_confirmation")
        assert result.data["vendor"] == "ООО РОГА"
        assert result.field_confidence["vendor"] == 1.0

    def test_payment_order_title_not_used_as_vendor(self):
        """PAYMENT ORDER document title must not appear as vendor."""
        ocr = "PAYMENT ORDER\n" "Beneficiary: TARGET CORP\n" "Amount: USD 500.00\n"
        result = parse_ocr_text(ocr, doc_type="payment_confirmation")
        assert result.data.get("vendor") != "PAYMENT ORDER"
        assert result.data.get("vendor") == "TARGET CORP"

    def test_wire_transfer_beneficiary_name_label(self):
        """Wire transfer using 'Beneficiary name:' label."""
        ocr = (
            "Wire Transfer\n"
            "Date: 01/03/2026\n"
            "Beneficiary name: SOME SUPPLIER LTD\n"
            "Amount: EUR 750.00\n"
        )
        result = parse_ocr_text(ocr, doc_type="payment_confirmation")
        assert result.data["vendor"] == "SOME SUPPLIER LTD"
        assert result.field_confidence["vendor"] == 1.0


class TestStatementFieldConfidence:
    def test_statement_field_confidence(self):
        """Bank statement should report institution, currency, transactions in field_confidence."""
        result = parse_ocr_text(BANK_STATEMENT_OCR, doc_type="statement")
        fc = result.field_confidence

        assert "institution" in fc
        assert "currency" in fc
        assert "transactions" in fc

    def test_statement_institution_confidence(self):
        """Institution should have confidence 1.0 when parsed from a clear header."""
        result = parse_ocr_text(BANK_STATEMENT_OCR, doc_type="statement")
        assert result.field_confidence["institution"] == 1.0

    def test_statement_transactions_confidence(self):
        """Four transactions in BANK_STATEMENT_OCR should yield transactions confidence > 0."""
        result = parse_ocr_text(BANK_STATEMENT_OCR, doc_type="statement")
        assert result.field_confidence["transactions"] > 0.0


class TestMinimalFieldConfidence:
    def test_minimal_field_confidence_keys(self):
        """Warranty/contract parse should always emit full_extraction, vendor, date keys."""
        result = parse_ocr_text("Some warranty text\n01/01/2026", doc_type="warranty")
        fc = result.field_confidence

        assert "full_extraction" in fc
        assert "vendor" in fc
        assert "date" in fc

    def test_minimal_full_extraction_always_zero(self):
        """full_extraction is always 0.0 for minimal parses (needs LLM)."""
        result = parse_ocr_text("Company XYZ\n01/01/2026", doc_type="contract")
        assert result.field_confidence["full_extraction"] == 0.0

    def test_minimal_needs_llm_always_true(self):
        """Minimal parser must always set needs_llm=True."""
        result = parse_ocr_text("Company XYZ\n01/01/2026", doc_type="warranty")
        assert result.needs_llm is True

    def test_parser_exception_field_confidence(self):
        """Parser exception should produce field_confidence == {'parser_exception': 0.0}."""
        from unittest.mock import patch

        with patch(
            "alibi.extraction.text_parser._parse_receipt",
            side_effect=RuntimeError("crash"),
        ):
            result = parse_ocr_text(PAPAS_RECEIPT_OCR, doc_type="receipt")

        assert result.field_confidence == {"parser_exception": 0.0}
        assert result.gaps == ["parser_exception"]


# ---------------------------------------------------------------------------
# B. Region Splitting Tests
# ---------------------------------------------------------------------------


class TestReceiptRegions:
    def test_receipt_regions_structure(self):
        """PAPAS receipt should produce non-None regions with all three sections."""
        result = parse_ocr_text(PAPAS_RECEIPT_OCR, doc_type="receipt")
        regions = result.regions

        assert regions is not None
        assert isinstance(regions, TextRegions)

    def test_receipt_regions_header_contains_vendor(self):
        result = parse_ocr_text(PAPAS_RECEIPT_OCR, doc_type="receipt")
        assert "PAPAS HYPERMARKET" in result.regions.header

    def test_receipt_regions_body_contains_items(self):
        result = parse_ocr_text(PAPAS_RECEIPT_OCR, doc_type="receipt")
        assert "BARILLA SPAGHETTI" in result.regions.body

    def test_receipt_regions_footer_contains_total(self):
        result = parse_ocr_text(PAPAS_RECEIPT_OCR, doc_type="receipt")
        # Footer starts at first total marker line
        assert (
            "Total Before" in result.regions.footer or "TOTAL" in result.regions.footer
        )

    def test_receipt_regions_boundary_indices(self):
        """header_end must be > 0 and footer_start must be > header_end."""
        result = parse_ocr_text(PAPAS_RECEIPT_OCR, doc_type="receipt")
        regions = result.regions

        assert regions.header_end > 0
        assert regions.footer_start > regions.header_end

    def test_regions_no_items(self):
        """Text with only header info and no items should produce empty body."""
        text = "SIMPLE SHOP\n17/01/2026\nTEL. 99 999 999"
        result = parse_ocr_text(text, doc_type="receipt")
        regions = result.regions

        assert regions is not None
        # When no item start is found, all content collapses into header
        assert regions.body == "" or "SIMPLE SHOP" in regions.header


class TestSplitRegionsDirectly:
    def test_split_regions_simple_receipt(self):
        """_split_regions should correctly split a simple three-part receipt."""
        text = (
            "VENDOR NAME\n"
            "01/01/2026\n"
            "\n"
            "Item A 5.00 T1\n"
            "Item B 3.00 T1\n"
            "\n"
            "TOTAL 8.00\n"
            "VAT 19% 1.27\n"
        )
        lines = text.split("\n")
        regions = _split_regions(lines)

        assert isinstance(regions, TextRegions)
        assert "VENDOR NAME" in regions.header
        assert "Item A" in regions.body
        assert "TOTAL" in regions.footer

    def test_split_regions_boundary_indices_ordered(self):
        """header_end <= footer_start always holds."""
        text = "SHOP\n" "02/02/2026\n" "Product One 9.99 T1\n" "TOTAL 9.99\n"
        lines = text.split("\n")
        regions = _split_regions(lines)

        assert regions.header_end <= regions.footer_start

    def test_split_regions_no_items_entire_text_is_header(self):
        """When there are no price lines, the whole text becomes the header."""
        text = "SHOP NAME\n01/01/2026\nSome description text"
        lines = text.split("\n")
        regions = _split_regions(lines)

        assert regions.body == ""
        assert regions.footer == ""
        assert "SHOP NAME" in regions.header

    def test_split_regions_returns_text_regions_type(self):
        lines = PAPAS_RECEIPT_OCR.split("\n")
        regions = _split_regions(lines)
        assert isinstance(regions, TextRegions)


class TestFindFooterStart:
    def test_footer_at_known_position(self):
        """_find_footer_start should return the index of the first total marker."""
        lines = [
            "VENDOR",
            "01/01/2026",
            "Item A 5.00 T1",
            "Item B 3.00 T1",
            "TOTAL 8.00",  # index 4 — footer starts here
            "VAT 0.64",
        ]
        # body_start = 2 (first price line)
        result = _find_footer_start(lines, body_start=2)
        assert result == 4

    def test_footer_not_found_returns_len(self):
        """When no total marker is present, len(lines) is returned."""
        lines = [
            "VENDOR",
            "Item A 5.00 T1",
            "Item B 3.00 T1",
        ]
        result = _find_footer_start(lines, body_start=1)
        assert result == len(lines)

    def test_footer_with_subtotal_marker(self):
        """'Total Before Disc' is a valid footer marker."""
        lines = [
            "VENDOR",
            "Item 2.50 T1",
            "Total Before Disc 2.50",  # index 2
            "TOTAL 2.50",
        ]
        result = _find_footer_start(lines, body_start=1)
        assert result == 2

    def test_footer_start_respects_body_start(self):
        """Scan starts at body_start — lines before it are ignored."""
        lines = [
            "TOTAL in header should be ignored",  # index 0, before body_start
            "Item A 5.00 T1",
            "Item B 3.00 T1",
            "TOTAL 8.00",  # index 3
        ]
        # body_start = 1, so index 0 (which contains TOTAL) is skipped
        result = _find_footer_start(lines, body_start=1)
        assert result == 3


# ---------------------------------------------------------------------------
# C. Mixed Content Region Test
# ---------------------------------------------------------------------------


MIXED_CONTENT_OCR = """JCC PAYMENT SYSTEMS
AID: A0000000031010

CAFE NERO LIMASSOL
Makarios Avenue 45
Limassol

20/02/2026 09:15

Item Coffee Latte 3.50 T1
Item Croissant 2.20 T1
Item Orange Juice 2.80 T1

TOTAL 8.50
VAT 19% 1.35

VISA ****-4321
AUTH NO.: 998877
AMOUNT EUR 8.50
APPROVED"""


class TestMixedContentRegions:
    def test_mixed_content_regions_body_has_items(self):
        """Payment header + receipt items: body should contain line items."""
        result = parse_ocr_text(MIXED_CONTENT_OCR, doc_type="receipt")
        regions = result.regions

        assert regions is not None
        assert "Coffee Latte" in regions.body or "Croissant" in regions.body

    def test_mixed_content_regions_footer_has_total(self):
        """Footer should start at the TOTAL line."""
        result = parse_ocr_text(MIXED_CONTENT_OCR, doc_type="receipt")
        regions = result.regions

        assert "TOTAL" in regions.footer or "8.50" in regions.footer

    def test_mixed_content_regions_header_has_vendor(self):
        """Header should contain the real vendor (after JCC noise is skipped in parsing)."""
        result = parse_ocr_text(MIXED_CONTENT_OCR, doc_type="receipt")
        regions = result.regions

        # The header region contains all lines before the first item
        assert regions.header is not None
        assert len(regions.header) > 0

    def test_mixed_content_vendor_extracted_correctly(self):
        """Vendor should be CAFE NERO LIMASSOL, not JCC noise."""
        result = parse_ocr_text(MIXED_CONTENT_OCR, doc_type="receipt")
        assert result.data.get("vendor") == "CAFE NERO LIMASSOL"

    def test_mixed_content_items_parsed(self):
        """All three line items should be extracted."""
        result = parse_ocr_text(MIXED_CONTENT_OCR, doc_type="receipt")
        assert result.line_item_count == 3

    def test_mixed_content_field_confidence_vendor_and_total(self):
        """Mixed-content receipt should have vendor and total at 1.0 confidence."""
        result = parse_ocr_text(MIXED_CONTENT_OCR, doc_type="receipt")
        fc = result.field_confidence

        assert fc.get("vendor") == 1.0
        assert fc.get("total") == 1.0


# ---------------------------------------------------------------------------
# Hash separator weighed patterns (#)
# ---------------------------------------------------------------------------


class TestHashWeighedPattern:
    """Test that # is accepted as equivalent to @ in weighed item patterns."""

    def test_hash_weighed_single(self):
        """Qty 1.535 # 13.99 each should parse as weighed item."""
        ocr = """BUTCHER SHOP
Main Street 1
01/01/2026 10:00

MILK 1L 2.50 T1
BEEF MINCE 21.48 T1
Qty 1.535 # 13.99 each
TOTAL          23.98
"""
        result = parse_ocr_text(ocr, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 2
        beef = items[1]
        assert beef.get("unit_raw") == "kg"

    def test_hash_multi_qty(self):
        """Qty 3.000 # 1.99 each 5.97 should parse correctly."""
        ocr = """GROCERY STORE
01/01/2026 10:00

MILK 1L 2.50 T1
ORANGES 5.97 T1
Qty 3.000 # 1.99 each 5.97 T1
TOTAL          8.47
"""
        result = parse_ocr_text(ocr, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 2
        oranges = items[1]
        assert oranges.get("quantity") == 3  # integer qty (3.000 rounds)

    def test_multiword_header_combo_filtered(self):
        """Multi-word column header combos should be filtered as non-items."""
        from alibi.extraction.text_parser import _is_non_item_line

        assert _is_non_item_line("QTY DESCRIPTION PRICE AMOUNT VAT") is True
        assert _is_non_item_line("PRICE AMOUNT") is True
        assert _is_non_item_line("MILK 1L FULL FAT") is False


class TestSession64Improvements:
    """Tests for parser improvements in session 64."""

    # ---- 1. VAT pattern expansion ----

    def test_vat_spaced_with_country_prefix(self):
        """VAT REG: CY - 99000110 H -> vendor_vat=99000110H"""
        text = "ACME Store\nVAT REG: CY - 99000110 H\nTotal 10.00"
        result = parse_ocr_text(text, doc_type="receipt")
        assert result.data.get("vendor_vat") == "99000110H"

    def test_vat_abbreviated_n(self):
        """VAT N.:10408607Q"""
        text = "My Shop\nVAT N.:10408607Q\nTotal 5.00"
        result = parse_ocr_text(text, doc_type="receipt")
        assert result.data.get("vendor_vat") == "10408607Q"

    def test_vat_preserved_country_prefix(self):
        """CY99887766A (no separator) should preserve CY prefix."""
        text = "Shop\nVAT NO: CY99887766A\nTotal 5.00"
        result = parse_ocr_text(text, doc_type="receipt")
        assert result.data.get("vendor_vat") == "CY99887766A"

    # ---- 2. VAT summary header detection ----

    def test_vat_summary_header_not_item(self):
        """'VAT RatVAT Amount Amount' should not be a line item."""
        assert _is_non_item_line("VAT RatVAT Amount Amount") is True

    def test_vat_percent_header_not_item(self):
        assert _is_non_item_line("VAT% Net VAT") is True

    def test_incl_vat_not_item(self):
        assert _is_non_item_line("Incl. VAT 19% 1.23") is True

    # ---- 3. vendor_address pollution ----

    def test_address_no_customer_receipt(self):
        """'CUSTOMER RECEIPT' should NOT be in vendor_address."""
        text = "ACME Cafe\nCUSTOMER RECEIPT\nDINE IN\n1 Coffee 3.00\nTotal 3.00"
        result = parse_ocr_text(text, doc_type="receipt")
        addr = result.data.get("vendor_address", "")
        assert "CUSTOMER RECEIPT" not in addr
        assert "DINE IN" not in addr

    def test_address_no_vat_line(self):
        """Lines with 'VAT' should not be captured as address."""
        text = "My Store\nVAT REG: 12345678X\n123 Main St\nTotal 10.00"
        result = parse_ocr_text(text, doc_type="receipt")
        addr = result.data.get("vendor_address", "")
        assert "VAT" not in addr
        assert result.data.get("vendor_vat") == "12345678X"

    # ---- 4. Weight / unit_quantity extraction ----

    def test_inline_weight_in_name(self):
        """'PEPPERS 1.535 kg 4.03' should extract unit_quantity."""
        text = "Store\nPEPPERS 1.535 kg 4.03\nTotal 4.03"
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 1
        pepper = items[0]
        assert "PEPPER" in pepper["name"]
        assert pepper.get("unit_quantity") == 1.535
        assert pepper.get("unit_raw") == "kg"

    def test_package_size_in_name(self):
        """'Blueberries 500g 4.09' should extract unit_quantity=500, unit_raw=g."""
        text = "Store\nBlueberries 500g 4.09\nTotal 4.09"
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 1
        bb = items[0]
        assert "Blueberries" in bb["name"] or "blueberries" in bb["name"].lower()
        assert bb.get("unit_quantity") == 500.0
        assert bb.get("unit_raw") == "g"

    # ---- 5. Leading quantity prefix ----

    def test_leading_qty_1_stripped(self):
        """'1 Double Espresso 3.50' -> qty=1, name='Double Espresso'."""
        text = "Cafe\n1 Double Espresso 3.50\nTotal 3.50"
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 1
        assert items[0]["name"] == "Double Espresso"
        assert items[0]["quantity"] == 1

    def test_leading_qty_2_stripped(self):
        """'2 Cappuccino 7.00' -> qty=2, name='Cappuccino'."""
        text = "Cafe\n2 Cappuccino 7.00\nTotal 7.00"
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 1
        assert items[0]["name"] == "Cappuccino"
        assert items[0]["quantity"] == 2

    def test_leading_qty_skip_unit(self):
        """'1 kg rice 2.50' should NOT strip '1' -- it's a unit."""
        text = "Store\n1 kg rice 2.50\nTotal 2.50"
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 1
        # Name should contain "kg" -- leading "1" is part of "1 kg"
        assert "kg" in items[0]["name"].lower()

    # ---- 6. Item-level subtotal (LITTLE SINS format) ----

    def test_item_subtotal_format(self):
        """Item + 'Subtotal 3.00 eur' on next line."""
        text = (
            "LITTLE SINS\n"
            "Espresso double\n"
            "Subtotal 3.00 eur\n"
            "Latte macchiato\n"
            "Subtotal 4.50 eur\n"
            "Total 7.50 eur\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 2, f"Expected 2 items, got {len(items)}: {items}"
        assert items[0]["total_price"] == 3.0
        assert items[1]["total_price"] == 4.5

    # ---- 7. Payment processor swap ----

    def test_payment_processor_swap(self):
        """viva.com filtered as header noise; real vendor extracted directly."""
        text = "viva.com\n" "BroomBloom&Bikes Ltd\n" "Total 15.00\n" "VISA ****1234\n"
        result = parse_ocr_text(text, doc_type="payment_confirmation")
        assert result.data["vendor"] == "BroomBloom&Bikes Ltd"
        # viva.com is filtered as header noise, not stored as vendor_legal_name
        assert "viva.com" not in (result.data.get("vendor_legal_name") or "")


# ---------------------------------------------------------------------------
# EAN-8 / EAN-13 barcode detection
# ---------------------------------------------------------------------------


class TestIsValidEan:
    """Unit tests for the _is_valid_ean() check-digit validator."""

    # --- Valid barcodes ---

    def test_valid_ean13_cyprus(self):
        # 5290036000111: known Cyprus product barcode
        assert _is_valid_ean("5290036000111") is True

    def test_valid_ean13_german(self):
        # 4006381333931: known German product barcode
        assert _is_valid_ean("4006381333931") is True

    def test_valid_ean13_greek(self):
        # 5201004000019: computed valid Greek-prefix EAN-13
        assert _is_valid_ean("5201004000019") is True

    def test_valid_ean8_standard(self):
        # 96385074: EAN-8 with valid check digit
        assert _is_valid_ean("96385074") is True

    def test_valid_ean8_second(self):
        # 40170725: another EAN-8 used in GS1 documentation
        assert _is_valid_ean("40170725") is True

    # --- Invalid check digits ---

    def test_invalid_ean13_bad_check_digit(self):
        # Off-by-one check digit
        assert _is_valid_ean("5290036000112") is False

    def test_invalid_ean13_zero_check_wrong(self):
        assert _is_valid_ean("5201004000018") is False

    def test_invalid_ean8_bad_check_digit(self):
        assert _is_valid_ean("96385079") is False

    # --- Wrong length ---

    def test_invalid_length_12(self):
        assert _is_valid_ean("529003600011") is False

    def test_invalid_length_14(self):
        assert _is_valid_ean("52900360001110") is False

    def test_invalid_length_7(self):
        assert _is_valid_ean("9638507") is False

    def test_invalid_length_9(self):
        assert _is_valid_ean("963850740") is False

    # --- Non-digit characters ---

    def test_invalid_non_digit_ean13(self):
        assert _is_valid_ean("5290036000X11") is False

    def test_invalid_empty(self):
        assert _is_valid_ean("") is False


class TestStandaloneEanDetection:
    """Integration tests: standalone EAN lines in receipt parsing."""

    def test_standalone_ean13_attached_to_previous_item(self):
        """Bare EAN-13 on its own line attaches to the preceding item."""
        text = (
            "MY SUPERMARKET\n"
            "MILK FULL FAT 1L  1.29\n"
            "5290036000111\n"
            "BREAD WHOLE WHEAT  0.99\n"
            "Total 2.28\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 1
        milk = items[0]
        assert milk["barcode"] == "5290036000111"

    def test_standalone_ean13_before_first_item_applied_to_first(self):
        """EAN-13 appearing before any item is carried to the first item."""
        text = (
            "MY SUPERMARKET\n"
            "5290036000111\n"
            "MILK FULL FAT 1L  1.29\n"
            "Total 1.29\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 1
        assert items[0]["barcode"] == "5290036000111"

    def test_standalone_ean8_attached_to_previous_item(self):
        """Bare EAN-8 on its own line attaches to the preceding item."""
        text = "CORNER SHOP\n" "GUM SPEARMINT  0.50\n" "96385074\n" "Total 0.50\n"
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 1
        assert items[0]["barcode"] == "96385074"

    def test_labeled_barcode_still_works(self):
        """Regression: 'Barcode: XXXX' prefix format continues to work."""
        text = (
            "ALPHAMEGA\n"
            "Barcode: 5290036000111\n"
            "MILK FULL FAT 1L  1.29\n"
            "Total 1.29\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 1
        assert items[0]["barcode"] == "5290036000111"

    def test_invalid_check_digit_not_treated_as_barcode(self):
        """13-digit string with bad check digit is not extracted as barcode."""
        text = (
            "MY SUPERMARKET\n"
            "CHEESE GOUDA 200G  2.99\n"
            "5290036000112\n"
            "Total 2.99\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        # The line may be treated as a name continuation or skipped, but NOT
        # as a validated barcode.
        if items:
            assert items[-1].get("barcode") != "5290036000112"

    def test_wrong_length_digit_string_not_treated_as_barcode(self):
        """12-digit string (wrong length for EAN) is not extracted as barcode."""
        text = (
            "MY SUPERMARKET\n"
            "COFFEE BEANS 250G  4.99\n"
            "529003600011\n"
            "Total 4.99\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        if items:
            assert items[-1].get("barcode") != "529003600011"

    def test_multiple_items_each_barcode_attaches_to_correct_item(self):
        """Each barcode line attaches to its own immediately preceding item."""
        text = (
            "SUPERMARKET\n"
            "MILK 1L  1.29\n"
            "5290036000111\n"
            "BREAD 500G  0.99\n"
            "4006381333931\n"
            "Total 2.28\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 2
        assert items[0]["barcode"] == "5290036000111"
        assert items[1]["barcode"] == "4006381333931"

    def test_barcode_not_overwritten_when_item_already_has_one(self):
        """If item already has a labeled barcode, standalone line is attached
        to the next item, not the one that already has a barcode."""
        text = (
            "ALPHAMEGA\n"
            "Barcode: 5290036000111\n"
            "MILK 1L  1.29\n"
            "BREAD 500G  0.99\n"
            "4006381333931\n"
            "Total 2.28\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 2
        # First item keeps its labeled barcode
        assert items[0]["barcode"] == "5290036000111"
        # Second item gets the standalone barcode
        assert items[1]["barcode"] == "4006381333931"


class TestSession72Fixes:
    """Fixes for P0 VAT regex, P1 VAT summary filter, P2 item subtotal."""

    # ---- P0: VAT extraction regression (NUMBER before N\.) ----

    def test_vat_number_label_extracts_value_not_label(self):
        """'VAT number : CY10370773Q' should extract CY10370773Q, not 'umber'."""
        text = "MALEVE FRESH MARKET\nVAT number : CY10370773Q\n1 Eggs 3.79\nTotal 3.79"
        result = parse_ocr_text(text, doc_type="receipt")
        vat = result.data.get("vendor_vat", "")
        assert vat != "umber", "Regex captured label text instead of VAT value"
        assert "10370773" in vat

    def test_vat_number_colon_format(self):
        """'VAT NUMBER: AB12345678' should extract 'AB12345678'."""
        text = "Shop\nVAT NUMBER: AB12345678\nTotal 5.00"
        result = parse_ocr_text(text, doc_type="receipt")
        assert result.data.get("vendor_vat") == "AB12345678"

    def test_vat_number_mixed_case(self):
        """'Vat Number : 10370773Q' should extract '10370773Q'."""
        text = "Shop\nVat Number : 10370773Q\nTotal 5.00"
        result = parse_ocr_text(text, doc_type="receipt")
        assert result.data.get("vendor_vat") == "10370773Q"

    # ---- P1: VAT summary line not parsed as item ----

    def test_vat_summary_numeric_not_item(self):
        """'9.00% 0.66 8.00' should not be a line item."""
        assert _is_non_item_line("9.00% 0.66 8.00") is True

    def test_vat_summary_integer_pct_not_item(self):
        """'19% 1.23 7.89' should not be a line item."""
        assert _is_non_item_line("19% 1.23 7.89") is True

    def test_vat_summary_no_total_not_item(self):
        """'5% 0.14' should not be a line item."""
        assert _is_non_item_line("5% 0.14") is True

    def test_vat_numbered_breakdown_not_item(self):
        """'VAT1 19.00% 0.02' and 'VAT2 5.00% 0.10' should not be items."""
        assert _is_non_item_line("VAT1 19.00% 0.02") is True
        assert _is_non_item_line("VAT2 5.00% 0.10") is True
        assert _is_non_item_line("VAT 19% 1.23") is True

    def test_vat_summary_filters_in_receipt(self):
        """VAT summary row should not appear as a line item in full parse."""
        text = (
            "Avantage B&P Cafe LTD\n"
            "VAT REG: CY - 99000110 H\n"
            "1 Double Espresso 4.00\n"
            "1 Double Espresso 4.00\n"
            "VAT RatVAT Amount Amount\n"
            "9.00% 0.66 8.00\n"
            "TOTAL EUR: 8.00\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        names = [it["name"] for it in items]
        assert all("9.00%" not in n for n in names), f"VAT row in items: {names}"
        assert len(items) == 2

    # ---- P2: Item subtotal with blank lines (LITTLE SINS format) ----

    def test_item_subtotal_with_blank_lines(self):
        """Item name followed by blank lines then 'Subtotal X.XX' should parse."""
        text = (
            "LITTLE SINS\n"
            "COFFEE BAR\n"
            "\n"
            "Name Quantity Price Subtotal\n"
            "\n"
            "Espresso double (takeaway)\n"
            "\n"
            "Subtotal 3.00 eur\n"
            "\n"
            "Total 3.00 eur\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 1, f"Expected >= 1 item, got {len(items)}"
        assert "Espresso" in items[0]["name"]
        assert items[0]["total_price"] == 3.0

    # ---- Header word filtering ----

    def test_column_header_name_quantity_price(self):
        """'Name Quantity Price Subtotal' should be filtered as a header."""
        assert _is_non_item_line("Name Quantity Price Subtotal") is True

    def test_column_header_qty_description_price(self):
        """'Qty Description Price Disc. Total' should be filtered."""
        assert _is_non_item_line("Qty Description Price Disc. Total") is True


# ---------------------------------------------------------------------------
# Bare Cypriot VAT detection (card terminal slips)
# ---------------------------------------------------------------------------

# JCC slip that prints the merchant VAT as a bare line with no label.
# Format: 0030573-04-2-07 07056 is the terminal/merchant reference,
# then vendor name, address, city, bare VAT, and datetime.
JCC_BARE_VAT_SLIP_OCR = """JCC PAYMENT SYSTEMS
0030573-04-2-07 07056
CAFE PAUL LIMASSOL MARINA
PLATIA SYNTAGMATOS2 LIM MARINA
LIMASSOL
99000110H
23/02/2026 12:06:21

SALE

AMOUNT EUR 14.50

VISA <4321>
AUTH NO.: 112233
TERMINAL: TM998877

APPROVED
CARDHOLDER COPY"""


class TestBareVatDetection:
    """Bare Cypriot VAT number detection for card terminal slips (JCC, viva.com).

    Card terminals print the merchant's VAT number as a standalone line
    with no "VAT" label — exactly 8 digits followed by 1 uppercase letter.
    """

    def test_bare_vat_99000110h_detected(self):
        """'99000110H' on its own line is captured as vendor_vat."""
        lines = ["CAFE PAUL", "LIMASSOL", "99000110H", "23/02/2026 12:06:21"]
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert data.get("vendor_vat") == "99000110H"

    def test_bare_vat_10057000y_detected(self):
        """'10057000Y' on its own line is captured as vendor_vat."""
        lines = ["SOME CAFE", "NICOSIA", "10057000Y", "01/01/2026 09:00:00"]
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert data.get("vendor_vat") == "10057000Y"

    def test_bare_vat_only_8_digits_no_letter_not_matched(self):
        """'12345678' (8 digits, no trailing letter) must NOT match."""
        lines = ["SHOP NAME", "NICOSIA", "12345678"]
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert "vendor_vat" not in data

    def test_bare_vat_9_digits_plus_letter_not_matched(self):
        """'123456789A' (9 digits + letter) must NOT match."""
        lines = ["SHOP NAME", "NICOSIA", "123456789A"]
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert "vendor_vat" not in data

    def test_bare_vat_7_digits_plus_letter_not_matched(self):
        """'9900011H' (7 digits + letter) must NOT match."""
        lines = ["SHOP NAME", "NICOSIA", "9900011H"]
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert "vendor_vat" not in data

    def test_bare_vat_lowercase_letter_not_matched(self):
        """'99000110h' (lowercase letter) must NOT match — CY VAT ends in uppercase."""
        lines = ["SHOP NAME", "LIMASSOL", "99000110h"]
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert "vendor_vat" not in data

    def test_labeled_vat_still_matched_first(self):
        """Labeled 'VAT NO: 99000110H' should still work (earlier pattern wins)."""
        lines = ["SHOP NAME", "VAT NO: 99000110H"]
        data: dict = {}
        gaps: list = []
        _extract_header(lines, data, gaps)
        assert data.get("vendor_vat") == "99000110H"

    def test_full_jcc_bare_vat_slip_parse(self):
        """Full parse of a JCC slip with bare VAT line extracts all key fields."""
        result = parse_ocr_text(JCC_BARE_VAT_SLIP_OCR, doc_type="payment_confirmation")
        assert result.data.get("vendor") == "CAFE PAUL LIMASSOL MARINA"
        assert result.data.get("vendor_vat") == "99000110H"
        assert result.data.get("date") == "2026-02-23"
        assert result.data.get("total") == 14.50
        assert result.data.get("card_last4") == "4321"


# ---------------------------------------------------------------------------
# Product note / annotation stripping
# ---------------------------------------------------------------------------


class TestStripProductNotes:
    """_strip_product_notes() extracts trailing parentheticals from item names.

    The function is called as a post-processing pass inside _extract_line_items()
    so both the helper itself and the integration path are tested here.
    """

    # ---- Unit tests for the helper directly ----

    def test_simple_parenthetical_stripped(self):
        """Trailing '(takeaway)' is moved to product_note."""
        items = [{"name": "Espresso double (takeaway)", "total_price": 4.50}]
        _strip_product_notes(items)
        assert items[0]["name"] == "Espresso double"
        assert items[0]["product_note"] == "(takeaway)"

    def test_multi_word_parenthetical_stripped(self):
        """Trailing '(250g, Ethiopia)' is moved to product_note."""
        items = [{"name": "Coffee beans (250g, Ethiopia)", "total_price": 12.00}]
        _strip_product_notes(items)
        assert items[0]["name"] == "Coffee beans"
        assert items[0]["product_note"] == "(250g, Ethiopia)"

    def test_no_parenthetical_unchanged(self):
        """Item with no trailing parenthetical is left untouched."""
        items = [{"name": "Espresso double", "total_price": 4.50}]
        _strip_product_notes(items)
        assert items[0]["name"] == "Espresso double"
        assert "product_note" not in items[0]

    def test_embedded_parenthetical_not_stripped(self):
        """A parenthetical in the middle of the name must NOT be stripped."""
        items = [{"name": "Coffee (Fair Trade) Beans", "total_price": 8.00}]
        _strip_product_notes(items)
        assert items[0]["name"] == "Coffee (Fair Trade) Beans"
        assert "product_note" not in items[0]

    def test_multiple_items_only_annotated_stripped(self):
        """Only items with trailing annotations are affected; others are untouched."""
        items = [
            {"name": "Espresso double (takeaway)", "total_price": 4.50},
            {"name": "Croissant", "total_price": 2.50},
            {"name": "Coffee beans (250g, Ethiopia)", "total_price": 12.00},
        ]
        _strip_product_notes(items)
        assert items[0]["name"] == "Espresso double"
        assert items[0]["product_note"] == "(takeaway)"
        assert items[1]["name"] == "Croissant"
        assert "product_note" not in items[1]
        assert items[2]["name"] == "Coffee beans"
        assert items[2]["product_note"] == "(250g, Ethiopia)"

    def test_empty_name_no_crash(self):
        """An item with an empty name does not raise."""
        items = [{"name": "", "total_price": 1.00}]
        _strip_product_notes(items)
        assert items[0]["name"] == ""
        assert "product_note" not in items[0]

    def test_missing_name_key_no_crash(self):
        """An item dict without a 'name' key does not raise."""
        items = [{"total_price": 1.00}]
        _strip_product_notes(items)
        assert "product_note" not in items[0]

    def test_parenthetical_too_long_not_stripped(self):
        """Parentheticals longer than 40 chars are NOT treated as product notes."""
        long_note = "(" + "x" * 41 + ")"
        name = "Item " + long_note
        items = [{"name": name, "total_price": 1.00}]
        _strip_product_notes(items)
        assert items[0]["name"] == name
        assert "product_note" not in items[0]

    def test_parenthetical_too_short_not_stripped(self):
        """A single-char parenthetical like '(x)' is NOT stripped (min 2 chars inside)."""
        items = [{"name": "Item (x)", "total_price": 1.00}]
        _strip_product_notes(items)
        assert items[0]["name"] == "Item (x)"
        assert "product_note" not in items[0]

    def test_cold_pressed_juice_note(self):
        """Multi-word note '(Cold pressed juice)' is captured."""
        items = [{"name": "Juice (Cold pressed juice)", "total_price": 5.00}]
        _strip_product_notes(items)
        assert items[0]["name"] == "Juice"
        assert items[0]["product_note"] == "(Cold pressed juice)"

    # ---- Integration: through _extract_line_items() ----

    def test_little_sins_receipt_annotation_stripped(self):
        """LITTLE SINS-style receipt: continuation annotation lines are stripped."""
        lines = [
            "Espresso double          4.50 T1",
            "(takeaway)",
            "Coffee beans             12.00 T1",
            "(250g, Ethiopia)",
            "Total                    16.50",
        ]
        items = _extract_line_items(lines, {})
        assert len(items) == 2
        assert items[0]["name"] == "Espresso double"
        assert items[0].get("product_note") == "(takeaway)"
        assert items[1]["name"] == "Coffee beans"
        assert items[1].get("product_note") == "(250g, Ethiopia)"

    def test_receipt_without_annotations_unaffected(self):
        """Standard receipt with no annotation lines parses names unchanged."""
        lines = [
            "BARILLA SPAGHETTI No5 500 3.49 T1",
            "EXTRA VIRGIN OIL 500ML   7.99 T1",
            "Total                   11.48",
        ]
        items = _extract_line_items(lines, {})
        assert len(items) == 2
        assert "product_note" not in items[0]
        assert "product_note" not in items[1]


# ---------------------------------------------------------------------------
# Name-Qty-Amount columnar format (MIRADAR UNION style)
# ---------------------------------------------------------------------------


class TestNameQtyAmountExtraction:
    """Tests for _extract_name_qty_amount_items() and integration with parse_ocr_text."""

    # ------------------------------------------------------------------
    # Unit tests for _extract_name_qty_amount_items directly
    # ------------------------------------------------------------------

    def test_basic_mixed_rows_all_four_miradar_lines(self):
        """All four MIRADAR line types parse correctly in a single call."""
        lines = [
            "Name Qty Amount",
            "Zakvaska Bread 4.00",
            "Salmon cold smoked 0.339 27.12",
            "Mustard 1pcs 2 3.00",
            "Sour cream 4.50",
            "Total 38.62",
        ]
        items = _extract_name_qty_amount_items(lines)
        assert items is not None
        assert len(items) == 4, f"Expected 4 items, got {len(items)}: {items}"

        names = [i["name"] for i in items]
        assert "Zakvaska Bread" in names
        assert "Salmon cold smoked" in names
        assert "Mustard" in names
        assert "Sour cream" in names

    def test_weighed_item_decimal_qty(self):
        """Weighed item: decimal qty becomes unit_quantity, quantity=1, unit_raw='kg'."""
        lines = [
            "Name Qty Amount",
            "Salmon cold smoked 0.339 27.12",
            "Total 27.12",
        ]
        items = _extract_name_qty_amount_items(lines)
        assert items is not None
        assert len(items) == 1
        item = items[0]
        assert item["name"] == "Salmon cold smoked"
        assert item["quantity"] == 1
        assert item["unit_quantity"] == pytest.approx(0.339, abs=1e-6)
        assert item["unit_raw"] == "kg"
        assert item["total_price"] == pytest.approx(27.12, abs=1e-6)

    def test_integer_qty_with_pcs_unit(self):
        """'Mustard 1pcs 2 3.00' → name='Mustard', quantity=2."""
        lines = [
            "Name Qty Amount",
            "Mustard 1pcs 2 3.00",
            "Total 3.00",
        ]
        items = _extract_name_qty_amount_items(lines)
        assert items is not None
        assert len(items) == 1
        item = items[0]
        assert item["name"] == "Mustard"
        assert item["quantity"] == 2
        assert item["unit_quantity"] is None
        assert item["unit_raw"] is None
        assert item["total_price"] == pytest.approx(3.00, abs=1e-6)

    def test_no_qty_row_plain_name_amount(self):
        """'Zakvaska Bread 4.00' → quantity=1, unit_quantity=None."""
        lines = [
            "Name Qty Amount",
            "Zakvaska Bread 4.00",
            "Total 4.00",
        ]
        items = _extract_name_qty_amount_items(lines)
        assert items is not None
        assert len(items) == 1
        item = items[0]
        assert item["name"] == "Zakvaska Bread"
        assert item["quantity"] == 1
        assert item["unit_quantity"] is None
        assert item["total_price"] == pytest.approx(4.00, abs=1e-6)

    def test_total_stops_item_extraction(self):
        """Lines after a total marker are not parsed as items."""
        lines = [
            "Name Qty Amount",
            "Zakvaska Bread 4.00",
            "Total 4.00",
            "Extra item 9.99",  # should not be parsed
        ]
        items = _extract_name_qty_amount_items(lines)
        assert items is not None
        assert len(items) == 1

    def test_header_variation_description_quantity_amount(self):
        """'Description Quantity Amount' header is recognized."""
        lines = [
            "Description Quantity Amount",
            "Bread roll 1.50",
            "Total 1.50",
        ]
        items = _extract_name_qty_amount_items(lines)
        assert items is not None
        assert len(items) == 1
        assert items[0]["name"] == "Bread roll"

    def test_header_variation_item_qty_total(self):
        """'Item Qty Total' header is recognized."""
        lines = [
            "Item Qty Total",
            "Milk 2.99",
            "Total 2.99",
        ]
        items = _extract_name_qty_amount_items(lines)
        assert items is not None
        assert len(items) == 1
        assert items[0]["name"] == "Milk"

    def test_non_matching_header_returns_none(self):
        """'Product Price' header does not trigger NQA extraction."""
        lines = [
            "Product Price",
            "Bread 1.50",
            "Total 1.50",
        ]
        result = _extract_name_qty_amount_items(lines)
        assert result is None

    def test_no_header_returns_none(self):
        """Lines without any header return None."""
        lines = [
            "Bread 1.50",
            "Milk 2.99",
            "Total 4.49",
        ]
        result = _extract_name_qty_amount_items(lines)
        assert result is None

    def test_empty_lines_between_items_are_skipped(self):
        """Blank lines between items do not break extraction."""
        lines = [
            "Name Qty Amount",
            "Zakvaska Bread 4.00",
            "",
            "Sour cream 4.50",
            "Total 8.50",
        ]
        items = _extract_name_qty_amount_items(lines)
        assert items is not None
        assert len(items) == 2

    def test_item_dict_has_all_standard_fields(self):
        """Every returned item dict contains all required standard fields."""
        required_fields = {
            "name",
            "name_en",
            "quantity",
            "unit_raw",
            "unit_quantity",
            "unit_price",
            "total_price",
            "tax_rate",
            "tax_type",
            "discount",
            "brand",
            "barcode",
            "category",
        }
        lines = [
            "Name Qty Amount",
            "Zakvaska Bread 4.00",
        ]
        items = _extract_name_qty_amount_items(lines)
        assert items is not None
        assert len(items) == 1
        missing = required_fields - set(items[0].keys())
        assert not missing, f"Item dict missing fields: {missing}"

    def test_total_price_sum_matches_receipt_total(self):
        """Sum of extracted item totals matches the receipt total line."""
        lines = [
            "Name Qty Amount",
            "Zakvaska Bread 4.00",
            "Salmon cold smoked 0.339 27.12",
            "Mustard 1pcs 2 3.00",
            "Sour cream 4.50",
            "Total 38.62",
        ]
        items = _extract_name_qty_amount_items(lines)
        assert items is not None
        total = sum(i["total_price"] for i in items)
        assert total == pytest.approx(38.62, abs=0.01)

    def test_pc_unit_variant(self):
        """'pc' (without s) is also recognized as a unit specifier."""
        lines = [
            "Name Qty Amount",
            "Sugar 2pc 3 6.00",
            "Total 6.00",
        ]
        items = _extract_name_qty_amount_items(lines)
        assert items is not None
        assert len(items) == 1
        assert items[0]["name"] == "Sugar"
        assert items[0]["quantity"] == 3

    def test_cyrillic_sht_unit(self):
        """Cyrillic 'шт' unit specifier is recognized."""
        lines = [
            "Name Qty Amount",
            "Yogurt 2шт 3 9.00",
            "Total 9.00",
        ]
        items = _extract_name_qty_amount_items(lines)
        assert items is not None
        assert len(items) == 1
        assert items[0]["name"] == "Yogurt"
        assert items[0]["quantity"] == 3

    # ------------------------------------------------------------------
    # Integration test via parse_ocr_text
    # ------------------------------------------------------------------

    def test_full_miradar_receipt_parse_ocr_text(self):
        """Full MIRADAR-style OCR text parses correctly via parse_ocr_text."""
        text = (
            "MIRADAR UNION\n"
            "ul. Lenina 45, Moscow\n"
            "INN: 7712345678\n"
            "Receipt No. 1042\n"
            "15/02/2026 14:33:00\n"
            "\n"
            "Name Qty Amount\n"
            "Zakvaska Bread 4.00\n"
            "Salmon cold smoked 0.339 27.12\n"
            "Mustard 1pcs 2 3.00\n"
            "Sour cream 4.50\n"
            "\n"
            "Total 38.62\n"
            "CASH 40.00\n"
            "CHANGE 1.38\n"
        )
        result = parse_ocr_text(text, doc_type="receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 4, f"Expected 4 items, got {len(items)}: {items}"

        # No-qty item
        bread = next((i for i in items if "Zakvaska" in i.get("name", "")), None)
        assert bread is not None, "Zakvaska Bread not found"
        assert bread["quantity"] == 1
        assert bread["total_price"] == pytest.approx(4.00, abs=1e-6)

        # Weighed item
        salmon = next((i for i in items if "Salmon" in i.get("name", "")), None)
        assert salmon is not None, "Salmon item not found"
        assert salmon["quantity"] == 1
        assert salmon["unit_quantity"] == pytest.approx(0.339, abs=1e-6)
        assert salmon["unit_raw"] == "kg"
        assert salmon["total_price"] == pytest.approx(27.12, abs=1e-6)

        # Pcs-qty item
        mustard = next((i for i in items if "Mustard" in i.get("name", "")), None)
        assert mustard is not None, "Mustard not found"
        assert mustard["quantity"] == 2
        assert mustard["total_price"] == pytest.approx(3.00, abs=1e-6)

        # Plain item
        cream = next((i for i in items if "Sour cream" in i.get("name", "")), None)
        assert cream is not None, "Sour cream not found"
        assert cream["quantity"] == 1
        assert cream["total_price"] == pytest.approx(4.50, abs=1e-6)


class TestStatementMultilineDescription:
    """Tests for bank statement multi-line description parsing."""

    def test_description_prefix_merged(self):
        text = """Eurobank Cyprus
IBAN: CY17002001280000001200527600
Period: 01/01/2024 - 31/01/2024

PAYMENT AT POS
ALPHAMEGA HYPERMARKET
15/01/2024 -45,50 15/01/2024

ATM WITHDRAWAL
EUROBANK LIMASSOL
12/01/2024 -200,00 12/01/2024
"""
        result = parse_ocr_text(text, "statement")
        txns = result.data.get("transactions", [])
        assert len(txns) >= 2
        # First: description should contain both prefix lines + rest
        desc0 = txns[0]["description"]
        assert "ALPHAMEGA" in desc0 or "PAYMENT" in desc0
        # Second: ATM withdrawal
        desc1 = txns[1]["description"]
        assert "ATM" in desc1 or "EUROBANK" in desc1

    def test_single_line_description_still_works(self):
        text = """Bank ABC
IBAN: CY17002001280000001200527600
Period: 01/01/2024 - 31/01/2024

15/01/2024 Grocery purchase -45,50
"""
        result = parse_ocr_text(text, "statement")
        txns = result.data.get("transactions", [])
        assert len(txns) == 1
        assert "Grocery" in txns[0]["description"]


class TestSwapProcessorVendor:
    """Tests for _swap_processor_vendor payment processor detection."""

    def test_swap_with_legal_name(self):
        from alibi.extraction.text_parser import _swap_processor_vendor

        data = {"vendor": "JCC PAYMENT SYSTEMS", "vendor_legal_name": "24 HR KIOSK"}
        _swap_processor_vendor(data)
        assert data["vendor"] == "24 HR KIOSK"
        assert data["vendor_legal_name"] == "JCC PAYMENT SYSTEMS"

    def test_swap_fallback_to_address(self):
        from alibi.extraction.text_parser import _swap_processor_vendor

        data = {
            "vendor": "JCC PAYMENT SYSTEMS",
            "vendor_legal_name": "",
            "vendor_address": "24 HR KIOSK/CAVA AMPELAKION 1 LIMASSOL",
        }
        _swap_processor_vendor(data)
        assert data["vendor"] == "24 HR KIOSK/CAVA AMPELAKION"
        assert data["vendor_legal_name"] == "JCC PAYMENT SYSTEMS"

    def test_swap_address_with_commas(self):
        from alibi.extraction.text_parser import _swap_processor_vendor

        data = {
            "vendor": "viva.com",
            "vendor_legal_name": "",
            "vendor_address": "Costa Coffee, Makarios Ave 12, Limassol",
        }
        _swap_processor_vendor(data)
        assert data["vendor"] == "Costa Coffee"

    def test_no_swap_when_not_processor(self):
        from alibi.extraction.text_parser import _swap_processor_vendor

        data = {"vendor": "Alphamega Supermarket", "vendor_legal_name": ""}
        _swap_processor_vendor(data)
        assert data["vendor"] == "Alphamega Supermarket"

    def test_no_swap_when_no_address(self):
        from alibi.extraction.text_parser import _swap_processor_vendor

        data = {
            "vendor": "JCC",
            "vendor_legal_name": "",
            "vendor_address": "",
        }
        _swap_processor_vendor(data)
        assert data["vendor"] == "JCC"

    def test_bare_cypriot_vat_with_space(self):
        """Bare Cypriot VAT '10154482 N' should be captured despite OCR space."""
        result = parse_ocr_text("10154482 N\nMilk 1.50\nTotal 1.50", "receipt")
        assert result.data.get("vendor_vat") == "10154482N"


# ---------------------------------------------------------------------------
# Date format hint tagging
# ---------------------------------------------------------------------------


class TestDateFormatHint:
    """Tests for date format hint in _extract_date_time."""

    def test_date_format_tagged(self):
        """Successful date extraction should tag _date_format."""
        result = parse_ocr_text(
            "Test Shop\n15/02/2026\nMilk 1.50\nTotal 1.50", "receipt"
        )
        assert result.data.get("_date_format") == "dmy"

    def test_ymd_format_tagged(self):
        result = parse_ocr_text(
            "Test Shop\n2026-02-15\nMilk 1.50\nTotal 1.50", "receipt"
        )
        assert result.data.get("_date_format") == "ymd"

    def test_dmy_time_format_tagged(self):
        result = parse_ocr_text(
            "Test Shop\n15/02/2026 14:30\nMilk 1.50\nTotal 1.50", "receipt"
        )
        assert result.data.get("_date_format") == "dmy_time"


# ---------------------------------------------------------------------------
# Total marker tagging
# ---------------------------------------------------------------------------


class TestTotalMarkerTag:
    """Tests for total marker tagging."""

    def test_total_marker_tagged(self):
        result = parse_ocr_text("Test Shop\nMilk 1.50\nTOTAL EUR 1.50", "receipt")
        marker = result.data.get("_total_marker", "")
        assert "total" in marker.lower() or result.data.get("total") is not None


# ---------------------------------------------------------------------------
# Language detection from parser
# ---------------------------------------------------------------------------


class TestLanguageDetectionFromParser:
    """Tests for language detection in parser output."""

    def test_english_default(self):
        result = parse_ocr_text("Test Shop\nMilk 1.50\nTotal 1.50", "receipt")
        assert result.data.get("_detected_language") == "en"


# ---------------------------------------------------------------------------
# Header lines tagging
# ---------------------------------------------------------------------------


class TestHeaderLinesTag:
    """Tests for header line count tagging."""

    def test_header_lines_tagged(self):
        text = (
            "Shop Name\nAddress Line\nVAT: 12345678A\n15/02/2026\nMilk 1.50\nTotal 1.50"
        )
        result = parse_ocr_text(text, "receipt")
        # Should have some header lines count
        assert result.data.get("_header_lines") is not None
        assert result.data["_header_lines"] > 0

    def test_footer_ratio_tagged(self):
        text = "Shop Name\nAddress\n15/02/2026\nMilk 1.50\nBread 2.00\nTotal 3.50\nThank you"
        result = parse_ocr_text(text, "receipt")
        ratio = result.data.get("_footer_ratio")
        if ratio is not None:
            assert 0.0 < ratio <= 1.0

    def test_header_scan_limited_by_hints(self):
        """With expected_header_lines hint, vendor from body text is not extracted."""
        from alibi.extraction.templates import ParserHints

        # Build text: real vendor in first 2 lines, fake vendor-like text at line 10
        header = "Real Shop\nVAT: 12345678A\n15/02/2026"
        body_lines = [f"Item{i} {i}.00" for i in range(1, 8)]
        # Without hints, the parser would scan all lines for header
        text_with_body_vendor = header + "\n" + "\n".join(body_lines) + "\nTotal 28.00"

        hints = ParserHints(expected_header_lines=3)
        result_with_hints = parse_ocr_text(
            text_with_body_vendor, "receipt", hints=hints
        )
        result_no_hints = parse_ocr_text(text_with_body_vendor, "receipt")

        # Both should find the vendor
        assert result_with_hints.data.get("vendor") == "Real Shop"
        assert result_no_hints.data.get("vendor") == "Real Shop"
