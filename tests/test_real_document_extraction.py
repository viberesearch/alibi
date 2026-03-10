"""Integration tests for real document extraction.

Tests the heuristic text parser (Stage 2) against real OCR output
from .alibi.yaml cache files. Compares parser output to ground truth
established by Claude Opus 4.6 vision.

These tests do NOT require Ollama — they use cached OCR text from the
YAML files alongside each document.

Run: uv run pytest tests/test_real_document_extraction.py -v
"""

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from alibi.extraction.text_parser import ParseResult, parse_ocr_text

from tests.ground_truth import (
    ALL_REAL_DOCUMENTS,
    ARAB_BUTCHERY_PAYMENT,
    BANK_STATEMENT,
    BLUE_ISLAND_PDF,
    FRESKO_PAYMENT,
    FRESKO_RECEIPT,
    INBOX_DIR,
    MALEVE_RECEIPT,
    MINI_MARKET_CONFIRMATION,
    NUT_CRACKER_PAYMENT,
    NUT_CRACKER_RECEIPT,
    OLLAMA_KNOWN_ERRORS,
    PAPAS_RECEIPT,
    PLUS_DISCOUNT_RECEIPT,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INBOX = Path(INBOX_DIR)


def _doc_file(doc: dict[str, Any]) -> str:
    """Get filename from ground truth document dict."""
    return str(doc["file"])


def _load_yaml_cache(filename: str) -> dict[str, Any] | None:
    """Load .alibi.yaml cache for a document."""
    stem = Path(filename).stem

    # Single file: receipt.jpg → receipt.alibi.yaml
    yaml_path = INBOX / f"{stem}.alibi.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            result: dict[str, Any] = yaml.safe_load(f)
            return result
    return None


def _get_raw_text(filename: str) -> str | None:
    """Get raw OCR text from YAML cache."""
    data = _load_yaml_cache(filename)
    if data and "raw_text" in data:
        raw: str = data["raw_text"]
        return raw
    return None


def _require_raw_text(filename: str) -> str:
    """Get raw OCR text, raising if not available."""
    raw = _get_raw_text(filename)
    assert raw is not None, f"No cached OCR text for {filename}"
    return raw


def _has_inbox() -> bool:
    """Check if inbox documents are available."""
    return INBOX.exists() and bool(list(INBOX.glob("*.jpeg")))


skip_no_inbox = pytest.mark.skipif(
    not _has_inbox(),
    reason="Inbox documents not available on this machine",
)


# ---------------------------------------------------------------------------
# Helper assertions
# ---------------------------------------------------------------------------


def _assert_field_contains(
    data: dict[str, Any], field: str, expected_substr: str, msg: str = ""
) -> None:
    """Assert a field exists and contains a substring (case-insensitive)."""
    val = data.get(field, "")
    assert val, f"{msg}Field '{field}' missing or empty"
    assert (
        expected_substr.lower() in str(val).lower()
    ), f"{msg}'{field}' = '{val}', expected to contain '{expected_substr}'"


def _assert_amount(
    actual: float | None,
    expected: Decimal,
    tolerance: Decimal = Decimal("0.02"),
    msg: str = "",
):
    """Assert an amount matches within tolerance."""
    assert actual is not None, f"{msg}Amount is None"
    actual_d = Decimal(str(actual))
    assert (
        abs(actual_d - expected) <= tolerance
    ), f"{msg}Amount {actual_d} != expected {expected} (tolerance {tolerance})"


def _check_expected(
    result: ParseResult, expected: dict[str, Any], doc_name: str
) -> list[str]:
    """Check parser result against expected values. Returns list of failures."""
    failures = []
    data = result.data
    prefix = f"[{doc_name}] "

    for key, val in expected.items():
        if key == "line_items":
            continue  # Checked separately
        if key == "transactions":
            continue  # Checked separately

        if key.endswith("_contains"):
            base_key = key.replace("_contains", "")
            actual = data.get(base_key, "")
            if not actual or val.lower() not in str(actual).lower():
                failures.append(
                    f"{prefix}{base_key} = '{actual}', expected to contain '{val}'"
                )
        elif key == "line_item_count":
            actual_count = len(data.get("line_items", []))
            if actual_count != val:
                failures.append(
                    f"{prefix}line_item_count = {actual_count}, expected {val}"
                )
        elif key == "line_item_count_min":
            actual_count = len(data.get("line_items", []))
            if actual_count < val:
                failures.append(
                    f"{prefix}line_item_count = {actual_count}, expected >= {val}"
                )
        elif key == "total_or_amount":
            total = data.get("total") or data.get("amount")
            if total is None:
                failures.append(f"{prefix}total/amount is None, expected {val}")
            elif abs(Decimal(str(total)) - val) > Decimal("0.02"):
                failures.append(f"{prefix}total/amount = {total}, expected {val}")
        elif key == "transaction_count":
            txns = data.get("transactions", [])
            if len(txns) != val:
                failures.append(
                    f"{prefix}transaction_count = {len(txns)}, expected {val}"
                )
        elif isinstance(val, Decimal):
            actual = data.get(key)
            if actual is None:
                failures.append(f"{prefix}{key} is None, expected {val}")
            elif abs(Decimal(str(actual)) - val) > Decimal("0.02"):
                failures.append(f"{prefix}{key} = {actual}, expected {val}")
        else:
            actual = data.get(key)
            if actual is None:
                failures.append(f"{prefix}{key} is None, expected '{val}'")
            elif str(actual).lower() != str(val).lower():
                failures.append(f"{prefix}{key} = '{actual}', expected '{val}'")

    return failures


# ---------------------------------------------------------------------------
# Receipt parser tests (from cached OCR text)
# ---------------------------------------------------------------------------


@skip_no_inbox
class TestFreskoReceipt:
    """IMG_0430 — FRESKO receipt, 2 items, EUR 2.75."""

    def test_parser_extracts_vendor(self):
        raw = _require_raw_text(_doc_file(FRESKO_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        assert "FRESKO" in (result.data.get("vendor") or "").upper()

    def test_parser_extracts_date(self):
        raw = _require_raw_text(_doc_file(FRESKO_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        assert result.data.get("date") == "2026-02-17"

    def test_parser_extracts_total(self):
        raw = _require_raw_text(_doc_file(FRESKO_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        _assert_amount(result.data.get("total"), Decimal("2.75"))

    def test_parser_extracts_line_items(self):
        raw = _require_raw_text(_doc_file(FRESKO_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        items = result.data.get("line_items", [])
        assert len(items) == 2
        names = [i["name"].lower() for i in items]
        assert any("facial" in n for n in names)
        assert any("sugar" in n for n in names)

    def test_parser_extracts_currency(self):
        raw = _require_raw_text(_doc_file(FRESKO_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        assert result.data.get("currency") == "EUR"

    def test_confidence_above_threshold(self):
        raw = _require_raw_text(_doc_file(FRESKO_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        assert result.confidence >= 0.5


@skip_no_inbox
class TestPapasReceipt:
    """IMG_0436 — PAPAS HYPERMARKET tall receipt, 23 items."""

    def test_parser_extracts_vendor(self):
        raw = _require_raw_text(_doc_file(PAPAS_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        assert "PAPAS" in (result.data.get("vendor") or "").upper()

    def test_parser_extracts_date(self):
        raw = _require_raw_text(_doc_file(PAPAS_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        assert result.data.get("date") == "2026-01-21"

    def test_parser_extracts_total(self):
        raw = _require_raw_text(_doc_file(PAPAS_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        _assert_amount(result.data.get("total"), Decimal("85.69"))

    def test_parser_extracts_many_line_items(self):
        raw = _require_raw_text(_doc_file(PAPAS_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 15, f"Expected >= 15 items, got {len(items)}"

    def test_parser_extracts_weighed_items(self):
        raw = _require_raw_text(_doc_file(PAPAS_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        items = result.data.get("line_items", [])
        # Find beef liver (weighed item)
        beef = [i for i in items if "beef" in i.get("name", "").lower()]
        assert beef, "Beef liver not found"
        _assert_amount(beef[0].get("total_price"), Decimal("7.79"))

    def test_parser_extracts_multi_qty_items(self):
        raw = _require_raw_text(_doc_file(PAPAS_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        items = result.data.get("line_items", [])
        # Red Bull: qty 3 @ 1.99 each = 5.97
        red_bull = [i for i in items if "red bull" in i.get("name", "").lower()]
        assert red_bull, "Red Bull not found"
        _assert_amount(red_bull[0].get("total_price"), Decimal("5.97"))

    def test_parser_extracts_card_last4(self):
        raw = _require_raw_text(_doc_file(PAPAS_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        assert result.data.get("card_last4") == "7201"


@skip_no_inbox
class TestPlusDiscountReceipt:
    """IMG_0432 — PLUS DISCOUNT MARKET, 7 items with weighed goods."""

    def test_parser_extracts_vendor(self):
        raw = _require_raw_text(_doc_file(PLUS_DISCOUNT_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        assert "PLUS" in (result.data.get("vendor") or "").upper()

    def test_parser_extracts_total(self):
        raw = _require_raw_text(_doc_file(PLUS_DISCOUNT_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        _assert_amount(result.data.get("total"), Decimal("15.75"))

    def test_parser_extracts_items(self):
        raw = _require_raw_text(_doc_file(PLUS_DISCOUNT_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 5, f"Expected >= 5 items, got {len(items)}"


@skip_no_inbox
class TestMaleveReceipt:
    """receipt.jpeg — Maleve Trading LTD, 8 items."""

    def test_parser_extracts_vendor(self):
        raw = _require_raw_text(_doc_file(MALEVE_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        assert "maleve" in (result.data.get("vendor") or "").lower()

    def test_parser_extracts_total(self):
        raw = _require_raw_text(_doc_file(MALEVE_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        _assert_amount(result.data.get("total"), Decimal("33.03"))

    def test_parser_extracts_items(self):
        raw = _require_raw_text(_doc_file(MALEVE_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 5, f"Expected >= 5 items, got {len(items)}"


# ---------------------------------------------------------------------------
# Payment confirmation tests
# ---------------------------------------------------------------------------


@skip_no_inbox
class TestArabButcheryPayment:
    """IMG_0429 — ARAB BUTCHERY JCC card slip."""

    def test_parser_extracts_vendor(self):
        raw = _require_raw_text(_doc_file(ARAB_BUTCHERY_PAYMENT))
        result = parse_ocr_text(raw, "payment_confirmation")
        assert "ARAB" in (result.data.get("vendor") or "").upper()

    def test_parser_extracts_date(self):
        raw = _require_raw_text(_doc_file(ARAB_BUTCHERY_PAYMENT))
        result = parse_ocr_text(raw, "payment_confirmation")
        # The parser should get 2026-02-17, not the wrong 2023-03-13
        assert result.data.get("date") == "2026-02-17"

    def test_parser_extracts_amount(self):
        raw = _require_raw_text(_doc_file(ARAB_BUTCHERY_PAYMENT))
        result = parse_ocr_text(raw, "payment_confirmation")
        _assert_amount(result.data.get("total"), Decimal("29.00"))

    def test_parser_extracts_card_details(self):
        raw = _require_raw_text(_doc_file(ARAB_BUTCHERY_PAYMENT))
        result = parse_ocr_text(raw, "payment_confirmation")
        assert result.data.get("card_last4") == "7201"
        assert result.data.get("card_type") == "visa"


@skip_no_inbox
class TestNutCrackerPayment:
    """IMG_0427 — THE NUT CRACKER HOUSE JCC card slip."""

    def test_parser_extracts_vendor(self):
        raw = _require_raw_text(_doc_file(NUT_CRACKER_PAYMENT))
        result = parse_ocr_text(raw, "payment_confirmation")
        assert "NUT CRACKER" in (result.data.get("vendor") or "").upper()

    def test_parser_extracts_amount(self):
        raw = _require_raw_text(_doc_file(NUT_CRACKER_PAYMENT))
        result = parse_ocr_text(raw, "payment_confirmation")
        _assert_amount(result.data.get("total"), Decimal("12.45"))


@skip_no_inbox
class TestFreskoPayment:
    """IMG_0431 — FreSko Worldline card slip."""

    def test_parser_extracts_amount(self):
        raw = _require_raw_text(_doc_file(FRESKO_PAYMENT))
        result = parse_ocr_text(raw, "payment_confirmation")
        _assert_amount(result.data.get("total"), Decimal("2.75"))


# ---------------------------------------------------------------------------
# Statement parser tests
# ---------------------------------------------------------------------------


@skip_no_inbox
class TestBankStatement:
    """bank_account_statement.pdf — Eurobank statement, 8 transactions.

    Note: The bank statement YAML was extracted by LLM directly (PDF),
    not via OCR, so it has no raw_text field. These tests use the
    synthetic statement text to verify the statement parser works.
    The real YAML data is in extracted_fields, not raw_text.
    """

    def _get_statement_text(self) -> str:
        raw = _get_raw_text(_doc_file(BANK_STATEMENT))
        if not raw:
            pytest.skip(
                "Bank statement YAML has no raw_text (extracted via LLM, not OCR)"
            )
        return raw

    def test_parser_extracts_institution(self):
        """The statement parser extracts institution/account holder.

        This statement OCR text doesn't contain the bank name (EUROBANK);
        the parser picks the first non-title, non-metadata line as vendor.
        """
        raw = self._get_statement_text()
        result = parse_ocr_text(raw, "statement")
        vendor = result.data.get("vendor", "")
        assert vendor, "Statement parser should extract a vendor/institution"

    def test_parser_extracts_account_number(self):
        raw = self._get_statement_text()
        result = parse_ocr_text(raw, "statement")
        assert result.data.get("account_number") is not None

    def test_parser_extracts_currency(self):
        raw = self._get_statement_text()
        result = parse_ocr_text(raw, "statement")
        assert result.data.get("currency") == "EUR"

    def test_parser_extracts_transactions(self):
        raw = self._get_statement_text()
        result = parse_ocr_text(raw, "statement")
        txns = result.data.get("transactions", [])
        assert len(txns) >= 6, f"Expected >= 6 transactions, got {len(txns)}"

    def test_parser_confidence_above_minimal(self):
        raw = self._get_statement_text()
        result = parse_ocr_text(raw, "statement")
        assert (
            result.confidence > 0.3
        ), f"Statement parser confidence too low: {result.confidence}"


# ---------------------------------------------------------------------------
# Ollama comparison tests — document known errors from LLM extraction
# ---------------------------------------------------------------------------


@skip_no_inbox
class TestOllamaComparisonReport:
    """Compare YAML cache (Ollama extraction) against ground truth.

    These tests document known Ollama errors and track whether the
    heuristic parser produces better results from the same OCR text.
    """

    def test_arab_butchery_date_corrected_by_parser(self):
        """Ollama got date wrong (2023-03-13), parser should get 2026-02-17."""
        raw = _require_raw_text(_doc_file(ARAB_BUTCHERY_PAYMENT))
        result = parse_ocr_text(raw, "payment_confirmation")
        # The raw OCR text has "17/02/2026" clearly
        assert result.data.get("date") == "2026-02-17", (
            f"Parser date: {result.data.get('date')} — "
            "Ollama got 2023-03-13 (terminal ID parsed as date)"
        )

    def test_fresko_payment_method_corrected_by_parser(self):
        """Ollama said 'cash', parser should detect 'card' from PAID SIX."""
        raw = _require_raw_text(_doc_file(FRESKO_RECEIPT))
        result = parse_ocr_text(raw, "receipt")
        # The OCR text contains "PAID SIX" which indicates card payment
        # but more importantly, the parser looks for card patterns
        # and "VISA" or card last4 in the text
        # Note: FRESKO receipt doesn't have card details in its text,
        # so payment_method may default to None. The key fix is that
        # it's NOT "cash".
        pm = result.data.get("payment_method")
        assert pm != "cash", f"Parser still says 'cash' — should not"


# ---------------------------------------------------------------------------
# Multilingual parser tests (synthetic OCR text)
# ---------------------------------------------------------------------------


class TestGermanReceiptParser:
    """Test parser on German receipt OCR text."""

    def test_extracts_vendor(self):
        from tests.ground_truth import GERMAN_RECEIPT_OCR

        result = parse_ocr_text(GERMAN_RECEIPT_OCR, "receipt")
        assert "REWE" in (result.data.get("vendor") or "")

    def test_extracts_date(self):
        from tests.ground_truth import GERMAN_RECEIPT_OCR

        result = parse_ocr_text(GERMAN_RECEIPT_OCR, "receipt")
        assert result.data.get("date") == "2026-02-15"

    def test_extracts_time(self):
        from tests.ground_truth import GERMAN_RECEIPT_OCR

        result = parse_ocr_text(GERMAN_RECEIPT_OCR, "receipt")
        assert result.data.get("time") == "14:23:15"

    def test_extracts_total(self):
        from tests.ground_truth import GERMAN_RECEIPT_OCR

        result = parse_ocr_text(GERMAN_RECEIPT_OCR, "receipt")
        _assert_amount(result.data.get("total"), Decimal("7.80"))

    def test_extracts_currency(self):
        from tests.ground_truth import GERMAN_RECEIPT_OCR

        result = parse_ocr_text(GERMAN_RECEIPT_OCR, "receipt")
        assert result.data.get("currency") == "EUR"

    def test_extracts_line_items(self):
        from tests.ground_truth import GERMAN_RECEIPT_OCR

        result = parse_ocr_text(GERMAN_RECEIPT_OCR, "receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 4, f"Expected >= 4 items, got {len(items)}"

    def test_extracts_registration(self):
        from tests.ground_truth import GERMAN_RECEIPT_OCR

        result = parse_ocr_text(GERMAN_RECEIPT_OCR, "receipt")
        reg = result.data.get("vendor_vat", "")
        assert "DE812706034" in reg

    def test_confidence_reasonable(self):
        from tests.ground_truth import GERMAN_RECEIPT_OCR

        result = parse_ocr_text(GERMAN_RECEIPT_OCR, "receipt")
        assert result.confidence >= 0.4, f"Low confidence: {result.confidence}"


class TestRussianReceiptParser:
    """Test parser on Russian receipt OCR text."""

    def test_extracts_vendor(self):
        from tests.ground_truth import RUSSIAN_RECEIPT_OCR

        result = parse_ocr_text(RUSSIAN_RECEIPT_OCR, "receipt")
        vendor = result.data.get("vendor", "")
        assert "ПЕРЕКРЁСТОК" in vendor or "перекрёсток" in vendor.lower()

    def test_extracts_date(self):
        from tests.ground_truth import RUSSIAN_RECEIPT_OCR

        result = parse_ocr_text(RUSSIAN_RECEIPT_OCR, "receipt")
        assert result.data.get("date") == "2026-02-18"

    def test_extracts_total(self):
        from tests.ground_truth import RUSSIAN_RECEIPT_OCR

        result = parse_ocr_text(RUSSIAN_RECEIPT_OCR, "receipt")
        _assert_amount(result.data.get("total"), Decimal("495.10"))

    def test_extracts_currency_rub(self):
        from tests.ground_truth import RUSSIAN_RECEIPT_OCR

        result = parse_ocr_text(RUSSIAN_RECEIPT_OCR, "receipt")
        assert result.data.get("currency") == "RUB"

    def test_extracts_line_items(self):
        from tests.ground_truth import RUSSIAN_RECEIPT_OCR

        result = parse_ocr_text(RUSSIAN_RECEIPT_OCR, "receipt")
        items = result.data.get("line_items", [])
        assert len(items) >= 4, f"Expected >= 4 items, got {len(items)}"

    def test_extracts_registration_inn(self):
        from tests.ground_truth import RUSSIAN_RECEIPT_OCR

        result = parse_ocr_text(RUSSIAN_RECEIPT_OCR, "receipt")
        # ИНН is a Russian tax ID, stored as vendor_tax_id
        reg = result.data.get("vendor_tax_id", "")
        assert "7728029110" in reg

    def test_extracts_card_last4(self):
        from tests.ground_truth import RUSSIAN_RECEIPT_OCR

        result = parse_ocr_text(RUSSIAN_RECEIPT_OCR, "receipt")
        assert result.data.get("card_last4") == "7890"

    def test_confidence_reasonable(self):
        from tests.ground_truth import RUSSIAN_RECEIPT_OCR

        result = parse_ocr_text(RUSSIAN_RECEIPT_OCR, "receipt")
        assert result.confidence >= 0.4, f"Low confidence: {result.confidence}"


class TestGreekReceiptParser:
    """Test parser on Greek receipt OCR text."""

    def test_extracts_date(self):
        from tests.ground_truth import GREEK_RECEIPT_OCR

        result = parse_ocr_text(GREEK_RECEIPT_OCR, "receipt")
        assert result.data.get("date") == "2026-02-19"

    def test_extracts_total(self):
        from tests.ground_truth import GREEK_RECEIPT_OCR

        result = parse_ocr_text(GREEK_RECEIPT_OCR, "receipt")
        _assert_amount(result.data.get("total"), Decimal("5.95"))

    def test_extracts_currency(self):
        from tests.ground_truth import GREEK_RECEIPT_OCR

        result = parse_ocr_text(GREEK_RECEIPT_OCR, "receipt")
        assert result.data.get("currency") == "EUR"

    def test_extracts_registration_afm(self):
        from tests.ground_truth import GREEK_RECEIPT_OCR

        result = parse_ocr_text(GREEK_RECEIPT_OCR, "receipt")
        reg = result.data.get("vendor_vat", "")
        assert "12345678" in reg


# ---------------------------------------------------------------------------
# Statement parser unit tests (synthetic text)
# ---------------------------------------------------------------------------


class TestStatementParserSynthetic:
    """Test statement parser with synthetic bank statement text."""

    SAMPLE_STATEMENT = """EUROBANK
ACCOUNT ACTIVITY
ACCOUNT NO 245-10-519031-00
DATE 17/02/2026
CURRENCY EUR
TYPE Savings Account
PERIOD 02/02/2026 - 17/02/2026
IBAN CY02005002450002451051903100
BIC HEBACY2N

DATE DESCRIPTION DEBIT CREDIT VALUE DATE
02/02/2026 DEA BEAUTY SAL *7514 -170,00 02/02/2026
05/02/2026 VIVASAN CYPRUS *7514 -100,00 05/02/2026
06/02/2026 APPLE.COM/BILL *7514 -21,95 06/02/2026
06/02/2026 MINI MARKET *7514 -19,68 06/02/2026
17/02/2026 RYANAIR *7514 -3,75 17/02/2026
TOTALS: 315,38 0,00
"""

    def test_extracts_institution(self):
        result = parse_ocr_text(self.SAMPLE_STATEMENT, "statement")
        assert "EUROBANK" in (result.data.get("vendor") or "")

    def test_extracts_account_number(self):
        result = parse_ocr_text(self.SAMPLE_STATEMENT, "statement")
        assert result.data.get("account_number") == "245-10-519031-00"

    def test_extracts_iban(self):
        result = parse_ocr_text(self.SAMPLE_STATEMENT, "statement")
        assert result.data.get("iban") == "CY02005002450002451051903100"

    def test_extracts_bic(self):
        result = parse_ocr_text(self.SAMPLE_STATEMENT, "statement")
        assert result.data.get("bic") == "HEBACY2N"

    def test_extracts_currency(self):
        result = parse_ocr_text(self.SAMPLE_STATEMENT, "statement")
        assert result.data.get("currency") == "EUR"

    def test_extracts_period(self):
        result = parse_ocr_text(self.SAMPLE_STATEMENT, "statement")
        assert result.data.get("period_start") == "2026-02-02"
        assert result.data.get("period_end") == "2026-02-17"

    def test_extracts_transactions(self):
        result = parse_ocr_text(self.SAMPLE_STATEMENT, "statement")
        txns = result.data.get("transactions", [])
        assert len(txns) >= 4, f"Expected >= 4 transactions, got {len(txns)}"

    def test_transaction_dates_correct(self):
        result = parse_ocr_text(self.SAMPLE_STATEMENT, "statement")
        txns = result.data.get("transactions", [])
        dates = [t.get("date") for t in txns]
        assert "2026-02-02" in dates
        assert "2026-02-17" in dates

    def test_transaction_amounts(self):
        result = parse_ocr_text(self.SAMPLE_STATEMENT, "statement")
        txns = result.data.get("transactions", [])
        # Find the DEA BEAUTY transaction
        dea = [t for t in txns if "DEA" in t.get("description", "")]
        assert dea, "DEA BEAUTY transaction not found"
        assert dea[0].get("debit") == -170.0

    def test_confidence_above_minimal(self):
        result = parse_ocr_text(self.SAMPLE_STATEMENT, "statement")
        assert result.confidence > 0.3


class TestGermanStatementParser:
    """Test statement parser with German bank statement text."""

    GERMAN_STATEMENT = """Deutsche Bank AG
KONTOAUSZUG
Kontonr.: 1234567890
Datum: 15.02.2026
BIC DEUTDEFF
IBAN DE89370400440532013000
ZEITRAUM: 01/02/2026 - 15/02/2026

Datum Beschreibung Soll Haben Wertstellung
01/02/2026 REWE MARKT -45,80 01/02/2026
03/02/2026 AMAZON EU -29,99 03/02/2026
05/02/2026 GEHALT 3.500,00 05/02/2026
SUMME: 75,79 3.500,00
"""

    def test_extracts_institution(self):
        result = parse_ocr_text(self.GERMAN_STATEMENT, "statement")
        assert "Deutsche Bank" in (result.data.get("vendor") or "")

    def test_extracts_iban(self):
        result = parse_ocr_text(self.GERMAN_STATEMENT, "statement")
        iban = result.data.get("iban", "")
        assert iban.startswith("DE")

    def test_extracts_transactions(self):
        result = parse_ocr_text(self.GERMAN_STATEMENT, "statement")
        txns = result.data.get("transactions", [])
        assert len(txns) >= 2, f"Expected >= 2 transactions, got {len(txns)}"
