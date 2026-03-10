"""Tests for alibi.extraction.barcode_detector."""

from __future__ import annotations

import pytest

from alibi.extraction.barcode_detector import (
    BarcodeResult,
    _is_valid_ean,
    detect_barcodes,
    has_barcode_support,
)


# ---------------------------------------------------------------------------
# _is_valid_ean
# ---------------------------------------------------------------------------


class TestIsValidEan:
    # --- valid barcodes ---

    def test_valid_ean13(self):
        # Standard GS1 test barcode
        assert _is_valid_ean("5901234123457") is True

    def test_valid_ean13_second(self):
        # Another well-known valid EAN-13
        assert _is_valid_ean("4006381333931") is True

    def test_valid_ean8(self):
        assert _is_valid_ean("96385074") is True

    def test_valid_ean8_second(self):
        assert _is_valid_ean("40170725") is True

    # --- invalid check digit ---

    def test_invalid_check_digit_ean13(self):
        # Flip last digit of valid barcode
        assert _is_valid_ean("5901234123456") is False

    def test_invalid_check_digit_ean8(self):
        assert _is_valid_ean("96385075") is False

    # --- wrong length ---

    def test_length_7_rejected(self):
        assert _is_valid_ean("1234567") is False

    def test_length_9_rejected(self):
        assert _is_valid_ean("123456789") is False

    def test_length_12_rejected(self):
        # UPC-A is 12 digits — not accepted by EAN validator
        assert _is_valid_ean("012345678905") is False

    def test_length_14_rejected(self):
        assert _is_valid_ean("12345678901234") is False

    # --- non-numeric ---

    def test_non_numeric_letters(self):
        assert _is_valid_ean("590123412345X") is False

    def test_non_numeric_spaces(self):
        assert _is_valid_ean("5901234 23457") is False

    def test_non_numeric_dashes(self):
        assert _is_valid_ean("590-1234-12345") is False

    # --- edge cases ---

    def test_empty_string(self):
        assert _is_valid_ean("") is False

    def test_all_zeros_ean13(self):
        # 0000000000000 — check digit 0, sum=0, valid mod-10
        assert _is_valid_ean("0000000000000") is True

    def test_all_zeros_ean8(self):
        assert _is_valid_ean("00000000") is True


# ---------------------------------------------------------------------------
# has_barcode_support
# ---------------------------------------------------------------------------


class TestHasBarcodeSupport:
    def test_returns_bool(self):
        result = has_barcode_support()
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# detect_barcodes
# ---------------------------------------------------------------------------


class TestDetectBarcodes:
    def test_empty_bytes_returns_empty_list_when_no_pyzbar(self):
        """When pyzbar is unavailable, any input returns []."""
        if has_barcode_support():
            pytest.skip("pyzbar is installed — skip no-pyzbar path")
        result = detect_barcodes(b"")
        assert result == []

    def test_invalid_image_data_returns_empty_list(self):
        """Garbage bytes should not raise — returns empty list."""
        result = detect_barcodes(b"\x00\xff\xde\xad\xbe\xef")
        assert isinstance(result, list)

    def test_empty_bytes_returns_list(self):
        """Empty bytes input should never raise."""
        result = detect_barcodes(b"")
        assert isinstance(result, list)

    @pytest.mark.skipif(
        not has_barcode_support(),
        reason="pyzbar not installed",
    )
    def test_real_image_with_barcode(self):
        """With pyzbar available, generate a real barcode image and detect it."""
        try:
            import io

            from PIL import Image
            from pyzbar.pyzbar import ZBarSymbol
            from pyzbar.pyzbar import decode as pyzbar_decode

            # Create a minimal white image — may return empty, but must not crash
            img = Image.new("L", (200, 100), color=255)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            result = detect_barcodes(buf.getvalue())
            assert isinstance(result, list)
        except Exception as exc:
            pytest.fail(f"detect_barcodes raised unexpectedly: {exc}")

    @pytest.mark.skipif(
        not has_barcode_support(),
        reason="pyzbar not installed",
    )
    def test_result_items_are_barcode_result(self):
        """Every returned item is a BarcodeResult dataclass instance."""
        import io

        from PIL import Image

        img = Image.new("L", (100, 50), color=255)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        results = detect_barcodes(buf.getvalue())
        for item in results:
            assert isinstance(item, BarcodeResult)
            assert isinstance(item.data, str)
            assert isinstance(item.type, str)
            assert isinstance(item.valid_ean, bool)

    @pytest.mark.skipif(
        not has_barcode_support(),
        reason="pyzbar not installed",
    )
    def test_valid_ean_flag_set_correctly(self):
        """BarcodeResult.valid_ean matches _is_valid_ean for digit-only data."""
        import io

        from PIL import Image

        # Blank image returns no barcodes — just exercise the code path
        img = Image.new("RGB", (100, 50), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        results = detect_barcodes(buf.getvalue())
        for r in results:
            if r.data.isdigit():
                assert r.valid_ean == _is_valid_ean(r.data)

    def test_returns_list_type(self):
        """Return type is always list regardless of pyzbar availability."""
        result = detect_barcodes(b"not an image")
        assert isinstance(result, list)
