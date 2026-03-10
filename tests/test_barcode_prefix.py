"""Tests for barcode-prefix item merging in _extract_line_items().

Alphamega and similar POS receipts print each item as two lines:
  Barcode: XXXXXXXXXXXXX
  PRODUCT NAME qty unit price tax_code

The parser should merge these into a single item with the product name
and the barcode extracted into the barcode field.
"""

from alibi.extraction.text_parser import parse_ocr_text


def _make_receipt(item_lines: str) -> str:
    """Wrap item lines in a minimal receipt template."""
    return (
        "ALPHAMEGA\n"
        "C.A. PAPAELLINAS EMPORIKI LTD\n"
        "Date: 15/01/26 11:24\n"
        "Description Amount\n"
        f"{item_lines}\n"
        "Total EURO 10.00\n"
    )


def test_barcode_prefix_simple():
    """Barcode line + product name line -> single item with barcode."""
    text = _make_receipt("Barcode: 5290036000111\n" "CHARAL/CHRISTIS STRA ea 3.99 D")
    result = parse_ocr_text(text, "receipt")
    items = result.data.get("line_items", [])
    names = [it["name"] for it in items]
    assert "CHARAL/CHRISTIS STRA ea" in names
    matched = [it for it in items if it["name"] == "CHARAL/CHRISTIS STRA ea"][0]
    assert matched["barcode"] == "5290036000111"
    assert matched["total_price"] == 3.99


def test_barcode_prefix_multiple_items():
    """Multiple barcode-prefixed items in sequence."""
    text = _make_receipt(
        "Barcode: 2001379000019\n"
        "LOLLO ROSSO/BIONDO C ea 1.25 A\n"
        "Barcode: 5290036000111\n"
        "CHARAL/CHRISTIS STRA ea 3.99 D\n"
        "Barcode: 5029053038759\n"
        "KLEENEX FACIAL EMOTI 1.89 C"
    )
    result = parse_ocr_text(text, "receipt")
    items = result.data.get("line_items", [])
    names = [it["name"] for it in items]
    assert "LOLLO ROSSO/BIONDO C ea" in names
    assert "CHARAL/CHRISTIS STRA ea" in names
    assert "KLEENEX FACIAL EMOTI" in names
    # All should have barcodes extracted
    for it in items:
        if it["name"] in (
            "LOLLO ROSSO/BIONDO C ea",
            "CHARAL/CHRISTIS STRA ea",
            "KLEENEX FACIAL EMOTI",
        ):
            assert it["barcode"] is not None


def test_barcode_prefix_with_embedded_qty():
    """Product line with embedded qty: 0.408 pce$9.99 4.08 D."""
    text = _make_receipt(
        "Barcode: 2106683004088\n" "IFANTIS HUMMUS CARAM 0.408 pce$9.99 4.08 D"
    )
    result = parse_ocr_text(text, "receipt")
    items = result.data.get("line_items", [])
    matched = [it for it in items if it["barcode"] == "2106683004088"]
    assert len(matched) == 1
    assert matched[0]["total_price"] == 4.08
    assert "IFANTIS" in matched[0]["name"]


def test_barcode_not_appended_to_previous_item():
    """Barcode line should NOT become continuation of previous item."""
    text = _make_receipt(
        "LOLLO ROSSO/BIONDO C ea 1.25 A\n"
        "Barcode: 5290036000111\n"
        "CHARAL/CHRISTIS STRA ea 3.99 D"
    )
    result = parse_ocr_text(text, "receipt")
    items = result.data.get("line_items", [])
    lollo = [it for it in items if "LOLLO" in it["name"]]
    assert len(lollo) == 1
    # Barcode should NOT be appended to LOLLO ROSSO name
    assert "Barcode" not in lollo[0]["name"]
    assert "5290036000111" not in lollo[0]["name"]


def test_barcode_prefix_no_price_on_next_line():
    """Barcode line followed by non-priced line: no merge, skip barcode."""
    text = _make_receipt(
        "Barcode: 9999999999999\n"
        "Some description without price\n"
        "ACTUAL ITEM 2.50 D"
    )
    result = parse_ocr_text(text, "receipt")
    items = result.data.get("line_items", [])
    # The barcode line shouldn't create a barcode-named item
    barcode_items = [it for it in items if "9999999999999" in str(it.get("name", ""))]
    assert len(barcode_items) == 0


def test_mixed_barcode_and_regular_items():
    """Mix of barcode-prefixed and regular items."""
    text = _make_receipt(
        "Barcode: 5290036000111\n"
        "CHARAL/CHRISTIS STRA ea 3.99 D\n"
        "FROM 9.99 TO 6.49 -1.43\n"
        "REGULAR ITEM NO BARCODE 2.50 D"
    )
    result = parse_ocr_text(text, "receipt")
    items = result.data.get("line_items", [])
    names = [it["name"] for it in items]
    # Barcode-prefixed item should have barcode
    charal = [it for it in items if "CHARAL" in it["name"]]
    assert charal[0]["barcode"] == "5290036000111"
    # Regular item should not have barcode
    regular = [it for it in items if "REGULAR" in it["name"]]
    assert len(regular) == 1
    assert regular[0]["barcode"] is None


def test_barcode_case_insensitive():
    """BARCODE: prefix (uppercase) should also match."""
    text = _make_receipt("BARCODE: 5290036000111\n" "CHARAL/CHRISTIS STRA ea 3.99 D")
    result = parse_ocr_text(text, "receipt")
    items = result.data.get("line_items", [])
    matched = [it for it in items if it.get("barcode") == "5290036000111"]
    assert len(matched) == 1


def test_barcode_three_line_pattern():
    """3-line: Barcode + name (no price) + standalone price on next line."""
    text = _make_receipt(
        "Barcode: 4750072612354\n" "BALTAIS TVOROG 9% 20\n" "2 ea$3.05 6.10 D"
    )
    result = parse_ocr_text(text, "receipt")
    items = result.data.get("line_items", [])
    matched = [it for it in items if "BALTAIS" in it["name"]]
    assert len(matched) == 1
    assert matched[0]["barcode"] == "4750072612354"
    assert matched[0]["total_price"] == 6.10


def test_barcode_three_line_short_barcode():
    """3-line pattern with short (8-digit) barcode."""
    text = _make_receipt("Barcode: 52905407\n" "CARLTONA SODA 175 g ea 1.15 D")
    result = parse_ocr_text(text, "receipt")
    items = result.data.get("line_items", [])
    matched = [it for it in items if "CARLTONA" in it["name"]]
    assert len(matched) == 1
    assert matched[0]["barcode"] == "52905407"


def test_barcode_three_line_mixed():
    """Mix of 2-line and 3-line barcode patterns."""
    text = _make_receipt(
        "Barcode: 5290036000111\n"
        "CHARAL/CHRISTIS STRA ea 3.99 D\n"
        "Barcode: 4750072612354\n"
        "BALTAIS TVOROG 9% 20\n"
        "2 ea$3.05 6.10 D\n"
        "Barcode: 52905407\n"
        "CARLTONA SODA 175 g ea 1.15 D"
    )
    result = parse_ocr_text(text, "receipt")
    items = result.data.get("line_items", [])
    charal = [it for it in items if "CHARAL" in it["name"]]
    baltais = [it for it in items if "BALTAIS" in it["name"]]
    carltona = [it for it in items if "CARLTONA" in it["name"]]
    assert charal[0]["barcode"] == "5290036000111"
    assert baltais[0]["barcode"] == "4750072612354"
    assert carltona[0]["barcode"] == "52905407"
