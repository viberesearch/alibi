"""Tests for line-item pollution filtering (non-item lines leaking into items)."""

from alibi.extraction.text_parser import (
    _is_pollution_item,
    filter_pollution_items,
)


class TestIsPollutionItem:
    def test_greek_total_kapta(self) -> None:
        assert _is_pollution_item("KAPTA")  # ΚΑΡΤΑ (card)
        assert _is_pollution_item("ΣΥΝΟΛO")  # total, Latin-O OCR variant
        assert _is_pollution_item("ΣΥΝΟΛΟ")

    def test_vat_analysis_letter_rows(self) -> None:
        assert _is_pollution_item("B 5 % 1,06 20,91")
        assert _is_pollution_item("D 19 % 0,08 0,41")
        assert _is_pollution_item("E 0 % 0,00 8,84")

    def test_multiplier_only_lines(self) -> None:
        assert _is_pollution_item("2 x")
        assert _is_pollution_item("3 x")
        assert _is_pollution_item("2 x 4.79'")

    def test_footer_blob(self) -> None:
        blob = (
            "ΣΥΝΟΛO 1,13 30,16 Lidl Plus 0125 205207 31.03.26 13:15 "
            "VAT NO. 30010823A AID: A0000000031010 PURCHASE 2.85 1.66 6.97 0.45"
        )
        assert _is_pollution_item(blob)

    def test_empty_name(self) -> None:
        assert _is_pollution_item("")
        assert _is_pollution_item("   ")

    def test_restaurant_header_metadata(self) -> None:
        assert _is_pollution_item("THU APRIL CHECK #309109-1")
        assert _is_pollution_item("TABLE #12")
        assert _is_pollution_item("TEL/FAX: 4790290066")
        assert _is_pollution_item("Server: Maria")

    def test_hotel_not_matched_as_tel(self) -> None:
        # "HOTEL" must not trip the anchored tel/fax rule.
        assert not _is_pollution_item("HOTEL CHOCOLAT TRUFFLES")

    def test_real_items_kept(self) -> None:
        # Genuine items (including weighed and Greek-name) must NOT be filtered.
        assert not _is_pollution_item("MITSIDE FLOUR 1KG")
        assert not _is_pollution_item("ΦPEEKO ΓΑΛΑ IL/FR. MILK")
        assert not _is_pollution_item("KAPOTA 1KG/CARROTS")
        assert not _is_pollution_item("Coffee V")
        assert not _is_pollution_item("BARILLA PENNE GENOVESE 190G")


class TestFilterPollutionItems:
    def test_filters_pollution_keeps_items(self) -> None:
        items = [
            {"name": "MITSIDE FLOUR 1KG", "total_price": 1.48},
            {"name": "2 x", "total_price": 1.49},
            {"name": "ΣΥΝΟΛO", "total_price": 31.29},
            {"name": "KAPTA", "total_price": 31.29},
            {"name": "B 5 % 1,06 20,91", "total_price": 21.96},
            {"name": "ΦPEEKO ΓΑΛΑ", "total_price": 2.96},
        ]
        out = filter_pollution_items(items)
        names = [i["name"] for i in out]
        assert names == ["MITSIDE FLOUR 1KG", "ΦPEEKO ΓΑΛΑ"]

    def test_empty_list(self) -> None:
        assert filter_pollution_items([]) == []

    def test_malformed_non_dict_entries_dropped(self) -> None:
        # A malformed float/str/None in line_items must not raise.
        items = [
            {"name": "MILK 1L", "total_price": 1.2},
            3.14,
            None,
            "garbage",
            {"name": "2 x", "total_price": 1.0},
        ]
        out = filter_pollution_items(items)
        assert out == [{"name": "MILK 1L", "total_price": 1.2}]
