"""Tests for jurisdiction inference and country-aware currency resolution."""

from alibi.normalizers.jurisdiction import (
    apply_jurisdiction,
    infer_jurisdiction,
    resolve_currency,
)


class TestInferJurisdiction:
    def test_canada_from_place_and_hst(self) -> None:
        ext = {
            "vendor": "University of Guelph",
            "raw_text": "Coffee 3.45\nHST 5% $\nHST 8% $\nTotal 9.16",
        }
        assert infer_jurisdiction(ext) == "CA"

    def test_northern_cyprus_from_place(self) -> None:
        ext = {
            "vendor_address": "Lefkoşa, KKTC",
            "raw_text": "KDV %18  Toplam 250.00 TL",
        }
        assert infer_jurisdiction(ext) == "CY-NORTH"

    def test_northern_cyprus_girne(self) -> None:
        ext = {"vendor_address": "Girne / Kyrenia", "raw_text": "Toplam 120 TL"}
        assert infer_jurisdiction(ext) == "CY-NORTH"

    def test_austria_from_place(self) -> None:
        ext = {"vendor_address": "Wien, Österreich", "raw_text": "MwSt 20% EUR"}
        assert infer_jurisdiction(ext) == "AT"

    def test_austria_from_atu_vat(self) -> None:
        ext = {"vendor_vat": "ATU12345678", "raw_text": "Rechnung"}
        assert infer_jurisdiction(ext) == "AT"

    def test_cyprus_from_place(self) -> None:
        ext = {"vendor_address": "Limassol 4047", "raw_text": "ΦΠΑ 19%"}
        assert infer_jurisdiction(ext) == "CY"

    def test_cyprus_from_greek_place(self) -> None:
        ext = {"raw_text": "ΛΕΜΕΣΟΣ ΦΠΑ 19% ΣΥΝΟΛΟ 12.00"}
        assert infer_jurisdiction(ext) == "CY"

    def test_cyprus_from_fpa_vat_marker_only(self) -> None:
        # No city, but ΦΠΑ present (every Cyprus receipt has it).
        ext = {"vendor": "ALPHAMEGA", "raw_text": "Description Amount ΦΠΑ 19% 14.00"}
        assert infer_jurisdiction(ext) == "CY"

    def test_turkey_mainland(self) -> None:
        ext = {"vendor_address": "Istanbul", "raw_text": "KDV %8 TL"}
        assert infer_jurisdiction(ext) == "TR"

    def test_canada_dry_does_not_trigger_canada(self) -> None:
        # A Cyprus restaurant selling "Canada Dry" must stay Cyprus.
        ext = {
            "vendor": "PARKLANE RESORT",
            "vendor_address": "Limassol",
            "raw_text": "CANADA DRY 3.50  ΦΠΑ 19%  Limassol",
        }
        assert infer_jurisdiction(ext) == "CY"

    def test_gst_as_guests_count_not_canada(self) -> None:
        # "GST 2" on a Cyprus restaurant slip means GUESTS, not Canadian tax.
        ext = {
            "vendor": "PARKLANE",
            "vendor_address": "11 Yianni Kranidioti, 4534 Limassol",
            "raw_text": "11086 TBL 42/1 GST 2  STEAK 44.00",
        }
        assert infer_jurisdiction(ext) == "CY"

    def test_real_canada_hst_still_detected(self) -> None:
        ext = {"vendor": "Tim Hortons", "raw_text": "Coffee 2.00 HST 13% Ontario"}
        assert infer_jurisdiction(ext) == "CA"

    def test_bare_canada_address_still_detected(self) -> None:
        ext = {"vendor_address": "123 Main St, Canada", "raw_text": "Total 5.00"}
        assert infer_jurisdiction(ext) == "CA"

    def test_smoked_turkey_product_does_not_trigger_turkey(self) -> None:
        # A Cyprus supermarket selling smoked turkey must stay Cyprus, not TR.
        ext = {
            "vendor": "ALPHAMEGA",
            "raw_text": "GRIDOUJ SMOKED TURKEY SLC 20 4.99  ΦΠΑ 19% Limassol",
        }
        assert infer_jurisdiction(ext) == "CY"

    def test_adana_kebab_does_not_trigger_turkey(self) -> None:
        ext = {
            "vendor": "Cyprus Grill",
            "raw_text": "Adana Kebab 12.50  ΦΠΑ 9% Nicosia",
        }
        assert infer_jurisdiction(ext) == "CY"

    def test_bare_tl_token_is_not_a_currency_cue(self) -> None:
        # A stray "TL" with no adjacent amount must not flip a Cyprus doc to TR.
        ext = {"vendor": "PAPAS", "raw_text": "ITEM TL CODE 4.99 EUR Limassol"}
        assert infer_jurisdiction(ext) == "CY"

    def test_tl_next_to_amount_is_a_lira_cue(self) -> None:
        ext = {"vendor": "Burhan Restaurant", "raw_text": "Toplam 250 TL"}
        assert infer_jurisdiction(ext) == "TR"

    def test_north_cyprus_village_token(self) -> None:
        ext = {"vendor_address": "Kaplıca", "raw_text": "Toplam 120 TL"}
        assert infer_jurisdiction(ext) == "CY-NORTH"

    def test_greek_script_fallback_to_cyprus(self) -> None:
        # Greek-text receipt with no city/ΦΠΑ still resolves to Cyprus.
        ext = {"vendor": "ΧΑΤΖΗΑΝΤΩΝΗΣ", "raw_text": "ΓΑΛΑ ΨΩΜΙ ΤΥΡΙ ΣΥΝΟΛΟ 8.40"}
        assert infer_jurisdiction(ext) == "CY"

    def test_no_signal_returns_none(self) -> None:
        ext = {"vendor": "Generic Shop", "raw_text": "Item 1.00 Total 1.00"}
        assert infer_jurisdiction(ext) is None

    def test_russia_from_cyrillic_place(self) -> None:
        ext = {"vendor_address": "Москва, ул. Тверская", "raw_text": "Молоко 80 ИТОГО"}
        assert infer_jurisdiction(ext) == "RU"

    def test_russia_from_rouble_and_nds(self) -> None:
        ext = {"vendor": "Пятёрочка", "raw_text": "ИТОГО 540 ₽  НДС 20%"}
        assert infer_jurisdiction(ext) == "RU"

    def test_russia_from_cyrillic_script_fallback(self) -> None:
        # Substantial Cyrillic, no place/currency token.
        ext = {"vendor": "Магазин", "raw_text": "Хлеб Масло Сыр Колбаса Чай"}
        assert infer_jurisdiction(ext) == "RU"

    def test_russia_latin_place(self) -> None:
        ext = {"vendor_address": "Moscow, Russia", "raw_text": "Total 540 RUB"}
        assert infer_jurisdiction(ext) == "RU"

    def test_spice_rub_word_does_not_trigger_russia(self) -> None:
        # A "DRY RUB" product with no Russian signal must not become RU.
        ext = {"vendor": "BBQ Shop", "raw_text": "DRY RUB SEASONING 4.99 Total 4.99"}
        assert infer_jurisdiction(ext) is None

    def test_north_cyprus_wins_over_turkey_tokens(self) -> None:
        # Both a north-Cyprus place and Turkish lira present.
        ext = {"vendor_address": "Gazimağusa, KKTC", "raw_text": "KDV TL Istanbul"}
        assert infer_jurisdiction(ext) == "CY-NORTH"


class TestResolveCurrency:
    def test_canada_dollar_becomes_cad_not_usd(self) -> None:
        ext = {"currency": "USD", "raw_text": "HST 5% $"}
        assert resolve_currency(ext, "CA") == "CAD"

    def test_northern_cyprus_multicurrency_is_try(self) -> None:
        # Receipt prints all three; canonical is the Lira.
        ext = {
            "currency": "EUR",
            "raw_text": "Toplam: 250 TL / 7.50 EUR / 8.00 USD  KDV %18",
        }
        assert resolve_currency(ext, "CY-NORTH") == "TRY"

    def test_austria_missing_currency_fills_eur(self) -> None:
        ext = {"raw_text": "MwSt 20%"}
        assert resolve_currency(ext, "AT") == "EUR"

    def test_explicit_gbp_kept(self) -> None:
        ext = {"currency": "GBP", "raw_text": "VAT 20%"}
        assert resolve_currency(ext, "GB") == "GBP"

    def test_unknown_jurisdiction_keeps_extracted(self) -> None:
        ext = {"currency": "EUR", "raw_text": "Total 5.00"}
        assert resolve_currency(ext, None) == "EUR"

    def test_russia_resolves_to_rub(self) -> None:
        ext = {"raw_text": "ИТОГО 540 ₽"}
        assert resolve_currency(ext, "RU") == "RUB"

    def test_rouble_cue_without_jurisdiction(self) -> None:
        ext = {"currency": "", "raw_text": "Сумма 540 ₽"}
        assert resolve_currency(ext, None) == "RUB"

    def test_printed_eur_overrides_cyrillic_inferred_rub(self) -> None:
        # A Russian-language venue that prices in EUR: the Cyrillic room name
        # forces RU jurisdiction, but the printed total is explicitly EUR with
        # no rouble cue anywhere -> the printed currency must win.
        ext = {
            "vendor": "LITTLE SINS COFFEE BAR",
            "currency": "RUB",
            "raw_text": "Table # 1 (Основной зал)\nTotal 3.00 eur\nCard 3.00 eur",
        }
        assert resolve_currency(ext, "RU") == "EUR"

    def test_printed_guard_keeps_rub_when_rouble_cue_present(self) -> None:
        # Both EUR and a rouble cue printed -> multi-currency, jurisdiction wins.
        ext = {"currency": "EUR", "raw_text": "Итого 540 ₽  (≈ 5 eur)  НДС 20%"}
        assert resolve_currency(ext, "RU") == "RUB"


class TestApplyJurisdiction:
    def test_canada_sets_country_and_corrects_currency(self) -> None:
        ext = {
            "vendor": "University of Guelph",
            "currency": "USD",
            "raw_text": "Coffee 3.45 HST 5% $ Total 9.16",
        }
        apply_jurisdiction(ext)
        assert ext["country"] == "CA"
        assert ext["currency"] == "CAD"

    def test_northern_cyprus(self) -> None:
        ext = {
            "vendor_address": "Lefkoşa KKTC",
            "currency": "EUR",
            "raw_text": "Toplam 250 TL KDV %18",
        }
        apply_jurisdiction(ext)
        assert ext["country"] == "CY-NORTH"
        assert ext["currency"] == "TRY"

    def test_cyrillic_language_with_eur_leaves_country_unknown(self) -> None:
        # Russian-speaking venue pricing in EUR: language is not a country.
        # Don't fabricate RU; leave country unknown and honour the printed EUR.
        ext = {
            "vendor": "LITTLE SINS COFFEE BAR",
            "currency": "RUB",
            "raw_text": "Table # 1 (Основной зал)\nTotal 3.00 eur\nCard 3.00 eur",
        }
        apply_jurisdiction(ext)
        assert ext.get("country") is None
        assert ext["currency"] == "EUR"

    def test_russian_place_name_pins_country_even_with_eur(self) -> None:
        # A real Russian place name pins RU; an EUR invoice keeps EUR currency.
        ext = {
            "currency": "RUB",
            "raw_text": "ООО Ромашка  Санкт-Петербург  Total 100.00 eur",
        }
        apply_jurisdiction(ext)
        assert ext["country"] == "RU"
        assert ext["currency"] == "EUR"

    def test_default_country_does_not_override_currency(self) -> None:
        # No jurisdiction signal: fall back to default country, keep currency.
        ext = {"currency": "EUR", "raw_text": "Item 1.00 Total 1.00"}
        apply_jurisdiction(ext, default_country="CY")
        assert ext["country"] == "CY"
        assert ext["currency"] == "EUR"

    def test_cyprus_eur_unchanged(self) -> None:
        ext = {
            "vendor_address": "Limassol",
            "currency": "EUR",
            "raw_text": "ΦΠΑ 19% ΣΥΝΟΛΟ 12.00",
        }
        apply_jurisdiction(ext)
        assert ext["country"] == "CY"
        assert ext["currency"] == "EUR"

    def test_empty_dict_safe(self) -> None:
        ext: dict = {}
        apply_jurisdiction(ext)
        assert ext == {}
