"""Ground truth data for real document extraction tests.

Each document has expected values established by Claude Opus 4.6 vision,
verified against the actual document images/PDFs. These represent the
"gold standard" for extraction accuracy comparison.
"""

import os
from decimal import Decimal

# ---------------------------------------------------------------------------
# Inbox path (configurable via env or fixture)
# ---------------------------------------------------------------------------
INBOX_DIR = os.environ.get("ALIBI_TEST_INBOX", "./tests/fixtures/inbox")

# ---------------------------------------------------------------------------
# IMG_0430 Medium.jpeg — FRESKO receipt (480x640, 2 items)
# ---------------------------------------------------------------------------
FRESKO_RECEIPT = {
    "file": "IMG_0430 Medium.jpeg",
    "doc_type": "receipt",
    "expected": {
        "vendor": "FRESKO",
        "vendor_address_contains": "KRANOU",
        "vendor_phone": "95772266",
        "date": "2026-02-17",
        "time": "12:30:47",
        "total": Decimal("2.75"),
        "currency": "EUR",
        "line_item_count": 2,
        "line_items": [
            {
                "name_contains": "Soft Facial",
                "quantity": 1,
                "total_price": Decimal("1.50"),
            },
            {
                "name_contains": "Sugar",
                "quantity": 1,
                "total_price": Decimal("1.25"),
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# IMG_0429 Medium.jpeg — ARAB BUTCHERY payment (480x640)
# ---------------------------------------------------------------------------
ARAB_BUTCHERY_PAYMENT = {
    "file": "IMG_0429 Medium.jpeg",
    "doc_type": "payment_confirmation",
    "expected": {
        "vendor": "ARAB BUTCHERY",
        "vendor_address_contains": "KRANOU",
        "date": "2026-02-17",
        "time": "12:38:37",
        "total": Decimal("29.00"),
        "currency": "EUR",
        "card_type": "visa",
        "card_last4": "7201",
        "authorization_code": "273428",
    },
}

# ---------------------------------------------------------------------------
# IMG_0432 Medium.jpeg — PLUS DISCOUNT MARKET receipt (480x640, 7 items)
# ---------------------------------------------------------------------------
PLUS_DISCOUNT_RECEIPT = {
    "file": "IMG_0432 Medium.jpeg",
    "doc_type": "receipt",
    "expected": {
        "vendor": "PLUS DISCOUNT MARKET",
        "vendor_address_contains": "GERMASOGEIA",
        "date": "2026-02-17",
        "time": "12:26:30",
        "total": Decimal("15.75"),
        "currency": "EUR",
        "payment_method": "card",
        "line_item_count_min": 5,
        "line_items": [
            {"name_contains": "PATATES", "total_price": Decimal("1.40")},
            {"name_contains": "LEMON", "total_price": Decimal("0.55")},
            {"name_contains": "Red Bull", "total_price": Decimal("5.40")},
        ],
    },
}

# ---------------------------------------------------------------------------
# receipt.jpeg — Maleve Trading LTD (222x640, 8 items)
# ---------------------------------------------------------------------------
MALEVE_RECEIPT = {
    "file": "receipt.jpeg",
    "doc_type": "receipt",
    "expected": {
        "vendor": "Maleve Trading LTD",
        "vendor_address_contains": "Profiti Elia",
        "vendor_phone": "25322957",
        "date": "2026-02-12",
        "time": "8:35:55",
        "total": Decimal("33.03"),
        "currency": "EUR",
        "payment_method": "card",
        "line_item_count": 8,
        "line_items": [
            {
                "name_contains": "Blueberries",
                "quantity": 1,
                "total_price": Decimal("4.09"),
            },
            {
                "name_contains": "Yogurt",
                "quantity": 2,
                "total_price": Decimal("9.18"),
            },
            {
                "name_contains": "Red Bull",
                "quantity": 1,
                "total_price": Decimal("1.09"),
            },
            {
                "name_contains": "Barilla",
                "quantity": 1,
                "total_price": Decimal("1.95"),
            },
            {
                "name_contains": "Avocado",
                "total_price": Decimal("3.97"),
            },
            {
                "name_contains": "St George",
                "total_price": Decimal("7.79"),
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# IMG_0436 Large.jpeg — PAPAS HYPERMARKET (278x1280, 23 items, tall)
# ---------------------------------------------------------------------------
PAPAS_RECEIPT = {
    "file": "IMG_0436 Large.jpeg",
    "doc_type": "receipt",
    "expected": {
        "vendor": "PAPAS HYPERMARKET",
        "vendor_address_contains": "LIMASSOL",
        "vendor_phone": "25574100",
        "date": "2026-01-21",
        "time": "13:56:11",
        "total": Decimal("85.69"),
        "currency": "EUR",
        "payment_method": "card",
        "card_last4": "7201",
        "line_item_count": 21,  # 23 items minus 2 merged (2 TOPPITS)
        "line_items": [
            {"name_contains": "GAEA", "total_price": Decimal("2.99")},
            {"name_contains": "RITTER DARK", "total_price": Decimal("3.59")},
            {
                "name_contains": "RED BULL",
                "quantity": 3,
                "total_price": Decimal("5.97"),
            },
            {
                "name_contains": "APPLES GRANNY",
                "total_price": Decimal("4.01"),
            },
            {
                "name_contains": "BEEF LIVER",
                "total_price": Decimal("7.79"),
            },
            {
                "name_contains": "CHICKEN",
                "total_price": Decimal("8.30"),
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# IMG_0435.jpeg — Blue Island PLC invoice (4032x3024, landscape photo)
# ---------------------------------------------------------------------------
BLUE_ISLAND_PHOTO = {
    "file": "IMG_0435.jpeg",
    "doc_type": "invoice",
    "expected": {
        "issuer": "Blue Island",
        "total_or_amount": Decimal("37.79"),
        "currency": "EUR",
        "line_item_count_min": 2,
    },
}

# ---------------------------------------------------------------------------
# IMG_0434.pdf — Blue Island PLC invoice (2 pages)
# ---------------------------------------------------------------------------
BLUE_ISLAND_PDF = {
    "file": "IMG_0434.pdf",
    "doc_type": "invoice",
    "expected": {
        "issuer": "Blue Island PLC",
        "invoice_number": "3115579",
        "issue_date": "2026-01-12",
        "customer_contains": "WOLT",
        "amount": Decimal("45.62"),
        "currency": "EUR",
        "payment_terms_contains": "Cash",
        "line_item_count": 5,
        "line_items": [
            {
                "name_contains": "Salmo Salar",
                "total_price": Decimal("8.66"),
            },
            {
                "name_contains": "Loligo",
                "quantity": Decimal("0.526"),
                "total_price": Decimal("10.01"),
            },
            {
                "name_contains": "Octopus",
                "quantity": Decimal("1.028"),
                "total_price": Decimal("18.59"),
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# bank_account_statement.pdf — Eurobank statement
# ---------------------------------------------------------------------------
BANK_STATEMENT = {
    "file": "bank_account_statement.pdf",
    "doc_type": "statement",
    "expected": {
        "institution": "EUROBANK",
        "account_number": "245-10-519031-00",
        "iban": "CY02005002450002451051903100",
        "currency": "EUR",
        "account_type": "Savings Account",
        "period_start": "2026-02-02",
        "period_end": "2026-02-17",
        "transaction_count": 8,
        "transactions": [
            {
                "date": "2026-02-02",
                "description_contains": "DEA BEAUTY",
                "amount": Decimal("-170.00"),
            },
            {
                "date": "2026-02-05",
                "description_contains": "VIVASAN",
                "amount": Decimal("-100.00"),
            },
            {
                "date": "2026-02-05",
                "description_contains": "ATM ERB 2183",
                "amount": Decimal("-500.00"),
            },
            {
                "date": "2026-02-05",
                "description_contains": "ATM ERB 2184",
                "amount": Decimal("-240.00"),
            },
            {
                "date": "2026-02-06",
                "description_contains": "ZHARNIKOV",
                "amount": Decimal("-100.00"),
            },
            {
                "date": "2026-02-06",
                "description_contains": "APPLE",
                "amount": Decimal("-21.95"),
            },
            {
                "date": "2026-02-06",
                "description_contains": "MINI MARKET",
                "amount": Decimal("-19.68"),
            },
            {
                "date": "2026-02-17",
                "description_contains": "RYANAIR",
                "amount": Decimal("-3.75"),
            },
        ],
        "total_debit": Decimal("1155.38"),
    },
}

# ---------------------------------------------------------------------------
# IMG_0427 Medium.jpeg — THE NUT CRACKER HOUSE JCC payment slip
# ---------------------------------------------------------------------------
NUT_CRACKER_PAYMENT = {
    "file": "IMG_0427 Medium.jpeg",
    "doc_type": "payment_confirmation",
    "expected": {
        "vendor": "THE NUT CRACKER HOUSE",
        "date": "2026-02-17",
        "time": "12:12:54",
        "total": Decimal("12.45"),
        "currency": "EUR",
        "card_type": "visa",
        "card_last4": "7201",
    },
}

# ---------------------------------------------------------------------------
# IMG_0428 Large.jpeg — TheNutCrackerHouse receipt (498x1280, tall)
# ---------------------------------------------------------------------------
NUT_CRACKER_RECEIPT = {
    "file": "IMG_0428 Large.jpeg",
    "doc_type": "receipt",
    "expected": {
        "vendor": "TheNutCrackerHouse",
        "vendor_phone": "25321010",
        "date": "2026-02-17",
        "total": Decimal("12.45"),
        "currency": "EUR",
        "payment_method": "card",
        "line_item_count": 1,
        "line_items": [
            {
                "name_contains": "THYME HONEY",
                "quantity": 1,
                "total_price": Decimal("12.45"),
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# IMG_0431 Medium.jpeg — FreSko BUTANOLO LTD payment (Worldline slip)
# ---------------------------------------------------------------------------
FRESKO_PAYMENT = {
    "file": "IMG_0431 Medium.jpeg",
    "doc_type": "payment_confirmation",
    "expected": {
        "vendor_contains": "FreSko",
        "date": "2026-02-17",
        "time": "12:30:14",
        "total": Decimal("2.75"),
        "currency": "EUR",
        "card_type": "visa",
        "card_last4": "7201",
    },
}

# ---------------------------------------------------------------------------
# transaction_confirmation.pdf — Eurobank MINI MARKET payment
# ---------------------------------------------------------------------------
MINI_MARKET_CONFIRMATION = {
    "file": "transaction_confirmation.pdf",
    "doc_type": "payment_confirmation",
    "expected": {
        "vendor_contains": "MINI MARKET",
        "date": "2026-02-06",
        "total": Decimal("19.68"),
        "currency": "EUR",
    },
}

# ---------------------------------------------------------------------------
# Ollama extraction comparison — known errors to track improvement
# ---------------------------------------------------------------------------
OLLAMA_KNOWN_ERRORS = {
    "IMG_0429 Medium.jpeg": {
        "date_wrong": "2023-03-13 (should be 2026-02-17)",
        "vendor_vat_garbled": "Parsed 'VAT NO. 17/02/2026' as registration",
    },
    "IMG_0432 Medium.jpeg": {
        "date_wrong": "2026-02-07 (should be 2026-02-17)",
        "address_wrong": "GEBMASIA (should be GERMASOGEIA)",
        "weighed_items_wrong": "Quantities/prices incorrect for weighed goods",
    },
    "receipt.jpeg": {
        "year_wrong": "2023 (should be 2026)",
        "address_garbled": "'Profin Elia 3A, Tel Aviv' (should be 'Profiti Elia 3A')",
        "phone_wrong": "25223957 (should be 25322957)",
        "avocado_qty_wrong": "1 (should be 1.1)",
    },
    "IMG_0430 Medium.jpeg": {
        "phone_wrong": "95777266 (should be 95772266)",
        "tax_rates_swapped": "Soft Facial T at 5% (should be 19%)",
        "payment_wrong": "cash (should be card — PAID SIX)",
    },
    "IMG_0435.jpeg": {
        "everything_garbled": "Greek text grossly misread from landscape photo",
        "type_wrong": "receipt (should be invoice)",
        "date_wrong": "2023-12-04",
    },
}

# ---------------------------------------------------------------------------
# German receipt sample (synthetic OCR text for parser testing)
# ---------------------------------------------------------------------------
GERMAN_RECEIPT_OCR = """REWE Markt GmbH
Domstr. 20
50668 Köln

Tel. 0221/1490
USt-IdNr. DE812706034

Datum: 15.02.2026 14:23:15
Bon-Nr. 4521

WEISSBROT 500G          1,29 B
BIO VOLLMILCH 1L        1,49 A
BANANEN                  1,89 B
  0,750 kg x 2,49 EUR/kg
EDEKA BUTTER 250G       2,19 A
MINERALWASSER 1,5L       0,69 B
  Pfand                  0,25

Summe                    7,80
  davon MwSt. A 19,0%   0,47
  davon MwSt. B  7,0%   0,31
Netto A                  2,48
Netto B                  4,54

GESAMT                EUR 7,80
EC-Karte                  7,80
Kartennummer: **** **** **** 3456

MwSt.   Netto    Steuer  Brutto
A 19,0%   2,48     0,47    2,95
B  7,0%   4,54     0,31    4,85

Vielen Dank für Ihren Einkauf!
"""

GERMAN_RECEIPT_EXPECTED = {
    "vendor": "REWE Markt GmbH",
    "vendor_address_contains": "Köln",
    "date": "2026-02-15",
    "time": "14:23:15",
    "total": Decimal("7.80"),
    "currency": "EUR",
    "payment_method": "card",
    "line_item_count_min": 4,
}

# ---------------------------------------------------------------------------
# Russian receipt sample (synthetic OCR text for parser testing)
# ---------------------------------------------------------------------------
RUSSIAN_RECEIPT_OCR = """ООО "ПЕРЕКРЁСТОК"
ИНН 7728029110
г. Москва, ул. Тверская, д. 15

Дата: 18.02.2026 10:45:32
Чек № 00142

ХЛЕБ БОРОДИНСКИЙ      89,90 А
МОЛОКО 3.2% 1Л       109,90 А
БАНАНЫ                 85,50 Б
  0,950 кг x 89,99 руб/кг
МАСЛО СЛИВОЧНОЕ 200Г  159,90 А
ВОДА МИН. 1.5Л         49,90 Б

ИТОГО                  495,10
  в т.ч. НДС 20%       58,78
  в т.ч. НДС 10%       12,31

ВСЕГО К ОПЛАТЕ    RUB 495,10
БАНКОВСКАЯ КАРТА       495,10
Карта: **** 7890

НДС        Без НДС   НДС     Итого
А 20,0%    293,92    58,78   352,70
Б 10,0%    123,09    12,31   135,40

СПАСИБО ЗА ПОКУПКУ!
"""

RUSSIAN_RECEIPT_EXPECTED = {
    "vendor_contains": "ПЕРЕКРЁСТОК",
    "date": "2026-02-18",
    "time": "10:45:32",
    "total": Decimal("495.10"),
    "currency": "RUB",
    "payment_method": "card",
    "card_last4": "7890",
    "line_item_count_min": 4,
}

# ---------------------------------------------------------------------------
# Greek receipt sample (for completeness — existing language)
# ---------------------------------------------------------------------------
GREEK_RECEIPT_OCR = """ΑΛΦΑ ΜΕΓΑ
ΑΦΜ: 12345678Α
ΛΕΜΕΣΟΣ

Ημερομηνία: 19/02/2026 09:15:22
Αρ. Απόδειξης: 7821

ΨΩΜΙ ΧΩΡΙΑΤΙΚΟ        1,50 Α
ΓΑΛΑ ΦΡΕΣΚΟ 1Λ        2,10 Α
ΜΠΑΝΑΝΕΣ               2,35 Α
  1,200 κιλά x 1,96 ανά κιλό

ΜΕΡΙΚΟ ΣΥΝΟΛΟ          5,95
ΦΠΑ 5%                 0,28
ΣΥΝΟΛΟ             EUR 5,95
VISA                    5,95

Ευχαριστούμε!
"""

GREEK_RECEIPT_EXPECTED = {
    "vendor_contains": "ΑΛΦΑ",
    "date": "2026-02-19",
    "time": "09:15:22",
    "total": Decimal("5.95"),
    "currency": "EUR",
    "payment_method": "card",
    "line_item_count_min": 2,
}

# ---------------------------------------------------------------------------
# All real documents for iteration
# ---------------------------------------------------------------------------
ALL_REAL_DOCUMENTS = [
    FRESKO_RECEIPT,
    ARAB_BUTCHERY_PAYMENT,
    PLUS_DISCOUNT_RECEIPT,
    MALEVE_RECEIPT,
    PAPAS_RECEIPT,
    NUT_CRACKER_PAYMENT,
    NUT_CRACKER_RECEIPT,
    FRESKO_PAYMENT,
    MINI_MARKET_CONFIRMATION,
    # PDF and photo documents have different pipeline paths:
    BLUE_ISLAND_PDF,
    BLUE_ISLAND_PHOTO,
    BANK_STATEMENT,
]
