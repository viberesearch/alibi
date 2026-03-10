"""Tests for canonical vendor_key resolution from identity members."""

from pathlib import Path
from unittest.mock import patch

import pytest

from alibi.db.connection import DatabaseManager
from alibi.db.models import DocumentType
from alibi.db import v2_store
from alibi.identities.matching import (
    _pick_canonical_vat,
    get_canonical_vendor_key,
)
from alibi.identities.store import add_member, create_identity


class TestPickCanonicalVat:
    """Unit tests for _pick_canonical_vat."""

    def test_single_value(self):
        """Single value returned as-is."""
        assert _pick_canonical_vat(["101800201"]) == "101800201"

    def test_prefers_letter_ending(self):
        """Prefers Cyprus-format VAT ending with letter."""
        assert _pick_canonical_vat(["101800201", "10180201N"]) == "10180201N"

    def test_prefers_letter_ending_reversed(self):
        """Order of input doesn't matter."""
        assert _pick_canonical_vat(["10180201N", "101800201"]) == "10180201N"

    def test_deterministic_tie_break(self):
        """When multiple letter-ending values, picks shortest then alphabetical."""
        result = _pick_canonical_vat(["10180201N", "10180201X"])
        assert result == "10180201N"  # alphabetical tie-break

    def test_no_letter_ending_picks_shortest(self):
        """Without letter-ending, picks shortest then alphabetical."""
        result = _pick_canonical_vat(["101800201", "10180201"])
        assert result == "10180201"

    def test_country_prefix_same_number(self):
        """CY10180201N and 10180201N are the same — prefer prefixed."""
        assert _pick_canonical_vat(["10180201N", "CY10180201N"]) == "CY10180201N"

    def test_country_prefix_reversed(self):
        """Order of input doesn't matter for prefix detection."""
        assert _pick_canonical_vat(["CY10180201N", "10180201N"]) == "CY10180201N"

    def test_country_prefix_with_ocr_error(self):
        """Prefixed + OCR error: pick prefixed letter-ending."""
        result = _pick_canonical_vat(["CY10180201N", "101800201"])
        assert result == "CY10180201N"

    def test_country_prefix_all_three_variants(self):
        """Prefixed + bare + OCR error: group then pick best."""
        result = _pick_canonical_vat(["101800201", "10180201N", "CY10180201N"])
        # CY10180201N and 10180201N group together, prefer CY-prefixed
        # Then CY10180201N vs 101800201: letter-ending wins
        assert result == "CY10180201N"

    def test_german_vat_prefix(self):
        """DE prefix recognized as EU country code."""
        assert _pick_canonical_vat(["123456789", "DE123456789"]) == "DE123456789"

    def test_non_eu_prefix_not_stripped(self):
        """Non-EU 2-letter prefix treated as part of the number."""
        # "US" is not an EU country code, so US12345 and 12345 are different
        result = _pick_canonical_vat(["US12345N", "12345N"])
        assert result == "12345N"  # shorter bare number


class TestGetCanonicalVendorKey:
    """Integration tests with real DB."""

    def test_from_vat_members(self, db):
        """Builds vendor_key from canonical VAT number."""
        identity_id = create_identity(db, "vendor", "Nut Cracker House")
        add_member(db, identity_id, "vat_number", "10180201N", source="extraction")
        add_member(db, identity_id, "vat_number", "101800201", source="extraction")

        result = get_canonical_vendor_key(db, identity_id)
        # Should pick letter-ending format and make_vendor_key uppercases
        assert result == "10180201N"

    def test_falls_back_to_vendor_key_member(self, db):
        """When no VAT numbers, uses existing vendor_key member."""
        identity_id = create_identity(db, "vendor", "Unknown Corp")
        add_member(db, identity_id, "vendor_key", "noid_abc123", source="extraction")

        result = get_canonical_vendor_key(db, identity_id)
        assert result == "noid_abc123"

    def test_returns_none_when_no_members(self, db):
        """Returns None when identity has no VAT or vendor_key members."""
        identity_id = create_identity(db, "vendor", "Name Only")
        add_member(db, identity_id, "name", "Name Only", source="extraction")

        result = get_canonical_vendor_key(db, identity_id)
        assert result is None


class TestPipelineCanonicalVendorKey:
    """Integration test: pipeline uses canonical vendor_key on facts."""

    def test_fact_vendor_key_uses_canonical_vat(self, db, tmp_path):
        """Process doc with OCR-error VAT, verify fact uses canonical."""
        from alibi.processing.pipeline import ProcessingPipeline

        # Pre-create identity with two VAT variants
        identity_id = create_identity(db, "vendor", "FRESKO")
        add_member(db, identity_id, "vat_number", "10057000Y", source="extraction")
        add_member(db, identity_id, "vat_number", "100570000", source="extraction")
        add_member(db, identity_id, "normalized_name", "fresko", source="extraction")
        add_member(db, identity_id, "name", "FRESKO HYPERMARKET", source="extraction")

        pipeline = ProcessingPipeline(db=db)

        extraction = {
            "document_type": "receipt",
            "vendor": "FRESKO HYPERMARKET",
            "vendor_vat": "100570000",  # OCR error variant
            "date": "2026-01-21",
            "total": "50.00",
            "currency": "EUR",
            "payment_method": "cash",
            "line_items": [],
            "raw_text": "FRESKO\n50.00 EUR\n",
        }

        img = tmp_path / "fresko.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0\x00\x01\xff\xd9")

        with (
            patch.object(pipeline, "_detect_document_type") as mock_detect,
            patch.object(pipeline, "_extract_document") as mock_extract,
        ):
            mock_detect.return_value = DocumentType.RECEIPT
            mock_extract.return_value = extraction
            pipeline.process_file(img)

        facts = db.fetchall("SELECT * FROM facts", ())
        assert len(facts) >= 1
        # Should use canonical VAT (letter-ending) not the OCR error
        fact = facts[0]
        assert fact["vendor_key"] == "10057000Y"
