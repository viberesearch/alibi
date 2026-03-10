"""Tests for vendor name canonicalization via YAML aliases.

Tests loading, matching, canonicalization, pipeline integration,
and graceful degradation.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from alibi.matching.duplicates import (
    canonicalize_vendor,
    init_vendor_mappings,
    load_vendor_aliases,
    normalize_vendor_name,
    reset_vendor_mappings,
)


@pytest.fixture(autouse=True)
def _clean_vendor_mappings():
    """Reset vendor mappings before and after each test."""
    reset_vendor_mappings()
    yield
    reset_vendor_mappings()


class TestLoadVendorAliases:
    """Tests for load_vendor_aliases() YAML loading."""

    def test_valid_aliases(self, tmp_path: Path):
        """Load valid vendor aliases from YAML."""
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text(
            "FRESKO: FreSko Butanolo\n" '"Nut Cracker House": The Nut Cracker House\n'
        )

        result = load_vendor_aliases(yaml_path)
        assert len(result) == 2
        # Keys are normalized
        assert "fresko" in result
        assert result["fresko"] == "FreSko Butanolo"
        assert "nutcrackerhouse" in result
        assert result["nutcrackerhouse"] == "The Nut Cracker House"

    def test_missing_file(self, tmp_path: Path):
        """Missing YAML file returns empty dict."""
        result = load_vendor_aliases(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_empty_file(self, tmp_path: Path):
        """Empty YAML file returns empty dict."""
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text("")

        result = load_vendor_aliases(yaml_path)
        assert result == {}

    def test_corrupted_yaml(self, tmp_path: Path):
        """Corrupted YAML returns empty dict."""
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text("{{broken yaml: [")

        result = load_vendor_aliases(yaml_path)
        assert result == {}

    def test_non_dict_yaml(self, tmp_path: Path):
        """YAML that parses to a list returns empty dict."""
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text("- item1\n- item2\n")

        result = load_vendor_aliases(yaml_path)
        assert result == {}

    def test_normalizes_keys(self, tmp_path: Path):
        """Keys are normalized (lowercase, stripped of punctuation/legal suffixes)."""
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text('"FreSko BUTANOLO LTD": FreSko Butanolo\n')

        result = load_vendor_aliases(yaml_path)
        # "FreSko BUTANOLO LTD" -> strip "ltd" -> "fresko butanolo" -> strip punct -> "freskobutanolo"
        assert "freskobutanolo" in result

    def test_empty_key_skipped(self, tmp_path: Path):
        """Empty keys after normalization are skipped."""
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text('"": Some Store\n"...": Another Store\n')

        result = load_vendor_aliases(yaml_path)
        assert len(result) == 0

    def test_empty_value_skipped(self, tmp_path: Path):
        """Empty values are skipped."""
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text('FRESKO: ""\n')

        result = load_vendor_aliases(yaml_path)
        assert len(result) == 0


class TestInitAndResetVendorMappings:
    """Tests for init_vendor_mappings() and reset_vendor_mappings()."""

    def test_init_loads_aliases(self, tmp_path: Path):
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text("FRESKO: FreSko Butanolo\n")

        init_vendor_mappings(yaml_path)
        assert canonicalize_vendor("FRESKO") == "FreSko Butanolo"

    def test_init_none_clears(self, tmp_path: Path):
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text("FRESKO: FreSko Butanolo\n")

        init_vendor_mappings(yaml_path)
        assert canonicalize_vendor("FRESKO") == "FreSko Butanolo"

        init_vendor_mappings(None)
        assert canonicalize_vendor("FRESKO") == "FRESKO"

    def test_reset_clears(self, tmp_path: Path):
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text("FRESKO: FreSko Butanolo\n")

        init_vendor_mappings(yaml_path)
        reset_vendor_mappings()
        assert canonicalize_vendor("FRESKO") == "FRESKO"


class TestCanonicalizeVendor:
    """Tests for canonicalize_vendor() function."""

    def test_exact_match(self, tmp_path: Path):
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text("FRESKO: FreSko Butanolo\n")
        init_vendor_mappings(yaml_path)

        assert canonicalize_vendor("FRESKO") == "FreSko Butanolo"

    def test_case_insensitive(self, tmp_path: Path):
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text("fresko: FreSko Butanolo\n")
        init_vendor_mappings(yaml_path)

        assert canonicalize_vendor("FRESKO") == "FreSko Butanolo"
        assert canonicalize_vendor("Fresko") == "FreSko Butanolo"
        assert canonicalize_vendor("fresko") == "FreSko Butanolo"

    def test_substring_match(self, tmp_path: Path):
        """Longer vendor name matches shorter alias via substring."""
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text("FRESKO: FreSko Butanolo\n")
        init_vendor_mappings(yaml_path)

        # "FreSko BUTANOLO LTD" normalizes to "freskobutanolo"
        # "FRESKO" normalizes to "fresko"
        # "fresko" is substring of "freskobutanolo" -> match
        assert canonicalize_vendor("FreSko BUTANOLO LTD") == "FreSko Butanolo"

    def test_no_match_returns_raw(self, tmp_path: Path):
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text("FRESKO: FreSko Butanolo\n")
        init_vendor_mappings(yaml_path)

        assert canonicalize_vendor("Unknown Store") == "Unknown Store"

    def test_none_input(self):
        assert canonicalize_vendor(None) is None

    def test_empty_input(self):
        assert canonicalize_vendor("") == ""

    def test_no_aliases_loaded(self):
        """Without aliases loaded, returns raw name."""
        assert canonicalize_vendor("FRESKO") == "FRESKO"

    def test_multiple_aliases(self, tmp_path: Path):
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text(
            "FRESKO: FreSko Butanolo\n"
            "Arab Butchery: Arab Butchery Ltd\n"
            "Nut Cracker: The Nut Cracker House\n"
        )
        init_vendor_mappings(yaml_path)

        assert canonicalize_vendor("FRESKO") == "FreSko Butanolo"
        assert canonicalize_vendor("Arab Butchery") == "Arab Butchery Ltd"
        assert canonicalize_vendor("THE NUT CRACKER HOUSE") == "The Nut Cracker House"

    def test_legal_suffix_stripped_before_match(self, tmp_path: Path):
        """Legal suffixes stripped from both alias key and input before matching."""
        yaml_path = tmp_path / "vendors.yaml"
        yaml_path.write_text("Plus Discount: Plus Discount Supermarket\n")
        init_vendor_mappings(yaml_path)

        assert canonicalize_vendor("Plus Discount GmbH") == "Plus Discount Supermarket"
        assert canonicalize_vendor("PLUS DISCOUNT") == "Plus Discount Supermarket"


class TestPipelineVendorCanonicalization:
    """Integration tests: vendor canonicalization through the pipeline."""

    @pytest.fixture
    def db_manager(self, tmp_path: Path):
        from alibi.config import Config
        from alibi.db.connection import DatabaseManager

        config = Config(db_path=tmp_path / "test.db")
        db = DatabaseManager(config)
        db.initialize()
        return db

    @pytest.fixture
    def pipeline(self, db_manager):
        from alibi.processing.pipeline import ProcessingPipeline

        return ProcessingPipeline(db=db_manager)

    def test_vendor_canonicalized_in_extraction(self, pipeline, tmp_path: Path):
        """Vendor name in extracted_data uses canonical form after processing."""
        # Set up vendor aliases
        alias_path = tmp_path / "vendors.yaml"
        alias_path.write_text("FRESKO: FreSko Butanolo\n")
        init_vendor_mappings(alias_path)

        extraction = {
            "vendor": "FRESKO",
            "total": 25.50,
            "date": "2025-03-15",
            "time": "10:30:00",
            "currency": "EUR",
            "raw_text": "Receipt from FRESKO",
            "line_items": [
                {
                    "name": "Apples",
                    "quantity": 1,
                    "unit_price": 2.50,
                    "total_price": 2.50,
                }
            ],
        }

        file_path = tmp_path / "receipt.jpg"
        file_path.write_text("mock")

        with (
            patch.object(pipeline, "_extract_document", return_value=extraction),
            patch(
                "alibi.processing.pipeline.vision_detect_document_type",
                return_value="receipt",
            ),
            patch(
                "alibi.matching.duplicates.compute_perceptual_hash",
                return_value="0000000000000000",
            ),
        ):
            result = pipeline.process_file(file_path)

        assert result.success
        # Canonical name applied to extracted_data before v2 pipeline
        assert result.extracted_data is not None
        assert result.extracted_data["vendor"] == "FreSko Butanolo"

    def test_unknown_vendor_unchanged(self, pipeline, tmp_path: Path):
        """Unknown vendor passes through unchanged."""
        extraction = {
            "vendor": "New Store ABC",
            "total": 10.0,
            "date": "2025-03-15",
            "currency": "EUR",
            "raw_text": "Receipt from New Store ABC",
            "line_items": [],
        }

        file_path = tmp_path / "receipt.jpg"
        file_path.write_text("mock")

        with (
            patch.object(pipeline, "_extract_document", return_value=extraction),
            patch(
                "alibi.processing.pipeline.vision_detect_document_type",
                return_value="receipt",
            ),
            patch(
                "alibi.matching.duplicates.compute_perceptual_hash",
                return_value="0000000000000000",
            ),
        ):
            result = pipeline.process_file(file_path)

        assert result.success
        assert result.extracted_data is not None
        assert result.extracted_data["vendor"] == "New Store ABC"
