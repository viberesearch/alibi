"""Tests for contract and warranty document type support.

Tests prompt routing, vision type detection, pipeline routing, refiners,
schema validation, and refiner registry integration.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from alibi.db.connection import DatabaseManager
from alibi.db.models import DocumentType, RecordType
from alibi.extraction.prompts import (
    CONTRACT_PROMPT,
    CONTRACT_PROMPT_V2,
    WARRANTY_PROMPT,
    WARRANTY_PROMPT_V2,
    get_prompt_for_type,
)
from alibi.extraction.schemas import (
    CONTRACT_SCHEMA,
    SCHEMAS,
    WARRANTY_SCHEMA,
    validate_extraction,
)
from alibi.extraction.vision import detect_document_type
from alibi.processing.pipeline import (
    ARTIFACT_TO_RECORD_TYPE,
    ProcessingPipeline,
)
from alibi.refiners import ContractRefiner, WarrantyRefiner, get_refiner


@pytest.fixture
def db_manager(tmp_path: Path) -> DatabaseManager:
    """Create a fresh DB for testing."""
    from alibi.config import Config

    config = Config(db_path=tmp_path / "test.db")
    db = DatabaseManager(config)
    db.initialize()
    return db


@pytest.fixture
def pipeline(db_manager: DatabaseManager) -> ProcessingPipeline:
    """Create a pipeline with the test DB."""
    return ProcessingPipeline(db=db_manager)


# ============================================================================
# 1. Prompt Routing Tests
# ============================================================================


class TestPromptRouting:
    """Test get_prompt_for_type routes to correct prompts."""

    def test_warranty_v2_prompt(self):
        """get_prompt_for_type returns WARRANTY_PROMPT_V2 for warranty v2."""
        prompt = get_prompt_for_type("warranty", version=2)
        assert prompt == WARRANTY_PROMPT_V2
        assert "warranty" in prompt.lower()

    def test_contract_v2_prompt(self):
        """get_prompt_for_type returns CONTRACT_PROMPT_V2 for contract v2."""
        prompt = get_prompt_for_type("contract", version=2)
        assert prompt == CONTRACT_PROMPT_V2
        assert "contract" in prompt.lower()

    def test_warranty_v1_prompt(self):
        """get_prompt_for_type returns WARRANTY_PROMPT for warranty v1."""
        prompt = get_prompt_for_type("warranty", version=1)
        assert prompt == WARRANTY_PROMPT
        assert "warranty" in prompt.lower()

    def test_contract_v1_prompt(self):
        """get_prompt_for_type returns CONTRACT_PROMPT for contract v1."""
        prompt = get_prompt_for_type("contract", version=1)
        assert prompt == CONTRACT_PROMPT
        assert "contract" in prompt.lower()


# ============================================================================
# 2. Vision Type Detection Tests
# ============================================================================


class TestVisionTypeDetection:
    """Test vision-based document type detection for contracts and warranties."""

    def test_vision_type_map_includes_contract(self):
        """_VISION_TYPE_MAP includes contract mapping."""
        assert "contract" in ProcessingPipeline._VISION_TYPE_MAP
        assert ProcessingPipeline._VISION_TYPE_MAP["contract"] == DocumentType.CONTRACT

    def test_vision_type_map_includes_warranty(self):
        """_VISION_TYPE_MAP includes warranty mapping."""
        assert "warranty" in ProcessingPipeline._VISION_TYPE_MAP
        assert ProcessingPipeline._VISION_TYPE_MAP["warranty"] == DocumentType.WARRANTY

    def test_detect_document_type_contract(self, tmp_path: Path):
        """detect_document_type returns 'contract' when vision detects contract."""
        file_path = tmp_path / "contract.jpg"
        file_path.write_bytes(b"mock contract image")

        with patch(
            "alibi.extraction.vision._call_ollama_vision",
            return_value={"response": "contract"},
        ):
            doc_type = detect_document_type(file_path)
            assert doc_type == "contract"

    def test_detect_document_type_warranty(self, tmp_path: Path):
        """detect_document_type returns 'warranty' when vision detects warranty."""
        file_path = tmp_path / "warranty.jpg"
        file_path.write_bytes(b"mock warranty image")

        with patch(
            "alibi.extraction.vision._call_ollama_vision",
            return_value={"response": "warranty"},
        ):
            doc_type = detect_document_type(file_path)
            assert doc_type == "warranty"

    def test_contract_in_valid_types(self, tmp_path: Path):
        """'contract' is a valid document type for vision detection."""
        file_path = tmp_path / "doc.jpg"
        file_path.write_bytes(b"mock")

        with patch(
            "alibi.extraction.vision._call_ollama_vision",
            return_value={"response": "contract"},
        ):
            result = detect_document_type(file_path)
            assert result == "contract"

    def test_warranty_in_valid_types(self, tmp_path: Path):
        """'warranty' is a valid document type for vision detection."""
        file_path = tmp_path / "doc.jpg"
        file_path.write_bytes(b"mock")

        with patch(
            "alibi.extraction.vision._call_ollama_vision",
            return_value={"response": "warranty"},
        ):
            result = detect_document_type(file_path)
            assert result == "warranty"


# ============================================================================
# 3. Pipeline Routing Tests
# ============================================================================


class TestPipelineRouting:
    """Test ARTIFACT_TO_RECORD_TYPE mapping and LLM type override."""

    def test_artifact_to_record_type_contract(self):
        """ARTIFACT_TO_RECORD_TYPE maps CONTRACT to CONTRACT."""
        assert ARTIFACT_TO_RECORD_TYPE[DocumentType.CONTRACT] == RecordType.CONTRACT

    def test_artifact_to_record_type_warranty(self):
        """ARTIFACT_TO_RECORD_TYPE maps WARRANTY to WARRANTY."""
        assert ARTIFACT_TO_RECORD_TYPE[DocumentType.WARRANTY] == RecordType.WARRANTY

    def test_llm_type_override_contract(
        self, pipeline: ProcessingPipeline, tmp_path: Path
    ):
        """LLM type override works for contract document_type."""
        extraction = {
            "vendor": "Acme Corp",
            "date": "2025-01-01",
            "document_type": "contract",
            "contract_type": "service",
            "payment_terms": "monthly",
            "raw_text": "Service contract",
        }

        file_path = tmp_path / "contract.pdf"
        file_path.write_text("mock")

        with (
            patch.object(pipeline, "_extract_document", return_value=extraction),
            patch(
                "alibi.processing.pipeline.vision_detect_document_type",
                return_value="other",  # Vision says other, LLM says contract
            ),
            patch(
                "alibi.matching.duplicates.compute_perceptual_hash",
                return_value="0000000000000000",
            ),
        ):
            result = pipeline.process_file(file_path)

        assert result.success
        assert result.record_type == RecordType.CONTRACT

    def test_llm_type_override_warranty(
        self, pipeline: ProcessingPipeline, tmp_path: Path
    ):
        """LLM type override works for warranty document_type."""
        extraction = {
            "vendor": "GadgetCo",
            "date": "2025-01-15",
            "document_type": "warranty",
            "warranty_type": "manufacturer",
            "warranty_expires": "2027-01-15",
            "raw_text": "Product warranty",
        }

        file_path = tmp_path / "warranty.jpg"
        file_path.write_text("mock")

        with (
            patch.object(pipeline, "_extract_document", return_value=extraction),
            patch(
                "alibi.processing.pipeline.vision_detect_document_type",
                return_value="other",
            ),
            patch(
                "alibi.matching.duplicates.compute_perceptual_hash",
                return_value="0000000000000000",
            ),
        ):
            result = pipeline.process_file(file_path)

        assert result.success
        assert result.record_type == RecordType.WARRANTY

    def test_yaml_cache_supports_contract(self, tmp_path: Path):
        """YAML cache write/read works for contract document type."""
        from alibi.extraction.yaml_cache import write_yaml_cache, read_yaml_cache

        file_path = tmp_path / "contract.pdf"
        file_path.write_text("mock")

        extraction = {
            "vendor": "Acme Corp",
            "date": "2025-01-01",
            "contract_type": "service",
        }

        yaml_path = write_yaml_cache(file_path, extraction, "contract")
        assert yaml_path is not None
        assert yaml_path.exists()

        loaded = read_yaml_cache(file_path, doc_type="contract")
        assert loaded is not None
        assert loaded["document_type"] == "contract"
        assert loaded["vendor"] == "Acme Corp"

    def test_yaml_cache_supports_warranty(self, tmp_path: Path):
        """YAML cache write/read works for warranty document type."""
        from alibi.extraction.yaml_cache import write_yaml_cache, read_yaml_cache

        file_path = tmp_path / "warranty.jpg"
        file_path.write_text("mock")

        extraction = {
            "vendor": "GadgetCo",
            "date": "2025-01-15",
            "warranty_type": "manufacturer",
        }

        yaml_path = write_yaml_cache(file_path, extraction, "warranty")
        assert yaml_path is not None
        assert yaml_path.exists()

        loaded = read_yaml_cache(file_path, doc_type="warranty")
        assert loaded is not None
        assert loaded["document_type"] == "warranty"
        assert loaded["vendor"] == "GadgetCo"


# ============================================================================
# 4. ContractRefiner Tests
# ============================================================================


class TestContractRefiner:
    """Test ContractRefiner normalization and field mapping."""

    def test_sets_record_type_contract(self):
        """ContractRefiner sets record_type to CONTRACT."""
        refiner = ContractRefiner()
        data = {
            "vendor": "Acme Corp",
            "date": "2025-01-01",
            "contract_type": "service",
        }
        refined = refiner.refine(data)
        assert refined["record_type"] == RecordType.CONTRACT

    def test_normalizes_payment_terms_monthly(self):
        """Normalizes payment_terms: monthly."""
        refiner = ContractRefiner()
        data = {
            "vendor": "Acme",
            "date": "2025-01-01",
            "payment_terms": "monthly",
        }
        refined = refiner.refine(data)
        assert refined["payment_terms"] == "monthly"

    def test_normalizes_payment_terms_annual(self):
        """Normalizes payment_terms: annual."""
        refiner = ContractRefiner()
        data = {
            "vendor": "Acme",
            "date": "2025-01-01",
            "payment_terms": "annual",
        }
        refined = refiner.refine(data)
        assert refined["payment_terms"] == "annual"

    def test_normalizes_payment_terms_yearly_to_annual(self):
        """Normalizes payment_terms: yearly -> annual."""
        refiner = ContractRefiner()
        data = {
            "vendor": "Acme",
            "date": "2025-01-01",
            "payment_terms": "yearly",
        }
        refined = refiner.refine(data)
        assert refined["payment_terms"] == "annual"

    def test_normalizes_payment_terms_one_time(self):
        """Normalizes payment_terms: one-time."""
        refiner = ContractRefiner()
        data = {
            "vendor": "Acme",
            "date": "2025-01-01",
            "payment_terms": "one-time",
        }
        refined = refiner.refine(data)
        assert refined["payment_terms"] == "one-time"

    def test_normalizes_vendor_name(self):
        """Normalizes vendor name (strips suffixes)."""
        refiner = ContractRefiner()
        data = {
            "vendor": "Acme Corp Ltd",
            "date": "2025-01-01",
        }
        refined = refiner.refine(data)
        # Vendor normalization strips legal suffixes (Corp + Ltd)
        assert refined["vendor"] == "Acme"

    def test_maps_issuer_to_vendor_when_vendor_absent(self):
        """Maps issuer to vendor when vendor is absent."""
        refiner = ContractRefiner()
        data = {
            "issuer": "Service Provider Inc",
            "date": "2025-01-01",
        }
        refined = refiner.refine(data)
        # Vendor normalization strips legal suffixes
        assert refined["vendor"] == "Service Provider"

    def test_maps_start_date_to_date_when_date_absent(self):
        """Maps start_date to date when date is absent."""
        refiner = ContractRefiner()
        data = {
            "vendor": "Acme",
            "start_date": "2025-02-01",
        }
        refined = refiner.refine(data)
        assert refined["date"] == "2025-02-01"

    def test_normalizes_renewal_auto(self):
        """Normalizes renewal: auto."""
        refiner = ContractRefiner()
        data = {
            "vendor": "Acme",
            "date": "2025-01-01",
            "renewal": "auto",
        }
        refined = refiner.refine(data)
        assert refined["renewal"] == "auto"

    def test_normalizes_renewal_automatic_to_auto(self):
        """Normalizes renewal: automatic -> auto."""
        refiner = ContractRefiner()
        data = {
            "vendor": "Acme",
            "date": "2025-01-01",
            "renewal": "automatic",
        }
        refined = refiner.refine(data)
        assert refined["renewal"] == "auto"

    def test_normalizes_renewal_manual(self):
        """Normalizes renewal: manual."""
        refiner = ContractRefiner()
        data = {
            "vendor": "Acme",
            "date": "2025-01-01",
            "renewal": "manual",
        }
        refined = refiner.refine(data)
        assert refined["renewal"] == "manual"

    def test_normalizes_renewal_none(self):
        """Normalizes renewal: none."""
        refiner = ContractRefiner()
        data = {
            "vendor": "Acme",
            "date": "2025-01-01",
            "renewal": "none",
        }
        refined = refiner.refine(data)
        assert refined["renewal"] == "none"

    def test_handles_missing_fields_gracefully(self):
        """Handles missing/None fields gracefully."""
        from datetime import date as date_type

        refiner = ContractRefiner()
        data = {
            "vendor": "Acme",
            "date": "2025-01-01",
            # No payment_terms, renewal, etc.
        }
        refined = refiner.refine(data)
        assert refined["vendor"] == "Acme"
        # Date gets normalized to date object by BaseRefiner
        assert refined["date"] == date_type(2025, 1, 1)
        assert refined["record_type"] == RecordType.CONTRACT

    def test_handles_none_payment_terms(self):
        """Handles None payment_terms gracefully."""
        refiner = ContractRefiner()
        data = {
            "vendor": "Acme",
            "date": "2025-01-01",
            "payment_terms": None,
        }
        refined = refiner.refine(data)
        assert refined.get("payment_terms") is None

    def test_handles_none_renewal(self):
        """Handles None renewal gracefully."""
        refiner = ContractRefiner()
        data = {
            "vendor": "Acme",
            "date": "2025-01-01",
            "renewal": None,
        }
        refined = refiner.refine(data)
        assert refined.get("renewal") is None


# ============================================================================
# 5. WarrantyRefiner Tests (verify existing functionality)
# ============================================================================


class TestWarrantyRefiner:
    """Test WarrantyRefiner functionality."""

    def test_sets_record_type_warranty(self):
        """WarrantyRefiner sets record_type to WARRANTY."""
        refiner = WarrantyRefiner()
        data = {
            "vendor": "GadgetCo",
            "date": "2025-01-01",
            "warranty_type": "manufacturer",
        }
        refined = refiner.refine(data)
        assert refined["record_type"] == RecordType.WARRANTY

    def test_normalizes_warranty_type(self):
        """Normalizes warranty_type."""
        refiner = WarrantyRefiner()
        data = {
            "vendor": "GadgetCo",
            "date": "2025-01-01",
            "warranty_type": "MANUFACTURER",
        }
        refined = refiner.refine(data)
        assert refined["warranty_type"] == "manufacturer"

    def test_maps_warranty_expires_to_date(self):
        """Maps warranty_expires to date when date is absent."""
        from datetime import date as date_type

        refiner = WarrantyRefiner()
        data = {
            "vendor": "GadgetCo",
            "warranty_expires": "2027-01-01",
        }
        refined = refiner.refine(data)
        # Date gets normalized to date object by BaseRefiner
        assert refined["date"] == date_type(2027, 1, 1)

    def test_extracts_product_name(self):
        """Extracts product_name if present."""
        refiner = WarrantyRefiner()
        data = {
            "vendor": "GadgetCo",
            "date": "2025-01-01",
            "product_name": "SuperWidget 3000",
        }
        refined = refiner.refine(data)
        # Product name is mapped to "product" field
        assert refined.get("product") == "SuperWidget 3000"

    def test_handles_missing_fields(self):
        """Handles missing fields gracefully."""
        from datetime import date as date_type

        refiner = WarrantyRefiner()
        data = {
            "vendor": "GadgetCo",
            "date": "2025-01-01",
        }
        refined = refiner.refine(data)
        assert refined["vendor"] == "Gadgetco"
        # Date gets normalized to date object by BaseRefiner
        assert refined["date"] == date_type(2025, 1, 1)
        assert refined["record_type"] == RecordType.WARRANTY


# ============================================================================
# 6. Schema Validation Tests
# ============================================================================


class TestSchemaValidation:
    """Test schema registration and validation."""

    def test_warranty_schema_exists(self):
        """WARRANTY_SCHEMA exists in SCHEMAS dict."""
        assert "warranty" in SCHEMAS
        assert SCHEMAS["warranty"] == WARRANTY_SCHEMA

    def test_contract_schema_exists(self):
        """CONTRACT_SCHEMA exists in SCHEMAS dict."""
        assert "contract" in SCHEMAS
        assert SCHEMAS["contract"] == CONTRACT_SCHEMA

    def test_validate_warranty_requires_vendor(self):
        """validate_extraction for warranty requires vendor."""
        data = {
            "date": "2025-01-01",
            "warranty_type": "manufacturer",
        }
        errors = validate_extraction(data, "warranty")
        assert len(errors) > 0
        assert any("vendor" in err.lower() for err in errors)

    def test_validate_warranty_with_vendor(self):
        """validate_extraction for warranty passes with vendor."""
        data = {
            "vendor": "GadgetCo",
            "date": "2025-01-01",
            "warranty_type": "manufacturer",
        }
        errors = validate_extraction(data, "warranty")
        assert errors == []

    def test_validate_contract_requires_vendor(self):
        """validate_extraction for contract requires vendor."""
        data = {
            "date": "2025-01-01",
            "contract_type": "service",
        }
        errors = validate_extraction(data, "contract")
        assert len(errors) > 0
        assert any("vendor" in err.lower() for err in errors)

    def test_validate_contract_with_vendor(self):
        """validate_extraction for contract passes with vendor."""
        data = {
            "vendor": "Acme Corp",
            "date": "2025-01-01",
            "contract_type": "service",
        }
        errors = validate_extraction(data, "contract")
        assert errors == []


# ============================================================================
# 7. Refiner Registry Tests
# ============================================================================


class TestRefinerRegistry:
    """Test refiner registry mapping."""

    def test_get_refiner_contract(self):
        """get_refiner returns ContractRefiner for CONTRACT."""
        refiner = get_refiner(RecordType.CONTRACT)
        assert isinstance(refiner, ContractRefiner)

    def test_get_refiner_warranty(self):
        """get_refiner returns WarrantyRefiner for WARRANTY."""
        refiner = get_refiner(RecordType.WARRANTY)
        assert isinstance(refiner, WarrantyRefiner)
