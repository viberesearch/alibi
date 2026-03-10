"""Tests for session 48 post-YAML-first improvements.

Covers:
- Template exposure (generate_blank_template, SUPPORTED_DOCUMENT_TYPES)
- Annotation durability (collect, migrate, cleanup orphans)
- Cloud re-linking (cleanup_document enriched return, reconcile_forming_clouds)
- Git versioning (YamlVersioner)
"""

import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
import yaml

from alibi.db.connection import DatabaseManager
from alibi.db.models import (
    Atom,
    AtomType,
    Bundle,
    BundleAtom,
    BundleAtomRole,
    BundleType,
    Cloud,
    CloudBundle,
    CloudMatchType,
    CloudStatus,
    Document,
    Fact,
    FactItem,
    FactType,
    UnitType,
)
from alibi.db import v2_store


# ---------------------------------------------------------------------------
# Template exposure
# ---------------------------------------------------------------------------


class TestGenerateBlankTemplate:
    """Tests for generate_blank_template() and SUPPORTED_DOCUMENT_TYPES."""

    def test_supported_types_list(self):
        from alibi.extraction.yaml_cache import SUPPORTED_DOCUMENT_TYPES

        assert "receipt" in SUPPORTED_DOCUMENT_TYPES
        assert "invoice" in SUPPORTED_DOCUMENT_TYPES
        assert "payment_confirmation" in SUPPORTED_DOCUMENT_TYPES
        assert "statement" in SUPPORTED_DOCUMENT_TYPES
        assert "contract" in SUPPORTED_DOCUMENT_TYPES
        assert "warranty" in SUPPORTED_DOCUMENT_TYPES
        assert len(SUPPORTED_DOCUMENT_TYPES) == 6

    def test_receipt_template_has_expected_fields(self):
        from alibi.extraction.yaml_cache import generate_blank_template

        tmpl = generate_blank_template("receipt")
        assert tmpl["document_type"] == "receipt"
        assert "vendor" in tmpl
        assert "total" in tmpl
        assert "date" in tmpl
        assert "line_items" in tmpl
        assert isinstance(tmpl["line_items"], list)
        assert len(tmpl["line_items"]) == 1
        assert "name" in tmpl["line_items"][0]

    def test_invoice_template(self):
        from alibi.extraction.yaml_cache import generate_blank_template

        tmpl = generate_blank_template("invoice")
        assert tmpl["document_type"] == "invoice"
        assert "issuer" in tmpl
        assert "invoice_number" in tmpl
        assert "line_items" in tmpl

    def test_statement_uses_transactions_key(self):
        from alibi.extraction.yaml_cache import generate_blank_template

        tmpl = generate_blank_template("statement")
        assert "transactions" in tmpl
        assert "line_items" not in tmpl

    def test_contract_no_items(self):
        from alibi.extraction.yaml_cache import generate_blank_template

        tmpl = generate_blank_template("contract")
        assert "line_items" not in tmpl
        assert "transactions" not in tmpl
        assert "vendor" in tmpl
        assert "effective_date" in tmpl

    def test_include_meta(self):
        from alibi.extraction.yaml_cache import (
            YAML_VERSION,
            generate_blank_template,
        )

        tmpl = generate_blank_template("receipt", include_meta=True)
        assert "_meta" in tmpl
        assert tmpl["_meta"]["version"] == YAML_VERSION

    def test_exclude_meta(self):
        from alibi.extraction.yaml_cache import generate_blank_template

        tmpl = generate_blank_template("receipt", include_meta=False)
        assert "_meta" not in tmpl

    def test_unknown_type_raises(self):
        from alibi.extraction.yaml_cache import generate_blank_template

        with pytest.raises(ValueError, match="Unknown document type"):
            generate_blank_template("bogus")

    def test_template_is_valid_yaml(self, tmp_path):
        from alibi.extraction.yaml_cache import generate_blank_template

        tmpl = generate_blank_template("receipt")
        out = tmp_path / "test.yaml"
        with open(out, "w") as f:
            yaml.dump(tmpl, f)
        with open(out) as f:
            loaded = yaml.safe_load(f)
        assert loaded["document_type"] == "receipt"


# ---------------------------------------------------------------------------
# Annotation durability
# ---------------------------------------------------------------------------


class TestAnnotationDurability:
    """Tests for annotation collection, migration, and orphan cleanup."""

    def test_collect_empty(self, db):
        """No annotations to collect."""
        from alibi.annotations.store import collect_annotations_for_cleanup

        saved = collect_annotations_for_cleanup(db, [], [])
        assert saved == []

    def test_collect_fact_annotations(self, db):
        """Collect annotations from facts about to be deleted."""
        from alibi.annotations.store import (
            add_annotation,
            collect_annotations_for_cleanup,
        )

        # Create a fake fact annotation
        ann_id = add_annotation(db, "note", "fact", "fact-001", "note", "birthday gift")

        saved = collect_annotations_for_cleanup(db, ["fact-001"], [])
        assert len(saved) == 1
        assert saved[0]["id"] == ann_id
        assert saved[0]["value"] == "birthday gift"

    def test_collect_fact_item_annotations(self, db):
        """Collect annotations from fact_items about to be deleted."""
        from alibi.annotations.store import (
            add_annotation,
            collect_annotations_for_cleanup,
        )

        ann_id = add_annotation(
            db, "person", "fact_item", "item-001", "bought_for", "Maria"
        )

        saved = collect_annotations_for_cleanup(db, [], ["item-001"])
        assert len(saved) == 1
        assert saved[0]["value"] == "Maria"

    def test_migrate_fact_annotation(self, db):
        """Migrate fact-level annotation to new fact."""
        from alibi.annotations.store import (
            get_annotations,
            migrate_annotations_to_fact,
        )

        saved = [
            {
                "id": "old-ann-1",
                "annotation_type": "note",
                "target_type": "fact",
                "target_id": "old-fact",
                "key": "note",
                "value": "birthday gift",
                "metadata": None,
                "source": "user",
            }
        ]

        count = migrate_annotations_to_fact(db, saved, "new-fact-id", [])
        assert count == 1

        anns = get_annotations(db, target_type="fact", target_id="new-fact-id")
        assert len(anns) == 1
        assert anns[0]["value"] == "birthday gift"

    def test_migrate_item_annotation_by_name(self, db):
        """Migrate fact_item annotation matching by item name in metadata."""
        from alibi.annotations.store import (
            get_annotations,
            migrate_annotations_to_fact,
        )

        saved = [
            {
                "id": "old-ann-2",
                "annotation_type": "person",
                "target_type": "fact_item",
                "target_id": "old-item",
                "key": "bought_for",
                "value": "Maria",
                "metadata": {"item_name": "Milk 1L"},
                "source": "user",
            }
        ]

        new_items = [{"id": "new-item-id", "name": "Milk 1L"}]
        count = migrate_annotations_to_fact(db, saved, "new-fact-id", new_items)
        assert count == 1

        anns = get_annotations(db, target_type="fact_item", target_id="new-item-id")
        assert len(anns) == 1
        assert anns[0]["value"] == "Maria"

    def test_migrate_no_match_skips(self, db):
        """Unmatched item annotations are skipped."""
        from alibi.annotations.store import migrate_annotations_to_fact

        saved = [
            {
                "id": "old-ann-3",
                "annotation_type": "person",
                "target_type": "fact_item",
                "target_id": "old-item",
                "key": "bought_for",
                "value": "John",
                "metadata": {"item_name": "Widget XYZ"},
                "source": "user",
            }
        ]

        new_items = [{"id": "new-item-id", "name": "Bread"}]
        count = migrate_annotations_to_fact(db, saved, "new-fact-id", new_items)
        assert count == 0

    def test_cleanup_orphaned_annotations(self, db):
        """Orphaned annotations are deleted."""
        from alibi.annotations.store import (
            add_annotation,
            cleanup_orphaned_annotations,
        )

        # Create annotation pointing to non-existent fact
        add_annotation(db, "note", "fact", "nonexistent-fact", "note", "orphan")

        deleted = cleanup_orphaned_annotations(db)
        assert deleted == 1


# ---------------------------------------------------------------------------
# Cloud re-linking (cleanup_document enriched result)
# ---------------------------------------------------------------------------


@pytest.fixture
def full_pipeline_data(db):
    """Create a complete document → atom → bundle → cloud → fact chain."""
    doc = Document(
        id=str(uuid4()),
        file_path="/test/receipt.jpg",
        file_hash="hash_full",
    )
    v2_store.store_document(db, doc)

    atoms = [
        Atom(
            id=str(uuid4()),
            document_id=doc.id,
            atom_type=AtomType.VENDOR,
            data={"name": "Test Store"},
        ),
        Atom(
            id=str(uuid4()),
            document_id=doc.id,
            atom_type=AtomType.ITEM,
            data={"name": "Milk", "quantity": 1, "unit_price": 2.0},
        ),
    ]
    v2_store.store_atoms(db, atoms)

    bundle = Bundle(
        id=str(uuid4()),
        document_id=doc.id,
        bundle_type=BundleType.BASKET,
    )
    bundle_atoms = [
        BundleAtom(
            bundle_id=bundle.id,
            atom_id=atoms[0].id,
            role=BundleAtomRole.VENDOR_INFO,
        ),
        BundleAtom(
            bundle_id=bundle.id,
            atom_id=atoms[1].id,
            role=BundleAtomRole.BASKET_ITEM,
        ),
    ]
    v2_store.store_bundle(db, bundle, bundle_atoms)

    cloud = Cloud(id=str(uuid4()))
    cb = CloudBundle(
        cloud_id=cloud.id,
        bundle_id=bundle.id,
        match_type=CloudMatchType.EXACT_AMOUNT,
        match_confidence=Decimal("1.0"),
    )
    v2_store.store_cloud(db, cloud, cb)

    fact = Fact(
        id=str(uuid4()),
        cloud_id=cloud.id,
        fact_type=FactType.PURCHASE,
        vendor="Test Store",
        total_amount=Decimal("10.00"),
        currency="EUR",
        event_date=date(2026, 1, 1),
    )
    item = FactItem(
        id=str(uuid4()),
        fact_id=fact.id,
        atom_id=atoms[1].id,
        name="Milk",
        quantity=Decimal("1"),
        unit=UnitType.PIECE,
        unit_price=Decimal("2.00"),
        total_price=Decimal("2.00"),
    )
    v2_store.store_fact(db, fact, [item])

    return {
        "doc": doc,
        "atoms": atoms,
        "bundle": bundle,
        "cloud": cloud,
        "fact": fact,
        "item": item,
    }


class TestCleanupDocumentEnriched:
    """Tests for enriched cleanup_document return."""

    def test_returns_dict_on_success(self, db, full_pipeline_data):
        result = v2_store.cleanup_document(db, full_pipeline_data["doc"].id)
        assert isinstance(result, dict)
        assert result["cleaned"] is True
        assert isinstance(result["saved_annotations"], list)
        assert isinstance(result["surviving_cloud_ids"], list)

    def test_returns_cleaned_false_for_missing(self, db):
        result = v2_store.cleanup_document(db, "nonexistent")
        assert result["cleaned"] is False

    def test_collects_annotations_on_cleanup(self, db, full_pipeline_data):
        from alibi.annotations.store import add_annotation

        add_annotation(
            db,
            "note",
            "fact",
            full_pipeline_data["fact"].id,
            "note",
            "test annotation",
        )

        result = v2_store.cleanup_document(db, full_pipeline_data["doc"].id)
        assert result["cleaned"] is True
        assert len(result["saved_annotations"]) == 1
        assert result["saved_annotations"][0]["value"] == "test annotation"


# ---------------------------------------------------------------------------
# Reconcile forming clouds
# ---------------------------------------------------------------------------


class TestReconcileFormingClouds:
    """Tests for reconcile_forming_clouds service function."""

    def test_no_forming_clouds(self, db):
        from alibi.services.correction import reconcile_forming_clouds

        count = reconcile_forming_clouds(db)
        assert count == 0


# ---------------------------------------------------------------------------
# Git versioning
# ---------------------------------------------------------------------------


class TestYamlVersioner:
    """Tests for YamlVersioner."""

    def test_track_file_within_vault(self, tmp_path):
        from alibi.mycelium.yaml_versioning import YamlVersioner

        versioner = YamlVersioner(vault_path=tmp_path)
        yaml_file = tmp_path / "test.alibi.yaml"
        yaml_file.touch()
        versioner.track(yaml_file)
        assert len(versioner._pending) == 1

    def test_track_file_outside_vault(self, tmp_path):
        from alibi.mycelium.yaml_versioning import YamlVersioner

        versioner = YamlVersioner(vault_path=tmp_path)
        outside = Path("/tmp/outside.alibi.yaml")
        versioner.track(outside)
        assert len(versioner._pending) == 0

    def test_track_disabled(self, tmp_path):
        from alibi.mycelium.yaml_versioning import YamlVersioner

        versioner = YamlVersioner(vault_path=tmp_path)
        yaml_file = tmp_path / "test.alibi.yaml"
        yaml_file.touch()

        with patch.dict("os.environ", {"ALIBI_YAML_GIT_VERSIONING": "false"}):
            versioner.track(yaml_file)
        assert len(versioner._pending) == 0

    def test_commit_pending_clears_list(self, tmp_path):
        from alibi.mycelium.yaml_versioning import YamlVersioner

        versioner = YamlVersioner(vault_path=tmp_path)
        yaml_file = tmp_path / "test.alibi.yaml"
        yaml_file.touch()
        versioner.track(yaml_file)

        # Not a git repo, so commit_pending returns False but clears list
        result = versioner.commit_pending()
        assert result is False
        assert len(versioner._pending) == 0

    def test_commit_pending_empty_is_noop(self, tmp_path):
        from alibi.mycelium.yaml_versioning import YamlVersioner

        versioner = YamlVersioner(vault_path=tmp_path)
        result = versioner.commit_pending()
        assert result is True

    def test_singleton_pattern(self, tmp_path):
        from alibi.mycelium.yaml_versioning import (
            get_yaml_versioner,
            reset_yaml_versioner,
        )

        reset_yaml_versioner()
        v1 = get_yaml_versioner(tmp_path)
        v2 = get_yaml_versioner()
        assert v1 is v2
        reset_yaml_versioner()


# ---------------------------------------------------------------------------
# CLI template command
# ---------------------------------------------------------------------------


class TestTemplateCommand:
    """Tests for `lt template` CLI command."""

    def test_template_receipt(self):
        from click.testing import CliRunner

        from alibi.cli import template

        runner = CliRunner()
        result = runner.invoke(template, ["receipt"])
        assert result.exit_code == 0
        assert "document_type: receipt" in result.output
        assert "vendor:" in result.output

    def test_template_unknown_type(self):
        from click.testing import CliRunner

        from alibi.cli import template

        runner = CliRunner()
        result = runner.invoke(template, ["bogus"])
        assert result.exit_code != 0

    def test_template_output_file(self, tmp_path):
        from click.testing import CliRunner

        from alibi.cli import template

        out = tmp_path / "test.yaml"
        runner = CliRunner()
        result = runner.invoke(template, ["invoice", "--output", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        with open(out) as f:
            data = yaml.safe_load(f)
        assert data["document_type"] == "invoice"


# ---------------------------------------------------------------------------
# MCP template tool
# ---------------------------------------------------------------------------


class TestMcpGetYamlTemplate:
    """Tests for the generate_blank_template used by MCP tool."""

    def test_valid_type_returns_template(self):
        from alibi.extraction.yaml_cache import generate_blank_template

        result = generate_blank_template("receipt")
        assert result["document_type"] == "receipt"
        assert "vendor" in result

    def test_all_types_produce_valid_templates(self):
        from alibi.extraction.yaml_cache import (
            SUPPORTED_DOCUMENT_TYPES,
            generate_blank_template,
        )

        for dtype in SUPPORTED_DOCUMENT_TYPES:
            tmpl = generate_blank_template(dtype)
            assert tmpl["document_type"] == dtype
            assert "_meta" in tmpl
