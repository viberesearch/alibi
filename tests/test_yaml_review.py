"""Tests for needs_review flag and scan_low_confidence_yamls."""

import pytest
from pathlib import Path

import yaml

from alibi.extraction.yaml_cache import (
    YAML_SUFFIX,
    YAML_VERSION,
    get_yaml_path,
    set_yaml_store_root,
    reset_yaml_store,
    write_yaml_cache,
)


class TestWriteYamlCacheNeedsReview:
    """Tests for needs_review flag in write_yaml_cache."""

    def test_needs_review_false_by_default(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        result = write_yaml_cache(source, {"vendor": "ACME", "total": 10.0}, "receipt")
        assert result is not None
        with open(result) as f:
            data = yaml.safe_load(f)
        assert "needs_review" not in data["_meta"]

    def test_needs_review_true_written(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        result = write_yaml_cache(
            source,
            {"vendor": "ACME", "total": 10.0},
            "receipt",
            needs_review=True,
        )
        assert result is not None
        with open(result) as f:
            data = yaml.safe_load(f)
        assert data["_meta"]["needs_review"] is True

    def test_needs_review_false_not_written(self, tmp_path: Path) -> None:
        """needs_review=False should not appear in _meta (no noise)."""
        source = tmp_path / "receipt.jpg"
        source.touch()
        result = write_yaml_cache(
            source,
            {"vendor": "ACME", "total": 10.0},
            "receipt",
            needs_review=False,
        )
        assert result is not None
        with open(result) as f:
            data = yaml.safe_load(f)
        assert "needs_review" not in data["_meta"]

    def test_needs_review_with_confidence(self, tmp_path: Path) -> None:
        """Both confidence and needs_review can coexist."""
        source = tmp_path / "invoice.pdf"
        source.touch()
        result = write_yaml_cache(
            source,
            {"vendor": "ACME", "total": 99.0},
            "invoice",
            confidence=0.3,
            needs_review=True,
        )
        assert result is not None
        with open(result) as f:
            data = yaml.safe_load(f)
        assert data["_meta"]["needs_review"] is True
        assert data["_meta"]["confidence"] == pytest.approx(0.3, abs=0.01)

    def test_needs_review_with_parser_gaps(self, tmp_path: Path) -> None:
        source = tmp_path / "receipt.jpg"
        source.touch()
        result = write_yaml_cache(
            source,
            {"vendor": "ACME"},
            "receipt",
            parser_gaps=["total", "date"],
            needs_review=True,
        )
        assert result is not None
        with open(result) as f:
            data = yaml.safe_load(f)
        assert data["_meta"]["needs_review"] is True
        assert data["_meta"]["parser_gaps"] == ["total", "date"]


class TestScanLowConfidenceYamls:
    """Tests for scan_low_confidence_yamls service function."""

    @pytest.fixture(autouse=True)
    def _setup_store(self, tmp_path: Path) -> None:
        """Set yaml store root to tmp_path so scan_yaml_store finds test files."""
        store = tmp_path / "yaml_store"
        set_yaml_store_root(store)
        yield
        set_yaml_store_root(None)
        reset_yaml_store()

    def _write_yaml(
        self,
        path: Path,
        confidence: float | None = None,
        needs_review: bool = False,
        parser_gaps: list[str] | None = None,
    ) -> Path:
        """Helper: write a YAML cache for the given source path."""
        path.touch()
        write_yaml_cache(
            path,
            {"vendor": "TEST", "total": 1.0},
            "receipt",
            confidence=confidence,
            needs_review=needs_review,
            parser_gaps=parser_gaps,
        )
        return path

    def test_finds_low_confidence(self, tmp_path: Path) -> None:
        from alibi.services.ingestion import scan_low_confidence_yamls

        self._write_yaml(tmp_path / "bad.jpg", confidence=0.2)
        self._write_yaml(tmp_path / "ok.jpg", confidence=0.9)

        results = scan_low_confidence_yamls(threshold=0.5)
        names = [p.name for p, _ in results]
        assert "bad.jpg" in names
        assert "ok.jpg" not in names

    def test_finds_flagged_needs_review(self, tmp_path: Path) -> None:
        from alibi.services.ingestion import scan_low_confidence_yamls

        self._write_yaml(tmp_path / "flagged.jpg", confidence=0.9, needs_review=True)
        self._write_yaml(tmp_path / "clean.jpg", confidence=0.9, needs_review=False)

        results = scan_low_confidence_yamls(threshold=0.5)
        names = [p.name for p, _ in results]
        assert "flagged.jpg" in names
        assert "clean.jpg" not in names

    def test_empty_store(self, tmp_path: Path) -> None:
        from alibi.services.ingestion import scan_low_confidence_yamls

        results = scan_low_confidence_yamls()
        assert results == []

    def test_no_low_confidence(self, tmp_path: Path) -> None:
        from alibi.services.ingestion import scan_low_confidence_yamls

        self._write_yaml(tmp_path / "good1.jpg", confidence=0.8)
        self._write_yaml(tmp_path / "good2.jpg", confidence=0.95)

        results = scan_low_confidence_yamls(threshold=0.5)
        assert results == []

    def test_sorted_by_confidence_ascending(self, tmp_path: Path) -> None:
        from alibi.services.ingestion import scan_low_confidence_yamls

        self._write_yaml(tmp_path / "mid.jpg", confidence=0.35)
        self._write_yaml(tmp_path / "low.jpg", confidence=0.1)
        self._write_yaml(tmp_path / "also_low.jpg", confidence=0.2)

        results = scan_low_confidence_yamls(threshold=0.5)
        confidences = [m.get("confidence") for _, m in results]
        assert confidences == sorted(confidences)

    def test_threshold_boundary(self, tmp_path: Path) -> None:
        from alibi.services.ingestion import scan_low_confidence_yamls

        # Exactly at threshold should NOT be flagged (strictly less than)
        self._write_yaml(tmp_path / "at_threshold.jpg", confidence=0.5)
        # Below threshold should be flagged
        self._write_yaml(tmp_path / "below.jpg", confidence=0.499)

        results = scan_low_confidence_yamls(threshold=0.5)
        names = [p.name for p, _ in results]
        assert "below.jpg" in names
        assert "at_threshold.jpg" not in names

    def test_no_confidence_field_with_needs_review(self, tmp_path: Path) -> None:
        from alibi.services.ingestion import scan_low_confidence_yamls

        self._write_yaml(tmp_path / "no_conf.jpg", needs_review=True)

        results = scan_low_confidence_yamls(threshold=0.5)
        names = [p.name for p, _ in results]
        assert "no_conf.jpg" in names

    def test_meta_returned_correctly(self, tmp_path: Path) -> None:
        from alibi.services.ingestion import scan_low_confidence_yamls

        self._write_yaml(
            tmp_path / "bad.jpg",
            confidence=0.3,
            needs_review=True,
            parser_gaps=["vendor"],
        )

        results = scan_low_confidence_yamls(threshold=0.5)
        assert len(results) == 1
        source_path, meta = results[0]
        assert source_path.name == "bad.jpg"
        assert meta["confidence"] == pytest.approx(0.3, abs=0.01)
        assert meta["needs_review"] is True
        assert meta["parser_gaps"] == ["vendor"]

    def test_scans_subdirectories(self, tmp_path: Path) -> None:
        from alibi.services.ingestion import scan_low_confidence_yamls

        subdir = tmp_path / "receipts"
        subdir.mkdir()
        self._write_yaml(subdir / "nested.jpg", confidence=0.2)

        results = scan_low_confidence_yamls(threshold=0.5)
        names = [p.name for p, _ in results]
        assert "nested.jpg" in names
