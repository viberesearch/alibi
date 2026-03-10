"""Tests for matching modules."""

import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from alibi.db.models import Artifact, DocumentStatus, DocumentType
from alibi.matching import (
    ComplementaryMatchResult,
    DuplicateCheckResult,
    check_content_duplicate,
    check_duplicate,
    compute_average_hash,
    compute_file_hash,
    compute_perceptual_hash,
    find_complementary_match,
    get_file_fingerprint,
    hash_distance,
    is_image_file,
    normalize_vendor_name,
)


class TestFileHashing:
    """Tests for file hashing functions."""

    def test_compute_file_hash(self, tmp_path):
        """Test computing SHA-256 hash of a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        hash1 = compute_file_hash(test_file)
        hash2 = compute_file_hash(test_file)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_compute_file_hash_different_content(self, tmp_path):
        """Test that different content produces different hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        assert compute_file_hash(file1) != compute_file_hash(file2)


class TestImageDetection:
    """Tests for image file detection."""

    def test_is_image_file_jpg(self):
        """Test detecting JPG files."""
        assert is_image_file(Path("test.jpg"))
        assert is_image_file(Path("test.jpeg"))
        assert is_image_file(Path("test.JPG"))

    def test_is_image_file_png(self):
        """Test detecting PNG files."""
        assert is_image_file(Path("test.png"))
        assert is_image_file(Path("test.PNG"))

    def test_is_image_file_other_formats(self):
        """Test detecting other image formats."""
        assert is_image_file(Path("test.gif"))
        assert is_image_file(Path("test.bmp"))
        assert is_image_file(Path("test.tiff"))
        assert is_image_file(Path("test.webp"))

    def test_is_image_file_non_image(self):
        """Test that non-image files return False."""
        assert not is_image_file(Path("test.pdf"))
        assert not is_image_file(Path("test.txt"))
        assert not is_image_file(Path("test.doc"))


class TestPerceptualHashing:
    """Tests for perceptual hashing."""

    def test_compute_perceptual_hash(self, tmp_path):
        """Test computing perceptual hash of an image."""
        from PIL import Image

        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        hash_value = compute_perceptual_hash(img_path)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 16  # dhash produces 16 hex chars

    def test_compute_average_hash(self, tmp_path):
        """Test computing average hash of an image."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="blue")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        hash_value = compute_average_hash(img_path)
        assert isinstance(hash_value, str)

    def test_similar_images_have_close_hashes(self, tmp_path):
        """Test that similar images have close perceptual hashes."""
        from PIL import Image

        # Create two very similar images
        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (100, 100), color=(255, 1, 1))  # Slightly different red

        path1 = tmp_path / "img1.png"
        path2 = tmp_path / "img2.png"
        img1.save(path1)
        img2.save(path2)

        hash1 = compute_perceptual_hash(path1)
        hash2 = compute_perceptual_hash(path2)

        distance = hash_distance(hash1, hash2)
        assert distance < 10  # Very similar images should have low distance

    def test_different_images_have_different_hashes(self, tmp_path):
        """Test that different images have different perceptual hashes."""
        from PIL import Image, ImageDraw

        # Create two images with different patterns (solid colors have same dhash)
        img1 = Image.new("RGB", (100, 100), color="white")
        draw1 = ImageDraw.Draw(img1)
        draw1.rectangle([0, 0, 50, 100], fill="black")  # Left half black

        img2 = Image.new("RGB", (100, 100), color="white")
        draw2 = ImageDraw.Draw(img2)
        draw2.rectangle([0, 0, 100, 50], fill="black")  # Top half black

        path1 = tmp_path / "img1.png"
        path2 = tmp_path / "img2.png"
        img1.save(path1)
        img2.save(path2)

        hash1 = compute_perceptual_hash(path1)
        hash2 = compute_perceptual_hash(path2)

        distance = hash_distance(hash1, hash2)
        assert distance > 0  # Different patterns should have different hashes


class TestDuplicateCheck:
    """Tests for duplicate checking."""

    def create_test_artifact(
        self,
        file_hash: str = "abc123",
        perceptual_hash: str | None = None,
        vendor: str | None = None,
        document_date: date | None = None,
        amount: Decimal | None = None,
    ) -> Artifact:
        """Create a test artifact."""
        return Artifact(
            id="test-id",
            space_id="space-1",
            type=DocumentType.RECEIPT,
            file_path="/path/to/file",
            file_hash=file_hash,
            perceptual_hash=perceptual_hash,
            vendor=vendor,
            document_date=document_date,
            amount=amount,
            status=DocumentStatus.PROCESSED,
        )

    def test_check_duplicate_exact_hash_match(self, tmp_path):
        """Test detecting exact hash match."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content")

        # Create artifact with same hash
        file_hash = compute_file_hash(test_file)
        existing = [self.create_test_artifact(file_hash=file_hash)]

        result = check_duplicate(test_file, existing)
        assert result.is_duplicate
        assert result.match_type == "exact_hash"
        assert result.similarity_score == 1.0

    def test_check_duplicate_no_match(self, tmp_path):
        """Test when no duplicate is found."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Unique content")

        existing = [self.create_test_artifact(file_hash="different_hash")]

        result = check_duplicate(test_file, existing)
        assert not result.is_duplicate

    def test_check_duplicate_perceptual_match(self, tmp_path):
        """Test detecting perceptual hash match for images."""
        from PIL import Image

        # Create test image
        img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        # Create artifact with same perceptual hash
        perceptual_hash = compute_perceptual_hash(img_path)
        existing = [
            self.create_test_artifact(
                file_hash="different", perceptual_hash=perceptual_hash
            )
        ]

        result = check_duplicate(img_path, existing)
        assert result.is_duplicate
        assert result.match_type == "perceptual"


class TestContentDuplicateCheck:
    """Tests for content-based duplicate detection."""

    def create_test_artifact(
        self,
        vendor: str | None = None,
        document_date: date | None = None,
        amount: Decimal | None = None,
        artifact_type: DocumentType = DocumentType.RECEIPT,
        transaction_time: str | None = None,
    ) -> Artifact:
        """Create a test artifact."""
        return Artifact(
            id="test-id",
            space_id="space-1",
            type=artifact_type,
            file_path="/path/to/file",
            file_hash="abc123",
            vendor=vendor,
            document_date=document_date,
            amount=amount,
            status=DocumentStatus.PROCESSED,
            transaction_time=transaction_time,
        )

    def test_check_content_duplicate_match(self):
        """Test detecting content match."""
        existing = [
            self.create_test_artifact(
                vendor="Test Store",
                document_date=date(2025, 1, 15),
                amount=Decimal("42.50"),
            )
        ]

        result = check_content_duplicate(
            vendor="Test Store",
            document_date=date(2025, 1, 15),
            amount=Decimal("42.50"),
            artifact_type="receipt",
            existing_artifacts=existing,
        )

        assert result.is_duplicate
        assert result.match_type == "content_match"

    def test_check_content_duplicate_different_vendor(self):
        """Test that different vendor is not a duplicate."""
        existing = [
            self.create_test_artifact(
                vendor="Store A",
                document_date=date(2025, 1, 15),
                amount=Decimal("42.50"),
            )
        ]

        result = check_content_duplicate(
            vendor="Store B",
            document_date=date(2025, 1, 15),
            amount=Decimal("42.50"),
            artifact_type="receipt",
            existing_artifacts=existing,
        )

        assert not result.is_duplicate

    def test_check_content_duplicate_missing_fields(self):
        """Test that missing fields returns not duplicate."""
        existing = [
            self.create_test_artifact(
                vendor="Test Store",
                document_date=date(2025, 1, 15),
                amount=Decimal("42.50"),
            )
        ]

        result = check_content_duplicate(
            vendor=None,
            document_date=date(2025, 1, 15),
            amount=Decimal("42.50"),
            artifact_type="receipt",
            existing_artifacts=existing,
        )

        assert not result.is_duplicate

    def test_check_content_duplicate_fuzzy_vendor_match(self):
        """Test that vendor names differing only in spacing/case are duplicates."""
        existing = [
            self.create_test_artifact(
                vendor="THE NUT CRACKER HOUSE",
                document_date=date(2025, 1, 15),
                amount=Decimal("12.45"),
            )
        ]

        result = check_content_duplicate(
            vendor="TheNutCrackerHouse",
            document_date=date(2025, 1, 15),
            amount=Decimal("12.45"),
            artifact_type="receipt",
            existing_artifacts=existing,
        )

        assert result.is_duplicate
        assert result.match_type == "content_match"

    def test_check_content_duplicate_fuzzy_vendor_with_punctuation(self):
        """Test vendor names with different punctuation are duplicates."""
        existing = [
            self.create_test_artifact(
                vendor="McDonald's",
                document_date=date(2025, 1, 15),
                amount=Decimal("8.99"),
            )
        ]

        result = check_content_duplicate(
            vendor="McDonalds",
            document_date=date(2025, 1, 15),
            amount=Decimal("8.99"),
            artifact_type="receipt",
            existing_artifacts=existing,
        )

        assert result.is_duplicate

    def test_check_content_duplicate_truly_different_vendors(self):
        """Test that genuinely different vendors are NOT duplicates."""
        existing = [
            self.create_test_artifact(
                vendor="Lidl",
                document_date=date(2025, 1, 15),
                amount=Decimal("42.50"),
            )
        ]

        result = check_content_duplicate(
            vendor="Aldi",
            document_date=date(2025, 1, 15),
            amount=Decimal("42.50"),
            artifact_type="receipt",
            existing_artifacts=existing,
        )

        assert not result.is_duplicate

    def test_check_content_duplicate_time_tolerance(self):
        """Test that card slip and receipt with close times are duplicates."""
        existing = [
            self.create_test_artifact(
                vendor="THE NUT CRACKER HOUSE",
                document_date=date(2025, 1, 15),
                amount=Decimal("12.45"),
                transaction_time="12:12:54",
            )
        ]

        # Receipt time differs by ~46 min — within 60 min tolerance
        result = check_content_duplicate(
            vendor="TheNutCrackerHouse",
            document_date=date(2025, 1, 15),
            amount=Decimal("12.45"),
            artifact_type="receipt",
            existing_artifacts=existing,
            transaction_time="12:58:00",
        )

        assert result.is_duplicate
        assert result.match_type == "content_match"

    def test_check_content_duplicate_time_too_far(self):
        """Test that same vendor/date/amount with very different times are NOT duplicates."""
        existing = [
            self.create_test_artifact(
                vendor="Test Store",
                document_date=date(2025, 1, 15),
                amount=Decimal("42.50"),
                transaction_time="09:00:00",
            )
        ]

        # 3 hours apart — clearly different transactions
        result = check_content_duplicate(
            vendor="Test Store",
            document_date=date(2025, 1, 15),
            amount=Decimal("42.50"),
            artifact_type="receipt",
            existing_artifacts=existing,
            transaction_time="12:00:00",
        )

        assert not result.is_duplicate


class TestNormalizeVendorName:
    """Tests for vendor name normalization."""

    def test_strips_spaces(self):
        assert normalize_vendor_name("THE NUT CRACKER HOUSE") == "thenutcrackerhouse"

    def test_camel_case(self):
        assert normalize_vendor_name("TheNutCrackerHouse") == "thenutcrackerhouse"

    def test_punctuation(self):
        assert normalize_vendor_name("McDonald's") == "mcdonalds"

    def test_hyphens_and_dots(self):
        assert normalize_vendor_name("H&M - Store.1") == "hmstore1"

    def test_identical_after_normalization(self):
        a = normalize_vendor_name("THE NUT CRACKER HOUSE")
        b = normalize_vendor_name("TheNutCrackerHouse")
        assert a == b

    def test_strips_legal_suffix_ltd(self):
        assert normalize_vendor_name("FreSko BUTANOLO LTD") == "freskobutanolo"

    def test_strips_legal_suffix_gmbh(self):
        assert normalize_vendor_name("Lidl GmbH") == "lidl"

    def test_strips_legal_suffix_inc(self):
        assert normalize_vendor_name("Apple Inc") == "apple"

    def test_strips_legal_suffix_srl(self):
        assert normalize_vendor_name("Acme S.R.L.") == "acme"


class TestVendorsMatch:
    """Tests for vendor substring matching."""

    def test_exact_match(self):
        from alibi.matching.duplicates import vendors_match

        assert vendors_match("fresko", "fresko")

    def test_substring_match_short_in_long(self):
        from alibi.matching.duplicates import vendors_match

        assert vendors_match("fresko", "freskobutanolo")

    def test_substring_match_long_in_short(self):
        from alibi.matching.duplicates import vendors_match

        assert vendors_match("freskobutanolo", "fresko")

    def test_no_match(self):
        from alibi.matching.duplicates import vendors_match

        assert not vendors_match("fresko", "lidl")

    def test_short_string_no_false_positive(self):
        """Names shorter than 4 chars should not substring-match."""
        from alibi.matching.duplicates import vendors_match

        assert not vendors_match("ab", "abc")

    def test_empty_strings(self):
        from alibi.matching.duplicates import vendors_match

        assert not vendors_match("", "fresko")
        assert not vendors_match("fresko", "")

    def test_content_duplicate_with_fuzzy_vendor(self):
        """Content duplicate check should match FRESKO vs FreSko BUTANOLO LTD."""
        existing = [
            Artifact(
                id="existing-id",
                space_id="space-1",
                type=DocumentType.RECEIPT,
                file_path="/path/to/file",
                file_hash="abc123",
                vendor="FreSko BUTANOLO LTD",
                document_date=date(2025, 1, 15),
                amount=Decimal("2.75"),
                status=DocumentStatus.PROCESSED,
            )
        ]

        result = check_content_duplicate(
            vendor="FRESKO",
            document_date=date(2025, 1, 15),
            amount=Decimal("2.75"),
            artifact_type="receipt",
            existing_artifacts=existing,
        )
        assert result.is_duplicate

    def test_complementary_match_with_fuzzy_vendor(self):
        """Complementary match should work with fuzzy vendor names."""
        existing = [
            Artifact(
                id="existing-id",
                space_id="space-1",
                type=DocumentType.RECEIPT,
                file_path="/path/to/file",
                file_hash="abc123",
                vendor="FreSko BUTANOLO LTD",
                document_date=date(2025, 1, 15),
                amount=Decimal("2.75"),
                status=DocumentStatus.PROCESSED,
            )
        ]

        result = find_complementary_match(
            vendor="FRESKO",
            document_date=date(2025, 1, 15),
            amount=Decimal("2.75"),
            artifact_type="payment_confirmation",
            existing_artifacts=existing,
        )
        assert result.is_match


class TestComplementaryMatch:
    """Tests for complementary proof matching (cross-type)."""

    def create_test_artifact(
        self,
        vendor: str | None = None,
        document_date: date | None = None,
        amount: Decimal | None = None,
        artifact_type: DocumentType = DocumentType.RECEIPT,
        transaction_time: str | None = None,
    ) -> Artifact:
        """Create a test artifact."""
        return Artifact(
            id="existing-id",
            space_id="space-1",
            type=artifact_type,
            file_path="/path/to/file",
            file_hash="abc123",
            vendor=vendor,
            document_date=document_date,
            amount=amount,
            status=DocumentStatus.PROCESSED,
            transaction_time=transaction_time,
        )

    def test_complementary_match_different_types(self):
        """Receipt + payment_confirmation for same purchase = complementary match."""
        existing = [
            self.create_test_artifact(
                vendor="Mini Market",
                document_date=date(2025, 1, 15),
                amount=Decimal("19.68"),
                artifact_type=DocumentType.RECEIPT,
            )
        ]

        result = find_complementary_match(
            vendor="Mini Market",
            document_date=date(2025, 1, 15),
            amount=Decimal("19.68"),
            artifact_type="payment_confirmation",
            existing_artifacts=existing,
        )

        assert result.is_match
        assert result.match_type == "complementary"
        assert result.original_artifact is not None
        assert result.original_artifact.type == DocumentType.RECEIPT

    def test_no_complementary_match_same_type(self):
        """Same type = NOT a complementary match (that's a duplicate)."""
        existing = [
            self.create_test_artifact(
                vendor="Mini Market",
                document_date=date(2025, 1, 15),
                amount=Decimal("19.68"),
                artifact_type=DocumentType.RECEIPT,
            )
        ]

        result = find_complementary_match(
            vendor="Mini Market",
            document_date=date(2025, 1, 15),
            amount=Decimal("19.68"),
            artifact_type="receipt",
            existing_artifacts=existing,
        )

        assert not result.is_match

    def test_complementary_match_different_vendor(self):
        """Different vendor = no match."""
        existing = [
            self.create_test_artifact(
                vendor="Store A",
                document_date=date(2025, 1, 15),
                amount=Decimal("19.68"),
                artifact_type=DocumentType.RECEIPT,
            )
        ]

        result = find_complementary_match(
            vendor="Store B",
            document_date=date(2025, 1, 15),
            amount=Decimal("19.68"),
            artifact_type="payment_confirmation",
            existing_artifacts=existing,
        )

        assert not result.is_match

    def test_complementary_match_different_amount(self):
        """Different amount = no match."""
        existing = [
            self.create_test_artifact(
                vendor="Mini Market",
                document_date=date(2025, 1, 15),
                amount=Decimal("19.68"),
                artifact_type=DocumentType.RECEIPT,
            )
        ]

        result = find_complementary_match(
            vendor="Mini Market",
            document_date=date(2025, 1, 15),
            amount=Decimal("25.00"),
            artifact_type="payment_confirmation",
            existing_artifacts=existing,
        )

        assert not result.is_match

    def test_complementary_match_date_tolerance(self):
        """Bank value date 2 days after payment = still matches (within 3-day tolerance)."""
        existing = [
            self.create_test_artifact(
                vendor="Mini Market",
                document_date=date(2025, 1, 15),
                amount=Decimal("19.68"),
                artifact_type=DocumentType.RECEIPT,
            )
        ]

        result = find_complementary_match(
            vendor="Mini Market",
            document_date=date(2025, 1, 17),  # 2 days later
            amount=Decimal("19.68"),
            artifact_type="statement",
            existing_artifacts=existing,
        )

        assert result.is_match

    def test_complementary_match_date_tolerance_exceeded(self):
        """5-day difference exceeds 3-day tolerance = no match."""
        existing = [
            self.create_test_artifact(
                vendor="Mini Market",
                document_date=date(2025, 1, 15),
                amount=Decimal("19.68"),
                artifact_type=DocumentType.RECEIPT,
            )
        ]

        result = find_complementary_match(
            vendor="Mini Market",
            document_date=date(2025, 1, 20),  # 5 days later
            amount=Decimal("19.68"),
            artifact_type="payment_confirmation",
            existing_artifacts=existing,
        )

        assert not result.is_match

    def test_complementary_match_fuzzy_vendor(self):
        """Fuzzy vendor matching works across types."""
        existing = [
            self.create_test_artifact(
                vendor="MINI MARKET",
                document_date=date(2025, 1, 15),
                amount=Decimal("19.68"),
                artifact_type=DocumentType.RECEIPT,
            )
        ]

        result = find_complementary_match(
            vendor="Mini Market",
            document_date=date(2025, 1, 15),
            amount=Decimal("19.68"),
            artifact_type="payment_confirmation",
            existing_artifacts=existing,
        )

        assert result.is_match

    def test_complementary_match_missing_fields(self):
        """Missing required fields = no match."""
        existing = [
            self.create_test_artifact(
                vendor="Mini Market",
                document_date=date(2025, 1, 15),
                amount=Decimal("19.68"),
            )
        ]

        result = find_complementary_match(
            vendor=None,
            document_date=date(2025, 1, 15),
            amount=Decimal("19.68"),
            artifact_type="payment_confirmation",
            existing_artifacts=existing,
        )

        assert not result.is_match

    def test_complementary_match_invoice_to_receipt(self):
        """Invoice + receipt for same purchase = complementary match."""
        existing = [
            self.create_test_artifact(
                vendor="Blue Island Ltd",
                document_date=date(2025, 1, 10),
                amount=Decimal("250.00"),
                artifact_type=DocumentType.INVOICE,
            )
        ]

        result = find_complementary_match(
            vendor="Blue Island Ltd",
            document_date=date(2025, 1, 10),
            amount=Decimal("250.00"),
            artifact_type="receipt",
            existing_artifacts=existing,
        )

        assert result.is_match
        assert result.original_artifact is not None
        assert result.original_artifact.type == DocumentType.INVOICE

    def test_complementary_match_time_tolerance_same_day(self):
        """Same day with close times = match."""
        existing = [
            self.create_test_artifact(
                vendor="Mini Market",
                document_date=date(2025, 1, 15),
                amount=Decimal("19.68"),
                artifact_type=DocumentType.RECEIPT,
                transaction_time="14:30:00",
            )
        ]

        result = find_complementary_match(
            vendor="Mini Market",
            document_date=date(2025, 1, 15),
            amount=Decimal("19.68"),
            artifact_type="payment_confirmation",
            existing_artifacts=existing,
            transaction_time="14:45:00",
        )

        assert result.is_match

    def test_complementary_match_time_too_far_same_day(self):
        """Same day but 3 hours apart = no match."""
        existing = [
            self.create_test_artifact(
                vendor="Mini Market",
                document_date=date(2025, 1, 15),
                amount=Decimal("19.68"),
                artifact_type=DocumentType.RECEIPT,
                transaction_time="09:00:00",
            )
        ]

        result = find_complementary_match(
            vendor="Mini Market",
            document_date=date(2025, 1, 15),
            amount=Decimal("19.68"),
            artifact_type="payment_confirmation",
            existing_artifacts=existing,
            transaction_time="12:00:00",
        )

        assert not result.is_match


class TestFileFingerprint:
    """Tests for file fingerprint generation."""

    def test_get_file_fingerprint_text_file(self, tmp_path):
        """Test fingerprint for text file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello")

        fp = get_file_fingerprint(test_file)
        assert "file_hash" in fp
        assert "file_size" in fp
        assert fp["extension"] == ".txt"
        assert "perceptual_hash" not in fp  # Not an image

    def test_get_file_fingerprint_image_file(self, tmp_path):
        """Test fingerprint for image file."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "test.png"
        img.save(img_path)

        fp = get_file_fingerprint(img_path)
        assert "file_hash" in fp
        assert "perceptual_hash" in fp
        assert "average_hash" in fp
        assert fp["width"] == 100
        assert fp["height"] == 100


class TestSuggestVendorCorrection:
    """Tests for OCR vendor name spell correction via identity database."""

    @pytest.fixture
    def db(self):
        import os

        os.environ.setdefault("ALIBI_TESTING", "1")
        from alibi.config import Config, reset_config
        from alibi.db.connection import DatabaseManager

        reset_config()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        config = Config(db_path=db_path)
        manager = DatabaseManager(config)
        if not manager.is_initialized():
            manager.initialize()
        yield manager
        manager.close()
        os.unlink(db_path)

    def _seed_vendor(self, db, canonical_name, name_members=None):
        from alibi.identities import store

        identity_id = store.create_identity(db, "vendor", canonical_name)
        store.add_member(db, identity_id, "name", canonical_name)
        for name in name_members or []:
            store.add_member(db, identity_id, "name", name)
        return identity_id

    def test_exact_match_returns_none(self, db):
        from alibi.identities.matching import suggest_vendor_correction

        self._seed_vendor(db, "Fresh Butanolo", ["FRESH BUTANOLO"])
        assert suggest_vendor_correction(db, "FRESH BUTANOLO") is None

    def test_similar_name_corrected(self, db):
        from alibi.identities.matching import suggest_vendor_correction

        self._seed_vendor(db, "Fresh Butanolo", ["FRESH BUTANOLO"])
        result = suggest_vendor_correction(db, "RRESH BURANOLO")
        assert result == "Fresh Butanolo"

    def test_too_different_returns_none(self, db):
        from alibi.identities.matching import suggest_vendor_correction

        self._seed_vendor(db, "Lidl", ["LIDL"])
        assert suggest_vendor_correction(db, "Aldi") is None

    def test_empty_db_returns_none(self, db):
        from alibi.identities.matching import suggest_vendor_correction

        assert suggest_vendor_correction(db, "RRESH BURANOLO") is None


# ---------------------------------------------------------------------------
# TestSuggestItemCorrection
# ---------------------------------------------------------------------------


class TestSuggestItemCorrection:
    """Tests for OCR spell correction of item names via identity database."""

    @pytest.fixture
    def db(self):
        import os
        from alibi.config import Config, reset_config
        from alibi.db.connection import DatabaseManager

        reset_config()
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        config = Config(db_path=db_path)
        manager = DatabaseManager(config)
        if not manager.is_initialized():
            manager.initialize()
        yield manager
        manager.close()
        os.unlink(db_path)

    def _seed_item(self, db, canonical_name, name_members=None):
        from alibi.identities import store

        identity_id = store.create_identity(db, "item", canonical_name)
        store.add_member(db, identity_id, "name", canonical_name)
        for name in name_members or []:
            store.add_member(db, identity_id, "name", name)
        return identity_id

    def test_exact_match_returns_none(self, db):
        from alibi.identities.matching import suggest_item_correction

        self._seed_item(db, "ΑΛΦΑΜΕΓΑ ΓΑΛΑ 1L", ["ALPHAMEGA MILK 1L"])
        assert suggest_item_correction(db, "ALPHAMEGA MILK 1L") is None

    def test_similar_name_corrected(self, db):
        from alibi.identities.matching import suggest_item_correction

        self._seed_item(db, "ALPHAMEGA MILK 1L", ["ALPHAMEGA MILK 1L"])
        result = suggest_item_correction(db, "ALPHAMEGA MILX 1L")
        assert result == "ALPHAMEGA MILK 1L"

    def test_too_different_returns_none(self, db):
        from alibi.identities.matching import suggest_item_correction

        self._seed_item(db, "Coca Cola 330ml", ["COCA COLA 330ML"])
        assert suggest_item_correction(db, "ORGANIC BANANAS") is None

    def test_empty_db_returns_none(self, db):
        from alibi.identities.matching import suggest_item_correction

        assert suggest_item_correction(db, "ALPHAMEGA MILX 1L") is None

    def test_short_name_returns_none(self, db):
        from alibi.identities.matching import suggest_item_correction

        self._seed_item(db, "AB")
        assert suggest_item_correction(db, "AC") is None

    def test_product_cache_correction(self, db):
        """Product cache names are used for spell correction."""
        import json

        from alibi.identities.matching import suggest_item_correction

        # No identity entries — only product_cache
        db.execute(
            "INSERT INTO product_cache (barcode, data, source) VALUES (?, ?, ?)",
            (
                "5201054025642",
                json.dumps({"product_name": "Alphamega Full Fat Milk 1L"}),
                "openfoodfacts",
            ),
        )
        result = suggest_item_correction(db, "Alphamega Full Fat Milx 1L")
        assert result == "Alphamega Full Fat Milk 1L"

    def test_product_cache_not_found_excluded(self, db):
        """Negative cache entries should not be used for correction."""
        import json

        from alibi.identities.matching import suggest_item_correction

        db.execute(
            "INSERT INTO product_cache (barcode, data, source) VALUES (?, ?, ?)",
            (
                "5201054025642",
                json.dumps({"_not_found": True}),
                "negative",
            ),
        )
        assert suggest_item_correction(db, "SOME PRODUCT XYZ") is None

    def test_product_cache_short_name_excluded(self, db):
        """Product names < 3 chars should be excluded."""
        import json

        from alibi.identities.matching import suggest_item_correction

        db.execute(
            "INSERT INTO product_cache (barcode, data, source) VALUES (?, ?, ?)",
            (
                "5201054025642",
                json.dumps({"product_name": "AB"}),
                "openfoodfacts",
            ),
        )
        assert suggest_item_correction(db, "AC") is None
