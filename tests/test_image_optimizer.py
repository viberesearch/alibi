"""Tests for alibi.processing.image_optimizer."""

import io
from pathlib import Path

import pytest
from PIL import Image

from alibi.processing.image_optimizer import (
    DEFAULT_MAX_DIM,
    DEFAULT_QUALITY,
    _MIN_FILE_SIZE,
    _MIN_OCR_DIM,
    _OPTIMIZED_MARKER,
    is_already_optimized,
    optimize_image,
)


def _create_test_image(
    path: Path,
    width: int = 4000,
    height: int = 3000,
    color: str = "red",
    fmt: str = "JPEG",
    add_exif: bool = True,
) -> Path:
    """Create a test image at the given path.

    Uses gradient fill to ensure file size exceeds _MIN_FILE_SIZE
    (solid colors compress too well for JPEG).
    """
    img = Image.new("RGB", (width, height))
    # Fill with gradient to avoid extreme compression
    pixels = []
    for y in range(height):
        for x in range(width):
            r = (x * 255 // max(width - 1, 1)) & 0xFF
            g = (y * 255 // max(height - 1, 1)) & 0xFF
            b = ((x + y) * 127 // max(width + height - 2, 1)) & 0xFF
            pixels.append((r, g, b))
    img.putdata(pixels)

    buf = io.BytesIO()
    if add_exif and fmt == "JPEG":
        # Create minimal EXIF data (Exif header)
        exif_data = (
            b"Exif\x00\x00MM\x00*\x00\x00\x00\x08" b"\x00\x00\x00\x00\x00\x00\x00"
        )
        img.save(buf, format=fmt, quality=95, exif=exif_data)
    else:
        img.save(buf, format=fmt, quality=95)
    path.write_bytes(buf.getvalue())
    return path


class TestIsAlreadyOptimized:
    """Tests for is_already_optimized()."""

    def test_unoptimized_image(self, tmp_path: Path) -> None:
        img_path = _create_test_image(tmp_path / "test.jpg")
        assert is_already_optimized(img_path) is False

    def test_optimized_image(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 100), "blue")
        img.save(img_path, format="JPEG", comment=_OPTIMIZED_MARKER)
        assert is_already_optimized(img_path) is True

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        assert is_already_optimized(tmp_path / "nope.jpg") is False


class TestOptimizeImage:
    """Tests for optimize_image()."""

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        result = optimize_image(tmp_path / "missing.jpg")
        assert result["optimized"] is False
        assert result["reason"] == "file_not_found"

    def test_unsupported_format(self, tmp_path: Path) -> None:
        txt = tmp_path / "test.txt"
        txt.write_text("not an image")
        result = optimize_image(txt)
        assert result["optimized"] is False
        assert result["reason"] == "unsupported_format"

    def test_too_small(self, tmp_path: Path) -> None:
        img_path = tmp_path / "tiny.jpg"
        img = Image.new("RGB", (10, 10), "green")
        img.save(img_path, format="JPEG")
        assert img_path.stat().st_size < _MIN_FILE_SIZE
        result = optimize_image(img_path)
        assert result["optimized"] is False
        assert result["reason"] == "too_small"

    def test_skip_already_optimized(self, tmp_path: Path) -> None:
        img_path = tmp_path / "done.jpg"
        # Create a noisy image large enough to exceed _MIN_FILE_SIZE
        import random

        random.seed(42)
        img = Image.new("RGB", (500, 500))
        pixels = [
            (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            for _ in range(500 * 500)
        ]
        img.putdata(pixels)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95, comment=_OPTIMIZED_MARKER)
        img_path.write_bytes(buf.getvalue())
        assert img_path.stat().st_size >= _MIN_FILE_SIZE
        result = optimize_image(img_path)
        assert result["optimized"] is False
        assert result["reason"] == "already_optimized"

    def test_resizes_large_image(self, tmp_path: Path) -> None:
        img_path = _create_test_image(tmp_path / "big.jpg", 4000, 3000)
        original_size = img_path.stat().st_size
        result = optimize_image(img_path, max_dim=2048)
        assert result["optimized"] is True
        assert result["resized"] is True
        assert result["original_dimensions"] == (4000, 3000)
        new_w, new_h = result["new_dimensions"]
        assert max(new_w, new_h) <= 2048
        assert result["new_size"] < original_size

    def test_strips_exif(self, tmp_path: Path) -> None:
        img_path = _create_test_image(tmp_path / "exif.jpg", 1000, 1000, add_exif=True)
        result = optimize_image(img_path, max_dim=2048)
        assert result["optimized"] is True
        assert result["stripped_exif"] is True
        # Verify marker present after optimization
        assert is_already_optimized(img_path)

    def test_no_resize_when_within_limit(self, tmp_path: Path) -> None:
        img_path = _create_test_image(tmp_path / "small.jpg", 1000, 800, add_exif=True)
        result = optimize_image(img_path, max_dim=2048)
        assert result["optimized"] is True
        assert result["resized"] is False
        # Still optimized because EXIF was stripped
        assert result["stripped_exif"] is True

    def test_png_converted_to_jpeg(self, tmp_path: Path) -> None:
        png_path = _create_test_image(
            tmp_path / "test.png", 2000, 1500, fmt="PNG", add_exif=False
        )
        assert png_path.exists()
        result = optimize_image(png_path, max_dim=2048)
        assert result["optimized"] is True
        # Original PNG should be gone
        assert not png_path.exists()
        # New JPEG should exist
        jpg_path = tmp_path / "test.jpg"
        assert jpg_path.exists()
        assert result["new_path"] == jpg_path

    def test_custom_quality(self, tmp_path: Path) -> None:
        img_path = _create_test_image(tmp_path / "q.jpg", 2000, 1500)
        result_high = optimize_image(img_path, quality=95)
        size_high = result_high["new_size"]
        # Re-create and optimize with low quality
        img_path2 = _create_test_image(tmp_path / "q2.jpg", 2000, 1500)
        result_low = optimize_image(img_path2, quality=50)
        size_low = result_low["new_size"]
        assert size_low < size_high

    def test_rgba_converted_to_rgb(self, tmp_path: Path) -> None:
        """RGBA images should be converted to RGB for JPEG output."""
        img_path = tmp_path / "rgba.png"
        import random

        random.seed(99)
        img = Image.new("RGBA", (500, 500))
        pixels = [
            (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(100, 255),
            )
            for _ in range(500 * 500)
        ]
        img.putdata(pixels)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_path.write_bytes(buf.getvalue())
        assert img_path.stat().st_size >= _MIN_FILE_SIZE

        result = optimize_image(img_path)
        assert result["optimized"] is True
        jpg_path = tmp_path / "rgba.jpg"
        assert jpg_path.exists()
        with Image.open(jpg_path) as loaded:
            assert loaded.mode == "RGB"

    def test_idempotent(self, tmp_path: Path) -> None:
        """Running optimize twice should skip the second time.

        The first optimization resizes 3000x2000 → ~2048x1365 and compresses.
        The result may drop below _MIN_FILE_SIZE, so verify the marker check
        or the too_small check prevents re-optimization.
        """
        img_path = _create_test_image(tmp_path / "idem.jpg", 3000, 2000)
        result1 = optimize_image(img_path)
        assert result1["optimized"] is True

        result2 = optimize_image(img_path)
        assert result2["optimized"] is False
        assert result2["reason"] in ("already_optimized", "too_small")


class TestMinOcrDimProtection:
    """Tests for _MIN_OCR_DIM protection on extreme aspect ratios."""

    def test_extreme_wide_no_upscale(self, tmp_path: Path) -> None:
        """10000x200 (50:1): min dim already < _MIN_OCR_DIM, scale would upscale → no resize."""
        img_path = _create_test_image(tmp_path / "wide.jpg", 10000, 200)
        result = optimize_image(img_path, max_dim=2048)
        assert result["optimized"] is True
        # Should NOT have been resized because protecting min dim requires upscale
        assert result.get("resized", False) is False

    def test_extreme_tall_no_upscale(self, tmp_path: Path) -> None:
        """200x10000 (tall): same logic, no resize."""
        img_path = _create_test_image(tmp_path / "tall.jpg", 200, 10000)
        result = optimize_image(img_path, max_dim=2048)
        assert result["optimized"] is True
        assert result.get("resized", False) is False

    def test_moderate_extreme_preserves_min_dim(self, tmp_path: Path) -> None:
        """10000x600 (17:1): scale=0.2048 → 600*0.2048=123 < 400.
        Adjusted scale=400/600=0.667 → 6667x400. Min dim preserved."""
        img_path = _create_test_image(tmp_path / "mod_wide.jpg", 10000, 600)
        result = optimize_image(img_path, max_dim=2048)
        assert result["optimized"] is True
        assert result["resized"] is True
        new_w, new_h = result["new_dimensions"]
        assert min(new_w, new_h) >= _MIN_OCR_DIM

    def test_normal_proportions_unaffected(self, tmp_path: Path) -> None:
        """3000x4000 (normal): standard resize, min dim already safe."""
        img_path = _create_test_image(tmp_path / "normal.jpg", 3000, 4000)
        result = optimize_image(img_path, max_dim=2048)
        assert result["optimized"] is True
        assert result["resized"] is True
        new_w, new_h = result["new_dimensions"]
        assert max(new_w, new_h) <= 2048
        assert min(new_w, new_h) >= _MIN_OCR_DIM

    def test_5_to_1_ratio_standard_resize(self, tmp_path: Path) -> None:
        """4000x800 (5:1): scale=0.512 → 2048x410. Min dim safe, normal resize."""
        img_path = _create_test_image(tmp_path / "five_one.jpg", 4000, 800)
        result = optimize_image(img_path, max_dim=2048)
        assert result["optimized"] is True
        assert result["resized"] is True
        new_w, new_h = result["new_dimensions"]
        assert max(new_w, new_h) <= 2048
        assert min(new_w, new_h) >= _MIN_OCR_DIM


class TestDocumentHandlerFolderContext:
    """Tests for folder context resolution in DocumentHandler."""

    def test_resolves_folder_context_for_typed_folder(self, tmp_path: Path) -> None:
        """Files in typed folders should get folder context with doc_type."""
        from alibi.daemon.handlers import DocumentHandler
        from alibi.processing.folder_router import FolderContext

        inbox = tmp_path / "inbox"
        receipts_dir = inbox / "receipts"
        receipts_dir.mkdir(parents=True)

        handler = DocumentHandler(inbox_root=inbox)
        ctx = handler._resolve_folder_context(receipts_dir / "test.jpg")
        assert ctx is not None
        assert ctx.doc_type is not None
        assert ctx.doc_type.value == "receipt"

    def test_resolves_none_outside_inbox(self, tmp_path: Path) -> None:
        """Files outside inbox root should get None context."""
        from alibi.daemon.handlers import DocumentHandler

        inbox = tmp_path / "inbox"
        inbox.mkdir()
        other = tmp_path / "other"
        other.mkdir()

        handler = DocumentHandler(inbox_root=inbox)
        ctx = handler._resolve_folder_context(other / "test.jpg")
        assert ctx is None

    def test_resolves_none_without_inbox_root(self) -> None:
        """Handler without inbox_root always returns None context."""
        from alibi.daemon.handlers import DocumentHandler

        handler = DocumentHandler(inbox_root=None)
        handler.inbox_root = None  # Force None
        ctx = handler._resolve_folder_context(Path("/some/file.jpg"))
        assert ctx is None

    def test_unsorted_folder_no_doc_type(self, tmp_path: Path) -> None:
        """Files in unsorted/ should have no doc_type."""
        from alibi.daemon.handlers import DocumentHandler

        inbox = tmp_path / "inbox"
        unsorted = inbox / "unsorted"
        unsorted.mkdir(parents=True)

        handler = DocumentHandler(inbox_root=inbox)
        ctx = handler._resolve_folder_context(unsorted / "mystery.pdf")
        assert ctx is not None
        assert ctx.doc_type is None

    def test_country_subfolder_detection(self, tmp_path: Path) -> None:
        """Files in country subfolders should resolve country code."""
        from alibi.daemon.handlers import DocumentHandler

        inbox = tmp_path / "inbox"
        gr_receipts = inbox / "GR" / "receipts"
        gr_receipts.mkdir(parents=True)

        handler = DocumentHandler(inbox_root=inbox)
        ctx = handler._resolve_folder_context(gr_receipts / "test.jpg")
        assert ctx is not None
        assert ctx.country == "GR"
        assert ctx.doc_type is not None
        assert ctx.doc_type.value == "receipt"
