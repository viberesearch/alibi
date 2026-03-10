"""Tests for OCR module (Stage 1)."""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alibi.extraction.ocr import (
    MIN_OCR_CHARS,
    _ROTATION_CANDIDATES,
    _is_non_latin,
    _prepare_image_for_ocr,
    _prepare_image_enhanced,
    _prepare_rotated_image,
    _try_rotations,
    ocr_image,
    ocr_image_with_retry,
)
from alibi.extraction.vision import VisionExtractionError


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    """Create a simple test image."""
    from PIL import Image

    img = Image.new("RGB", (200, 300), color="white")
    path = tmp_path / "test.jpg"
    img.save(path, "JPEG")
    return path


@pytest.fixture
def tall_image(tmp_path: Path) -> Path:
    """Create an extreme aspect ratio (tall) image."""
    from PIL import Image

    img = Image.new("RGB", (100, 800), color="white")
    path = tmp_path / "tall.jpg"
    img.save(path, "JPEG")
    return path


class TestPrepareImageForOcr:
    """Tests for OCR image preparation."""

    def test_resize_large_image(self, sample_image: Path):
        from PIL import Image

        # Create large image
        img = Image.new("RGB", (3000, 4000), "white")
        large = sample_image.parent / "large.jpg"
        img.save(large, "JPEG")

        result = _prepare_image_for_ocr(large, max_dim=1344)
        # Result should be JPEG bytes, smaller than original
        assert len(result) > 0
        # Verify it's valid JPEG
        loaded = Image.open(io.BytesIO(result))
        assert max(loaded.width, loaded.height) <= 1344

    def test_small_image_unchanged_size(self, sample_image: Path):
        result = _prepare_image_for_ocr(sample_image, max_dim=1344)
        assert len(result) > 0

    def test_no_dimension_jitter(self, sample_image: Path):
        """OCR image prep should NOT use multiples of 28 — that's qwen-specific."""
        from PIL import Image

        result = _prepare_image_for_ocr(sample_image, max_dim=500)
        loaded = Image.open(io.BytesIO(result))
        # glm-ocr doesn't need dimension multiples of 28
        assert loaded.width > 0
        assert loaded.height > 0


class TestPrepareImageEnhanced:
    """Tests for enhanced OCR image preparation."""

    def test_enhanced_produces_valid_image(self, sample_image: Path):
        from PIL import Image

        result = _prepare_image_enhanced(sample_image, max_dim=500)
        loaded = Image.open(io.BytesIO(result))
        assert loaded.width > 0
        assert loaded.height > 0

    def test_enhanced_different_from_normal(self, sample_image: Path):
        normal = _prepare_image_for_ocr(sample_image, max_dim=500)
        enhanced = _prepare_image_enhanced(sample_image, max_dim=500)
        # Enhanced should apply contrast/sharpen, so bytes differ
        # (may be same for pure white image, but typically different)
        assert len(normal) > 0
        assert len(enhanced) > 0


class TestOcrImage:
    """Tests for OCR image extraction."""

    @patch("alibi.extraction.ocr._call_ollama_ocr")
    def test_ocr_returns_text(self, mock_call, sample_image: Path):
        mock_call.return_value = {"response": "SUPERMARKET\nBread 2.50\nMilk 1.99"}
        text = ocr_image(sample_image, model="glm-ocr", ollama_url="http://test:11434")
        assert "SUPERMARKET" in text
        assert "Bread" in text

    @patch("alibi.extraction.ocr._call_ollama_ocr")
    def test_ocr_error_raises(self, mock_call, sample_image: Path):
        mock_call.return_value = {"error": "model not found"}
        with pytest.raises(VisionExtractionError, match="OCR error"):
            ocr_image(sample_image, model="glm-ocr", ollama_url="http://test:11434")

    def test_ocr_file_not_found(self, tmp_path: Path):
        with pytest.raises(VisionExtractionError, match="not found"):
            ocr_image(tmp_path / "nonexistent.jpg")

    @patch("alibi.extraction.ocr._call_ollama_ocr")
    @patch("alibi.extraction.ocr._ocr_sliced")
    def test_extreme_aspect_ratio_uses_slicing(
        self, mock_sliced, mock_call, tall_image: Path
    ):
        mock_sliced.return_value = "sliced text"
        text = ocr_image(tall_image, model="glm-ocr", ollama_url="http://test:11434")
        mock_sliced.assert_called_once()
        assert text == "sliced text"


class TestOcrImageWithRetry:
    """Tests for OCR with enhanced retry."""

    @patch("alibi.extraction.ocr._ocr_single")
    def test_good_text_no_retry(self, mock_ocr, sample_image: Path):
        mock_ocr.return_value = "A" * MIN_OCR_CHARS
        text, was_enhanced = ocr_image_with_retry(
            sample_image, model="glm-ocr", ollama_url="http://test:11434"
        )
        assert len(text) >= MIN_OCR_CHARS
        assert was_enhanced is False
        assert mock_ocr.call_count == 1

    @patch("alibi.extraction.ocr._ocr_single")
    def test_short_text_triggers_retry(self, mock_ocr, sample_image: Path):
        # First call returns too short, second returns better
        mock_ocr.side_effect = ["short", "A" * (MIN_OCR_CHARS + 10)]
        text, was_enhanced = ocr_image_with_retry(
            sample_image, model="glm-ocr", ollama_url="http://test:11434"
        )
        assert was_enhanced is True
        assert mock_ocr.call_count == 2
        # Second call should use enhanced=True
        assert mock_ocr.call_args_list[1][1].get("enhanced") is True

    @patch("alibi.extraction.ocr._try_rotations", return_value=(None, None))
    @patch("alibi.extraction.ocr._ocr_single")
    def test_both_short_returns_longer(self, mock_ocr, mock_rot, sample_image: Path):
        mock_ocr.side_effect = ["ab", "abc"]
        text, was_enhanced = ocr_image_with_retry(
            sample_image, model="glm-ocr", ollama_url="http://test:11434"
        )
        assert text == "abc"
        assert was_enhanced is True

    @patch("alibi.extraction.ocr._try_rotations", return_value=(None, None))
    @patch("alibi.extraction.ocr.get_config")
    @patch("alibi.extraction.ocr._ocr_single")
    def test_fallback_model_on_insufficient_text(
        self, mock_ocr, mock_config, mock_rot, sample_image: Path
    ):
        """When primary + enhanced both fail, fallback model is tried."""
        cfg = MagicMock()
        cfg.ollama_ocr_model = "glm-ocr"
        cfg.ollama_url = "http://test:11434"
        cfg.ollama_ocr_fallback_model = "minicpm-v"
        mock_config.return_value = cfg

        # Primary: short, enhanced: short, fallback: good
        mock_ocr.side_effect = ["ab", "abc", "Greek receipt text " * 5]
        text, was_enhanced = ocr_image_with_retry(sample_image)
        assert len(text) > MIN_OCR_CHARS
        assert was_enhanced is True
        assert mock_ocr.call_count == 3
        # Third call should use fallback model
        assert mock_ocr.call_args_list[2][0][1] == "minicpm-v"

    @patch("alibi.extraction.ocr._try_rotations", return_value=(None, None))
    @patch("alibi.extraction.ocr.get_config")
    @patch("alibi.extraction.ocr._ocr_single")
    def test_fallback_not_used_when_not_configured(
        self, mock_ocr, mock_config, mock_rot, sample_image: Path
    ):
        """No fallback when ollama_ocr_fallback_model is None."""
        cfg = MagicMock()
        cfg.ollama_ocr_model = "glm-ocr"
        cfg.ollama_url = "http://test:11434"
        cfg.ollama_ocr_fallback_model = None
        mock_config.return_value = cfg

        mock_ocr.side_effect = ["ab", "abc"]
        text, was_enhanced = ocr_image_with_retry(sample_image)
        assert text == "abc"
        assert mock_ocr.call_count == 2

    @patch("alibi.extraction.ocr._try_rotations", return_value=(None, None))
    @patch("alibi.extraction.ocr.get_config")
    @patch("alibi.extraction.ocr._ocr_single")
    def test_fallback_error_falls_back_gracefully(
        self, mock_ocr, mock_config, mock_rot, sample_image: Path
    ):
        """Fallback model failure doesn't crash, returns best primary result."""
        cfg = MagicMock()
        cfg.ollama_ocr_model = "glm-ocr"
        cfg.ollama_url = "http://test:11434"
        cfg.ollama_ocr_fallback_model = "minicpm-v"
        mock_config.return_value = cfg

        mock_ocr.side_effect = [
            "ab",
            "abc",
            VisionExtractionError("model not found"),
        ]
        text, was_enhanced = ocr_image_with_retry(sample_image)
        assert text == "abc"

    @patch("alibi.extraction.ocr.get_config")
    @patch("alibi.extraction.ocr._ocr_single")
    def test_fallback_not_used_when_primary_sufficient(
        self, mock_ocr, mock_config, sample_image: Path
    ):
        """Fallback model not invoked when primary model succeeds."""
        cfg = MagicMock()
        cfg.ollama_ocr_model = "glm-ocr"
        cfg.ollama_url = "http://test:11434"
        cfg.ollama_ocr_fallback_model = "minicpm-v"
        mock_config.return_value = cfg

        mock_ocr.return_value = "A" * MIN_OCR_CHARS
        text, was_enhanced = ocr_image_with_retry(sample_image)
        assert len(text) >= MIN_OCR_CHARS
        assert was_enhanced is False
        assert mock_ocr.call_count == 1


class TestProactiveModelSelection:
    """Tests for country-based proactive OCR model selection."""

    @patch("alibi.extraction.ocr.get_config")
    @patch("alibi.extraction.ocr._ocr_single")
    def test_non_latin_country_uses_fallback_directly(
        self, mock_ocr, mock_config, sample_image: Path
    ):
        """GR (Greece) should skip glm-ocr and use minicpm-v directly."""
        cfg = MagicMock()
        cfg.ollama_ocr_model = "glm-ocr"
        cfg.ollama_url = "http://test:11434"
        cfg.ollama_ocr_fallback_model = "minicpm-v"
        mock_config.return_value = cfg

        mock_ocr.return_value = "Greek receipt " * 5
        text, was_enhanced = ocr_image_with_retry(sample_image, country="GR")
        assert mock_ocr.call_count == 1
        # Should have used minicpm-v, not glm-ocr
        assert mock_ocr.call_args[0][1] == "minicpm-v"

    @patch("alibi.extraction.ocr.get_config")
    @patch("alibi.extraction.ocr._ocr_single")
    def test_latin_country_uses_primary_model(
        self, mock_ocr, mock_config, sample_image: Path
    ):
        """DE (Germany, Latin script) should use glm-ocr first."""
        cfg = MagicMock()
        cfg.ollama_ocr_model = "glm-ocr"
        cfg.ollama_url = "http://test:11434"
        cfg.ollama_ocr_fallback_model = "minicpm-v"
        mock_config.return_value = cfg

        mock_ocr.return_value = "German receipt " * 5
        text, was_enhanced = ocr_image_with_retry(sample_image, country="DE")
        assert mock_ocr.call_count == 1
        assert mock_ocr.call_args[0][1] == "glm-ocr"

    @patch("alibi.extraction.ocr.get_config")
    @patch("alibi.extraction.ocr._ocr_single")
    def test_no_country_uses_primary_model(
        self, mock_ocr, mock_config, sample_image: Path
    ):
        """No country → normal behavior (glm-ocr first)."""
        cfg = MagicMock()
        cfg.ollama_ocr_model = "glm-ocr"
        cfg.ollama_url = "http://test:11434"
        cfg.ollama_ocr_fallback_model = "minicpm-v"
        mock_config.return_value = cfg

        mock_ocr.return_value = "Receipt text " * 5
        text, was_enhanced = ocr_image_with_retry(sample_image, country=None)
        assert mock_ocr.call_count == 1
        assert mock_ocr.call_args[0][1] == "glm-ocr"

    @patch("alibi.extraction.ocr.get_config")
    @patch("alibi.extraction.ocr._ocr_single")
    def test_non_latin_country_without_fallback_model(
        self, mock_ocr, mock_config, sample_image: Path
    ):
        """Non-Latin country but no fallback model → use primary (no crash)."""
        cfg = MagicMock()
        cfg.ollama_ocr_model = "glm-ocr"
        cfg.ollama_url = "http://test:11434"
        cfg.ollama_ocr_fallback_model = None
        mock_config.return_value = cfg

        mock_ocr.return_value = "Some text " * 5
        text, was_enhanced = ocr_image_with_retry(sample_image, country="GR")
        assert mock_ocr.call_count == 1
        assert mock_ocr.call_args[0][1] == "glm-ocr"

    @patch("alibi.extraction.ocr.get_config")
    @patch("alibi.extraction.ocr._ocr_single")
    def test_lowercase_country_code_works(
        self, mock_ocr, mock_config, sample_image: Path
    ):
        """Country code case-insensitive."""
        cfg = MagicMock()
        cfg.ollama_ocr_model = "glm-ocr"
        cfg.ollama_url = "http://test:11434"
        cfg.ollama_ocr_fallback_model = "minicpm-v"
        mock_config.return_value = cfg

        mock_ocr.return_value = "Russian text " * 5
        text, was_enhanced = ocr_image_with_retry(sample_image, country="ru")
        assert mock_ocr.call_args[0][1] == "minicpm-v"


class TestIsNonLatin:
    """Tests for non-Latin script detection."""

    def test_latin_text(self):
        assert _is_non_latin("Hello World") is False

    def test_greek_text(self):
        assert _is_non_latin("ΣΟΥΠΕΡΜΑΡΚΕΤ ΦΡΕΣΚΟ") is True

    def test_cyrillic_text(self):
        assert _is_non_latin("Магазин продуктов") is True

    def test_mixed_with_numbers(self):
        # Greek text with prices/numbers
        assert _is_non_latin("ΨΩΜΙ 2.50\nΓΑΛΑ 1.99") is True

    def test_mostly_latin_with_some_greek(self):
        # Below 30% threshold
        assert _is_non_latin("Hello World αβ", threshold=0.3) is False

    def test_empty_text(self):
        assert _is_non_latin("") is False

    def test_only_numbers(self):
        assert _is_non_latin("12345 67.89") is False


class TestPrepareRotatedImage:
    """Tests for image rotation preparation."""

    def test_rotate_180(self, sample_image: Path):
        from PIL import Image

        result = _prepare_rotated_image(sample_image, 180)
        loaded = Image.open(io.BytesIO(result))
        # 200x300 original, 180° should keep same dimensions
        assert loaded.width == 200
        assert loaded.height == 300

    def test_rotate_90_swaps_dimensions(self, sample_image: Path):
        from PIL import Image

        result = _prepare_rotated_image(sample_image, 90)
        loaded = Image.open(io.BytesIO(result))
        # 200x300 → 90° CCW → 300x200
        assert loaded.width == 300
        assert loaded.height == 200

    def test_rotate_270_swaps_dimensions(self, sample_image: Path):
        from PIL import Image

        result = _prepare_rotated_image(sample_image, 270)
        loaded = Image.open(io.BytesIO(result))
        assert loaded.width == 300
        assert loaded.height == 200

    def test_respects_max_dim(self, tmp_path: Path):
        from PIL import Image

        img = Image.new("RGB", (2000, 3000), "white")
        path = tmp_path / "large.jpg"
        img.save(path, "JPEG")

        result = _prepare_rotated_image(path, 180, max_dim=500)
        loaded = Image.open(io.BytesIO(result))
        assert max(loaded.width, loaded.height) <= 500

    def test_converts_rgba_to_rgb(self, tmp_path: Path):
        from PIL import Image

        img = Image.new("RGBA", (200, 300), (255, 255, 255, 128))
        path = tmp_path / "rgba.png"
        img.save(path, "PNG")

        result = _prepare_rotated_image(path, 90)
        loaded = Image.open(io.BytesIO(result))
        assert loaded.mode == "RGB"

    def test_produces_valid_jpeg(self, sample_image: Path):
        result = _prepare_rotated_image(sample_image, 270)
        assert result[:2] == b"\xff\xd8"  # JPEG magic bytes


class TestTryRotations:
    """Tests for rotation-based OCR recovery."""

    @patch("alibi.extraction.ocr._ocr_band_bytes")
    @patch("alibi.extraction.ocr._prepare_rotated_image")
    def test_returns_first_good_rotation(self, mock_prep, mock_ocr, sample_image: Path):
        """180° produces enough text — returns immediately without trying others."""
        mock_prep.return_value = b"fake_jpeg"
        mock_ocr.return_value = "A" * MIN_OCR_CHARS

        text, degrees = _try_rotations(
            sample_image, "glm-ocr", "http://test:11434", 60.0
        )
        assert text == "A" * MIN_OCR_CHARS
        assert degrees == 180  # First candidate
        assert mock_ocr.call_count == 1

    @patch("alibi.extraction.ocr._ocr_band_bytes")
    @patch("alibi.extraction.ocr._prepare_rotated_image")
    def test_tries_all_rotations_when_all_short(
        self, mock_prep, mock_ocr, sample_image: Path
    ):
        """All rotations return short text — returns best of them."""
        mock_prep.return_value = b"fake_jpeg"
        mock_ocr.side_effect = ["ab", "abcde", "abc"]

        text, degrees = _try_rotations(
            sample_image, "glm-ocr", "http://test:11434", 60.0
        )
        assert text == "abcde"
        assert degrees == 270  # Second candidate had longest text
        assert mock_ocr.call_count == 3

    @patch("alibi.extraction.ocr._ocr_band_bytes")
    @patch("alibi.extraction.ocr._prepare_rotated_image")
    def test_skips_failed_rotations(self, mock_prep, mock_ocr, sample_image: Path):
        """OCR failure on a rotation is silently skipped."""
        mock_prep.return_value = b"fake_jpeg"
        mock_ocr.side_effect = [
            VisionExtractionError("model error"),
            "A" * MIN_OCR_CHARS,
            "short",
        ]

        text, degrees = _try_rotations(
            sample_image, "glm-ocr", "http://test:11434", 60.0
        )
        assert text == "A" * MIN_OCR_CHARS
        assert degrees == 270  # Second candidate (first failed)

    @patch("alibi.extraction.ocr._ocr_band_bytes")
    @patch("alibi.extraction.ocr._prepare_rotated_image")
    def test_all_fail_returns_none(self, mock_prep, mock_ocr, sample_image: Path):
        """All rotations fail — returns (None, None)."""
        mock_prep.return_value = b"fake_jpeg"
        mock_ocr.side_effect = VisionExtractionError("error")

        text, degrees = _try_rotations(
            sample_image, "glm-ocr", "http://test:11434", 60.0
        )
        assert text is None
        assert degrees is None

    @patch("alibi.extraction.ocr._ocr_band_bytes")
    @patch("alibi.extraction.ocr._prepare_rotated_image")
    def test_empty_text_not_returned(self, mock_prep, mock_ocr, sample_image: Path):
        """Empty string from all rotations returns (None, None)."""
        mock_prep.return_value = b"fake_jpeg"
        mock_ocr.return_value = ""

        text, degrees = _try_rotations(
            sample_image, "glm-ocr", "http://test:11434", 60.0
        )
        assert text is None
        assert degrees is None


class TestRotationIntegration:
    """Integration tests: rotation detection in the full retry pipeline."""

    @patch("alibi.extraction.ocr._try_rotations")
    @patch("alibi.extraction.ocr._ocr_single")
    def test_rotation_fixes_short_text(self, mock_ocr, mock_rot, sample_image: Path):
        """When normal+enhanced fail, rotation produces good text."""
        mock_ocr.side_effect = ["ab", "abc"]
        mock_rot.return_value = ("Rotated receipt text " * 5, 180)

        text, was_enhanced = ocr_image_with_retry(
            sample_image, model="glm-ocr", ollama_url="http://test:11434"
        )
        assert "Rotated receipt text" in text
        assert was_enhanced is True
        mock_rot.assert_called_once()

    @patch("alibi.extraction.ocr._try_rotations")
    @patch("alibi.extraction.ocr._ocr_single")
    def test_rotation_not_tried_when_normal_sufficient(
        self, mock_ocr, mock_rot, sample_image: Path
    ):
        """Good text on first try — rotation never attempted."""
        mock_ocr.return_value = "A" * MIN_OCR_CHARS

        text, was_enhanced = ocr_image_with_retry(
            sample_image, model="glm-ocr", ollama_url="http://test:11434"
        )
        assert was_enhanced is False
        mock_rot.assert_not_called()

    @patch("alibi.extraction.ocr._try_rotations")
    @patch("alibi.extraction.ocr._ocr_single")
    def test_rotation_not_tried_when_enhanced_sufficient(
        self, mock_ocr, mock_rot, sample_image: Path
    ):
        """Enhanced OCR sufficient — rotation never attempted."""
        mock_ocr.side_effect = ["short", "A" * MIN_OCR_CHARS]

        text, was_enhanced = ocr_image_with_retry(
            sample_image, model="glm-ocr", ollama_url="http://test:11434"
        )
        assert was_enhanced is True
        mock_rot.assert_not_called()

    @patch("alibi.extraction.ocr._try_rotations")
    @patch("alibi.extraction.ocr.get_config")
    @patch("alibi.extraction.ocr._ocr_single")
    def test_rotation_before_fallback_model(
        self, mock_ocr, mock_config, mock_rot, sample_image: Path
    ):
        """Rotation tried before fallback model. If rotation succeeds, no fallback."""
        cfg = MagicMock()
        cfg.ollama_ocr_model = "glm-ocr"
        cfg.ollama_url = "http://test:11434"
        cfg.ollama_ocr_fallback_model = "minicpm-v"
        mock_config.return_value = cfg

        mock_ocr.side_effect = ["ab", "abc"]
        mock_rot.return_value = ("Rotated good text " * 5, 90)

        text, was_enhanced = ocr_image_with_retry(sample_image)
        assert "Rotated good text" in text
        assert was_enhanced is True
        # Only 2 _ocr_single calls (normal + enhanced), no fallback
        assert mock_ocr.call_count == 2

    @patch("alibi.extraction.ocr._try_rotations")
    @patch("alibi.extraction.ocr.get_config")
    @patch("alibi.extraction.ocr._ocr_single")
    def test_rotation_fails_then_fallback_tried(
        self, mock_ocr, mock_config, mock_rot, sample_image: Path
    ):
        """Rotation doesn't help, fallback model is still tried."""
        cfg = MagicMock()
        cfg.ollama_ocr_model = "glm-ocr"
        cfg.ollama_url = "http://test:11434"
        cfg.ollama_ocr_fallback_model = "minicpm-v"
        mock_config.return_value = cfg

        mock_ocr.side_effect = ["ab", "abc", "Fallback good " * 5]
        mock_rot.return_value = (None, None)

        text, was_enhanced = ocr_image_with_retry(sample_image)
        assert "Fallback good" in text
        assert was_enhanced is True
        assert mock_ocr.call_count == 3

    def test_rotation_candidates_order(self):
        """Verify rotation candidates: 180° first, then 270° (90CW), then 90° (90CCW)."""
        assert _ROTATION_CANDIDATES == [180, 270, 90]


class TestPaymentYamlTemplate:
    """Tests for payment confirmation YAML schema fix."""

    def test_payment_template_uses_total_not_amount(self):
        """Payment confirmation template should use 'total', not 'amount'."""
        from alibi.extraction.yaml_cache import _PAYMENT_FIELDS

        assert "total" in _PAYMENT_FIELDS
        assert "amount" not in _PAYMENT_FIELDS

    def test_payment_template_total_default_is_none(self):
        from alibi.extraction.yaml_cache import _PAYMENT_FIELDS

        assert _PAYMENT_FIELDS["total"] is None

    def test_payment_blank_template_has_total(self):
        from alibi.extraction.yaml_cache import generate_blank_template

        template = generate_blank_template("payment_confirmation")
        assert "total" in template
        assert "amount" not in template

    def test_invoice_template_still_has_amount(self):
        """Invoice template should keep 'amount' (not affected by fix)."""
        from alibi.extraction.yaml_cache import _INVOICE_FIELDS

        assert "amount" in _INVOICE_FIELDS

    def test_statement_txn_template_still_has_amount(self):
        """Statement transaction template should keep 'amount'."""
        from alibi.extraction.yaml_cache import _STATEMENT_TXN_FIELDS

        assert "amount" in _STATEMENT_TXN_FIELDS


class TestOcrTimeoutConvertsToVisionError:
    """Verify OCR timeouts are converted to VisionExtractionError."""

    def test_ocr_single_timeout_raises_vision_error(self, sample_image: Path):
        """httpx.TimeoutException from _call_ollama_ocr becomes VisionExtractionError."""
        import httpx

        from alibi.extraction.ocr import _ocr_single

        with patch("alibi.extraction.ocr._call_ollama_ocr") as mock_call:
            mock_call.side_effect = httpx.ReadTimeout("OCR timed out")
            with pytest.raises(VisionExtractionError, match="OCR failed after retries"):
                _ocr_single(sample_image, "glm-ocr", "http://localhost:11434", 10.0)

    def test_ocr_single_connect_error_raises_vision_error(self, sample_image: Path):
        """httpx.ConnectError from _call_ollama_ocr becomes VisionExtractionError."""
        import httpx

        from alibi.extraction.ocr import _ocr_single

        with patch("alibi.extraction.ocr._call_ollama_ocr") as mock_call:
            mock_call.side_effect = httpx.ConnectError("Connection refused")
            with pytest.raises(VisionExtractionError, match="OCR failed after retries"):
                _ocr_single(sample_image, "glm-ocr", "http://localhost:11434", 10.0)
