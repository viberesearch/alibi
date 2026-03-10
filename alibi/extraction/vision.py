"""Ollama Vision extraction for images.

Three-stage pipeline (default):
  Stage 1: glm-ocr (fast OCR) → raw text
  Stage 2: Heuristic text parser → pre-structured data
  Stage 3: qwen3:8b (text-only) → correction/enrichment JSON

Falls back to two-stage (skip parser) when heuristic confidence < 0.3,
and to legacy single-stage vision (qwen3-vl:30b) when verification fails.
"""

import base64
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, cast

import httpx

from alibi.config import get_config
from alibi.extraction.prompts import get_prompt_for_type
from alibi.extraction.templates import ParserHints
from alibi.utils.retry import with_retry

logger = logging.getLogger(__name__)

# Retry configuration for Ollama API calls
OLLAMA_RETRY_EXCEPTIONS = (httpx.TimeoutException, httpx.ConnectError)

# Default timeout for LLM calls (seconds). Override with ALIBI_LLM_TIMEOUT env var.
DEFAULT_LLM_TIMEOUT = float(os.environ.get("ALIBI_LLM_TIMEOUT", "600"))


class VisionExtractionError(Exception):
    """Error during vision extraction."""

    pass


# qwen vision models need dimensions as multiples of 28 and certain dimension
# combinations trigger GGML assertion failures. We try progressively
# smaller sizes until one works.
DIMENSION_MULTIPLE = 28
# Ordered from largest (best quality) to smallest (safest)
MAX_DIMENSION_ATTEMPTS = [1344, 1008, 756]

# Slicing configuration for extreme aspect ratio images.
# Images exceeding this ratio (max/min dimension) are sliced into bands.
EXTREME_ASPECT_RATIO = 3.0
# Target aspect ratio for each band (max/min of band dimensions)
MAX_BAND_ASPECT_RATIO = 2.0
# Fraction of band height/width that overlaps with the adjacent band
SLICE_OVERLAP_RATIO = 0.15


def _compute_dimensions(
    orig_w: int, orig_h: int, max_dimension: int
) -> tuple[int, int]:
    """Compute target dimensions fitting within max_dimension, rounded to multiples of 28."""
    w, h = orig_w, orig_h
    if max(w, h) > max_dimension:
        scale = max_dimension / max(w, h)
        w = int(w * scale)
        h = int(h * scale)
    w = max((w // DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE, DIMENSION_MULTIPLE)
    h = max((h // DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE, DIMENSION_MULTIPLE)
    return w, h


def _dimension_jitter(w: int, h: int) -> list[tuple[int, int]]:
    """Generate nearby dimension variants to work around GGML assertion failures.

    Tries +28 and +56 offsets on width, height, and both axes.
    Some dimension combos trigger GGML bugs; a small shift often avoids them.
    """
    seen = set()
    variants = []
    m = DIMENSION_MULTIPLE
    for dw, dh in [(m, 0), (0, m), (m, m), (2 * m, 0), (0, 2 * m), (2 * m, 2 * m)]:
        nw = w + dw
        nh = h + dh
        if (nw, nh) not in seen and nw >= m and nh >= m:
            seen.add((nw, nh))
            variants.append((nw, nh))
    return variants


def _needs_slicing(width: int, height: int) -> bool:
    """Check if image has extreme aspect ratio requiring sliced extraction."""
    if width <= 0 or height <= 0:
        return False
    ratio = max(width, height) / min(width, height)
    return ratio > EXTREME_ASPECT_RATIO


def _create_image_bands(image_path: Path) -> list[bytes]:
    """Slice an extreme aspect ratio image into overlapping JPEG bands.

    For tall images, creates horizontal bands. For wide images, vertical bands.
    Each band has aspect ratio <= MAX_BAND_ASPECT_RATIO with SLICE_OVERLAP_RATIO
    overlap between adjacent bands.

    Returns list of JPEG bytes for each band, in reading order.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 200_000_000

    with Image.open(image_path) as img:
        w, h = img.width, img.height

        if h >= w:
            # Tall image — slice into horizontal bands
            band_h = int(w * MAX_BAND_ASPECT_RATIO)
            overlap = int(band_h * SLICE_OVERLAP_RATIO)
            return _slice_axis(img, w, h, band_h, overlap, axis="vertical")
        else:
            # Wide image — slice into vertical bands
            band_w = int(h * MAX_BAND_ASPECT_RATIO)
            overlap = int(band_w * SLICE_OVERLAP_RATIO)
            return _slice_axis(img, w, h, band_w, overlap, axis="horizontal")


def _slice_axis(
    img: Any,
    w: int,
    h: int,
    band_size: int,
    overlap: int,
    axis: str,
) -> list[bytes]:
    """Slice image along one axis into overlapping bands.

    Args:
        img: PIL Image object.
        w: Image width.
        h: Image height.
        band_size: Size of each band along the slicing axis.
        overlap: Overlap in pixels between adjacent bands.
        axis: "vertical" slices top-to-bottom, "horizontal" slices left-to-right.
    """
    from PIL import Image as _PILImage

    bands: list[bytes] = []
    total = h if axis == "vertical" else w
    pos = 0

    while pos < total:
        end = min(pos + band_size, total)

        if axis == "vertical":
            band = img.crop((0, pos, w, end))
        else:
            band = img.crop((pos, 0, end, h))

        if band.mode in ("RGBA", "P"):
            band = band.convert("RGB")

        buf = io.BytesIO()
        band.save(buf, format="JPEG", quality=90)
        bands.append(buf.getvalue())

        if end >= total:
            break
        pos = end - overlap

    return bands


def _prepare_image(
    image_path: Path, max_dimension: int, target_dims: tuple[int, int] | None = None
) -> bytes:
    """Resize image to fit within max_dimension, rounding to multiples of 28.

    Args:
        image_path: Path to image file.
        max_dimension: Maximum dimension for longest side.
        target_dims: If provided, use these exact (w, h) instead of computing.

    Returns JPEG bytes. Falls back to raw file bytes if PIL fails.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 200_000_000

    try:
        with Image.open(image_path) as raw_img:
            result: Image.Image = raw_img
            if target_dims:
                w, h = target_dims
            else:
                w, h = _compute_dimensions(result.width, result.height, max_dimension)

            logger.debug(
                f"Preparing {image_path.name}: {result.width}x{result.height} -> {w}x{h}"
            )
            result = result.resize((w, h), Image.LANCZOS)  # type: ignore[attr-defined]

            buf = io.BytesIO()
            if result.mode in ("RGBA", "P"):
                result = result.convert("RGB")
            result.save(buf, format="JPEG", quality=90)
            return buf.getvalue()
    except Exception as e:
        logger.debug(f"PIL could not process {image_path.name}: {e}, using raw bytes")
        return image_path.read_bytes()


def extract_json_from_response(response_text: str) -> dict[str, Any]:
    """Extract JSON from LLM response text.

    Handles responses that may have markdown code blocks or extra text.
    """
    # Try to find JSON in code blocks first
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)
    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise VisionExtractionError(
                f"No JSON found in response: {response_text[:200]}"
            )

    try:
        return cast(dict[str, Any], json.loads(json_str))
    except json.JSONDecodeError:
        # Clean control characters and retry
        cleaned = re.sub(
            r"[\x00-\x1f\x7f]", lambda m: " " if m.group() in "\r\n\t" else "", json_str
        )
        try:
            return cast(dict[str, Any], json.loads(cleaned))
        except json.JSONDecodeError as e2:
            raise VisionExtractionError(f"Invalid JSON in response: {e2}") from e2


# ------------------------------------------------------------------
# Two-stage pipeline entry point
# ------------------------------------------------------------------


def extract_from_image(
    image_path: Path,
    doc_type: str | None = "receipt",
    ollama_url: str | None = None,
    model: str | None = None,
    timeout: float = DEFAULT_LLM_TIMEOUT,
    skip_llm_threshold: float | None = None,
    country: str | None = None,
    hints: ParserHints | None = None,
) -> dict[str, Any]:
    """Extract data from image using the three-stage pipeline.

    Stage 1: OCR via glm-ocr → raw text
    Stage 1b: If doc_type is None, classify from OCR text (post-OCR)
    Stage 2: Heuristic parser → pre-structured data (deterministic)
    Stage 3: LLM correction/enrichment → final JSON
    Verify: arithmetic checks → confidence score
    Fallback: if parser confidence < 0.3, skip to full LLM structuring
              if verification confidence < 0.3, use legacy qwen3-vl

    The raw OCR text is injected into the result as ``raw_text`` if not
    already present.

    Args:
        image_path: Path to the image file
        doc_type: Type of document, or None to auto-classify from OCR text
        ollama_url: Ollama API URL (defaults to config)
        model: Model to use for legacy fallback (defaults to config)
        timeout: Request timeout in seconds

    Returns:
        Extracted data as dictionary (includes ``raw_text`` from OCR)

    Raises:
        VisionExtractionError: If extraction fails
    """
    if not image_path.exists():
        raise VisionExtractionError(f"Image file not found: {image_path}")

    config = get_config()
    ollama_url = ollama_url or config.ollama_url

    # Import two-stage modules (avoids circular imports at module level)
    from alibi.extraction.ocr import ocr_image_with_retry
    from alibi.extraction.structurer import structure_ocr_text
    from alibi.extraction.verification import (
        FALLBACK_THRESHOLD,
        build_emphasis_prompt,
        verify_extraction,
    )

    try:
        import time

        t_start = time.monotonic()

        # Stage 1: OCR
        ocr_text, was_enhanced = ocr_image_with_retry(
            image_path,
            ollama_url=ollama_url,
            timeout=config.ocr_timeout,
            country=country,
        )
        t_ocr = time.monotonic()
        logger.info(
            f"Three-stage OCR for {image_path.name}: {len(ocr_text)} chars"
            f"{' (enhanced)' if was_enhanced else ''}"
            f" [{t_ocr - t_start:.1f}s]"
        )

        # Stage 1b: Post-OCR classification (when doc_type not pre-determined)
        if doc_type is None:
            from alibi.extraction.text_parser import classify_ocr_text

            doc_type = classify_ocr_text(ocr_text)
            logger.info(f"Post-OCR classification for {image_path.name}: {doc_type}")

        # Stage 2: Heuristic parse
        from alibi.extraction.text_parser import parse_ocr_text

        parse_result = None
        try:
            parse_result = parse_ocr_text(ocr_text, doc_type=doc_type, hints=hints)
            t_parse = time.monotonic()
            logger.info(
                f"Heuristic parser for {image_path.name}: "
                f"confidence={parse_result.confidence:.2f}, "
                f"items={parse_result.line_item_count}, "
                f"gaps={parse_result.gaps[:5]}"
                f" [{(t_parse - t_ocr) * 1000:.1f}ms]"
            )
        except Exception as e:
            logger.warning(
                f"Heuristic parser failed for {image_path.name}: {e}, "
                f"falling through to full LLM structuring"
            )

        # Stage 3: LLM structuring (correction, full, or skip)
        t_llm_start = time.monotonic()
        skip_threshold = (
            skip_llm_threshold
            if skip_llm_threshold is not None
            else config.skip_llm_threshold
        )
        skipped_llm = False

        if (
            parse_result
            and parse_result.confidence >= skip_threshold
            and doc_type in ("receipt", "payment_confirmation", "statement", "invoice")
        ):
            # High-confidence parser output — skip LLM entirely
            extracted = dict(parse_result.data)
            extracted["_parser_confidence"] = parse_result.confidence
            extracted["_pipeline"] = "parser_only"
            skipped_llm = True
            logger.info(
                f"Skipping LLM for {image_path.name}: "
                f"parser confidence {parse_result.confidence:.2f} "
                f">= threshold {skip_threshold}"
            )
        elif parse_result and parse_result.confidence >= 0.3:
            # Parser extracted meaningful data — route via available backend
            if config.gemini_extraction_enabled:
                # Gemini does comprehensive extraction; skip micro-prompts
                # to avoid loading Ollama models (VRAM contention with OCR)
                extracted = structure_ocr_text(
                    ocr_text,
                    doc_type=doc_type,
                    ollama_url=ollama_url,
                    timeout=timeout,
                )
                extracted["_parser_confidence"] = parse_result.confidence
                extracted["_pipeline"] = "three_stage_gemini"
            else:
                # Ollama path: try micro-prompts first, then correction
                from alibi.extraction.micro_prompts import run_micro_prompts

                micro_result = run_micro_prompts(
                    parse_result,
                    ocr_text,
                    doc_type,
                    ollama_url=ollama_url,
                    timeout=timeout,
                )

                if micro_result is not None:
                    extracted = micro_result
                else:
                    from alibi.extraction.prompts import get_correction_prompt

                    correction_prompt = get_correction_prompt(
                        parse_result.data,
                        ocr_text,
                        doc_type,
                        field_confidence=parse_result.field_confidence,
                        regions=parse_result.regions,
                    )
                    extracted = structure_ocr_text(
                        ocr_text,
                        doc_type=doc_type,
                        ollama_url=ollama_url,
                        timeout=timeout,
                        emphasis_prompt=correction_prompt,
                    )
                    extracted["_parser_confidence"] = parse_result.confidence
                    extracted["_pipeline"] = "three_stage"
        else:
            # Parser couldn't extract much — full LLM structuring
            extracted = structure_ocr_text(
                ocr_text,
                doc_type=doc_type,
                ollama_url=ollama_url,
                timeout=timeout,
            )
            extracted["_pipeline"] = "two_stage"

        t_llm = time.monotonic()
        if not skipped_llm:
            logger.info(
                f"LLM structuring for {image_path.name}: "
                f"[{t_llm - t_llm_start:.1f}s]"
            )

        # Verify (skip verification retry when LLM was skipped)
        verification = verify_extraction(extracted, ocr_text=ocr_text)
        logger.info(
            f"Two-stage verification for {image_path.name}: "
            f"confidence={verification.confidence:.2f}, "
            f"flags={verification.flags}"
        )

        # If verification recommends re-run, try Stage 2 again with emphasis
        # (only when LLM was used — no point retrying if we skipped it)
        if verification.rerun_recommended and not skipped_llm:
            failed_checks = {
                k: v for k, v in verification.check_scores.items() if v < 0.7
            }
            emphasis = build_emphasis_prompt(
                ocr_text,
                doc_type,
                failed_checks,
                prompt_mode=config.prompt_mode,
            )
            extracted_retry = structure_ocr_text(
                ocr_text,
                doc_type=doc_type,
                ollama_url=ollama_url,
                timeout=timeout,
                emphasis_prompt=emphasis,
            )
            verification_retry = verify_extraction(extracted_retry, ocr_text=ocr_text)
            logger.info(
                f"Two-stage retry for {image_path.name}: "
                f"confidence={verification_retry.confidence:.2f}"
            )

            # Use retry result if it's better
            if verification_retry.confidence > verification.confidence:
                extracted = extracted_retry
                verification = verification_retry

        # Gap-fill from parser: if parser extracted fields the LLM missed,
        # carry them forward (e.g. amount_tendered, change_due)
        if parse_result and not skipped_llm:
            _PARSER_GAP_FILL_FIELDS = (
                "amount_tendered",
                "change_due",
            )
            for field in _PARSER_GAP_FILL_FIELDS:
                val = parse_result.data.get(field)
                if val is not None and field not in extracted:
                    extracted[field] = val

        # Inject raw_text from OCR if not already present
        if not extracted.get("raw_text"):
            extracted["raw_text"] = ocr_text

        # Store confidence + timing + OCR metadata
        extracted["_two_stage_confidence"] = verification.confidence
        extracted["_ocr_enhanced"] = was_enhanced
        t_total = time.monotonic() - t_start
        extracted["_timing"] = {
            "ocr_s": round(t_ocr - t_start, 2),
            "parser_ms": round((t_llm_start - t_ocr) * 1000, 1),
            "llm_s": round(t_llm - t_llm_start, 2),
            "total_s": round(t_total, 2),
        }
        logger.info(
            f"Pipeline total for {image_path.name}: {t_total:.1f}s "
            f"(OCR={t_ocr - t_start:.1f}s, "
            f"parser={(t_llm_start - t_ocr) * 1000:.0f}ms, "
            f"LLM={t_llm - t_llm_start:.1f}s)"
        )

        # If still very low confidence, fall back to vision
        if verification.confidence < FALLBACK_THRESHOLD:
            logger.warning(
                f"Two-stage confidence {verification.confidence:.2f} < "
                f"{FALLBACK_THRESHOLD} for {image_path.name}, "
                f"falling back to vision"
            )
            return _fallback_vision(
                image_path, doc_type or "receipt", ollama_url, model, timeout
            )

        return extracted

    except VisionExtractionError:
        # Two-stage failed entirely — fall back to vision
        logger.warning(
            f"Two-stage pipeline failed for {image_path.name}, "
            f"falling back to vision"
        )
        return _fallback_vision(
            image_path, doc_type or "receipt", ollama_url, model, timeout
        )
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        # OCR timeout that wasn't converted — defense-in-depth
        logger.warning(
            f"OCR timeout for {image_path.name}: {e}, " f"falling back to vision"
        )
        return _fallback_vision(
            image_path, doc_type or "receipt", ollama_url, model, timeout
        )


def extract_from_images(
    image_paths: list[Path],
    doc_type: str | None = "receipt",
    ollama_url: str | None = None,
    model: str | None = None,
    timeout: float = 180.0,
    skip_llm_threshold: float | None = None,
    country: str | None = None,
    hints: ParserHints | None = None,
) -> dict[str, Any]:
    """Extract data from multiple images (pages) of a single document.

    Uses three-stage pipeline matching single-image extraction:
    1. OCR each page separately with retry/fallback
    2. Heuristic parser on combined text (deterministic, instant)
    3. LLM correction only if parser confidence is low

    Falls back to single-page extraction if only one image.

    Args:
        image_paths: Paths to image files, sorted by page order
        doc_type: Type of document, or None to auto-classify from OCR text
        ollama_url: Ollama API URL (defaults to config)
        model: Model to use (defaults to config)
        timeout: Request timeout in seconds (higher default for multi-page)
        skip_llm_threshold: Override for LLM skip threshold (defaults to config)
        country: ISO country code for OCR model selection

    Returns:
        Extracted data as dictionary

    Raises:
        VisionExtractionError: If extraction fails
    """
    if not image_paths:
        raise VisionExtractionError("No image paths provided")

    if len(image_paths) == 1:
        return extract_from_image(
            image_paths[0],
            doc_type,
            ollama_url,
            model,
            timeout,
            skip_llm_threshold=skip_llm_threshold,
            country=country,
            hints=hints,
        )

    config = get_config()
    ollama_url = ollama_url or config.ollama_url

    from alibi.extraction.ocr import ocr_image_with_retry
    from alibi.extraction.structurer import structure_ocr_text

    # Stage 1: OCR each page separately with retry/fallback
    page_texts: list[str] = []
    for i, path in enumerate(image_paths):
        if not path.exists():
            raise VisionExtractionError(f"Image file not found: {path}")
        try:
            text, was_enhanced = ocr_image_with_retry(
                path, ollama_url=ollama_url, timeout=config.ocr_timeout, country=country
            )
            page_texts.append(f"--- PAGE {i + 1} ---\n{text}")
            logger.debug(
                f"OCR page {i + 1}: {len(text)} chars"
                f"{' (enhanced)' if was_enhanced else ''}"
            )
        except (VisionExtractionError, httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning(f"OCR page {i + 1} failed: {e}")

    if not page_texts:
        raise VisionExtractionError("All page OCR attempts failed")

    combined_text = "\n\n".join(page_texts)

    # Post-OCR classification when type not pre-determined
    if doc_type is None:
        from alibi.extraction.text_parser import classify_ocr_text

        doc_type = classify_ocr_text(combined_text)
        logger.info(f"Post-OCR classification for multi-image: {doc_type}")
    if doc_type is None:
        doc_type = "receipt"

    # Stage 2: Heuristic parse
    from alibi.extraction.text_parser import parse_ocr_text

    skip_threshold = (
        skip_llm_threshold
        if skip_llm_threshold is not None
        else config.skip_llm_threshold
    )

    parse_result = None
    try:
        parse_result = parse_ocr_text(combined_text, doc_type=doc_type, hints=hints)
        logger.info(
            f"Multi-image parser: confidence={parse_result.confidence:.2f}, "
            f"items={parse_result.line_item_count}, "
            f"gaps={parse_result.gaps[:5]}"
        )
    except Exception as e:
        logger.warning(f"Multi-image parser failed: {e}, falling through to LLM")

    # Stage 3: LLM structuring based on parser confidence
    skippable_types = ("receipt", "payment_confirmation", "statement", "invoice")

    if (
        parse_result
        and parse_result.confidence >= skip_threshold
        and doc_type in skippable_types
    ):
        extracted = dict(parse_result.data)
        extracted["_parser_confidence"] = parse_result.confidence
        extracted["_pipeline"] = "multi_image_parser_only"
        if not extracted.get("raw_text"):
            extracted["raw_text"] = combined_text
        logger.info(
            f"Skipping LLM for multi-image: "
            f"parser confidence {parse_result.confidence:.2f} "
            f">= threshold {skip_threshold}"
        )
        return extracted

    if parse_result and parse_result.confidence >= 0.3:
        config = get_config()
        if config.gemini_extraction_enabled:
            # Gemini does comprehensive extraction; skip micro-prompts
            # to avoid loading Ollama models (VRAM contention with OCR)
            try:
                extracted = structure_ocr_text(
                    combined_text,
                    doc_type=doc_type,
                    ollama_url=ollama_url,
                    timeout=timeout,
                )
                extracted["_parser_confidence"] = parse_result.confidence
                extracted["_pipeline"] = "multi_image_three_stage_gemini"
                if not extracted.get("raw_text"):
                    extracted["raw_text"] = combined_text
                return extracted
            except Exception as e:
                logger.warning(
                    f"Multi-image Gemini extraction failed: {e}, " "using parser result"
                )
                extracted = dict(parse_result.data)
                extracted["_parser_confidence"] = parse_result.confidence
                extracted["_pipeline"] = "multi_image_parser_fallback"
                extracted["raw_text"] = combined_text
                return extracted
        else:
            # Ollama path: try micro-prompts first, then correction
            from alibi.extraction.micro_prompts import run_micro_prompts

            micro_result = run_micro_prompts(
                parse_result,
                combined_text,
                doc_type,
                ollama_url=ollama_url,
                timeout=timeout,
            )
            if micro_result is not None:
                if not micro_result.get("raw_text"):
                    micro_result["raw_text"] = combined_text
                micro_result["_pipeline"] = "multi_image_three_stage"
                return micro_result

            # Correction prompt fallback
            from alibi.extraction.prompts import get_correction_prompt

            correction_prompt = get_correction_prompt(
                parse_result.data,
                combined_text,
                doc_type,
                field_confidence=parse_result.field_confidence,
                regions=parse_result.regions,
            )
            try:
                extracted = structure_ocr_text(
                    combined_text,
                    doc_type=doc_type,
                    ollama_url=ollama_url,
                    timeout=timeout,
                    emphasis_prompt=correction_prompt,
                )
                extracted["_parser_confidence"] = parse_result.confidence
                extracted["_pipeline"] = "multi_image_three_stage"
                if not extracted.get("raw_text"):
                    extracted["raw_text"] = combined_text
                return extracted
            except Exception as e:
                logger.warning(
                    f"Multi-image LLM correction failed: {e}, " "using parser result"
                )
                extracted = dict(parse_result.data)
                extracted["_parser_confidence"] = parse_result.confidence
                extracted["_pipeline"] = "multi_image_parser_fallback"
                extracted["raw_text"] = combined_text
                return extracted

    # Low confidence or parser failed — full LLM structuring
    extracted = structure_ocr_text(
        combined_text,
        doc_type=doc_type,
        ollama_url=ollama_url,
        timeout=timeout,
    )
    extracted["_pipeline"] = "multi_image_llm_full"

    if not extracted.get("raw_text"):
        extracted["raw_text"] = combined_text

    return extracted


# ------------------------------------------------------------------
# Fallback vision: Gemini Vision (when enabled) or legacy Ollama
# ------------------------------------------------------------------


def _fallback_vision(
    image_path: Path,
    doc_type: str = "receipt",
    ollama_url: str | None = None,
    model: str | None = None,
    timeout: float = DEFAULT_LLM_TIMEOUT,
) -> dict[str, Any]:
    """Fallback when 3-stage pipeline fails or produces low confidence.

    Routes to Gemini Vision when enabled, otherwise uses legacy Ollama vision.
    """
    config = get_config()
    if config.gemini_extraction_enabled:
        try:
            from alibi.extraction.gemini_structurer import (
                GeminiExtractionError,
                extract_from_image_gemini,
            )

            logger.info(f"Using Gemini Vision fallback for {image_path.name}")
            return extract_from_image_gemini(str(image_path), doc_type=doc_type)
        except GeminiExtractionError as e:
            logger.warning(
                "Gemini Vision failed for %s, falling back to legacy: %s",
                image_path.name,
                e,
            )
        except Exception as e:
            logger.warning(
                "Gemini Vision unexpected error for %s, falling back to legacy: %s",
                image_path.name,
                e,
            )

    return _extract_from_image_legacy(image_path, doc_type, ollama_url, model, timeout)


def _extract_from_image_legacy(
    image_path: Path,
    doc_type: str = "receipt",
    ollama_url: str | None = None,
    model: str | None = None,
    timeout: float = DEFAULT_LLM_TIMEOUT,
) -> dict[str, Any]:
    """Legacy extraction: single vision model does OCR + structuring.

    Used as fallback when two-stage pipeline fails or produces low
    confidence results. Tall-image slicing is handled at the OCR level
    (ocr.py), so this simply delegates to single-image extraction.
    """
    return _extract_single_image_legacy(
        image_path, doc_type, ollama_url, model, timeout
    )


def _extract_single_image_legacy(
    image_path: Path,
    doc_type: str = "receipt",
    ollama_url: str | None = None,
    model: str | None = None,
    timeout: float = DEFAULT_LLM_TIMEOUT,
) -> dict[str, Any]:
    """Extract data from a single image with dimension jitter (legacy).

    Tries progressively smaller image dimensions if the vision model
    returns 500 errors (qwen vision model GGML assertion bug).
    """
    config = get_config()
    ollama_url = ollama_url or config.ollama_url
    model = model or config.ollama_model

    prompt = get_prompt_for_type(doc_type, version=2, mode=config.prompt_mode)
    last_error: Exception | None = None

    # Get original dimensions for jitter computation
    orig_dims: tuple[int, int] | None = None
    try:
        from PIL import Image as _PILImage

        with _PILImage.open(image_path) as _img:
            orig_dims = (_img.width, _img.height)
    except Exception:
        pass

    for max_dim in MAX_DIMENSION_ATTEMPTS:
        dims_to_try: list[tuple[int, int] | None]
        if orig_dims:
            base_dims = _compute_dimensions(*orig_dims, max_dim)
            dims_to_try = [base_dims, *_dimension_jitter(*base_dims)]
        else:
            dims_to_try = [None]

        for dims in dims_to_try:
            image_bytes = _prepare_image(image_path, max_dim, target_dims=dims)
            image_b64 = base64.b64encode(image_bytes).decode()

            try:
                result = _call_ollama_vision(
                    ollama_url, model, prompt, [image_b64], timeout
                )
            except VisionExtractionError as e:
                if "500" in str(e):
                    dim_str = f"{dims[0]}x{dims[1]}" if dims else f"max_dim={max_dim}"
                    logger.warning(
                        f"Vision model 500 at {dim_str} for "
                        f"{image_path.name}, trying next dimensions"
                    )
                    last_error = e
                    continue
                raise

            if "error" in result:
                raise VisionExtractionError(f"Ollama error: {result['error']}")

            if "response" not in result:
                raise VisionExtractionError(f"Unexpected response format: {result}")

            return extract_json_from_response(result["response"])

    raise last_error or VisionExtractionError(
        f"All dimension attempts failed for {image_path}"
    )


# ------------------------------------------------------------------
# Shared Ollama API helpers
# ------------------------------------------------------------------


@with_retry(max_attempts=3, base_delay=2.0, exceptions=OLLAMA_RETRY_EXCEPTIONS)
def _call_ollama_vision(
    ollama_url: str,
    model: str,
    prompt: str,
    images_b64: list[str],
    timeout: float,
) -> dict[str, Any]:
    """Make HTTP request to Ollama Vision API with retry support.

    Args:
        ollama_url: Ollama API URL
        model: Model name
        prompt: Prompt text
        images_b64: List of base64-encoded images
        timeout: Request timeout
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": images_b64,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
            response.raise_for_status()
        return cast(dict[str, Any], response.json())
    except httpx.HTTPStatusError as e:
        raise VisionExtractionError(f"HTTP error: {e.response.status_code}") from e
    except httpx.RequestError as e:
        if isinstance(e, (httpx.TimeoutException, httpx.ConnectError)):
            raise  # Re-raise for retry
        raise VisionExtractionError(f"Request failed: {e}") from e


# ------------------------------------------------------------------
# Document type detection (uses OCR model for speed)
# ------------------------------------------------------------------


def detect_document_type(image_path: Path) -> str:
    """Detect document type from image using OCR vision model.

    Uses the lightweight OCR model (glm-ocr) for faster classification.
    Falls back to the main vision model if OCR model fails.

    Returns one of: receipt, invoice, statement, warranty, contract,
    payment_confirmation, other
    """
    config = get_config()

    if not image_path.exists():
        raise VisionExtractionError(f"Image file not found: {image_path}")

    prompt = """Look at this document image and determine its type.
Return ONLY one of these words: receipt, invoice, statement, warranty, contract, payment_confirmation, other

Use "payment_confirmation" for card terminal slips, bank transfer confirmations,
or similar payment-only documents that show payment details but no item list.
Use "receipt" for documents with an itemized list of purchased goods/services.
Use "contract" for agreements, service contracts, leases, or subscription agreements.
Use "warranty" for warranty certificates, warranty cards, or guarantee documents.

Do not include any other text."""

    from alibi.extraction.ocr import _prepare_image_for_ocr

    # Use OCR model for type detection (faster, no GGML jitter needed)
    try:
        image_bytes = _prepare_image_for_ocr(image_path, max_dim=1344)
        image_b64 = base64.b64encode(image_bytes).decode()
        result = _call_ollama_vision(
            config.ollama_url,
            config.ollama_ocr_model,
            prompt,
            [image_b64],
            30.0,
        )
    except VisionExtractionError as e:
        logger.warning(
            f"OCR model type detection failed: {e}, trying legacy vision model"
        )
        # Fallback to legacy vision model with jitter
        return _detect_document_type_legacy(image_path)
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        logger.warning(f"Failed to detect document type: {e}")
        return "other"

    if result is None:
        return "other"

    response_text = result.get("response", "").strip().lower()

    valid_types = {
        "receipt",
        "invoice",
        "statement",
        "warranty",
        "contract",
        "payment_confirmation",
        "other",
    }
    for doc_type in valid_types:
        if doc_type in response_text:
            return doc_type

    return "other"


def _detect_document_type_legacy(image_path: Path) -> str:
    """Detect document type using legacy vision model.

    Tries progressively smaller image sizes (no jitter needed — the short
    type-detection prompt is much less susceptible to GGML assertion failures
    than the full extraction prompt).
    """
    config = get_config()

    prompt = """Look at this document image and determine its type.
Return ONLY one of these words: receipt, invoice, statement, warranty, contract, payment_confirmation, other

Use "payment_confirmation" for card terminal slips, bank transfer confirmations,
or similar payment-only documents that show payment details but no item list.
Use "receipt" for documents with an itemized list of purchased goods/services.
Use "contract" for agreements, service contracts, leases, or subscription agreements.
Use "warranty" for warranty certificates, warranty cards, or guarantee documents.

Do not include any other text."""

    result = None
    for max_dim in MAX_DIMENSION_ATTEMPTS:
        image_bytes = _prepare_image(image_path, max_dim)
        image_b64 = base64.b64encode(image_bytes).decode()
        try:
            result = _call_ollama_vision(
                config.ollama_url, config.ollama_model, prompt, [image_b64], 30.0
            )
            break
        except VisionExtractionError as e:
            if "500" in str(e):
                logger.warning(
                    f"Type detection 500 at max_dim={max_dim}, trying smaller"
                )
                continue
            logger.warning(f"Failed to detect document type: {e}")
            return "other"
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            logger.warning(f"Failed to detect document type: {e}")
            return "other"

    if result is None:
        logger.warning("All dimension attempts failed for type detection")
        return "other"

    response_text = result.get("response", "").strip().lower()

    valid_types = {
        "receipt",
        "invoice",
        "statement",
        "warranty",
        "contract",
        "payment_confirmation",
        "other",
    }
    for doc_type in valid_types:
        if doc_type in response_text:
            return doc_type

    return "other"
