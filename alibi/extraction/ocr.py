"""Stage 1: OCR extraction using a lightweight vision model (glm-ocr).

Converts document images to raw text via a fast OCR model. No structured
extraction happens here — just faithful text transcription.

Four-tier retry strategy for robust text extraction:
1. Normal OCR with primary model
2. Enhanced preprocessing (contrast + sharpen)
2.5. Rotation detection (180°, 90° CW, 90° CCW) for rotated scans
3. Fallback model (e.g., minicpm-v) for non-Latin scripts

For extreme aspect ratio images (>3:1), slices into overlapping bands,
OCRs each band, and concatenates the text.
"""

import base64
import io
import logging
from pathlib import Path
from typing import Any, cast

import httpx

from alibi.config import get_config
from alibi.extraction.prompts import OCR_PROMPT
from alibi.extraction.vision import (
    EXTREME_ASPECT_RATIO,
    VisionExtractionError,
    _create_image_bands,
    _needs_slicing,
)
from alibi.utils.retry import with_retry

logger = logging.getLogger(__name__)

# Retry on transient Ollama errors
_RETRY_EXCEPTIONS = (httpx.TimeoutException, httpx.ConnectError)

# Minimum chars of OCR output before we consider it a success.
MIN_OCR_CHARS = 30

# Countries where the primary script is non-Latin.
# When folder context indicates these, use fallback model directly.
NON_LATIN_COUNTRIES: set[str] = {
    "GR",  # Greece (Greek)
    "RU",  # Russia (Cyrillic)
    "UA",  # Ukraine (Cyrillic)
    "BG",  # Bulgaria (Cyrillic)
    "RS",  # Serbia (Cyrillic)
    "MK",  # North Macedonia (Cyrillic)
    "BY",  # Belarus (Cyrillic)
    "GE",  # Georgia (Georgian)
    "AM",  # Armenia (Armenian)
    "CN",  # China (CJK)
    "JP",  # Japan (CJK)
    "KR",  # South Korea (Hangul)
    "TH",  # Thailand (Thai)
    "IL",  # Israel (Hebrew)
    "SA",  # Saudi Arabia (Arabic)
    "AE",  # UAE (Arabic)
    "EG",  # Egypt (Arabic)
}

# Default max image dimension for OCR. 1008 is the sweet spot:
# 2.7x faster than 1344 with equal or better parser confidence.
OCR_MAX_DIM = 1008

# Rotation candidates for auto-rotation detection (PIL CCW degrees).
# 180° first (upside-down most common), then 90° CW, then 90° CCW.
_ROTATION_CANDIDATES = [180, 270, 90]

# Max OCR output length (chars) before truncation. Prevents hallucination bloat.
_MAX_OCR_LEN = 10_000

# Minimum repetitions to consider a phrase as hallucinated.
_HALLUCINATION_REPEAT_THRESHOLD = 5


def _prepare_image_for_ocr(image_path: Path, max_dim: int = OCR_MAX_DIM) -> bytes:
    """Resize image for OCR. No dimension jitter needed (glm-ocr has no GGML bug).

    Returns JPEG bytes. Falls back to raw file bytes if PIL fails.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 200_000_000

    try:
        with Image.open(image_path) as raw_img:
            result: Image.Image = raw_img
            w, h = result.width, result.height
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                w = int(w * scale)
                h = int(h * scale)

            result = result.resize((w, h), Image.LANCZOS)  # type: ignore[attr-defined]

            if result.mode in ("RGBA", "P"):
                result = result.convert("RGB")

            buf = io.BytesIO()
            result.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
    except Exception as e:
        logger.debug(f"PIL could not process {image_path.name}: {e}, using raw bytes")
        return image_path.read_bytes()


def _prepare_image_enhanced(image_path: Path, max_dim: int = OCR_MAX_DIM) -> bytes:
    """Prepare image with enhanced preprocessing for OCR retry.

    Applies contrast boost and sharpening to improve OCR on faded/blurry docs.
    """
    from PIL import Image, ImageEnhance, ImageFilter

    Image.MAX_IMAGE_PIXELS = 200_000_000

    try:
        with Image.open(image_path) as raw_img:
            result: Image.Image = raw_img
            w, h = result.width, result.height
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                w = int(w * scale)
                h = int(h * scale)

            result = result.resize((w, h), Image.LANCZOS)  # type: ignore[attr-defined]

            if result.mode in ("RGBA", "P"):
                result = result.convert("RGB")

            # Enhance contrast (+20%) and sharpen
            result = ImageEnhance.Contrast(result).enhance(1.2)
            result = result.filter(ImageFilter.SHARPEN)

            buf = io.BytesIO()
            result.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
    except Exception as e:
        logger.debug(f"Enhanced prep failed for {image_path.name}: {e}")
        return _prepare_image_for_ocr(image_path, max_dim)


def _prepare_rotated_image(
    image_path: Path, degrees: int, max_dim: int = OCR_MAX_DIM
) -> bytes:
    """Prepare a rotated copy of an image for OCR.

    Uses PIL transpose for exact 90-degree rotations (faster than rotate).
    Image is resized and converted to JPEG bytes in memory — the original
    file on disk is never modified.

    Args:
        image_path: Path to the source image.
        degrees: Rotation in degrees (90, 180, 270 — PIL CCW convention).
        max_dim: Maximum dimension after rotation.

    Returns:
        JPEG bytes of the rotated, resized image.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = 200_000_000

    _TRANSPOSE_MAP = {
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }

    with Image.open(image_path) as raw_img:
        result: Image.Image = raw_img
        transpose_method = _TRANSPOSE_MAP.get(degrees)
        if transpose_method is not None:
            result = result.transpose(transpose_method)
        else:
            result = result.rotate(degrees, expand=True)

        w, h = result.width, result.height
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            w = int(w * scale)
            h = int(h * scale)

        result = result.resize((w, h), Image.LANCZOS)  # type: ignore[attr-defined]

        if result.mode in ("RGBA", "P"):
            result = result.convert("RGB")

        buf = io.BytesIO()
        result.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def _try_rotations(
    image_path: Path,
    model: str,
    ollama_url: str,
    timeout: float,
    min_chars: int = MIN_OCR_CHARS,
) -> tuple[str | None, int | None]:
    """Try OCR with rotated variants when normal OCR produces insufficient text.

    Attempts 180° (upside-down), 90° CW, and 90° CCW rotations. Returns
    immediately when a rotation produces >= min_chars of text.

    Args:
        image_path: Path to the image file.
        model: OCR model name.
        ollama_url: Ollama API URL.
        timeout: Per-request timeout in seconds.
        min_chars: Minimum chars to accept a rotation result.

    Returns:
        (text, degrees) if a rotation improved OCR, (None, None) otherwise.
        Degrees is the correction applied in PIL CCW convention.
    """
    best_text = ""
    best_rotation: int | None = None

    for degrees in _ROTATION_CANDIDATES:
        try:
            rotated_bytes = _prepare_rotated_image(image_path, degrees)
            text = _ocr_band_bytes(rotated_bytes, model, ollama_url, timeout)

            if len(text) >= min_chars:
                return text, degrees

            if len(text) > len(best_text):
                best_text = text
                best_rotation = degrees

        except (VisionExtractionError, Exception) as e:
            logger.debug(f"Rotation {degrees}° OCR failed: {e}")
            continue

    if best_text and best_rotation is not None:
        return best_text, best_rotation

    return None, None


@with_retry(max_attempts=3, base_delay=2.0, exceptions=_RETRY_EXCEPTIONS)
def _call_ollama_ocr(
    ollama_url: str,
    model: str,
    prompt: str,
    image_b64: str,
    timeout: float,
) -> dict[str, Any]:
    """Call Ollama API with a single image for OCR."""
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
            )
            response.raise_for_status()
        return cast(dict[str, Any], response.json())
    except httpx.HTTPStatusError as e:
        raise VisionExtractionError(f"OCR HTTP error: {e.response.status_code}") from e
    except httpx.RequestError as e:
        if isinstance(e, (httpx.TimeoutException, httpx.ConnectError)):
            raise  # Re-raise for retry
        raise VisionExtractionError(f"OCR request failed: {e}") from e


def ocr_image(
    image_path: Path,
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: float = 60.0,
) -> str:
    """Run OCR on a single image and return raw text.

    For extreme aspect ratio images, slices into bands and concatenates.

    Args:
        image_path: Path to the image file.
        model: OCR model name (defaults to config.ollama_ocr_model).
        ollama_url: Ollama API URL (defaults to config).
        timeout: Per-request timeout in seconds.

    Returns:
        Raw OCR text.

    Raises:
        VisionExtractionError: If OCR fails.
    """
    if not image_path.exists():
        raise VisionExtractionError(f"Image file not found: {image_path}")

    config = get_config()
    model = model or config.ollama_ocr_model
    ollama_url = ollama_url or config.ollama_url

    # Check for extreme aspect ratio — slice if needed
    try:
        from PIL import Image as _PILImage

        with _PILImage.open(image_path) as _img:
            if _needs_slicing(_img.width, _img.height):
                logger.info(
                    f"Extreme aspect ratio ({_img.width}x{_img.height}) "
                    f"for {image_path.name}, using sliced OCR"
                )
                return _ocr_sliced(image_path, model, ollama_url, timeout)
    except VisionExtractionError:
        raise
    except Exception:
        pass  # PIL failed, try direct OCR

    return _ocr_single(image_path, model, ollama_url, timeout)


def _ocr_single(
    image_path: Path,
    model: str,
    ollama_url: str,
    timeout: float,
    enhanced: bool = False,
) -> str:
    """OCR a single image (no slicing)."""
    if enhanced:
        image_bytes = _prepare_image_enhanced(image_path)
    else:
        image_bytes = _prepare_image_for_ocr(image_path)

    image_b64 = base64.b64encode(image_bytes).decode()

    try:
        result = _call_ollama_ocr(ollama_url, model, OCR_PROMPT, image_b64, timeout)
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        raise VisionExtractionError(
            f"OCR failed after retries for {image_path.name}: {e}"
        ) from e

    if "error" in result:
        raise VisionExtractionError(f"OCR error: {result['error']}")

    text: str = result.get("response", "").strip()
    logger.debug(f"OCR {image_path.name}: {len(text)} chars")
    return text


def _ocr_band_bytes(
    band_bytes: bytes,
    model: str,
    ollama_url: str,
    timeout: float,
) -> str:
    """OCR a single band (already JPEG bytes)."""
    image_b64 = base64.b64encode(band_bytes).decode()

    try:
        result = _call_ollama_ocr(ollama_url, model, OCR_PROMPT, image_b64, timeout)
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        raise VisionExtractionError(f"OCR band failed after retries: {e}") from e

    if "error" in result:
        raise VisionExtractionError(f"OCR band error: {result['error']}")

    text: str = result.get("response", "").strip()
    return text


def _ocr_sliced(
    image_path: Path,
    model: str,
    ollama_url: str,
    timeout: float,
) -> str:
    """OCR an extreme aspect ratio image by slicing into bands.

    Creates overlapping bands, OCRs each, concatenates text with separators.
    """
    bands = _create_image_bands(image_path)
    logger.info(f"Sliced {image_path.name} into {len(bands)} bands for OCR")

    texts: list[str] = []
    for i, band_bytes in enumerate(bands):
        try:
            text = _ocr_band_bytes(band_bytes, model, ollama_url, timeout)
            if text:
                texts.append(text)
            logger.debug(f"OCR band {i}: {len(text)} chars")
        except VisionExtractionError as e:
            logger.warning(f"OCR band {i} failed: {e}")

    if not texts:
        raise VisionExtractionError(
            f"All {len(bands)} OCR bands failed for {image_path}"
        )

    return "\n".join(texts)


def _is_non_latin(text: str, threshold: float = 0.3) -> bool:
    """Check if text contains a significant proportion of non-Latin characters.

    Args:
        text: OCR output text.
        threshold: Minimum ratio of non-Latin letters to trigger True.

    Returns:
        True if non-Latin script characters exceed threshold.
    """
    if not text:
        return False

    from alibi.normalizers.language import _get_unicode_script

    letter_count = 0
    non_latin_count = 0

    for char in text:
        if char.isspace() or char.isdigit() or not char.isalpha():
            continue
        letter_count += 1
        script = _get_unicode_script(char)
        if script != "LATIN":
            non_latin_count += 1

    if letter_count == 0:
        return False

    return (non_latin_count / letter_count) >= threshold


def _clean_ocr_hallucinations(text: str) -> str:
    """Remove repeated lines/phrases that indicate OCR hallucination.

    Some vision models (notably glm-ocr) occasionally emit thousands of
    repeated words or phrases on complex documents. This function detects
    and collapses such repetitions to keep the OCR text usable.
    """
    lines = text.split("\n")
    if len(lines) <= 20:
        return text

    # Count consecutive duplicate lines
    cleaned: list[str] = []
    prev = None
    dup_count = 0
    for line in lines:
        stripped = line.strip()
        if stripped == prev and stripped:
            dup_count += 1
            if dup_count < _HALLUCINATION_REPEAT_THRESHOLD:
                cleaned.append(line)
            elif dup_count == _HALLUCINATION_REPEAT_THRESHOLD:
                logger.warning(
                    "OCR hallucination detected: line %r repeated %d+ times, truncating",
                    stripped[:50],
                    _HALLUCINATION_REPEAT_THRESHOLD,
                )
        else:
            prev = stripped
            dup_count = 0
            cleaned.append(line)

    result = "\n".join(cleaned)

    # Hard truncation safety net
    if len(result) > _MAX_OCR_LEN:
        logger.warning(
            "OCR output too long (%d chars), truncating to %d",
            len(result),
            _MAX_OCR_LEN,
        )
        result = result[:_MAX_OCR_LEN]

    return result


def ocr_image_with_retry(
    image_path: Path,
    model: str | None = None,
    ollama_url: str | None = None,
    timeout: float = 60.0,
    country: str | None = None,
) -> tuple[str, bool]:
    """Run OCR with enhanced retry, rotation detection, and model fallback.

    Four-tier retry strategy:
    1. Normal OCR with primary model (glm-ocr)
    2. Enhanced preprocessing (contrast + sharpen) with primary model
    2.5. Rotation detection — try 180°, 90° CW, 90° CCW
    3. Fallback model (e.g., minicpm-v) if configured and still insufficient

    When ``country`` is a non-Latin country code and a fallback model is
    configured, the fallback model is used directly (skipping the primary
    model that performs poorly on non-Latin scripts).

    Args:
        image_path: Path to image.
        model: OCR model (defaults to config).
        ollama_url: Ollama URL (defaults to config).
        timeout: Per-request timeout.
        country: 2-letter ISO country code from folder context.

    Returns:
        Tuple of (ocr_text, was_enhanced). was_enhanced is True if the
        enhanced preprocessing or fallback model was used.

    Raises:
        VisionExtractionError: If all attempts fail.
    """
    config = get_config()
    model = model or config.ollama_ocr_model
    ollama_url = ollama_url or config.ollama_url
    fallback_model = config.ollama_ocr_fallback_model

    # Proactive model selection: non-Latin countries skip primary OCR model
    if country and country.upper() in NON_LATIN_COUNTRIES and fallback_model:
        logger.info(
            f"Non-Latin country {country}: using {fallback_model} directly "
            f"for {image_path.name}"
        )
        model = fallback_model

    text, was_enhanced = _ocr_with_retry_inner(
        image_path, model, ollama_url, timeout, fallback_model
    )
    return _clean_ocr_hallucinations(text), was_enhanced


def _ocr_with_retry_inner(
    image_path: Path,
    model: str,
    ollama_url: str,
    timeout: float,
    fallback_model: str | None,
) -> tuple[str, bool]:
    """Inner retry logic for ocr_image_with_retry."""
    # Check for slicing first (enhanced retry doesn't apply to sliced images)
    try:
        from PIL import Image as _PILImage

        with _PILImage.open(image_path) as _img:
            if _needs_slicing(_img.width, _img.height):
                text = _ocr_sliced(image_path, model, ollama_url, timeout)
                if len(text) < MIN_OCR_CHARS and fallback_model:
                    logger.info(
                        f"Sliced OCR too short ({len(text)} chars), "
                        f"retrying with fallback model {fallback_model}"
                    )
                    fb_text = _ocr_sliced(
                        image_path, fallback_model, ollama_url, timeout
                    )
                    if len(fb_text) > len(text):
                        return fb_text, True
                return text, False
    except VisionExtractionError:
        raise
    except Exception:
        pass

    # Tier 1: Normal OCR
    text = _ocr_single(image_path, model, ollama_url, timeout, enhanced=False)
    if len(text) >= MIN_OCR_CHARS:
        return text, False

    # Tier 2: Enhanced preprocessing
    logger.info(
        f"OCR output too short ({len(text)} chars) for {image_path.name}, "
        f"retrying with enhanced preprocessing"
    )
    enhanced_text = _ocr_single(image_path, model, ollama_url, timeout, enhanced=True)
    best_text = enhanced_text if len(enhanced_text) >= len(text) else text
    was_enhanced = len(enhanced_text) >= len(text)

    if len(best_text) >= MIN_OCR_CHARS:
        return best_text, was_enhanced

    # Tier 2.5: Rotation detection — try 180°, 90° CW, 90° CCW
    logger.info(
        f"OCR still short ({len(best_text)} chars) for {image_path.name}, "
        f"trying rotation detection"
    )
    rot_text, rot_deg = _try_rotations(image_path, model, ollama_url, timeout)
    if rot_text and len(rot_text) > len(best_text):
        logger.info(
            f"Rotation correction ({rot_deg}°) improved OCR: "
            f"{len(best_text)} → {len(rot_text)} chars for {image_path.name}"
        )
        best_text = rot_text
        was_enhanced = True

        if len(best_text) >= MIN_OCR_CHARS:
            return best_text, True

    # Tier 3: Fallback model (e.g., minicpm-v for non-Latin scripts)
    if fallback_model:
        logger.info(
            f"OCR still insufficient ({len(best_text)} chars) for "
            f"{image_path.name}, trying fallback model {fallback_model}"
        )
        try:
            fb_text = _ocr_single(
                image_path, fallback_model, ollama_url, timeout, enhanced=False
            )
            if len(fb_text) > len(best_text):
                if _is_non_latin(fb_text):
                    logger.info(
                        f"Fallback model detected non-Latin script "
                        f"({len(fb_text)} chars) for {image_path.name}"
                    )
                return fb_text, True
        except VisionExtractionError as e:
            logger.warning(f"Fallback OCR model failed: {e}")

    return best_text, was_enhanced
