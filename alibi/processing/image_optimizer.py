"""Image optimization module for document ingest (Phase 6).

Optimizes source images on disk before pipeline processing:
- Strips EXIF metadata (privacy: GPS, camera serial, etc.)
- Resizes images exceeding max_dim (phone photos are often 4000x3000+)
- Compresses to JPEG at configurable quality
- Skips already-optimized images (checks for marker in EXIF comment)

Original files are replaced in-place. The pipeline's hash computation
runs AFTER optimization, so the hash reflects the optimized image.
"""

import io
import logging
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# Marker written to JPEG comment to skip re-optimization
_OPTIMIZED_MARKER = "alibi-optimized"

# Default settings (overridable via config)
DEFAULT_MAX_DIM = 2048
DEFAULT_QUALITY = 85

# Formats that benefit from optimization
_OPTIMIZABLE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Minimum file size worth optimizing (skip tiny images)
_MIN_FILE_SIZE = 50_000  # 50 KB

# Minimum pixel dimension for OCR readability.  When resizing to max_dim
# would shrink the shorter side below this, the scale factor is raised so
# the shorter side stays at _MIN_OCR_DIM (or the image is left untouched
# if that would require upscaling).
_MIN_OCR_DIM = 400


def is_already_optimized(image_path: Path) -> bool:
    """Check if an image has already been optimized by alibi."""
    try:
        with Image.open(image_path) as img:
            comment = img.info.get("comment", b"")
            if isinstance(comment, bytes):
                comment = comment.decode("utf-8", errors="ignore")
            return _OPTIMIZED_MARKER in comment
    except Exception:
        return False


def optimize_image(
    image_path: Path,
    max_dim: int = DEFAULT_MAX_DIM,
    quality: int = DEFAULT_QUALITY,
) -> dict[str, Any]:
    """Optimize an image file in place.

    Strips EXIF, resizes if needed, compresses to JPEG. Writes the
    optimized version back to the same path (preserving the filename
    but potentially changing the extension to .jpg).

    Args:
        image_path: Path to the image file.
        max_dim: Maximum dimension (longest side) in pixels.
        quality: JPEG compression quality (1-100).

    Returns:
        Dict with optimization stats:
        - optimized: bool — whether any changes were made
        - original_size: int — file size before (bytes)
        - new_size: int — file size after (bytes)
        - original_dimensions: tuple — (width, height) before
        - new_dimensions: tuple — (width, height) after
        - stripped_exif: bool — whether EXIF was stripped
    """
    if not image_path.exists():
        return {"optimized": False, "reason": "file_not_found"}

    suffix = image_path.suffix.lower()
    if suffix not in _OPTIMIZABLE_EXTENSIONS:
        return {"optimized": False, "reason": "unsupported_format"}

    original_size = image_path.stat().st_size
    if original_size < _MIN_FILE_SIZE:
        return {"optimized": False, "reason": "too_small"}

    if is_already_optimized(image_path):
        return {"optimized": False, "reason": "already_optimized"}

    try:
        with Image.open(image_path) as img:
            original_dims = img.size  # (width, height)

            # Check if image has EXIF data
            has_exif = bool(img.info.get("exif"))

            # Handle rotation from EXIF orientation tag
            from PIL import ImageOps

            result: Image.Image = ImageOps.exif_transpose(img)

            # Resize if needed
            w, h = result.size
            resized = False
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                # Protect minimum dimension for OCR readability
                if min(w, h) * scale < _MIN_OCR_DIM:
                    scale = max(scale, _MIN_OCR_DIM / min(w, h))
                # Never upscale
                scale = min(scale, 1.0)
                if scale < 1.0:
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    result = result.resize((new_w, new_h), Image.Resampling.LANCZOS)
                    resized = True

            # Convert to RGB (strip alpha, handle palette mode)
            if result.mode in ("RGBA", "P", "LA"):
                result = result.convert("RGB")
            elif result.mode != "RGB":
                result = result.convert("RGB")

            new_dims = result.size

            # Save to buffer first (check if it's actually smaller)
            buf = io.BytesIO()
            result.save(
                buf,
                format="JPEG",
                quality=quality,
                optimize=True,
                comment=_OPTIMIZED_MARKER,
            )
            new_size = buf.tell()

            # Only write back if we actually reduced size or stripped EXIF
            needs_write = resized or has_exif or new_size < original_size
            if not needs_write:
                return {
                    "optimized": False,
                    "reason": "no_improvement",
                    "original_size": original_size,
                    "original_dimensions": original_dims,
                }

            # Write optimized image
            # If the original wasn't JPEG, write to .jpg path
            if suffix not in (".jpg", ".jpeg"):
                new_path = image_path.with_suffix(".jpg")
                new_path.write_bytes(buf.getvalue())
                # Remove original non-JPEG file
                image_path.unlink()
                logger.info(
                    f"Converted {image_path.name} → {new_path.name} "
                    f"({original_size:,} → {new_size:,} bytes)"
                )
            else:
                image_path.write_bytes(buf.getvalue())
                new_path = image_path
                logger.info(
                    f"Optimized {image_path.name}: "
                    f"{original_size:,} → {new_size:,} bytes "
                    f"({100 - new_size * 100 // original_size}% reduction)"
                )

            return {
                "optimized": True,
                "original_size": original_size,
                "new_size": new_size,
                "original_dimensions": original_dims,
                "new_dimensions": new_dims,
                "stripped_exif": has_exif,
                "resized": resized,
                "new_path": new_path,
            }

    except Exception as e:
        logger.warning(f"Image optimization failed for {image_path}: {e}")
        return {"optimized": False, "reason": f"error: {e}"}
