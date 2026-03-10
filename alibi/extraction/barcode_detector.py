"""Barcode detection from images using pyzbar.

Detects EAN-8, EAN-13, UPC-A, and QR codes from image bytes.
Falls back gracefully if pyzbar or libzbar is not installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    from pyzbar.pyzbar import ZBarSymbol  # type: ignore[import-untyped]
    from pyzbar.pyzbar import decode as _pyzbar_decode  # type: ignore[import-untyped]

    _HAS_PYZBAR = True
except ImportError:
    _HAS_PYZBAR = False


@dataclass
class BarcodeResult:
    """A detected barcode from an image."""

    data: str  # The barcode value (digits)
    type: str  # Barcode type: EAN13, EAN8, UPCA, QRCODE, etc.
    valid_ean: bool  # Whether it passes GS1 check-digit validation


def _is_valid_ean(digits: str) -> bool:
    """Validate EAN-8 or EAN-13 check digit (GS1 mod-10 algorithm)."""
    if len(digits) not in (8, 13):
        return False
    try:
        total = 0
        for i, ch in enumerate(reversed(digits)):
            weight = 1 if i % 2 == 0 else 3
            total += int(ch) * weight
        return total % 10 == 0
    except (ValueError, IndexError):
        return False


def detect_barcodes(image_data: bytes) -> list[BarcodeResult]:
    """Detect barcodes from image bytes.

    Args:
        image_data: Raw image bytes (JPEG, PNG, etc.)

    Returns:
        List of BarcodeResult. Empty list if no barcodes found
        or if pyzbar is not available.
    """
    if not _HAS_PYZBAR:
        logger.warning("pyzbar not installed — barcode detection unavailable")
        return []

    try:
        import io

        from PIL import Image

        raw = Image.open(io.BytesIO(image_data))
        # Convert to grayscale for better detection
        img = raw.convert("L")

        decoded = _pyzbar_decode(
            img,
            symbols=[
                ZBarSymbol.EAN13,
                ZBarSymbol.EAN8,
                ZBarSymbol.UPCA,
                ZBarSymbol.QRCODE,
                ZBarSymbol.CODE128,
            ],
        )

        results: list[BarcodeResult] = []
        seen: set[str] = set()  # Deduplicate

        for obj in decoded:
            data = obj.data.decode("utf-8", errors="replace").strip()
            if not data or data in seen:
                continue
            seen.add(data)

            barcode_type = obj.type
            is_ean = _is_valid_ean(data) if data.isdigit() else False

            results.append(
                BarcodeResult(
                    data=data,
                    type=barcode_type,
                    valid_ean=is_ean,
                )
            )

        if results:
            logger.info(
                "Detected %d barcode(s): %s",
                len(results),
                ", ".join(f"{r.data} ({r.type})" for r in results),
            )

        return results

    except Exception:
        logger.exception("Barcode detection failed")
        return []


def has_barcode_support() -> bool:
    """Check if barcode detection library is available."""
    return _HAS_PYZBAR
