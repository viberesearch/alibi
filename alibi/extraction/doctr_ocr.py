"""python-doctr OCR integration for alibi.

Optional alternative to Ollama vision OCR (Stage 1).
Requires: `uv pip install -e ".[ocr]"`

python-doctr provides:
- Word-level bounding boxes with page->block->line->word hierarchy
- High accuracy (97%+ on printed documents)
- Layout analysis for spatial parsing
- 90+ language support
- No Ollama/network dependency — runs locally
- No pillow version conflict (unlike surya-ocr)

Usage:
    from alibi.extraction.doctr_ocr import doctr_ocr_image, is_doctr_available

    if is_doctr_available():
        result = doctr_ocr_image(Path("receipt.jpg"))
        print(result.text)           # Full OCR text
        print(result.lines)          # Lines with word-level bboxes
        print(result.ocr_time_s)     # Time taken
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_doctr_available: bool | None = None


def is_doctr_available() -> bool:
    """Check if python-doctr is installed and importable."""
    global _doctr_available
    if _doctr_available is None:
        try:
            import doctr  # noqa: F401

            _doctr_available = True
        except ImportError:
            _doctr_available = False
    return _doctr_available


@dataclass
class OcrWord:
    """A single recognized word with spatial info."""

    text: str
    confidence: float
    geometry: tuple[tuple[float, float], tuple[float, float]]
    # ((x_min, y_min), (x_max, y_max)) relative 0-1


@dataclass
class OcrLine:
    """A line of text composed of words, with spatial info."""

    text: str
    confidence: float  # mean of word confidences
    geometry: tuple[tuple[float, float], tuple[float, float]]
    words: list[OcrWord] = field(default_factory=list)


@dataclass
class DoctrOcrResult:
    """Result of doctr OCR on a single image/page."""

    text: str  # Full text, lines joined by newline
    lines: list[OcrLine] = field(default_factory=list)
    page_dimensions: tuple[int, int] = (0, 0)  # (height, width) pixels
    ocr_time_s: float = 0.0
    image_path: str = ""
    raw_export: dict[str, Any] = field(default_factory=dict)


# Module-level predictor cache
_predictor: Any = None


def _get_predictor() -> Any:
    """Lazy-load doctr OCR predictor (cached after first call)."""
    global _predictor
    if _predictor is not None:
        return _predictor

    from doctr.models import ocr_predictor

    logger.info("Initializing doctr OCR predictor (first call)...")
    t0 = time.monotonic()

    _predictor = ocr_predictor(
        "db_resnet50",
        "crnn_vgg16_bn",
        pretrained=True,
        assume_straight_pages=True,
        preserve_aspect_ratio=True,
        symmetric_pad=True,
        resolve_lines=True,
        resolve_blocks=True,
    )

    elapsed = time.monotonic() - t0
    logger.info(f"doctr predictor loaded in {elapsed:.1f}s")

    return _predictor


def _build_result(
    doc_result: Any,
    image_path: str,
    elapsed: float,
    page_dims: tuple[int, int],
) -> DoctrOcrResult:
    """Convert doctr Document to DoctrOcrResult."""
    export = doc_result.export()
    page_data = export.get("pages", [{}])[0] if export.get("pages") else {}

    lines: list[OcrLine] = []
    text_parts: list[str] = []

    for block in page_data.get("blocks", []):
        for line_data in block.get("lines", []):
            words: list[OcrWord] = []
            word_texts: list[str] = []
            word_confs: list[float] = []

            for word_data in line_data.get("words", []):
                geo = word_data.get("geometry", ((0, 0), (1, 1)))
                word = OcrWord(
                    text=word_data.get("value", ""),
                    confidence=word_data.get("confidence", 0.0),
                    geometry=(tuple(geo[0]), tuple(geo[1])),
                )
                words.append(word)
                word_texts.append(word.text)
                word_confs.append(word.confidence)

            if not words:
                continue

            line_geo = line_data.get("geometry", ((0, 0), (1, 1)))
            line_text = " ".join(word_texts)
            line = OcrLine(
                text=line_text,
                confidence=(sum(word_confs) / len(word_confs) if word_confs else 0.0),
                geometry=(tuple(line_geo[0]), tuple(line_geo[1])),
                words=words,
            )
            lines.append(line)
            text_parts.append(line_text)

    return DoctrOcrResult(
        text="\n".join(text_parts),
        lines=lines,
        page_dimensions=page_dims,
        ocr_time_s=round(elapsed, 2),
        image_path=image_path,
        raw_export=export,
    )


def doctr_ocr_image(image_path: Path) -> DoctrOcrResult:
    """Run doctr OCR on a single image.

    Args:
        image_path: Path to the image file.

    Returns:
        DoctrOcrResult with full text, per-line/word info, and timing.

    Raises:
        ImportError: If python-doctr is not installed.
        FileNotFoundError: If image doesn't exist.
    """
    if not is_doctr_available():
        raise ImportError(
            "python-doctr is not installed. Run: uv pip install -e '.[ocr]'"
        )

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    import numpy as np
    from PIL import Image

    predictor = _get_predictor()

    img = Image.open(image_path)
    page_dims = (img.height, img.width)
    np_img = np.array(img)

    t0 = time.monotonic()
    result = predictor([np_img])
    elapsed = time.monotonic() - t0

    return _build_result(result, str(image_path), elapsed, page_dims)


def doctr_ocr_pdf(pdf_path: Path, max_pages: int = 10) -> list[DoctrOcrResult]:
    """Run doctr OCR on a PDF file (one result per page).

    Args:
        pdf_path: Path to the PDF file.
        max_pages: Maximum number of pages to process.

    Returns:
        List of DoctrOcrResult, one per page.
    """
    if not is_doctr_available():
        raise ImportError(
            "python-doctr is not installed. Run: uv pip install -e '.[ocr]'"
        )

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    import numpy as np

    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError("PDF image conversion requires: uv sync --extra pdf")

    predictor = _get_predictor()
    images = convert_from_path(str(pdf_path), first_page=1, last_page=max_pages)

    results: list[DoctrOcrResult] = []
    for i, pil_img in enumerate(images):
        page_dims = (pil_img.height, pil_img.width)
        np_img = np.array(pil_img)

        t0 = time.monotonic()
        doc_result = predictor([np_img])
        elapsed = time.monotonic() - t0

        results.append(
            _build_result(
                doc_result,
                f"{pdf_path}:page_{i + 1}",
                elapsed,
                page_dims,
            )
        )

    return results


def compare_ocr(image_path: Path, ollama_url: str | None = None) -> dict[str, Any]:
    """A/B compare doctr OCR vs Ollama vision OCR on the same image.

    Returns dict with both results and timing comparison.

    Args:
        image_path: Path to the image file.
        ollama_url: Ollama API URL for the vision OCR.

    Returns:
        Dict with 'doctr', 'ollama', and 'comparison' keys.
    """
    results: dict[str, Any] = {"image": str(image_path)}

    # doctr OCR
    if is_doctr_available():
        try:
            doctr_result = doctr_ocr_image(image_path)
            results["doctr"] = {
                "text": doctr_result.text,
                "lines": len(doctr_result.lines),
                "words": sum(len(ln.words) for ln in doctr_result.lines),
                "time_s": doctr_result.ocr_time_s,
                "avg_confidence": (
                    sum(ln.confidence for ln in doctr_result.lines)
                    / len(doctr_result.lines)
                    if doctr_result.lines
                    else 0.0
                ),
            }
        except Exception as e:
            results["doctr"] = {"error": str(e)}
    else:
        results["doctr"] = {"error": "python-doctr not installed"}

    # Ollama vision OCR
    try:
        from alibi.extraction.ocr import ocr_image_with_retry

        t0 = time.monotonic()
        ollama_text, was_enhanced = ocr_image_with_retry(
            image_path, ollama_url=ollama_url, timeout=60.0
        )
        elapsed = time.monotonic() - t0
        results["ollama"] = {
            "text": ollama_text,
            "chars": len(ollama_text),
            "time_s": round(elapsed, 2),
            "enhanced": was_enhanced,
        }
    except Exception as e:
        results["ollama"] = {"error": str(e)}

    # Comparison
    doctr_data = results.get("doctr", {})
    ollama_data = results.get("ollama", {})
    if "text" in doctr_data and "text" in ollama_data:
        results["comparison"] = {
            "doctr_chars": len(doctr_data["text"]),
            "ollama_chars": len(ollama_data["text"]),
            "doctr_time_s": doctr_data["time_s"],
            "ollama_time_s": ollama_data["time_s"],
            "faster": (
                "doctr" if doctr_data["time_s"] < ollama_data["time_s"] else "ollama"
            ),
        }

    return results
