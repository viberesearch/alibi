#!/usr/bin/env python3
"""A/B test all OCR backends on real documents.

Tests: doctr, glm-ocr, deepseek-ocr, Nanonets-OCR2-3B
Measures: speed, text quality, parser compatibility.
"""

import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alibi.extraction.text_parser import parse_ocr_text

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

INBOX = Path(os.environ.get("ALIBI_BENCHMARK_INBOX", "./tests/fixtures/inbox"))
OLLAMA_URL = "http://localhost:11434"

OCR_PROMPT = (
    "Read ALL text visible in this document image. "
    "Return the complete text exactly as written, preserving layout and line breaks. "
    "Include every number, price, date, and symbol you can see."
)

# Test documents — representative mix
TEST_DOCS = [
    ("IMG_0430 Medium.jpeg", "receipt"),  # standard portrait receipt
    ("IMG_0436 Large.jpeg", "receipt"),  # extremely tall receipt (278x1280)
    ("receipt.jpeg", "receipt"),  # tall receipt (222x640)
    ("IMG_0429 Medium.jpeg", "payment_confirmation"),  # payment slip
    ("IMG_0435.jpeg", "receipt"),  # high-res landscape photo (4032x3024)
]

# Ollama OCR models to test
OLLAMA_MODELS = [
    "glm-ocr",  # current default (1.1B)
    "deepseek-ocr",  # dedicated OCR (3.3B)
    "yasserrmd/Nanonets-OCR2-3B",  # dedicated OCR, qwen2vl family (3.09B)
]


def prepare_image(image_path: Path, max_dim: int = 1344) -> bytes:
    """Resize image for OCR, return JPEG bytes."""
    Image.MAX_IMAGE_PIXELS = 200_000_000
    with Image.open(image_path) as img:
        w, h = img.width, img.height
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            w, h = int(w * scale), int(h * scale)
            img = img.resize((w, h), Image.LANCZOS)  # type: ignore[attr-defined]
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def ocr_ollama(image_path: Path, model: str) -> tuple[str, float]:
    """Run OCR via Ollama. Returns (text, time_seconds)."""
    image_bytes = prepare_image(image_path)
    image_b64 = base64.b64encode(image_bytes).decode()

    payload = {
        "model": model,
        "prompt": OCR_PROMPT,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{OLLAMA_URL}/api/generate", json=payload)
            resp.raise_for_status()
            result = resp.json()
    except Exception as e:
        return f"ERROR: {e}", time.monotonic() - t0

    elapsed = time.monotonic() - t0
    text = result.get("response", "").strip()
    return text, elapsed


def ocr_doctr(image_path: Path) -> tuple[str, float]:
    """Run OCR via python-doctr. Returns (text, time_seconds)."""
    from alibi.extraction.doctr_ocr import doctr_ocr_image

    t0 = time.monotonic()
    try:
        result = doctr_ocr_image(image_path)
        return result.text, result.ocr_time_s
    except Exception as e:
        return f"ERROR: {e}", time.monotonic() - t0


def run_parser(text: str, doc_type: str) -> dict[str, Any]:
    """Feed OCR text through the heuristic parser."""
    try:
        result = parse_ocr_text(text, doc_type)
        return {
            "confidence": round(result.confidence, 2),
            "gaps": result.gaps,
            "line_items": result.line_item_count,
            "needs_llm": result.needs_llm,
            "vendor": result.data.get("vendor") or result.data.get("issuer", ""),
            "total": result.data.get("total") or result.data.get("amount", ""),
            "date": result.data.get("document_date", ""),
        }
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    print("=" * 80)
    print("OCR A/B TEST — All backends on real documents")
    print("=" * 80)

    # Warm up doctr (first call loads model)
    print("\nWarming up doctr predictor...")
    t0 = time.monotonic()
    warmup_path = INBOX / "receipt.jpeg"
    ocr_doctr(warmup_path)
    print(f"  doctr predictor loaded in {time.monotonic() - t0:.1f}s\n")

    backends = ["doctr"] + OLLAMA_MODELS
    all_results = []

    for doc_name, doc_type in TEST_DOCS:
        doc_path = INBOX / doc_name
        if not doc_path.exists():
            print(f"SKIP: {doc_name} not found")
            continue

        with Image.open(doc_path) as img:
            dims = f"{img.width}x{img.height}"
            aspect = img.width / img.height

        print(f"\n{'─' * 80}")
        print(f"Document: {doc_name} ({dims}, aspect={aspect:.2f}, type={doc_type})")
        print(f"{'─' * 80}")

        doc_results: dict[str, Any] = {
            "document": doc_name,
            "type": doc_type,
            "dims": dims,
            "backends": {},
        }

        for backend in backends:
            print(f"\n  [{backend}]")

            if backend == "doctr":
                text, elapsed = ocr_doctr(doc_path)
            else:
                text, elapsed = ocr_ollama(doc_path, backend)

            is_error = text.startswith("ERROR:")
            char_count = len(text) if not is_error else 0
            line_count = text.count("\n") + 1 if not is_error else 0

            print(f"    Time: {elapsed:.2f}s")
            print(f"    Chars: {char_count}")
            print(f"    Lines: {line_count}")

            if is_error:
                print(f"    {text}")
                doc_results["backends"][backend] = {
                    "time_s": round(elapsed, 2),
                    "error": text,
                }
                continue

            # Show first 200 chars of OCR text
            preview = text[:200].replace("\n", "\\n")
            print(f"    Preview: {preview}...")

            # Run through parser
            parser_result = run_parser(text, doc_type)
            print(
                f"    Parser: conf={parser_result.get('confidence', '?')}, "
                f"items={parser_result.get('line_items', '?')}, "
                f"vendor={parser_result.get('vendor', '?')!r}, "
                f"total={parser_result.get('total', '?')}"
            )

            doc_results["backends"][backend] = {
                "time_s": round(elapsed, 2),
                "chars": char_count,
                "lines": line_count,
                "parser": parser_result,
            }

        all_results.append(doc_results)

    # Summary table
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    print(
        f"\n{'Document':<30} {'Backend':<30} {'Time':>6} {'Chars':>6} {'Conf':>5} {'Items':>5} {'Vendor':<20}"
    )
    print("─" * 102)

    for doc in all_results:
        for backend, data in doc["backends"].items():
            if "error" in data:
                print(
                    f"{doc['document']:<30} {backend:<30} {data['time_s']:>5.1f}s {'ERR':>6} {'':>5} {'':>5} {'':<20}"
                )
                continue
            p = data.get("parser", {})
            vendor = str(p.get("vendor", ""))[:20]
            print(
                f"{doc['document']:<30} {backend:<30} {data['time_s']:>5.1f}s {data['chars']:>6} "
                f"{p.get('confidence', 0):>5.2f} {p.get('line_items', 0):>5} {vendor:<20}"
            )

    # Save raw results
    out_path = Path(__file__).parent / "ocr_ab_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
