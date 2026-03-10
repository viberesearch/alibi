#!/usr/bin/env python3
"""Compare vision models (qwen2.5vl vs glm-ocr) for document extraction.

Sends the same images to both models and compares:
- Raw OCR quality (text recognition)
- Structured extraction (JSON parsing from alibi prompts)
- Speed (time per request)
"""

import base64
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from alibi.extraction.prompts import get_prompt_for_type
from alibi.extraction.vision import _prepare_image, extract_json_from_response

OLLAMA_URL = "http://localhost:11434"
MODELS = ["qwen3-vl:30b", "glm-ocr"]

DOCS_DIR = Path(os.environ.get("ALIBI_BENCHMARK_INBOX", "./tests/fixtures/inbox"))

# Test images — representative sample
TEST_IMAGES = [
    "receipt.jpeg",  # Small receipt, 40KB
    "IMG_0428 Large.jpeg",  # Larger image, 152KB
    "IMG_0436 Large.jpeg",  # Complex receipt, 21 items, 121KB
]


def prepare_image_b64(path: Path, max_dim: int = 1344) -> str:
    """Prepare image as base64 for Ollama API."""
    img_bytes = _prepare_image(path, max_dimension=max_dim)
    return base64.b64encode(img_bytes).decode("utf-8")


def call_model(
    model: str,
    prompt: str,
    image_b64: str,
    timeout: float = 300.0,
) -> tuple[str, float]:
    """Call Ollama model and return (response_text, elapsed_seconds)."""
    start = time.monotonic()
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
                "options": {"temperature": 0.1},
            },
        )
        resp.raise_for_status()
    elapsed = time.monotonic() - start
    return resp.json().get("response", ""), elapsed


def test_ocr_mode(model: str, image_b64: str) -> tuple[str, float]:
    """Test pure OCR / text recognition mode."""
    prompt = "Extract all text from this document image. Return the raw text exactly as written."
    return call_model(model, prompt, image_b64)


def test_structured_extraction(model: str, image_b64: str) -> tuple[str, float]:
    """Test structured receipt extraction with alibi prompt."""
    prompt = get_prompt_for_type("receipt", version=2)
    return call_model(model, prompt, image_b64)


def count_json_fields(response: str) -> dict[str, Any]:
    """Try to parse JSON from response and count fields."""
    try:
        data = extract_json_from_response(response)
        if data is None:
            return {"parsed": False, "fields": 0, "items": 0}
        items = data.get("line_items") or []
        return {
            "parsed": True,
            "fields": len([k for k, v in data.items() if v is not None]),
            "items": len(items),
            "vendor": data.get("vendor", ""),
            "total": data.get("total", ""),
            "date": data.get("date", ""),
        }
    except Exception as e:
        return {"parsed": False, "error": str(e)[:80], "fields": 0, "items": 0}


def print_separator() -> None:
    print("=" * 80)


def main() -> None:
    print_separator()
    print("MODEL COMPARISON: qwen2.5vl:32b vs glm-ocr")
    print_separator()

    # Phase 1: Check model availability
    print("\n[Phase 0] Checking model availability...")
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{OLLAMA_URL}/api/tags")
        models = [m["name"] for m in resp.json().get("models", [])]
        for m in MODELS:
            tag = m if ":" in m else f"{m}:latest"
            found = any(tag in name or m in name for name in models)
            print(f"  {m}: {'AVAILABLE' if found else 'NOT FOUND'}")
            if not found:
                print(f"    Available models: {models}")
                return

    # Phase 1: Raw OCR comparison
    print(f"\n[Phase 1] Raw OCR text extraction")
    print_separator()

    for img_name in TEST_IMAGES:
        img_path = DOCS_DIR / img_name
        if not img_path.exists():
            print(f"  SKIP: {img_name} not found")
            continue

        print(f"\n--- {img_name} ---")
        image_b64 = prepare_image_b64(img_path)
        print(f"  Image prepared ({len(image_b64) // 1024}KB base64)")

        for model in MODELS:
            try:
                response, elapsed = test_ocr_mode(model, image_b64)
                lines = response.strip().split("\n")
                char_count = len(response)
                print(
                    f"\n  [{model}] {elapsed:.1f}s | {char_count} chars | {len(lines)} lines"
                )
                # Show first 5 lines
                for line in lines[:5]:
                    print(f"    {line[:100]}")
                if len(lines) > 5:
                    print(f"    ... ({len(lines) - 5} more lines)")
            except Exception as e:
                print(f"\n  [{model}] ERROR: {e}")

    # Phase 2: Structured extraction comparison
    print(f"\n\n[Phase 2] Structured receipt extraction (alibi prompt)")
    print_separator()

    for img_name in TEST_IMAGES:
        img_path = DOCS_DIR / img_name
        if not img_path.exists():
            continue

        print(f"\n--- {img_name} ---")
        image_b64 = prepare_image_b64(img_path)

        for model in MODELS:
            try:
                response, elapsed = test_structured_extraction(model, image_b64)
                stats = count_json_fields(response)
                parsed_status = "OK" if stats["parsed"] else "FAIL"
                print(
                    f"  [{model}] {elapsed:.1f}s | JSON: {parsed_status} | "
                    f"fields: {stats['fields']} | items: {stats['items']}"
                )
                if stats.get("vendor"):
                    print(
                        f"    vendor={stats['vendor']}, total={stats['total']}, date={stats['date']}"
                    )
                if not stats["parsed"]:
                    # Show raw response snippet
                    print(f"    Raw: {response[:200]}")
            except Exception as e:
                print(f"  [{model}] ERROR: {e}")

    # Phase 3: glm-ocr with its native prompt format
    print(f"\n\n[Phase 3] glm-ocr native prompt ('Text Recognition')")
    print_separator()

    for img_name in TEST_IMAGES[:1]:  # Just first image
        img_path = DOCS_DIR / img_name
        if not img_path.exists():
            continue

        print(f"\n--- {img_name} ---")
        image_b64 = prepare_image_b64(img_path)

        try:
            response, elapsed = call_model("glm-ocr", "Text Recognition", image_b64)
            lines = response.strip().split("\n")
            print(
                f"  [glm-ocr native] {elapsed:.1f}s | {len(response)} chars | {len(lines)} lines"
            )
            for line in lines[:15]:
                print(f"    {line[:120]}")
            if len(lines) > 15:
                print(f"    ... ({len(lines) - 15} more lines)")
        except Exception as e:
            print(f"  [glm-ocr native] ERROR: {e}")

    # Phase 4: Speed comparison (lightweight prompt)
    print(f"\n\n[Phase 4] Speed comparison — document type detection")
    print_separator()

    img_path = DOCS_DIR / TEST_IMAGES[0]
    image_b64 = prepare_image_b64(img_path)
    detect_prompt = (
        "What type of document is this? Reply with ONE word: "
        "receipt, invoice, statement, contract, warranty, or other."
    )

    for model in MODELS:
        times = []
        for i in range(3):
            try:
                response, elapsed = call_model(
                    model, detect_prompt, image_b64, timeout=60
                )
                times.append(elapsed)
                if i == 0:
                    print(f"  [{model}] response: {response.strip()[:50]}")
            except Exception as e:
                print(f"  [{model}] run {i+1} ERROR: {e}")

        if times:
            avg = sum(times) / len(times)
            print(f"  [{model}] avg={avg:.1f}s | times={[f'{t:.1f}' for t in times]}")

    print_separator()
    print("DONE")


if __name__ == "__main__":
    main()
