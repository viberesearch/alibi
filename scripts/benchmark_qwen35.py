#!/usr/bin/env python3
"""Benchmark Qwen 3.5 models against current pipeline models.

Tests 3 pipeline roles:
1. OCR (vision): qwen3.5 vs glm-ocr — image → raw text
2. Structuring (text-only): qwen3.5 vs qwen3:8b — OCR text → JSON
3. Document classification (vision): qwen3.5 vs qwen3-vl:30b — image → type

Uses ground truth documents from the alibi test suite.
"""

import base64
import io
import json
import logging
import re
import sys
import time
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alibi.extraction.prompts import OCR_PROMPT, get_text_extraction_prompt
from alibi.extraction.text_parser import parse_ocr_text
from alibi.extraction.vision import extract_json_from_response

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Unbuffered output for background runs
sys.stdout.reconfigure(line_buffering=True)

import os

OLLAMA_URL = "http://localhost:11434"
INBOX = Path(os.environ.get("ALIBI_BENCHMARK_INBOX", "./tests/fixtures/inbox"))

# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------
# Current baseline models
BASELINE_OCR = "glm-ocr"
BASELINE_STRUCTURE = "qwen3:8b"
BASELINE_CLASSIFY = "qwen3-vl:30b"

# Qwen 3.5 candidates
QWEN35_MODELS = ["qwen3.5:27b"]  # 35b-a3b added when available

# ---------------------------------------------------------------------------
# Test documents (subset of ground_truth.py for speed)
# ---------------------------------------------------------------------------
OCR_TEST_DOCS = [
    {
        "file": "receipts/fresko/IMG_0430 Medium.jpeg",
        "type": "receipt",
        "expect_vendor": "FRESKO",
        "expect_total": "2.75",
        "expect_items": 2,
    },
    {
        "file": "receipts/maleve/receipt.jpeg",
        "type": "receipt",
        "expect_vendor": "Maleve",
        "expect_total": "33.03",
        "expect_items": 8,
    },
    {
        "file": "receipts/IMG_0436 Large.jpeg",
        "type": "receipt",
        "expect_vendor": "PAPAS",
        "expect_total": "85.69",
        "expect_items": 21,
    },
    {
        "file": "receipts/plus-discount/IMG_0432 Medium.jpeg",
        "type": "receipt",
        "expect_vendor": "PLUS DISCOUNT",
        "expect_total": "15.75",
        "expect_items": 5,
    },
    {
        "file": "receipts/nut-cracker-house/IMG_0428 Large.jpeg",
        "type": "receipt",
        "expect_vendor": "NUT CRACKER",
        "expect_total": "12.45",
        "expect_items": 1,
    },
]

CLASSIFY_TEST_DOCS = [
    {"file": "receipts/fresko/IMG_0430 Medium.jpeg", "expected_type": "receipt"},
    {"file": "receipts/IMG_0436 Large.jpeg", "expected_type": "receipt"},
    {
        "file": "payments/arab-butchery/IMG_0429 Medium.jpeg",
        "expected_type": "payment_confirmation",
    },
    {
        "file": "payments/fresko/IMG_0431 Medium.jpeg",
        "expected_type": "payment_confirmation",
    },
]


def prepare_image(image_path: Path, max_dim: int = 1008) -> bytes:
    """Resize image for OCR, return JPEG bytes."""
    Image.MAX_IMAGE_PIXELS = 200_000_000
    with Image.open(image_path) as img:
        w, h = img.width, img.height
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            w, h = int(w * scale), int(h * scale)
            img = img.resize((w, h), Image.LANCZOS)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def _maybe_no_think(model: str, prompt: str) -> str:
    """Prefix prompt with /no_think for Qwen models to skip reasoning."""
    if "qwen3" in model:
        return "/no_think\n" + prompt
    return prompt


def warmup_model(model: str) -> None:
    """Load a model into VRAM by sending a trivial request."""
    print(f"  Warming up {model}...")
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=300.0) as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": "/no_think\nSay OK",
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 5},
                },
            )
            resp.raise_for_status()
        print(f"  {model} loaded in {time.monotonic() - t0:.1f}s")
    except Exception as e:
        print(f"  Warmup failed for {model}: {e}")


def call_ollama_vision(
    model: str, prompt: str, image_b64: str, timeout: float = 300.0
) -> tuple[str, float, dict[str, Any]]:
    """Call Ollama vision API. Returns (response_text, elapsed_s, raw_stats)."""
    prompt = _maybe_no_think(model, prompt)
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
            )
            resp.raise_for_status()
        elapsed = time.monotonic() - t0
        data = resp.json()
        text = _strip_think_blocks(data.get("response", "").strip())
        stats = {
            "total_duration_ns": data.get("total_duration", 0),
            "eval_count": data.get("eval_count", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
        }
        return text, elapsed, stats
    except Exception as e:
        return f"ERROR: {e}", time.monotonic() - t0, {}


def call_ollama_text(
    model: str, prompt: str, timeout: float = 300.0
) -> tuple[str, float, dict[str, Any]]:
    """Call Ollama text-only API. Returns (response_text, elapsed_s, raw_stats)."""
    prompt = _maybe_no_think(model, prompt)
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
            resp.raise_for_status()
        elapsed = time.monotonic() - t0
        data = resp.json()
        text = _strip_think_blocks(data.get("response", "").strip())
        stats = {
            "total_duration_ns": data.get("total_duration", 0),
            "eval_count": data.get("eval_count", 0),
            "prompt_eval_count": data.get("prompt_eval_count", 0),
        }
        return text, elapsed, stats
    except Exception as e:
        return f"ERROR: {e}", time.monotonic() - t0, {}


def check_model_available(model: str) -> bool:
    """Check if model is available in Ollama."""
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check both exact and prefix match
            return any(model in m for m in models)
    except Exception:
        return False


def tokens_per_sec(stats: dict[str, Any], elapsed: float) -> float:
    """Calculate tokens/sec from Ollama stats."""
    eval_count = stats.get("eval_count", 0)
    if eval_count and elapsed > 0:
        return eval_count / elapsed
    return 0.0


# ===========================================================================
# BENCHMARK 1: OCR Quality
# ===========================================================================
def benchmark_ocr(models: list[str]) -> list[dict[str, Any]]:
    """Compare OCR quality: image → raw text → parser confidence."""
    print("\n" + "=" * 80)
    print("BENCHMARK 1: OCR QUALITY (image → raw text)")
    print("=" * 80)

    results = []

    for doc in OCR_TEST_DOCS:
        img_path = INBOX / doc["file"]
        if not img_path.exists():
            print(f"  SKIP: {doc['file']} not found")
            continue

        image_bytes = prepare_image(img_path)
        image_b64 = base64.b64encode(image_bytes).decode()

        with Image.open(img_path) as img:
            dims = f"{img.width}x{img.height}"

        print(f"\n{'─' * 80}")
        print(f"Document: {Path(doc['file']).name} ({dims})")
        print(f"Expected: vendor={doc['expect_vendor']}, total={doc['expect_total']}")
        print(f"{'─' * 80}")

        for model in models:
            print(f"\n  [{model}]")
            text, elapsed, stats = call_ollama_vision(model, OCR_PROMPT, image_b64)

            if text.startswith("ERROR:"):
                print(f"    {text}")
                results.append(
                    {
                        "doc": doc["file"],
                        "model": model,
                        "error": text,
                        "elapsed_s": round(elapsed, 2),
                    }
                )
                continue

            # Run through heuristic parser
            parse_result = parse_ocr_text(text, doc["type"])
            vendor = parse_result.data.get("vendor", "")
            total = parse_result.data.get("total", "")
            item_count = parse_result.line_item_count

            # Check accuracy
            vendor_match = (
                doc["expect_vendor"].lower() in vendor.lower() if vendor else False
            )
            total_match = str(total) == doc["expect_total"] if total else False
            items_ok = item_count >= doc["expect_items"]

            tps = tokens_per_sec(stats, elapsed)

            print(f"    Time: {elapsed:.2f}s ({tps:.0f} tok/s)")
            print(f"    Chars: {len(text)}, Lines: {text.count(chr(10)) + 1}")
            print(f"    Parser confidence: {parse_result.confidence:.2f}")
            print(f"    Vendor: {vendor!r} ({'OK' if vendor_match else 'MISS'})")
            print(f"    Total: {total} ({'OK' if total_match else 'MISS'})")
            print(
                f"    Items: {item_count}/{doc['expect_items']} ({'OK' if items_ok else 'MISS'})"
            )
            print(f"    Gaps: {parse_result.gaps}")

            results.append(
                {
                    "doc": doc["file"],
                    "model": model,
                    "elapsed_s": round(elapsed, 2),
                    "tokens_per_sec": round(tps, 1),
                    "chars": len(text),
                    "confidence": round(parse_result.confidence, 3),
                    "vendor": vendor,
                    "vendor_match": vendor_match,
                    "total": str(total),
                    "total_match": total_match,
                    "items": item_count,
                    "items_expected": doc["expect_items"],
                    "items_ok": items_ok,
                    "gaps": parse_result.gaps,
                    "needs_llm": parse_result.needs_llm,
                }
            )

    return results


# ===========================================================================
# BENCHMARK 2: Structuring Quality
# ===========================================================================
def benchmark_structuring(models: list[str]) -> list[dict[str, Any]]:
    """Compare structuring: OCR text → structured JSON extraction."""
    print("\n" + "=" * 80)
    print("BENCHMARK 2: STRUCTURING QUALITY (OCR text → JSON)")
    print("=" * 80)

    results = []

    for doc in OCR_TEST_DOCS:
        img_path = INBOX / doc["file"]
        if not img_path.exists():
            continue

        # First get OCR text from glm-ocr (the baseline OCR model)
        image_bytes = prepare_image(img_path)
        image_b64 = base64.b64encode(image_bytes).decode()

        print(f"\n{'─' * 80}")
        print(f"Document: {Path(doc['file']).name}")
        print(f"{'─' * 80}")

        print("  Getting OCR text from glm-ocr...")
        ocr_text, ocr_elapsed, _ = call_ollama_vision(
            BASELINE_OCR, OCR_PROMPT, image_b64
        )

        if ocr_text.startswith("ERROR:"):
            print(f"  OCR FAILED: {ocr_text}")
            continue

        print(f"  OCR: {len(ocr_text)} chars in {ocr_elapsed:.1f}s")

        # Build the structuring prompt
        prompt = get_text_extraction_prompt(
            ocr_text, doc["type"], version=2, mode="specialized"
        )

        for model in models:
            print(f"\n  [{model}] structuring...")

            text, elapsed, stats = call_ollama_text(model, prompt, timeout=300.0)

            if text.startswith("ERROR:"):
                print(f"    {text}")
                results.append(
                    {
                        "doc": doc["file"],
                        "model": model,
                        "role": "structuring",
                        "error": text,
                        "elapsed_s": round(elapsed, 2),
                    }
                )
                continue

            # Parse JSON
            try:
                extraction = extract_json_from_response(text)
            except Exception as e:
                print(f"    JSON parse failed: {e}")
                print(f"    Raw (first 200): {text[:200]}")
                results.append(
                    {
                        "doc": doc["file"],
                        "model": model,
                        "role": "structuring",
                        "error": f"JSON parse: {e}",
                        "elapsed_s": round(elapsed, 2),
                    }
                )
                continue

            if extraction is None:
                print(f"    No JSON found in response")
                print(f"    Raw (first 200): {text[:200]}")
                results.append(
                    {
                        "doc": doc["file"],
                        "model": model,
                        "role": "structuring",
                        "error": "No JSON in response",
                        "elapsed_s": round(elapsed, 2),
                    }
                )
                continue

            vendor = extraction.get("vendor", "")
            total = extraction.get("total", "")
            items = extraction.get("line_items") or []
            tps = tokens_per_sec(stats, elapsed)

            vendor_match = (
                doc["expect_vendor"].lower() in str(vendor).lower() if vendor else False
            )
            total_match = str(total) == doc["expect_total"] if total else False

            fields_present = len(
                [
                    k
                    for k, v in extraction.items()
                    if v is not None and v != "" and v != []
                ]
            )

            print(f"    Time: {elapsed:.2f}s ({tps:.0f} tok/s)")
            print(f"    Fields: {fields_present}, Items: {len(items)}")
            print(f"    Vendor: {vendor!r} ({'OK' if vendor_match else 'MISS'})")
            print(f"    Total: {total} ({'OK' if total_match else 'MISS'})")

            results.append(
                {
                    "doc": doc["file"],
                    "model": model,
                    "role": "structuring",
                    "elapsed_s": round(elapsed, 2),
                    "tokens_per_sec": round(tps, 1),
                    "fields": fields_present,
                    "items": len(items),
                    "vendor": str(vendor),
                    "vendor_match": vendor_match,
                    "total": str(total),
                    "total_match": total_match,
                }
            )

    return results


# ===========================================================================
# BENCHMARK 3: Document Classification
# ===========================================================================
CLASSIFY_PROMPT = (
    "What type of document is this image? "
    "Reply with ONLY one of these exact words: "
    "receipt, invoice, statement, payment_confirmation, contract, warranty"
)


def benchmark_classification(models: list[str]) -> list[dict[str, Any]]:
    """Compare document type classification accuracy."""
    print("\n" + "=" * 80)
    print("BENCHMARK 3: DOCUMENT CLASSIFICATION (image → type)")
    print("=" * 80)

    results = []

    for doc in CLASSIFY_TEST_DOCS:
        img_path = INBOX / doc["file"]
        if not img_path.exists():
            print(f"  SKIP: {doc['file']} not found")
            continue

        image_bytes = prepare_image(img_path, max_dim=512)  # small for speed
        image_b64 = base64.b64encode(image_bytes).decode()

        print(f"\n{'─' * 80}")
        print(f"Document: {Path(doc['file']).name} (expected: {doc['expected_type']})")
        print(f"{'─' * 80}")

        for model in models:
            print(f"\n  [{model}]")
            text, elapsed, stats = call_ollama_vision(
                model, CLASSIFY_PROMPT, image_b64, timeout=300.0
            )

            if text.startswith("ERROR:"):
                print(f"    {text}")
                results.append(
                    {
                        "doc": doc["file"],
                        "model": model,
                        "role": "classification",
                        "error": text,
                    }
                )
                continue

            # Normalize response
            response_clean = text.strip().lower().replace(" ", "_")
            # Extract just the type keyword
            for t in [
                "receipt",
                "invoice",
                "statement",
                "payment_confirmation",
                "contract",
                "warranty",
            ]:
                if t in response_clean:
                    response_clean = t
                    break

            correct = response_clean == doc["expected_type"]
            tps = tokens_per_sec(stats, elapsed)

            print(f"    Time: {elapsed:.2f}s ({tps:.0f} tok/s)")
            print(f"    Response: {text.strip()!r}")
            print(
                f"    Classified: {response_clean} ({'CORRECT' if correct else 'WRONG'})"
            )

            results.append(
                {
                    "doc": doc["file"],
                    "model": model,
                    "role": "classification",
                    "elapsed_s": round(elapsed, 2),
                    "tokens_per_sec": round(tps, 1),
                    "response": text.strip(),
                    "classified_as": response_clean,
                    "expected": doc["expected_type"],
                    "correct": correct,
                }
            )

    return results


# ===========================================================================
# Summary
# ===========================================================================
def print_summary(
    ocr_results: list[dict],
    struct_results: list[dict],
    class_results: list[dict],
) -> None:
    """Print a consolidated summary table."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # OCR summary
    print("\n--- OCR Quality ---")
    print(
        f"{'Model':<25} {'Avg Time':>8} {'Avg Conf':>8} {'Vendor%':>8} "
        f"{'Total%':>8} {'Items%':>8}"
    )
    print("─" * 65)

    ocr_models = sorted(set(r["model"] for r in ocr_results if "error" not in r))
    for model in ocr_models:
        rows = [r for r in ocr_results if r["model"] == model and "error" not in r]
        if not rows:
            continue
        avg_time = sum(r["elapsed_s"] for r in rows) / len(rows)
        avg_conf = sum(r["confidence"] for r in rows) / len(rows)
        vendor_pct = sum(1 for r in rows if r["vendor_match"]) / len(rows) * 100
        total_pct = sum(1 for r in rows if r["total_match"]) / len(rows) * 100
        items_pct = sum(1 for r in rows if r["items_ok"]) / len(rows) * 100
        print(
            f"{model:<25} {avg_time:>7.1f}s {avg_conf:>8.2f} {vendor_pct:>7.0f}% "
            f"{total_pct:>7.0f}% {items_pct:>7.0f}%"
        )

    # Structuring summary
    print("\n--- Structuring Quality ---")
    print(
        f"{'Model':<25} {'Avg Time':>8} {'Vendor%':>8} {'Total%':>8} "
        f"{'Avg Fields':>10} {'Avg Items':>10}"
    )
    print("─" * 71)

    struct_models = sorted(set(r["model"] for r in struct_results if "error" not in r))
    for model in struct_models:
        rows = [r for r in struct_results if r["model"] == model and "error" not in r]
        if not rows:
            continue
        avg_time = sum(r["elapsed_s"] for r in rows) / len(rows)
        vendor_pct = sum(1 for r in rows if r["vendor_match"]) / len(rows) * 100
        total_pct = sum(1 for r in rows if r["total_match"]) / len(rows) * 100
        avg_fields = sum(r["fields"] for r in rows) / len(rows)
        avg_items = sum(r["items"] for r in rows) / len(rows)
        print(
            f"{model:<25} {avg_time:>7.1f}s {vendor_pct:>7.0f}% {total_pct:>7.0f}% "
            f"{avg_fields:>10.1f} {avg_items:>10.1f}"
        )

    # Classification summary
    print("\n--- Classification Accuracy ---")
    print(f"{'Model':<25} {'Avg Time':>8} {'Accuracy':>8}")
    print("─" * 41)

    class_models = sorted(set(r["model"] for r in class_results if "error" not in r))
    for model in class_models:
        rows = [r for r in class_results if r["model"] == model and "error" not in r]
        if not rows:
            continue
        avg_time = sum(r["elapsed_s"] for r in rows) / len(rows)
        accuracy = sum(1 for r in rows if r["correct"]) / len(rows) * 100
        print(f"{model:<25} {avg_time:>7.1f}s {accuracy:>7.0f}%")


def main() -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Qwen 3.5 Benchmark — {timestamp}")
    print(f"Ollama: {OLLAMA_URL}")

    # Check which models are available
    available_qwen35 = [m for m in QWEN35_MODELS if check_model_available(m)]
    if not available_qwen35:
        print("\nERROR: No Qwen 3.5 models available. Run: ollama pull qwen3.5:27b")
        sys.exit(1)

    # Also check if 35b-a3b is available (add dynamically)
    if check_model_available("qwen3.5:35b-a3b"):
        available_qwen35.append("qwen3.5:35b-a3b")

    print(f"\nAvailable Qwen 3.5 models: {available_qwen35}")
    print(
        f"Baseline models: OCR={BASELINE_OCR}, Structure={BASELINE_STRUCTURE}, Classify={BASELINE_CLASSIFY}"
    )

    # Warmup: load each model into VRAM once
    all_models = sorted(
        set([BASELINE_OCR, BASELINE_STRUCTURE, BASELINE_CLASSIFY] + available_qwen35)
    )
    print(f"\nWarming up {len(all_models)} models...")
    for m in all_models:
        warmup_model(m)

    # Benchmark 1: OCR
    ocr_models = [BASELINE_OCR] + available_qwen35
    ocr_results = benchmark_ocr(ocr_models)

    # Benchmark 2: Structuring (text-only, no vision needed)
    struct_models = [BASELINE_STRUCTURE] + available_qwen35
    struct_results = benchmark_structuring(struct_models)

    # Benchmark 3: Classification (vision models only)
    class_models = [BASELINE_CLASSIFY] + available_qwen35
    class_results = benchmark_classification(class_models)

    # Summary
    print_summary(ocr_results, struct_results, class_results)

    # Save results
    all_results = {
        "timestamp": timestamp,
        "models_tested": available_qwen35,
        "baselines": {
            "ocr": BASELINE_OCR,
            "structuring": BASELINE_STRUCTURE,
            "classification": BASELINE_CLASSIFY,
        },
        "ocr": ocr_results,
        "structuring": struct_results,
        "classification": class_results,
    }

    out_path = Path(__file__).parent / "qwen35_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
