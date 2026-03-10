#!/usr/bin/env python3
"""Benchmark Qwen 3.5 models — v2 (fixed: sequential, cached OCR for structuring).

NOTE: Ollama 0.17.4 broke glm-ocr (hallucination loop). This benchmark
compares only working models.

Tests:
1. OCR (vision): qwen3.5:27b vs qwen3.5:35b-a3b — image → raw text
2. Structuring (text-only): qwen3.5 models vs qwen3:8b — cached OCR text → JSON
3. Document classification (vision): qwen3.5 models vs qwen3-vl:30b
"""

import base64
import io
import json
import re
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from alibi.extraction.prompts import OCR_PROMPT, get_text_extraction_prompt
from alibi.extraction.text_parser import parse_ocr_text
from alibi.extraction.vision import extract_json_from_response

import os

OLLAMA_URL = "http://localhost:11434"
INBOX = Path(os.environ.get("ALIBI_BENCHMARK_INBOX", "./tests/fixtures/inbox"))

# ---------------------------------------------------------------------------
# Test documents with cached OCR text paths
# ---------------------------------------------------------------------------
TEST_DOCS = [
    {
        "file": "receipts/fresko/IMG_0430 Medium.jpeg",
        "yaml": "receipts/fresko/IMG_0430 Medium.alibi.yaml",
        "type": "receipt",
        "expect_vendor": "FRESKO",
        "expect_total": "2.75",
        "expect_items": 2,
    },
    {
        "file": "receipts/maleve/receipt.jpeg",
        "yaml": "receipts/maleve/receipt.alibi.yaml",
        "type": "receipt",
        "expect_vendor": "Maleve",
        "expect_total": "33.03",
        "expect_items": 8,
    },
    {
        "file": "receipts/IMG_0436 Large.jpeg",
        "yaml": "receipts/IMG_0436 Large.alibi.yaml",
        "type": "receipt",
        "expect_vendor": "PAPAS",
        "expect_total": "85.69",
        "expect_items": 21,
    },
    {
        "file": "receipts/plus-discount/IMG_0432 Medium.jpeg",
        "yaml": "receipts/plus-discount/IMG_0432 Medium.alibi.yaml",
        "type": "receipt",
        "expect_vendor": "PLUS DISCOUNT",
        "expect_total": "15.75",
        "expect_items": 5,
    },
    {
        "file": "receipts/nut-cracker-house/IMG_0428 Large.jpeg",
        "yaml": "receipts/nut-cracker-house/IMG_0428 Large.alibi.yaml",
        "type": "receipt",
        "expect_vendor": "NUT CRACKER",
        "expect_total": "12.45",
        "expect_items": 1,
    },
]

CLASSIFY_DOCS = [
    {"file": "receipts/fresko/IMG_0430 Medium.jpeg", "expected": "receipt"},
    {"file": "receipts/IMG_0436 Large.jpeg", "expected": "receipt"},
    {
        "file": "payments/arab-butchery/IMG_0429 Medium.jpeg",
        "expected": "payment_confirmation",
    },
    {
        "file": "payments/fresko/IMG_0431 Medium.jpeg",
        "expected": "payment_confirmation",
    },
]


def prepare_image(image_path: Path, max_dim: int = 1008) -> bytes:
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


def strip_think(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


def no_think(model: str, prompt: str) -> str:
    if "qwen3" in model:
        return "/no_think\n" + prompt
    return prompt


def call_vision(
    model: str, prompt: str, image_b64: str, timeout: float = 300.0
) -> tuple[str, float, dict]:
    prompt = no_think(model, prompt)
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
            )
            r.raise_for_status()
        elapsed = time.monotonic() - t0
        d = r.json()
        return (
            strip_think(d.get("response", "")),
            elapsed,
            {
                "eval_count": d.get("eval_count", 0),
                "prompt_eval_count": d.get("prompt_eval_count", 0),
            },
        )
    except Exception as e:
        return f"ERROR: {e}", time.monotonic() - t0, {}


def call_text(
    model: str, prompt: str, timeout: float = 300.0
) -> tuple[str, float, dict]:
    prompt = no_think(model, prompt)
    t0 = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1},
                },
            )
            r.raise_for_status()
        elapsed = time.monotonic() - t0
        d = r.json()
        return (
            strip_think(d.get("response", "")),
            elapsed,
            {
                "eval_count": d.get("eval_count", 0),
                "prompt_eval_count": d.get("prompt_eval_count", 0),
            },
        )
    except Exception as e:
        return f"ERROR: {e}", time.monotonic() - t0, {}


def unload_all():
    """Unload all models to free VRAM."""
    try:
        with httpx.Client(timeout=10) as c:
            r = c.get(f"{OLLAMA_URL}/api/ps")
            for m in r.json().get("models", []):
                name = m.get("name", "")
                if name:
                    c.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={"model": name, "keep_alive": 0},
                    )
    except Exception:
        pass


def tps(stats: dict, elapsed: float) -> float:
    ec = stats.get("eval_count", 0)
    return ec / elapsed if ec and elapsed > 0 else 0.0


def get_cached_ocr(yaml_path: Path) -> str:
    """Load cached OCR text from .alibi.yaml file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return data.get("_meta", {}).get("ocr_text", "")


# ===========================================================================
# BENCHMARK 1: OCR Quality (Qwen 3.5 models only — glm-ocr broken)
# ===========================================================================
def benchmark_ocr(models: list[str]) -> list[dict]:
    print("\n" + "=" * 80)
    print("BENCHMARK 1: OCR QUALITY")
    print("Models: " + ", ".join(models))
    print("(glm-ocr baseline broken on Ollama 0.17.4 — using cached text as reference)")
    print("=" * 80)

    results = []

    for doc in TEST_DOCS:
        img_path = INBOX / doc["file"]
        yaml_path = INBOX / doc["yaml"]
        if not img_path.exists():
            print(f"  SKIP: {doc['file']}")
            continue

        image_bytes = prepare_image(img_path)
        image_b64 = base64.b64encode(image_bytes).decode()

        # Load reference OCR from cache (produced by working glm-ocr pre-0.17.4)
        ref_ocr = get_cached_ocr(yaml_path) if yaml_path.exists() else ""
        ref_parse = parse_ocr_text(ref_ocr, doc["type"]) if ref_ocr else None

        with Image.open(img_path) as img:
            dims = f"{img.width}x{img.height}"

        print(f"\n{'─' * 80}")
        print(f"Doc: {Path(doc['file']).name} ({dims})")
        print(f"Expected: vendor={doc['expect_vendor']}, total={doc['expect_total']}")
        if ref_parse:
            print(
                f"Reference (cached glm-ocr): conf={ref_parse.confidence:.2f}, "
                f"vendor={ref_parse.data.get('vendor', '')!r}, "
                f"total={ref_parse.data.get('total', '')}"
            )

        for model in models:
            # Unload previous model to avoid VRAM contention
            unload_all()
            time.sleep(1)

            print(f"\n  [{model}]")
            text, elapsed, stats = call_vision(model, OCR_PROMPT, image_b64)

            if text.startswith("ERROR:"):
                print(f"    {text}")
                results.append({"doc": doc["file"], "model": model, "error": text})
                continue

            pr = parse_ocr_text(text, doc["type"])
            vendor = pr.data.get("vendor", "")
            total = pr.data.get("total", "")
            items = pr.line_item_count

            vm = doc["expect_vendor"].lower() in vendor.lower() if vendor else False
            tm = str(total) == doc["expect_total"] if total else False
            im = items >= doc["expect_items"]
            t = tps(stats, elapsed)

            print(f"    Time: {elapsed:.1f}s ({t:.0f} tok/s)")
            print(f"    Chars: {len(text)}, Lines: {text.count(chr(10)) + 1}")
            print(f"    Confidence: {pr.confidence:.2f}")
            print(f"    Vendor: {vendor!r} ({'OK' if vm else 'MISS'})")
            print(f"    Total: {total} ({'OK' if tm else 'MISS'})")
            print(
                f"    Items: {items}/{doc['expect_items']} ({'OK' if im else 'MISS'})"
            )

            results.append(
                {
                    "doc": doc["file"],
                    "model": model,
                    "elapsed_s": round(elapsed, 1),
                    "tps": round(t, 0),
                    "chars": len(text),
                    "confidence": round(pr.confidence, 3),
                    "vendor": vendor,
                    "vendor_match": vm,
                    "total": str(total),
                    "total_match": tm,
                    "items": items,
                    "items_ok": im,
                    "gaps": pr.gaps,
                }
            )

    return results


# ===========================================================================
# BENCHMARK 2: Structuring (uses cached OCR text — avoids broken glm-ocr)
# ===========================================================================
def benchmark_structuring(models: list[str]) -> list[dict]:
    print("\n" + "=" * 80)
    print("BENCHMARK 2: STRUCTURING QUALITY (cached OCR → JSON)")
    print("Models: " + ", ".join(models))
    print("=" * 80)

    results = []

    for doc in TEST_DOCS:
        yaml_path = INBOX / doc["yaml"]
        if not yaml_path.exists():
            print(f"  SKIP: {doc['yaml']}")
            continue

        ocr_text = get_cached_ocr(yaml_path)
        if not ocr_text:
            print(f"  SKIP: No cached OCR for {doc['file']}")
            continue

        prompt = get_text_extraction_prompt(
            ocr_text, doc["type"], version=2, mode="specialized"
        )

        print(f"\n{'─' * 80}")
        print(f"Doc: {Path(doc['file']).name} (cached OCR: {len(ocr_text)} chars)")

        for model in models:
            unload_all()
            time.sleep(1)

            print(f"\n  [{model}]")
            text, elapsed, stats = call_text(model, prompt, timeout=300.0)

            if text.startswith("ERROR:"):
                print(f"    {text}")
                results.append({"doc": doc["file"], "model": model, "error": text})
                continue

            try:
                extraction = extract_json_from_response(text)
            except Exception as e:
                print(f"    JSON parse failed: {e}")
                print(f"    Raw: {text[:200]}")
                results.append(
                    {"doc": doc["file"], "model": model, "error": f"JSON: {e}"}
                )
                continue

            if extraction is None:
                print(f"    No JSON found")
                print(f"    Raw: {text[:200]}")
                results.append({"doc": doc["file"], "model": model, "error": "No JSON"})
                continue

            vendor = extraction.get("vendor", "")
            total = extraction.get("total", "")
            items = extraction.get("line_items") or []
            t = tps(stats, elapsed)
            vm = (
                doc["expect_vendor"].lower() in str(vendor).lower() if vendor else False
            )
            tm = str(total) == doc["expect_total"] if total else False
            fields = len([k for k, v in extraction.items() if v not in (None, "", [])])

            print(f"    Time: {elapsed:.1f}s ({t:.0f} tok/s)")
            print(f"    Fields: {fields}, Items: {len(items)}")
            print(f"    Vendor: {vendor!r} ({'OK' if vm else 'MISS'})")
            print(f"    Total: {total} ({'OK' if tm else 'MISS'})")

            results.append(
                {
                    "doc": doc["file"],
                    "model": model,
                    "elapsed_s": round(elapsed, 1),
                    "tps": round(t, 0),
                    "fields": fields,
                    "items": len(items),
                    "vendor": str(vendor),
                    "vendor_match": vm,
                    "total": str(total),
                    "total_match": tm,
                }
            )

    return results


# ===========================================================================
# BENCHMARK 3: Classification
# ===========================================================================
CLASSIFY_PROMPT = (
    "What type of document is this image? "
    "Reply with ONLY one of these exact words: "
    "receipt, invoice, statement, payment_confirmation, contract, warranty"
)


def benchmark_classification(models: list[str]) -> list[dict]:
    print("\n" + "=" * 80)
    print("BENCHMARK 3: DOCUMENT CLASSIFICATION")
    print("Models: " + ", ".join(models))
    print("=" * 80)

    results = []

    for doc in CLASSIFY_DOCS:
        img_path = INBOX / doc["file"]
        if not img_path.exists():
            print(f"  SKIP: {doc['file']}")
            continue

        image_bytes = prepare_image(img_path, max_dim=512)
        image_b64 = base64.b64encode(image_bytes).decode()

        print(f"\n{'─' * 80}")
        print(f"Doc: {Path(doc['file']).name} (expected: {doc['expected']})")

        for model in models:
            unload_all()
            time.sleep(1)

            print(f"\n  [{model}]")
            text, elapsed, stats = call_vision(model, CLASSIFY_PROMPT, image_b64)

            if text.startswith("ERROR:"):
                print(f"    {text}")
                results.append({"doc": doc["file"], "model": model, "error": text})
                continue

            clean = text.strip().lower().replace(" ", "_")
            for t_name in [
                "payment_confirmation",
                "receipt",
                "invoice",
                "statement",
                "contract",
                "warranty",
            ]:
                if t_name in clean:
                    clean = t_name
                    break

            correct = clean == doc["expected"]
            t = tps(stats, elapsed)

            print(f"    Time: {elapsed:.1f}s ({t:.0f} tok/s)")
            print(
                f"    Response: {text.strip()!r} → {clean} ({'CORRECT' if correct else 'WRONG'})"
            )

            results.append(
                {
                    "doc": doc["file"],
                    "model": model,
                    "elapsed_s": round(elapsed, 1),
                    "tps": round(t, 0),
                    "classified": clean,
                    "expected": doc["expected"],
                    "correct": correct,
                }
            )

    return results


# ===========================================================================
# Summary
# ===========================================================================
def print_summary(ocr: list, struct: list, cls: list):
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    for label, data, metrics in [
        (
            "OCR Quality",
            ocr,
            ["elapsed_s", "confidence", "vendor_match", "total_match", "items_ok"],
        ),
        (
            "Structuring",
            struct,
            ["elapsed_s", "vendor_match", "total_match", "fields", "items"],
        ),
    ]:
        print(f"\n--- {label} ---")
        models = sorted(set(r["model"] for r in data if "error" not in r))
        if not models:
            print("  No valid results")
            continue

        if "confidence" in metrics:
            print(
                f"{'Model':<25} {'Time':>7} {'Conf':>6} {'Vend%':>6} {'Tot%':>6} {'Item%':>6}"
            )
        else:
            print(
                f"{'Model':<25} {'Time':>7} {'Vend%':>6} {'Tot%':>6} {'Fields':>7} {'Items':>6}"
            )
        print("─" * 65)

        for model in models:
            rows = [r for r in data if r["model"] == model and "error" not in r]
            if not rows:
                continue
            at = sum(r["elapsed_s"] for r in rows) / len(rows)
            vp = sum(1 for r in rows if r["vendor_match"]) / len(rows) * 100
            tp = sum(1 for r in rows if r["total_match"]) / len(rows) * 100

            if "confidence" in metrics:
                ac = sum(r["confidence"] for r in rows) / len(rows)
                ip = sum(1 for r in rows if r["items_ok"]) / len(rows) * 100
                print(
                    f"{model:<25} {at:>6.1f}s {ac:>6.2f} {vp:>5.0f}% {tp:>5.0f}% {ip:>5.0f}%"
                )
            else:
                af = sum(r["fields"] for r in rows) / len(rows)
                ai = sum(r["items"] for r in rows) / len(rows)
                print(
                    f"{model:<25} {at:>6.1f}s {vp:>5.0f}% {tp:>5.0f}% {af:>7.1f} {ai:>6.1f}"
                )

    print(f"\n--- Classification ---")
    models = sorted(set(r["model"] for r in cls if "error" not in r))
    if models:
        print(f"{'Model':<25} {'Time':>7} {'Accuracy':>8}")
        print("─" * 40)
        for model in models:
            rows = [r for r in cls if r["model"] == model and "error" not in r]
            if not rows:
                continue
            at = sum(r["elapsed_s"] for r in rows) / len(rows)
            acc = sum(1 for r in rows if r["correct"]) / len(rows) * 100
            print(f"{model:<25} {at:>6.1f}s {acc:>7.0f}%")


def main():
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Qwen 3.5 Benchmark v2 — {ts}")
    print(f"Ollama: {OLLAMA_URL}")

    # Check available models
    with httpx.Client(timeout=10) as c:
        r = c.get(f"{OLLAMA_URL}/api/tags")
        installed = [m["name"] for m in r.json().get("models", [])]

    qwen35 = []
    for m in ["qwen3.5:27b", "qwen3.5:35b-a3b"]:
        if any(m in n for n in installed):
            qwen35.append(m)

    print(f"\nQwen 3.5 available: {qwen35}")
    print(f"NOTE: glm-ocr broken on Ollama 0.17.4 — excluded from OCR benchmark")
    print(f"NOTE: Structuring uses cached OCR text from pre-0.17.4 glm-ocr runs")

    # Benchmark 1: OCR (vision models only)
    ocr_results = benchmark_ocr(qwen35)

    # Benchmark 2: Structuring (text-only)
    struct_models = ["qwen3:8b"] + qwen35
    struct_results = benchmark_structuring(struct_models)

    # Benchmark 3: Classification (vision)
    class_models = ["qwen3-vl:30b"] + qwen35
    class_results = benchmark_classification(class_models)

    # Summary
    print_summary(ocr_results, struct_results, class_results)

    # Save
    all_results = {
        "timestamp": ts,
        "ollama_version": "0.17.4",
        "note": "glm-ocr broken (hallucination) on Ollama 0.17.4",
        "models_tested": qwen35,
        "ocr": ocr_results,
        "structuring": struct_results,
        "classification": class_results,
    }
    out = Path(__file__).parent / "qwen35_benchmark_v2_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
