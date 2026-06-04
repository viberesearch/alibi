# ADR: Schema-Enforced Local Structuring (Ollama `format` + `think=false`)

## Status

Accepted and implemented.

## Context

Stage 3 of the pipeline turns raw OCR text into a structured extraction dict.
The cloud path (Gemini) already enforced a contract: it passes the Pydantic
extraction models (`ReceiptExtraction`, `InvoiceExtraction`, …) as
`response_schema`, so the model can only emit schema-conforming JSON.

The local-first default path (Ollama, `qwen3.5:9b`) did not. It sent a prose
prompt ending in "Return a JSON object with: {…}" and parsed whatever came
back — a *HOW* prompt describing the output in prose rather than a *WHAT*
prompt enforcing it. This is the exact failure mode described in "Tell The LLM
What. Stop Telling It How.": the schema, not the prose, is what pulls the model
toward correct, parseable output.

Two problems surfaced when measured on real receipts (`qwen3.5:9b`, Ollama
0.30.4):

1. **The local path was effectively non-functional.** On Ollama ≥ 0.30 the
   in-prompt `/no_think` directive is silently ignored. `qwen3.5` is a hybrid
   reasoning model; left in reasoning mode it spent its **entire** `num_predict`
   budget (4096 tokens) on chain-of-thought, hit `done_reason=length`, and
   returned an **empty `response`** field (the reasoning went to a separate
   `thinking` field). Net result: 0/10 receipts produced parseable JSON; the
   path silently failed or fell back.

2. **Free-form output, even once reasoning was fixed, hallucinated structure.**
   Without a schema the model invented optional fields — e.g. `unit_raw="kg"`,
   `unit_quantity=1.0`, `brand="MILKMAN"`, `category="dairy"` on items that
   stated none of them — and sometimes returned an empty `line_items` array.

This matters across countries and scripts. Receipts arrive in many tax regimes
(German MwSt, French TVA, Greek ΦΠΑ, Russian НДС, sales tax, GST) and many
scripts (Latin, Cyrillic, Greek, Arabic, CJK). The canonical schema fields
(`tax_rate`, `tax_type`, `unit_raw`, `name` + `name_en`, `language`) are the
normalization target that makes those documents comparable. The schema is the
capability.

## Decision

Constrain the local Ollama structuring call to the **same** Pydantic schemas
the Gemini path already uses, and disable reasoning at the API level:

- **`get_extraction_json_schema(doc_type)`** derives the JSON schema from the
  existing extraction models (`_get_extraction_model(...).model_json_schema()`).
  No new schema is introduced — the contract is shared with the cloud path.
- **`structure_ocr_text`** passes that schema as Ollama's `format` field for
  normal extraction (config flag `ollama_structured_output`, default `True`).
  Emphasis/correction retries stay free-form (their prompts ask for shapes the
  strict schema would reject).
- **`_call_ollama_text`** sets `think=false` for hybrid reasoning models
  (`qwen3`, `gemma4`, `gemma3`, `exaone`, `sarvam`). The ineffective in-prompt
  `/no_think` prefix is removed. The model-name lists are split into
  `_CHAT_ENDPOINT_MODELS` (routing) and `_THINKING_CAPABLE_MODELS` (the
  `think=false` switch).

## Consequences

Measured on 10 real receipts (Cyprus/EU, mixed Latin + Russian-origin product
names), `qwen3.5:9b`, single OCR per image, identical prompt — isolating each
variable:

| Variant | JSON-OK | avg items | avg fill | avg verify | out tokens | latency |
|---------|--------:|----------:|---------:|-----------:|-----------:|--------:|
| A — reasoning on, no schema (prior behaviour) | **0%** | — | — | — | 4096 (cap) | 119 s |
| B — `think=false`, free-form JSON | 100% | 2.5 | 0.687 | 0.784 | 876 | 24.7 s |
| C — `think=false` + schema (**shipped**) | 100% | **2.7** | **0.767** | **0.82** | **601** | **17.0 s** |

- **A→B** is the correctness rescue: a broken default path becomes 100%
  parseable.
- **B→C** is the schema-enforcement win: more line items recovered (the schema
  pulls the model to populate `line_items` where free-form returned `[]`),
  higher field fill *with less hallucination* (free-form's occasional higher
  fill was invented values), higher arithmetic-verification confidence, and
  **−31% output tokens / −31% latency**.

Non-Latin handling is preserved (JSON-schema `type: string` permits any
Unicode): Cyrillic / Greek / Arabic product names are kept in `name`, the
English rendering goes to `name_en`, `language` gets the correct ISO code, and
НДС 20% / ΦΠΑ 24% / VAT 5% all land in the canonical `tax_rate` field.

The same `think=false` fix also rescues the correction/enrichment retry paths,
which were silently broken by the same reasoning-budget bug.

## Notes

- Requires Ollama ≥ 0.30 (the `think` request field). The internal pins ≥ 0.19.x;
  this raises the effective floor for the local structuring path.
- Cloud (Gemini) path unchanged — it already enforced the contract.
- The general lesson generalizes to any local structured-output task on
  Ollama: pass the JSON schema as `format` and disable reasoning with
  `think=false` for hybrid models.
