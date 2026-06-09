# ADR: Jurisdiction inference & country-aware currency resolution

## Status

Accepted and implemented (schema v36).

## Context

Alibi was built and tuned on Cyprus (EUR) receipts. A real corpus added
documents from **Austria (EUR), Canada (CAD), and Northern Cyprus (Turkish
Lira)**. Two failures surfaced:

1. **Currency was symbol-blind.** The normaliser maps a bare `$` to USD, so a
   Canadian receipt (CAD) was stored as USD. Country always defaulted to `CY`.
2. **Multi-currency receipts.** Northern-Cyprus receipts print the same total in
   Lira + EUR + USD; the canonical amount is always the Lira, but the extractor
   picked one arbitrarily.

The capability that makes cross-vendor/-country comparison possible is the
*schema* — so the jurisdiction and canonical currency must be first-class,
correct fields, not a Cyprus-shaped default.

## Decision

A single module — `alibi/normalizers/jurisdiction.py` — is the source of truth.
It runs once post-extraction (in `ProcessingPipeline._fill_locale_gaps`) and:

- **`infer_jurisdiction(extracted)`** returns a jurisdiction code from, in
  priority order: Northern-Cyprus place tokens → Canada (provinces/cities, or a
  GST-*with-amount*/HST/PST/QST/TPS/TVQ tax line) → Austria (places or `ATU`
  VAT) → mainland Turkey → Republic of Cyprus (places, or the Greek VAT marker
  ΦΠΑ) → currency-only fallback (lira cue → TR, CAD → CA) → Greek-script
  fallback → None. Codes are ISO 3166-1 alpha-2 plus the sentinel **`CY-NORTH`**
  for the TRNC (no ISO code; distinct currency TRY and tax regime KDV).
- **`resolve_currency(extracted, jurisdiction)`** treats a known jurisdiction's
  currency as canonical (it is the currency of the printed total) — this fixes
  the `$`→CAD case and self-heals mis-tags. Unknown jurisdiction falls back to
  unambiguous text symbols/cues, not a possibly-stale currency field.
- A new **`country`** column on `facts` (migration 036) is threaded
  vendor-atom → collapse → fact → DB. Read paths use `SELECT *`, so the column
  flows through automatically.

`scripts/backfill_jurisdiction.py --all` re-derives jurisdiction for every fact
from its source extraction, so logic refinements apply retroactively.

## Precision decisions (learned from the corpus)

Country detection is a precision/recall balance; these were false-positive
sources fixed with unit tests:

- **"SMOKED TURKEY"** (poultry) matched the Turkey token → excluded "TURKEY"
  and "ADANA" (a kebab) from the token list; the country is `TÜRKİYE`.
- **"GST 2"** on restaurant slips means *guests*, not Canadian tax → bare "GST"
  dropped; a real GST line (followed by % or an amount) is matched by regex.
- **"Canada Dry" / "Canadian Club"** → bare "CANADA" ignored inside a brand
  blacklist.
- **bare "TL"** is noisy → only counts as a lira cue next to an amount.

## Deployment assumption

This is a **Cyprus-based deployment with no Greek-jurisdiction documents**. Two
fallbacks rely on that: the Greek VAT marker **ΦΠΑ** and a **dominant Greek
script** both resolve to `CY` (in a multi-country Greek/Cypriot deployment these
would need disambiguation; the *currency* — EUR — is correct either way).

## Consequences

On the 2026-06-04 corpus (268 docs) the stored facts resolved to CY 194 / CA 47
/ CY-NORTH 9 / AT 2, with currencies CAD/TRY/EUR correct, and Northern-Cyprus
multi-currency totals normalised to TRY. 42 unit tests cover the module and the
line-item pollution filter. Remaining `None`-jurisdiction facts (payment/ATM
slips, badly-OCR'd docs) keep a correct currency and are surfaced for review.
