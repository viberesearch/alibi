# alibi

Local-first financial document intelligence. Extracts structured data from photos of receipts, invoices, and financial documents using local OCR. Matches documents across sources. Learns from corrections. Privacy-first: 64% of documents processed entirely locally with zero cloud API cost.

**API-first architecture**: every feature is available through the REST API. The CLI, Telegram bot, MCP server, and web UI are all thin clients over the same service layer.

## Why alibi?

Receipt scanners extract vendor + total. Expense trackers require manual data entry. Cloud APIs charge per scan and see all your data.

Alibi does more:
- **Line-item extraction**: individual products with quantity, unit, price, brand, category, tax, barcode -- not just totals
- **Cross-document matching**: a receipt + a bank statement + a payment confirmation = one confirmed financial fact, with split payment detection
- **Cross-vendor price comparison**: `comparable_unit_price` normalizes EUR/kg, EUR/L, EUR/pcs across vendors. `comparable_name` translates product names across languages. Answer "where is milk cheapest?" across stores and countries
- **Product variants**: tracks that 3% milk differs from 1.5%, L eggs from M eggs, organic from conventional -- at the schema level, not as free text
- **Price factor analysis**: discovers *why* prices differ (brand premium, organic markup, vendor pricing) from your data
- **Vendor intelligence**: VAT-based identity resolution, POS system signature detection (16 systems), template learning per vendor
- **Privacy controls**: 4-tier data masking (public/private/trusted/secret) and 3-level anonymization export (categories-only, pseudonymized, statistical)
- **Product enrichment**: 8-tier cascade from Open Food Facts to Gemini, with provenance tracking and optional contribution back to OFF
- **Adaptive learning**: gets smarter with every correction -- templates, categories, vendor defaults, extraction quality self-diagnostics
- **Local-first**: Ollama (glm-ocr 1.1B) for OCR, heuristic parser handles 64% of documents without any LLM

The core data model -- **Atoms** (raw observations) -> **Clouds** (probabilistic clusters) -> **Facts** (confirmed events) -- is a general epistemological pattern for turning uncertain observations into confirmed knowledge.

### Document-agnostic by design

Alibi works with whatever you have. A single POS slip records the amount and vendor. A receipt adds item-level detail. A bank statement confirms the transaction from the bank's side. When multiple documents describe the same purchase, they validate each other and collapse into a single fact with richer data -- each document type is another dimension, not a duplicate. The system never requires a specific document type; one is enough, several are better.

### Human-editable YAML intermediaries

Every processed document produces a `.alibi.yaml` file alongside the original image/PDF. This YAML file is the single source of truth between the document and the database:

```yaml
document_type: receipt
vendor: Alphamega
vendor_vat: "10XXXXXXY"
date: "2026-03-01"
currency: EUR
total: 45.67
subtotal: 43.25
tax_total: 2.42
payment_method: card
line_items:
  - name: Organic Milk 1L
    quantity: 2
    unit: l                  # measurement dimension (l for liquids, kg for weight, pcs for counted)
    unit_quantity: 1.0       # package size in unit dimension (1.0L) — enables EUR/L comparison
    unit_price: 2.35
    total_price: 4.70
    tax_rate: 5.0
    brand: ""                # filled by enrichment cascade
    category: ""             # filled by enrichment cascade
    product_variant: ""      # subcategory: "3%" for milk fat, "L" for egg size
    barcode: ""              # detected from receipt or photo
```

If the extraction got something wrong, you edit the YAML (it is designed to be human-readable) and re-run `lt process` — the updated values replace the previous database records. The `unit` + `unit_quantity` pair is key to comparison: `unit` is the measurement dimension (`l` for liquids, `kg` for weight, `pcs` for counted items) and `unit_quantity` is the package size in that dimension. This enables `comparable_unit_price` (e.g., EUR/L across different milk brands and package sizes). This gives you full control over your data without needing to understand the database schema.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Ollama](https://ollama.ai/) with the `glm-ocr` model

Optional system dependencies:
- `poppler` — for PDF processing (`brew install poppler` / `apt install poppler-utils`)
- `zbar` — for barcode scanning (`brew install zbar` / `apt install libzbar0`)

### Install

```bash
git clone https://github.com/viberesearch/alibi.git
cd alibi

# Core install (CLI + API + OCR processing)
uv sync

# Full install (all optional features)
uv sync --extra all

# Pull the OCR model
ollama pull glm-ocr

# Optional: pull the local LLM for Stage 3 correction
ollama pull qwen3:8b
```

### Configure

```bash
cp .env.example .env
# Edit .env — only ALIBI_OLLAMA_URL is truly required
# See .env.example for the full list of options

cp data/vendor_aliases.example.yaml data/vendor_aliases.yaml
cp data/unit_aliases.example.yaml data/unit_aliases.yaml
```

### Run

```bash
# Initialize the database
uv run lt init

# Process a receipt photo
uv run lt process -p ~/path/to/receipt.jpg

# View extracted facts
uv run lt facts list

# Start the API server (interactive docs at http://localhost:3100/docs)
uv run lt serve
```

## Architecture

```
Client (CLI / REST API / Telegram / MCP / Web UI)
  |
  v
Service Layer (alibi/services/ — 13 facades, ~88 functions)
  |
  +---> Extraction Pipeline (3-stage hybrid)
  |       Stage 1: OCR (glm-ocr, local, 1.5-6s)
  |       Stage 2: Heuristic parser (~2ms, handles 64% of docs)
  |       Stage 3: LLM correction (local qwen3:8b OR cloud Gemini 2.5 Flash)
  |
  +---> Atom-Cloud-Fact Pipeline
  |       atoms/parser -> clouds/formation -> clouds/collapse
  |
  +---> SQLite (schema v33, 20 tables + FTS5 index)
  |
  +---> Event Bus -> Obsidian notes, webhooks, analytics export
```

**Stack**: Python 3.12, FastAPI, SQLite, Ollama, pydantic-settings, httpx, Pillow.
Optional: Google Gemini, Anthropic Claude, aiogram (Telegram), pyzbar (barcodes), lancedb (vectors).

### Stage 3: Local vs Cloud

The extraction pipeline's Stage 3 (LLM correction) supports two backends:

| | Local (Ollama) | Cloud (Gemini) |
|---|---|---|
| Model | qwen3:8b | Gemini 2.5 Flash |
| Cost | Free | ~$0.001/doc |
| Speed | 2-5s | 0.5-1s |
| RAM | ~8GB | None |
| Privacy | Full | Product names only (no PII) |
| Config | Default | `ALIBI_GEMINI_EXTRACTION_ENABLED=true` |

Most documents (64%) are handled by the heuristic parser alone, making Stage 3 a fallback for complex/unusual documents.

<details>
<summary><strong>Features</strong></summary>

### Extraction
- Human-editable YAML intermediaries (`.alibi.yaml`) -- edit and re-ingest to correct data
- 3-stage hybrid pipeline: OCR -> heuristic parser -> conditional LLM
- Per-item extraction: quantity, unit, price, brand, category, tax, barcode
- Product variant extraction: fat %, egg size (L/M/S), organic, light, strained, unsalted
- Cash flow tracking: amount tendered and change due (when payment_method=cash)
- Folder-based document routing with country/vendor/type inference from directory structure
- Multi-language OCR with fallback models for non-Latin scripts (Greek, Russian, CJK)
- POS system signature detection (16 systems) to bootstrap vendor templates
- Image optimization on ingest (EXIF stripping, resize, compression)
- Duplicate detection: file hash (MD5) + perceptual hash (aHash) + fuzzy vendor matching
- Barcode detection from receipt photos (EAN/UPC/QR via pyzbar)
- CSV and OFX import for bank statements and transaction data

### Data Model
- Atom-Cloud-Fact pipeline: observations -> probabilistic clusters -> confirmed events
- Cross-document matching (receipt + payment confirmation + bank statement = one fact)
- Split payment detection: multiple payment methods summing to one transaction
- Vendor identity resolution (VAT number, tax ID, fuzzy name matching)
- Vendor template learning with reliability tracking (6 learned fields per vendor)
- Annotations: open-ended key-value metadata on any entity (tags, notes, product attributes)
- Budget system: scenarios with parent/child categories, variance tracking, period-based alerts

### Enrichment
- 8-tier cascade: historical -> Open Food Facts -> UPCitemdb -> GS1 prefix -> fuzzy name -> local LLM -> Gemini -> Anthropic
- Cross-language product comparison via `comparable_name` (e.g., "ΓΑΛΑ ΠΛΗΡΕΣ" -> "whole milk")
- Enrichment provenance tracking with user feedback loop (confirm/reject via Web UI or Telegram)
- Cross-vendor barcode matching (shared EAN/UPC propagation)
- Nutritional tracking: joins purchased items with OFF product data for per-item nutritional estimates
- Open Food Facts contribution: optionally submit enriched products back to OFF
- Scheduled enrichment daemon: automatic background processing on staggered intervals

### Analytics & Comparison
- **Cross-vendor price comparison**: `comparable_unit_price` normalizes to EUR/kg, EUR/L, EUR/pcs across vendors and package sizes
- **Price factor analysis**: discovers price drivers from data (brand premiums, organic markups, vendor pricing strategies)
- **Product variant comparison**: compare 3% milk vs 1.5% milk, L eggs vs M eggs across vendors
- Subscription detection (k-means on recurring amounts with period classification)
- Anomaly detection (statistical outlier detection with baseline drift and severity scoring)
- Correction confusion matrix: self-diagnostics showing which extraction fields need improvement
- Location analytics (spending heatmap, branch comparison, nearby vendors)
- Period comparison (month-over-month, year-over-year, category trend analysis)
- MindsDB predictors (spending forecast, category classification)
- All data accessible via MCP for LLM-powered natural language analytics

### Privacy & Data Control
- **5-tier data masking** (T0-T4): amounts hidden/rounded/exact, vendor=category/name, dates=month/exact, line items included/excluded, provenance included/excluded
  - CLI: `lt export masked output.json --tier 2`, API: `GET /api/v1/export/masked/transactions?tier=2`
- **3-level anonymization export**: categories_only (safe for any external use), pseudonymized (consistent fake names, shifted amounts, preserves patterns), statistical (aggregates only)
  - CLI: `lt export anonymized data.json --level pseudonymized --key-file key.json`, API: `POST /api/v1/export/anonymized?level=statistical`
- **Cloud AI masking**: automatically masks sensitive entities before sending to cloud APIs, with reversible masking maps
- **Snapshot & diff**: detect out-of-band edits (e.g., direct SQL changes in TablePlus) and track them as correction events
- Backup and restore: compressed archives with manifest and checksum verification (`lt db backup/restore`)

### Adaptive Learning
- Template reliability tracking per vendor (6 fields: date format, total marker, header/footer ratio, language, layout type)
- Correction feedback loops with sibling propagation (correct one item, similar items update)
- Cloud formation weight learning from merge/split history
- OCR spell correction using identity database as dictionary
- Barcode position learning per POS/vendor
- Extraction quality self-assessment: correction confusion matrix highlights weak fields

</details>

## Interfaces

Alibi is **API-first** — the REST API is the primary interface. All other interfaces (CLI, Telegram, MCP) call the same service layer.

### REST API

Start the server:
```bash
uv run lt serve
```

- **Interactive API docs**: `http://localhost:3100/docs` (Swagger UI)
- **Web UI**: `http://localhost:3100/web` (user management and enrichment dashboard)

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/process` | Process a document |
| GET | `/api/v1/facts` | List facts (paginated) |
| GET | `/api/v1/facts/{id}` | Fact detail with items |
| POST | `/api/v1/search` | Search by vendor/items |
| GET | `/api/v1/analytics/*` | Spending, subscriptions, anomalies |
| GET | `/api/v1/identities` | Vendor identities |
| PATCH | `/api/v1/line-items/{id}` | Update item enrichment |
| POST | `/api/v1/enrichment/run/*` | Trigger enrichment (cloud/llm/gemini) |
| GET | `/api/v1/corrections` | Correction event log |
| GET | `/api/v1/nutrition/summary` | Nutritional tracking |
| GET | `/health` | Health check with DB stats |

See the interactive docs for the full 97-endpoint reference.

Authentication: single-user mode by default (no API key needed). Set `ALIBI_API_KEY` to require a key, or use the mnemonic API key system for multi-user setups (`uv run lt user api-key create`).

### Web UI

The web UI is a single-page application served by the API server at `/web`:

- **Dashboard**: system stats, user list, quick actions
- **Facts browser**: paginated fact list with drill-down to items, bundles, and clouds
- **Document upload**: drag-and-drop with type selection, real-time processing feedback
- **Analytics**: spending by month/vendor/category, subscription detection, bar charts
- **Search**: cross-entity search (documents, facts, items) with type filtering
- **Enrichment review**: confidence-based queue, confirm/reject with inline editing, coverage analytics
- **User management**: create users, manage contacts (Telegram/email), generate and revoke API keys

No separate build step required -- the UI is a self-contained HTML file. Start the API server and navigate to `http://localhost:3100/web`.

### CLI (`lt`)

```bash
uv run lt process -p <file>          # Process a document
uv run lt process -p <folder>        # Process folder as multi-page document
uv run lt facts list                 # List all facts
uv run lt facts show <id>            # Inspect a fact with items
uv run lt enrich cascade [-l 100]    # Multi-source barcode enrichment
uv run lt enrich gemini [-l 500]     # Gemini mega-batch enrichment
uv run lt analytics corrections      # Correction confusion matrix
uv run lt export masked out.json -t 2   # Tier-masked export
uv run lt export anonymized data.json   # Privacy-preserving export
uv run lt budget list                # List budget scenarios
uv run lt budget actual 2026-03      # Actual spending for a period
uv run lt vectordb search "groceries"   # Semantic search
uv run lt maintain run               # Full maintenance cycle
uv run lt db info                    # Schema version and row counts
```

The `lt` command stands for "Life Tracker" (alibi's original working name).

### MCP Server

```bash
uv run python -m alibi.mcp           # Start MCP server (25 tools)
```

Alibi is MCP-native -- AI assistants can process documents, query facts, update enrichment, and run analytics through the MCP protocol. Tools include document processing, fact/item queries, vendor identity management, line item updates (barcode, brand, category, unit), analytics summaries, and search.

### Telegram Bot

The Telegram bot provides mobile document capture and conversational queries. Setup:

1. Create a bot via [@BotFather](https://t.me/BotFather) on Telegram
2. Install the telegram extra: `uv sync --extra telegram`
3. Set the token: `export TELEGRAM_BOT_TOKEN=<your-token>`
4. Start the bot: `uv run lt telegram start`

**Document capture** -- send a photo or file to the bot, or use a type command first:

| Command | Description |
|---------|-------------|
| `/receipt` | Next upload is a receipt |
| `/invoice` | Next upload is an invoice |
| `/warranty` | Next upload is a warranty document |
| (send photo) | Auto-detect document type |

**Queries and corrections:**

| Command | Description |
|---------|-------------|
| `/expenses [N]` | Recent transactions (default: 10) |
| `/summary [period]` | Spending summary (week/month/year) |
| `/find <query>` | Search facts by vendor or item |
| `/fix` | Correct a fact (reply to a bot message) |
| `/merge` | Merge duplicate vendors |
| `/tag <key> <value>` | Add annotation to a fact |
| `/untag <key>` | Remove annotation |

**Enrichment and management:**

| Command | Description |
|---------|-------------|
| `/enrich` | Review pending enrichment items |
| `/scan` | Scan a barcode from a photo |
| `/barcode` | Manual barcode entry |
| `/lineitem` | Edit a line item |
| `/budget` | Budget management |
| `/language` | Set preferred language |
| `/help` | List all commands |

The bot supports multi-user mode -- each Telegram user can be linked to an alibi user account via `/link`.

**Security: restrict bot access** to your family by setting allowed Telegram user IDs:

```bash
# Find your Telegram user ID by messaging @userinfobot
export ALIBI_TELEGRAM_ALLOWED_USERS=123456789,987654321
```

When set, the bot silently ignores messages from all other users. **Strongly recommended** for any internet-facing bot.

### File Watcher

```bash
uv run lt daemon start                # Watch inbox for new documents
uv run lt daemon start --foreground   # Run in foreground (see logs)
uv run lt daemon status               # Check if watcher is running
uv run lt daemon stop                 # Stop the watcher
```

Auto-processes documents dropped into the configured inbox folder. Supports typed subfolders (`receipts/`, `invoices/`, etc.) for automatic document type routing.

### Predictions (MindsDB)

Alibi integrates with [MindsDB](https://mindsdb.com/) for ML-powered predictions on your financial data. Two models are available:

**Spending forecast** -- time-series model that predicts future spending by category, trained on your historical purchase data. Uses a configurable lookback window (default: 6 months) to forecast upcoming months.

**Category classifier** -- learns from your existing categorized items (vendor + item name + amount) and predicts categories for new uncategorized items. Requires at least 50 categorized items to train.

**Setup:**

```bash
# 1. Run MindsDB (self-hosted, no cloud dependency)
docker run -p 47334:47334 mindsdb/mindsdb:lightwood

# 2. Enable in .env
ALIBI_MINDSDB_ENABLED=true
ALIBI_MINDSDB_URL=http://127.0.0.1:47334

# 3. Train models (requires existing data in alibi)
uv run lt predictions train-forecast     # ~1-5 min depending on data volume
uv run lt predictions train-category     # needs 50+ categorized items
```

**CLI usage:**

```bash
uv run lt predictions status                           # Model status
uv run lt predictions forecast                         # 3-month spending forecast
uv run lt predictions forecast -m 6 -c Groceries      # 6-month, one category
uv run lt predictions classify "Alphamega" "Milk" 2.50 # Single item
uv run lt predictions classify-pending                 # Preview uncategorized
uv run lt predictions classify-pending --apply         # Apply to database
```

**API endpoints** (all under `/api/v1/predictions`):

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/train/forecast` | Train spending forecast (window, horizon params) |
| POST | `/train/category` | Train category classifier |
| GET | `/forecast` | Get spending predictions (months, category filter) |
| POST | `/classify` | Classify a single item (vendor, name, amount) |
| GET | `/classify/uncategorized` | Batch classify uncategorized items |
| GET | `/models` | List all models and their training status |
| GET | `/models/{name}/status` | Check specific model status |
| DELETE | `/models/{name}` | Delete a model |

Retrain models periodically as you accumulate more data for improved accuracy.

### Analytics Export

Alibi can push its structured data to an external analytics service for dashboarding, BI, or LLM-powered analysis. When enabled, every fact creation/update triggers an HTTP POST with the full dataset:

```bash
ALIBI_ANALYTICS_EXPORT_ENABLED=true
ALIBI_ANALYTICS_STACK_URL=http://localhost:8070
```

The payload is a JSON snapshot posted to `{url}/v1/ingest/alibi`:

```json
{
  "facts": [{"id": "...", "vendor": "...", "total_amount": 42.50, ...}],
  "fact_items": [{"id": "...", "name": "Milk", "brand": "...", "category": "dairy", ...}],
  "annotations": [{"key": "project", "value": "kitchen", ...}],
  "documents": [{"id": "...", "source": "telegram", "user_id": "...", ...}]
}
```

Build your own receiver with any framework (FastAPI, Express, etc.) and store in any database (PostgreSQL, ClickHouse, DuckDB). The export is full-replace (not incremental), so your receiver can simply truncate and reload. See [`.env.example`](.env.example) for the full payload schema.

### Vector Search (LanceDB)

Optional semantic search across documents, facts, and items using `nomic-embed-text` embeddings via Ollama.

```bash
# Install vector dependencies
uv sync --extra vector

# Set storage path in .env
ALIBI_LANCE_PATH=data/lancedb

# Build the index from existing data
uv run lt vectordb init --rebuild

# Search
uv run lt vectordb search "grocery shopping"
```

API: `GET /api/v1/search?q=milk&semantic=true` uses vector similarity when LanceDB is configured, with automatic SQL fallback.

### Budgets

Budget scenarios with actual-vs-target comparison and variance tracking.

```bash
uv run lt budget create "March 2026" --type target
uv run lt budget add-entry <id> groceries 500 2026-03
uv run lt budget actual 2026-03
uv run lt budget compare <base-id> <compare-id>
```

API: full CRUD at `/api/v1/budgets/` (scenarios, entries, comparison). Also available via Telegram (`/budget`) and MCP.

## Project Structure

```
alibi/
  config.py                # pydantic-settings (ALIBI_ prefix)
  cli.py                   # Click CLI entry point (lt command)
  commands/                # CLI command modules (10 modules, 60+ commands)
  services/                # Service layer (13 facades + events + 4 subscribers)
  api/routers/             # FastAPI REST API (19 routers, ~107 endpoints)
  web/                     # Web UI (single-page application)
  mcp/                     # MCP server (25 tools)
  telegram/                # Telegram bot (13 handlers + middleware)
  extraction/              # 3-stage pipeline: OCR, heuristic parser, LLM structurer
  processing/              # Pipeline orchestration, folder routing, image optimizer
  atoms/                   # Atom parsing from extraction data
  clouds/                  # Cloud formation, collapse, split payment detection
  identities/              # Vendor identity resolution and matching
  enrichment/              # Product enrichment cascade (8 tiers)
  analytics/               # Spending, subscriptions, anomalies, nutrition, price factors
  normalizers/             # Pure functions: vendors, numbers, units, dates, tax
  refiners/                # Per-type refiners: purchase, invoice, warranty, contract
  predictions/             # MindsDB predictors (spending forecast, category classifier)
  masking/                 # 4-tier data masking for privacy-controlled disclosure
  anonymization/           # 3-level anonymization export
  annotations/             # Open-ended metadata on entities (tags, notes, attributes)
  budgets/                 # Budget scenarios with category hierarchies
  matching/                # Duplicate detection (hash, perceptual, fuzzy)
  backup.py                # Database backup and restore with checksums
  maintenance/             # Learning aggregation, template reliability
  db/                      # SQLite schema, connection, migrations (35 migrations)
  daemon/                  # File watcher + enrichment scheduler
  i18n/                    # Multi-language support
  auth/                    # API key generation and validation (PBKDF2+salt)
tests/                     # ~4650 tests
data/                      # Runtime data (gitignored)
docs/                      # Architecture docs and ADRs
```

## Configuration

All settings use `ALIBI_` prefix. See [`.env.example`](.env.example) for the full reference.

**Minimal setup** (local-only processing):
```bash
ALIBI_OLLAMA_URL=http://127.0.0.1:11434
ALIBI_DEFAULT_CURRENCY=EUR
```

**With cloud Stage 3** (faster, higher accuracy on complex receipts):
```bash
ALIBI_GEMINI_API_KEY=your-key-here
ALIBI_GEMINI_EXTRACTION_ENABLED=true
```

**Full setup** (all features):
```bash
ALIBI_GEMINI_API_KEY=...                    # Gemini extraction + enrichment
ALIBI_GEMINI_ENRICHMENT_ENABLED=true
ALIBI_CLOUD_ENRICHMENT_ENABLED=true         # Anthropic enrichment
ALIBI_ANTHROPIC_API_KEY=sk-ant-...
TELEGRAM_BOT_TOKEN=...                      # Telegram bot
ALIBI_MINDSDB_ENABLED=true                  # MindsDB predictions
ALIBI_ENRICHMENT_SCHEDULE_ENABLED=true      # Background enrichment daemon
```

Every variable is documented with its purpose and default value in [`.env.example`](.env.example).

## Tested On

Alibi has been developed and tested primarily on **Cyprus receipts and invoices** (Greek + English, EUR). The extraction pipeline handles:
- Supermarket receipts (Alphamega, Papas, and others)
- Fuel station receipts (EKO, Petrolina)
- Restaurant and retail receipts
- Bank statements and payment confirmations
- Card terminal slips (JCC, viva.com, Worldline)

The heuristic parser and LLM correction are language-agnostic. Non-Latin scripts (Greek, Russian, CJK) are supported via the OCR fallback model (`minicpm-v`).

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System design, data model, service layer, extraction pipeline |
| [Enrichment Pipeline ADR](docs/adr/enrichment-pipeline.md) | 8-tier product enrichment cascade design decision |
| [Bulk Corrections Guide](docs/guides/bulk-corrections.md) | Edit data in SQLite GUI, sync back to correction system |
| [iOS Capture Guide](docs/guides/ios-capture.md) | iOS Shortcuts for receipt photo capture |
| [Obsidian Note Example](docs/examples/obsidian-note.md) | Sample generated Obsidian note from a receipt |

## Development

```bash
uv run pytest -x                             # Run tests (~4650 passing)
uv run black . && uv run flake8 && uv run mypy   # Lint + type check
```

## Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Write tests for new functionality
4. Run `uv run pytest -x` and `uv run black . && uv run flake8` before submitting
5. Open a pull request

## Research Paper

The Atom-Cloud-Fact data model and the epistemological foundations of multi-source data reconciliation are described in:

> Zharnikov, D. (2026). *The atom-cloud-fact epistemological pipeline: From financial document processing to brand perception modeling*. Available at: https://github.com/spectralbranding/sbt-papers/tree/main/alibi-epistemology

## How to Cite

If you use Alibi in your research or build on its concepts, please cite the paper above. A `CITATION.cff` file is included for automated citation tools.

## License

[MIT](LICENSE)
