# Contributing to Alibi

Thank you for your interest in contributing to Alibi. This guide covers the development setup, conventions, and process for submitting changes.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- SQLite 3.35+ (ships with Python 3.12)
- For OCR: [Ollama](https://ollama.com/) with `glm-ocr` model
- For barcode scanning: `zbar` library (`brew install zbar` / `apt install libzbar0`)
- For PDF processing: `poppler` (`brew install poppler` / `apt install poppler-utils`)

## Development Setup

```bash
# Clone and install
git clone https://github.com/viberesearch/alibi.git
cd alibi
uv sync --group dev

# Install optional dependencies as needed
uv sync --group dev --extra telegram   # Telegram bot
uv sync --group dev --extra gemini     # Gemini enrichment
uv sync --group dev --extra barcode    # Barcode scanning
uv sync --group dev --extra all        # Everything

# Initialize database
uv run lt init

# Verify setup
uv run lt health
```

## Running Tests

```bash
# Full test suite (4,490+ tests)
uv run pytest

# Stop on first failure
uv run pytest -x

# Run specific test file
uv run pytest tests/test_text_parser.py

# Run specific test
uv run pytest tests/test_text_parser.py::TestReceiptParsing::test_basic_receipt -v

# Skip integration tests (require running services)
uv run pytest --ignore=tests/integration
```

## Code Quality

```bash
# Format
uv run black alibi/ tests/

# Lint
uv run flake8 alibi/ tests/

# Type check
uv run mypy alibi/

# All checks (mirrors pre-commit hook)
uv run black alibi/ tests/ && uv run flake8 alibi/ tests/ && uv run mypy alibi/
```

## Branch Conventions

- **Never commit directly to `main`**
- Branch naming: `type/description` (e.g., `feature/barcode-enrichment`, `fix/parser-amount-detection`)
- Types: `feature/`, `fix/`, `refactor/`, `docs/`, `test/`

## Project Structure

```
alibi/
  api/              # FastAPI REST API + middleware
  auth/             # API key hashing and validation
  clouds/           # Cloud formation, collapse, learning
  commands/         # CLI command modules (Click)
  db/               # SQLite schema, migrations, v2_store
  enrichment/       # OFF, UPCitemdb, GS1, Gemini enrichment
  extraction/       # OCR, vision, text parsing, structuring
  ingestion/        # CSV/OFX import
  matching/         # Vendor matching and identity resolution
  mcp/              # Model Context Protocol server
  normalizers/      # Amount, date, vendor normalization
  predictions/      # MindsDB integration
  processing/       # Document pipeline, folder routing
  services/         # Service layer (facade over internal modules)
  telegram/         # Telegram bot handlers
tests/              # 145 test files, mirrors source structure
data/               # Runtime data (gitignored, except examples)
docs/               # Architecture and design documentation
```

## Key Patterns

### Service Layer

All external interfaces (CLI, API, MCP, Telegram) should route through `alibi/services/`. The service layer provides the public API for internal modules:

```python
# Good: use service layer
from alibi.services import search_facts, update_fact

# Avoid: direct module access from interfaces
from alibi.db.v2_store import fetch_facts  # internal
```

### Database Access

- All SQL uses parameterized queries (no string formatting)
- `DatabaseManager` provides `fetchone()`, `fetchall()`, `execute()`, `transaction()`
- Migrations in `alibi/db/migrations/` with up/down SQL blocks
- Schema version tracked in `schema_version` table

### Adding a Migration

1. Create `alibi/db/migrations/NNN_description.sql`
2. Include both `-- UP` and `-- DOWN` blocks
3. End the UP block with `INSERT OR IGNORE INTO schema_version (version) VALUES (NNN);`
4. Update the version in `alibi/db/schema.sql` seed list
5. Update schema version assertions in tests if needed

### Document Processing Pipeline

1. **Ingest**: Image optimization, hash computation, deduplication
2. **Extract**: OCR (glm-ocr) -> text parser -> LLM structuring (optional)
3. **Atomize**: Parse structured data into typed Atoms
4. **Bundle**: Group related atoms into Bundles
5. **Cloud**: Match bundles across documents into Clouds
6. **Collapse**: Derive Facts from validated Clouds

## Pull Request Process

1. Create a feature branch from `main`
2. Make changes with tests
3. Ensure all checks pass: `uv run black . && uv run flake8 && uv run mypy alibi/ && uv run pytest`
4. Write a clear PR description explaining the "why"
5. Link to any related issues

## Commit Messages

Follow conventional commits style:
- `feat:` new feature
- `fix:` bug fix
- `refactor:` code restructuring (no behavior change)
- `docs:` documentation only
- `test:` adding/fixing tests
- `chore:` tooling, CI, dependencies

## Environment Variables

All configuration is via environment variables (no secrets in code). See `.env.example` for the full list. Key variables:

| Variable | Purpose |
|----------|---------|
| `ALIBI_DB_PATH` | SQLite database path |
| `ALIBI_INBOX_PATH` | Document inbox directory |
| `ALIBI_OLLAMA_URL` | Ollama API endpoint |
| `ALIBI_GEMINI_API_KEY` | Gemini API key (optional cloud escalation) |
| `ALIBI_ANTHROPIC_API_KEY` | Anthropic API key (optional cloud escalation) |

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
