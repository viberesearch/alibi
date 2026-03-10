FROM python:3.12-slim

# System dependencies for PDF and barcode processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cache layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --frozen

# Copy application code
COPY alibi/ alibi/

# Create non-root user and data directories
RUN groupadd -r alibi && useradd -r -g alibi -d /app alibi \
    && mkdir -p data/yaml_store data/inbox data/backups \
    && chown -R alibi:alibi /app

USER alibi

# Default environment
ENV ALIBI_DB_PATH=/app/data/alibi.db
ENV ALIBI_INBOX_PATH=/app/data/inbox
ENV ALIBI_OLLAMA_URL=http://ollama:11434

EXPOSE 3100

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:3100/api/v1/health')" || exit 1

# API server by default
CMD ["uv", "run", "uvicorn", "alibi.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "3100"]
