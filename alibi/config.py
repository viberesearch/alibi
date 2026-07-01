"""Configuration management for Alibi."""

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Alibi configuration settings with environment variable support.

    Environment variables use ALIBI_ prefix (e.g., ALIBI_DB_PATH).
    """

    model_config = SettingsConfigDict(
        env_prefix="ALIBI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore unknown env vars
    )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize config, skipping .env in test mode."""
        if os.getenv("ALIBI_TESTING") and "_env_file" not in kwargs:
            kwargs["_env_file"] = None
        super().__init__(**kwargs)

    # Database
    db_path: Path = Field(default=Path("data/alibi.db"))

    # Obsidian vault
    vault_path: Optional[Path] = Field(default=None)

    # Inbox folder for document capture (relative to vault)
    inbox_folder: str = Field(default="inbox/documents")

    # Ollama settings
    ollama_url: str = Field(default="http://127.0.0.1:11434")
    ollama_model: str = Field(default="qwen3-vl:30b")
    ollama_ocr_model: str = Field(default="glm-ocr")
    ollama_ocr_fallback_model: Optional[str] = Field(default=None)
    # Structuring model (Stage 3, OCR text -> schema JSON). gemma4:12b-mlx
    # (MLX/MTP engine, Ollama 0.31+) replaced plain gemma4:12b after a 6-receipt
    # production-path A/B: better quality (verify 0.925 vs 0.848, fill 0.883 vs
    # 0.85; json_ok + items tied) AND ~44% faster (13.6s vs 24.2s/doc). MTP draft
    # tokens raise eval_count (754 vs 470) but wall-clock still wins. Text-only,
    # which is correct here: Stage 3 is text->JSON; vision stays on qwen3-vl:30b.
    # History: gemma4:12b had beaten qwen3.5:9b (verify 0.87 vs 0.76, fill 0.85
    # vs 0.77). Routes via /api/chat with think=false + format schema.
    ollama_structure_model: str = Field(default="gemma4:12b-mlx")
    ollama_keep_alive: int = Field(default=300)  # seconds; 0 = unload immediately
    ollama_num_predict: int = Field(
        default=4096
    )  # tokens; must be high for thinking models
    # Context window for OCR + structuring calls. alibi documents are tiny
    # (largest receipt OCR ~2.6k tokens; +prompt/schema ~5k worst case), so a
    # small ctx is plenty. Setting it explicitly decouples alibi from each
    # machine's Ollama GUI default (e.g. 256k) — a large default reserves a
    # huge KV cache per loaded model and causes needless VRAM pressure.
    ollama_num_ctx: int = Field(default=8192)
    # Per-document escalation: a large receipt (many line items) can exhaust
    # the output budget and return truncated, unparseable JSON. When Ollama
    # signals this (done_reason="length") the structuring call is retried once
    # with these larger budgets. num_ctx must also grow since prompt + output
    # must fit the context window.
    ollama_num_predict_escalated: int = Field(default=8192)
    ollama_num_ctx_escalated: int = Field(default=16384)
    ocr_backend: str = Field(default="ollama")  # "ollama" or "doctr"
    # Skip LLM (Stage 3) when parser confidence is at or above this threshold.
    # Set to 1.1 to never skip. Default 0.9 skips LLM for well-parsed receipts.
    skip_llm_threshold: float = Field(default=0.9)

    # Processing
    auto_process: bool = Field(default=False)

    # OCR timeout per request (seconds)
    ocr_timeout: float = Field(default=120.0)

    # Structuring (Stage 3) request timeout. gemma4:12b can exceed 120s on a
    # long receipt; raised so long documents finish rather than time out.
    structure_timeout: float = Field(default=240.0)

    # Hard-document handling -------------------------------------------------
    # Orientation sweep: when the local OCR base read looks mis-oriented
    # (short, script-incoherent, or symbol-noisy), run a 4-way 90-degree sweep
    # and keep the orientation that reads the most DISTINCT priced lines
    # (robust to glm-ocr hallucination loops). A clean upright receipt is not
    # flagged, so the extra OCR passes are skipped for the common case.
    ocr_orientation_sweep: bool = Field(default=True)

    # Confidence-gated cloud escalation. When the local pipeline's verification
    # confidence is below escalation_confidence_threshold OR the amount-line
    # coverage check shows dropped items, re-run the document through the cloud
    # vision model and keep the better result (by coverage + reconciliation).
    # Independent of gemini_extraction_enabled — cloud stays off for the common
    # path; this is a targeted rescue for hard docs the local model can't read.
    # Requires gemini_api_key to be set.
    gemini_escalation_enabled: bool = Field(default=True)
    escalation_confidence_threshold: float = Field(default=0.6)

    # Image optimization on ingest (Phase 6)
    optimize_images: bool = Field(default=True)
    image_max_dim: int = Field(default=2048)
    image_quality: int = Field(default=85)

    # Prompt mode: 'specialized' uses type-specific V2 prompts,
    # 'universal' uses UNIVERSAL_PROMPT_V2 for all document types.
    prompt_mode: str = Field(default="specialized")

    # Constrain the local Ollama structuring model to the extraction JSON
    # schema (Ollama `format` param). Turns the prose "return JSON" prompt
    # into an enforced contract — the model can only emit schema-conforming
    # output. Mirrors the Gemini path's Pydantic response_schema. The schema
    # fields are the canonical normalization target, so country-specific
    # receipts (MwSt, TVA, sales tax, GST) all collapse into the same fields.
    ollama_structured_output: bool = Field(default=True)

    # Default currency
    default_currency: str = Field(default="EUR")

    # Display language for multi-language support
    display_language: str = Field(default="original")

    # Telegram bot (uses TELEGRAM_BOT_TOKEN, not ALIBI_ prefix for compatibility)
    telegram_token: str = Field(
        default="",
        validation_alias="TELEGRAM_BOT_TOKEN",
    )

    # Comma-separated Telegram user IDs allowed to use the bot.
    # When set, only these users can interact with the bot; all others are ignored.
    # Find your Telegram user ID by messaging @userinfobot on Telegram.
    # Example: ALIBI_TELEGRAM_ALLOWED_USERS=123456789,987654321
    telegram_allowed_users: str = Field(default="")

    # API settings
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=3100)
    api_key: Optional[str] = Field(default=None)

    # Unit alias overrides (YAML file)
    unit_aliases_path: Optional[Path] = Field(default=Path("data/unit_aliases.yaml"))

    # Vendor alias overrides (YAML file)
    vendor_aliases_path: Optional[Path] = Field(
        default=Path("data/vendor_aliases.yaml")
    )

    # YAML store: separate tree for .alibi.yaml caches (decoupled from source dirs).
    yaml_store: Path = Field(default=Path("data/yaml_store"))

    # Analytics stack export
    analytics_export_enabled: bool = Field(default=False)
    analytics_stack_url: str = Field(default="http://localhost:8070")

    # Cloud enrichment (Anthropic API)
    cloud_enrichment_enabled: bool = Field(default=False)
    anthropic_api_key: Optional[str] = Field(default=None)
    cloud_enrichment_model: str = Field(default="claude-sonnet-4-6")
    llm_enrichment_timeout: float = Field(default=60.0)

    # MindsDB predictors
    mindsdb_enabled: bool = Field(default=False)
    mindsdb_url: str = Field(default="http://127.0.0.1:47334")

    # Gemini enrichment
    gemini_api_key: Optional[str] = Field(default=None)
    gemini_enrichment_enabled: bool = Field(default=False)
    gemini_enrichment_model: str = Field(default="gemini-3.5-flash")

    # Gemini extraction (Stage 3 replacement for Ollama qwen3.5:9b)
    gemini_extraction_enabled: bool = Field(default=False)
    gemini_extraction_model: str = Field(default="gemini-3.5-flash")
    # Output token budget for Gemini extraction/vision. A long receipt (30+
    # line items) overruns the old 2048 default and returns truncated,
    # unparseable JSON; 8192 covers the largest documents in the corpus.
    gemini_max_output_tokens: int = Field(default=8192)

    # Post-batch duplicate-fact resolution. When enabled, a batch ingest runs a
    # gated dedup pass at the end, auto-merging only corroborated duplicates
    # (price-overlap / perceptual-hash / zero-item twin) and leaving anything
    # ambiguous for manual review. Off by default — deletion is opt-in.
    dedup_after_batch: bool = Field(default=False)

    # Open Food Facts contribution (submit enriched products back to OFF)
    off_contribution_enabled: bool = Field(default=False)

    # Scheduled enrichment
    enrichment_schedule_enabled: bool = Field(default=False)
    enrichment_schedule_interval: int = Field(default=21600)  # 6 hours
    enrichment_schedule_gemini_interval: int = Field(default=259200)  # 3 days
    enrichment_schedule_maintenance_interval: int = Field(default=604800)  # 7 days
    enrichment_schedule_limit: int = Field(default=500)

    # LanceDB vector path
    lance_path: Optional[Path] = Field(default=None)

    def get_absolute_db_path(self) -> Path:
        """Get absolute path to database file."""
        if self.db_path.is_absolute():
            return self.db_path
        # Relative to project root
        return Path(__file__).parent.parent / self.db_path

    def get_unit_aliases_path(self) -> Optional[Path]:
        """Get absolute path to unit aliases YAML file."""
        if self.unit_aliases_path is None:
            return None
        if self.unit_aliases_path.is_absolute():
            return self.unit_aliases_path
        return Path(__file__).parent.parent / self.unit_aliases_path

    def get_vendor_aliases_path(self) -> Optional[Path]:
        """Get absolute path to vendor aliases YAML file."""
        if self.vendor_aliases_path is None:
            return None
        if self.vendor_aliases_path.is_absolute():
            return self.vendor_aliases_path
        return Path(__file__).parent.parent / self.vendor_aliases_path

    def get_inbox_path(self) -> Optional[Path]:
        """Get absolute path to inbox folder."""
        if self.vault_path is None:
            return None
        return self.vault_path / self.inbox_folder

    def get_yaml_store_path(self) -> Path:
        """Get absolute path to YAML store directory."""
        if self.yaml_store.is_absolute():
            return self.yaml_store
        return Path(__file__).parent.parent / self.yaml_store

    def get_lance_path(self) -> Path:
        """Get absolute path to LanceDB directory."""
        if self.lance_path is not None:
            if self.lance_path.is_absolute():
                return self.lance_path
            return Path(__file__).parent.parent / self.lance_path
        # Default: data/lancedb relative to project root
        return Path(__file__).parent.parent / "data" / "lancedb"

    def validate_paths(self) -> list[str]:
        """Validate that required paths exist. Returns list of errors."""
        errors = []

        # Check vault path if set
        if self.vault_path is not None and not self.vault_path.exists():
            errors.append(f"Vault path does not exist: {self.vault_path}")

        return errors


# Global config singleton
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create config singleton."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reset_config() -> None:
    """Reset config singleton (useful for testing)."""
    global _config
    _config = None


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent
