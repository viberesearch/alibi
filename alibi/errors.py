"""Actionable error messages for Alibi CLI."""

from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel


@dataclass
class ActionableError:
    """Error with actionable suggestions."""

    message: str
    suggestions: list[str] = field(default_factory=list)
    error_code: str | None = None

    def display(self, console: Console) -> None:
        """Display error with suggestions using Rich."""
        lines = [f"[red bold]Error:[/red bold] {self.message}"]
        if self.suggestions:
            lines.append("")
            lines.append("[yellow]Suggestions:[/yellow]")
            for i, s in enumerate(self.suggestions, 1):
                lines.append(f"  {i}. {s}")
        console.print(Panel("\n".join(lines), border_style="red"))


# Pre-defined common errors
OLLAMA_CONNECTION_ERROR = ActionableError(
    message="Cannot connect to Ollama",
    suggestions=[
        "Start Ollama: ollama serve",
        "Check ALIBI_OLLAMA_URL in .env (current: {url})",
        "Verify Ollama is running: curl {url}/api/tags",
    ],
    error_code="E001",
)

OLLAMA_MODEL_NOT_FOUND = ActionableError(
    message="Ollama model '{model}' not found",
    suggestions=[
        "Pull the model: ollama pull {model}",
        "List available models: ollama list",
        "Change model in .env: ALIBI_OLLAMA_MODEL=<model>",
    ],
    error_code="E002",
)

VAULT_NOT_FOUND = ActionableError(
    message="Vault path does not exist: {path}",
    suggestions=[
        "Create the directory: mkdir -p {path}",
        "Update ALIBI_VAULT_PATH in .env",
        "Run 'lt setup' to reconfigure",
    ],
    error_code="E003",
)

DATABASE_NOT_FOUND = ActionableError(
    message="Database not found at: {path}",
    suggestions=[
        "Initialize database: lt init",
        "Check ALIBI_DB_PATH in .env",
    ],
    error_code="E004",
)

UNSUPPORTED_FILE_TYPE = ActionableError(
    message="Unsupported file type: {extension}",
    suggestions=[
        "Supported formats: .jpg, .jpeg, .png, .gif, .pdf",
        "Convert file to supported format",
    ],
    error_code="E005",
)

NO_INBOX_CONFIGURED = ActionableError(
    message="No inbox path configured",
    suggestions=[
        "Set ALIBI_VAULT_PATH environment variable",
        "Run 'lt setup' to configure",
        "Use --path option to specify a path directly",
    ],
    error_code="E006",
)

IMPORT_FAILED = ActionableError(
    message="Import failed: {reason}",
    suggestions=[
        "Check file format matches expected type",
        "Verify file is not corrupted",
        "Try specifying format explicitly with --format",
    ],
    error_code="E007",
)


def format_error(error: ActionableError, **kwargs: str) -> ActionableError:
    """Format error message with dynamic values."""
    return ActionableError(
        message=error.message.format(**kwargs),
        suggestions=[s.format(**kwargs) for s in error.suggestions],
        error_code=error.error_code,
    )
