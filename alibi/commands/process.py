"""Process and scan commands."""

from __future__ import annotations

from typing import Any

import click

from alibi.commands.shared import console, format_amount, is_quiet
from alibi.config import get_config
from alibi.db.connection import get_db
from alibi.errors import (
    NO_INBOX_CONFIGURED,
    UNSUPPORTED_FILE_TYPE,
    VAULT_NOT_FOUND,
    format_error,
)


@click.command()
@click.option("--path", "-p", type=click.Path(exists=True), help="Custom path to scan")
def scan(path: str | None) -> None:
    """Scan inbox for new documents."""
    from pathlib import Path

    from rich.table import Table

    from alibi.processing.watcher import scan_inbox

    config = get_config()

    scan_path = Path(path) if path else None
    inbox_path = scan_path or config.get_inbox_path()

    if inbox_path is None:
        NO_INBOX_CONFIGURED.display(console)
        raise click.Abort()

    if not inbox_path.exists():
        err = format_error(VAULT_NOT_FOUND, path=str(inbox_path))
        err.display(console)
        raise click.Abort()

    console.print(f"[bold blue]Scanning:[/bold blue] {inbox_path}\n")

    try:
        files = scan_inbox(inbox_path)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise click.Abort() from e

    if not files:
        console.print("[yellow]No supported files found.[/yellow]")
        return

    table = Table(title=f"Found {len(files)} document(s)")
    table.add_column("File", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Size", justify="right")

    for f in files:
        size = f.stat().st_size
        size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
        table.add_row(f.name, f.suffix.upper()[1:], size_str)

    console.print(table)
    console.print(f"\nRun 'lt process' to process these files.")


@click.command()
@click.option(
    "--path", "-p", type=click.Path(exists=True), help="Specific file or folder"
)
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be processed")
def process(path: str | None, dry_run: bool) -> None:
    """Process documents with OCR/Vision extraction."""
    from pathlib import Path

    from alibi.processing.folder_router import (
        FolderContext,
        scan_inbox_recursive,
    )
    from alibi.processing.pipeline import ProcessingResult
    from alibi.processing.watcher import is_supported_file, scan_inbox
    from alibi.services.ingestion import (
        process_batch,
        process_document_group,
        process_file,
    )

    config = get_config()
    db_manager = get_db()

    # Ensure database is initialized
    if not db_manager.is_initialized():
        console.print("[yellow]Database not initialized. Initializing...[/yellow]")
        db_manager.initialize()

    # Determine files to process
    document_groups: list[tuple[Path, list[Path]]] = []  # (folder, files) pairs
    single_files: list[Path] = []
    file_contexts: list[FolderContext] = []

    if path:
        path_obj = Path(path)
        if path_obj.is_file():
            if not is_supported_file(path_obj):
                err = format_error(UNSUPPORTED_FILE_TYPE, extension=path_obj.suffix)
                err.display(console)
                raise click.Abort()
            single_files = [path_obj]
            # Try to resolve folder context for single file
            inbox_path = config.get_inbox_path()
            if inbox_path and path_obj.resolve().is_relative_to(inbox_path.resolve()):
                from alibi.processing.folder_router import resolve_folder_context

                file_contexts = [resolve_folder_context(path_obj, inbox_path)]
            else:
                file_contexts = [FolderContext()]
        else:
            # Folder passed with -p: group all files as one document
            folder_files = sorted(
                [f for f in path_obj.iterdir() if f.is_file() and is_supported_file(f)]
            )
            if folder_files:
                document_groups.append((path_obj, folder_files))
            else:
                console.print("[yellow]No supported files in folder.[/yellow]")
                return
    else:
        inbox_path = config.get_inbox_path()
        if inbox_path is None:
            NO_INBOX_CONFIGURED.display(console)
            raise click.Abort()

        if not inbox_path.exists():
            console.print("[yellow]No files to process.[/yellow]")
            return

        # Recursive inbox scan with folder context
        scan_results = scan_inbox_recursive(inbox_path)
        for file_path, ctx in scan_results:
            single_files.append(file_path)
            file_contexts.append(ctx)

    total_items = len(single_files) + len(document_groups)
    if total_items == 0:
        console.print("[yellow]No files to process.[/yellow]")
        return

    # Show what will be processed
    total_pages = len(single_files) + sum(len(g[1]) for g in document_groups)
    if document_groups:
        console.print(
            f"[bold blue]Processing {total_items} document(s) "
            f"({total_pages} pages total)...[/bold blue]\n"
        )
        for folder, files in document_groups:
            console.print(f"  [cyan]Group:[/cyan] {folder.name}/ ({len(files)} pages)")
    else:
        console.print(
            f"[bold blue]Processing {len(single_files)} file(s)...[/bold blue]\n"
        )
        for f, ctx in zip(single_files, file_contexts):
            type_label = ctx.doc_type.value if ctx.doc_type else "unsorted"
            vendor_label = f" ({ctx.vendor_hint})" if ctx.vendor_hint else ""
            console.print(f"  {f.name} [dim]{type_label}{vendor_label}[/dim]")
        console.print()

    if dry_run:
        for f, ctx in zip(single_files, file_contexts):
            type_label = ctx.doc_type.value if ctx.doc_type else "auto-detect"
            console.print(f"  Would process: {f.name} [{type_label}]")
        for folder, files in document_groups:
            console.print(f"  Would process group: {folder.name}/ ({len(files)} pages)")
            for f in files:
                console.print(f"    - {f.name}")
        return

    # Process via service layer
    results: list[ProcessingResult] = []

    # Set CLI provenance on all folder contexts
    for ctx in file_contexts:
        if ctx.source is None:
            ctx.source = "cli"
        if ctx.user_id is None:
            ctx.user_id = "system"

    # Process single files with folder contexts
    for f, ctx in zip(single_files, file_contexts):
        result = process_file(db_manager, f, folder_context=ctx)
        results.append(result)

    # Process document groups
    for folder, files in document_groups:
        group_ctx = FolderContext(source="cli", user_id="system")
        result = process_document_group(db_manager, files, folder_context=group_ctx)
        results.append(result)

    # Show results
    success_count = sum(1 for r in results if r.success and not r.is_duplicate)
    duplicate_count = sum(1 for r in results if r.is_duplicate)
    error_count = sum(1 for r in results if not r.success)

    console.print()
    for result in results:
        name = result.file_path.name
        if result.success:
            if result.is_duplicate:
                console.print(f"[yellow]Duplicate:[/yellow] {name}")
            else:
                console.print(f"[green]Processed:[/green] {name}")
                if result.extracted_data:
                    vendor = result.extracted_data.get("vendor", "Unknown")
                    total = result.extracted_data.get("total", "N/A")
                    console.print(f"  Vendor: {vendor}, Total: {total}")
        else:
            console.print(f"[red]Error:[/red] {name}")
            if result.error:
                console.print(f"  {result.error}")

    console.print()
    console.print("[bold]Summary:[/bold]")
    console.print(f"  Processed: {success_count}")
    console.print(f"  Duplicates: {duplicate_count}")
    console.print(f"  Errors: {error_count}")


@click.command("gemini-batch")
@click.option("--limit", "-l", default=10, help="Maximum documents to process")
@click.option(
    "--threshold",
    "-t",
    default=0.9,
    help="Only process docs below this parser confidence",
)
@click.option("--dry-run", "-n", is_flag=True, help="Show what would be processed")
def gemini_batch(limit: int, threshold: float, dry_run: bool) -> None:
    """Batch-extract documents via Gemini (Stage 3 replacement).

    Finds YAML files where parser confidence is below threshold and
    re-extracts them using Gemini in a single batch API call.
    """
    from pathlib import Path

    import yaml

    from alibi.extraction.gemini_structurer import (
        GeminiExtractionError,
        structure_ocr_texts_gemini,
    )

    config = get_config()

    if not config.gemini_extraction_enabled:
        console.print(
            "[red]Gemini extraction not enabled.[/red] "
            "Set ALIBI_GEMINI_EXTRACTION_ENABLED=true"
        )
        return

    if not config.gemini_api_key:
        console.print("[red]ALIBI_GEMINI_API_KEY not configured.[/red]")
        return

    # Scan YAML store for files with low parser confidence
    yaml_store = config.get_yaml_store_path()
    if not yaml_store.exists():
        console.print("[yellow]No YAML store found.[/yellow]")
        return

    candidates: list[tuple[Path, dict[str, Any]]] = []
    for yaml_path in sorted(yaml_store.rglob("*.alibi.yaml")):
        try:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            if not data or not isinstance(data, dict):
                continue

            parser_conf = data.get("_parser_confidence", 1.0)
            raw_text = data.get("raw_text", "")
            if parser_conf < threshold and raw_text:
                candidates.append((yaml_path, data))
                if len(candidates) >= limit:
                    break
        except Exception:
            continue

    if not candidates:
        console.print(
            f"[yellow]No YAML files with parser confidence < {threshold}.[/yellow]"
        )
        return

    console.print(f"Found {len(candidates)} documents for Gemini extraction:")
    for yaml_path, data in candidates:
        vendor = data.get("vendor", "Unknown")
        conf = data.get("_parser_confidence", "?")
        console.print(f"  {yaml_path.name}: {vendor} (confidence={conf})")

    if dry_run:
        console.print("\n[yellow]Dry run — no changes made.[/yellow]")
        return

    # Build batch
    documents = []
    for yaml_path, data in candidates:
        doc_type = data.get("document_type", "receipt")
        raw_text = data.get("raw_text", "")
        documents.append({"raw_text": raw_text, "doc_type": doc_type})

    console.print(f"\nSending {len(documents)} documents to Gemini...")
    try:
        results = structure_ocr_texts_gemini(documents)
    except GeminiExtractionError as e:
        console.print(f"[red]Gemini batch failed:[/red] {e}")
        return

    # Update YAML files with Gemini results
    updated = 0
    for (yaml_path, original_data), result in zip(candidates, results):
        if "_error" in result:
            console.print(f"  [red]Failed:[/red] {yaml_path.name}")
            continue

        # Merge Gemini result into existing YAML (preserve _meta, raw_text)
        merged = dict(original_data)
        for key, value in result.items():
            if key.startswith("_"):
                continue
            if value is not None:
                merged[key] = value

        merged["_pipeline"] = "gemini_batch_extraction"

        with open(yaml_path, "w") as f:
            yaml.dump(merged, f, default_flow_style=False, allow_unicode=True)

        vendor = result.get("vendor", original_data.get("vendor", "?"))
        console.print(f"  [green]Updated:[/green] {yaml_path.name} ({vendor})")
        updated += 1

    console.print(f"\n[bold]Updated {updated}/{len(candidates)} YAML files.[/bold]")
    console.print("Run 'lt reingest --scan' to re-ingest updated files.")


@click.command("gemini-vision")
@click.option(
    "--path",
    "-p",
    type=click.Path(exists=True),
    required=True,
    help="Image file to extract from",
)
@click.option("--type", "-t", "doc_type", default="receipt", help="Document type")
def gemini_vision(path: str, doc_type: str) -> None:
    """Extract structured data from an image via Gemini Vision (bypasses OCR)."""
    from alibi.extraction.gemini_structurer import (
        GeminiExtractionError,
        extract_from_image_gemini,
    )

    config = get_config()

    if not config.gemini_extraction_enabled:
        console.print(
            "[red]Gemini extraction not enabled.[/red] "
            "Set ALIBI_GEMINI_EXTRACTION_ENABLED=true"
        )
        return

    if not config.gemini_api_key:
        console.print("[red]ALIBI_GEMINI_API_KEY not configured.[/red]")
        return

    console.print(f"Extracting from {path} via Gemini Vision...")
    try:
        result = extract_from_image_gemini(path, doc_type=doc_type)
    except GeminiExtractionError as e:
        console.print(f"[red]Gemini vision failed:[/red] {e}")
        return

    import json

    console.print(json.dumps(result, indent=2, ensure_ascii=False))
