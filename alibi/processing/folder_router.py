"""Folder-based document routing.

Resolves a file's path relative to the inbox root into a FolderContext
containing document type, country, vendor hint, and loaded configs.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from alibi.db.models import DocumentType
from alibi.processing.config_loader import (
    InboxConfig,
    VendorConfig,
    load_country_config,
    load_inbox_config,
    load_vendor_config,
    merge_configs,
)

logger = logging.getLogger(__name__)

# Folder name → DocumentType mapping
_FOLDER_TYPE_MAP: dict[str, DocumentType] = {
    "receipts": DocumentType.RECEIPT,
    "invoices": DocumentType.INVOICE,
    "payments": DocumentType.PAYMENT_CONFIRMATION,
    "statements": DocumentType.STATEMENT,
    "warranties": DocumentType.WARRANTY,
    "contracts": DocumentType.CONTRACT,
}

# 2-letter ISO 3166-1 alpha-2 country codes
_ISO_COUNTRY_RE = re.compile(r"^[A-Z]{2}$")

# Files to skip during inbox scanning
SKIP_FILES = {
    "_config.yaml",
    "_vendor.yaml",
    ".DS_Store",
    ".gitkeep",
}


@dataclass
class FolderContext:
    """Context derived from a file's folder path."""

    doc_type: DocumentType | None = None  # None = unsorted, needs classification
    country: str | None = None  # 2-letter ISO code
    vendor_hint: str | None = None  # Vendor folder name
    inbox_config: InboxConfig | None = None
    vendor_config: VendorConfig | None = None
    source: str | None = None  # Entry point: telegram, api, cli, watcher, mcp
    user_id: str | None = None  # User who submitted the document
    map_url: str | None = None  # User-provided Google Maps URL
    lat: float | None = None  # Parsed latitude from map_url
    lng: float | None = None  # Parsed longitude from map_url


def is_country_code(name: str) -> bool:
    """Check if a folder name is a 2-letter ISO 3166-1 alpha-2 code."""
    return bool(_ISO_COUNTRY_RE.match(name))


def resolve_folder_context(file_path: Path, inbox_root: Path) -> FolderContext:
    """Walk ancestors from file to inbox root, extract routing context.

    Resolution order:
    1. Type: nearest ancestor matching _FOLDER_TYPE_MAP keys
    2. Country: any ancestor that is a 2-letter uppercase ISO code
    3. Vendor: immediate parent if it's not a type folder or country code
    4. Files directly in inbox root: treated as unsorted

    Args:
        file_path: Path to the document file
        inbox_root: Path to the inbox root directory

    Returns:
        FolderContext with resolved type, country, vendor hint, and configs
    """
    inbox_root = inbox_root.resolve()
    file_path = file_path.resolve()

    # Collect ancestors between file and inbox root
    ancestors: list[str] = []
    current = file_path.parent
    while current != inbox_root and current != current.parent:
        ancestors.append(current.name)
        current = current.parent

    if current != inbox_root:
        # File is not inside inbox_root
        return FolderContext()

    # Resolve type, country, vendor from ancestors (nearest-first)
    doc_type: DocumentType | None = None
    country: str | None = None
    vendor_hint: str | None = None
    type_folder_name: str | None = None
    country_folder: Path | None = None
    vendor_folder: Path | None = None

    for folder_name in ancestors:
        lower = folder_name.lower()

        # Type detection
        if doc_type is None and lower in _FOLDER_TYPE_MAP:
            doc_type = _FOLDER_TYPE_MAP[lower]
            type_folder_name = lower

        # Country detection
        if country is None and is_country_code(folder_name):
            country = folder_name
            country_folder = file_path.parent
            # Walk up to find the actual country folder
            p = file_path.parent
            while p.name != folder_name and p != inbox_root:
                p = p.parent
            if p.name == folder_name:
                country_folder = p

        # Vendor detection: immediate parent that isn't a type or country folder
        if vendor_hint is None and folder_name == file_path.parent.name:
            if lower not in _FOLDER_TYPE_MAP and not is_country_code(folder_name):
                if lower != "unsorted":
                    vendor_hint = folder_name
                    vendor_folder = file_path.parent

    # "unsorted" folder means explicit classification needed
    if any(a.lower() == "unsorted" for a in ancestors):
        doc_type = None

    # Load configs
    inbox_config = load_inbox_config(inbox_root)
    country_cfg = load_country_config(country_folder) if country_folder else None
    merged_config = merge_configs(inbox_config, country_cfg)

    vendor_config = load_vendor_config(vendor_folder) if vendor_folder else None

    return FolderContext(
        doc_type=doc_type,
        country=country or (merged_config.default_country if merged_config else None),
        vendor_hint=vendor_hint,
        inbox_config=merged_config,
        vendor_config=vendor_config,
    )


def scan_inbox_recursive(inbox_root: Path) -> list[tuple[Path, FolderContext]]:
    """Recursively scan inbox for supported files with folder context.

    Returns a list of (file_path, folder_context) tuples, sorted by path.
    Skips config/meta files and .alibi.yaml caches.

    If yaml_store is configured and is a subdirectory of inbox_root, the
    yaml_store directory is also skipped entirely (it contains only YAML files).
    """
    from alibi.extraction.yaml_cache import _get_yaml_store_root
    from alibi.processing.watcher import is_supported_file

    results: list[tuple[Path, FolderContext]] = []
    inbox_root = inbox_root.resolve()

    if not inbox_root.is_dir():
        return results

    # Build set of directory prefixes to skip entirely
    skip_dirs: set[Path] = set()
    yaml_store_root = _get_yaml_store_root()
    if yaml_store_root is not None:
        yaml_store_root = yaml_store_root.resolve()
        try:
            yaml_store_root.relative_to(inbox_root)
            # yaml_store is inside inbox — exclude it from scanning
            skip_dirs.add(yaml_store_root)
        except ValueError:
            pass  # yaml_store is outside inbox — nothing to skip here

    for path in sorted(inbox_root.rglob("*")):
        if not path.is_file():
            continue

        # Skip files inside any excluded directory tree
        if any(path.is_relative_to(skip_dir) for skip_dir in skip_dirs):
            continue

        # Skip meta/config files
        if path.name in SKIP_FILES:
            continue
        if path.name.endswith(".alibi.yaml"):
            continue

        # Skip unsupported file types
        if not is_supported_file(path):
            continue

        ctx = resolve_folder_context(path, inbox_root)
        results.append((path, ctx))

    return results
