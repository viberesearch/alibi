"""Config loading for folder-based document routing.

Loads _config.yaml (inbox/country level) and _vendor.yaml (vendor level)
to provide locale defaults and vendor identity hints.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ImageOptConfig:
    """Image optimization settings."""

    enabled: bool = False
    max_dim: int = 1500
    quality: int = 85
    strip_exif: bool = True


@dataclass
class InboxConfig:
    """Inbox-level configuration from _config.yaml."""

    default_country: str = "CY"
    default_currency: str = "EUR"
    default_language: str = "el"
    image_optimization: ImageOptConfig | None = None


@dataclass
class VendorLocation:
    """A vendor's physical location."""

    address: str = ""
    map_url: str = ""


@dataclass
class VendorConfig:
    """Vendor identity from _vendor.yaml."""

    trade_name: str = ""
    legal_name: str = ""
    country: str = ""
    vat_number: str = ""
    tax_id: str = ""
    locations: list[VendorLocation] = field(default_factory=list)
    phone: str = ""
    website: str = ""
    default_currency: str = ""
    loyalty_number: str = ""
    account_number: str = ""
    notes: str = ""


def load_inbox_config(inbox_root: Path) -> InboxConfig | None:
    """Load _config.yaml from inbox root.

    Returns None if the file doesn't exist.
    """
    config_path = inbox_root / "_config.yaml"
    if not config_path.is_file():
        return None
    return _parse_inbox_config(config_path)


def load_country_config(country_dir: Path) -> InboxConfig | None:
    """Load _config.yaml from a country subfolder.

    Returns None if the file doesn't exist.
    """
    config_path = country_dir / "_config.yaml"
    if not config_path.is_file():
        return None
    return _parse_inbox_config(config_path)


def load_vendor_config(vendor_dir: Path) -> VendorConfig | None:
    """Load _vendor.yaml from a vendor subfolder.

    Returns None if the file doesn't exist.
    """
    vendor_path = vendor_dir / "_vendor.yaml"
    if not vendor_path.is_file():
        return None

    try:
        with open(vendor_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load {vendor_path}: {e}")
        return None

    locations: list[VendorLocation] = []
    for loc in raw.get("locations") or []:
        if isinstance(loc, dict):
            locations.append(
                VendorLocation(
                    address=str(loc.get("address", "")),
                    map_url=str(loc.get("map_url", "")),
                )
            )

    return VendorConfig(
        trade_name=str(raw.get("trade_name", "")),
        legal_name=str(raw.get("legal_name", "")),
        country=str(raw.get("country", "")),
        vat_number=str(raw.get("vat_number", "")),
        tax_id=str(raw.get("tax_id", "")),
        locations=locations,
        phone=str(raw.get("phone", "")),
        website=str(raw.get("website", "")),
        default_currency=str(raw.get("default_currency", "")),
        loyalty_number=str(raw.get("loyalty_number", "")),
        account_number=str(raw.get("account_number", "")),
        notes=str(raw.get("notes", "")),
    )


def merge_configs(
    inbox_config: InboxConfig | None, country_config: InboxConfig | None
) -> InboxConfig:
    """Merge inbox and country configs. Country overrides inbox defaults."""
    base = inbox_config or InboxConfig()
    if country_config is None:
        return base
    return InboxConfig(
        default_country=country_config.default_country or base.default_country,
        default_currency=country_config.default_currency or base.default_currency,
        default_language=country_config.default_language or base.default_language,
        image_optimization=country_config.image_optimization or base.image_optimization,
    )


def _parse_inbox_config(config_path: Path) -> InboxConfig | None:
    """Parse an InboxConfig from a YAML file."""
    try:
        with open(config_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load {config_path}: {e}")
        return None

    img_opt = None
    img_raw = raw.get("image_optimization")
    if isinstance(img_raw, dict):
        img_opt = ImageOptConfig(
            enabled=bool(img_raw.get("enabled", False)),
            max_dim=int(img_raw.get("max_dim", 1500)),
            quality=int(img_raw.get("quality", 85)),
            strip_exif=bool(img_raw.get("strip_exif", True)),
        )

    return InboxConfig(
        default_country=str(raw.get("default_country", "")),
        default_currency=str(raw.get("default_currency", "")),
        default_language=str(raw.get("default_language", "")),
        image_optimization=img_opt,
    )
