"""Tests for config loading for folder-based document routing."""

import os
from pathlib import Path

import pytest
import yaml

os.environ["ALIBI_TESTING"] = "1"

from alibi.processing.config_loader import (
    ImageOptConfig,
    InboxConfig,
    VendorConfig,
    VendorLocation,
    load_country_config,
    load_inbox_config,
    load_vendor_config,
    merge_configs,
)


class TestLoadInboxConfig:
    """Tests for load_inbox_config()."""

    def test_loads_valid_config(self, tmp_path: Path) -> None:
        config = {
            "default_country": "GR",
            "default_currency": "EUR",
            "default_language": "el",
        }
        (tmp_path / "_config.yaml").write_text(yaml.dump(config))
        result = load_inbox_config(tmp_path)
        assert result is not None
        assert result.default_country == "GR"
        assert result.default_currency == "EUR"
        assert result.default_language == "el"

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        result = load_inbox_config(tmp_path)
        assert result is None

    def test_handles_malformed_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "_config.yaml").write_text(": : : [invalid yaml\n\t\x00")
        result = load_inbox_config(tmp_path)
        assert result is None

    def test_handles_empty_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "_config.yaml").write_text("")
        result = load_inbox_config(tmp_path)
        # Empty YAML parses as None, handled gracefully
        assert result is not None
        assert result.default_country == ""

    def test_loads_image_optimization(self, tmp_path: Path) -> None:
        config = {
            "default_country": "CY",
            "image_optimization": {
                "enabled": True,
                "max_dim": 1200,
                "quality": 90,
                "strip_exif": False,
            },
        }
        (tmp_path / "_config.yaml").write_text(yaml.dump(config))
        result = load_inbox_config(tmp_path)
        assert result is not None
        assert result.image_optimization is not None
        assert result.image_optimization.enabled is True
        assert result.image_optimization.max_dim == 1200
        assert result.image_optimization.quality == 90
        assert result.image_optimization.strip_exif is False

    def test_image_optimization_defaults(self, tmp_path: Path) -> None:
        config = {
            "default_country": "CY",
            "image_optimization": {},
        }
        (tmp_path / "_config.yaml").write_text(yaml.dump(config))
        result = load_inbox_config(tmp_path)
        assert result is not None
        assert result.image_optimization is not None
        assert result.image_optimization.enabled is False
        assert result.image_optimization.max_dim == 1500
        assert result.image_optimization.quality == 85
        assert result.image_optimization.strip_exif is True

    def test_no_image_optimization(self, tmp_path: Path) -> None:
        config = {"default_country": "CY"}
        (tmp_path / "_config.yaml").write_text(yaml.dump(config))
        result = load_inbox_config(tmp_path)
        assert result is not None
        assert result.image_optimization is None

    def test_partial_fields(self, tmp_path: Path) -> None:
        config = {"default_currency": "USD"}
        (tmp_path / "_config.yaml").write_text(yaml.dump(config))
        result = load_inbox_config(tmp_path)
        assert result is not None
        assert result.default_currency == "USD"
        assert result.default_country == ""
        assert result.default_language == ""


class TestLoadVendorConfig:
    """Tests for load_vendor_config()."""

    def test_loads_full_config(self, tmp_path: Path) -> None:
        vendor = {
            "trade_name": "FreSko",
            "legal_name": "FRESKO BUTANOLO LTD",
            "country": "CY",
            "vat_number": "HE123456",
            "phone": "+357-22-123456",
            "website": "https://fresko.com.cy",
            "default_currency": "EUR",
            "loyalty_number": "FRES-001",
            "account_number": "ACC-999",
            "notes": "Main supermarket",
            "locations": [
                {
                    "address": "123 Ledra Street, Nicosia",
                    "map_url": "https://maps.google.com/?q=...",
                },
                {
                    "address": "45 Makarios Ave, Limassol",
                    "map_url": "https://maps.google.com/?q=...",
                },
            ],
        }
        (tmp_path / "_vendor.yaml").write_text(yaml.dump(vendor))
        result = load_vendor_config(tmp_path)
        assert result is not None
        assert result.trade_name == "FreSko"
        assert result.legal_name == "FRESKO BUTANOLO LTD"
        assert result.country == "CY"
        assert result.vat_number == "HE123456"
        assert result.phone == "+357-22-123456"
        assert result.website == "https://fresko.com.cy"
        assert result.default_currency == "EUR"
        assert result.loyalty_number == "FRES-001"
        assert result.account_number == "ACC-999"
        assert result.notes == "Main supermarket"
        assert len(result.locations) == 2
        assert result.locations[0].address == "123 Ledra Street, Nicosia"
        assert result.locations[1].address == "45 Makarios Ave, Limassol"

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        result = load_vendor_config(tmp_path)
        assert result is None

    def test_handles_missing_optional_fields(self, tmp_path: Path) -> None:
        vendor = {"trade_name": "Test Shop"}
        (tmp_path / "_vendor.yaml").write_text(yaml.dump(vendor))
        result = load_vendor_config(tmp_path)
        assert result is not None
        assert result.trade_name == "Test Shop"
        assert result.legal_name == ""
        assert result.country == ""
        assert result.vat_number == ""
        assert result.locations == []
        assert result.phone == ""
        assert result.website == ""
        assert result.default_currency == ""
        assert result.loyalty_number == ""
        assert result.account_number == ""
        assert result.notes == ""

    def test_handles_malformed_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "_vendor.yaml").write_text(": : invalid [yaml\n\t\x00")
        result = load_vendor_config(tmp_path)
        assert result is None

    def test_handles_empty_yaml(self, tmp_path: Path) -> None:
        (tmp_path / "_vendor.yaml").write_text("")
        result = load_vendor_config(tmp_path)
        assert result is not None
        assert result.trade_name == ""

    def test_locations_ignores_non_dict_entries(self, tmp_path: Path) -> None:
        vendor = {
            "trade_name": "Test",
            "locations": ["string_entry", 42, {"address": "Valid"}],
        }
        (tmp_path / "_vendor.yaml").write_text(yaml.dump(vendor))
        result = load_vendor_config(tmp_path)
        assert result is not None
        assert len(result.locations) == 1
        assert result.locations[0].address == "Valid"

    def test_locations_none_value(self, tmp_path: Path) -> None:
        vendor = {"trade_name": "Test", "locations": None}
        (tmp_path / "_vendor.yaml").write_text(yaml.dump(vendor))
        result = load_vendor_config(tmp_path)
        assert result is not None
        assert result.locations == []


class TestLoadCountryConfig:
    """Tests for load_country_config()."""

    def test_loads_country_config(self, tmp_path: Path) -> None:
        country_dir = tmp_path / "GR"
        country_dir.mkdir()
        config = {
            "default_country": "GR",
            "default_currency": "EUR",
            "default_language": "el",
        }
        (country_dir / "_config.yaml").write_text(yaml.dump(config))
        result = load_country_config(country_dir)
        assert result is not None
        assert result.default_country == "GR"
        assert result.default_language == "el"

    def test_returns_none_when_missing(self, tmp_path: Path) -> None:
        country_dir = tmp_path / "GR"
        country_dir.mkdir()
        result = load_country_config(country_dir)
        assert result is None


class TestMergeConfigs:
    """Tests for merge_configs()."""

    def test_country_overrides_inbox(self) -> None:
        inbox = InboxConfig(
            default_country="CY",
            default_currency="EUR",
            default_language="el",
        )
        country = InboxConfig(
            default_country="GR",
            default_currency="EUR",
            default_language="el",
        )
        merged = merge_configs(inbox, country)
        assert merged.default_country == "GR"

    def test_country_overrides_currency(self) -> None:
        inbox = InboxConfig(default_currency="EUR")
        country = InboxConfig(default_currency="GBP")
        merged = merge_configs(inbox, country)
        assert merged.default_currency == "GBP"

    def test_country_overrides_language(self) -> None:
        inbox = InboxConfig(default_language="el")
        country = InboxConfig(default_language="de")
        merged = merge_configs(inbox, country)
        assert merged.default_language == "de"

    def test_country_overrides_image_optimization(self) -> None:
        inbox_img = ImageOptConfig(enabled=True, max_dim=1500)
        country_img = ImageOptConfig(enabled=False, max_dim=1000)
        inbox = InboxConfig(image_optimization=inbox_img)
        country = InboxConfig(image_optimization=country_img)
        merged = merge_configs(inbox, country)
        assert merged.image_optimization is not None
        assert merged.image_optimization.max_dim == 1000
        assert merged.image_optimization.enabled is False

    def test_none_country_returns_inbox(self) -> None:
        inbox = InboxConfig(
            default_country="CY",
            default_currency="EUR",
            default_language="el",
        )
        merged = merge_configs(inbox, None)
        assert merged.default_country == "CY"
        assert merged.default_currency == "EUR"
        assert merged.default_language == "el"

    def test_none_inbox_returns_defaults(self) -> None:
        merged = merge_configs(None, None)
        assert merged.default_country == "CY"
        assert merged.default_currency == "EUR"
        assert merged.default_language == "el"

    def test_none_inbox_with_country(self) -> None:
        country = InboxConfig(
            default_country="DE",
            default_currency="EUR",
            default_language="de",
        )
        merged = merge_configs(None, country)
        assert merged.default_country == "DE"
        assert merged.default_language == "de"

    def test_country_empty_field_falls_back_to_inbox(self) -> None:
        """Empty string in country config should fall back to inbox value."""
        inbox = InboxConfig(default_country="CY", default_language="el")
        country = InboxConfig(default_country="", default_language="de")
        merged = merge_configs(inbox, country)
        # Empty string is falsy, so inbox default should win
        assert merged.default_country == "CY"
        assert merged.default_language == "de"

    def test_inbox_image_opt_preserved_when_country_has_none(self) -> None:
        img = ImageOptConfig(enabled=True, max_dim=1200)
        inbox = InboxConfig(image_optimization=img)
        country = InboxConfig()
        merged = merge_configs(inbox, country)
        assert merged.image_optimization is not None
        assert merged.image_optimization.max_dim == 1200


class TestDataclassDefaults:
    """Verify dataclass default values are correct."""

    def test_inbox_config_defaults(self) -> None:
        cfg = InboxConfig()
        assert cfg.default_country == "CY"
        assert cfg.default_currency == "EUR"
        assert cfg.default_language == "el"
        assert cfg.image_optimization is None

    def test_image_opt_config_defaults(self) -> None:
        cfg = ImageOptConfig()
        assert cfg.enabled is False
        assert cfg.max_dim == 1500
        assert cfg.quality == 85
        assert cfg.strip_exif is True

    def test_vendor_config_defaults(self) -> None:
        cfg = VendorConfig()
        assert cfg.trade_name == ""
        assert cfg.legal_name == ""
        assert cfg.locations == []
        assert cfg.phone == ""

    def test_vendor_location_defaults(self) -> None:
        loc = VendorLocation()
        assert loc.address == ""
        assert loc.map_url == ""
