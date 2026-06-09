"""Regression tests: local enrichment CLI passes refresh the item_stars mirror.

The five decoupled enrichment passes (categorize, comparable-names,
comparable-prices, units, attributes) write straight to ``fact_items``. They
previously only printed "Run `lt items rebuild`", leaving the materialised
``item_stars`` analytics surface stale until a human rebuilt it. Each command
must now refresh the affected facts itself via
``refresh_item_stars_for_items`` (only for the items it actually changed).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from alibi.commands.enrich import enrich
from alibi.enrichment.attributes import AttributeResult
from alibi.enrichment.categorize import CategoryResult
from alibi.enrichment.comparable_names import ComparableNameResult
from alibi.enrichment.comparable_prices import ComparablePriceResult
from alibi.enrichment.units import UnitResult

# (command args, dotted path of the pass fn, results, expected refreshed ids)
_CASES = [
    pytest.param(
        ["categorize"],
        "alibi.enrichment.categorize.enrich_pending_categories",
        [
            CategoryResult("i1", "food > dairy", "dairy", success=True),
            CategoryResult("i2", None, None, success=False),
        ],
        ["i1"],
        id="categorize",
    ),
    pytest.param(
        ["comparable-names"],
        "alibi.enrichment.comparable_names.enrich_pending_comparable_names",
        [
            ComparableNameResult("i1", "milk", success=True),
            ComparableNameResult("i2", None, success=False),
        ],
        ["i1"],
        id="comparable-names",
    ),
    pytest.param(
        ["comparable-prices"],
        "alibi.enrichment.comparable_prices.recompute_pending_comparable_prices",
        [
            ComparablePriceResult("i1", "kg", "5.00", changed=True),
            ComparablePriceResult("i2", None, None, changed=False),
        ],
        ["i1"],
        id="comparable-prices",
    ),
    pytest.param(
        ["units"],
        "alibi.enrichment.units.enrich_pending_units",
        [
            UnitResult("i1", "g", "450", success=True),
            UnitResult("i2", None, None, success=False),
        ],
        ["i1"],
        id="units",
    ),
    pytest.param(
        ["attributes"],
        "alibi.enrichment.attributes.enrich_pending_attributes",
        [
            AttributeResult("i1", {"organic": True}, None, changed=True),
            AttributeResult("i2", {}, None, changed=False),
        ],
        ["i1"],
        id="attributes",
    ),
]


@pytest.mark.parametrize("args, pass_path, results, expected_ids", _CASES)
def test_pass_refreshes_item_stars_for_changed_items(
    args: list[str],
    pass_path: str,
    results: list,
    expected_ids: list[str],
) -> None:
    fake_db = MagicMock()
    fake_db.is_initialized.return_value = True

    with (
        patch("alibi.commands.enrich.get_db", return_value=fake_db),
        patch(pass_path, return_value=results),
        patch(
            "alibi.services.refresh_item_stars_for_items", return_value=1
        ) as mock_refresh,
    ):
        result = CliRunner().invoke(enrich, args)

    assert result.exit_code == 0, result.output
    mock_refresh.assert_called_once_with(fake_db, expected_ids)


def test_no_results_does_not_refresh() -> None:
    """An empty pass must not call the refresh helper at all."""
    fake_db = MagicMock()
    fake_db.is_initialized.return_value = True

    with (
        patch("alibi.commands.enrich.get_db", return_value=fake_db),
        patch("alibi.enrichment.units.enrich_pending_units", return_value=[]),
        patch(
            "alibi.services.refresh_item_stars_for_items", return_value=0
        ) as mock_refresh,
    ):
        result = CliRunner().invoke(enrich, ["units"])

    assert result.exit_code == 0, result.output
    mock_refresh.assert_not_called()
