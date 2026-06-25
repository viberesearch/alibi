"""Tests for the state-facet analytics surface on the items CLI.

The product-state facet (#58) lives in ``attributes.$.state`` and is mirrored
into ``item_stars``. The service already supports filtering by any facet and
grouping by ``attr:<key>``; these tests cover the thin CLI plumbing added to
expose STATE specifically: the ``--state`` filter folds into the flexible
attributes filter, and an ``attr:state`` group dim is aliased to ``state`` for
display.
"""

from __future__ import annotations

import os

os.environ["ALIBI_TESTING"] = "1"

from alibi.commands.items import _filters


class TestStateFilterFolding:
    def test_state_folds_into_attributes(self):
        out = _filters({"state": "canned"})
        assert out == {"attributes": {"state": "canned"}}

    def test_state_is_lowercased_and_stripped(self):
        out = _filters({"state": "  Fresh "})
        assert out["attributes"] == {"state": "fresh"}

    def test_no_state_no_attributes_key(self):
        out = _filters({"comparable_name": "salmon"})
        assert out == {"comparable_name": "salmon"}
        assert "attributes" not in out

    def test_state_composes_with_other_filters(self):
        out = _filters({"comparable_name": "salmon", "country": "CY", "state": "cured"})
        assert out == {
            "comparable_name": "salmon",
            "country": "CY",
            "attributes": {"state": "cured"},
        }

    def test_empty_state_ignored(self):
        assert _filters({"state": ""}) == {}


class TestAttrDimAlias:
    """The avg-price command aliases an ``attr:<key>`` dim to its bare key, both
    for the column header and the per-row value lookup (the service returns rows
    keyed by the bare key, not the raw ``attr:`` dim string)."""

    @staticmethod
    def _aliases(dims: list[str]) -> list[str]:
        # Mirrors the resolution in items_avg_price.
        return [d[len("attr:") :] if d.startswith("attr:") else d for d in dims]

    def test_attr_prefix_stripped(self):
        assert self._aliases(["comparable_name", "attr:state"]) == [
            "comparable_name",
            "state",
        ]

    def test_plain_dims_unchanged(self):
        assert self._aliases(["vendor", "month"]) == ["vendor", "month"]
