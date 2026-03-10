"""Tests for alibi.predictions.category — category inference predictor."""

from __future__ import annotations

import os
from datetime import date
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.predictions.category import (
    MODEL_NAME,
    TABLE_NAME,
    _MIN_EXAMPLES_PER_CATEGORY,
    _MIN_TOTAL_EXAMPLES,
    classify,
    classify_batch,
    prepare_category_training_data,
)

# Pandas-dependent tests are conditionally skipped
_has_pandas = False
try:
    import pandas  # noqa: F401

    _has_pandas = True
except ImportError:
    pass

_skip_no_pandas = pytest.mark.skipif(not _has_pandas, reason="pandas not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.wait_model_ready.return_value = "complete"
    return client


def _make_fact(fact_id: str, vendor: str) -> dict[str, Any]:
    return {
        "id": fact_id,
        "vendor": vendor,
        "event_date": date(2025, 1, 15),
    }


def _make_item(
    name: str,
    category: str | None,
    price: float = 5.0,
    name_normalized: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "name_normalized": name_normalized,
        "category": category,
        "total_price": price,
        "unit_price": price,
    }


def _make_mock_df(rows: int = 60, empty: bool = False) -> MagicMock:
    df = MagicMock()
    df.empty = empty
    df.__len__ = MagicMock(return_value=0 if empty else rows)
    return df


# ---------------------------------------------------------------------------
# prepare_category_training_data() — pandas-dependent
# ---------------------------------------------------------------------------


@_skip_no_pandas
class TestPrepareCategoryTrainingData:
    def test_empty_facts_returns_empty_dataframe(self, db: Any) -> None:
        with patch("alibi.predictions.category.v2_store") as mock_store:
            mock_store.list_facts.return_value = []
            df = prepare_category_training_data(db)

        assert df.empty
        assert list(df.columns) == ["vendor_name", "item_name", "amount", "category"]

    def test_items_without_category_are_skipped(self, db: Any) -> None:
        facts = [_make_fact("f1", "Lidl")]
        items = [
            _make_item("bread", None),
            _make_item("milk", ""),
        ]

        with patch("alibi.predictions.category.v2_store") as mock_store:
            mock_store.list_facts.return_value = facts
            mock_store.get_fact_items.return_value = items

            df = prepare_category_training_data(db)

        assert df.empty

    def test_filters_categories_with_few_examples(self, db: Any) -> None:
        """Categories with < _MIN_EXAMPLES_PER_CATEGORY items are removed."""
        # 4 items in "rare" (below threshold of 5), 6 items in "common"
        facts = [_make_fact(f"f{i}", "Store") for i in range(10)]

        def get_items(db_arg: Any, fact_id: str) -> list[dict[str, Any]]:
            fid = int(fact_id[1:])
            if fid < 4:
                return [_make_item(f"item{fid}", "rare")]
            return [_make_item(f"item{fid}", "common")]

        with patch("alibi.predictions.category.v2_store") as mock_store:
            mock_store.list_facts.return_value = facts
            mock_store.get_fact_items.side_effect = get_items

            df = prepare_category_training_data(db)

        assert "rare" not in df["category"].values
        assert "common" in df["category"].values

    def test_name_normalized_preferred_over_name(self, db: Any) -> None:
        """name_normalized is used for item_name when present."""
        facts = [_make_fact(f"f{i}", "Store") for i in range(6)]

        with patch("alibi.predictions.category.v2_store") as mock_store:
            mock_store.list_facts.return_value = facts
            mock_store.get_fact_items.return_value = [
                _make_item("Whole Wheat Bread 500g", "food", name_normalized="bread")
            ]

            df = prepare_category_training_data(db)

        if not df.empty:
            assert "bread" in df["item_name"].values

    def test_uses_total_price_for_amount(self, db: Any) -> None:
        facts = [_make_fact(f"f{i}", "Store") for i in range(6)]

        with patch("alibi.predictions.category.v2_store") as mock_store:
            mock_store.list_facts.return_value = facts
            mock_store.get_fact_items.return_value = [
                _make_item("bread", "food", price=3.50)
            ]

            df = prepare_category_training_data(db)

        if not df.empty:
            assert abs(df["amount"].iloc[0] - 3.50) < 0.01


# ---------------------------------------------------------------------------
# train_category_classifier() — uses mocked DataFrame so no pandas needed
# ---------------------------------------------------------------------------


class TestTrainCategoryClassifier:
    def test_train_uploads_and_creates_model(
        self, db: Any, mock_client: MagicMock
    ) -> None:
        from alibi.predictions.category import train_category_classifier

        fake_df = _make_mock_df(rows=60)

        with patch(
            "alibi.predictions.category.prepare_category_training_data",
            return_value=fake_df,
        ):
            result = train_category_classifier(mock_client, db)

        mock_client.upload_dataframe.assert_called_once_with(TABLE_NAME, fake_df)
        mock_client.create_model.assert_called_once()
        create_kwargs = mock_client.create_model.call_args[1]
        assert create_kwargs["name"] == MODEL_NAME
        assert create_kwargs["predict"] == "category"
        mock_client.wait_model_ready.assert_called_once_with(MODEL_NAME)
        assert result == "complete"

    def test_train_insufficient_data_raises_value_error(
        self, db: Any, mock_client: MagicMock
    ) -> None:
        from alibi.predictions.category import train_category_classifier

        # Only 10 rows — below _MIN_TOTAL_EXAMPLES (50)
        small_df = _make_mock_df(rows=10)
        with patch(
            "alibi.predictions.category.prepare_category_training_data",
            return_value=small_df,
        ):
            with pytest.raises(ValueError, match="Insufficient training data"):
                train_category_classifier(mock_client, db)

    def test_train_empty_data_raises_value_error(
        self, db: Any, mock_client: MagicMock
    ) -> None:
        from alibi.predictions.category import train_category_classifier

        empty_df = _make_mock_df(empty=True)
        with patch(
            "alibi.predictions.category.prepare_category_training_data",
            return_value=empty_df,
        ):
            with pytest.raises(ValueError, match="Insufficient training data"):
                train_category_classifier(mock_client, db)


# ---------------------------------------------------------------------------
# classify() — no pandas dependency
# ---------------------------------------------------------------------------


class TestClassify:
    def test_classify_single_item(self, mock_client: MagicMock) -> None:
        mock_client.predict.return_value = [
            {"category": "food", "category_confidence": 0.92}
        ]

        result = classify(mock_client, "Lidl", "bread", 1.5)

        assert result["category"] == "food"
        assert abs(result["category_confidence"] - 0.92) < 0.001
        assert result["vendor_name"] == "Lidl"
        assert result["item_name"] == "bread"
        assert result["amount"] == 1.5

    def test_classify_empty_result_returns_none_category(
        self, mock_client: MagicMock
    ) -> None:
        mock_client.predict.return_value = []

        result = classify(mock_client, "Store", "unknown item", 9.99)

        assert result["category"] is None
        assert result["category_confidence"] == 0.0

    def test_classify_forwards_correct_model_name(self, mock_client: MagicMock) -> None:
        mock_client.predict.return_value = [
            {"category": "x", "category_confidence": 0.5}
        ]

        classify(mock_client, "V", "I", 1.0)

        mock_client.predict.assert_called_once()
        model_arg = mock_client.predict.call_args[0][0]
        assert model_arg == MODEL_NAME

    def test_classify_uses_first_result_only(self, mock_client: MagicMock) -> None:
        mock_client.predict.return_value = [
            {"category": "first", "category_confidence": 0.9},
            {"category": "second", "category_confidence": 0.7},
        ]

        result = classify(mock_client, "V", "I", 1.0)

        assert result["category"] == "first"

    def test_classify_passes_correct_input_fields(self, mock_client: MagicMock) -> None:
        mock_client.predict.return_value = [
            {"category": "food", "category_confidence": 0.8}
        ]

        classify(mock_client, "Lidl", "milk", 0.99)

        data_arg = mock_client.predict.call_args[0][1]
        assert data_arg["vendor_name"] == "Lidl"
        assert data_arg["item_name"] == "milk"
        assert data_arg["amount"] == 0.99


# ---------------------------------------------------------------------------
# classify_batch() — no pandas dependency
# ---------------------------------------------------------------------------


class TestClassifyBatch:
    def test_classify_batch_empty_returns_empty(self, mock_client: MagicMock) -> None:
        result = classify_batch(mock_client, [])
        assert result == []
        mock_client.predict.assert_not_called()

    def test_classify_batch_multiple_items(self, mock_client: MagicMock) -> None:
        input_items = [
            {"vendor_name": "Lidl", "item_name": "bread", "amount": 1.5},
            {"vendor_name": "Aldi", "item_name": "milk", "amount": 0.9},
        ]
        mock_client.predict.return_value = [
            {
                "vendor_name": "Lidl",
                "item_name": "bread",
                "amount": 1.5,
                "category": "food",
                "category_confidence": 0.9,
            },
            {
                "vendor_name": "Aldi",
                "item_name": "milk",
                "amount": 0.9,
                "category": "dairy",
                "category_confidence": 0.85,
            },
        ]

        result = classify_batch(mock_client, input_items)

        assert len(result) == 2
        assert result[0]["category"] == "food"
        assert result[1]["category"] == "dairy"

    def test_classify_batch_uses_model_name(self, mock_client: MagicMock) -> None:
        mock_client.predict.return_value = [
            {"category": "x", "category_confidence": 0.5}
        ]

        classify_batch(
            mock_client,
            [{"vendor_name": "V", "item_name": "I", "amount": 1.0}],
        )

        model_arg = mock_client.predict.call_args[0][0]
        assert model_arg == MODEL_NAME

    def test_classify_batch_handles_missing_confidence(
        self, mock_client: MagicMock
    ) -> None:
        mock_client.predict.return_value = [{"category": "food"}]

        result = classify_batch(
            mock_client, [{"vendor_name": "V", "item_name": "I", "amount": 1.0}]
        )

        assert result[0]["category_confidence"] == 0.0

    def test_classify_batch_passes_all_items_at_once(
        self, mock_client: MagicMock
    ) -> None:
        """classify_batch sends all items in a single predict() call."""
        items = [
            {"vendor_name": f"V{i}", "item_name": f"item{i}", "amount": float(i)}
            for i in range(5)
        ]
        mock_client.predict.return_value = [
            {"category": "x", "category_confidence": 0.5}
        ] * 5

        classify_batch(mock_client, items)

        assert mock_client.predict.call_count == 1
        passed_items = mock_client.predict.call_args[0][1]
        assert passed_items == items
