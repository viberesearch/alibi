"""Tests for alibi.predictions.client — MindsDB REST API client."""

from __future__ import annotations

import os
import time
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

os.environ["ALIBI_TESTING"] = "1"

from alibi.predictions.client import MindsDBClient, MindsDBError, QueryResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    status_code: int = 200,
    json_body: dict[str, Any] | None = None,
) -> MagicMock:
    """Build a fake httpx Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}

    if status_code >= 400:
        import httpx

        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=MagicMock(),
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------


class TestQueryResult:
    def test_to_dicts_empty(self) -> None:
        qr = QueryResult(column_names=["a", "b"], data=[])
        assert qr.to_dicts() == []

    def test_to_dicts_single_row(self) -> None:
        qr = QueryResult(
            column_names=["name", "status"], data=[["my_model", "complete"]]
        )
        result = qr.to_dicts()
        assert result == [{"name": "my_model", "status": "complete"}]

    def test_to_dicts_multiple_rows(self) -> None:
        qr = QueryResult(
            column_names=["x", "y"],
            data=[[1, 2], [3, 4]],
        )
        result = qr.to_dicts()
        assert result == [{"x": 1, "y": 2}, {"x": 3, "y": 4}]

    def test_default_type_is_table(self) -> None:
        qr = QueryResult()
        assert qr.type == "table"


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_success(self) -> None:
        client = MindsDBClient("http://test:47334")
        resp = _make_response(
            200,
            {"column_names": ["id", "val"], "data": [["a", 1]], "type": "table"},
        )
        with patch.object(client._client, "post", return_value=resp):
            result = client.query("SELECT 1")

        assert result.column_names == ["id", "val"]
        assert result.data == [["a", 1]]
        assert result.type == "table"

    def test_query_http_error_raises_mindsdb_error(self) -> None:
        import httpx

        client = MindsDBClient("http://test:47334")
        with patch.object(
            client._client,
            "post",
            side_effect=httpx.HTTPError("connection refused"),
        ):
            with pytest.raises(MindsDBError, match="HTTP error"):
                client.query("SELECT 1")

    def test_query_http_status_error_raises_mindsdb_error(self) -> None:
        client = MindsDBClient("http://test:47334")
        resp = _make_response(500)
        with patch.object(client._client, "post", return_value=resp):
            with pytest.raises(MindsDBError):
                client.query("SELECT 1")

    def test_query_body_error_raises_mindsdb_error(self) -> None:
        client = MindsDBClient("http://test:47334")
        resp = _make_response(200, {"error": "syntax error near SELECT"})
        with patch.object(client._client, "post", return_value=resp):
            with pytest.raises(MindsDBError, match="syntax error"):
                client.query("BAD SQL")

    def test_query_missing_fields_use_defaults(self) -> None:
        client = MindsDBClient("http://test:47334")
        resp = _make_response(200, {})
        with patch.object(client._client, "post", return_value=resp):
            result = client.query("SELECT 1")

        assert result.column_names == []
        assert result.data == []
        assert result.type == "table"


# ---------------------------------------------------------------------------
# is_healthy()
# ---------------------------------------------------------------------------


class TestIsHealthy:
    def test_is_healthy_true(self) -> None:
        client = MindsDBClient("http://test:47334")
        resp = MagicMock()
        resp.status_code = 200
        with patch.object(client._client, "get", return_value=resp):
            assert client.is_healthy() is True

    def test_is_healthy_false_non_200(self) -> None:
        client = MindsDBClient("http://test:47334")
        resp = MagicMock()
        resp.status_code = 503
        with patch.object(client._client, "get", return_value=resp):
            assert client.is_healthy() is False

    def test_is_healthy_false_http_error(self) -> None:
        import httpx

        client = MindsDBClient("http://test:47334")
        with patch.object(
            client._client, "get", side_effect=httpx.HTTPError("timeout")
        ):
            assert client.is_healthy() is False


# ---------------------------------------------------------------------------
# upload_dataframe()
# ---------------------------------------------------------------------------


class TestUploadDataframe:
    def _make_mock_df(self) -> MagicMock:
        """Return a mock object that behaves like a minimal pandas DataFrame."""
        df = MagicMock()
        df.to_csv = MagicMock(side_effect=lambda buf, **kw: buf.write("a,b\n1,2\n"))
        df.__len__ = MagicMock(return_value=2)
        return df

    def test_upload_dataframe_success(self) -> None:
        client = MindsDBClient("http://test:47334")
        df = self._make_mock_df()

        drop_resp = _make_response(200, {"column_names": [], "data": []})
        put_resp = MagicMock()
        put_resp.status_code = 200
        put_resp.raise_for_status.return_value = None

        with (
            patch.object(client._client, "post", return_value=drop_resp),
            patch.object(client._client, "put", return_value=put_resp) as mock_put,
        ):
            client.upload_dataframe("my_table", df)

        mock_put.assert_called_once()
        call_kwargs = mock_put.call_args
        assert "my_table" in call_kwargs[0][0]

    def test_upload_dataframe_put_error_raises_mindsdb_error(self) -> None:
        import httpx

        client = MindsDBClient("http://test:47334")
        df = self._make_mock_df()

        drop_resp = _make_response(200, {"column_names": [], "data": []})

        with (
            patch.object(client._client, "post", return_value=drop_resp),
            patch.object(
                client._client, "put", side_effect=httpx.HTTPError("upload failed")
            ),
        ):
            with pytest.raises(MindsDBError, match="Upload failed"):
                client.upload_dataframe("my_table", df)


# ---------------------------------------------------------------------------
# create_model()
# ---------------------------------------------------------------------------


class TestCreateModel:
    def test_create_model_basic(self) -> None:
        client = MindsDBClient("http://test:47334")
        sql_calls: list[str] = []

        def fake_query(sql: str) -> QueryResult:
            sql_calls.append(sql)
            return QueryResult()

        with patch.object(client, "query", side_effect=fake_query):
            client.create_model(
                name="my_model",
                predict="amount",
                from_table="files.`my_table`",
            )

        # Two calls: DROP + CREATE
        assert len(sql_calls) == 2
        create_sql = sql_calls[1]
        assert "CREATE MODEL" in create_sql
        assert "my_model" in create_sql
        assert "PREDICT" in create_sql
        assert "amount" in create_sql

    def test_create_model_timeseries(self) -> None:
        client = MindsDBClient("http://test:47334")
        sql_calls: list[str] = []

        def fake_query(sql: str) -> QueryResult:
            sql_calls.append(sql)
            return QueryResult()

        with patch.object(client, "query", side_effect=fake_query):
            client.create_model(
                name="ts_model",
                predict="amount",
                from_table="files.`data`",
                timeseries_options={
                    "order": "date",
                    "group": "category",
                    "window": 6,
                    "horizon": 3,
                },
            )

        create_sql = sql_calls[1]
        assert "ORDER BY" in create_sql
        assert "GROUP BY" in create_sql
        assert "WINDOW 6" in create_sql
        assert "HORIZON 3" in create_sql

    def test_create_model_with_using(self) -> None:
        client = MindsDBClient("http://test:47334")
        sql_calls: list[str] = []

        def fake_query(sql: str) -> QueryResult:
            sql_calls.append(sql)
            return QueryResult()

        with patch.object(client, "query", side_effect=fake_query):
            client.create_model(
                name="my_model",
                predict="label",
                from_table="files.`data`",
                using={"engine": "lightwood"},
            )

        create_sql = sql_calls[1]
        assert "USING" in create_sql
        assert "lightwood" in create_sql

    def test_create_model_drop_error_is_ignored(self) -> None:
        """MindsDBError during DROP is swallowed so create proceeds."""
        client = MindsDBClient("http://test:47334")
        call_count = 0

        def fake_query(sql: str) -> QueryResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise MindsDBError("cannot drop")
            return QueryResult()

        with patch.object(client, "query", side_effect=fake_query):
            client.create_model(name="my_model", predict="x", from_table="files.`t`")

        assert call_count == 2


# ---------------------------------------------------------------------------
# get_model_status()
# ---------------------------------------------------------------------------


class TestGetModelStatus:
    def test_get_model_status_returns_string(self) -> None:
        client = MindsDBClient("http://test:47334")
        qr = QueryResult(column_names=["status"], data=[["complete"]])
        with patch.object(client, "query", return_value=qr):
            status = client.get_model_status("my_model")
        assert status == "complete"

    def test_get_model_status_not_found_raises(self) -> None:
        client = MindsDBClient("http://test:47334")
        qr = QueryResult(column_names=["status"], data=[])
        with patch.object(client, "query", return_value=qr):
            with pytest.raises(MindsDBError, match="Model not found"):
                client.get_model_status("missing_model")


# ---------------------------------------------------------------------------
# wait_model_ready()
# ---------------------------------------------------------------------------


class TestWaitModelReady:
    def test_wait_model_ready_immediate(self) -> None:
        client = MindsDBClient("http://test:47334")
        with patch.object(client, "get_model_status", return_value="complete"):
            status = client.wait_model_ready(
                "my_model", timeout=10.0, poll_interval=0.01
            )
        assert status == "complete"

    def test_wait_model_ready_polls_until_complete(self) -> None:
        client = MindsDBClient("http://test:47334")
        statuses = ["training", "training", "complete"]
        idx = 0

        def fake_status(name: str) -> str:
            nonlocal idx
            s = statuses[idx]
            idx += 1
            return s

        with (
            patch.object(client, "get_model_status", side_effect=fake_status),
            patch("time.sleep"),
        ):
            status = client.wait_model_ready(
                "my_model", timeout=30.0, poll_interval=0.01
            )

        assert status == "complete"

    def test_wait_model_ready_error_state_raises(self) -> None:
        client = MindsDBClient("http://test:47334")
        with patch.object(client, "get_model_status", return_value="error"):
            with pytest.raises(MindsDBError, match="training failed"):
                client.wait_model_ready("my_model", timeout=10.0, poll_interval=0.01)

    def test_wait_model_ready_timeout_raises(self) -> None:
        client = MindsDBClient("http://test:47334")

        # Always return "training" so it never finishes
        with (
            patch.object(client, "get_model_status", return_value="training"),
            patch("time.sleep"),
            patch("time.monotonic", side_effect=[0.0, 0.0, 9999.0]),
        ):
            with pytest.raises(MindsDBError, match="timed out"):
                client.wait_model_ready("my_model", timeout=1.0, poll_interval=0.01)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------


class TestPredict:
    def test_predict_single_dict(self) -> None:
        client = MindsDBClient("http://test:47334")
        fake_result = [{"category": "food", "category_confidence": 0.9}]
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = fake_result

        with patch.object(client._client, "post", return_value=resp):
            result = client.predict("my_model", {"vendor_name": "Lidl", "amount": 5.0})

        assert result == fake_result

    def test_predict_batch(self) -> None:
        client = MindsDBClient("http://test:47334")
        items = [
            {"vendor_name": "Lidl", "item_name": "bread", "amount": 1.5},
            {"vendor_name": "Aldi", "item_name": "milk", "amount": 0.9},
        ]
        fake_result = [
            {"category": "food", "category_confidence": 0.95},
            {"category": "dairy", "category_confidence": 0.88},
        ]
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        resp.json.return_value = fake_result

        with patch.object(client._client, "post", return_value=resp) as mock_post:
            result = client.predict("my_model", items)

        assert result == fake_result
        # Verify list was passed directly (not wrapped again)
        call_json = mock_post.call_args[1]["json"]
        assert call_json["data"] == items

    def test_predict_http_error_raises_mindsdb_error(self) -> None:
        import httpx

        client = MindsDBClient("http://test:47334")
        with patch.object(
            client._client, "post", side_effect=httpx.HTTPError("timeout")
        ):
            with pytest.raises(MindsDBError, match="Prediction failed"):
                client.predict("my_model", {"x": 1})


# ---------------------------------------------------------------------------
# predict_sql()
# ---------------------------------------------------------------------------


class TestPredictSql:
    def test_predict_sql_returns_dicts(self) -> None:
        client = MindsDBClient("http://test:47334")
        qr = QueryResult(
            column_names=["forecast_date", "amount"],
            data=[["2025-04-01", 150.0]],
        )
        with patch.object(client, "query", return_value=qr):
            result = client.predict_sql("SELECT ...")

        assert result == [{"forecast_date": "2025-04-01", "amount": 150.0}]


# ---------------------------------------------------------------------------
# drop_model() / list_models()
# ---------------------------------------------------------------------------


class TestDropAndListModels:
    def test_drop_model_calls_query(self) -> None:
        client = MindsDBClient("http://test:47334")
        with patch.object(client, "query", return_value=QueryResult()) as mock_q:
            client.drop_model("my_model")

        sql = mock_q.call_args[0][0]
        assert "DROP MODEL" in sql
        assert "my_model" in sql

    def test_list_models_returns_dicts(self) -> None:
        client = MindsDBClient("http://test:47334")
        qr = QueryResult(
            column_names=["name", "status", "predict", "engine"],
            data=[["spending_model", "complete", "amount", "lightwood"]],
        )
        with patch.object(client, "query", return_value=qr):
            result = client.list_models()

        assert result == [
            {
                "name": "spending_model",
                "status": "complete",
                "predict": "amount",
                "engine": "lightwood",
            }
        ]


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_context_manager_closes_client(self) -> None:
        client = MindsDBClient("http://test:47334")
        with patch.object(client, "close") as mock_close:
            with client:
                pass
        mock_close.assert_called_once()

    def test_context_manager_returns_self(self) -> None:
        client = MindsDBClient("http://test:47334")
        with patch.object(client, "close"):
            with client as c:
                assert c is client
