"""MindsDB REST API client.

Communicates with a self-hosted MindsDB instance via its HTTP SQL API.
No SDK dependency — uses httpx for all requests.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0
_TRAINING_POLL_INTERVAL = 5.0
_TRAINING_TIMEOUT = 600.0


class MindsDBError(Exception):
    """Raised when MindsDB returns an error."""


@dataclass
class QueryResult:
    """Result of a MindsDB SQL query."""

    column_names: list[str] = field(default_factory=list)
    data: list[list[Any]] = field(default_factory=list)
    type: str = "table"

    def to_dicts(self) -> list[dict[str, Any]]:
        """Convert rows to list of dicts."""
        return [dict(zip(self.column_names, row)) for row in self.data]


class MindsDBClient:
    """Synchronous client for the MindsDB HTTP SQL API."""

    def __init__(self, url: str = "http://127.0.0.1:47334") -> None:
        self.url = url.rstrip("/")
        self._sql_endpoint = f"{self.url}/api/sql/query"
        self._client = httpx.Client(timeout=_DEFAULT_TIMEOUT)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "MindsDBClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @staticmethod
    def _validate_identifier(name: str) -> None:
        """Validate that a name is a safe SQL identifier."""
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(f"Invalid model name: {name}")

    # -- Low-level --------------------------------------------------------

    def query(self, sql: str) -> QueryResult:
        """Execute a SQL statement against MindsDB.

        Args:
            sql: Any valid MindsDB SQL statement.

        Returns:
            QueryResult with column_names and data rows.

        Raises:
            MindsDBError: On HTTP or MindsDB-level errors.
        """
        logger.debug("MindsDB SQL: %s", sql[:200])
        try:
            resp = self._client.post(self._sql_endpoint, json={"query": sql})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise MindsDBError(f"HTTP error: {exc}") from exc

        body = resp.json()
        if "error" in body:
            raise MindsDBError(body["error"])

        return QueryResult(
            column_names=body.get("column_names", []),
            data=body.get("data", []),
            type=body.get("type", "table"),
        )

    def is_healthy(self) -> bool:
        """Check if MindsDB is reachable."""
        try:
            resp = self._client.get(f"{self.url}/api/status")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    # -- File / table upload ----------------------------------------------

    def upload_dataframe(self, name: str, df: Any) -> None:
        """Upload a pandas DataFrame as a MindsDB files table.

        If a table with the same name exists, it is replaced.

        Args:
            name: Table name in the MindsDB ``files`` database.
            df: A pandas DataFrame.
        """
        self._validate_identifier(name)

        import io

        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode("utf-8")

        # Drop existing table silently
        try:
            self.query(f"DROP TABLE IF EXISTS files.`{name}`")
        except MindsDBError:
            pass

        try:
            resp = self._client.put(
                f"{self.url}/api/files/{name}",
                files={"file": (f"{name}.csv", csv_bytes, "text/csv")},
                timeout=120.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise MindsDBError(f"Upload failed: {exc}") from exc

        logger.info("Uploaded table files.%s (%d rows)", name, len(df))

    # -- Model lifecycle --------------------------------------------------

    def create_model(
        self,
        name: str,
        predict: str,
        from_table: str,
        select_sql: str | None = None,
        *,
        timeseries_options: dict[str, Any] | None = None,
        using: dict[str, Any] | None = None,
    ) -> None:
        """Create (train) a new MindsDB model.

        Args:
            name: Model name.
            predict: Target column to predict.
            from_table: Source table (e.g. ``files.spending_data``).
            select_sql: Optional SELECT override (defaults to ``SELECT * FROM <from_table>``).
            timeseries_options: Dict with order, group, window, horizon for time-series.
            using: Dict of USING clause parameters (engine, model_name, etc.).
        """
        self._validate_identifier(name)
        self._validate_identifier(predict)

        # Drop existing model
        try:
            self.query(f"DROP MODEL IF EXISTS mindsdb.`{name}`")
        except MindsDBError:
            pass

        select = select_sql or f"SELECT * FROM {from_table}"
        parts = [
            f"CREATE MODEL mindsdb.`{name}`",
            f"FROM files ({select})",
            f"PREDICT `{predict}`",
        ]

        if timeseries_options:
            if "order" in timeseries_options:
                parts.append(f"ORDER BY `{timeseries_options['order']}`")
            if "group" in timeseries_options:
                parts.append(f"GROUP BY `{timeseries_options['group']}`")
            if "window" in timeseries_options:
                parts.append(f"WINDOW {timeseries_options['window']}")
            if "horizon" in timeseries_options:
                parts.append(f"HORIZON {timeseries_options['horizon']}")

        if using:
            using_pairs = [f"{k} = '{v}'" for k, v in using.items()]
            parts.append(f"USING {', '.join(using_pairs)}")

        sql = "\n".join(parts)
        self.query(sql)
        logger.info("Created model mindsdb.%s (target: %s)", name, predict)

    def get_model_status(self, name: str) -> str:
        """Get training status of a model.

        Returns:
            One of: generating, training, complete, error.
        """
        self._validate_identifier(name)
        result = self.query(f"SELECT status FROM mindsdb.models WHERE name = '{name}'")
        if not result.data:
            raise MindsDBError(f"Model not found: {name}")
        return str(result.data[0][0])

    def wait_model_ready(
        self,
        name: str,
        timeout: float = _TRAINING_TIMEOUT,
        poll_interval: float = _TRAINING_POLL_INTERVAL,
    ) -> str:
        """Block until a model finishes training.

        Args:
            name: Model name.
            timeout: Max seconds to wait.
            poll_interval: Seconds between status checks.

        Returns:
            Final status string.

        Raises:
            MindsDBError: If model errors or timeout exceeded.
        """
        start = time.monotonic()
        while True:
            status = self.get_model_status(name)
            if status == "complete":
                logger.info("Model %s training complete", name)
                return status
            if status == "error":
                raise MindsDBError(f"Model {name} training failed")
            if time.monotonic() - start > timeout:
                raise MindsDBError(
                    f"Model {name} training timed out after {timeout}s "
                    f"(status: {status})"
                )
            time.sleep(poll_interval)

    def predict(
        self, model_name: str, data: dict[str, Any] | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Query predictions from a trained model.

        Args:
            model_name: Name of the trained model.
            data: Single dict or list of dicts with input features.

        Returns:
            List of prediction result dicts.
        """
        self._validate_identifier(model_name)
        if isinstance(data, dict):
            data = [data]

        try:
            resp = self._client.post(
                f"{self.url}/api/projects/mindsdb/models/{model_name}/predict",
                json={"data": data},
                timeout=60.0,
            )
            resp.raise_for_status()
            return resp.json()  # type: ignore[no-any-return]
        except httpx.HTTPError as exc:
            raise MindsDBError(f"Prediction failed: {exc}") from exc

    def predict_sql(self, sql: str) -> list[dict[str, Any]]:
        """Run a prediction via SQL SELECT.

        Useful for time-series predictions that require a JOIN.

        Args:
            sql: A SELECT ... FROM mindsdb.model_name ... query.

        Returns:
            List of result dicts.
        """
        result = self.query(sql)
        return result.to_dicts()

    def drop_model(self, name: str) -> None:
        """Delete a model."""
        self._validate_identifier(name)
        self.query(f"DROP MODEL IF EXISTS mindsdb.`{name}`")
        logger.info("Dropped model mindsdb.%s", name)

    def list_models(self) -> list[dict[str, Any]]:
        """List all models in the mindsdb project."""
        result = self.query("SELECT name, status, predict, engine FROM mindsdb.models")
        return result.to_dicts()
