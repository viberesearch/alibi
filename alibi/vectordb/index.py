"""LanceDB index management for semantic search."""

import json
import logging
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Callable

try:
    import lancedb  # type: ignore[import-untyped]
    import pyarrow as pa  # type: ignore[import-untyped]
except ImportError:
    lancedb = None  # type: ignore[assignment]
    pa = None  # type: ignore[assignment]

from alibi.config import get_config
from alibi.db.connection import DatabaseManager
from alibi.vectordb.embeddings import (
    EMBEDDING_DIM,
    EmbeddingError,
    get_embedding,
)

logger = logging.getLogger(__name__)

# Default vector storage path
DEFAULT_VECTOR_PATH = Path.home() / ".alibi" / "vectors"


class IndexType(str, Enum):
    """Type of indexed content."""

    TRANSACTION = "transaction"
    ARTIFACT = "artifact"
    ITEM = "item"


def _build_vector_schema() -> Any:
    """Build PyArrow schema for LanceDB tables (deferred to avoid import error)."""
    if pa is None:
        raise ImportError("Vector search requires: uv sync --extra vector")
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("type", pa.string()),  # IndexType value
            pa.field("text", pa.string()),  # Searchable text content
            pa.field("vendor", pa.string()),
            pa.field("amount", pa.float64()),
            pa.field("currency", pa.string()),
            pa.field("date", pa.string()),  # ISO date string
            pa.field("metadata", pa.string()),  # JSON string for extra data
            pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
        ]
    )


class VectorIndex:
    """Manages LanceDB vector index for semantic search."""

    def __init__(
        self,
        db_path: Path | None = None,
        embedding_fn: Callable[[str], list[float]] | None = None,
    ):
        """Initialize vector index.

        Args:
            db_path: Path to LanceDB directory (default: ~/.alibi/vectors/)
            embedding_fn: Custom embedding function (default: Ollama)
        """
        if lancedb is None:
            raise ImportError("Vector search requires: uv sync --extra vector")
        self.db_path = db_path or DEFAULT_VECTOR_PATH
        self.db_path.mkdir(parents=True, exist_ok=True)

        self._embedding_fn = embedding_fn or get_embedding
        self._db: lancedb.DBConnection | None = None
        self._table_name = "alibi_vectors"

    @property
    def db(self) -> lancedb.DBConnection:
        """Get LanceDB connection (lazy init)."""
        if self._db is None:
            self._db = lancedb.connect(str(self.db_path))
        return self._db

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text, handling errors gracefully."""
        try:
            return self._embedding_fn(text)
        except EmbeddingError as e:
            logger.error(f"Embedding error: {e}")
            raise

    def is_initialized(self) -> bool:
        """Check if the vector index is initialized."""
        result = self.db.list_tables()
        # Handle both old API (list) and new API (response object with .tables)
        tables = result.tables if hasattr(result, "tables") else result
        return self._table_name in tables

    def initialize(self) -> None:
        """Initialize the vector index (create empty table)."""
        if self.is_initialized():
            logger.info("Vector index already initialized")
            return

        # Create empty table with schema
        self.db.create_table(
            self._table_name,
            schema=_build_vector_schema(),
        )
        logger.info(f"Created vector index at {self.db_path}")

    def drop(self) -> None:
        """Drop the vector index."""
        if self.is_initialized():
            self.db.drop_table(self._table_name)
            logger.info("Dropped vector index")

    def get_stats(self) -> dict[str, int]:
        """Get index statistics."""
        if not self.is_initialized():
            return {"total": 0, "transactions": 0, "artifacts": 0, "items": 0}

        table = self.db.open_table(self._table_name)
        arrow_table = table.to_arrow()

        total = arrow_table.num_rows
        type_column = arrow_table.column("type").to_pylist()

        stats = {
            "total": total,
            "transactions": sum(
                1 for t in type_column if t == IndexType.TRANSACTION.value
            ),
            "artifacts": sum(1 for t in type_column if t == IndexType.ARTIFACT.value),
            "items": sum(1 for t in type_column if t == IndexType.ITEM.value),
        }

        return stats

    def _build_searchable_text(
        self,
        index_type: IndexType,
        vendor: str | None = None,
        description: str | None = None,
        name: str | None = None,
        category: str | None = None,
        raw_text: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Build searchable text from entity fields."""
        parts = []

        if index_type == IndexType.TRANSACTION:
            if vendor:
                parts.append(f"vendor: {vendor}")
            if description:
                parts.append(f"description: {description}")

        elif index_type == IndexType.ARTIFACT:
            if vendor:
                parts.append(f"vendor: {vendor}")
            if raw_text:
                # nomic-embed-text context is ~8192 tokens. JSON content
                # tokenizes at ~2.5 chars/token, so cap at 2500 chars to
                # stay safely within limits for structured text.
                parts.append(raw_text[:2500])

        elif index_type == IndexType.ITEM:
            if name:
                parts.append(f"item: {name}")
            if category:
                parts.append(f"category: {category}")

        return " ".join(parts) if parts else ""

    def index_transaction(
        self,
        transaction_id: str,
        vendor: str | None,
        description: str | None,
        amount: float | None,
        currency: str,
        transaction_date: date | str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index a transaction for semantic search.

        Args:
            transaction_id: Unique transaction ID
            vendor: Vendor name
            description: Transaction description
            amount: Transaction amount
            currency: Currency code
            transaction_date: Transaction date
            metadata: Additional metadata to store
        """
        text = self._build_searchable_text(
            IndexType.TRANSACTION,
            vendor=vendor,
            description=description,
        )

        if not text:
            logger.debug(f"Skipping transaction {transaction_id}: no searchable text")
            return

        embedding = self._get_embedding(text)

        date_str = (
            transaction_date.isoformat()
            if isinstance(transaction_date, date)
            else str(transaction_date) if transaction_date else ""
        )

        record = {
            "id": transaction_id,
            "type": IndexType.TRANSACTION.value,
            "text": text,
            "vendor": vendor or "",
            "amount": float(amount) if amount else 0.0,
            "currency": currency,
            "date": date_str,
            "metadata": json.dumps(metadata or {}),
            "vector": embedding,
        }

        self._upsert_record(record)
        logger.debug(f"Indexed transaction: {transaction_id}")

    def index_artifact(
        self,
        artifact_id: str,
        artifact_type: str,
        vendor: str | None,
        amount: float | None,
        currency: str,
        document_date: date | str | None,
        raw_text: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index an artifact for semantic search.

        Args:
            artifact_id: Unique artifact ID
            artifact_type: Type of artifact (receipt, invoice, etc.)
            vendor: Vendor name
            amount: Document amount
            currency: Currency code
            document_date: Document date
            raw_text: Extracted text from document
            metadata: Additional metadata to store
        """
        text = self._build_searchable_text(
            IndexType.ARTIFACT,
            vendor=vendor,
            raw_text=raw_text,
        )

        if not text:
            logger.debug(f"Skipping artifact {artifact_id}: no searchable text")
            return

        embedding = self._get_embedding(text)

        date_str = (
            document_date.isoformat()
            if isinstance(document_date, date)
            else str(document_date) if document_date else ""
        )

        record = {
            "id": artifact_id,
            "type": IndexType.ARTIFACT.value,
            "text": text,
            "vendor": vendor or "",
            "amount": float(amount) if amount else 0.0,
            "currency": currency,
            "date": date_str,
            "metadata": json.dumps(
                {**(metadata or {}), "artifact_type": artifact_type}
            ),
            "vector": embedding,
        }

        self._upsert_record(record)
        logger.debug(f"Indexed artifact: {artifact_id}")

    def index_item(
        self,
        item_id: str,
        name: str,
        category: str | None,
        purchase_price: float | None,
        currency: str,
        purchase_date: date | str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Index an item for semantic search.

        Args:
            item_id: Unique item ID
            name: Item name
            category: Item category
            purchase_price: Purchase price
            currency: Currency code
            purchase_date: Purchase date
            metadata: Additional metadata to store
        """
        text = self._build_searchable_text(
            IndexType.ITEM,
            name=name,
            category=category,
        )

        if not text:
            logger.debug(f"Skipping item {item_id}: no searchable text")
            return

        embedding = self._get_embedding(text)

        date_str = (
            purchase_date.isoformat()
            if isinstance(purchase_date, date)
            else str(purchase_date) if purchase_date else ""
        )

        record = {
            "id": item_id,
            "type": IndexType.ITEM.value,
            "text": text,
            "vendor": "",  # Items don't have vendors
            "amount": float(purchase_price) if purchase_price else 0.0,
            "currency": currency,
            "date": date_str,
            "metadata": json.dumps(
                {**(metadata or {}), "name": name, "category": category}
            ),
            "vector": embedding,
        }

        self._upsert_record(record)
        logger.debug(f"Indexed item: {item_id}")

    def _upsert_record(self, record: dict[str, Any]) -> None:
        """Insert or update a record in the index."""
        if not self.is_initialized():
            self.initialize()

        table = self.db.open_table(self._table_name)

        # Check if record exists
        existing = (
            table.search()
            .where(f"id = '{record['id']}'", prefilter=True)
            .limit(1)
            .to_list()
        )

        if existing:
            # Delete existing record first
            table.delete(f"id = '{record['id']}'")

        # Add new record
        table.add([record])

    def remove(self, entity_id: str) -> bool:
        """Remove an entity from the index.

        Args:
            entity_id: ID of entity to remove

        Returns:
            True if entity was removed, False if not found
        """
        if not self.is_initialized():
            return False

        table = self.db.open_table(self._table_name)

        # Check if exists
        existing = (
            table.search()
            .where(f"id = '{entity_id}'", prefilter=True)
            .limit(1)
            .to_list()
        )

        if not existing:
            return False

        table.delete(f"id = '{entity_id}'")
        logger.debug(f"Removed from index: {entity_id}")
        return True

    def search(
        self,
        query: str,
        limit: int = 10,
        index_types: list[IndexType] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the index for similar content.

        Args:
            query: Search query text
            limit: Maximum results to return
            index_types: Filter by content type (default: all)

        Returns:
            List of matching records with similarity scores
        """
        if not self.is_initialized():
            return []

        query_embedding = self._get_embedding(query)

        table = self.db.open_table(self._table_name)

        search = table.search(query_embedding)

        # Apply type filter if specified
        if index_types:
            type_values = [t.value for t in index_types]
            type_filter = " OR ".join([f"type = '{t}'" for t in type_values])
            search = search.where(f"({type_filter})", prefilter=True)

        results = search.limit(limit).to_list()

        # Parse metadata and add similarity score
        processed = []
        for r in results:
            try:
                metadata = json.loads(r.get("metadata", "{}"))
            except json.JSONDecodeError:
                metadata = {}

            processed.append(
                {
                    "id": r["id"],
                    "type": r["type"],
                    "text": r["text"],
                    "vendor": r["vendor"],
                    "amount": r["amount"],
                    "currency": r["currency"],
                    "date": r["date"],
                    "metadata": metadata,
                    "score": r.get("_distance", 0.0),  # LanceDB returns distance
                }
            )

        return processed

    def rebuild_from_db(
        self,
        db_manager: DatabaseManager,
        space_id: str = "default",
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, int]:
        """Rebuild the entire index from database.

        Args:
            db_manager: Database manager instance
            space_id: Space ID to index
            progress_callback: Optional callback(entity_type, current, total)

        Returns:
            Dictionary with counts of indexed entities
        """
        # Drop and recreate
        self.drop()
        self.initialize()

        counts = {"transactions": 0, "artifacts": 0, "items": 0}

        # Index facts (v2 transactions)
        txn_rows = db_manager.fetchall(
            """
            SELECT id, vendor, fact_type, total_amount, currency, event_date
            FROM facts
            """,
        )

        total_txns = len(txn_rows)
        for i, row in enumerate(txn_rows):
            try:
                self.index_transaction(
                    transaction_id=str(row[0]),
                    vendor=row[1],
                    description=row[2],
                    amount=float(row[3]) if row[3] else None,
                    currency=row[4] or "EUR",
                    transaction_date=row[5],
                )
                counts["transactions"] += 1

                if progress_callback:
                    progress_callback("transactions", i + 1, total_txns)

            except EmbeddingError as e:
                logger.warning(f"Failed to index fact {row[0]}: {e}")

        # Index documents (v2 artifacts)
        artifact_rows = db_manager.fetchall(
            """
            SELECT d.id, d.file_path, d.raw_extraction,
                   f.fact_type, f.vendor, f.total_amount, f.currency, f.event_date
            FROM documents d
            LEFT JOIN bundles b ON b.document_id = d.id
            LEFT JOIN cloud_bundles cb ON cb.bundle_id = b.id
            LEFT JOIN facts f ON f.cloud_id = cb.cloud_id
            GROUP BY d.id
            """,
        )

        total_artifacts = len(artifact_rows)
        for i, row in enumerate(artifact_rows):
            try:
                self.index_artifact(
                    artifact_id=row[0],
                    artifact_type=row[3] or "document",
                    vendor=row[4],
                    amount=float(row[5]) if row[5] else None,
                    currency=row[6] or "EUR",
                    document_date=row[7],
                    raw_text=row[2],
                )
                counts["artifacts"] += 1

                if progress_callback:
                    progress_callback("artifacts", i + 1, total_artifacts)

            except EmbeddingError as e:
                logger.warning(f"Failed to index document {row[0]}: {e}")

        # Index items
        item_rows = db_manager.fetchall(
            """
            SELECT id, name, category, purchase_price, currency, purchase_date
            FROM items
            WHERE space_id = ?
            """,
            (space_id,),
        )

        total_items = len(item_rows)
        for i, row in enumerate(item_rows):
            try:
                self.index_item(
                    item_id=row[0],
                    name=row[1],
                    category=row[2],
                    purchase_price=float(row[3]) if row[3] else None,
                    currency=row[4] or "EUR",
                    purchase_date=row[5],
                )
                counts["items"] += 1

                if progress_callback:
                    progress_callback("items", i + 1, total_items)

            except EmbeddingError as e:
                logger.warning(f"Failed to index item {row[0]}: {e}")

        logger.info(
            f"Rebuilt index: {counts['transactions']} txns, "
            f"{counts['artifacts']} artifacts, {counts['items']} items"
        )

        return counts
