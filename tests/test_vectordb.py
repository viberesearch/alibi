"""Tests for the vectordb module."""

import json
import tempfile
from collections.abc import Generator
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alibi.vectordb.embeddings import (
    EMBEDDING_DIM,
    EmbeddingError,
    create_mock_embedding_fn,
    get_embedding,
)
from alibi.vectordb.index import IndexType, VectorIndex
from alibi.vectordb.search import (
    SearchResult,
    semantic_search,
    sql_search,
    unified_search,
)


class TestEmbeddings:
    """Tests for embedding functions."""

    def test_mock_embedding_fn_returns_correct_dimension(self) -> None:
        """Mock embedding should return correct dimension."""
        mock_fn = create_mock_embedding_fn()
        embedding = mock_fn("test text")

        assert len(embedding) == EMBEDDING_DIM
        assert all(isinstance(x, float) for x in embedding)

    def test_mock_embedding_fn_deterministic(self) -> None:
        """Mock embedding should be deterministic for same input."""
        mock_fn = create_mock_embedding_fn()

        emb1 = mock_fn("hello world")
        emb2 = mock_fn("hello world")

        assert emb1 == emb2

    def test_mock_embedding_fn_different_for_different_inputs(self) -> None:
        """Mock embedding should differ for different inputs."""
        mock_fn = create_mock_embedding_fn()

        emb1 = mock_fn("hello world")
        emb2 = mock_fn("goodbye world")

        assert emb1 != emb2

    def test_mock_embedding_fn_custom_dimension(self) -> None:
        """Mock embedding should support custom dimensions."""
        mock_fn = create_mock_embedding_fn(dim=128)
        embedding = mock_fn("test")

        assert len(embedding) == 128

    def test_get_embedding_error_on_connection_failure(self) -> None:
        """get_embedding should raise EmbeddingError on connection failure."""
        import httpx

        with patch("alibi.vectordb.embeddings.httpx.Client") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value.__enter__.return_value = mock_instance
            mock_instance.post.side_effect = httpx.RequestError(
                "Connection failed", request=MagicMock()
            )

            with pytest.raises(EmbeddingError):
                get_embedding("test", ollama_url="http://invalid:11434")


class TestVectorIndex:
    """Tests for VectorIndex class."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create a temporary directory for test index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_embedding_fn(self):
        """Create a mock embedding function."""
        return create_mock_embedding_fn()

    @pytest.fixture
    def index(self, temp_dir: Path, mock_embedding_fn) -> VectorIndex:
        """Create a test index with mock embeddings."""
        return VectorIndex(db_path=temp_dir, embedding_fn=mock_embedding_fn)

    def test_index_not_initialized_by_default(self, index: VectorIndex) -> None:
        """New index should not be initialized."""
        assert not index.is_initialized()

    def test_initialize_creates_table(self, index: VectorIndex) -> None:
        """Initialize should create the table."""
        index.initialize()
        assert index.is_initialized()

    def test_initialize_idempotent(self, index: VectorIndex) -> None:
        """Multiple initializations should be safe."""
        index.initialize()
        index.initialize()
        assert index.is_initialized()

    def test_drop_removes_table(self, index: VectorIndex) -> None:
        """Drop should remove the table."""
        index.initialize()
        index.drop()
        assert not index.is_initialized()

    def test_get_stats_empty_index(self, index: VectorIndex) -> None:
        """Stats on empty index should show zeros."""
        index.initialize()
        stats = index.get_stats()

        assert stats["total"] == 0
        assert stats["transactions"] == 0
        assert stats["artifacts"] == 0
        assert stats["items"] == 0

    def test_index_transaction(self, index: VectorIndex) -> None:
        """Should index a transaction."""
        index.initialize()

        index.index_transaction(
            transaction_id="txn-001",
            vendor="Amazon",
            description="Books purchase",
            amount=29.99,
            currency="EUR",
            transaction_date=date(2024, 1, 15),
        )

        stats = index.get_stats()
        assert stats["transactions"] == 1
        assert stats["total"] == 1

    def test_index_artifact(self, index: VectorIndex) -> None:
        """Should index an artifact."""
        index.initialize()

        index.index_artifact(
            artifact_id="art-001",
            artifact_type="receipt",
            vendor="Grocery Store",
            amount=85.50,
            currency="EUR",
            document_date=date(2024, 1, 16),
            raw_text="Receipt from grocery store. Items: milk, bread, eggs.",
        )

        stats = index.get_stats()
        assert stats["artifacts"] == 1
        assert stats["total"] == 1

    def test_index_item(self, index: VectorIndex) -> None:
        """Should index an item."""
        index.initialize()

        index.index_item(
            item_id="item-001",
            name="MacBook Pro",
            category="Electronics",
            purchase_price=2499.00,
            currency="EUR",
            purchase_date=date(2024, 1, 10),
        )

        stats = index.get_stats()
        assert stats["items"] == 1
        assert stats["total"] == 1

    def test_upsert_updates_existing_record(self, index: VectorIndex) -> None:
        """Indexing same ID should update, not duplicate."""
        index.initialize()

        # Index twice with same ID
        index.index_transaction(
            transaction_id="txn-001",
            vendor="Amazon",
            description="First purchase",
            amount=29.99,
            currency="EUR",
            transaction_date=date(2024, 1, 15),
        )

        index.index_transaction(
            transaction_id="txn-001",
            vendor="Amazon Updated",
            description="Updated purchase",
            amount=39.99,
            currency="EUR",
            transaction_date=date(2024, 1, 15),
        )

        stats = index.get_stats()
        assert stats["transactions"] == 1  # Not 2

    def test_remove_existing_record(self, index: VectorIndex) -> None:
        """Should remove an existing record."""
        index.initialize()

        index.index_transaction(
            transaction_id="txn-001",
            vendor="Amazon",
            description="Purchase",
            amount=29.99,
            currency="EUR",
            transaction_date=date(2024, 1, 15),
        )

        result = index.remove("txn-001")
        assert result is True

        stats = index.get_stats()
        assert stats["transactions"] == 0

    def test_remove_nonexistent_record(self, index: VectorIndex) -> None:
        """Removing nonexistent record should return False."""
        index.initialize()

        result = index.remove("nonexistent-id")
        assert result is False

    def test_search_returns_results(self, index: VectorIndex) -> None:
        """Search should return matching results."""
        index.initialize()

        # Index some test data
        index.index_transaction(
            transaction_id="txn-001",
            vendor="Amazon",
            description="Book about Python programming",
            amount=29.99,
            currency="EUR",
            transaction_date=date(2024, 1, 15),
        )

        index.index_transaction(
            transaction_id="txn-002",
            vendor="Grocery Store",
            description="Weekly groceries",
            amount=85.50,
            currency="EUR",
            transaction_date=date(2024, 1, 16),
        )

        results = index.search("Python book", limit=5)

        assert len(results) > 0
        assert results[0]["id"] == "txn-001"

    def test_search_with_type_filter(self, index: VectorIndex) -> None:
        """Search should filter by type."""
        index.initialize()

        index.index_transaction(
            transaction_id="txn-001",
            vendor="Amazon",
            description="Purchase",
            amount=29.99,
            currency="EUR",
            transaction_date=date(2024, 1, 15),
        )

        index.index_item(
            item_id="item-001",
            name="Amazon Echo",
            category="Electronics",
            purchase_price=99.00,
            currency="EUR",
            purchase_date=date(2024, 1, 10),
        )

        # Search only items
        results = index.search("Amazon", limit=5, index_types=[IndexType.ITEM])

        assert len(results) == 1
        assert results[0]["type"] == IndexType.ITEM.value

    def test_search_empty_index_returns_empty(self, index: VectorIndex) -> None:
        """Search on empty index should return empty list."""
        index.initialize()
        results = index.search("anything", limit=5)
        assert results == []

    def test_search_uninitialized_index_returns_empty(self, index: VectorIndex) -> None:
        """Search on uninitialized index should return empty list."""
        results = index.search("anything", limit=5)
        assert results == []


class TestSearch:
    """Tests for search functions."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create a temporary directory for test index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_embedding_fn(self):
        """Create a mock embedding function."""
        return create_mock_embedding_fn()

    @pytest.fixture
    def index(self, temp_dir: Path, mock_embedding_fn) -> VectorIndex:
        """Create a test index with mock embeddings."""
        idx = VectorIndex(db_path=temp_dir, embedding_fn=mock_embedding_fn)
        idx.initialize()
        return idx

    @pytest.fixture
    def mock_db(self) -> MagicMock:
        """Create a mock database manager."""
        db = MagicMock()
        db.is_initialized.return_value = True
        return db

    def test_semantic_search_returns_results(self, index: VectorIndex) -> None:
        """semantic_search should return SearchResult objects."""
        index.index_transaction(
            transaction_id="txn-001",
            vendor="Amazon",
            description="Book purchase",
            amount=29.99,
            currency="EUR",
            transaction_date=date(2024, 1, 15),
        )

        results = semantic_search(index, "book", limit=5)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "txn-001"
        assert results[0].source == "vector"
        assert results[0].entity_type == "transaction"

    def test_semantic_search_uninitialized_returns_empty(
        self, temp_dir: Path, mock_embedding_fn
    ) -> None:
        """semantic_search on uninitialized index returns empty."""
        index = VectorIndex(db_path=temp_dir, embedding_fn=mock_embedding_fn)
        results = semantic_search(index, "anything", limit=5)
        assert results == []

    def test_sql_search_transactions(self, mock_db: MagicMock) -> None:
        """sql_search should find facts (transactions)."""
        mock_db.fetchall.side_effect = [
            # Facts
            [("txn-001", "Amazon", "purchase", 29.99, "EUR", "2024-01-15")],
            # Documents
            [],
            # Fact items
            [],
        ]

        results = sql_search(mock_db, "amazon", limit=10)

        assert len(results) == 1
        assert results[0].id == "txn-001"
        assert results[0].source == "sql"
        assert results[0].entity_type == "transaction"

    def test_sql_search_fact_items(self, mock_db: MagicMock) -> None:
        """sql_search should find fact items (line items)."""
        mock_db.fetchall.side_effect = [
            # Facts
            [],
            # Documents
            [],
            # Fact items: id, name, quantity, total_price, category, brand, vendor, event_date
            [
                (
                    "fi-001",
                    "MacBook Pro",
                    1,
                    2499.00,
                    "Electronics",
                    "Apple",
                    "Apple Store",
                    "2024-01-10",
                )
            ],
        ]

        results = sql_search(mock_db, "macbook", limit=10)

        assert len(results) == 1
        assert results[0].id == "fi-001"
        assert results[0].entity_type == "line_item"

    def test_sql_search_documents(self, mock_db: MagicMock) -> None:
        """sql_search should find documents."""
        mock_db.fetchall.side_effect = [
            # Facts
            [],
            # Documents: id, file_path, raw_extraction, vendor, total_amount, currency, event_date, fact_type
            [
                (
                    "doc-001",
                    "/path/receipt.jpg",
                    "Grocery Store receipt",
                    "Grocery Store",
                    85.50,
                    "EUR",
                    "2024-01-16",
                    "purchase",
                )
            ],
            # Fact items
            [],
        ]

        results = sql_search(mock_db, "grocery", limit=10)

        assert len(results) == 1
        assert results[0].id == "doc-001"
        assert results[0].entity_type == "artifact"

    def test_unified_search_combines_results(
        self, index: VectorIndex, mock_db: MagicMock
    ) -> None:
        """unified_search should combine SQL and vector results."""
        # Add to vector index
        index.index_transaction(
            transaction_id="txn-vector",
            vendor="Amazon",
            description="Vector indexed purchase",
            amount=29.99,
            currency="EUR",
            transaction_date=date(2024, 1, 15),
        )

        # Mock SQL results
        mock_db.fetchall.side_effect = [
            # Facts
            [("txn-sql", "Amazon SQL", "purchase", 39.99, "EUR", "2024-01-16")],
            # Documents
            [],
            # Fact items
            [],
        ]

        results = unified_search(
            db=mock_db,
            index=index,
            query="Amazon",
            limit=10,
        )

        # Should have both results
        ids = {r.id for r in results}
        assert "txn-vector" in ids
        assert "txn-sql" in ids

    def test_unified_search_deduplicates(
        self, index: VectorIndex, mock_db: MagicMock
    ) -> None:
        """unified_search should deduplicate by ID."""
        # Add to vector index
        index.index_transaction(
            transaction_id="txn-001",
            vendor="Amazon",
            description="Purchase",
            amount=29.99,
            currency="EUR",
            transaction_date=date(2024, 1, 15),
        )

        # Mock SQL results with same ID
        mock_db.fetchall.side_effect = [
            # Facts - same ID
            [("txn-001", "Amazon", "purchase", 29.99, "EUR", "2024-01-15")],
            # Documents
            [],
            # Fact items
            [],
        ]

        results = unified_search(
            db=mock_db,
            index=index,
            query="Amazon",
            limit=10,
        )

        # Should only have one result
        assert len(results) == 1
        assert results[0].id == "txn-001"

    def test_unified_search_sql_only_mode(
        self, index: VectorIndex, mock_db: MagicMock
    ) -> None:
        """unified_search with use_vector=False should only use SQL."""
        mock_db.fetchall.side_effect = [
            [("txn-sql", "Amazon", "purchase", 29.99, "EUR", "2024-01-15")],
            [],
            [],
        ]

        results = unified_search(
            db=mock_db,
            index=index,
            query="Amazon",
            limit=10,
            use_vector=False,
        )

        # Should only have SQL result
        assert len(results) == 1
        assert results[0].source == "sql"


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_sorting(self) -> None:
        """SearchResults should sort by score descending."""
        results = [
            SearchResult(id="1", source="sql", entity_type="transaction", score=0.5),
            SearchResult(id="2", source="sql", entity_type="transaction", score=0.9),
            SearchResult(id="3", source="sql", entity_type="transaction", score=0.3),
        ]

        sorted_results = sorted(results)

        assert sorted_results[0].id == "2"  # Highest score first
        assert sorted_results[1].id == "1"
        assert sorted_results[2].id == "3"  # Lowest score last

    def test_search_result_default_values(self) -> None:
        """SearchResult should have sensible defaults."""
        result = SearchResult(
            id="test",
            source="sql",
            entity_type="transaction",
            score=0.5,
        )

        assert result.vendor is None
        assert result.description is None
        assert result.amount is None
        assert result.currency == "EUR"
        assert result.date is None
        assert result.metadata is None


class TestIndexType:
    """Tests for IndexType enum."""

    def test_index_types_values(self) -> None:
        """IndexType should have expected values."""
        assert IndexType.TRANSACTION.value == "transaction"
        assert IndexType.ARTIFACT.value == "artifact"
        assert IndexType.ITEM.value == "item"
