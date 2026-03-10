"""Vector database module for semantic search using LanceDB."""

from alibi.vectordb.embeddings import (
    EmbeddingError,
    get_embedding,
    get_embedding_async,
)
from alibi.vectordb.index import VectorIndex
from alibi.vectordb.search import (
    SearchResult,
    semantic_search,
    unified_search,
)

__all__ = [
    "EmbeddingError",
    "get_embedding",
    "get_embedding_async",
    "VectorIndex",
    "SearchResult",
    "semantic_search",
    "unified_search",
]
