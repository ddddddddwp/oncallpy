"""External capabilities required by the RAG use cases."""

# ports 是依赖倒置的边界：用例依赖能力定义，infrastructure 再实现这些能力。
from app.rag.ports.embedding_provider import EmbeddingProvider
from app.rag.ports.sparse_searcher import SparseSearcher
from app.rag.ports.sparse_writer import SparseWriter
from app.rag.ports.vector_searcher import VectorSearcher
from app.rag.ports.vector_writer import VectorWriter

__all__ = [
    "EmbeddingProvider",
    "SparseSearcher",
    "SparseWriter",
    "VectorSearcher",
    "VectorWriter",
]
