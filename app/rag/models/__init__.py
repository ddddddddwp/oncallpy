"""Framework-neutral models used by the RAG domain."""

# 统一从这里导出领域对象，调用方无需知道每个 dataclass 的物理文件位置。
from app.rag.models.document import Document
from app.rag.models.indexing_result import IndexingResult
from app.rag.models.search_hit import SearchHit
from app.rag.models.search_query import RetrievalProfile, SearchQuery

__all__ = [
    "Document",
    "IndexingResult",
    "RetrievalProfile",
    "SearchHit",
    "SearchQuery",
]
