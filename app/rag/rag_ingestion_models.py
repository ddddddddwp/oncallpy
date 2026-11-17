"""Compatibility exports for RAG ingestion data models."""

# 这些别名让旧调用方继续工作；领域数据结构的单一来源是 indexing.contracts。
from app.rag.indexing.contracts import (
    FileIndexingOutcome,
    IngestionDecision,
    LoadedDocument,
    ParsedDocument,
)

__all__ = [
    "FileIndexingOutcome",
    "IngestionDecision",
    "LoadedDocument",
    "ParsedDocument",
]
