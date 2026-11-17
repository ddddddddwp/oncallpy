"""Compatibility exports for RAG ingestion deduplication."""

# 该文件只维持历史导入路径；真正实现集中在 indexing.deduplicator。
from app.rag.indexing.deduplicator import RagDeduplicator

__all__ = ["RagDeduplicator"]
