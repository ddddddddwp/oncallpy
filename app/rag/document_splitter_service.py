"""Compatibility exports for the RAG document splitter."""

# 旧导入路径继续转发到新的 indexing 分层；这里不保留第二份实现，避免行为分叉。
from app.rag.indexing.document_splitter import (
    DocumentSplitterService,
    config as config,
    document_splitter_service,
)

__all__ = ["DocumentSplitterService", "document_splitter_service"]
