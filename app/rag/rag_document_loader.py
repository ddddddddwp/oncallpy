"""Compatibility exports for the RAG document loader."""

# 兼容层不复制逻辑，旧名称与新分层始终引用同一个实现类。
from app.rag.indexing.document_loader import RagDocumentLoader

__all__ = ["RagDocumentLoader"]
