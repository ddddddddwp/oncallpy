"""Compatibility exports for the RAG metadata builder."""

# 旧模块名只做转发，避免两套 metadata schema 随时间产生差异。
from app.rag.indexing.metadata_builder import RagMetadataBuilder

__all__ = ["RagMetadataBuilder"]
