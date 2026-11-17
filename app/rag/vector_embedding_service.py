"""Compatibility exports for the DashScope embedding service."""

# 旧调用方仍可导入全局 service；实际生命周期由 bootstrap 的 lru_cache 管理。
from app.rag.bootstrap import get_embedding_provider
from app.rag.infrastructure.dashscope_embeddings import DashScopeEmbeddings

# 这里保留历史上的 eager 全局名称，具体 provider 仍由组合根统一构造。
vector_embedding_service = get_embedding_provider()

__all__ = ["DashScopeEmbeddings", "vector_embedding_service"]
