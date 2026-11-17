"""Compatibility exports for the Milvus vector store."""

# 兼容层保留可 patch 的 Milvus/connections 模块全局，便于旧测试与调用方继续注入替身。
from typing import Any, cast

from langchain_milvus import Milvus
from pymilvus import connections

from app.config import config as config
from app.rag.bootstrap import get_vector_store
from app.rag.infrastructure.milvus_vector_store import MilvusVectorStore


class VectorStoreManager(MilvusVectorStore):
    """Backward-compatible Milvus adapter with patchable constructor globals."""

    def __init__(self) -> None:
        # 延迟导入可打破 bootstrap → compatibility → bootstrap 的模块初始化环。
        from app.rag.vector_embedding_service import vector_embedding_service

        super().__init__(
            vector_embedding_service,
            milvus_cls=Milvus,
            connections_api=connections,
        )


class _LazyVectorStoreManager:
    """Delay the legacy global's Milvus connection until its first use."""

    def __getattr__(self, name: str) -> Any:
        # 首次访问旧全局的任意方法时，才经组合根解析真正的 Milvus 适配器。
        return getattr(get_vector_store(), name)


# cast 只服务静态类型检查；运行时仍是透明代理对象。
vector_store_manager = cast(MilvusVectorStore, _LazyVectorStoreManager())

__all__ = ["VectorStoreManager", "vector_store_manager"]
