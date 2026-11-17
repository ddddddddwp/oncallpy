"""Compatibility exports for the SQLite sparse index."""

# 新实现位于 infrastructure；本模块保留旧类名和旧全局对象契约。
from app.rag.bootstrap import get_sparse_index
from app.rag.infrastructure.sqlite_sparse_index import SqliteSparseIndex


class SparseIndexService(SqliteSparseIndex):
    """Backward-compatible name for the SQLite implementation."""


# 通过 bootstrap 的缓存入口取得共享实例，避免兼容全局绕过组合根重复打开数据库。
sparse_index_service = get_sparse_index()

__all__ = ["SparseIndexService", "sparse_index_service"]
