"""Dense retrieval capability used by the query use case."""

# 该端口隐藏 Milvus/LangChain 返回类型，让查询编排只处理领域 SearchHit。
from __future__ import annotations

from typing import Any, Protocol

from app.rag.models.search_hit import SearchHit


class VectorSearcher(Protocol):
    """Search dense candidates without exposing a vendor result type."""

    def search(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]: ...

    # 查询路由会据此决定是否安全地下推过滤条件，避免适配器静默忽略约束。
    def supports_metadata_filter(self) -> bool: ...
