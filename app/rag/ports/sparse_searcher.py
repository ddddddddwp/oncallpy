"""Sparse retrieval capability used by the query use case."""

# 用例只接收领域 SearchHit，不让 SQLite FTS 的 row/score 类型越过端口边界。
from __future__ import annotations

from typing import Any, Protocol

from app.rag.models.search_hit import SearchHit


class SparseSearcher(Protocol):
    """Search sparse candidates without exposing a storage result type."""

    # metadata_filter 与 dense 端口形状一致，使 QueryService 可以并行组织两路召回。
    def search_hits(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]: ...
