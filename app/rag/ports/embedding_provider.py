"""Embedding capability required by vector infrastructure."""

# Protocol 只描述结构类型；实现无需继承它，便于替换供应商或测试桩。
from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Create vectors for document batches and individual queries."""

    # 批量文档与单条查询分开建模，对应多数 embedding SDK 的两种调用路径。
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
