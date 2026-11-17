"""Document model owned by the RAG domain."""

# 这是跨索引、查询和端口传递的最小文档类型，刻意不继承 LangChain Document。
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Text and metadata passed across RAG module boundaries."""

    # 正文用于 embedding、FTS 与最终上下文；metadata 承载路由、generation 和来源信息。
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # id 可映射到稳定 chunk 主键；尚未持久化的中间文档允许为空。
    id: str | None = None
