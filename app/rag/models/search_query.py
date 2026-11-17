"""Query inputs and supported retrieval profiles."""

# 查询输入保持框架无关，既可从 Agent tool 构造，也可由普通应用代码直接使用。
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class RetrievalProfile(StrEnum):
    """Supported retrieval profiles."""

    # profile 决定默认 top_k、路由约束和 metadata boost，而不是选择不同存储。
    CHAT = "chat"
    AIOPS = "aiops"
    PROTOCOL = "protocol"


@dataclass
class SearchQuery:
    """Framework-neutral request accepted by the query use case."""

    text: str
    # top_k=None 让 QueryService 根据 profile 选择默认值；显式 filter 会与推断结果合并。
    profile: RetrievalProfile | str = RetrievalProfile.CHAT
    top_k: int | None = None
    metadata_filter: dict[str, Any] | None = None
