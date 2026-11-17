"""Candidate returned by a retrieval port."""

# SearchHit 把“文档、分数、召回通道”绑定在一起，供融合与重排阶段统一处理。
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.rag.models.document import Document


@dataclass
class SearchHit:
    """A scored domain document and its retrieval channel."""

    document: Document
    # dense 与 sparse 的原始分数含义不同，后续 fusion/reranker 负责归一化解释。
    score: float
    source: Literal["dense", "sparse", "fused"]
