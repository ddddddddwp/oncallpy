"""Sparse indexing capabilities used by the indexing use case."""

# sparse 写端口同时承担 generation 完整性检查，支持 dense 已写而 FTS 缺失时的修复。
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from app.rag.models.document import Document


class SparseWriter(Protocol):
    """Persist and inspect sparse document generations."""

    def upsert_documents(self, documents: Sequence[Document]) -> None: ...

    # replace 要求一个事务内替换某文件的全部 FTS chunk，防止新旧 generation 混杂。
    def replace_file_documents(
        self,
        file_id: str,
        documents: Sequence[Document],
    ) -> int: ...

    # file_id + file_hash 共同标识内容 generation，而非仅凭路径判断是否完整。
    def contains_file_hash(self, file_id: str, file_hash: str) -> bool: ...
