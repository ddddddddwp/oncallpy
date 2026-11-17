"""Vector indexing capabilities used by the indexing use case."""

# 写端口以 generation 为核心管理 dense 数据，并包含兼容旧主键数据所需的清理能力。
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from app.rag.models.document import Document


class VectorWriter(Protocol):
    """Persist and manage dense document generations."""

    def add_documents(self, documents: Sequence[Document]) -> list[str]: ...

    # has_file_id 判断路径身份是否出现过，has_file_hash 再判断当前内容是否已写入。
    def has_file_id(self, file_id: str) -> bool: ...

    def has_file_hash(self, file_id: str, file_hash: str) -> bool: ...

    def get_legacy_ids_by_source(self, source: str) -> list[str]: ...

    def delete_by_ids(self, document_ids: list[str]) -> int: ...

    def delete_by_file_generation(self, file_id: str, file_hash: str) -> int: ...

    # 新 generation 成功后才删除旧 generation，并明确保留刚写入的 file_hash。
    def delete_file_generations(self, file_id: str, keep_file_hash: str) -> int: ...

    def optimize_collection(self) -> dict[str, Any]: ...
