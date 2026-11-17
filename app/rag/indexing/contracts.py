"""RAG ingestion pipeline data models."""

# 本模块描述索引流水线各阶段交换的数据和能力，不包含任何具体存储实现。
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from app.rag.models.document import Document
from app.rag.models.indexing_result import IndexingResult


@dataclass(frozen=True)
class LoadedDocument:
    """A document loaded from disk with stable file identity."""

    # path 用于实际访问，normalized_path 用于跨阶段稳定比较与生成 file_id。
    path: str
    normalized_path: str
    file_name: str
    extension: str
    file_size: int
    modified_time: float
    # 同时保留原始字节与解析后正文：前者计算内容 hash，后者进入切分器。
    raw_bytes: bytes
    text: str
    # file_id 标识“同一路径”，file_hash 标识“该路径下的这一版内容”。
    file_id: str
    file_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    """Text content with parser metadata selected from the file type."""

    # ParsedDocument 预留解析器阶段的结果形状，使解析元数据不污染磁盘加载事实。
    loaded: LoadedDocument
    text: str
    parser: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionDecision:
    """Decision made before inserting a loaded file into the vector store."""

    # action 决定主流程是否写新 generation；reason 则保留可观测的判断依据。
    action: Literal["skip", "insert", "replace"]
    reason: str
    deleted_count: int = 0
    # legacy_ids 是旧版仅按 _source 识别的记录，待新 generation 安全后再清理。
    legacy_ids: list[str] = field(default_factory=list)
    can_rollback_generation: bool = False

    @property
    def skipped(self) -> bool:
        # 属性隐藏字符串比较，让调用方以领域语义判断是否可跳过。
        return self.action == "skip"


@dataclass
class FileIndexingOutcome:
    """Indexing outcome for a single file."""

    # 单文件结果既供目录聚合计数，也记录写入 ID、删除量和 best-effort 优化结果。
    file_path: str
    file_id: str = ""
    file_hash: str = ""
    parser: str = ""
    skipped: bool = False
    reason: str = ""
    deleted_count: int = 0
    inserted_chunks: int = 0
    document_ids: list[str] = field(default_factory=list)
    optimization: dict[str, Any] = field(default_factory=dict)


class DocumentLoader(Protocol):
    """Load one supported file into a stable ingestion model."""

    # Protocol 让用例可注入测试 loader，并防止文件系统细节扩散到主编排。
    def load(self, file_path: str) -> LoadedDocument: ...


class DocumentSplitter(Protocol):
    """Split raw text into framework-neutral chunks."""

    def split_document(self, content: str, file_path: str = "") -> list[Document]: ...


class MetadataBuilder(Protocol):
    """Enrich chunks with stable indexing metadata."""

    # 输入 Iterable 允许切分器以任意可迭代形式提供 chunk，输出统一物化为列表。
    def build(
        self,
        loaded: LoadedDocument,
        documents: Iterable[Document],
        parser: str,
    ) -> list[Document]: ...


class Deduplicator(Protocol):
    """Choose the ingestion action for a loaded generation."""

    def prepare(self, loaded: LoadedDocument) -> IngestionDecision: ...


class IndexingUseCase(Protocol):
    """Public indexing operations."""

    def index_directory(self, directory_path: str | None = None) -> IndexingResult: ...

    def index_single_file(self, file_path: str) -> FileIndexingOutcome: ...
