# RAG Modular Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 RAG 等价迁移为索引、查询、端口、基础设施和领域模型五个清晰边界，同时保留所有旧导入和运行行为。

**Architecture:** 新的 `IndexingService` 与 `QueryService` 只依赖项目领域模型和结构化 `Protocol`；Milvus、SQLite FTS 与 DashScope 位于基础设施层。原有 `app/rag/*.py` 文件保留为兼容入口，通过重导出、薄适配和默认工厂委托给新实现。

**Tech Stack:** Python 3.11、dataclasses、typing.Protocol、LangChain、langchain-milvus、SQLite FTS5、DashScope/OpenAI-compatible API、pytest、Ruff、Pyright

---

## 文件职责映射

### 新增领域模型

- `app/rag/models/document.py`：项目 `Document` 数据模型。
- `app/rag/models/search_query.py`：`RetrievalProfile` 与 `SearchQuery`。
- `app/rag/models/search_hit.py`：稠密、稀疏和融合候选 `SearchHit`。
- `app/rag/models/indexing_result.py`：迁移现有 `IndexingResult`。
- `app/rag/models/__init__.py`：稳定的模型导出。

### 新增端口

- `app/rag/ports/vector_writer.py`：索引写入和 generation 生命周期能力。
- `app/rag/ports/vector_searcher.py`：领域化向量搜索能力。
- `app/rag/ports/sparse_writer.py`：稀疏索引写入和 generation 检查能力。
- `app/rag/ports/sparse_searcher.py`：领域化稀疏搜索能力。
- `app/rag/ports/embedding_provider.py`：文档和查询 embedding 能力。
- `app/rag/ports/__init__.py`：端口导出。

### 新增索引模块

- `app/rag/indexing/contracts.py`：ingestion 数据类以及 loader/splitter/builder/deduplicator/use-case Protocol。
- `app/rag/indexing/document_loader.py`：迁移 `RagDocumentLoader`。
- `app/rag/indexing/document_splitter.py`：迁移切分算法，公开返回领域 `Document`。
- `app/rag/indexing/metadata_builder.py`：迁移元数据增强算法。
- `app/rag/indexing/deduplicator.py`：迁移 generation 决策算法。
- `app/rag/indexing/indexing_service.py`：索引事务编排。
- `app/rag/indexing/__init__.py`：索引用例导出。

### 新增查询模块

- `app/rag/query/contracts.py`：router/fusion/reranker/use-case Protocol。
- `app/rag/query/query_router.py`：过滤推断、过滤回退和 route metadata 映射。
- `app/rag/query/retrieval_fusion.py`：领域 `SearchHit` 版本的 RRF。
- `app/rag/query/reranker.py`：当前本地打分与排序公式。
- `app/rag/query/query_service.py`：检索编排、回退、trace 和上下文替换。
- `app/rag/query/__init__.py`：查询用例导出。

### 新增基础设施和组合根

- `app/rag/infrastructure/dashscope_embeddings.py`：迁移当前 embedding 实现。
- `app/rag/infrastructure/milvus_vector_store.py`：迁移 Milvus 行为并增加领域搜索入口。
- `app/rag/infrastructure/sqlite_sparse_index.py`：迁移 SQLite FTS 行为并增加领域搜索入口。
- `app/rag/infrastructure/__init__.py`：基础设施导出。
- `app/rag/bootstrap.py`：按能力惰性创建默认 embedding、vector store、sparse index、indexing service 和 query service。

### 改为兼容入口

- `app/rag/rag_ingestion_models.py`
- `app/rag/rag_document_loader.py`
- `app/rag/document_splitter_service.py`
- `app/rag/rag_metadata_builder.py`
- `app/rag/rag_deduplicator.py`
- `app/rag/vector_embedding_service.py`
- `app/rag/vector_store_manager.py`
- `app/rag/sparse_index_service.py`
- `app/rag/vector_index_service.py`
- `app/rag/retrieval_fusion.py`
- `app/rag/rag_retrieval_service.py`

`app/services/*` 的现有 module-alias shim 保持不变，它们会继续指向上述 `app.rag` 兼容入口。

## Task 1: 建立基线并添加领域模型

**Files:**
- Create: `app/rag/models/__init__.py`
- Create: `app/rag/models/document.py`
- Create: `app/rag/models/search_query.py`
- Create: `app/rag/models/search_hit.py`
- Create: `app/rag/models/indexing_result.py`
- Create: `tests/test_rag_models.py`

- [ ] **Step 1: 运行现有 RAG 基线**

Run:

```bash
uv run pytest \
  tests/test_rag_ingestion_pipeline.py \
  tests/test_document_splitter_runbook.py \
  tests/test_sparse_index_service.py \
  tests/test_vector_embedding_service.py \
  tests/test_vector_store_filtering.py \
  tests/test_retrieval_fusion.py \
  tests/test_rag_retrieval_service.py -q
```

Expected: 全部通过；若环境问题发生在测试收集阶段，先记录具体模块和异常，不修改业务期望绕过失败。

- [ ] **Step 2: 写领域模型失败测试**

`tests/test_rag_models.py` 覆盖以下代码：

```python
from datetime import datetime, timedelta

from app.rag.models import Document, IndexingResult, RetrievalProfile, SearchHit, SearchQuery


def test_document_metadata_is_not_shared() -> None:
    first = Document(page_content="one")
    second = Document(page_content="two")
    first.metadata["key"] = "value"
    assert second.metadata == {}


def test_search_query_and_hit_use_domain_documents() -> None:
    query = SearchQuery(text="CPUHigh", profile=RetrievalProfile.AIOPS, top_k=2)
    hit = SearchHit(document=Document("runbook"), score=0.8, source="dense")
    assert query.profile == RetrievalProfile.AIOPS
    assert hit.document.page_content == "runbook"


def test_indexing_result_preserves_legacy_dictionary() -> None:
    result = IndexingResult(directory_path="/tmp/docs", total_files=1)
    result.start_time = datetime(2026, 7, 22, 10, 0, 0)
    result.end_time = result.start_time + timedelta(milliseconds=25)
    result.increment_success_count()
    result.add_failed_file("bad.md", "broken")

    assert result.to_dict() == {
        "success": False,
        "directory_path": "/tmp/docs",
        "total_files": 1,
        "success_count": 1,
        "skipped_count": 0,
        "fail_count": 0,
        "deleted_count": 0,
        "inserted_chunk_count": 0,
        "duration_ms": 25,
        "error_message": "",
        "failed_files": {"bad.md": "broken"},
    }
```

- [ ] **Step 3: 确认测试因模块不存在而失败**

Run: `uv run pytest tests/test_rag_models.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: app.rag.models`.

- [ ] **Step 4: 实现领域模型**

实现以下稳定签名：

```python
@dataclass
class Document:
    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str | None = None


class RetrievalProfile(StrEnum):
    CHAT = "chat"
    AIOPS = "aiops"
    PROTOCOL = "protocol"


@dataclass
class SearchQuery:
    text: str
    profile: RetrievalProfile | str = RetrievalProfile.CHAT
    top_k: int | None = None
    metadata_filter: dict[str, Any] | None = None


@dataclass
class SearchHit:
    document: Document
    score: float
    source: Literal["dense", "sparse", "fused"]
```

`IndexingResult` 使用 dataclass 默认值实现，但保留当前四个计数/失败辅助方法、`get_duration_ms()` 和完全一致的 `to_dict()` 键顺序与内容。

- [ ] **Step 5: 运行模型测试**

Run: `uv run pytest tests/test_rag_models.py -q`

Expected: PASS.

- [ ] **Step 6: 提交领域模型**

```bash
git add app/rag/models tests/test_rag_models.py
git commit -m "refactor(rag): add framework-neutral domain models"
```

## Task 2: 定义外部端口

**Files:**
- Create: `app/rag/ports/__init__.py`
- Create: `app/rag/ports/vector_writer.py`
- Create: `app/rag/ports/vector_searcher.py`
- Create: `app/rag/ports/sparse_writer.py`
- Create: `app/rag/ports/sparse_searcher.py`
- Create: `app/rag/ports/embedding_provider.py`
- Create: `tests/test_rag_ports.py`

- [ ] **Step 1: 写端口消费测试和类型夹具**

测试文件定义 `FakeVectorStore`、`FakeSparseIndex`、`FakeEmbeddings`，并显式赋值：

```python
vector_writer: VectorWriter = FakeVectorStore()
vector_searcher: VectorSearcher = FakeVectorStore()
sparse_writer: SparseWriter = FakeSparseIndex()
sparse_searcher: SparseSearcher = FakeSparseIndex()
embedding_provider: EmbeddingProvider = FakeEmbeddings()
```

运行时断言分别调用 `add_documents`、`search`、`replace_file_documents`、`search_hits`、`embed_query`，确认端口可由不继承任何基类的 fake 满足。

- [ ] **Step 2: 确认端口测试失败**

Run: `uv run pytest tests/test_rag_ports.py -q`

Expected: FAIL during collection with missing `app.rag.ports`.

- [ ] **Step 3: 实现五个 Protocol**

端口使用以下方法集合：

```python
class VectorWriter(Protocol):
    def add_documents(self, documents: Sequence[Document]) -> list[str]: ...
    def has_file_id(self, file_id: str) -> bool: ...
    def has_file_hash(self, file_id: str, file_hash: str) -> bool: ...
    def get_legacy_ids_by_source(self, source: str) -> list[str]: ...
    def delete_by_ids(self, document_ids: list[str]) -> int: ...
    def delete_by_file_generation(self, file_id: str, file_hash: str) -> int: ...
    def delete_file_generations(self, file_id: str, keep_file_hash: str) -> int: ...
    def optimize_collection(self) -> dict[str, Any]: ...


class VectorSearcher(Protocol):
    def search(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]: ...
    def supports_metadata_filter(self) -> bool: ...


class SparseWriter(Protocol):
    def upsert_documents(self, documents: Sequence[Document]) -> None: ...
    def replace_file_documents(self, file_id: str, documents: Sequence[Document]) -> int: ...
    def contains_file_hash(self, file_id: str, file_hash: str) -> bool: ...


class SparseSearcher(Protocol):
    def search_hits(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]: ...


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

不添加 `runtime_checkable`，不提供默认实现。

- [ ] **Step 4: 运行端口测试和 Pyright**

Run:

```bash
uv run pytest tests/test_rag_ports.py -q
uv run pyright app/rag/ports tests/test_rag_ports.py
```

Expected: 两条命令均成功，Pyright 为 0 errors。

- [ ] **Step 5: 提交端口**

```bash
git add app/rag/ports tests/test_rag_ports.py
git commit -m "refactor(rag): define storage and embedding ports"
```

## Task 3: 迁移索引协作者和内部契约

**Files:**
- Create: `app/rag/indexing/__init__.py`
- Create: `app/rag/indexing/contracts.py`
- Create: `app/rag/indexing/document_loader.py`
- Create: `app/rag/indexing/document_splitter.py`
- Create: `app/rag/indexing/metadata_builder.py`
- Create: `app/rag/indexing/deduplicator.py`
- Modify: `app/rag/rag_ingestion_models.py`
- Modify: `app/rag/rag_document_loader.py`
- Modify: `app/rag/document_splitter_service.py`
- Modify: `app/rag/rag_metadata_builder.py`
- Modify: `app/rag/rag_deduplicator.py`
- Modify: `tests/test_rag_ingestion_pipeline.py`
- Modify: `tests/test_document_splitter_runbook.py`

- [ ] **Step 1: 添加新旧入口等价失败测试**

增加测试，直接从新路径导入五类对象，并验证：

```python
assert LegacyLoadedDocument is LoadedDocument
assert LegacyRagDocumentLoader is RagDocumentLoader
assert LegacyRagMetadataBuilder is RagMetadataBuilder
assert LegacyRagDeduplicator is RagDeduplicator
```

对 `DocumentSplitterService.split_document()` 和 `RagMetadataBuilder.build()` 增加 `isinstance(result[0], app.rag.models.Document)` 断言，同时保留原有内容和 metadata 断言。

- [ ] **Step 2: 确认测试因新路径不存在而失败**

Run:

```bash
uv run pytest tests/test_rag_ingestion_pipeline.py tests/test_document_splitter_runbook.py -q
```

Expected: FAIL during import of `app.rag.indexing`.

- [ ] **Step 3: 迁移 ingestion 数据类和 Protocol**

把当前 `LoadedDocument`、`ParsedDocument`、`IngestionDecision`、`FileIndexingOutcome` 字段原样迁入 `indexing/contracts.py`，并增加：

```python
class DocumentLoader(Protocol):
    def load(self, file_path: str) -> LoadedDocument: ...


class DocumentSplitter(Protocol):
    def split_document(self, content: str, file_path: str = "") -> list[Document]: ...


class MetadataBuilder(Protocol):
    def build(
        self,
        loaded: LoadedDocument,
        documents: Iterable[Document],
        parser: str,
    ) -> list[Document]: ...


class Deduplicator(Protocol):
    def prepare(self, loaded: LoadedDocument) -> IngestionDecision: ...


class IndexingUseCase(Protocol):
    def index_directory(self, directory_path: str | None = None) -> IndexingResult: ...
    def index_single_file(self, file_path: str) -> FileIndexingOutcome: ...
```

`rag_ingestion_models.py` 改成显式重导出这些数据类。

- [ ] **Step 4: 迁移 loader、metadata builder 和 deduplicator**

复制现有算法，仅替换为新 contracts、领域 `Document` 和 `VectorWriter` 类型；延迟读取 config 的逻辑保持原样。三个旧文件改成显式重导出，不创建第二份实现。

- [ ] **Step 5: 迁移 splitter 并在公开边界转换 Document**

保留当前 LangChain splitter、runbook、protocol table 和 sentence-window 算法。公开方法返回前使用：

```python
def _to_domain_document(document: LangChainDocument) -> Document:
    return Document(
        page_content=document.page_content,
        metadata=dict(document.metadata),
        id=getattr(document, "id", None),
    )
```

内部临时列表继续使用 `LangChainDocument`，`split_markdown`、`split_text`、`split_document` 的返回注解改为领域 `Document`。旧 `document_splitter_service.py` 重导出类并保留同名全局实例。

- [ ] **Step 6: 运行索引协作者测试**

Run:

```bash
uv run pytest tests/test_rag_ingestion_pipeline.py tests/test_document_splitter_runbook.py -q
uv run ruff check app/rag/indexing app/rag/models
uv run pyright app/rag/indexing app/rag/models
```

Expected: 全部成功；隔离进程中的 config 导入断言保持原结果。

- [ ] **Step 7: 提交索引协作者**

```bash
git add app/rag/indexing app/rag/rag_ingestion_models.py \
  app/rag/rag_document_loader.py app/rag/document_splitter_service.py \
  app/rag/rag_metadata_builder.py app/rag/rag_deduplicator.py \
  tests/test_rag_ingestion_pipeline.py tests/test_document_splitter_runbook.py
git commit -m "refactor(rag): move indexing collaborators behind contracts"
```

## Task 4: 迁移三项基础设施实现

**Files:**
- Create: `app/rag/infrastructure/__init__.py`
- Create: `app/rag/infrastructure/dashscope_embeddings.py`
- Create: `app/rag/infrastructure/milvus_vector_store.py`
- Create: `app/rag/infrastructure/sqlite_sparse_index.py`
- Create: `app/rag/bootstrap.py`
- Modify: `app/rag/vector_embedding_service.py`
- Modify: `app/rag/vector_store_manager.py`
- Modify: `app/rag/sparse_index_service.py`
- Modify: `tests/test_vector_embedding_service.py`
- Modify: `tests/test_vector_store_filtering.py`
- Modify: `tests/test_sparse_index_service.py`

- [ ] **Step 1: 添加新基础设施入口失败测试**

在三个现有测试文件中增加新路径导入，验证旧类名仍可用，并验证：

```python
assert issubclass(LegacyDashScopeEmbeddings, DashScopeEmbeddings)
assert issubclass(LegacyVectorStoreManager, MilvusVectorStore)
assert issubclass(LegacySparseIndexService, SqliteSparseIndex)
```

为 SQLite 增加 `search_hits()` 测试，断言返回 `SearchHit` 和领域 `Document`；原 `search()` 的 LangChain tuple 返回测试保持不变。

- [ ] **Step 2: 确认测试因基础设施模块不存在而失败**

Run:

```bash
uv run pytest tests/test_vector_embedding_service.py \
  tests/test_vector_store_filtering.py tests/test_sparse_index_service.py -q
```

Expected: FAIL during import of `app.rag.infrastructure`.

- [ ] **Step 3: 迁移 DashScopeEmbeddings**

将当前类完整迁入 `dashscope_embeddings.py`，保留 `Embeddings` 基类、batch size、OpenAI-compatible URL、参数、日志和异常链。新模块不创建全局实例。旧模块重导出兼容子类，并通过 `bootstrap.get_embedding_provider()` 暴露 `vector_embedding_service`。

- [ ] **Step 4: 迁移 MilvusVectorStore**

把 `VectorStoreManager` 方法完整迁入 `MilvusVectorStore`。构造器增加可注入参数：

```python
def __init__(
    self,
    embedding_provider: EmbeddingProvider,
    *,
    milvus_cls: type[Milvus] = Milvus,
    connections_api: Any = connections,
) -> None:
```

保留原 `similarity_search*` 方法作为第三方兼容 API，并增加领域端口方法：

```python
def search(self, query: str, k: int, metadata_filter=None) -> list[SearchHit]:
    return [
        SearchHit(_to_domain_document(doc), float(score), "dense")
        for doc, score in self.similarity_search_with_score(query, k, metadata_filter)
    ]

def supports_metadata_filter(self) -> bool:
    return True
```

`add_documents` 在调用 LangChain Milvus 前，把领域对象转换为 LangChain `Document`；已经是 LangChain 对象时仍复制 page content 和 metadata，避免修改调用方对象。

- [ ] **Step 5: 迁移 SqliteSparseIndex**

保留 schema、FTS query、metadata filter 和 legacy `search()` tuple API。增加 `search_hits()`，将 tuple 转为 `SearchHit(source="sparse")` 和领域 `Document`。写入方法接受结构上具有 `page_content`/`metadata` 的领域对象。

- [ ] **Step 6: 添加基础惰性工厂和旧类包装**

`bootstrap.py` 先实现三个独立缓存工厂：

```python
@lru_cache(maxsize=1)
def get_embedding_provider() -> DashScopeEmbeddings:
    return DashScopeEmbeddings(
        api_key=config.dashscope_api_key,
        model=config.dashscope_embedding_model,
        dimensions=1024,
    )

@lru_cache(maxsize=1)
def get_vector_store() -> MilvusVectorStore:
    return MilvusVectorStore(get_embedding_provider())

@lru_cache(maxsize=1)
def get_sparse_index() -> SqliteSparseIndex:
    return SqliteSparseIndex(
        db_path=getattr(config, "rag_sparse_index_path", "data/rag_sparse.sqlite")
    )
```

旧 `VectorStoreManager` 子类在直接构造时从旧模块的 `vector_embedding_service`、`Milvus` 和 `connections` 注入父类，以继续支持现有 monkeypatch；旧全局变量使用缓存工厂以与新入口共享实例。旧 embedding 和 sparse 模块采取相同的“兼容类 + 共享默认实例”方式。

- [ ] **Step 7: 运行基础设施测试**

Run:

```bash
uv run pytest tests/test_vector_embedding_service.py \
  tests/test_vector_store_filtering.py tests/test_sparse_index_service.py \
  tests/test_milvus_database_selection.py -q
uv run ruff check app/rag/infrastructure app/rag/bootstrap.py
uv run pyright app/rag/infrastructure app/rag/bootstrap.py
```

Expected: 全部成功；Milvus database name、过滤 fail-closed 和 FTS 排序断言不变。

- [ ] **Step 8: 提交基础设施**

```bash
git add app/rag/infrastructure app/rag/bootstrap.py \
  app/rag/vector_embedding_service.py app/rag/vector_store_manager.py \
  app/rag/sparse_index_service.py tests/test_vector_embedding_service.py \
  tests/test_vector_store_filtering.py tests/test_sparse_index_service.py
git commit -m "refactor(rag): isolate vector sparse and embedding infrastructure"
```

## Task 5: 迁移 IndexingService 并保留旧入口

**Files:**
- Create: `app/rag/indexing/indexing_service.py`
- Modify: `app/rag/indexing/contracts.py`
- Modify: `app/rag/indexing/__init__.py`
- Modify: `app/rag/bootstrap.py`
- Modify: `app/rag/vector_index_service.py`
- Modify: `tests/test_rag_ingestion_pipeline.py`

- [ ] **Step 1: 添加新索引用例失败测试**

沿用现有 `FakeLoader`、`FakeSplitter`、`FakeMetadataBuilder`、`FakeStore` 和 `FakeSparseIndex`，直接构造：

```python
service = IndexingService(
    loader=FakeLoader(),
    splitter=FakeSplitter(),
    metadata_builder=FakeMetadataBuilder(),
    vector_writer=store,
    sparse_writer=sparse,
    deduplicator=FakeDeduplicator(
        IngestionDecision(action="insert", reason="new")
    ),
)
```

断言新入口返回领域 `FileIndexingOutcome`，旧 `VectorIndexService` 对同一 fixture 保持相同 `inserted_chunks`、删除调用顺序和 route summary metadata。

- [ ] **Step 2: 确认新服务不存在**

Run: `uv run pytest tests/test_rag_ingestion_pipeline.py -q`

Expected: FAIL importing `app.rag.indexing.indexing_service`.

- [ ] **Step 3: 迁移索引编排**

将当前 `VectorIndexService` 的 `index_directory`、`index_single_file`、sparse repair、rollback 和 route summary 方法完整迁入 `IndexingService`。仅进行以下机械变化：

- `IndexingResult` 从 `app.rag.models` 导入。
- 文档类型使用领域 `Document`。
- `vector_store` 属性改为 `vector_writer`。
- `sparse_index` 属性改为 `sparse_writer`。
- 构造器要求显式注入 `VectorWriter`，并允许 `sparse_writer` 或惰性 `sparse_writer_factory`。
- `_get_sparse_index()` 只读取注入对象或工厂，不导入旧模块。

所有 catch、日志消息、删除顺序和结果计数保持原样。

- [ ] **Step 4: 实现旧 VectorIndexService 包装**

旧类保留原构造签名，把 `vector_store` 映射为 `vector_writer`，把 `sparse_index` 映射为 `sparse_writer`。缺省依赖继续从旧兼容模块取得；sparse 继续延迟导入。旧模块显式重导出 `IndexingResult`，并保留 `vector_index_service` 全局实例。

- [ ] **Step 5: 扩展默认索引工厂**

在 `bootstrap.py` 增加缓存的 `get_indexing_service()`，显式创建 loader、splitter、metadata builder、deduplicator，并注入共享 vector/sparse 实例。工厂不得被导入 `app.rag.indexing` 的动作自动调用。

- [ ] **Step 6: 运行全部索引特征测试**

Run:

```bash
uv run pytest tests/test_rag_ingestion_pipeline.py \
  tests/test_document_splitter_runbook.py tests/test_sparse_index_service.py -q
uv run ruff check app/rag/indexing app/rag/vector_index_service.py
uv run pyright app/rag/indexing app/rag/vector_index_service.py
```

Expected: 全部成功，尤其是 generation rollback、sparse repair、legacy-id 删除和 route summary 测试。

- [ ] **Step 7: 提交索引用例**

```bash
git add app/rag/indexing app/rag/bootstrap.py app/rag/vector_index_service.py \
  tests/test_rag_ingestion_pipeline.py
git commit -m "refactor(rag): move indexing orchestration behind ports"
```

## Task 6: 提取查询路由、融合和重排

**Files:**
- Create: `app/rag/query/__init__.py`
- Create: `app/rag/query/contracts.py`
- Create: `app/rag/query/query_router.py`
- Create: `app/rag/query/retrieval_fusion.py`
- Create: `app/rag/query/reranker.py`
- Modify: `app/rag/retrieval_fusion.py`
- Create: `tests/test_rag_query_components.py`
- Modify: `tests/test_retrieval_fusion.py`
- Modify: `tests/test_rag_retrieval_service.py`

- [ ] **Step 1: 写组件级等价失败测试**

测试以下固定行为：

- `DefaultQueryRouter.infer_metadata_filter("英威腾逆变器 0x3601", PROTOCOL)` 同时包含 protocol 文档类型、设备、厂商和寄存器地址。
- `metadata_filter_fallbacks()` 只按当前规则移除 manufacturer，且去重。
- 新 RRF 对 dense/sparse 重复 chunk 只返回一次，写入 `_retrieval_sources` 和 relevance score kind。
- `LocalReranker` 在现有权重下把匹配 alert/runbook 的候选排在普通文档前。

- [ ] **Step 2: 确认组件模块不存在**

Run:

```bash
uv run pytest tests/test_rag_query_components.py tests/test_retrieval_fusion.py -q
```

Expected: FAIL importing `app.rag.query`.

- [ ] **Step 3: 实现查询 contracts 与 router**

`contracts.py` 声明：

```python
class QueryRouter(Protocol):
    def infer_metadata_filter(self, query: str, profile: RetrievalProfile) -> dict[str, Any]: ...
    def metadata_filter_fallbacks(self, value: dict[str, Any]) -> list[dict[str, Any]]: ...
    def route_summary_filter(self, value: dict[str, Any] | None) -> dict[str, Any]: ...
    def filter_from_route_metadata(
        self, metadata: dict[str, Any], profile: RetrievalProfile
    ) -> dict[str, Any]: ...
    def requires_taxonomy_match(self, value: dict[str, Any]) -> bool: ...

class RetrievalFusion(Protocol):
    def fuse(
        self, dense_results: list[SearchHit], sparse_results: list[SearchHit], k: int
    ) -> list[SearchHit]: ...

class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[SearchHit], profile: RetrievalProfile
    ) -> list[SearchHit]: ...


class QueryUseCase(Protocol):
    def retrieve(self, query: SearchQuery) -> list[Document]: ...
```

`DefaultQueryRouter` 原样接收当前 `_infer_*`、filter fallback、route filter 和 default document type 逻辑。

- [ ] **Step 4: 实现领域 RRF 和本地 reranker**

RRF 保留 `_chunk_id`/`id`/`chunk_id`/source+content key 规则和 `1 / (k + rank)` 公式，输出 `SearchHit(source="fused")`。`LocalReranker` 原样接收三个权重、term extraction、vector relevance、keyword score 和 metadata boost 公式。

旧 `retrieval_fusion.py` 把 legacy tuple 转为 `SearchHit`，调用新 fusion，再把结果转换回 tuple；旧函数签名保持不变。

- [ ] **Step 5: 运行组件与原查询测试**

Run:

```bash
uv run pytest tests/test_rag_query_components.py \
  tests/test_retrieval_fusion.py tests/test_rag_retrieval_service.py -q
uv run ruff check app/rag/query app/rag/retrieval_fusion.py
uv run pyright app/rag/query
```

Expected: 全部成功；当前 `RagRetrievalService` 尚未迁移，因此查询特征测试作为对照继续通过。

- [ ] **Step 6: 提交查询组件**

```bash
git add app/rag/query app/rag/retrieval_fusion.py \
  tests/test_rag_query_components.py tests/test_retrieval_fusion.py \
  tests/test_rag_retrieval_service.py
git commit -m "refactor(rag): extract query routing fusion and reranking"
```

## Task 7: 迁移 QueryService 并适配旧检索入口

**Files:**
- Create: `app/rag/query/query_service.py`
- Modify: `app/rag/query/contracts.py`
- Modify: `app/rag/query/__init__.py`
- Modify: `app/rag/bootstrap.py`
- Modify: `app/rag/rag_retrieval_service.py`
- Modify: `tests/test_rag_retrieval_service.py`

- [ ] **Step 1: 添加新查询用例失败测试**

为新端口 fake 实现 `search()`、`supports_metadata_filter()` 和 `search_hits()`，构造：

```python
service = QueryService(
    vector_searcher=FakeVectorSearcher(
        [SearchHit(Document("CPUHigh troubleshooting"), 0.2, "dense")]
    ),
    sparse_searcher=FakeSparseSearcher([]),
    router=DefaultQueryRouter(),
    fusion=ReciprocalRankFusion(),
    reranker=LocalReranker(),
)
documents = service.retrieve(SearchQuery("CPUHigh 怎么排查", RetrievalProfile.AIOPS, 1))
assert isinstance(documents[0], DomainDocument)
```

再对旧 `RagRetrievalService` 使用现有 legacy fake，断言返回值仍为 LangChain `Document`，内容、metadata、过滤调用和排序与迁移前相同。

- [ ] **Step 2: 确认 QueryService 不存在**

Run: `uv run pytest tests/test_rag_retrieval_service.py -q`

Expected: FAIL importing `app.rag.query.query_service`.

- [ ] **Step 3: 迁移查询编排**

把当前 `retrieve`、search candidates、route-first、safe dense/sparse search、trace、profile/top-k/config 和 context-window 逻辑迁入 `QueryService`。机械变化为：

- 入口变为 `retrieve(self, query: SearchQuery) -> list[Document]`。
- 候选统一为 `SearchHit`。
- 路由、融合和重排分别调用注入 Protocol。
- `_vector_store_accepts_metadata_filter()` 改为调用 `supports_metadata_filter()`。
- 不再导入旧 vector/sparse 全局对象。
- trace 名称、输入字段、输出字段、catch 和空列表行为保持不变。

- [ ] **Step 4: 实现 legacy 搜索适配器**

在旧模块内定义私有适配器。向量适配器调用 legacy `similarity_search_with_score()`，将每个 tuple 转为领域 `SearchHit(source="dense")`；稀疏适配器调用 legacy `search()` 并转为 `SearchHit(source="sparse")`。向量适配器使用当前 `inspect.signature` 规则报告 metadata-filter 支持情况。

适配器接受 provider callable，使未显式注入依赖的全局 `rag_retrieval_service` 在模块导入时不连接 Milvus 或创建 sparse DB。

- [ ] **Step 5: 实现旧 RagRetrievalService 包装**

旧类保留：

```python
def retrieve(
    self,
    query: str,
    profile: RetrievalProfile | str = RetrievalProfile.CHAT,
    top_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
) -> list[LangChainDocument]:
```

它构造 `SearchQuery`，调用新服务，并把每个领域对象转换为新的 LangChain `Document`。构造器仍接受 `vector_store` 和 `sparse_search`，显式 fake 优先于默认 provider。

- [ ] **Step 6: 扩展默认查询工厂**

`bootstrap.py` 增加缓存的 `get_query_service()`，注入共享 Milvus/SQLite 实例和默认 router/fusion/reranker。调用工厂之前不建立外部连接。

- [ ] **Step 7: 运行所有查询特征测试**

Run:

```bash
uv run pytest tests/test_rag_query_components.py \
  tests/test_retrieval_fusion.py tests/test_rag_retrieval_service.py -q
uv run ruff check app/rag/query app/rag/rag_retrieval_service.py
uv run pyright app/rag/query app/rag/rag_retrieval_service.py
```

Expected: 全部成功，尤其是显式 filter fail-closed、route fallback、hybrid RRF 顺序、unknown profile、candidate-k 和 trace 测试。

- [ ] **Step 8: 提交查询用例**

```bash
git add app/rag/query app/rag/bootstrap.py app/rag/rag_retrieval_service.py \
  tests/test_rag_retrieval_service.py
git commit -m "refactor(rag): move query orchestration behind ports"
```

## Task 8: 固化兼容性和依赖方向

**Files:**
- Modify: `app/rag/__init__.py`
- Modify: `tests/test_compatibility.py`
- Create: `tests/test_rag_architecture.py`

- [ ] **Step 1: 写兼容和架构失败测试**

增加测试覆盖：

- `app.services.*` shim 仍与对应 `app.rag.*` 模块对象相同。
- 旧类、函数和全局实例全部可导入。
- 单独导入 `app.rag.models`、`app.rag.ports`、`app.rag.indexing.contracts`、`app.rag.query.contracts` 不加载 `langchain_milvus`、`pymilvus` 或 `openai`。
- 用 AST 读取 `indexing/indexing_service.py`、`query/query_service.py`、router、fusion、reranker，断言 import target 不以 `app.rag.infrastructure`、`langchain`、`pymilvus` 或 `openai` 开头。

- [ ] **Step 2: 运行测试并观察尚未满足的导入边界**

Run:

```bash
uv run pytest tests/test_compatibility.py tests/test_rag_architecture.py -q
```

Expected: 在 package 导出或依赖方向尚未完成的位置 FAIL，并明确列出违规 import。

- [ ] **Step 3: 完成惰性包导出**

`app/rag/__init__.py` 只声明包和 `__all__`，不 eagerly import bootstrap 或具体基础设施。需要公共便捷导出时使用 PEP 562 `__getattr__` 延迟导入，确保 `import app.rag` 不建立外部连接。

- [ ] **Step 4: 修正依赖方向并运行兼容测试**

将检测到的具体实现 import 移至 `bootstrap.py` 或旧兼容模块；不得用测试忽略列表掩盖业务服务到基础设施的直接依赖。

Run:

```bash
uv run pytest tests/test_compatibility.py tests/test_rag_architecture.py \
  tests/test_app_startup.py -q
```

Expected: 全部成功，且启动隔离测试无新增 stderr。

- [ ] **Step 5: 提交兼容和架构保护**

```bash
git add app/rag/__init__.py tests/test_compatibility.py tests/test_rag_architecture.py
git commit -m "test(rag): guard compatibility and dependency direction"
```

## Task 9: 全量验证与收尾

**Files:**
- Modify: only files reported by the verification commands below

- [ ] **Step 1: 运行 RAG 全套测试**

Run:

```bash
uv run pytest \
  tests/test_rag_models.py tests/test_rag_ports.py \
  tests/test_rag_query_components.py tests/test_rag_architecture.py \
  tests/test_rag_ingestion_pipeline.py tests/test_document_splitter_runbook.py \
  tests/test_sparse_index_service.py tests/test_vector_embedding_service.py \
  tests/test_vector_store_filtering.py tests/test_milvus_database_selection.py \
  tests/test_retrieval_fusion.py tests/test_rag_retrieval_service.py \
  tests/test_compatibility.py -q
```

Expected: PASS.

- [ ] **Step 2: 运行 Ruff 并只修复本次范围问题**

Run:

```bash
uv run ruff check app/rag tests/test_rag_models.py tests/test_rag_ports.py \
  tests/test_rag_query_components.py tests/test_rag_architecture.py
```

Expected: `All checks passed!`。不得格式化或改写与本次变更无关的用户文件。

- [ ] **Step 3: 运行 Pyright**

Run: `uv run pyright app/rag`

Expected: 0 errors。已有 information/warning 只有在命令退出码为 0 且与本次签名兼容时可保留并记录。

- [ ] **Step 4: 运行完整测试集**

Run: `uv run pytest -q`

Expected: PASS；若只剩需要真实容器或凭据的失败，保存完整失败信息并运行所有不依赖该外部状态的测试证明代码路径通过。

- [ ] **Step 5: 检查 diff 和旧入口**

Run:

```bash
git diff --check
git status --short
uv run python -c "from app.rag.vector_index_service import VectorIndexService; from app.rag.rag_retrieval_service import RagRetrievalService; print(VectorIndexService.__name__, RagRetrievalService.__name__)"
```

Expected: diff 无 whitespace error；状态中只出现本次 RAG 文件和用户原有的 `pyproject.toml`、`pyrightconfig.json`、`uv.lock`、`.understand-anything/`；导入命令打印两个兼容类名。

- [ ] **Step 6: 提交验证修正**

如 Step 1–5 产生本次范围内修正：

```bash
git add app/rag tests
git commit -m "test(rag): verify modular boundary migration"
```

如果没有产生修正，不创建空提交。

## 完成定义

- 新目录结构完整且每个文件职责单一。
- 领域模型和 Protocol 不依赖第三方框架类型。
- `IndexingService`、`QueryService`、router、fusion、reranker 不导入基础设施或第三方 SDK。
- splitter 仅在内部使用 LangChain 临时对象，公开返回领域 `Document`。
- 默认新旧入口共享 embedding、Milvus 和 SQLite 实例。
- 旧 import、构造签名、全局变量和旧查询 LangChain 返回类型可用。
- 索引 transaction、查询 fallback、排序、trace 和日志语义由原测试证明保持。
- Ruff、Pyright、RAG 定向测试和可运行的完整测试集通过。
