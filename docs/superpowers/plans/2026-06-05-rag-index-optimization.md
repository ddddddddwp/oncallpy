# RAG Index Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the dependency-free RAG ingestion and retrieval enhancement approved in `docs/superpowers/specs/2026-06-05-rag-index-optimization-design.md`.

**Architecture:** Keep public APIs stable while adding explicit ingestion stages for loading, metadata, deduplication, vector writing, and optimization. Retrieval remains exposed through `retrieve_knowledge`, but internally uses profiles, larger recall, keyword filtering, and local reranking.

**Tech Stack:** Python 3.11, FastAPI, LangChain `Document`, LangChain Milvus, pymilvus, pytest.

---

## File Structure

- Create `app/services/rag_ingestion_models.py`: dataclasses for loaded/parsed documents and indexing outcomes.
- Create `app/services/rag_document_loader.py`: UTF-8 file loader, SHA-256 file hash and stable file id.
- Create `app/services/rag_metadata_builder.py`: deterministic chunk ids and complete metadata.
- Create `app/services/rag_deduplicator.py`: incremental indexing decisions against the vector store boundary.
- Create `app/services/rag_retrieval_service.py`: profile selection, keyword filtering, local rerank.
- Modify `app/config.py`: add RAG/Milvus tuning fields.
- Modify `app/core/milvus_client.py`: read collection/index configuration from config.
- Modify `app/services/document_splitter_service.py`: add operations-runbook metadata and final chunk indexes.
- Modify `app/services/vector_store_manager.py`: delete/query by file id/hash, search with scores, optimize collection.
- Modify `app/services/vector_index_service.py`: orchestrate the ingestion pipeline and report skipped/inserted/deleted stats.
- Modify `app/tools/knowledge_tool.py`: call the retrieval service and support chat/AIOps profile selection.
- Modify `app/agent/aiops_multi/nodes/knowledge_agent.py`: request the AIOps retrieval profile.
- Add `tests/test_rag_ingestion_pipeline.py`: loader, metadata builder, dedup, index orchestration tests.
- Add `tests/test_rag_retrieval_service.py`: rerank/profile/tool behavior tests.
- Add `tests/test_document_splitter_runbook.py`: operations Markdown structure tests.

## Task 1: Loader And Metadata Builder

**Files:**
- Create: `app/services/rag_ingestion_models.py`
- Create: `app/services/rag_document_loader.py`
- Create: `app/services/rag_metadata_builder.py`
- Test: `tests/test_rag_ingestion_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

from langchain_core.documents import Document

from app.services.rag_document_loader import RagDocumentLoader
from app.services.rag_metadata_builder import RagMetadataBuilder


def test_loader_computes_stable_file_identity(tmp_path: Path):
    file_path = tmp_path / "cpu.md"
    file_path.write_bytes("# CPU告警\n处理步骤".encode("utf-8"))

    loaded = RagDocumentLoader().load(str(file_path))

    assert loaded.normalized_path == file_path.resolve().as_posix()
    assert loaded.file_id
    assert loaded.file_hash
    assert loaded.file_size == len(file_path.read_bytes())
    assert loaded.extension == ".md"
    assert loaded.text == "# CPU告警\n处理步骤"


def test_metadata_builder_adds_stable_chunk_metadata(tmp_path: Path):
    file_path = tmp_path / "cpu.md"
    file_path.write_text("# CPU告警\n## 处理方案\n扩容", encoding="utf-8")
    loaded = RagDocumentLoader().load(str(file_path))
    docs = [
        Document(
            page_content="## 处理方案\n扩容",
            metadata={"alert_name": "CPU告警", "runbook_section": "resolution"},
        )
    ]

    enriched = RagMetadataBuilder(schema_version=7, collection_name="biz_v7").build(
        loaded=loaded,
        documents=docs,
        parser="markdown",
    )

    metadata = enriched[0].metadata
    assert metadata["_schema_version"] == 7
    assert metadata["_collection_name"] == "biz_v7"
    assert metadata["_file_id"] == loaded.file_id
    assert metadata["_file_hash"] == loaded.file_hash
    assert metadata["_chunk_id"]
    assert metadata["_chunk_hash"]
    assert metadata["_chunk_index"] == 0
    assert metadata["_chunk_count"] == 1
    assert metadata["alert_name"] == "CPU告警"
    assert metadata["runbook_section"] == "resolution"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_ingestion_pipeline.py -q`

Expected: FAIL because the new modules do not exist.

- [ ] **Step 3: Implement minimal loader and metadata builder**

Create dataclasses for `LoadedDocument`, `ParsedDocument`, `IngestionDecision`, and `FileIndexingOutcome`. Implement SHA-256 helpers and deterministic chunk metadata. Preserve existing splitter metadata while adding underscore-prefixed system fields.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rag_ingestion_pipeline.py -q`

Expected: PASS for loader and metadata tests.

## Task 2: Operations Markdown Chunk Metadata

**Files:**
- Modify: `app/services/document_splitter_service.py`
- Test: `tests/test_document_splitter_runbook.py`

- [ ] **Step 1: Write failing tests**

```python
from app.services.document_splitter_service import DocumentSplitterService


def test_markdown_splitter_preserves_runbook_structure():
    content = """
# CPU 使用率过高

## 告警详情
- 告警名称: CPUHigh
- 服务: checkout

## 排查步骤
1. 查看 CPU 指标
2. 查询错误日志

## 处理方案
扩容 checkout 实例。
""".strip()

    docs = DocumentSplitterService().split_markdown(content, "/tmp/cpu.md")

    assert docs
    assert all("_chunk_index" in doc.metadata for doc in docs)
    assert all("_chunk_count" in doc.metadata for doc in docs)
    assert any(doc.metadata.get("alert_name") == "CPUHigh" for doc in docs)
    assert any(doc.metadata.get("runbook_section") == "troubleshooting" for doc in docs)
    assert any(doc.metadata.get("runbook_section") == "resolution" for doc in docs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_document_splitter_runbook.py -q`

Expected: FAIL because `alert_name`, `runbook_section`, and `_chunk_index` are not emitted.

- [ ] **Step 3: Implement minimal runbook metadata extraction**

Add helpers that infer `alert_name` from common labels and infer `runbook_section` from headings. Assign `_chunk_index` and `_chunk_count` after final splitting/merging.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_document_splitter_runbook.py -q`

Expected: PASS.

## Task 3: Vector Store Deduplication Boundary

**Files:**
- Modify: `app/services/vector_store_manager.py`
- Create: `app/services/rag_deduplicator.py`
- Test: `tests/test_rag_ingestion_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
from app.services.rag_deduplicator import RagDeduplicator
from app.services.rag_ingestion_models import LoadedDocument


class FakeVectorStore:
    def __init__(self, has_same_hash=False, existing_file=False):
        self.has_same_hash = has_same_hash
        self.existing_file = existing_file
        self.deleted_file_ids = []
        self.deleted_sources = []

    def has_file_hash(self, file_id, file_hash):
        return self.has_same_hash

    def has_file_id(self, file_id):
        return self.existing_file

    def delete_by_file_id(self, file_id):
        self.deleted_file_ids.append(file_id)
        return 3

    def delete_by_source(self, source):
        self.deleted_sources.append(source)
        return 0


def _loaded():
    return LoadedDocument(
        path="/tmp/cpu.md",
        normalized_path="/tmp/cpu.md",
        file_name="cpu.md",
        extension=".md",
        file_size=8,
        modified_time=1.0,
        raw_bytes=b"content",
        text="content",
        file_id="file-id",
        file_hash="file-hash",
    )


def test_deduplicator_skips_unchanged_file():
    store = FakeVectorStore(has_same_hash=True)

    decision = RagDeduplicator(store).prepare(_loaded())

    assert decision.action == "skip"
    assert decision.deleted_count == 0


def test_deduplicator_deletes_changed_file_by_file_id():
    store = FakeVectorStore(existing_file=True)

    decision = RagDeduplicator(store).prepare(_loaded())

    assert decision.action == "replace"
    assert decision.deleted_count == 3
    assert store.deleted_file_ids == ["file-id"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_ingestion_pipeline.py -q`

Expected: FAIL because `RagDeduplicator` and vector-store methods do not exist.

- [ ] **Step 3: Implement vector-store query/delete helpers and deduplicator**

Add JSON metadata expressions in one place. Add `has_file_id`, `has_file_hash`, `delete_by_file_id`, `delete_by_file_hash`, and keep `delete_by_source` as fallback.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rag_ingestion_pipeline.py -q`

Expected: PASS for dedup tests.

## Task 4: Index Service Pipeline Orchestration

**Files:**
- Modify: `app/services/vector_index_service.py`
- Test: `tests/test_rag_ingestion_pipeline.py`

- [ ] **Step 1: Write failing orchestration tests**

```python
from langchain_core.documents import Document

from app.services.rag_ingestion_models import IngestionDecision, LoadedDocument
from app.services.vector_index_service import VectorIndexService


class FakeLoader:
    def load(self, file_path):
        return LoadedDocument(
            path=file_path,
            normalized_path=file_path,
            file_name="cpu.md",
            extension=".md",
            file_size=7,
            modified_time=1.0,
            raw_bytes=b"content",
            text="content",
            file_id="file-id",
            file_hash="file-hash",
        )


class FakeDeduplicator:
    def __init__(self, decision):
        self.decision = decision

    def prepare(self, loaded):
        return self.decision


class FakeSplitter:
    def split_document(self, content, file_path):
        return [Document(page_content="content", metadata={})]


class FakeMetadataBuilder:
    def build(self, loaded, documents, parser):
        return documents


class FakeStore:
    def __init__(self):
        self.added = []
        self.optimized = False

    def add_documents(self, docs):
        self.added.extend(docs)
        return ["id-1"]

    def optimize_collection(self):
        self.optimized = True
        return {"optimized": True}


def test_index_single_file_skips_unchanged_file(tmp_path):
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = FakeStore()
    service = VectorIndexService(
        loader=FakeLoader(),
        splitter=FakeSplitter(),
        metadata_builder=FakeMetadataBuilder(),
        deduplicator=FakeDeduplicator(IngestionDecision(action="skip", reason="unchanged")),
        vector_store=store,
    )

    outcome = service.index_single_file(str(path))

    assert outcome.skipped is True
    assert store.added == []


def test_index_single_file_adds_enriched_documents(tmp_path):
    path = tmp_path / "cpu.md"
    path.write_text("content", encoding="utf-8")
    store = FakeStore()
    service = VectorIndexService(
        loader=FakeLoader(),
        splitter=FakeSplitter(),
        metadata_builder=FakeMetadataBuilder(),
        deduplicator=FakeDeduplicator(IngestionDecision(action="insert", reason="new")),
        vector_store=store,
    )

    outcome = service.index_single_file(str(path))

    assert outcome.inserted_chunks == 1
    assert len(store.added) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_ingestion_pipeline.py -q`

Expected: FAIL because `VectorIndexService` does not accept injectable dependencies and `index_single_file` returns no outcome.

- [ ] **Step 3: Implement orchestration**

Refactor `VectorIndexService.__init__` to accept optional dependencies. `index_single_file` should return `FileIndexingOutcome`, count skipped files, and call `optimize_collection` after successful writes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rag_ingestion_pipeline.py -q`

Expected: PASS.

## Task 5: Retrieval Profiles And Local Rerank

**Files:**
- Create: `app/services/rag_retrieval_service.py`
- Modify: `app/services/vector_store_manager.py`
- Modify: `app/tools/knowledge_tool.py`
- Modify: `app/agent/aiops_multi/nodes/knowledge_agent.py`
- Test: `tests/test_rag_retrieval_service.py`

- [ ] **Step 1: Write failing tests**

```python
from langchain_core.documents import Document

from app.services.rag_retrieval_service import RagRetrievalService, RetrievalProfile


class FakeStore:
    def similarity_search_with_score(self, query, k):
        return [
            (
                Document(
                    page_content="普通说明",
                    metadata={"runbook_section": "other"},
                ),
                0.1,
            ),
            (
                Document(
                    page_content="CPUHigh 排查步骤: 查询 CPU 指标",
                    metadata={"alert_name": "CPUHigh", "runbook_section": "troubleshooting"},
                ),
                0.4,
            ),
        ]


def test_aiops_retrieval_reranks_alert_runbook_chunks_first():
    service = RagRetrievalService(vector_store=FakeStore())

    docs = service.retrieve("CPUHigh 怎么排查", profile=RetrievalProfile.AIOPS, top_k=1)

    assert docs[0].metadata["alert_name"] == "CPUHigh"


def test_unknown_profile_falls_back_to_chat():
    service = RagRetrievalService(vector_store=FakeStore())

    docs = service.retrieve("CPUHigh", profile="unknown", top_k=1)

    assert len(docs) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_rag_retrieval_service.py -q`

Expected: FAIL because `RagRetrievalService` does not exist.

- [ ] **Step 3: Implement retrieval service and tool wiring**

Add `RetrievalProfile` enum or string constants, profile defaults, keyword overlap, metadata boost, and score combination. Update `retrieve_knowledge(query, profile="chat", top_k=None)` to call the service. Update AIOps knowledge agent to call `retrieve_knowledge.ainvoke({"query": query, "profile": "aiops"})`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_rag_retrieval_service.py -q`

Expected: PASS.

## Task 6: Configuration And Milvus Index Management

**Files:**
- Modify: `app/config.py`
- Modify: `app/core/milvus_client.py`
- Modify: `app/services/vector_store_manager.py`
- Test: `tests/test_rag_ingestion_pipeline.py`

- [ ] **Step 1: Write failing config tests**

```python
from app.config import Settings


def test_rag_index_defaults_are_configurable():
    settings = Settings(_env_file=None)

    assert settings.milvus_collection_name == "biz"
    assert settings.rag_schema_version == 1
    assert settings.rag_aiops_top_k >= settings.rag_top_k
    assert settings.rag_vector_metric_type == "L2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_rag_ingestion_pipeline.py::test_rag_index_defaults_are_configurable -q`

Expected: FAIL because the new settings do not exist.

- [ ] **Step 3: Add settings and wire collection/index params**

Use config-backed collection name and index/search settings. Keep defaults identical to current behavior where possible.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_rag_ingestion_pipeline.py::test_rag_index_defaults_are_configurable -q`

Expected: PASS.

## Task 7: Final Verification

**Files:**
- All files touched above.

- [ ] **Step 1: Run focused RAG tests**

Run:

```bash
pytest tests/test_rag_ingestion_pipeline.py tests/test_document_splitter_runbook.py tests/test_rag_retrieval_service.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run existing contract tests touched by tool changes**

Run:

```bash
pytest tests/test_multi_agent_aiops_contract.py tests/test_redis_checkpoint.py -q
```

Expected: all selected existing tests pass.

- [ ] **Step 3: Run full test suite if local dependencies allow it**

Run:

```bash
pytest -q
```

Expected: full suite passes, or failures are documented if they require unavailable live services.

- [ ] **Step 4: Inspect worktree**

Run:

```bash
git status --short
```

Expected: only intended RAG files and pre-existing unrelated user changes are present.
