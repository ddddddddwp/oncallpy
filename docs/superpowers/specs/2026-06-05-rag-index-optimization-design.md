# RAG Index Optimization Design

## Goal

Strengthen the existing RAG ingestion and retrieval path without adding external rerank dependencies. The system should skip unchanged files, write stable metadata, delete by stable file identifiers, preserve operations-runbook structure in chunks, and improve retrieval quality through larger recall plus local keyword filtering and reranking.

## Context

The current ingestion path is:

```text
VectorIndexService.index_single_file
  -> read file text
  -> delete old Milvus rows by metadata["_source"]
  -> DocumentSplitterService.split_document
  -> VectorStoreManager.add_documents
```

The current retrieval path is:

```text
retrieve_knowledge
  -> vector_store.as_retriever(search_kwargs={"k": config.rag_top_k})
```

This works for basic `.txt` and `.md` files, but it has five practical weaknesses:

- unchanged files are re-embedded and rewritten;
- deletes depend on `_source`, which is path-sensitive and less stable than a file identifier;
- Markdown chunks do not explicitly preserve alert, troubleshooting, and resolution structure;
- `biz` is hard-coded as a collection name with no schema-version configuration;
- retrieval uses only vector similarity with a single low default `top_k`.

## Chosen Approach

Implement a conservative, dependency-free RAG enhancement.

The public API remains stable:

- file upload keeps auto-indexing after upload;
- `POST /index_directory` keeps returning the existing result envelope;
- `retrieve_knowledge(query)` remains the LangChain tool exposed to chat and AIOps;
- Milvus continues to support the existing `biz` collection by default.

Internally, ingestion becomes an explicit pipeline:

```text
Document Loader
  -> Parser
  -> Chunker
  -> Metadata Builder
  -> Deduplicator
  -> Embedding Batch
  -> Vector Writer
  -> Index Optimizer
```

The first implementation should keep this as focused Python services rather than a heavy framework. The pipeline boundary is valuable because each stage can be tested without live DashScope or Milvus.

## Module Boundaries

### `app/services/rag_ingestion_models.py`

Defines small dataclasses or Pydantic models used between ingestion stages:

- `LoadedDocument`: file path, normalized path, raw content bytes/text, file stats, `file_id`, `file_hash`, extension.
- `ParsedDocument`: normalized path, text content, parser type, base metadata.
- `IngestionDecision`: whether to skip, replace, or insert and the reason.
- `IngestionStats`: counts for indexed, skipped, deleted, inserted chunks, and optimization status.

These models keep the pipeline explicit without forcing Milvus or LangChain imports into every step.

### `app/services/rag_document_loader.py`

Owns file loading and stable identity:

- reads `.txt` and `.md` as UTF-8 text;
- computes `file_hash` as SHA-256 over file bytes;
- computes `file_id` as SHA-256 over the normalized absolute path;
- records `_source`, `_file_name`, `_extension`, `_file_size`, `_modified_time`, and parser hints.

The hash uses bytes, not text, so line ending and encoding changes are reflected accurately.

### `app/services/document_splitter_service.py`

Keeps the current LangChain splitters but gains an operations-aware Markdown path.

For Markdown documents, the service should preserve metadata for common runbook structures:

- `alert_name`: from headings or labels such as `告警名称`, `告警名`, `Alert`;
- `runbook_section`: one of `alert`, `symptom`, `troubleshooting`, `resolution`, `rollback`, `verification`, `other`;
- `h1`, `h2`, and existing heading metadata;
- `chunk_index` and `chunk_count`.

Chunk content should retain headings and step lists. The splitter can still fall back to recursive character splitting when a section is too large.

### `app/services/rag_metadata_builder.py`

Adds stable chunk metadata before write:

- `_schema_version`;
- `_collection_name`;
- `_source`;
- `_file_id`;
- `_file_hash`;
- `_chunk_id`;
- `_chunk_hash`;
- `_chunk_index`;
- `_chunk_count`;
- `_content_length`;
- `_parser`;
- `_indexed_at`;
- `_file_name`;
- `_extension`;
- runbook fields copied from splitter metadata.

`_chunk_id` should be deterministic: `sha256(file_id + ":" + chunk_index + ":" + chunk_hash)`. That makes re-indexed unchanged content stable and makes future upsert behavior possible.

### `app/services/rag_deduplicator.py`

Owns incremental indexing decisions:

- if Milvus already has rows for the same `_file_id` and `_file_hash`, skip the file;
- if the same `_file_id` exists with a different hash, delete old rows by `_file_id` before inserting;
- if only legacy `_source` rows exist, delete by `_source` as a fallback during the transition.

The first pass can query Milvus JSON metadata fields through the existing `Collection.query` API. It should degrade to reindex on query errors rather than silently dropping new content.

### `app/services/vector_store_manager.py`

Remains the Milvus write boundary.

Add methods:

- `delete_by_file_id(file_id: str) -> int`;
- `delete_by_file_hash(file_hash: str) -> int`;
- `has_file_hash(file_id: str, file_hash: str) -> bool`;
- `optimize_collection() -> dict[str, Any]`;
- `similarity_search_with_profile(query, profile)`.

Keep `delete_by_source` as a legacy fallback for current data.

### `app/core/milvus_client.py`

Add schema and collection version configuration while keeping `biz` as the default.

Configuration fields:

- `milvus_collection_name: str = "biz"`;
- `rag_schema_version: int = 1`;
- `rag_vector_metric_type: str = "L2"`;
- `rag_vector_index_type: str = "IVF_FLAT"`;
- `rag_vector_nlist: int = 128`;
- `rag_vector_nprobe: int = 10`.

The physical Milvus schema can remain `id`, `vector`, `content`, `metadata` for compatibility. Schema versioning is stored in metadata and configuration so the next migration can be deliberate instead of implicit.

### `app/services/rag_retrieval_service.py`

Owns local retrieval policy:

- choose a retrieval profile: `chat` or `aiops`;
- call vector search with a larger candidate count than final `top_k`;
- apply optional keyword filtering;
- rerank candidates locally by combining vector distance, keyword overlap, and runbook metadata boosts;
- return final documents and scores.

Default profiles:

- `chat`: final `top_k=config.rag_top_k`, candidate count near `max(top_k * 2, 6)`;
- `aiops`: final `top_k=config.rag_aiops_top_k`, candidate count near `max(top_k * 3, 15)`, keyword/runbook boost enabled.

No external rerank model is introduced in this change.

## Retrieval Scoring

Milvus currently uses L2 distance, where smaller is better. Local rerank should normalize that into a higher-is-better score:

```text
vector_score = 1 / (1 + distance)
final_score =
  vector_score * 0.70
  + keyword_score * 0.20
  + metadata_boost * 0.10
```

`keyword_score` is based on token overlap between the query and chunk content/title/alert metadata. For Chinese text, use a simple character n-gram or substring strategy rather than adding a tokenizer dependency.

`metadata_boost` rewards chunks that look operationally useful for an AIOps query:

- alert-name match;
- section is `troubleshooting`, `resolution`, or `verification`;
- heading contains query keywords;
- metadata has an explicit `alert_name`.

The scoring constants should live in code as named defaults so later tuning is localized.

## Error Handling

Ingestion should fail per file, not per directory:

- unreadable or unsupported files are counted as failed;
- unchanged files are counted as skipped, not failed;
- delete failure before replacement fails that file, because writing duplicate chunks would corrupt retrieval;
- collection optimization failure is recorded in stats but should not roll back successfully inserted chunks.

Retrieval should degrade safely:

- if vector search fails, return no documents and log the error;
- if local rerank fails for one candidate, keep that candidate with vector-only score;
- if profile selection receives an unknown value, fall back to `chat`.

## Testing Strategy

Use TDD with tests that avoid live DashScope and Milvus.

Focused tests should cover:

- loader computes stable `file_id` and byte-based `file_hash`;
- metadata builder writes `_file_id`, `_file_hash`, `_chunk_id`, `_schema_version`, and runbook fields;
- deduplicator skips unchanged files and replaces changed files;
- Markdown splitter preserves alert/runbook metadata for operations documents;
- retrieval reranker prioritizes exact alert and resolution/troubleshooting chunks over plain vector order;
- `retrieve_knowledge` uses the chat profile by default and allows AIOps callers to request the AIOps profile;
- vector store delete expressions target `_file_id` or `_file_hash`, with `_source` retained only as fallback.

Integration tests requiring live Milvus or DashScope are out of scope for this pass.

## Migration And Compatibility

Existing rows only have metadata such as `_source`, `_extension`, and `_file_name`. New logic must tolerate that.

During reindex:

1. compute new `file_id` and `file_hash`;
2. query for rows with matching `_file_id` and `_file_hash`;
3. if unchanged rows exist, skip;
4. if rows with matching `_file_id` but different hash exist, delete by `_file_id`;
5. if no new-metadata rows exist, delete legacy rows by `_source`;
6. insert new chunks with complete metadata.

This lets the current `biz` collection evolve in place.

## Out Of Scope

- Adding an external rerank model or rerank API key.
- Migrating data into multiple physical collections.
- Supporting PDF, DOCX, HTML, or binary documents.
- Adding a web UI for index administration.
- Replacing LangChain Milvus with direct `pymilvus` inserts.

## Risks

- Milvus JSON query syntax can vary by version. Keep query/delete expressions isolated in `VectorStoreManager`.
- Local keyword scoring for Chinese is approximate without a tokenizer. It is still useful for alert names and runbook section labels.
- Stable chunk IDs depend on deterministic chunk order. The metadata builder should assign chunk indexes after final splitting and merging.
- Collection optimization may be limited by Milvus version and permissions. Treat optimization as best-effort with explicit status reporting.
