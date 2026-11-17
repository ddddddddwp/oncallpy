# RAG 索引与查询模块化重构设计

日期：2026-07-22

## 背景

当前 RAG 能力已经具备文件加载、文档切分、元数据增强、去重、Milvus 稠密索引、SQLite 稀疏索引、查询路由、RRF 融合和本地重排等完整流程，但模块边界仍以具体实现为中心：

- `VectorIndexService` 虽然允许构造器注入，但依赖未通过稳定契约表达，默认路径仍直接获取具体类或全局单例。
- `RagRetrievalService` 同时承担查询参数规范化、过滤推断、路由、召回、融合、重排和上下文替换，并在未注入依赖时回退到全局单例。
- 领域逻辑直接使用 LangChain `Document`，上层契约与第三方框架类型耦合。
- Milvus、SQLite 稀疏索引和 DashScope embedding 的能力边界没有按读写职责拆分。
- CodeGraph 影响分析显示，索引核心影响约 35 个符号，查询核心影响约 52 个符号，因此一次性删除旧入口的回归风险较高。

本设计采用渐进式替换：建立新的模块边界和 Protocol 端口，将现有行为等价迁移到新实现，同时保留旧模块作为兼容层。

## 目标

1. 将 RAG 明确划分为索引和查询两个业务模块。
2. 通过小型 `Protocol` 隔离业务编排与 Milvus、SQLite、DashScope、LangChain。
3. 使用项目自有领域模型表达文档、查询、命中结果和索引结果。
4. 保留现有导入路径、构造方式、全局入口和返回类型兼容性。
5. 保持索引事务语义、查询回退语义、排序结果和可观测行为不变。
6. 允许在单元测试中直接注入 fake，实现不依赖真实基础设施的测试。

## 非目标

- 不调整文档切分规则、元数据规则或去重算法。
- 不调整查询路由、过滤推断、RRF、权重、候选数量或重排公式。
- 不将同步接口改成异步接口。
- 不更换 Milvus、SQLite 或 DashScope。
- 不在本次变更中删除旧模块。
- 不以“抽象完整”为目标给所有类增加接口，只抽象业务边界和外部能力。

## 目录与职责

目标目录如下：

```text
app/rag/
├── bootstrap.py
├── indexing/
│   ├── __init__.py
│   ├── indexing_service.py
│   ├── document_loader.py
│   ├── document_splitter.py
│   ├── metadata_builder.py
│   ├── deduplicator.py
│   └── contracts.py
├── query/
│   ├── __init__.py
│   ├── query_service.py
│   ├── query_router.py
│   ├── retrieval_fusion.py
│   ├── reranker.py
│   └── contracts.py
├── ports/
│   ├── __init__.py
│   ├── vector_writer.py
│   ├── vector_searcher.py
│   ├── sparse_writer.py
│   ├── sparse_searcher.py
│   └── embedding_provider.py
├── infrastructure/
│   ├── __init__.py
│   ├── milvus_vector_store.py
│   ├── sqlite_sparse_index.py
│   └── dashscope_embeddings.py
└── models/
    ├── __init__.py
    ├── document.py
    ├── search_query.py
    ├── search_hit.py
    └── indexing_result.py
```

职责约束：

- `models` 只包含项目领域数据，不导入 LangChain、Milvus 或 DashScope。
- `ports` 只声明外部能力契约，不创建具体实现，不读取全局配置。
- `infrastructure` 实现外部系统端口，并负责领域对象与 Milvus、SQLite、DashScope
  等第三方对象之间的转换。
- `indexing` 负责索引用例和索引内部协作者。
- `query` 负责查询用例和查询内部协作者。
- `bootstrap.py` 是新实现的组合根，集中创建和连接具体依赖。
- 原有 `app/rag/*.py` 入口暂时保留，只做重导出、参数/对象转换或委托。

依赖方向必须保持：

```text
API / Agent / Scripts
        │
        ├── 旧入口兼容层
        └── 新用例入口
                 │
         indexing / query
          │             │
        models         ports
                          ▲
                          │
                  infrastructure
```

业务模块不能反向导入基础设施模块；具体实现只在组合根和兼容层中出现。

## 领域模型

### Document

`models/document.py` 定义项目自己的可变数据类，至少保留：

- `page_content: str`
- `metadata: dict[str, Any]`
- `id: str | None`

沿用 `page_content` 和 `metadata` 命名，以减少现有业务逻辑迁移成本。模型本身不继承或导入 LangChain `Document`。

### SearchQuery

`models/search_query.py` 表达一次查询请求：

- `text: str`
- `profile: str`
- `top_k: int | None`
- `metadata_filter: dict[str, Any] | None`

profile 的规范化和默认 top-k 仍由查询模块按现有规则处理。

### SearchHit

`models/search_hit.py` 表达召回和融合过程中的候选：

- `document: Document`
- `score: float`
- `source: str`，标记 dense、sparse 或 fused 来源
- 必要的可选排名/追踪字段

只迁移当前算法真正使用或已有可观测输出需要的字段，不引入新的排序语义。

### IndexingResult

`models/indexing_result.py` 保留当前全部状态：

- `success`
- `directory_path`
- `total_files`
- `success_count`
- `skipped_count`
- `fail_count`
- `deleted_count`
- `inserted_chunk_count`
- `start_time`
- `end_time`
- `error_message`
- `failed_files`

并保持现有计数方法、`get_duration_ms()` 和 `to_dict()` 输出不变。

索引流程专用的 `LoadedDocument`、`FileIndexingOutcome` 等内部数据结构保留现有字段，归入 `indexing/contracts.py` 或由该模块重导出，不扩大为跨模块公共模型。

## Protocol 边界

### 外部能力端口

`ports` 使用 `typing.Protocol`，按调用方需要拆分能力：

- `VectorWriter`：新增文档、替换文件 generation、删除和索引写入所需的查询操作。
- `VectorSearcher`：相似度搜索和元数据过滤搜索。
- `SparseWriter`：upsert、replace、delete、contains 及索引修复判断。
- `SparseSearcher`：稀疏候选搜索。
- `EmbeddingProvider`：`embed_documents` 与 `embed_query`。

端口的最终方法名、参数、返回值和异常语义机械对齐当前调用，不在本次重构中重新设计存储语义。

一个具体实现可以同时满足多个小接口：

```text
MilvusVectorStore  -> VectorWriter + VectorSearcher
SqliteSparseIndex  -> SparseWriter + SparseSearcher
DashScopeEmbeddings -> EmbeddingProvider
```

Protocol 不使用 `runtime_checkable`，除非现有运行时确实需要 `isinstance` 判断；静态结构化类型已经足够。

### 索引内部契约

`indexing/contracts.py` 定义索引用例和可替换协作者所需的最小 Protocol：

- `IndexingUseCase`
- `DocumentLoader`
- `DocumentSplitter`
- `MetadataBuilder`
- `Deduplicator`

具体类可以继续使用当前易识别的名称，例如 `RagDocumentLoader`、`DocumentSplitterService`、`RagMetadataBuilder` 和 `RagDeduplicator`，不要求显式继承 Protocol。

为保持现有切分结果，具体 `DocumentSplitterService` 可以在实现内部继续使用
LangChain 的 Markdown 和递归文本切分器，但它的公开签名只接受字符串并返回领域
`Document`。第三方 `Document` 只能作为方法内部的临时对象，不能进入 Protocol 或
其他业务服务。

### 查询内部契约

`query/contracts.py` 定义：

- `QueryUseCase`
- `QueryRouter`
- `RetrievalFusion`
- `Reranker`

`QueryService` 只负责编排；现有过滤推断和路由逻辑迁入 `query_router.py`，RRF 迁入 `retrieval_fusion.py`，本地打分与排序迁入 `reranker.py`。

## 索引流程

索引执行顺序保持不变：

```text
扫描文件
  -> 加载文档
  -> 判断是否重复或需要修复
  -> 文档切分
  -> 构建元数据和路由摘要
  -> 写入新一代稠密向量
  -> 同步稀疏索引
  -> 清理旧一代稠密向量
  -> 汇总 IndexingResult
```

必须由现有特征测试保护的事务语义：

1. splitter 或稠密写入失败时保留旧 generation。
2. 稀疏索引同步成功后才能清理旧稠密数据。
3. 文件未变化时仍可修复缺失的稀疏数据，并重试失败的旧数据清理。
4. generation 或 file-id 验证失败时，不开始不可验证的写入或错误回滚。
5. 空切分结果作为失败处理，旧 generation 保留。
6. legacy rows 在失败路径存活；成功时只删除已捕获的 legacy ids。
7. 文件路由摘要、runbook 元数据、设备分类和协议元数据保持现有内容与生成条件。

## 查询流程

查询执行顺序保持不变：

```text
SearchQuery
  -> 规范化 profile、top_k 和候选数量
  -> 接受显式过滤或推断 metadata_filter
  -> 可选的路由摘要检索
  -> 稠密召回与可选稀疏召回
  -> RRF 融合
  -> 现有本地重排
  -> 上下文窗口替换
  -> Document 列表
```

必须保持的行为：

1. 稀疏索引不可用时回退为稠密检索。
2. 路由摘要缺失时执行现有回退路径。
3. 明确分类过滤以及要求 taxonomy 匹配的查询继续 fail-closed。
4. taxonomy 查询在允许时只按现有顺序放宽过滤，不增加无过滤查询。
5. 未知 profile 回退到 chat。
6. profile 默认 top-k、候选数量、RRF 参数和所有重排权重不变。
7. 上层检索异常继续记录日志和 trace，并返回空列表。
8. trace 的名称、关键输入字段和输出字段保持不变。

## 基础设施适配器

### MilvusVectorStore

将现有 `VectorStoreManager` 的 Milvus 建连、集合管理、写入、删除、generation 查询、相似度检索和过滤能力迁入新实现。该实现接受 `EmbeddingProvider`，并在边界处把领域 `Document` 转换为 Milvus/LangChain 所需对象，再把搜索结果转换为 `SearchHit`。

### SqliteSparseIndex

将现有 `SparseIndexService` 的写入与搜索能力迁入同一个具体实现，但分别通过 `SparseWriter` 和 `SparseSearcher` 暴露。SQLite schema、tokenization、排序和持久化位置不变。

### DashScopeEmbeddings

包装当前 DashScope embedding 行为并满足 `EmbeddingProvider`。模型、批量策略、重试和异常行为保持现状。

## 组合根与依赖注入

`bootstrap.py` 是新代码唯一知道所有具体实现的位置。它负责：

1. 从现有 config 读取配置。
2. 构造 `DashScopeEmbeddings`。
3. 构造 `MilvusVectorStore` 和 `SqliteSparseIndex`。
4. 将 writer/searcher 端口注入 `IndexingService` 和 `QueryService`。
5. 提供缓存的默认服务，供兼容层使用。

新业务服务的构造器全部接受 Protocol，而不是在业务方法中导入全局单例。为了保持旧行为，旧包装类在依赖缺省时仍通过组合根取得默认实现；显式传入 fake 时必须优先使用 fake。

## 兼容策略

原有模块暂不删除，包括但不限于：

- `vector_index_service.py`
- `rag_retrieval_service.py`
- `vector_store_manager.py`
- `sparse_index_service.py`
- `vector_embedding_service.py`
- `rag_document_loader.py`
- `document_splitter_service.py`
- `rag_metadata_builder.py`
- `rag_deduplicator.py`
- `retrieval_fusion.py`

兼容层遵循以下规则：

1. 原有类名、函数名和全局实例名继续可导入。
2. 原有构造参数继续有效，尤其是测试中直接注入 fake 的方式。
3. 旧 `RagRetrievalService.retrieve(query, profile, top_k, metadata_filter)` 签名和 `list[langchain_core.documents.Document]` 返回类型保持不变。
4. 旧入口把参数转换为 `SearchQuery`，调用新 `QueryService`，再把领域 `Document` 转为 LangChain `Document`。
5. 新入口只暴露领域模型；第三方类型不穿过新业务边界。
6. 兼容层不得复制路由、融合、重排或索引事务逻辑。
7. 新旧入口必须共享同一个默认基础设施实例，避免重复连接或状态分叉。

## 错误处理与可观测性

- 基础设施异常可以在适配器层保留原异常或增加异常链，但业务层观察到的成功、失败和回退行为不变。
- 查询服务继续在顶层捕获检索异常、记录错误、完成 trace 并返回空列表。
- 稀疏能力初始化或查询失败继续记录 warning 并回退稠密检索。
- 索引服务继续通过 `IndexingResult` 累计失败；失败路径不得提前删除旧数据。
- 不在本次重构中新增静默吞错路径。
- 日志和 trace 的关键字段保持兼容，使重构前后的观测结果可比较。

## 测试策略

实施前先运行现有 RAG 定向测试，建立基线。实施过程中采用以下层次：

1. **领域模型测试**：默认值、可变元数据、`IndexingResult.to_dict()` 和耗时计算。
2. **端口契约测试**：用 fake 证明各 Protocol 可由独立实现满足，并由 Pyright 检查结构兼容性。
3. **适配器测试**：领域 Document 与 LangChain Document 转换、Milvus/SQLite/DashScope 现有行为代理。
4. **索引特征测试**：现有 ingestion 测试不改期望，验证 generation、稀疏同步、回滚和路由摘要语义。
5. **查询特征测试**：现有 retrieval 测试不改期望，验证过滤、路由、混合召回、RRF、重排、回退和 trace。
6. **兼容测试**：旧导入、旧构造器、旧全局实例和旧返回类型继续工作。
7. **新入口测试**：新服务只依赖 Protocol，可以使用纯 fake 完成索引与查询。
8. **一致性测试**：相同 fixture 通过新旧入口得到相同文档内容、元数据和排序。

最终验证顺序：

```text
RAG 定向测试
-> Ruff
-> Pyright
-> 完整 pytest
```

如完整测试依赖本地 Milvus、Redis 或其他容器，应区分环境不可用与代码回归，并在交付报告中明确记录。

## 迁移原则

迁移按依赖方向自底向上进行：领域模型、端口、基础设施、索引模块、查询模块、组合根、兼容层、调用方。每一批迁移都先增加失败测试，再写最小实现使其通过；旧实现只有在新实现及兼容测试通过后才改成薄包装。

本次变更完成时不删除兼容层。删除旧入口应作为后续独立变更，在确认仓库内外调用方均已迁移后进行。

## 验收标准

1. 目标目录和模块全部建立，职责符合本设计。
2. 索引与查询业务服务只依赖领域模型和 Protocol。
3. Milvus、SQLite 稀疏索引和 DashScope 实现位于 `infrastructure`。
4. `models`、`ports`、`IndexingService`、`QueryService`、router、fusion 和 reranker
   不导入或暴露 LangChain、Milvus、DashScope 类型；具体 splitter 仅可在内部使用
   LangChain 切分实现，并在返回前转换为领域 `Document`。
5. 旧导入路径、构造参数、全局实例和查询返回类型保持兼容。
6. 现有索引和查询测试无需改变业务期望即可通过。
7. 新增端口、依赖注入、新旧一致性和类型检查测试通过。
8. Ruff、Pyright 和可运行的完整测试集通过。
9. 本次重构不改变索引生成结果、查询排序、回退策略或可观测输出。
