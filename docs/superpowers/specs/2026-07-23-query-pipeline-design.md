# 独立 Query Pipeline 设计

状态：已确认  
日期：2026-07-23

## 1. 背景

当前 RAG 查询能力已经按 `QueryService`、`DefaultQueryRouter`、`RetrievalFusion`、`LocalReranker` 和存储端口做了初步模块化，但查询编排仍集中在 `QueryService`。Profile 归一化、Metadata 推断、route-first、Dense/Sparse 召回、过滤回退、融合、重排、上下文窗口和 tracing 共同存在于一条服务流程中。

当前实现不会显式生成独立的改写结果。Agent 可能在工具调用参数中隐式整理用户问题，但 Query 层自身主要保留原始文本，通过确定性规则推断 Metadata Filter，再使用同一文本执行一次或多次检索。这使得以下工作不够方便：

- 单独观察 Query 分析和改写结果。
- 独立修改改写 Prompt 或规则。
- 控制多 Query 的检索次数和成本。
- 在不启动 Agent、MCP 或 API 的情况下调试完整检索流程。
- 区分规则、LLM、路由、存储和重排阶段的失败与降级。

## 2. 目标

- 将 Query 建设为独立 Python 领域模块，不拆成微服务。
- 提供唯一公开入口 `QueryEngine`，供 Agent、API、CLI 和测试共同使用。
- 使用规则优先、复杂问题可选 LLM 的改写策略。
- 使用结构化 `QueryPlan` 表达分析、改写和检索计划。
- 支持有限、可预算的多 Query 检索。
- 提供不依赖 Agent 的全阶段 CLI 调试能力。
- 保持显式 Metadata Filter、严格实体过滤和现有存储降级语义。
- 通过兼容适配器渐进迁移现有 `QueryService` 和 `retrieve_knowledge`。

## 3. 非目标

- 本次不把 Query 拆成独立 HTTP 微服务。
- 本次不引入动态工作流或图执行框架。
- 本次不让 LLM 直接生成 Milvus 表达式或控制存储调用。
- 本次不允许无限子查询、无限重试或自主迭代检索。
- 本次不改变索引构建、Document、SearchHit 或存储 SDK 的领域边界。
- 本次不立即删除 `QueryService` 兼容入口。

## 4. 总体架构

```text
API / Agent / CLI
        │
        ▼
    QueryEngine
        │
        ├── QueryAnalyzer
        ├── RewritePolicy
        ├── RuleQueryRewriter
        ├── LLMQueryRewriter
        ├── QueryPlanner
        ├── RetrievalExecutor
        ├── RetrievalFusion
        ├── Reranker
        └── QueryDebugger
```

依赖方向固定为：

```text
调用方 → QueryEngine → Query contracts/models ← LLM 与存储适配器
```

边界要求：

- Query 模块不依赖 Agent、MCP、FastAPI 或具体模型厂商。
- LLM 通过 `QueryRewriteModel` Protocol 注入。
- Milvus 和 SQLite 继续通过 `VectorSearcher`、`SparseSearcher` 注入。
- CLI 和 Agent 使用同一个 `QueryEngine`，不复制查询逻辑。
- `QueryService.retrieve(SearchQuery)` 暂时作为兼容适配器，内部委托 `QueryEngine`。

## 5. 建议目录

```text
app/rag/query/
├── engine.py
├── models.py
├── contracts.py
├── analyzer.py
├── rewrite_policy.py
├── planner.py
├── executor.py
├── debugger.py
├── cli.py
├── rewriters/
│   ├── __init__.py
│   ├── rule_rewriter.py
│   └── llm_rewriter.py
├── query_router.py
├── query_service.py
├── retrieval_fusion.py
└── reranker.py
```

现有 `query_router.py` 在迁移期保留。稳定后，其字段识别逻辑进入 `QueryAnalyzer`，路由计划逻辑进入 `QueryPlanner`。

## 6. 核心领域模型

### 6.1 QueryRequest

`QueryRequest` 是所有调用方提交给 Query 层的统一请求：

- `text`：用户原始文本，始终保留。
- `profile`：`chat`、`aiops` 或 `protocol`。
- `top_k`：最终文档数量，可为空并由 profile 决定。
- `metadata_filter`：调用方显式过滤条件。
- `conversation_context`：框架无关的 `QueryTurn` 序列。
- `rewrite_mode`：`auto`、`rule`、`llm` 或 `off`。
- `debug`：是否返回完整阶段 trace。

显式 `metadata_filter` 优先级最高，规则和 LLM 均不得覆盖或放宽。

### 6.2 QueryAnalysis

`QueryAnalysis` 保存纯规则分析结果：

- `normalized_text`
- `intents`
- `entities`
- `inferred_filter`
- `complexity`
- `confidence`
- `context_dependent`
- `reasons`

分析阶段不访问 LLM、Milvus 或 SQLite，保证确定性和可单元测试性。

### 6.3 RewriteDecision

`RewriteDecision` 显式保存是否调用 LLM、触发原因和失败回退目标：

- `use_llm`
- `reasons`
- `fallback`

### 6.4 RewrittenQuery

规则改写器和 LLM 改写器返回统一结构：

- `primary_text`
- `subqueries`
- `preserved_terms`
- `source`
- `confidence`
- `reason`

子查询数量受配置上限约束。设备型号、告警名、寄存器地址和功能码等关键实体必须进入 `preserved_terms`，不得在改写中丢失。

### 6.5 QueryPlan

`QueryPlan` 是 Query 层最重要的可观测产物，包含：

- 原始 `QueryRequest`
- `QueryAnalysis`
- `RewriteDecision`
- `RewrittenQuery`
- 主查询和去重后的子查询
- 经过校验的 Metadata Filter
- route-first、hybrid 和 retrieval strategy
- `candidate_k`
- 最大子查询数和最大检索尝试数
- 超时和过滤放宽策略

`QueryPlan` 必须可序列化，不得包含 Milvus、SQLite、LangChain、LLM Client 或其他运行时对象。

### 6.6 QueryResult

`QueryResult` 包含：

- `documents`
- `plan`
- `trace`
- `degraded`
- `degradation_reasons`
- 查询状态和错误摘要

生产调用方可以只消费 `documents`；CLI、测试和可观测系统可以消费完整结果。

## 7. MetadataFieldRegistry

新增 `MetadataFieldRegistry`，替代 Router 中分散的硬编码字段列表。每个字段通过 `MetadataFieldSpec` 描述：

- 规范字段名。
- 用户语言别名。
- extractor 和 normalizer。
- 是否允许作为硬过滤条件。
- 是否允许降级放宽。
- 是否属于严格实体字段。
- 支持的 profile。

新增 Metadata 字段时，只修改 Registry 和对应 extractor，不修改 QueryEngine 主流程。

LLM 产生的 entity candidate 必须满足：

- 字段存在于 Registry。
- 值通过字段 normalizer。
- 能映射到原始 Query 或会话上下文中的证据。

无法验证的候选只能用于文本 Query，不得进入硬过滤条件。

## 8. Query 改写策略

### 8.1 规则优先

`QueryAnalyzer` 首先完成：

- 空白和标点规范化。
- Taxonomy、协议、告警和 runbook 意图识别。
- 多意图、指代、省略和复杂度判断。
- 规则置信度计算。

规范化不得改变技术词、设备型号、地址、功能码或告警名。

### 8.2 自动触发 LLM

默认 `rewrite_mode=auto`。以下情况触发 LLM：

- 规则置信度低于阈值。
- 同时存在多个意图。
- Query 依赖会话上下文或包含指代。
- Query 较长、结构复杂或包含多个问题。
- 实体识别与 profile 存在冲突。

以下情况只走规则：

- 问题单一且明确。
- 调用方传入显式 Metadata Filter。
- 型号、寄存器等关键实体已经明确且规则置信度足够。
- 调用方指定 `rule` 或 `off`。

`llm` 模式用于 CLI 和受控实验，允许强制调用 LLM。

### 8.3 受约束 LLM 输出

LLM 只允许返回结构化字段：

- `primary_text`
- `subqueries`
- `intents`
- `entity_candidates`
- `confidence`
- `reason`

LLM 不得决定存储表达式、检索次数、route-first、hybrid 权重或 Metadata 放宽策略。输出必须经过 schema、长度、数量、关键实体和 Registry 校验。

LLM 超时、异常、格式错误或校验失败时，使用规则改写结果。

## 9. QueryPlanner

`QueryPlanner` 将请求、分析和改写结果转换为声明式 `QueryPlan`。

默认策略：

- 最多一个主查询和两个子查询。
- 相同或高度相似的子查询执行前去重。
- Route Summary 只使用主查询执行一次。
- 子查询共享主查询解析出的路由和 Metadata Filter。
- 主查询权重高于子查询。
- 所有 route、fallback 和子查询共享全局检索预算。

显式 Filter 直接生成受约束计划，不进入自动推断或放宽流程。

## 10. RetrievalExecutor

`RetrievalExecutor` 只执行 `QueryPlan`，不负责理解自然语言。它承接当前 `QueryService` 中的：

- 候选搜索。
- Dense/Sparse 安全调用。
- route-first 两阶段搜索。
- Metadata fallback。
- 检索预算。
- 存储失败降级。

多 Query 执行流程：

```text
主查询 → Route Summary，一次
              ↓
         内容 Metadata Filter
              ↓
       ┌──────┼──────┐
       ↓      ↓      ↓
     主查询  子查询1  子查询2
       │      │      │
   Dense + Sparse 检索
       └──────┼──────┘
              ↓
       跨 Query 加权 RRF
              ↓
           Reranker
```

每个 Query 内先融合 Dense/Sparse 排名，再使用加权 RRF 融合主查询和子查询排名。主查询权重最高，避免子查询结果淹没原始意图。

## 11. 调试与可观测性

### 11.1 Debugger 接口

定义 `QueryDebugger` Protocol，并提供：

- `NoOpQueryDebugger`：生产默认实现。
- `ConsoleQueryDebugger`：CLI 表格或文本输出。
- `MemoryQueryDebugger`：测试断言。
- 后续可选的 observability adapter。

### 11.2 阶段事件

支持以下结构化事件：

- `request_received`
- `analysis_completed`
- `rewrite_decided`
- `llm_rewrite_started`
- `llm_rewrite_completed`
- `llm_rewrite_failed`
- `plan_created`
- `route_search_started`
- `route_search_completed`
- `retrieval_attempt_started`
- `retrieval_attempt_completed`
- `fusion_completed`
- `rerank_completed`
- `query_completed`

事件记录阶段耗时、Query 变体、Filter、通道、候选数量、分数摘要、预算消耗和降级原因。默认不记录完整文档正文、密钥或连接信息；调试预览长度受配置限制。

### 11.3 CLI

```bash
uv run python -m app.rag.query.cli analyze "问题" --profile aiops
uv run python -m app.rag.query.cli plan "问题" --rewrite-mode auto
uv run python -m app.rag.query.cli run "问题" --profile aiops --debug
uv run python -m app.rag.query.cli run "问题" --profile protocol --format json
```

`analyze` 和 `plan` 不连接 Milvus 或 SQLite。`run` 执行完整计划。

## 12. 失败和降级

| 失败位置 | 降级行为 |
| --- | --- |
| LLM 超时、异常或非法输出 | 使用规则改写 |
| LLM Metadata 非法 | 丢弃非法字段，保留文本 Query |
| Route Summary 未命中 | 进入普通 Metadata 检索 |
| Sparse 失败 | 保留 Dense 结果 |
| Dense 失败 | Sparse 可用时使用 Sparse |
| 单个子查询失败 | 保留其他 Query 结果 |
| RRF 失败 | 使用去重后的原始候选 |
| Reranker 失败 | 使用融合排序 |
| 全部检索通道失败 | 返回结构化失败结果，不向 Agent 抛出未处理异常 |

安全约束：

- 显式 Filter 不自动放宽。
- 明确设备、型号、寄存器或功能码时，不执行跨实体无过滤回退。
- 达到全局检索预算后停止后续 route、fallback 和子查询。
- LLM 和存储调用分别设置超时。
- 所有降级必须记录在 `QueryResult` 和 trace 中。

## 13. 配置

新增以下配置项：

- `rag_query_rewrite_mode`
- `rag_query_llm_enabled`
- `rag_query_llm_model`
- `rag_query_llm_timeout_seconds`
- `rag_query_rewrite_confidence_threshold`
- `rag_query_max_subqueries`
- `rag_query_max_retrieval_attempts`
- `rag_query_debug_preview_length`

上线顺序固定为 `off → rule → auto`。首次迁移以 `off` 保持当前行为，规则模式稳定后再启用自动 LLM。

## 14. 迁移方案

### 阶段一：领域模型和 Protocol

新增模型与接口，不改变检索行为。使用现有测试锁定行为基线。

### 阶段二：Analyzer 和 Planner

抽离 Taxonomy、协议、profile、top-k、route、hybrid 和 fallback 决策。`rewrite_mode=off` 生成与当前行为一致的计划。

### 阶段三：RetrievalExecutor

迁移当前 `QueryService` 内部搜索方法。`QueryService.retrieve()` 转换 `SearchQuery` 为 `QueryRequest`，委托 `QueryEngine` 并返回 documents。

### 阶段四：CLI 和 Trace

加入 analyze、plan、run、JSON 输出和全阶段调试能力。

### 阶段五：LLM Rewrite

先在 CLI 和测试中启用，随后按配置从 rule 逐步切换到 auto。旧实现只有在行为、性能和质量门禁全部通过后才能删除。

## 15. 测试策略

### 15.1 单元测试

- Analyzer 的实体、意图、复杂度和置信度。
- RewritePolicy 触发条件。
- LLM schema 校验与规则降级。
- Planner 的预算、Filter 权限和 Query 去重。
- 关键实体保留。
- Registry 字段白名单和放宽策略。

### 15.2 集成测试

- Fake LLM、Fake Dense 和 Fake Sparse。
- Route-first 命中与失败。
- Dense/Sparse 单边故障。
- 多 Query 融合。
- LLM 超时和非法输出。
- 显式 Filter 不被覆盖。
- 达到检索预算后停止。

### 15.3 回归测试

- 保留现有 QueryService 测试。
- `rewrite_mode=off` 下保持当前输入输出行为。
- `retrieve_knowledge` 输出契约不变。
- Agent、API 和 bootstrap 的公开入口不发生破坏性变化。

### 15.4 质量评估

建立覆盖 chat、aiops 和 protocol 的领域查询集，持续比较旧版与新版：

- Recall@K、MRR 或 nDCG。
- 无结果率。
- Route 命中率。
- LLM 调用和降级比例。
- 平均存储查询次数。
- P50/P95 延迟。
- 错设备、错型号和错协议的越界召回率。

## 16. 验收标准

- Agent、API 和 CLI 均通过同一个 `QueryEngine` 查询。
- `analyze` 和 `plan` 可在不连接存储的情况下运行。
- 每次查询都能生成可序列化 `QueryPlan`。
- `rewrite_mode=off` 通过现有 Query 回归测试。
- 简单查询不调用 LLM。
- 复杂查询最多产生配置允许数量的子查询。
- Route Summary 每个 QueryPlan 最多执行一次。
- 所有检索尝试受到统一预算约束。
- 显式 Filter 和严格实体过滤安全语义不回退。
- LLM、Dense、Sparse、Fusion 或 Reranker 单点失败均有明确降级结果和 trace。
- CLI 能展示分析、改写、计划、每次检索、融合和重排阶段。

## 17. 已确认决策

- 采用独立 Python 模块，不采用微服务。
- 采用分阶段 Query Pipeline，不采用动态 Query Graph。
- 采用规则优先、复杂查询可选 LLM。
- 采用自动 LLM 触发策略。
- 采用结构化 `QueryPlan`。
- 采用受预算约束的多 Query 检索。
- 子查询共享主查询的一次 route-first 结果。
- 提供全阶段 CLI 调试，不在首期建设持久化实验回放平台。
