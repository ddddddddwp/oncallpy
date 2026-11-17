# Remove RAG Compatibility Entrypoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 删除 `VectorIndexService` / `RagRetrievalService` 旧入口层，让生产代码直接依赖 `IndexingService`、`QueryService` 和领域模型，同时保持索引与检索行为不变。

**Architecture:** `app/rag/bootstrap.py` 仍是组合根。API 和脚本通过 `get_indexing_service()` 获取索引用例；知识工具通过 `get_query_service()` 提交 `SearchQuery`，并只在 LangChain tool 边界把领域 `Document` 转为 LangChain `Document`。删除 `app/rag` 两个旧 facade 及 `app/services` 两个 module-alias shim；Milvus、SQLite 和 embedding 的底层适配器本次不动。

**Tech Stack:** Python 3.11、FastAPI、LangChain、pytest、Ruff、Pyright

---

## Task 1: 用失败测试固定新边界

**Files:**
- Modify: `tests/test_rag_architecture.py`
- Modify: `tests/test_rag_retrieval_service.py`

- [ ] 添加架构测试，要求四个旧入口文件不存在，且生产调用方不再导入旧模块。
- [ ] 改写知识工具边界测试：fake `QueryService` 接收 `SearchQuery`，工具仍返回 LangChain `Document` artifact。
- [ ] 运行定向测试并确认它们在旧实现上按预期失败。

Run:

```bash
PYTHONPATH=. /root/code/EnergyAgent/.venv/bin/pytest \
  tests/test_rag_architecture.py \
  tests/test_rag_retrieval_service.py::test_knowledge_tool_passes_profile_to_query_service -q --no-cov
```

Expected: FAIL，因为旧文件仍存在且知识工具仍调用 `rag_retrieval_service`。

## Task 2: 迁移生产调用方

**Files:**
- Modify: `app/api/file.py`
- Modify: `app/tools/knowledge_tool.py`
- Modify: `scripts/benchmark_rag_ivf_params.py`

- [ ] API 在请求处理函数中惰性调用 `get_indexing_service()`。
- [ ] 知识工具构造 `SearchQuery`，调用 `get_query_service().retrieve()`，并在工具边界转换 artifact 类型。
- [ ] benchmark 直接使用两个新用例工厂和领域查询对象。
- [ ] 运行知识工具与相关 API 测试。

## Task 3: 将测试迁到新用例

**Files:**
- Modify: `tests/test_rag_ingestion_pipeline.py`
- Modify: `tests/test_rag_retrieval_service.py`
- Modify: `tests/test_compatibility.py`

- [ ] 用测试 helper 显式组装 `IndexingService`，删除对 `VectorIndexService` 构造器的依赖。
- [ ] 用端口 fake/adapter 显式组装 `QueryService`，查询输入改为 `SearchQuery`。
- [ ] 从兼容性清单移除四个旧模块，并断言它们不可导入。
- [ ] 保留原有结果、排序、过滤、trace、generation 和失败补偿断言。

## Task 4: 删除旧入口并更新文档

**Files:**
- Delete: `app/rag/vector_index_service.py`
- Delete: `app/rag/rag_retrieval_service.py`
- Delete: `app/services/vector_index_service.py`
- Delete: `app/services/rag_retrieval_service.py`
- Modify: `app/rag/README.md`

- [ ] 删除两个 facade 和两个 module-alias shim。
- [ ] README 移除“旧兼容入口”描述，调用示例只展示新用例。
- [ ] 全库搜索，确保没有生产或测试代码引用旧模块/对象。

## Task 5: 全量验证

- [ ] 运行 RAG 定向测试。
- [ ] 运行全量 pytest（不收集覆盖率）。
- [ ] 对修改文件运行 Ruff 和 Pyright。
- [ ] 检查 diff 只包含本次兼容入口移除。

Run:

```bash
DASHSCOPE_API_KEY=sk-test PYTHONPATH=. /root/code/EnergyAgent/.venv/bin/pytest -q --no-cov
/root/code/EnergyAgent/.venv/bin/ruff check app tests scripts
/root/code/EnergyAgent/.venv/bin/pyright
```

Expected: 全部通过，且 RAG 行为断言不变。
