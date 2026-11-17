# Independent Query Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independently callable and debuggable QueryEngine with rule-first analysis, optional schema-constrained LLM rewriting, bounded multi-query retrieval, structured plans, and compatibility with the existing RAG tool and QueryService API.

**Architecture:** Introduce immutable query-domain models and a staged pipeline: analyze → decide rewrite → rewrite → plan → execute → fuse/rerank → return. QueryEngine is the public facade; external LLM and storage dependencies remain behind Protocols, while QueryService becomes a compatibility adapter and the CLI calls the same engine.

**Tech Stack:** Python 3.11-compatible syntax, dataclasses, Pydantic v2, LangChain ChatOpenAI structured output, Milvus/SQLite ports, pytest, Ruff, Pyright, uv.

---

## File Structure

### New production files

- `app/rag/query/models.py`: Query request, analysis, rewrite, plan, trace, execution, and result dataclasses.
- `app/rag/query/metadata_registry.py`: Filterable Metadata field definitions, validation, strictness, and relaxation policy.
- `app/rag/query/analyzer.py`: Deterministic text normalization, intent/entity extraction, complexity, and confidence.
- `app/rag/query/rewrite_policy.py`: `off|rule|auto|llm` decision logic.
- `app/rag/query/rewriters/__init__.py`: Rewriter exports.
- `app/rag/query/rewriters/rule_rewriter.py`: Deterministic fallback rewrite.
- `app/rag/query/rewriters/llm_rewriter.py`: Structured LLM output validation and protected-term enforcement.
- `app/rag/query/planner.py`: Query deduplication, route/hybrid policy, candidate sizing, and retrieval budget.
- `app/rag/query/debugger.py`: No-op, memory, and console debugger implementations.
- `app/rag/query/executor.py`: Route-first, Dense/Sparse, fallback, budget, and cross-query execution.
- `app/rag/query/engine.py`: Pipeline facade and final result assembly.
- `app/rag/query/cli.py`: `analyze`, `plan`, and `run` commands.
- `app/rag/infrastructure/langchain_query_rewrite_model.py`: LangChain/OpenAI-compatible implementation of `QueryRewriteModel`.

### Modified production files

- `app/rag/query/contracts.py`: Add analyzer, policy, rewriter, model, planner, executor, debugger, and multi-query fusion Protocols.
- `app/rag/query/retrieval_fusion.py`: Add weighted cross-query RRF while preserving existing Dense/Sparse RRF.
- `app/rag/query/query_service.py`: Replace orchestration with a compatibility facade over QueryEngine.
- `app/rag/query/__init__.py`: Export the stable Query API.
- `app/rag/bootstrap.py`: Lazily assemble and cache QueryEngine and the compatibility QueryService.
- `app/config.py`: Add query rewrite, timeout, subquery, budget, and debug settings.
- `app/core/llm_factory.py`: Pass an optional request timeout to ChatOpenAI.
- `app/tools/knowledge_tool.py`: Continue the public tool contract while resolving the shared QueryEngine path.
- `app/rag/README.md`: Document the new Query pipeline and CLI.

### New test files

- `tests/test_query_pipeline_models.py`
- `tests/test_query_analyzer.py`
- `tests/test_query_rewriters.py`
- `tests/test_query_planner.py`
- `tests/test_query_debugger.py`
- `tests/test_query_executor.py`
- `tests/test_query_engine.py`
- `tests/test_query_cli.py`

### Existing tests extended

- `tests/test_rag_query_components.py`
- `tests/test_rag_retrieval_service.py`
- `tests/test_rag_bootstrap.py`
- `tests/test_llm_factory_and_observability.py`
- `tests/test_layered_config.py`

## Execution Preconditions

- The current working tree contains existing user-owned changes, including comment-only edits in `app/rag`. Preserve them.
- At execution time use `superpowers:using-git-worktrees` if a clean isolated branch can be integrated without losing those comments. Otherwise execute inline and stage only task-owned hunks.
- Never reset, restore, or discard the existing working tree.
- Before each commit inspect `git diff --cached --name-status`; commit only files and hunks owned by the current task.

### Task 1: Query Domain Models and Public Contracts

**Files:**
- Create: `app/rag/query/models.py`
- Modify: `app/rag/query/contracts.py`
- Modify: `app/rag/query/__init__.py`
- Create: `tests/test_query_pipeline_models.py`

- [ ] **Step 1: Write failing model tests**

```python
from dataclasses import asdict

from app.rag.models import RetrievalProfile, SearchQuery
from app.rag.query.models import QueryRequest, RewriteMode


def test_query_request_copies_explicit_filter_from_search_query() -> None:
    original_filter = {"device_type": "inverter"}
    request = QueryRequest.from_search_query(
        SearchQuery(
            text="逆变器怎么排查？",
            profile=RetrievalProfile.AIOPS,
            top_k=5,
            metadata_filter=original_filter,
        )
    )

    original_filter["manufacturer"] = "huawei"

    assert request.text == "逆变器怎么排查？"
    assert request.profile == RetrievalProfile.AIOPS
    assert request.top_k == 5
    assert request.metadata_filter == {"device_type": "inverter"}
    assert request.rewrite_mode == RewriteMode.AUTO


def test_query_request_and_nested_values_are_serializable() -> None:
    request = QueryRequest(text="寄存器 40001 是什么？", profile="protocol")

    payload = asdict(request)

    assert payload["text"] == "寄存器 40001 是什么？"
    assert payload["conversation_context"] == ()
```

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `uv run pytest tests/test_query_pipeline_models.py -q`

Expected: FAIL during collection with `ModuleNotFoundError: app.rag.query.models`.

- [ ] **Step 3: Add the complete query-domain model set**

Implement `app/rag/query/models.py` with these exact public types and defaults:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from app.rag.models import Document, RetrievalProfile, SearchHit, SearchQuery


class RewriteMode(StrEnum):
    OFF = "off"
    RULE = "rule"
    AUTO = "auto"
    LLM = "llm"


@dataclass(frozen=True, slots=True)
class QueryTurn:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class QueryRequest:
    text: str
    profile: RetrievalProfile | str = RetrievalProfile.CHAT
    top_k: int | None = None
    metadata_filter: dict[str, Any] | None = None
    conversation_context: tuple[QueryTurn, ...] = ()
    rewrite_mode: RewriteMode | str = RewriteMode.AUTO
    debug: bool = False

    @classmethod
    def from_search_query(cls, query: SearchQuery) -> QueryRequest:
        return cls(
            text=query.text,
            profile=query.profile,
            top_k=query.top_k,
            metadata_filter=dict(query.metadata_filter) if query.metadata_filter else None,
        )


@dataclass(frozen=True, slots=True)
class EntityCandidate:
    field: str
    value: str
    evidence: str


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    normalized_text: str
    profile: RetrievalProfile
    intents: tuple[str, ...] = ()
    entities: dict[str, str] = field(default_factory=dict)
    inferred_filter: dict[str, Any] = field(default_factory=dict)
    complexity: Literal["simple", "compound", "contextual"] = "simple"
    confidence: float = 1.0
    context_dependent: bool = False
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RewriteDecision:
    use_llm: bool
    reasons: tuple[str, ...] = ()
    fallback: Literal["rule"] = "rule"


@dataclass(frozen=True, slots=True)
class RewrittenQuery:
    primary_text: str
    subqueries: tuple[str, ...] = ()
    preserved_terms: tuple[str, ...] = ()
    entity_candidates: tuple[EntityCandidate, ...] = ()
    source: Literal["off", "rule", "llm", "fallback"] = "rule"
    confidence: float = 1.0
    reason: str = ""


@dataclass(frozen=True, slots=True)
class QueryPlan:
    request: QueryRequest
    analysis: QueryAnalysis
    decision: RewriteDecision
    rewrite: RewrittenQuery
    queries: tuple[str, ...]
    metadata_filter: dict[str, Any] | None
    final_top_k: int
    candidate_k: int
    route_first: bool
    hybrid: bool
    rrf_k: int
    max_attempts: int


@dataclass(frozen=True, slots=True)
class QueryTraceEvent:
    stage: str
    status: Literal["started", "completed", "failed", "degraded"]
    duration_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryExecution:
    candidates: list[SearchHit] = field(default_factory=list)
    degraded: bool = False
    degradation_reasons: list[str] = field(default_factory=list)
    attempts_used: int = 0


@dataclass(slots=True)
class QueryResult:
    documents: list[Document]
    plan: QueryPlan
    trace: tuple[QueryTraceEvent, ...] = ()
    degraded: bool = False
    degradation_reasons: tuple[str, ...] = ()
    error: str | None = None
```

Extend `contracts.py` with Protocols whose method signatures consume these models: `QueryAnalyzer.analyze`, `RewritePolicy.decide`, `QueryRewriter.rewrite`, `QueryRewriteModel.generate`, `QueryPlanner.plan`, `QueryExecutor.execute`, `QueryDebugger.emit`, and `MultiQueryFusion.fuse_ranked`.

- [ ] **Step 4: Export the stable model names and run tests**

Export `QueryRequest`, `QueryPlan`, `QueryResult`, `RewriteMode`, and `QueryEngine`-independent model names from `app/rag/query/__init__.py`.

Run: `uv run pytest tests/test_query_pipeline_models.py -q`

Expected: `2 passed`.

- [ ] **Step 5: Commit only Task 1 changes**

```bash
git add app/rag/query/models.py app/rag/query/contracts.py app/rag/query/__init__.py tests/test_query_pipeline_models.py
git diff --cached --name-status
git commit -m "feat(query): add pipeline domain models"
```

### Task 2: Metadata Registry and Deterministic Analyzer

**Files:**
- Create: `app/rag/query/metadata_registry.py`
- Create: `app/rag/query/analyzer.py`
- Create: `tests/test_query_analyzer.py`
- Modify: `tests/test_rag_query_components.py`

- [ ] **Step 1: Write failing analyzer and registry tests**

```python
from app.rag.query.analyzer import DefaultQueryAnalyzer
from app.rag.query.metadata_registry import default_metadata_registry
from app.rag.query.models import QueryRequest


def test_analyzer_extracts_compound_aiops_intents_and_taxonomy() -> None:
    analysis = DefaultQueryAnalyzer().analyze(
        QueryRequest(text="华为逆变器告警怎么排查、处理并验证？", profile="aiops")
    )

    assert analysis.intents == ("troubleshooting", "resolution", "verification")
    assert analysis.inferred_filter["document_type"] == "runbook"
    assert analysis.inferred_filter["device_type"] == "inverter"
    assert analysis.inferred_filter["manufacturer"] == "huawei"
    assert analysis.complexity == "compound"


def test_analyzer_marks_contextual_reference() -> None:
    analysis = DefaultQueryAnalyzer().analyze(
        QueryRequest(text="它处理完以后怎么验证？", profile="aiops")
    )

    assert analysis.context_dependent is True
    assert analysis.complexity == "contextual"


def test_registry_relaxes_only_manufacturer_and_protects_strict_fields() -> None:
    registry = default_metadata_registry()
    filters = registry.fallbacks(
        {"device_type": "inverter", "manufacturer": "huawei", "register_address": "40001"}
    )

    assert filters == [
        {"device_type": "inverter", "manufacturer": "huawei", "register_address": "40001"},
        {"device_type": "inverter", "register_address": "40001"},
    ]
    assert registry.requires_strict_match(filters[0]) is True
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/test_query_analyzer.py -q`

Expected: FAIL with missing `analyzer` and `metadata_registry` modules.

- [ ] **Step 3: Implement MetadataFieldRegistry**

Create a frozen `MetadataFieldSpec` with `name`, `aliases`, `filterable`, `relaxable`, `strict`, and `profiles`. Implement `MetadataFieldRegistry.validate_filter`, `validate_candidate`, `fallbacks`, and `requires_strict_match`.

The default registry must include these filterable fields:

```python
DEFAULT_FIELD_SPECS = (
    MetadataFieldSpec("document_type"),
    MetadataFieldSpec("runbook_section"),
    MetadataFieldSpec("alert_name", strict=True),
    MetadataFieldSpec("business_domain"),
    MetadataFieldSpec("device_type", strict=True),
    MetadataFieldSpec("manufacturer", relaxable=True),
    MetadataFieldSpec("model", strict=True),
    MetadataFieldSpec("protocol_type"),
    MetadataFieldSpec("register_address", strict=True),
    MetadataFieldSpec("register_name"),
    MetadataFieldSpec("function_code", strict=True),
    MetadataFieldSpec("data_type"),
    MetadataFieldSpec("unit"),
    MetadataFieldSpec("scale"),
    MetadataFieldSpec("read_write"),
    MetadataFieldSpec("route_key"),
    MetadataFieldSpec("_source"),
)
```

`validate_candidate` must reject unknown fields, empty evidence, and evidence absent from the original text plus serialized conversation context.

- [ ] **Step 4: Implement DefaultQueryAnalyzer**

Reuse `infer_taxonomy_from_text` and `infer_protocol_metadata_from_text`. Normalize with `" ".join(text.split())`; infer runbook intents using the existing keyword groups; normalize unknown profiles to `chat`; compute `context_dependent` from the explicit pronoun set `("它", "这个", "那个", "刚才", "上一个")`.

Confidence rules must be deterministic:

```python
confidence = 0.9
reasons: list[str] = []
if context_dependent:
    confidence -= 0.35
    reasons.append("context_reference")
if len(intents) > 1:
    confidence -= 0.2
    reasons.append("multiple_intents")
if not inferred_filter and len(normalized_text) > 80:
    confidence -= 0.2
    reasons.append("long_unstructured_query")
confidence = max(0.0, min(confidence, 1.0))
```

For `aiops`, add `document_type=runbook` and one or multiple `runbook_section` values. For `protocol`, add `document_type=protocol` and protocol metadata. For `chat`, use taxonomy only.

- [ ] **Step 5: Preserve existing router behavior and run tests**

Refactor `DefaultQueryRouter` to delegate inferred-filter and fallback decisions to `DefaultQueryAnalyzer` and `MetadataFieldRegistry`, while keeping its public Protocol unchanged during migration.

Run:

```bash
uv run pytest tests/test_query_analyzer.py tests/test_rag_query_components.py -q
```

Expected: all tests pass, including the existing manufacturer-only relaxation assertions.

- [ ] **Step 6: Commit Task 2**

```bash
git add app/rag/query/metadata_registry.py app/rag/query/analyzer.py app/rag/query/query_router.py tests/test_query_analyzer.py tests/test_rag_query_components.py
git diff --cached --name-status
git commit -m "feat(query): extract deterministic query analysis"
```

### Task 3: Rewrite Policy and Rule/LLM Rewriters

**Files:**
- Create: `app/rag/query/rewrite_policy.py`
- Create: `app/rag/query/rewriters/__init__.py`
- Create: `app/rag/query/rewriters/rule_rewriter.py`
- Create: `app/rag/query/rewriters/llm_rewriter.py`
- Modify: `app/rag/query/contracts.py`
- Create: `tests/test_query_rewriters.py`

- [ ] **Step 1: Write failing rewrite-policy tests**

```python
import pytest

from app.rag.models import RetrievalProfile
from app.rag.query.models import QueryAnalysis, QueryRequest, RewriteMode
from app.rag.query.rewrite_policy import DefaultRewritePolicy, RewritePolicySettings
from app.rag.query.rewriters.llm_rewriter import LLMQueryRewriter, QueryRewriteValidationError
from app.rag.query.rewriters.rule_rewriter import RuleQueryRewriter


def analysis(**overrides):
    values = {
        "normalized_text": "逆变器怎么排查？",
        "profile": RetrievalProfile.AIOPS,
        "confidence": 0.9,
    }
    values.update(overrides)
    return QueryAnalysis(**values)


def test_auto_policy_uses_rules_for_simple_high_confidence_query() -> None:
    decision = DefaultRewritePolicy(RewritePolicySettings(llm_enabled=True)).decide(
        QueryRequest(text="逆变器怎么排查？"), analysis()
    )
    assert decision.use_llm is False


def test_auto_policy_uses_llm_for_multiple_intents() -> None:
    decision = DefaultRewritePolicy(RewritePolicySettings(llm_enabled=True)).decide(
        QueryRequest(text="排查并验证", rewrite_mode=RewriteMode.AUTO),
        analysis(intents=("troubleshooting", "verification"), confidence=0.7),
    )
    assert decision.use_llm is True
    assert "multiple_intents" in decision.reasons


def test_explicit_filter_does_not_auto_call_llm() -> None:
    decision = DefaultRewritePolicy(RewritePolicySettings(llm_enabled=True)).decide(
        QueryRequest(text="排查", metadata_filter={"device_type": "inverter"}),
        analysis(confidence=0.2),
    )
    assert decision.use_llm is False


def test_rule_rewriter_preserves_entities_without_generating_subqueries() -> None:
    rewrite = RuleQueryRewriter().rewrite(
        QueryRequest(text="华为逆变器怎么排查？"),
        analysis(entities={"manufacturer": "huawei", "device_type": "inverter"}),
    )
    assert rewrite.primary_text == "逆变器怎么排查？"
    assert rewrite.subqueries == ()
    assert rewrite.preserved_terms == ("huawei", "inverter")


def test_llm_rewriter_rejects_output_that_drops_protected_term() -> None:
    class FakeModel:
        def generate(self, payload):
            return {
                "primary_text": "告警怎么处理",
                "subqueries": [],
                "entity_candidates": [],
                "confidence": 0.9,
                "reason": "rewrite",
            }

    with pytest.raises(QueryRewriteValidationError, match="protected term"):
        LLMQueryRewriter(FakeModel(), max_subqueries=2).rewrite(
            QueryRequest(text="SUN2000 告警怎么处理"),
            analysis(entities={"model": "SUN2000"}),
        )
```

- [ ] **Step 2: Verify the tests fail**

Run: `uv run pytest tests/test_query_rewriters.py -q`

Expected: FAIL because the policy and rewriter modules do not exist.

- [ ] **Step 3: Implement RewritePolicy**

Add frozen `RewritePolicySettings(llm_enabled: bool, confidence_threshold: float = 0.75)`. Normalize `rewrite_mode` safely. Apply this order:

1. `off` and `rule` return `use_llm=False`.
2. Explicit Metadata Filter returns `use_llm=False` in `auto`.
3. `llm` returns the configured `llm_enabled` value and records `llm_disabled` if unavailable.
4. `auto` uses LLM when enabled and any of: confidence below threshold, multiple intents, contextual query, or compound complexity.

- [ ] **Step 4: Implement the rewriters**

`RuleQueryRewriter` returns normalized text, no subqueries, sorted unique entity values as protected terms, and source `off` or `rule` based on request mode.

`LLMQueryRewriter` sends this bounded payload through `QueryRewriteModel.generate`:

```python
payload = {
    "text": request.text,
    "profile": analysis.profile.value,
    "normalized_text": analysis.normalized_text,
    "intents": list(analysis.intents),
    "entities": dict(analysis.entities),
    "conversation_context": [
        {"role": turn.role, "content": turn.content[-1000:]}
        for turn in request.conversation_context[-6:]
    ],
    "max_subqueries": self.max_subqueries,
}
```

Validate non-empty primary text, maximum subquery count, 500-character limit per query, deduplicate exact subqueries, validate entity candidate keys, and require every protected term to occur case-insensitively in primary text or a subquery. Raise `QueryRewriteValidationError` for violations.

- [ ] **Step 5: Run tests and commit**

Run: `uv run pytest tests/test_query_rewriters.py -q`

Expected: all tests pass.

```bash
git add app/rag/query/rewrite_policy.py app/rag/query/rewriters app/rag/query/contracts.py tests/test_query_rewriters.py
git diff --cached --name-status
git commit -m "feat(query): add bounded query rewriting"
```

### Task 4: Declarative Query Planner

**Files:**
- Create: `app/rag/query/planner.py`
- Create: `tests/test_query_planner.py`

- [ ] **Step 1: Write failing planner tests**

```python
from app.rag.models import RetrievalProfile
from app.rag.query.models import (
    QueryAnalysis,
    QueryRequest,
    RewriteDecision,
    RewrittenQuery,
)
from app.rag.query.planner import DefaultQueryPlanner, QueryPlannerSettings


def make_analysis() -> QueryAnalysis:
    return QueryAnalysis(
        normalized_text="华为逆变器告警怎么排查并验证？",
        profile=RetrievalProfile.AIOPS,
        intents=("troubleshooting", "verification"),
        inferred_filter={"document_type": "runbook", "manufacturer": "huawei"},
    )


def test_planner_deduplicates_and_bounds_queries() -> None:
    planner = DefaultQueryPlanner(QueryPlannerSettings(max_subqueries=2, max_attempts=7))
    plan = planner.plan(
        QueryRequest(text="原始问题", profile="aiops", top_k=5),
        make_analysis(),
        RewriteDecision(use_llm=True),
        RewrittenQuery(
            primary_text="主查询",
            subqueries=("子查询", "子查询", "第二子查询", "第三子查询"),
            source="llm",
        ),
    )

    assert plan.queries == ("主查询", "子查询", "第二子查询")
    assert plan.final_top_k == 5
    assert plan.max_attempts == 7
    assert plan.route_first is True


def test_explicit_filter_disables_route_and_is_copied() -> None:
    explicit_filter = {"model": "SUN2000"}
    plan = DefaultQueryPlanner(QueryPlannerSettings()).plan(
        QueryRequest(text="状态", metadata_filter=explicit_filter),
        make_analysis(),
        RewriteDecision(use_llm=False),
        RewrittenQuery(primary_text="状态", source="rule"),
    )

    explicit_filter["manufacturer"] = "huawei"
    assert plan.metadata_filter == {"model": "SUN2000"}
    assert plan.route_first is False
```

- [ ] **Step 2: Verify failure and implement settings**

Run: `uv run pytest tests/test_query_planner.py -q`

Expected: FAIL because `planner.py` does not exist.

Add `QueryPlannerSettings` with exact defaults:

```python
@dataclass(frozen=True, slots=True)
class QueryPlannerSettings:
    top_k: int = 3
    aiops_top_k: int = 8
    route_enabled: bool = True
    hybrid_enabled: bool = True
    hybrid_profiles: tuple[RetrievalProfile, ...] = (
        RetrievalProfile.AIOPS,
        RetrievalProfile.PROTOCOL,
    )
    rrf_k: int = 60
    max_subqueries: int = 2
    max_attempts: int = 8
```

- [ ] **Step 3: Implement DefaultQueryPlanner**

Normalize `top_k` to at least 1. Preserve the current candidate rule: `aiops` uses `max(final_top_k * 3, final_top_k)` and other profiles use `max(final_top_k * 2, final_top_k)`. Build queries from primary plus at most `max_subqueries` non-empty exact-deduplicated subqueries. Validate LLM candidates through MetadataFieldRegistry before merging them into inferred filters. Never merge into or relax an explicit filter.

Set `route_first=settings.route_enabled and request.metadata_filter is None`. Set `hybrid` from the configured profile list. Clamp `max_attempts` to at least 1.

- [ ] **Step 4: Run planner tests and commit**

Run: `uv run pytest tests/test_query_planner.py -q`

Expected: all tests pass.

```bash
git add app/rag/query/planner.py tests/test_query_planner.py
git diff --cached --name-status
git commit -m "feat(query): add declarative retrieval planning"
```

### Task 5: Structured Query Debuggers

**Files:**
- Create: `app/rag/query/debugger.py`
- Create: `tests/test_query_debugger.py`

- [ ] **Step 1: Write failing debugger tests**

```python
from app.rag.query.debugger import MemoryQueryDebugger, sanitize_trace_data
from app.rag.query.models import QueryTraceEvent


def test_memory_debugger_preserves_order() -> None:
    debugger = MemoryQueryDebugger()
    debugger.emit(QueryTraceEvent(stage="analysis", status="started"))
    debugger.emit(QueryTraceEvent(stage="analysis", status="completed", duration_ms=1.2))

    assert [event.status for event in debugger.events] == ["started", "completed"]


def test_trace_sanitizer_redacts_secrets_and_limits_document_preview() -> None:
    payload = sanitize_trace_data(
        {"api_key": "secret", "document": "x" * 100, "metadata_filter": {"model": "A"}},
        preview_length=12,
    )

    assert payload["api_key"] == "***"
    assert payload["document"] == "xxxxxxxxxxxx…"
    assert payload["metadata_filter"] == {"model": "A"}
```

- [ ] **Step 2: Verify failure and implement debugger variants**

Run: `uv run pytest tests/test_query_debugger.py -q`

Expected: FAIL because `debugger.py` does not exist.

Implement:

- `NoOpQueryDebugger.emit`: immediately returns `None`.
- `MemoryQueryDebugger.events`: append emitted events.
- `ConsoleQueryDebugger.emit`: write one JSON object per line to the injected text stream.
- `sanitize_trace_data`: recursively redact keys containing `api_key`, `token`, `password`, or `secret`; truncate only `document`, `content`, and `preview` strings.

- [ ] **Step 3: Run tests and commit**

Run: `uv run pytest tests/test_query_debugger.py -q`

Expected: all tests pass.

```bash
git add app/rag/query/debugger.py tests/test_query_debugger.py
git diff --cached --name-status
git commit -m "feat(query): add structured query debugging"
```

### Task 6: Retrieval Executor and Weighted Multi-Query Fusion

**Files:**
- Create: `app/rag/query/executor.py`
- Modify: `app/rag/query/retrieval_fusion.py`
- Modify: `app/rag/query/contracts.py`
- Create: `tests/test_query_executor.py`
- Modify: `tests/test_retrieval_fusion.py`

- [ ] **Step 1: Write failing weighted-fusion tests**

```python
from app.rag.models import Document, SearchHit
from app.rag.query.retrieval_fusion import WeightedQueryFusion


def hit(chunk_id: str) -> SearchHit:
    return SearchHit(Document(chunk_id, {"_chunk_id": chunk_id}), score=1.0, source="dense")


def test_weighted_query_fusion_prioritizes_primary_query() -> None:
    fused = WeightedQueryFusion().fuse_ranked(
        [(1.0, [hit("primary")]), (0.6, [hit("secondary"), hit("primary")])],
        k=60,
    )

    assert [item.document.metadata["_chunk_id"] for item in fused] == ["primary", "secondary"]
    assert fused[0].document.metadata["_retrieval_score_kind"] == "relevance"
```

- [ ] **Step 2: Write failing executor tests**

Use fake vector and sparse searchers that append `(query, filter)` calls. Add tests proving:

```python
def test_executor_routes_once_and_reuses_filter_for_subqueries():
    # Plan contains three query strings and route_first=True.
    # Fake vector store returns one route_summary, then one content hit per query.
    # Assert exactly one call has document_type=route_summary.
    # Assert all three content calls use the route-derived _source filter.


def test_executor_stops_at_global_attempt_budget():
    # Plan max_attempts=2 and all stores miss.
    # Assert total Dense attempts equals 2 and execution.attempts_used == 2.


def test_executor_keeps_dense_when_sparse_raises():
    # Dense returns a hit and sparse raises RuntimeError.
    # Assert the hit remains and degradation_reasons contains sparse_search_failed.
```

- [ ] **Step 3: Verify failures**

Run:

```bash
uv run pytest tests/test_query_executor.py tests/test_retrieval_fusion.py -q
```

Expected: FAIL because `WeightedQueryFusion` and `RetrievalExecutor` do not exist.

- [ ] **Step 4: Implement WeightedQueryFusion**

For every `(weight, ranked_hits)` pair, add `weight / (k + rank)` per stable document key. Preserve the representative Document, combine source labels, mark `_retrieval_score_kind=relevance`, and return descending `SearchHit(source="fused")` values. Reject non-positive `k`; ignore non-positive weights.

- [ ] **Step 5: Implement bounded RetrievalExecutor**

Constructor dependencies:

```python
def __init__(
    self,
    *,
    vector_searcher: VectorSearcher,
    sparse_searcher: SparseSearcher | None,
    router: QueryRouter,
    fusion: RetrievalFusion,
    multi_query_fusion: MultiQueryFusion,
) -> None: ...
```

Execution algorithm:

1. Create an attempt counter and `consume_attempt()` guard; every Dense call consumes one attempt and its paired Sparse call does not consume an additional logical attempt.
2. If `route_first`, search `route_summary` once using the primary query.
3. Select the first route whose derived content filter returns results; route-generated filters may relax only through registry/router fallbacks.
4. Execute each planned query under the shared content filter until budget exhaustion.
5. Within each query, call Dense, optionally Sparse, and existing RRF.
6. Fuse per-query ranked lists with weights `1.0` for the primary query and `0.6` for subqueries.
7. If route misses, use inferred-filter fallbacks; if strict fields are present and all miss, return empty rather than unfiltered results.
8. Record `sparse_search_failed`, `dense_search_failed`, `route_miss`, `budget_exhausted`, and `filtered_search_empty` exactly once per execution.

Preserve the current TypeError fail-closed behavior when a vector adapter cannot accept Metadata Filter.

- [ ] **Step 6: Run executor and existing retrieval tests**

Run:

```bash
uv run pytest tests/test_query_executor.py tests/test_retrieval_fusion.py tests/test_rag_retrieval_service.py -q
```

Expected: new executor tests pass; existing retrieval tests remain unchanged because QueryService has not yet switched.

- [ ] **Step 7: Commit Task 6**

```bash
git add app/rag/query/executor.py app/rag/query/retrieval_fusion.py app/rag/query/contracts.py tests/test_query_executor.py tests/test_retrieval_fusion.py
git diff --cached --name-status
git commit -m "feat(query): execute bounded multi-query retrieval"
```

### Task 7: QueryEngine and QueryService Compatibility Facade

**Files:**
- Create: `app/rag/query/engine.py`
- Modify: `app/rag/query/query_service.py`
- Modify: `app/rag/query/__init__.py`
- Create: `tests/test_query_engine.py`
- Modify: `tests/test_rag_retrieval_service.py`

- [ ] **Step 1: Write failing engine tests**

Create fakes for analyzer, policy, rule/LLM rewriters, planner, executor, reranker, and debugger. Test:

```python
def test_engine_does_not_call_llm_for_simple_query():
    # Policy returns use_llm=False.
    # Assert rule rewriter called once and LLM rewriter never called.


def test_engine_falls_back_to_rules_when_llm_rewrite_fails():
    # LLM rewriter raises QueryRewriteValidationError.
    # Assert result.degraded and llm_rewrite_failed is recorded.


def test_engine_reranks_primary_query_and_replaces_context_window():
    # Executor returns a Document with _context_window.
    # Assert reranker receives plan.rewrite.primary_text.
    # Assert result document page_content equals _context_window and metadata is preserved.
```

- [ ] **Step 2: Verify failure**

Run: `uv run pytest tests/test_query_engine.py -q`

Expected: FAIL because `engine.py` does not exist.

- [ ] **Step 3: Implement QueryEngine orchestration**

Constructor:

```python
def __init__(
    self,
    *,
    analyzer: QueryAnalyzer,
    rewrite_policy: RewritePolicy,
    rule_rewriter: QueryRewriter,
    llm_rewriter: QueryRewriter | None,
    planner: QueryPlanner,
    executor: QueryExecutor,
    reranker: Reranker,
    debugger: QueryDebugger,
) -> None: ...
```

`analyze(request)` delegates only to analyzer. `plan(request)` performs analysis, decision, guarded rewrite, and planning without invoking storage. `query(request)` executes the plan, reranks with primary text, applies `_context_window`, slices `final_top_k`, and returns QueryResult.

Every stage emits started/completed events with monotonic duration. Catch only stage-boundary exceptions: rewrite exceptions fall back to rules; executor exceptions produce a failed QueryResult; reranker exceptions preserve fused order.

- [ ] **Step 4: Convert QueryService into a compatibility facade**

Support both construction forms:

```python
QueryService(engine=query_engine)
QueryService(
    vector_searcher=...,
    sparse_searcher=...,
    router=...,
    fusion=...,
    reranker=...,
)
```

The legacy dependency form must construct a default analyzer/policy with rewriting disabled and a planner/executor matching current config behavior. `retrieve(SearchQuery)` must return `engine.query(QueryRequest.from_search_query(query)).documents`.

Preserve the existing `trace_span_factory` constructor argument. The compatibility `retrieve` method must wrap the engine call in `trace_span_factory("rag_retrieve", input_data=...)` and call `span.set_output(...)` with status, document count, candidate size, final Metadata Filter, route attempt/match, and degradation reason. This keeps current observability tests and callers stable while QueryEngine emits its independent stage events.

Delete migrated private orchestration methods only after all existing retrieval tests pass against the facade.

- [ ] **Step 5: Run engine and compatibility tests**

Run:

```bash
uv run pytest tests/test_query_engine.py tests/test_rag_retrieval_service.py tests/test_rag_query_components.py -q
```

Expected: all tests pass; the existing QueryService public contract remains intact.

- [ ] **Step 6: Commit Task 7**

```bash
git add app/rag/query/engine.py app/rag/query/query_service.py app/rag/query/__init__.py tests/test_query_engine.py tests/test_rag_retrieval_service.py
git diff --cached --name-status
git commit -m "refactor(query): route retrieval through QueryEngine"
```

### Task 8: LLM Adapter, Configuration, and Lazy Bootstrap

**Files:**
- Create: `app/rag/infrastructure/langchain_query_rewrite_model.py`
- Modify: `app/config.py`
- Modify: `app/core/llm_factory.py`
- Modify: `app/rag/bootstrap.py`
- Modify: `tests/test_layered_config.py`
- Modify: `tests/test_llm_factory_and_observability.py`
- Modify: `tests/test_rag_bootstrap.py`

- [ ] **Step 1: Add failing configuration tests**

Assert default values:

```python
assert settings.rag_query_rewrite_mode == "off"
assert settings.rag_query_llm_enabled is False
assert settings.rag_query_llm_timeout_seconds == 3.0
assert settings.rag_query_rewrite_confidence_threshold == 0.75
assert settings.rag_query_max_subqueries == 2
assert settings.rag_query_max_retrieval_attempts == 8
assert settings.rag_query_debug_preview_length == 240
```

Add an LLMFactory test calling `create_chat_model(timeout=2.5)` and asserting `captured["timeout"] == 2.5`.

- [ ] **Step 2: Verify failures and add settings**

Run:

```bash
uv run pytest tests/test_layered_config.py tests/test_llm_factory_and_observability.py -q
```

Expected: FAIL for missing fields and unsupported timeout.

Add these typed Settings fields with the exact defaults above plus `rag_query_llm_model: str = ""`. Extend `LLMFactory.create_chat_model` with `timeout: float | None = None` and pass `timeout=timeout` to ChatOpenAI.

- [ ] **Step 3: Implement the LangChain structured adapter**

Define a private Pydantic response schema with bounded `primary_text`, `subqueries`, `entity_candidates`, `confidence`, and `reason`. `LangChainQueryRewriteModel.__init__` receives a chat model and calls `with_structured_output`. `generate(payload)` invokes a fixed system prompt plus JSON payload and returns `model_dump()`.

The system prompt must state:

```text
Rewrite only for retrieval. Preserve every technical identifier exactly.
Return one primary query and at most max_subqueries focused subqueries.
Do not invent devices, manufacturers, models, alarms, register addresses, or function codes.
Entity candidates require an exact evidence substring from the supplied text or context.
Do not output database filters or retrieval settings.
```

- [ ] **Step 4: Add lazy QueryEngine assembly tests**

Extend `test_rag_bootstrap.py` to prove:

- `get_query_engine()` is cached.
- Calling `get_query_engine()` with rewrite mode `off` does not create ChatOpenAI, Milvus, or SQLite.
- A simple `chat` plan does not initialize the sparse index.
- `get_query_service().engine is get_query_engine()`.

- [ ] **Step 5: Implement bootstrap composition**

Add cached providers `get_query_rewrite_model()` and `get_query_engine()`. Build planner settings from `config`. Use a lazy model wrapper so ChatOpenAI is resolved only when `LLMQueryRewriter.rewrite` runs. Keep vector and sparse adapters lazy through `_LazyPort`.

`get_query_service()` must return `QueryService(engine=get_query_engine())`.

- [ ] **Step 6: Run configuration and bootstrap tests**

Run:

```bash
uv run pytest tests/test_layered_config.py tests/test_llm_factory_and_observability.py tests/test_rag_bootstrap.py -q
```

Expected: all tests pass with no external service connections.

- [ ] **Step 7: Commit Task 8**

```bash
git add app/config.py app/core/llm_factory.py app/rag/infrastructure/langchain_query_rewrite_model.py app/rag/bootstrap.py tests/test_layered_config.py tests/test_llm_factory_and_observability.py tests/test_rag_bootstrap.py
git diff --cached --name-status
git commit -m "feat(query): wire optional LLM rewriting"
```

### Task 9: Independent Query CLI and Tool Integration

**Files:**
- Create: `app/rag/query/cli.py`
- Create: `tests/test_query_cli.py`
- Modify: `app/tools/knowledge_tool.py`
- Modify: `tests/test_rag_retrieval_service.py`

- [ ] **Step 1: Write failing CLI tests**

Expose `main(argv: list[str] | None = None, engine: QueryEngine | None = None) -> int` so tests can inject a fake engine.

```python
def test_plan_command_outputs_json_without_calling_query(capsys):
    engine = FakeEngine()
    exit_code = main(["plan", "逆变器怎么排查？", "--profile", "aiops", "--format", "json"], engine)
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["rewrite"]["primary_text"] == "逆变器怎么排查？"
    assert engine.query_calls == 0


def test_run_command_returns_nonzero_for_structured_failure(capsys):
    engine = FailingFakeEngine()
    assert main(["run", "问题", "--format", "json"], engine) == 1
```

- [ ] **Step 2: Verify failure and implement argparse CLI**

Run: `uv run pytest tests/test_query_cli.py -q`

Expected: FAIL because `cli.py` does not exist.

Implement subcommands:

- `analyze <text> --profile --format table|json`
- `plan <text> --profile --top-k --rewrite-mode --format table|json`
- `run <text> --profile --top-k --rewrite-mode --format table|json --debug`

`analyze` calls only `engine.analyze`; `plan` calls only `engine.plan`; `run` calls `engine.query`. JSON output uses a recursive serializer for dataclasses, StrEnum, tuples, Document, and SearchHit. Table output must include stage, rewrite source, queries, filter, route/hybrid, attempts, degraded status, and document source previews.

- [ ] **Step 3: Preserve knowledge-tool behavior through the shared engine**

Keep the tool signature and LangChain artifact conversion unchanged. Resolve `get_query_service()` as before; bootstrap now guarantees it wraps the shared QueryEngine. Extend the existing tool test to assert QueryRequest profile/top-k reach the engine through the compatibility facade.

- [ ] **Step 4: Run CLI and tool tests**

Run:

```bash
uv run pytest tests/test_query_cli.py tests/test_rag_retrieval_service.py::test_knowledge_tool_passes_profile_to_query_service -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 9**

```bash
git add app/rag/query/cli.py app/tools/knowledge_tool.py tests/test_query_cli.py tests/test_rag_retrieval_service.py
git diff --cached --name-status
git commit -m "feat(query): add independent query CLI"
```

### Task 10: Documentation, Regression, and Quality Gates

**Files:**
- Modify: `app/rag/README.md`
- Modify: `tests/test_rag_architecture.py`
- Create: `tests/fixtures/query_quality_cases.json`
- Create: `tests/test_query_quality_cases.py`

- [ ] **Step 1: Add architecture and quality-fixture tests**

Extend architecture tests to assert Query core files do not import `langchain`, `fastapi`, `pymilvus`, or Agent modules; only `app/rag/infrastructure/langchain_query_rewrite_model.py` may import LangChain model types.

Create fixture cases covering chat, aiops, and protocol. Each entry must include `text`, `profile`, `expected_intents`, `expected_filter_subset`, and `llm_expected`. Add a parameterized test that runs Analyzer and RewritePolicy without external services and asserts those fields.

- [ ] **Step 2: Update the RAG README**

Replace the Query section diagram with the staged QueryEngine flow. Document:

- QueryRequest and QueryPlan.
- Rule/auto/llm/off modes.
- Automatic LLM triggers.
- Shared route result and retrieval budget.
- CLI analyze/plan/run examples.
- Degradation fields and protected-filter behavior.
- The migration compatibility of SearchQuery and QueryService.

- [ ] **Step 3: Run focused lint, type, and tests**

Run:

```bash
uv run ruff check app/rag/query app/rag/infrastructure/langchain_query_rewrite_model.py tests/test_query_*.py
uv run pyright app/rag/query app/rag/infrastructure/langchain_query_rewrite_model.py
uv run pytest tests/test_query_pipeline_models.py tests/test_query_analyzer.py tests/test_query_rewriters.py tests/test_query_planner.py tests/test_query_debugger.py tests/test_query_executor.py tests/test_query_engine.py tests/test_query_cli.py tests/test_query_quality_cases.py -q
```

Expected: Ruff exits 0, targeted Pyright exits 0, and all new Query tests pass.

- [ ] **Step 4: Run existing RAG regression tests**

Run:

```bash
uv run pytest tests/test_rag_query_components.py tests/test_rag_retrieval_service.py tests/test_rag_bootstrap.py tests/test_retrieval_fusion.py tests/test_rag_architecture.py tests/test_rag_models.py tests/test_vector_store_filtering.py -q
```

Expected: all existing RAG query and architecture tests pass.

- [ ] **Step 5: Run the complete project test suite**

Run: `uv run pytest -q`

Expected: all project tests pass. If an external service test is explicitly skipped by its existing marker, record the skip; do not convert failures into skips.

- [ ] **Step 6: Verify CLI without storage access**

Run:

```bash
uv run python -m app.rag.query.cli analyze "华为逆变器告警怎么排查并验证？" --profile aiops --format json
uv run python -m app.rag.query.cli plan "寄存器 40001 是什么？" --profile protocol --rewrite-mode rule --format json
```

Expected: both commands exit 0, print valid JSON, and do not connect to Milvus or SQLite.

- [ ] **Step 7: Commit final tests and documentation**

```bash
git add app/rag/README.md tests/test_rag_architecture.py tests/fixtures/query_quality_cases.json tests/test_query_quality_cases.py
git diff --cached --name-status
git commit -m "docs(query): document and verify query pipeline"
```

## Final Verification Checklist

- [ ] `QueryEngine` is the common execution path for Agent, API compatibility service, and CLI.
- [ ] `analyze` and `plan` run without initializing external storage.
- [ ] `QueryPlan` JSON includes original text, analysis, rewrite source, queries, filter, strategy, and budget.
- [ ] Simple auto-mode queries do not call LLM.
- [ ] Complex auto-mode queries call LLM only when enabled.
- [ ] LLM failure falls back to rules with an explicit degradation reason.
- [ ] Route Summary executes at most once per plan.
- [ ] Primary and subqueries share one route-derived content filter.
- [ ] Global attempt budget bounds route, fallback, and multi-query expansion.
- [ ] Explicit Filter is copied, never overwritten, and never relaxed.
- [ ] Strict entity misses never fall back to cross-entity unfiltered retrieval.
- [ ] Dense, Sparse, fusion, and rerank failures each preserve the documented fallback.
- [ ] Existing QueryService and `retrieve_knowledge` contracts remain compatible.
- [ ] Ruff, targeted Pyright, focused tests, RAG regressions, and the complete suite pass.
