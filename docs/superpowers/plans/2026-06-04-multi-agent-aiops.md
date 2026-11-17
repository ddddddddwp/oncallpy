---
archived-with: 2026-06-04-add-multi-agent-aiops
status: final
---
# Multi-Agent AIOps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal multi-agent AIOps diagnosis workflow while preserving the existing `/api/aiops` SSE contract.

**Architecture:** Add a new `app/agent/aiops_multi/` package for shared state, tool loading, and specialist nodes. Add `MultiAgentAIOpsService` as the graph compiler and event formatter, then wire `/api/aiops` to the new service while keeping the old `AIOpsService` as fallback.

**Tech Stack:** Python 3.11, FastAPI, LangGraph, LangChain tools, `langchain_qwq.ChatQwen`, MCP tools, pytest.

---

## Metadata

- change: `add-multi-agent-aiops`
- design-doc: `docs/superpowers/specs/2026-06-04-multi-agent-aiops-design.md`
- base-ref: `no-git-repository`

## File Structure

Worker 1 owns only the multi-agent package:

- Create `app/agent/aiops_multi/__init__.py`
- Create `app/agent/aiops_multi/state.py`
- Create `app/agent/aiops_multi/tool_registry.py`
- Create `app/agent/aiops_multi/nodes/__init__.py`
- Create `app/agent/aiops_multi/nodes/planner.py`
- Create `app/agent/aiops_multi/nodes/knowledge_agent.py`
- Create `app/agent/aiops_multi/nodes/ops_evidence_agent.py`
- Create `app/agent/aiops_multi/nodes/diagnosis_agent.py`
- Create `app/agent/aiops_multi/nodes/report_agent.py`

Worker 2 owns service/API/tests:

- Create `app/services/multi_agent_aiops_service.py`
- Modify `app/api/aiops.py`
- Create `tests/test_multi_agent_aiops_contract.py`
- Modify `openspec/changes/add-multi-agent-aiops/tasks.md` only after integration is verified

## Task 1: Worker 1 - Multi-Agent Core Package

**Files:**

- Create: `app/agent/aiops_multi/__init__.py`
- Create: `app/agent/aiops_multi/state.py`
- Create: `app/agent/aiops_multi/tool_registry.py`
- Create: `app/agent/aiops_multi/nodes/__init__.py`
- Create: `app/agent/aiops_multi/nodes/planner.py`
- Create: `app/agent/aiops_multi/nodes/knowledge_agent.py`
- Create: `app/agent/aiops_multi/nodes/ops_evidence_agent.py`
- Create: `app/agent/aiops_multi/nodes/diagnosis_agent.py`
- Create: `app/agent/aiops_multi/nodes/report_agent.py`

- [ ] **Step 1: Use CodeGraph before editing**

Run CodeGraph context for `AIOpsService`, `planner`, `executor`, `replanner`, and `get_mcp_client_with_retry`. Confirm the new package will not modify existing `app/agent/aiops/*`.

- [ ] **Step 2: Create shared state**

Create `MultiAgentAIOpsState` with fields:

- `input: str`
- `session_id: str`
- `plan: list[str]`
- `past_steps: Annotated[list[tuple[str, str]], operator.add]`
- `knowledge_evidence: Annotated[list[dict[str, Any]], operator.add]`
- `metrics_evidence: Annotated[list[dict[str, Any]], operator.add]`
- `log_evidence: Annotated[list[dict[str, Any]], operator.add]`
- `tool_results: Annotated[list[dict[str, Any]], operator.add]`
- `diagnosis: dict[str, Any]`
- `recommendations: list[dict[str, Any]]`
- `errors: Annotated[list[dict[str, Any]], operator.add]`
- `current_stage: str`
- `report: str`
- `response: str`

Also define helper type aliases for evidence dictionaries.

- [ ] **Step 3: Create tool registry**

Create helpers that return local tools and MCP tools:

- `get_local_tools()`
- `get_all_tools()`
- `format_available_tools()`

Use existing `get_current_time`, `retrieve_knowledge`, `get_mcp_client_with_retry`, and `format_tools_description`. On MCP failure, return local tools and an error dictionary rather than raising.

- [ ] **Step 4: Create planner node**

Implement `planner(state)` to return:

- `plan`
- `current_stage`
- optional `errors`

Use a structured `Plan` model like the existing planner. If LLM/tool setup fails, return a deterministic fallback plan that still drives downstream nodes.

- [ ] **Step 5: Create knowledge agent node**

Implement `knowledge_agent(state)` to call `retrieve_knowledge.ainvoke({"query": state["input"]})`. Return `knowledge_evidence`, `past_steps`, `current_stage`, and `errors` on failure.

- [ ] **Step 6: Create operational evidence node**

Implement `ops_evidence_agent(state)` to bind all available tools to `ChatQwen`, ask for operational evidence relevant to the plan, execute tool calls through `ToolNode`, and store summarized results under `tool_results`. Classify result summaries into `metrics_evidence` or `log_evidence` by tool name where possible.

- [ ] **Step 7: Create diagnosis node**

Implement `diagnosis_agent(state)` to use collected evidence and errors to produce a structured diagnosis dictionary and recommendations list. Use deterministic fallback output if LLM structured output fails.

- [ ] **Step 8: Create report node**

Implement `report_agent(state)` to produce a Markdown report using `input`, `plan`, `knowledge_evidence`, `metrics_evidence`, `log_evidence`, `diagnosis`, `recommendations`, and `errors`. Return both `report` and `response` with the same Markdown text.

- [ ] **Step 9: Run syntax validation for owned files**

Run:

`python -m py_compile app/agent/aiops_multi/*.py app/agent/aiops_multi/nodes/*.py`

Expected: command exits 0.

## Task 2: Worker 2 - Service, API Compatibility, and Tests

**Files:**

- Create: `app/services/multi_agent_aiops_service.py`
- Modify: `app/api/aiops.py`
- Create: `tests/test_multi_agent_aiops_contract.py`

- [ ] **Step 1: Use CodeGraph before editing**

Run CodeGraph context for `diagnose_stream`, `AIOpsService.diagnose`, and `/api/aiops`. Confirm the existing frontend expects `report` before `complete`.

- [ ] **Step 2: Write failing contract tests first**

Create `tests/test_multi_agent_aiops_contract.py` with tests that do not call DashScope or MCP:

- formatter turns planner node output into `type == "plan"`
- formatter turns agent progress into `type == "step_complete"`
- formatter turns report state into `type == "report"` with `report`
- formatter turns final state into `type == "complete"` with `diagnosis.report`
- API wrapper emits `event: message` and JSON `data` when the service is mocked

Run:

`pytest tests/test_multi_agent_aiops_contract.py -q`

Expected before implementation: fail because `app.services.multi_agent_aiops_service` does not exist.

- [ ] **Step 3: Implement service skeleton**

Create `MultiAgentAIOpsService` with:

- constants for node names;
- `_build_graph()` compiling a linear LangGraph from Worker 1 nodes;
- `execute(user_input, session_id)` yielding formatted events;
- `diagnose(session_id)` using the same fixed AIOps task text as the old service;
- event formatter helpers for plan, step completion, report, complete, and error.

Do not instantiate `ChatQwen` in the service module itself; node modules own model calls.

- [ ] **Step 4: Wire API with fallback**

Modify `app/api/aiops.py` to import `multi_agent_aiops_service`. Use it for `/api/aiops`. If it raises before streaming starts, fall back to `aiops_service.diagnose(session_id=session_id)`.

Keep the SSE frame shape:

- `event` is `message`
- `data` is `json.dumps(event, ensure_ascii=False)`
- break on `complete` or `error`

- [ ] **Step 5: Run contract tests**

Run:

`pytest tests/test_multi_agent_aiops_contract.py -q`

Expected after implementation: pass.

- [ ] **Step 6: Run syntax/import validation**

Run:

`python -m py_compile app/services/multi_agent_aiops_service.py app/api/aiops.py tests/test_multi_agent_aiops_contract.py`

Expected: command exits 0.

## Task 3: Integration

**Files:**

- Modify: files from Worker 1 and Worker 2 only if interface mismatches exist
- Modify: `openspec/changes/add-multi-agent-aiops/tasks.md`

- [ ] **Step 1: Reconcile worker interfaces**

Confirm Worker 2 imports exactly these node functions from Worker 1:

- `planner`
- `knowledge_agent`
- `ops_evidence_agent`
- `diagnosis_agent`
- `report_agent`

Confirm all nodes accept `MultiAgentAIOpsState` and return partial state dictionaries.

- [ ] **Step 2: Run focused tests**

Run:

`pytest tests/test_multi_agent_aiops_contract.py -q`

Expected: pass.

- [ ] **Step 3: Run compile checks**

Run:

`python -m compileall app/agent/aiops_multi app/services/multi_agent_aiops_service.py app/api/aiops.py tests/test_multi_agent_aiops_contract.py`

Expected: command exits 0.

- [ ] **Step 4: Update OpenSpec tasks**

Mark implementation tasks complete in `openspec/changes/add-multi-agent-aiops/tasks.md` after tests and compile checks pass.

## Self-Review

Spec coverage:

- Multi-agent orchestration: Task 1 creates specialist nodes; Task 2 compiles the graph.
- Compatible streaming events: Task 2 adds formatter tests and API wiring.
- Structured diagnosis evidence: Task 1 defines state and evidence nodes.
- Error handling: Task 1 stores errors; Task 2 formats graph exceptions.

No placeholders remain. The plan intentionally excludes indexing optimization because the OpenSpec scope excludes Milvus and embedding changes.
