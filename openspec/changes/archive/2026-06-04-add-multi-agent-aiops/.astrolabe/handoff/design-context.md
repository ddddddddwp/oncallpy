# Astrolabe Design Handoff

- Change: add-multi-agent-aiops
- Phase: design
- Mode: compact
- Context hash: 27412e9284090f3e99832d4443e34b386fd7610501ce8518e2c56c657aa84ef3

Generated-by: ast-handoff.sh

OpenSpec remains the canonical capability spec. This handoff is a deterministic, source-traceable context pack, not an agent-authored summary.

## openspec/changes/add-multi-agent-aiops/proposal.md

- Source: openspec/changes/add-multi-agent-aiops/proposal.md
- Lines: 1-34
- SHA256: 69c91b6a64955a15ee0191b61f2da5962dcc59236303eaffc43e528acf435578

```md
## Why

The current AIOps diagnosis flow uses a single planner/executor/replanner loop, so tool selection, evidence collection, root-cause reasoning, and report generation are tightly coupled. Splitting diagnosis into specialized agents will make the workflow easier to reason about, test, and extend while preserving the existing `/api/aiops` streaming API.

## What Changes

- Introduce a multi-agent AIOps orchestration flow for the existing Python FastAPI/LangGraph application.
- Split AIOps responsibilities into dedicated agents for planning, knowledge retrieval, metrics collection, diagnosis, and report generation.
- Preserve the existing `/api/aiops` SSE event contract so the current frontend can keep working.
- Add structured shared state for evidence, diagnosis, recommendations, and report output.
- Keep the existing single-flow AIOps implementation available as a compatibility fallback during migration.
- Do not change document upload, vector indexing, embedding, or Milvus schema in this change.

## Capabilities

### New Capabilities

- `multi-agent-aiops`: Multi-agent AIOps diagnosis orchestration with compatible streaming events.

### Modified Capabilities

- None.

## Impact

- Affected backend modules:
  - `app/services/aiops_service.py`
  - new multi-agent service/module files under `app/services/` and `app/agent/`
  - `app/api/aiops.py` only if wiring changes are needed
- Affected runtime integrations:
  - Existing local tools: knowledge retrieval and current time
  - Existing MCP tools: CLS and Monitor MCP servers
  - Existing `/api/aiops` frontend flow in `static/app.js`
- No database migration, no Milvus collection change, and no new external dependency is planned.
```

## openspec/changes/add-multi-agent-aiops/design.md

- Source: openspec/changes/add-multi-agent-aiops/design.md
- Lines: 1-71
- SHA256: f54e6515ea6476be1f08835354ea05b38ffb78761093997f8ffa8fc613cf0a3e

```md
## Context

The project already exposes an AIOps streaming endpoint through FastAPI. Internally, `AIOpsService` builds a LangGraph state machine with three nodes: planner, executor, and replanner. This works, but the executor is responsible for too much: deciding which tools to call, collecting evidence, and returning step results that the replanner must interpret from text.

The existing tool surface is already suitable for a multi-agent workflow. Local tools provide knowledge retrieval and time lookup. MCP tools provide log and monitoring capabilities. The main implementation challenge is orchestration: introduce clearer agent boundaries without breaking the current `/api/aiops` SSE response shape.

## Goals / Non-Goals

**Goals:**

- Add a minimal multi-agent AIOps workflow that can be called from the existing AIOps API.
- Separate planning, knowledge retrieval, metrics/log evidence collection, diagnosis, and report generation into distinct nodes or agent functions.
- Keep current frontend-compatible streaming events: `plan`, `step_complete`, `status`, `report`, `complete`, and `error`.
- Use structured shared state so report generation is based on named evidence fields, not only free-form step history.
- Keep the implementation small enough to verify in this repository without introducing new infrastructure.

**Non-Goals:**

- Do not optimize document chunking, embedding, or Milvus indexing in this change.
- Do not replace the RAG chat agent.
- Do not add a new frontend UI for agent internals.
- Do not add a new LLM provider or external dependency.
- Do not remove the current AIOps service until the multi-agent path is verified.

## Decisions

### Decision 1: Use LangGraph nodes as specialized agents

The multi-agent flow will use LangGraph nodes/functions for each specialist role instead of adding a separate agent framework. This matches the existing codebase and keeps state transitions explicit.

Alternative considered: create independent LangChain agents that call each other dynamically. That would be more flexible, but harder to test and more likely to break the existing streaming contract.

### Decision 2: Preserve `/api/aiops` event compatibility

The API will continue to stream the same event types that the frontend already understands. Internally the event source may come from a new `MultiAgentAIOpsService`, but the public contract should not require frontend changes.

Alternative considered: expose a new `/api/aiops_multi` endpoint first. That is safer during migration, but it delays real integration and leaves two user-facing flows to maintain. A fallback path is enough for this project.

### Decision 3: Use structured evidence fields

The shared state will include fields for plan, knowledge evidence, log evidence, metrics evidence, diagnosis, recommendations, report, and errors. This makes each agent's responsibility visible and gives tests stable assertions.

Alternative considered: keep using `past_steps` only. That preserves the current model but repeats the core problem: downstream nodes must infer evidence from natural-language step output.

### Decision 4: Start with a minimal role set

The first implementation should include planner, knowledge agent, metrics/log evidence agent, diagnosis agent, and report agent. A separate orchestrator can be represented by the graph routing itself for now.

Alternative considered: add a full orchestrator agent that dynamically selects specialists. That is useful later, but too large for the first step and harder to verify.

## Risks / Trade-offs

- More LLM calls can increase latency and cost. Mitigation: keep the first workflow minimal and use deterministic aggregation where possible.
- MCP services are mock/local and may fail if not running. Mitigation: capture tool errors in state and still produce an honest final report.
- Streaming event semantics can drift from frontend expectations. Mitigation: keep event formatting in a service layer and verify event types.
- The old and new AIOps services may overlap. Mitigation: keep wiring narrow and avoid deleting the old implementation in this change.

## Migration Plan

1. Add the new multi-agent state and node modules.
2. Add a new service that compiles and streams the multi-agent graph.
3. Wire `/api/aiops` to the new service or add a feature-flagged fallback path if needed.
4. Run syntax/import checks and focused tests or lightweight async smoke checks.
5. Keep the old `AIOpsService` available until the new flow is verified.

Rollback is straightforward: restore `/api/aiops` to use the existing `aiops_service.diagnose` path.

## Open Questions

- Whether the first implementation should wire `/api/aiops` directly to the new service or expose it behind a temporary compatibility alias.
- Whether log evidence and metrics evidence should be two separate nodes or one combined evidence node in the first pass.
```

## openspec/changes/add-multi-agent-aiops/tasks.md

- Source: openspec/changes/add-multi-agent-aiops/tasks.md
- Lines: 1-23
- SHA256: 8c0744a22e9a5be775d95fdd920c77c8bdb322ca53f78c0a54049f37f4fd1a74

```md
## 1. Core Multi-Agent Structure

- [ ] 1.1 Add shared multi-agent AIOps state definitions.
- [ ] 1.2 Add specialist agent nodes for planning, knowledge evidence, operational evidence, diagnosis, and report generation.
- [ ] 1.3 Add a multi-agent AIOps service that compiles and streams the LangGraph workflow.

## 2. API Compatibility

- [ ] 2.1 Preserve the existing `/api/aiops` SSE event types and payload shape expected by the frontend.
- [ ] 2.2 Keep the old AIOps service available as a rollback or fallback path.
- [ ] 2.3 Ensure tool failures are surfaced as structured errors without crashing the whole stream.

## 3. Parallel Implementation Split

- [ ] 3.1 Agent 1 owns the core workflow modules and shared state.
- [ ] 3.2 Agent 2 owns API/SSE compatibility, validation, and focused tests or smoke checks.
- [ ] 3.3 Integrate both work streams and resolve any interface mismatch.

## 4. Verification

- [ ] 4.1 Run Python syntax/import validation for touched modules.
- [ ] 4.2 Run focused tests if a test suite exists; otherwise add or run a lightweight smoke check.
- [ ] 4.3 Verify OpenSpec/Astrolabe tasks and status before moving to archive.
```

## openspec/changes/add-multi-agent-aiops/specs/multi-agent-aiops/spec.md

- Source: openspec/changes/add-multi-agent-aiops/specs/multi-agent-aiops/spec.md
- Lines: 1-43
- SHA256: e7b518108e10fb814ddd851bac8276b4d141647da7577c788854ed38cbc23da4

```md
## ADDED Requirements

### Requirement: Multi-agent AIOps orchestration

The system SHALL provide an AIOps diagnosis workflow composed of specialized agent nodes for planning, evidence collection, diagnosis, and report generation.

#### Scenario: Diagnosis runs through specialist agents

- **WHEN** a client starts an AIOps diagnosis
- **THEN** the system SHALL execute a multi-agent workflow that creates a plan, collects evidence, produces a diagnosis, and generates a final report

#### Scenario: Agent failures are captured

- **WHEN** an agent node or tool call fails during diagnosis
- **THEN** the system SHALL capture the failure in workflow state and continue to produce an honest error-aware output when possible

### Requirement: Compatible AIOps streaming events

The system SHALL preserve the existing `/api/aiops` streaming event types consumed by the frontend.

#### Scenario: Frontend receives compatible events

- **WHEN** the frontend calls `/api/aiops`
- **THEN** the streamed events SHALL use compatible event types including `plan`, `step_complete`, `status`, `report`, `complete`, or `error`

#### Scenario: Final report remains Markdown

- **WHEN** the diagnosis completes successfully
- **THEN** the final report SHALL be returned as Markdown text in the streamed report or completion payload

### Requirement: Structured diagnosis evidence

The system SHALL keep structured evidence in the multi-agent workflow state.

#### Scenario: Evidence is available to diagnosis and report agents

- **WHEN** knowledge, log, or metrics evidence is collected
- **THEN** the workflow SHALL store that evidence in named state fields that downstream diagnosis and report agents can read

#### Scenario: Report is based on collected evidence

- **WHEN** the report agent generates the final report
- **THEN** it SHALL base the report on collected evidence, diagnosis, recommendations, and recorded errors instead of inventing unavailable data
```

