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
