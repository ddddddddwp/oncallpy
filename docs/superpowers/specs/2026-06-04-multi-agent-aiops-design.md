---
astrolabe_change: add-multi-agent-aiops
role: technical-design
canonical_spec: openspec
archived-with: 2026-06-04-add-multi-agent-aiops
status: final
---

# Multi-Agent AIOps Technical Design

## Context

The existing AIOps path exposes `POST /api/aiops` and streams JSON events over SSE. Internally, `AIOpsService` compiles a LangGraph loop with planner, executor, and replanner nodes. That implementation proves the workflow shape, but it keeps evidence collection and reasoning coupled through free-form text in `past_steps`.

The new implementation will add a separate multi-agent workflow while keeping the existing service available as fallback. OpenSpec remains the canonical behavior contract for this change.

## Confirmed Approach

Use a minimal linear LangGraph workflow:

1. `planner`
2. `knowledge_agent`
3. `ops_evidence_agent`
4. `diagnosis_agent`
5. `report_agent`

This is intentionally not a dynamic supervisor implementation. A dynamic router can be added later after the evidence model and API contract are stable.

## Module Layout

Add a new multi-agent package under `app/agent/aiops_multi/`:

- `state.py`: shared `MultiAgentAIOpsState` and typed helper aliases.
- `tool_registry.py`: shared tool loading helpers for local tools and MCP tools.
- `nodes/planner.py`: produces a compact diagnosis plan.
- `nodes/knowledge_agent.py`: retrieves relevant runbook or knowledge-base context.
- `nodes/ops_evidence_agent.py`: collects operational evidence through available tools and records tool failures.
- `nodes/diagnosis_agent.py`: turns structured evidence into root-cause findings and recommendations.
- `nodes/report_agent.py`: generates the final Markdown report from evidence, diagnosis, recommendations, and errors.

Add `app/services/multi_agent_aiops_service.py` to compile the graph, stream updates, and format frontend-compatible SSE payload dictionaries.

## Shared State

The new state must keep existing conceptual fields and add named evidence fields:

- `input`
- `session_id`
- `plan`
- `past_steps`
- `knowledge_evidence`
- `metrics_evidence`
- `log_evidence`
- `tool_results`
- `diagnosis`
- `recommendations`
- `errors`
- `current_stage`
- `report`
- `response`

List-like fields that can be appended by multiple nodes should use LangGraph reducers. This prevents later node updates from accidentally overwriting prior evidence.

## API Compatibility

`POST /api/aiops` must keep its current external shape:

- request body: `session_id`
- SSE frame event name: `message`
- JSON event types: `plan`, `step_complete`, `status`, `report`, `complete`, `error`
- final stream termination after `complete` or `error`

The frontend displays the final report from the `report` event. The `complete` event should remain compact and should not replace the `report` event as the only place containing the report.

## Error Handling

Agent or tool failures should be captured in state under `errors`. The workflow should continue when possible and produce an honest report that names missing evidence or failed tools. Only unrecoverable graph-level exceptions should become an `error` stream event.

## Testing Strategy

The first verification layer should test event-contract formatting without requiring live DashScope or MCP services. Focused tests should assert:

- planner output formats into a `plan` event;
- specialist-node updates format into `step_complete` or `status` events;
- report output formats into a `report` event;
- final state formats into a `complete` event;
- exceptions format into `error` events.

If test scaffolding is too expensive for the first pass, run Python compile/import checks plus a lightweight async smoke check with mocked node outputs.

## Build Split

Two workers will implement disjoint scopes:

- Worker 1 owns the multi-agent package: state, tool registry, nodes, and core graph construction.
- Worker 2 owns API/SSE compatibility and verification: service formatting, endpoint wiring, tests or smoke checks.

The integration point between workers is `MultiAgentAIOpsService.diagnose(session_id)`, which yields dictionaries compatible with the existing `app/api/aiops.py` wrapper.

## Risks

- Multi-agent execution increases latency because more model calls may occur.
- MCP services may be unavailable in local development; failures must be non-fatal when possible.
- The frontend manually parses streamed lines, so each SSE payload must remain a single JSON object.
- This directory is not currently a Git repository, so branch/worktree isolation and milestone commits are unavailable unless the project is initialized as a Git repo later.
