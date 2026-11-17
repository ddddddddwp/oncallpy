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
