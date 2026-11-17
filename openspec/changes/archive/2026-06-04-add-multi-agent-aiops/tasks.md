## 1. Core Multi-Agent Structure

- [x] 1.1 Add shared multi-agent AIOps state definitions.
- [x] 1.2 Add specialist agent nodes for planning, knowledge evidence, operational evidence, diagnosis, and report generation.
- [x] 1.3 Add a multi-agent AIOps service that compiles and streams the LangGraph workflow.

## 2. API Compatibility

- [x] 2.1 Preserve the existing `/api/aiops` SSE event types and payload shape expected by the frontend.
- [x] 2.2 Keep the old AIOps service available as a rollback or fallback path.
- [x] 2.3 Ensure tool failures are surfaced as structured errors without crashing the whole stream.

## 3. Parallel Implementation Split

- [x] 3.1 Agent 1 owns the core workflow modules and shared state.
- [x] 3.2 Agent 2 owns API/SSE compatibility, validation, and focused tests or smoke checks.
- [x] 3.3 Integrate both work streams and resolve any interface mismatch.

## 4. Verification

- [x] 4.1 Run Python syntax/import validation for touched modules.
- [x] 4.2 Run focused tests if a test suite exists; otherwise add or run a lightweight smoke check.
- [x] 4.3 Verify OpenSpec/Astrolabe tasks and status before moving to archive.
