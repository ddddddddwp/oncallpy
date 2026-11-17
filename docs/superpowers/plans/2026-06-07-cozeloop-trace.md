# CozeLoop Trace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Trace-only CozeLoop monitoring for the existing FastAPI/RAG/AIOps service without changing prompt management.

**Architecture:** Keep `app/core/observability.py` as the only CozeLoop integration boundary. It initializes the SDK client from settings, exposes a safe `trace_span()` context manager, and degrades to no-op behavior when the SDK or config is missing.

**Tech Stack:** FastAPI, LangChain/LangGraph, CozeLoop Python SDK, pytest, uv.

---

### Task 1: Client Initialization

**Files:**
- Modify: `app/config.py`
- Modify: `app/core/observability.py`
- Test: `tests/test_llm_factory_and_observability.py`

- [ ] Add failing tests for `COZELOOP_API_BASE_URL` settings and `cozeloop.new_client(...)` initialization with workspace ID, API token, and optional API base URL.
- [ ] Implement the config field and CozeLoop client initialization.
- [ ] Verify SDK absence/config absence still leaves business code in no-op mode.

### Task 2: Trace Span Behavior

**Files:**
- Modify: `app/core/observability.py`
- Test: `tests/test_llm_factory_and_observability.py`

- [ ] Add failing tests that `trace_span()` uses the initialized client, records input/output, and closes spans even on exceptions.
- [ ] Implement client-backed span creation with safe fallback behavior.
- [ ] Verify `shutdown_cozeloop()` closes the initialized client.

### Task 3: Dependency and Runtime Verification

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] Add `cozeloop>=0.1.28`.
- [ ] Run the targeted observability tests.
- [ ] Restart the local stack and verify `/health` returns 200 with CozeLoop initialized or safely skipped.
