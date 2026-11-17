# Redis Checkpoint Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist LangGraph checkpoints in Redis through `REDIS_URL` instead of process-local `MemorySaver`.

**Architecture:** Add one checkpoint module that owns Redis saver creation and version-specific helper methods. Services initialize Redis-backed checkpointers through this module and keep using `session_id` as LangGraph `thread_id`. RAG session read/clear APIs become async so they can use the async Redis checkpointer required by async LangGraph execution.

**Tech Stack:** FastAPI, LangGraph, `langgraph-checkpoint-redis`, Pydantic Settings, pytest, pytest-asyncio.

---

## File Structure

- Create `app/core/checkpoint.py`: async Redis checkpointer factory plus helper functions for latest-checkpoint lookup and thread deletion.
- Modify `app/config.py`: add `redis_url` loaded from `REDIS_URL` with the local Docker default.
- Modify `pyproject.toml`: add `langgraph-checkpoint-redis`.
- Modify `app/services/rag_agent_service.py`: remove direct `MemorySaver`, initialize Redis checkpointer asynchronously, make session history and clear methods async.
- Modify `app/services/aiops_service.py`: remove direct `MemorySaver`, add async Redis initialization before graph execution.
- Modify `app/services/multi_agent_aiops_service.py`: remove direct `MemorySaver`, add async Redis initialization before graph execution and lazy retry paths.
- Modify `app/api/chat.py`: await async RAG session history and clear calls.
- Modify `app/main.py`: initialize Redis-backed graph services during FastAPI lifespan startup and close them during shutdown if the saver exposes a close method.
- Modify `docs/ONBOARDING.md`: replace memory-only checkpoint notes with Redis-backed behavior and local Redis requirements.
- Test `tests/test_redis_checkpoint.py`: checkpoint config, factory, service integration, async session helpers.
- Test existing `tests/test_llm_factory_and_observability.py`: add config default/env assertions or keep config tests in the new test file.

## Task 1: Add Redis Config And Dependency

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/config.py`
- Test: `tests/test_redis_checkpoint.py`

- [ ] **Step 1: Write the failing config tests**

Create `tests/test_redis_checkpoint.py` with:

```python
from __future__ import annotations

import pytest

from app.config import Settings


def test_settings_defaults_redis_url_to_local_docker(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)

    settings = Settings(_env_file=None)

    assert settings.redis_url == "redis://:your-redis-password@localhost:6379/0"


def test_settings_reads_redis_url_env(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://:secret@redis.local:6379/2")

    settings = Settings(_env_file=None)

    assert settings.redis_url == "redis://:secret@redis.local:6379/2"
```

- [ ] **Step 2: Run the config tests and verify they fail**

Run:

```bash
uv run pytest tests/test_redis_checkpoint.py::test_settings_defaults_redis_url_to_local_docker tests/test_redis_checkpoint.py::test_settings_reads_redis_url_env -q
```

Expected: FAIL because `Settings` has no `redis_url` field.

- [ ] **Step 3: Add the config field**

In `app/config.py`, add this field after the app network settings:

```python
    # Redis checkpoint 配置
    redis_url: str = "redis://:your-redis-password@localhost:6379/0"
```

- [ ] **Step 4: Add the Redis checkpoint dependency**

In `pyproject.toml` dependencies, add:

```toml
    "langgraph-checkpoint-redis>=0.4.0",
```

Then run:

```bash
uv lock
```

Expected: `uv.lock` updates and the package resolves.

- [ ] **Step 5: Run the config tests and verify they pass**

Run:

```bash
uv run pytest tests/test_redis_checkpoint.py::test_settings_defaults_redis_url_to_local_docker tests/test_redis_checkpoint.py::test_settings_reads_redis_url_env -q
```

Expected: PASS.

## Task 2: Add Central Redis Checkpoint Module

**Files:**
- Create: `app/core/checkpoint.py`
- Modify: `tests/test_redis_checkpoint.py`

- [ ] **Step 1: Write failing factory and helper tests**

Append to `tests/test_redis_checkpoint.py`:

```python
from types import SimpleNamespace


@pytest.mark.asyncio
async def test_create_redis_checkpointer_uses_configured_url_and_setup(monkeypatch):
    import app.core.checkpoint as checkpoint_module

    calls: dict[str, object] = {}

    class FakeAsyncRedisSaver:
        def __init__(self, redis_url: str):
            calls["redis_url"] = redis_url
            self.setup_called = False

        async def asetup(self):
            self.setup_called = True
            calls["setup_called"] = True

    monkeypatch.setattr(checkpoint_module, "AsyncRedisSaver", FakeAsyncRedisSaver)
    monkeypatch.setattr(checkpoint_module.config, "redis_url", "redis://:pw@localhost:6379/4")

    saver = await checkpoint_module.create_redis_checkpointer()

    assert isinstance(saver, FakeAsyncRedisSaver)
    assert calls == {
        "redis_url": "redis://:pw@localhost:6379/4",
        "setup_called": True,
    }


@pytest.mark.asyncio
async def test_create_redis_checkpointer_fails_loudly_when_dependency_missing(monkeypatch):
    import app.core.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "AsyncRedisSaver", None)

    with pytest.raises(RuntimeError, match="langgraph-checkpoint-redis"):
        await checkpoint_module.create_redis_checkpointer()


@pytest.mark.asyncio
async def test_get_latest_checkpoint_prefers_async_get():
    from app.core.checkpoint import get_latest_checkpoint

    class FakeCheckpointer:
        async def aget(self, config):
            return {"channel_values": {"messages": ["stored"]}}

    result = await get_latest_checkpoint(
        FakeCheckpointer(),
        {"configurable": {"thread_id": "session-1"}},
    )

    assert result == {"channel_values": {"messages": ["stored"]}}


@pytest.mark.asyncio
async def test_delete_checkpoint_thread_prefers_async_delete():
    from app.core.checkpoint import delete_checkpoint_thread

    calls = []

    class FakeCheckpointer:
        async def adelete_thread(self, thread_id):
            calls.append(thread_id)

    await delete_checkpoint_thread(FakeCheckpointer(), "session-1")

    assert calls == ["session-1"]


@pytest.mark.asyncio
async def test_close_checkpointer_uses_available_async_close():
    from app.core.checkpoint import close_checkpointer

    calls = []

    class FakeCheckpointer:
        async def aclose(self):
            calls.append("closed")

    await close_checkpointer(FakeCheckpointer())

    assert calls == ["closed"]
```

- [ ] **Step 2: Run the checkpoint module tests and verify they fail**

Run:

```bash
uv run pytest tests/test_redis_checkpoint.py -q
```

Expected: FAIL because `app.core.checkpoint` does not exist.

- [ ] **Step 3: Implement the checkpoint module**

Create `app/core/checkpoint.py`:

```python
"""Redis-backed LangGraph checkpoint helpers."""

from __future__ import annotations

import inspect
from typing import Any

from loguru import logger

from app.config import config

try:
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
except ImportError:  # pragma: no cover - exercised by tests through monkeypatching.
    AsyncRedisSaver = None  # type: ignore[assignment]


async def create_redis_checkpointer(redis_url: str | None = None) -> Any:
    """Create and initialize the async Redis checkpointer used by LangGraph."""
    if AsyncRedisSaver is None:
        raise RuntimeError(
            "Redis checkpoint support is not installed. "
            "Install the langgraph-checkpoint-redis package."
        )

    resolved_url = redis_url or config.redis_url
    try:
        saver = AsyncRedisSaver(redis_url=resolved_url)
        await saver.asetup()
        logger.info("Redis LangGraph checkpointer initialized")
        return saver
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize Redis LangGraph checkpointer. "
            f"Check REDIS_URL and Redis availability. REDIS_URL={resolved_url!r}"
        ) from exc


async def get_latest_checkpoint(checkpointer: Any, config_dict: dict[str, Any]) -> Any:
    """Return the latest checkpoint data for a LangGraph config."""
    if hasattr(checkpointer, "aget"):
        return await checkpointer.aget(config_dict)

    if hasattr(checkpointer, "aget_tuple"):
        checkpoint_tuple = await checkpointer.aget_tuple(config_dict)
        return _extract_checkpoint(checkpoint_tuple)

    if hasattr(checkpointer, "get"):
        result = checkpointer.get(config_dict)
        if inspect.isawaitable(result):
            result = await result
        return _extract_checkpoint(result)

    if hasattr(checkpointer, "get_tuple"):
        result = checkpointer.get_tuple(config_dict)
        if inspect.isawaitable(result):
            result = await result
        return _extract_checkpoint(result)

    raise RuntimeError("Configured LangGraph checkpointer cannot read checkpoints")


async def delete_checkpoint_thread(checkpointer: Any, thread_id: str) -> None:
    """Delete all checkpoints for a LangGraph thread."""
    if hasattr(checkpointer, "adelete_thread"):
        await checkpointer.adelete_thread(thread_id)
        return

    if hasattr(checkpointer, "delete_thread"):
        result = checkpointer.delete_thread(thread_id)
        if inspect.isawaitable(result):
            await result
        return

    raise RuntimeError("Configured LangGraph checkpointer cannot delete threads")


async def close_checkpointer(checkpointer: Any) -> None:
    """Close a checkpointer if the implementation exposes a close method."""
    for method_name in ("aclose", "close"):
        close = getattr(checkpointer, method_name, None)
        if close is None:
            continue
        result = close()
        if inspect.isawaitable(result):
            await result
        return


def _extract_checkpoint(value: Any) -> Any:
    if hasattr(value, "checkpoint"):
        return value.checkpoint
    if isinstance(value, tuple) and value:
        return value[0]
    return value
```

- [ ] **Step 4: Run checkpoint module tests and verify they pass**

Run:

```bash
uv run pytest tests/test_redis_checkpoint.py -q
```

Expected: PASS.

## Task 3: Wire Redis Checkpointer Into Services

**Files:**
- Modify: `app/services/rag_agent_service.py`
- Modify: `app/services/aiops_service.py`
- Modify: `app/services/multi_agent_aiops_service.py`
- Modify: `app/api/chat.py`
- Modify: `app/main.py`
- Modify: `tests/test_redis_checkpoint.py`

- [ ] **Step 1: Write failing service integration tests**

Append to `tests/test_redis_checkpoint.py`:

```python
@pytest.mark.asyncio
async def test_rag_service_initializes_checkpointer_with_factory(monkeypatch):
    import app.services.rag_agent_service as rag_module

    fake_checkpointer = object()

    async def fake_create():
        return fake_checkpointer

    service = rag_module.RagAgentService.__new__(rag_module.RagAgentService)
    service.checkpointer = None

    monkeypatch.setattr(rag_module, "create_redis_checkpointer", fake_create)

    await service._initialize_checkpointer()

    assert service.checkpointer is fake_checkpointer


@pytest.mark.asyncio
async def test_rag_session_history_reads_async_checkpoint_helper(monkeypatch):
    import app.services.rag_agent_service as rag_module

    service = rag_module.RagAgentService.__new__(rag_module.RagAgentService)
    service.checkpointer = object()

    async def fake_get_latest(checkpointer, config_dict):
        return {
            "channel_values": {
                "messages": [
                    rag_module.SystemMessage(content="system"),
                    rag_module.HumanMessage(content="hello"),
                    SimpleNamespace(content="world"),
                ]
            }
        }

    monkeypatch.setattr(rag_module, "get_latest_checkpoint", fake_get_latest)

    history = await service.get_session_history("session-1")

    assert [item["role"] for item in history] == ["user", "assistant"]
    assert [item["content"] for item in history] == ["hello", "world"]


@pytest.mark.asyncio
async def test_rag_clear_session_deletes_checkpoint_thread(monkeypatch):
    import app.services.rag_agent_service as rag_module

    calls = []
    service = rag_module.RagAgentService.__new__(rag_module.RagAgentService)
    service.checkpointer = object()

    async def fake_delete(checkpointer, thread_id):
        calls.append((checkpointer, thread_id))

    monkeypatch.setattr(rag_module, "delete_checkpoint_thread", fake_delete)

    assert await service.clear_session("session-1") is True
    assert calls == [(service.checkpointer, "session-1")]


@pytest.mark.asyncio
async def test_aiops_service_ensure_graph_initializes_redis_checkpointer(monkeypatch):
    import app.services.aiops_service as aiops_module

    fake_checkpointer = object()

    async def fake_create():
        return fake_checkpointer

    service = aiops_module.AIOpsService.__new__(aiops_module.AIOpsService)
    service.checkpointer = None
    service.graph = None
    service._initialization_error = None

    monkeypatch.setattr(aiops_module, "create_redis_checkpointer", fake_create)
    monkeypatch.setattr(service, "_build_graph", lambda: "graph")

    graph = await service._ensure_graph()

    assert graph == "graph"
    assert service.checkpointer is fake_checkpointer


@pytest.mark.asyncio
async def test_multi_agent_ensure_graph_initializes_redis_checkpointer(monkeypatch):
    import app.services.multi_agent_aiops_service as multi_module

    fake_checkpointer = object()

    async def fake_create():
        return fake_checkpointer

    service = multi_module.MultiAgentAIOpsService.__new__(multi_module.MultiAgentAIOpsService)
    service.checkpointer = None
    service.graph = None
    service._initialization_error = None

    monkeypatch.setattr(multi_module, "create_redis_checkpointer", fake_create)
    monkeypatch.setattr(service, "_build_graph", lambda: "graph")

    graph = await service._ensure_graph()

    assert graph == "graph"
    assert service.checkpointer is fake_checkpointer
```

- [ ] **Step 2: Run the service tests and verify they fail**

Run:

```bash
uv run pytest tests/test_redis_checkpoint.py -q
```

Expected: FAIL because services still use `MemorySaver`, sync methods, and sync `_ensure_graph`.

- [ ] **Step 3: Update `RagAgentService`**

Change imports in `app/services/rag_agent_service.py`:

```python
from app.core.checkpoint import (
    close_checkpointer,
    create_redis_checkpointer,
    delete_checkpoint_thread,
    get_latest_checkpoint,
)
```

Remove:

```python
from langgraph.checkpoint.memory import MemorySaver
```

In `__init__`, replace `self.checkpointer = MemorySaver()` with:

```python
        self.checkpointer = None
```

Add before `_initialize_agent`:

```python
    async def _initialize_checkpointer(self):
        if self.checkpointer is None:
            self.checkpointer = await create_redis_checkpointer()
```

At the top of `_initialize_agent`, before MCP setup:

```python
        await self._initialize_checkpointer()
```

Change `get_session_history` signature and body:

```python
    async def get_session_history(self, session_id: str) -> list:
        """获取会话历史（从 Redis checkpointer 中读取）"""
        try:
            await self._initialize_checkpointer()
            config_dict = {"configurable": {"thread_id": session_id}}
            checkpoint_data = await get_latest_checkpoint(self.checkpointer, config_dict)

            if not checkpoint_data:
                logger.info(f"获取会话历史: {session_id}, 消息数量: 0")
                return []

            messages = checkpoint_data.get("channel_values", {}).get("messages", [])
            history = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    continue

                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, "content") else str(msg)
                timestamp = getattr(msg, "timestamp", None)
                if not timestamp:
                    from datetime import datetime

                    timestamp = datetime.now().isoformat()

                history.append(
                    {
                        "role": role,
                        "content": content,
                        "timestamp": timestamp,
                    }
                )

            logger.info(f"获取会话历史: {session_id}, 消息数量: {len(history)}")
            return history
        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []
```

Change `clear_session`:

```python
    async def clear_session(self, session_id: str) -> bool:
        """清空会话历史（从 Redis checkpointer 中删除）"""
        try:
            await self._initialize_checkpointer()
            await delete_checkpoint_thread(self.checkpointer, session_id)
            logger.info(f"已清除会话历史: {session_id}")
            return True
        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False
```

In `cleanup`, add:

```python
            if self.checkpointer is not None:
                await close_checkpointer(self.checkpointer)
                self.checkpointer = None
                self.agent = None
                self._agent_initialized = False
```

- [ ] **Step 4: Update `AIOpsService`**

Change imports in `app/services/aiops_service.py`:

```python
from app.core.checkpoint import close_checkpointer, create_redis_checkpointer
```

Remove:

```python
from langgraph.checkpoint.memory import MemorySaver
```

Change `__init__`:

```python
    def __init__(self):
        """初始化服务"""
        self.checkpointer = None
        self.graph = None
        self._initialization_error: Exception | None = None
        logger.info("Plan-Execute-Replan Service 初始化完成")
```

Add:

```python
    async def initialize(self) -> None:
        await self._ensure_graph()

    async def cleanup(self) -> None:
        if self.checkpointer is not None:
            await close_checkpointer(self.checkpointer)
            self.checkpointer = None
        self.graph = None

    async def _ensure_graph(self):
        if self.graph is not None:
            return self.graph

        if self.checkpointer is None:
            self.checkpointer = await create_redis_checkpointer()

        self.graph = self._build_graph()
        self._initialization_error = None
        return self.graph
```

At the start of `execute`, before streaming:

```python
            graph = await self._ensure_graph()
```

Then replace `self.graph.astream` and `self.graph.get_state` in `execute` with `graph.astream` and `graph.get_state`.

- [ ] **Step 5: Update `MultiAgentAIOpsService`**

Change imports in `app/services/multi_agent_aiops_service.py`:

```python
from app.core.checkpoint import close_checkpointer, create_redis_checkpointer
```

Remove:

```python
from langgraph.checkpoint.memory import MemorySaver
```

Change `__init__`:

```python
    def __init__(self) -> None:
        self.checkpointer = None
        self.graph = None
        self._initialization_error: Exception | None = None
        logger.info("Multi-agent AIOps Service 初始化完成")
```

Add:

```python
    async def initialize(self) -> None:
        await self._ensure_graph()

    async def cleanup(self) -> None:
        if self.checkpointer is not None:
            await close_checkpointer(self.checkpointer)
            self.checkpointer = None
        self.graph = None
```

Change `_ensure_graph` to:

```python
    async def _ensure_graph(self):
        if getattr(self, "graph", None) is not None:
            return self.graph

        if getattr(self, "checkpointer", None) is None:
            self.checkpointer = await create_redis_checkpointer()

        self.graph = self._build_graph()
        self._initialization_error = None
        return self.graph
```

In `execute`, replace:

```python
            graph = self._ensure_graph()
```

with:

```python
            graph = await self._ensure_graph()
```

- [ ] **Step 6: Update chat API session calls**

In `app/api/chat.py`, change:

```python
        success = rag_agent_service.clear_session(request.session_id)
```

to:

```python
        success = await rag_agent_service.clear_session(request.session_id)
```

Change:

```python
        history = rag_agent_service.get_session_history(session_id)
```

to:

```python
        history = await rag_agent_service.get_session_history(session_id)
```

- [ ] **Step 7: Initialize services in FastAPI lifespan**

In `app/main.py`, add imports:

```python
from app.services.aiops_service import aiops_service
from app.services.multi_agent_aiops_service import multi_agent_aiops_service
from app.services.rag_agent_service import rag_agent_service
```

After Milvus connects in lifespan startup, add:

```python
        logger.info("🔌 正在初始化 Redis checkpoint...")
        await aiops_service.initialize()
        await multi_agent_aiops_service.initialize()
        await rag_agent_service._initialize_checkpointer()
        logger.info("✅ Redis checkpoint 初始化完成")
```

Before `shutdown_cozeloop()` during shutdown, add:

```python
        await rag_agent_service.cleanup()
        await aiops_service.cleanup()
        await multi_agent_aiops_service.cleanup()
```

- [ ] **Step 8: Run service integration tests and verify they pass**

Run:

```bash
uv run pytest tests/test_redis_checkpoint.py -q
```

Expected: PASS.

## Task 4: Update Documentation And Run Regression Tests

**Files:**
- Modify: `docs/ONBOARDING.md`
- Modify: `README.md`

- [ ] **Step 1: Update documentation**

In `docs/ONBOARDING.md`, replace the memory-only checkpoint notes with:

```markdown
6. 会话 ID 被用作 LangGraph 的 `thread_id`，用于在 Redis checkpointer 中区分会话。
```

Replace:

```markdown
后端使用 MemorySaver，进程重启会丢失。前端 localStorage 只影响当前浏览器体验，不等于服务端持久会话。
```

with:

```markdown
后端使用 Redis checkpointer 存储 LangGraph checkpoint。默认本地连接是 `redis://:your-redis-password@localhost:6379/0`，通过 `REDIS_URL` 覆盖。只要 Redis 数据还在，FastAPI 进程重启后服务端会话 checkpoint 仍可继续读取。前端 localStorage 只影响当前浏览器体验，不等于服务端 checkpoint 存储。
```

In `README.md`, add Redis to environment requirements:

```markdown
- Redis 8 或 Redis Stack（需要 RedisJSON 和 RediSearch），本地默认 `redis://:your-redis-password@localhost:6379/0`
```

- [ ] **Step 2: Run targeted tests**

Run:

```bash
uv run pytest tests/test_redis_checkpoint.py tests/test_multi_agent_aiops_contract.py tests/test_llm_factory_and_observability.py -q
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 4: Run import smoke test**

Run:

```bash
uv run python - <<'PY'
from app.config import config
from app.core.checkpoint import create_redis_checkpointer
print(config.redis_url)
print(create_redis_checkpointer.__name__)
PY
```

Expected: prints the Redis URL and `create_redis_checkpointer`.

## Self-Review

- Spec coverage: the plan covers `REDIS_URL`, centralized factory, all three LangGraph services, async checkpointer usage, startup failure behavior, tests without live Redis, and documentation updates.
- Placeholder scan: no unresolved placeholder markers or deferred implementation steps remain.
- Type consistency: helper names are `create_redis_checkpointer`, `get_latest_checkpoint`, `delete_checkpoint_thread`, and `close_checkpointer` throughout; async service methods are awaited in API and lifespan.
