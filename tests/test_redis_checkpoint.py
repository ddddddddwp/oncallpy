from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_isolated_settings_probe(tmp_path: Path, redis_url: str) -> None:
    fake_config = tmp_path / "config.yaml"
    fake_config.write_text("{}\n", encoding="utf-8")
    environment = {
        "ENERGY_AGENT_CONFIG_FILES": str(fake_config),
        "EXPECTED_REDIS_URL": redis_url,
        "HOME": str(tmp_path),
        "PYTHONPATH": str(PROJECT_ROOT),
    }
    if redis_url != "redis://localhost:6379/0":
        environment["REDIS_URL"] = redis_url
    probe = """
import os
from urllib.parse import urlsplit
from app.config import Settings, config

expected = os.environ["EXPECTED_REDIS_URL"]
assert config.redis_url == expected
assert Settings(_env_file=None).redis_url == expected
parsed = urlsplit(config.redis_url)
if expected == "redis://localhost:6379/0":
    assert parsed.username is None
    assert parsed.password is None
    assert "@" not in parsed.netloc
"""

    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_settings_defaults_redis_url_to_local_docker(tmp_path: Path) -> None:
    _run_isolated_settings_probe(tmp_path, "redis://localhost:6379/0")


def test_settings_reads_redis_url_env(tmp_path: Path) -> None:
    _run_isolated_settings_probe(
        tmp_path,
        "redis://:fake-env-password@redis.invalid:6379/2",
    )


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


def _fresh_import(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _stub_rag_import_dependencies(monkeypatch):
    fake_tools = ModuleType("app.tools")
    fake_tools.get_current_time = object()
    fake_tools.retrieve_knowledge = object()
    monkeypatch.setitem(sys.modules, "app.tools", fake_tools)


def _stub_aiops_import_dependencies(monkeypatch):
    fake_aiops = ModuleType("app.agent.aiops")

    async def fake_node(state):
        return {}

    fake_aiops.PlanExecuteState = dict
    fake_aiops.planner = fake_node
    fake_aiops.executor = fake_node
    fake_aiops.replanner = fake_node
    monkeypatch.setitem(sys.modules, "app.agent.aiops", fake_aiops)


@pytest.mark.asyncio
async def test_rag_service_initializes_checkpointer_with_factory(monkeypatch):
    _stub_rag_import_dependencies(monkeypatch)
    rag_module = _fresh_import("app.rag.rag_agent_service")

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
    _stub_rag_import_dependencies(monkeypatch)
    rag_module = _fresh_import("app.rag.rag_agent_service")

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
    _stub_rag_import_dependencies(monkeypatch)
    rag_module = _fresh_import("app.rag.rag_agent_service")

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
    _stub_aiops_import_dependencies(monkeypatch)
    aiops_module = _fresh_import("app.services.aiops_service")

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
    multi_module = _fresh_import("app.services.multi_agent_aiops_service")

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
