from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings


def test_settings_reads_minimax_and_cozeloop_env(monkeypatch):
    monkeypatch.setenv("CHAT_PROVIDER", "minimax")
    monkeypatch.setenv("MINIMAX_API_KEY", "mini-key")
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimax.local/v1")
    monkeypatch.setenv("MINIMAX_MODEL", "MiniMax-M2")
    monkeypatch.setenv("COZELOOP_WORKSPACE_ID", "ws-abc")
    monkeypatch.setenv("COZELOOP_API_TOKEN", "token-xyz")
    monkeypatch.setenv("COZELOOP_API_BASE_URL", "https://loop.example/api")
    monkeypatch.setenv("LOG_DIR", "/ast/oncallpy-logs")

    settings = Settings()

    assert settings.chat_provider == "minimax"
    assert settings.minimax_api_key == "mini-key"
    assert settings.minimax_base_url == "https://api.minimax.local/v1"
    assert settings.minimax_model == "MiniMax-M2"
    assert settings.cozeloop_workspace_id == "ws-abc"
    assert settings.cozeloop_api_token == "token-xyz"
    assert settings.cozeloop_api_base_url == "https://loop.example/api"
    assert settings.log_dir == "/ast/oncallpy-logs"


def test_settings_reads_deepseek_env(monkeypatch):
    monkeypatch.setenv("CHAT_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.local")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("RAG_MODEL", "deepseek-v4-flash")

    settings = Settings()

    assert settings.chat_provider == "deepseek"
    assert settings.deepseek_api_key == "deepseek-key"
    assert settings.deepseek_base_url == "https://api.deepseek.local"
    assert settings.deepseek_model == "deepseek-v4-flash"
    assert settings.rag_model == "deepseek-v4-flash"


def test_app_logger_uses_configured_log_dir(monkeypatch):
    import app.utils.logger as logger_module

    monkeypatch.setattr(logger_module.config, "log_dir", "runtime-logs", raising=False)

    assert logger_module.resolve_app_log_path() == Path("runtime-logs") / "app_{time:YYYY-MM-DD}.log"


def test_llm_factory_defaults_to_minimax(monkeypatch):
    captured = {}

    class DummyChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs.get("model")

    import app.core.llm_factory as factory

    monkeypatch.setattr(factory, "ChatOpenAI", DummyChatOpenAI)
    monkeypatch.setattr(factory.config, "chat_provider", "minimax")
    monkeypatch.setattr(factory.config, "minimax_api_key", "mini-key")
    monkeypatch.setattr(factory.config, "minimax_base_url", "https://api.minimax.local/v1")
    monkeypatch.setattr(factory.config, "minimax_model", "MiniMax-M2")

    model = factory.LLMFactory.create_chat_model()

    assert isinstance(model, DummyChatOpenAI)
    assert captured["api_key"] == "mini-key"
    assert captured["base_url"] == "https://api.minimax.local/v1"
    assert captured["model"] == "MiniMax-M2"
    assert captured["timeout"] is None


def test_llm_factory_passes_optional_timeout(monkeypatch):
    captured = {}

    class DummyChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import app.core.llm_factory as factory

    monkeypatch.setattr(factory, "ChatOpenAI", DummyChatOpenAI)

    factory.LLMFactory.create_chat_model(timeout=2.5)

    assert captured["timeout"] == 2.5


def test_llm_factory_uses_deepseek_provider(monkeypatch):
    captured = {}

    class DummyChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = kwargs.get("model")

    import app.core.llm_factory as factory

    monkeypatch.setattr(factory, "ChatOpenAI", DummyChatOpenAI)
    monkeypatch.setattr(factory.config, "chat_provider", "deepseek")
    monkeypatch.setattr(factory.config, "deepseek_api_key", "deepseek-key", raising=False)
    monkeypatch.setattr(factory.config, "deepseek_base_url", "https://api.deepseek.local", raising=False)
    monkeypatch.setattr(factory.config, "deepseek_model", "deepseek-v4-flash", raising=False)

    model = factory.LLMFactory.create_chat_model()

    assert isinstance(model, DummyChatOpenAI)
    assert captured["api_key"] == "deepseek-key"
    assert captured["base_url"] == "https://api.deepseek.local"
    assert captured["model"] == "deepseek-v4-flash"
    assert captured["extra_body"] is None


@pytest.mark.asyncio
async def test_legacy_aiops_planner_uses_llm_factory(monkeypatch):
    import app.agent.aiops.planner as planner_module

    call_state = {"called": False}

    class FakeStructuredLLM:
        async def ainvoke(self, *_args, **_kwargs):
            return {"steps": ["step-1", "step-2"]}

    class FakeLLM:
        def with_structured_output(self, *args, **kwargs):
            call_state["called"] = True
            return FakeStructuredLLM()

    class DummyKnowledgeTool:
        async def ainvoke(self, *_args, **_kwargs):
            return ""

    class DummyMCPTool:
        async def get_tools(self):
            return []

    async def get_client():
        return DummyMCPTool()

    monkeypatch.setattr(planner_module, "llm_factory", SimpleNamespace(create_chat_model=lambda **_: FakeLLM()))
    monkeypatch.setattr(planner_module, "retrieve_knowledge", DummyKnowledgeTool())
    monkeypatch.setattr(planner_module, "get_mcp_client_with_retry", get_client)

    result = await planner_module.planner({"input": "检查 CPU 异常"})

    assert call_state["called"] is True
    assert "plan" in result
    assert result["plan"] == ["step-1", "step-2"]


@pytest.mark.asyncio
async def test_multi_agent_planner_uses_llm_factory(monkeypatch):
    import app.agent.aiops_multi.nodes.planner as planner_module

    call_state = {"called": False}

    class FakeStructuredLLM:
        async def ainvoke(self, *_args, **_kwargs):
            return {"steps": ["a", "b"]}

    class FakeLLM:
        def with_structured_output(self, *args, **kwargs):
            call_state["called"] = True
            return FakeStructuredLLM()

    async def fake_available_tools(include_mcp=True):
        return [], []

    monkeypatch.setattr(planner_module, "llm_factory", SimpleNamespace(create_chat_model=lambda **_: FakeLLM()))
    monkeypatch.setattr(planner_module, "get_available_tools", fake_available_tools)

    result = await planner_module.planner({"input": "检查 CPU 异常", "tool_descriptions": ""})

    assert call_state["called"] is True
    assert result["plan"] == ["a", "b"]


def test_cozeloop_missing_sdk_not_breaking(monkeypatch):
    import app.core.observability as observability

    monkeypatch.setattr(observability.config, "cozeloop_workspace_id", "ws-abc")
    monkeypatch.setattr(observability.config, "cozeloop_api_token", "token-abc")
    monkeypatch.setattr(observability, "_load_cozeloop_module", lambda: None)

    assert observability.initialize_cozeloop() is False
    assert observability.is_cozeloop_enabled() is False

    with observability.trace_span("startup"):
        pass


def test_cozeloop_initialize_creates_client_with_api_token(monkeypatch):
    import app.core.observability as observability

    class FakeClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeModule:
        def __init__(self):
            self.client = FakeClient()
            self.new_client_kwargs = None

        def new_client(self, **kwargs):
            self.new_client_kwargs = kwargs
            return self.client

    fake_module = FakeModule()
    monkeypatch.setattr(observability.config, "cozeloop_workspace_id", "ws-abc")
    monkeypatch.setattr(observability.config, "cozeloop_api_token", "token-abc")
    monkeypatch.setattr(observability.config, "cozeloop_api_base_url", "https://loop.example")
    monkeypatch.setattr(observability, "_load_cozeloop_module", lambda: fake_module)

    assert observability.initialize_cozeloop() is True

    assert observability.is_cozeloop_enabled() is True
    assert fake_module.new_client_kwargs == {
        "workspace_id": "ws-abc",
        "api_token": "token-abc",
        "api_base_url": "https://loop.example",
    }

    observability.shutdown_cozeloop()

    assert fake_module.client.closed is True
    assert observability.is_cozeloop_enabled() is False


def test_trace_span_uses_initialized_cozeloop_client(monkeypatch):
    import app.core.observability as observability

    class FakeSpan:
        def __init__(self):
            self.input_data = None
            self.output_data = None
            self.finished = False

        def set_input(self, data):
            self.input_data = data

        def set_output(self, data):
            self.output_data = data

        def finish(self):
            self.finished = True

    class FakeClient:
        def __init__(self):
            self.span = FakeSpan()
            self.start_calls = []

        def start_span(self, name, span_type):
            self.start_calls.append((name, span_type))
            return self.span

        def close(self):
            return None

    class FakeModule:
        def __init__(self):
            self.client = FakeClient()

        def new_client(self, **_kwargs):
            return self.client

    fake_module = FakeModule()
    monkeypatch.setattr(observability.config, "cozeloop_workspace_id", "ws-abc")
    monkeypatch.setattr(observability.config, "cozeloop_api_token", "token-abc")
    monkeypatch.setattr(observability.config, "cozeloop_api_base_url", "")
    monkeypatch.setattr(observability, "_load_cozeloop_module", lambda: fake_module)
    observability.initialize_cozeloop()

    with observability.trace_span(
        "chat_request",
        span_type="custom",
        input_data={"session_id": "s1"},
        set_output=lambda: {"status": "success"},
    ):
        pass

    assert fake_module.client.start_calls == [("chat_request", "custom")]
    assert fake_module.client.span.input_data == {"session_id": "s1"}
    assert fake_module.client.span.output_data == {"status": "success"}
    assert fake_module.client.span.finished is True


def test_cozeloop_runtime_span_failure_not_breaking(monkeypatch):
    import app.core.observability as observability

    class _BadModule:
        def new_client(self, **_kwargs):
            return self

        def start_span(self, *_args, **_kwargs):
            raise RuntimeError("cozeloop runtime failure")

        def close(self):  # pragma: no cover - 兜底行为
            return None

    monkeypatch.setattr(observability.config, "cozeloop_workspace_id", "ws-abc")
    monkeypatch.setattr(observability.config, "cozeloop_api_token", "token-abc")
    monkeypatch.setattr(observability, "_load_cozeloop_module", lambda: _BadModule())

    assert observability.initialize_cozeloop() is True
    assert observability.is_cozeloop_enabled() is True

    with observability.trace_span("runtime_failure"):
        pass


def test_new_turn_id_and_request_id():
    import app.core.observability as observability

    turn_id = observability.new_turn_id("session-123")
    request_id = observability.new_request_id()

    assert turn_id.startswith("session-123:")
    assert len(turn_id.split(":", 1)[1]) == 12
    assert len(request_id) == 32


def test_trace_span_merges_trace_context_and_sets_tags(monkeypatch):
    import app.core.observability as observability

    class FakeSpan:
        def __init__(self):
            self.input_data = None
            self.output_data = None
            self.finished = False
            self.thread_id = None
            self.tags = {}

        def set_input(self, data):
            self.input_data = data

        def set_output(self, data):
            self.output_data = data

        def set_tag_item(self, key, value):
            self.tags[key] = value

        def set_thread_id(self, thread_id):
            self.thread_id = thread_id

        def finish(self):
            self.finished = True

    class FakeClient:
        def __init__(self):
            self.start_calls = []
            self.spans = []

        def start_span(self, name, span_type):
            self.start_calls.append((name, span_type))
            span = FakeSpan()
            self.spans.append(span)
            return span

        def close(self):
            return None

    class FakeModule:
        def __init__(self):
            self.client = FakeClient()

        def new_client(self, **_kwargs):
            return self.client

    fake_module = FakeModule()
    monkeypatch.setattr(observability.config, "cozeloop_workspace_id", "ws-abc")
    monkeypatch.setattr(observability.config, "cozeloop_api_token", "token-abc")
    monkeypatch.setattr(observability, "_load_cozeloop_module", lambda: fake_module)

    observability.initialize_cozeloop()

    trace_context = {
        "session_id": "session-1",
        "turn_id": "session-1:abc123",
        "request_id": "req-1",
        "api": "chat",
    }

    with observability.trace_span(
        "chat_request",
        input_data={"question": "hi"},
        trace_context=trace_context,
    ) as _span:
        assert observability.get_trace_context()["session_id"] == "session-1"
        with observability.trace_span("inner_span") as inner_span:
            assert inner_span.input_data["session_id"] == "session-1"

    assert observability.get_trace_context() == {}
    assert fake_module.client.start_calls == [("chat_request", "custom"), ("inner_span", "custom")]
    assert fake_module.client.spans[0].thread_id == "session-1"
    assert fake_module.client.spans[0].input_data["session_id"] == "session-1"
    assert fake_module.client.spans[0].input_data["turn_id"] == "session-1:abc123"
    assert fake_module.client.spans[0].input_data["request_id"] == "req-1"
    assert fake_module.client.spans[0].input_data["api"] == "chat"
    assert fake_module.client.spans[0].input_data["question"] == "hi"
    assert fake_module.client.spans[0].tags == {
        "session_id": "session-1",
        "turn_id": "session-1:abc123",
        "request_id": "req-1",
        "api": "chat",
    }
    assert fake_module.client.spans[0].finished is True
    assert fake_module.client.spans[1].input_data["session_id"] == "session-1"
    assert fake_module.client.spans[1].input_data["turn_id"] == "session-1:abc123"
    assert fake_module.client.spans[1].input_data["api"] == "chat"
    assert "question" not in fake_module.client.spans[1].input_data
