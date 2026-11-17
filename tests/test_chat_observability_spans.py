from __future__ import annotations

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import app.api.chat as chat_module
from app.models.request import ChatRequest, ClearRequest
from app.models.response import ApiResponse, SessionInfoResponse


class FakeSpan:
    def __init__(self):
        self.input_data = None
        self.output_data = None

    def set_input(self, data):
        self.input_data = data

    def set_output(self, data):
        self.output_data = data

    def finish(self):
        self.finished = True


class FakeEventSourceResponse:
    def __init__(self, body_iterator):
        self.body_iterator = body_iterator


def _prepare_fake_span_hooks():
    calls: list[dict] = []
    spans: list[FakeSpan] = []

    @contextmanager
    def fake_trace_span(
        name: str,
        span_type: str = "custom",
        input_data=None,
        set_output=None,
        trace_context=None,
    ):
        del set_output
        calls.append(
            {
                "name": name,
                "span_type": span_type,
                "input_data": input_data,
                "trace_context": trace_context,
            }
        )
        span = FakeSpan()
        spans.append(span)
        yield span

    return calls, spans, fake_trace_span


@pytest.mark.asyncio
async def test_chat_endpoint_records_session_turn_request_id_in_span_input_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def fake_query(question: str, session_id: str) -> str:
        calls.append((question, session_id))
        return "mock answer"

    calls_span, spans, fake_trace_span = _prepare_fake_span_hooks()
    fake_service = SimpleNamespace(query=fake_query)
    monkeypatch.setattr(chat_module, "rag_agent_service", fake_service)
    monkeypatch.setattr(chat_module, "new_request_id", lambda: "req-001")
    monkeypatch.setattr(chat_module, "new_turn_id", lambda _session_id: "session-001:turn-abc")
    monkeypatch.setattr(chat_module, "trace_span", fake_trace_span)

    response = await chat_module.chat(ChatRequest(id="session-001", question="今天的告警怎么处理？"))

    assert response == {
        "code": 200,
        "message": "success",
        "data": {"success": True, "answer": "mock answer", "errorMessage": None},
    }
    assert calls == [("今天的告警怎么处理？", "session-001")]
    assert len(calls_span) == 1
    assert calls_span[0]["name"] == "chat_request"
    assert calls_span[0]["span_type"] == "custom"
    assert calls_span[0]["trace_context"] == {
        "session_id": "session-001",
        "request_id": "req-001",
        "turn_id": "session-001:turn-abc",
        "api": "chat",
    }
    assert calls_span[0]["input_data"]["session_id"] == "session-001"
    assert calls_span[0]["input_data"]["question"] == "今天的告警怎么处理？"
    assert calls_span[0]["input_data"]["streaming"] is False
    assert calls_span[0]["input_data"]["request_id"] == "req-001"
    assert calls_span[0]["input_data"]["turn_id"] == "session-001:turn-abc"
    assert spans[0].output_data == {
        "status": "success",
        "session_id": "session-001",
        "request_id": "req-001",
        "turn_id": "session-001:turn-abc",
    }


@pytest.mark.asyncio
async def test_chat_stream_endpoint_records_session_turn_request_id_in_span_input_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def fake_query_stream(question: str, session_id: str):
        calls.append((question, session_id))
        yield {"type": "content", "data": "chunk-1"}
        yield {"type": "complete", "data": {"answer": "mock stream answer"}}

    calls_span, spans, fake_trace_span = _prepare_fake_span_hooks()
    fake_service = SimpleNamespace(query_stream=fake_query_stream)
    monkeypatch.setattr(chat_module, "rag_agent_service", fake_service)
    monkeypatch.setattr(chat_module, "new_request_id", lambda: "req-stream-001")
    monkeypatch.setattr(
        chat_module,
        "new_turn_id",
        lambda _session_id: "session-001:stream-turn-abc",
    )
    monkeypatch.setattr(chat_module, "trace_span", fake_trace_span)
    monkeypatch.setattr(chat_module, "EventSourceResponse", FakeEventSourceResponse)

    response = await chat_module.chat_stream(
        ChatRequest(id="session-001", question="今天有异常日志么？")
    )

    events: list[dict] = []
    async for event in response.body_iterator:
        events.append(event)

    assert calls == [("今天有异常日志么？", "session-001")]
    assert len(calls_span) == 1
    assert calls_span[0]["name"] == "chat_stream_request"
    assert calls_span[0]["trace_context"] == {
        "session_id": "session-001",
        "request_id": "req-stream-001",
        "turn_id": "session-001:stream-turn-abc",
        "api": "chat_stream",
    }
    assert calls_span[0]["input_data"]["session_id"] == "session-001"
    assert calls_span[0]["input_data"]["question"] == "今天有异常日志么？"
    assert calls_span[0]["input_data"]["request_id"] == "req-stream-001"
    assert calls_span[0]["input_data"]["turn_id"] == "session-001:stream-turn-abc"
    assert spans[0].output_data == {
        "status": "success",
        "session_id": "session-001",
        "request_id": "req-stream-001",
        "turn_id": "session-001:stream-turn-abc",
    }

    decoded = [
        json.loads(item["data"]) if isinstance(item["data"], str) else item["data"]
        for item in events
    ]
    assert decoded[0]["type"] == "content"
    assert decoded[1]["type"] == "done"
    assert decoded[1]["data"]["answer"] == "mock stream answer"


@pytest.mark.asyncio
async def test_chat_stream_endpoint_records_error_output_on_query_stream_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def fake_query_stream(question: str, session_id: str):
        calls.append((question, session_id))
        raise RuntimeError("upstream failed")
        if False:
            yield {"type": "skip", "data": None}

    calls_span, spans, fake_trace_span = _prepare_fake_span_hooks()
    fake_service = SimpleNamespace(query_stream=fake_query_stream)
    monkeypatch.setattr(chat_module, "rag_agent_service", fake_service)
    monkeypatch.setattr(chat_module, "new_request_id", lambda: "req-stream-err-001")
    monkeypatch.setattr(
        chat_module,
        "new_turn_id",
        lambda _session_id: "session-001:stream-turn-error",
    )
    monkeypatch.setattr(chat_module, "trace_span", fake_trace_span)
    monkeypatch.setattr(chat_module, "EventSourceResponse", FakeEventSourceResponse)

    response = await chat_module.chat_stream(
        ChatRequest(id="session-001", question="触发查询异常")
    )

    events: list[dict] = []
    async for event in response.body_iterator:
        events.append(event)

    assert calls == [("触发查询异常", "session-001")]
    assert len(calls_span) == 1
    assert calls_span[0]["name"] == "chat_stream_request"
    assert calls_span[0]["trace_context"] == {
        "session_id": "session-001",
        "request_id": "req-stream-err-001",
        "turn_id": "session-001:stream-turn-error",
        "api": "chat_stream",
    }
    assert spans[0].output_data == {
        "status": "error",
        "session_id": "session-001",
        "request_id": "req-stream-err-001",
        "turn_id": "session-001:stream-turn-error",
    }

    decoded = [
        json.loads(item["data"]) if isinstance(item["data"], str) else item["data"]
        for item in events
    ]
    assert decoded == [{"type": "error", "data": "upstream failed"}]


@pytest.mark.asyncio
async def test_chat_endpoint_records_error_output_on_query_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    async def fake_query(question: str, session_id: str) -> str:
        calls.append((question, session_id))
        raise RuntimeError("chat failed")

    calls_span, spans, fake_trace_span = _prepare_fake_span_hooks()
    fake_service = SimpleNamespace(query=fake_query)
    monkeypatch.setattr(chat_module, "rag_agent_service", fake_service)
    monkeypatch.setattr(chat_module, "new_request_id", lambda: "req-chat-err-001")
    monkeypatch.setattr(
        chat_module,
        "new_turn_id",
        lambda _session_id: "session-001:chat-turn-error",
    )
    monkeypatch.setattr(chat_module, "trace_span", fake_trace_span)

    response = await chat_module.chat(
        ChatRequest(id="session-001", question="chat 异常测试")
    )

    assert calls == [("chat 异常测试", "session-001")]
    assert response == {
        "code": 500,
        "message": "error",
        "data": {
            "success": False,
            "answer": None,
            "errorMessage": "chat failed",
        },
    }
    assert len(calls_span) == 1
    assert calls_span[0]["name"] == "chat_request"
    assert calls_span[0]["trace_context"] == {
        "session_id": "session-001",
        "request_id": "req-chat-err-001",
        "turn_id": "session-001:chat-turn-error",
        "api": "chat",
    }
    assert spans[0].output_data == {
        "status": "error",
        "session_id": "session-001",
        "request_id": "req-chat-err-001",
        "turn_id": "session-001:chat-turn-error",
    }


@pytest.mark.asyncio
async def test_clear_session_endpoint_records_session_turn_request_id_in_span_input_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def fake_clear_session(session_id: str) -> bool:
        calls.append(session_id)
        return True

    calls_span, spans, fake_trace_span = _prepare_fake_span_hooks()
    fake_service = SimpleNamespace(clear_session=fake_clear_session)
    monkeypatch.setattr(chat_module, "rag_agent_service", fake_service)
    monkeypatch.setattr(chat_module, "new_request_id", lambda: "req-clear-001")
    monkeypatch.setattr(
        chat_module,
        "new_turn_id",
        lambda _session_id: "session-001:clear-turn-abc",
    )
    monkeypatch.setattr(chat_module, "trace_span", fake_trace_span)

    response = await chat_module.clear_session(ClearRequest(session_id="session-001"))

    assert calls == ["session-001"]
    assert calls_span[0]["name"] == "chat_clear_request"
    assert calls_span[0]["trace_context"] == {
        "session_id": "session-001",
        "request_id": "req-clear-001",
        "turn_id": "session-001:clear-turn-abc",
        "api": "chat_clear",
    }
    assert calls_span[0]["input_data"]["session_id"] == "session-001"
    assert calls_span[0]["input_data"]["request_id"] == "req-clear-001"
    assert calls_span[0]["input_data"]["turn_id"] == "session-001:clear-turn-abc"
    assert spans[0].output_data == {
        "status": "success",
        "session_id": "session-001",
        "request_id": "req-clear-001",
        "turn_id": "session-001:clear-turn-abc",
    }
    assert response == ApiResponse(
        status="success",
        message="会话已清空",
        data=None,
    )


@pytest.mark.asyncio
async def test_clear_session_endpoint_records_error_output_on_clear_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_clear_session(session_id: str) -> bool:
        del session_id
        raise RuntimeError("clear failed")

    calls_span, spans, fake_trace_span = _prepare_fake_span_hooks()
    fake_service = SimpleNamespace(clear_session=fake_clear_session)
    monkeypatch.setattr(chat_module, "rag_agent_service", fake_service)
    monkeypatch.setattr(chat_module, "new_request_id", lambda: "req-clear-err-001")
    monkeypatch.setattr(
        chat_module,
        "new_turn_id",
        lambda _session_id: "session-001:clear-turn-error",
    )
    monkeypatch.setattr(chat_module, "trace_span", fake_trace_span)

    with pytest.raises(HTTPException) as exc:
        await chat_module.clear_session(ClearRequest(session_id="session-001"))

    assert exc.value.status_code == 500
    assert exc.value.detail == "clear failed"
    assert calls_span[0]["name"] == "chat_clear_request"
    assert calls_span[0]["trace_context"] == {
        "session_id": "session-001",
        "request_id": "req-clear-err-001",
        "turn_id": "session-001:clear-turn-error",
        "api": "chat_clear",
    }
    assert spans[0].output_data == {
        "status": "error",
        "session_id": "session-001",
        "request_id": "req-clear-err-001",
        "turn_id": "session-001:clear-turn-error",
    }


@pytest.mark.asyncio
async def test_get_session_info_endpoint_records_session_turn_request_id_in_span_input_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = [{"role": "user", "content": "hi"}]

    async def fake_get_session_history(session_id: str):
        assert session_id == "session-001"
        return history

    calls_span, spans, fake_trace_span = _prepare_fake_span_hooks()
    fake_service = SimpleNamespace(get_session_history=fake_get_session_history)
    monkeypatch.setattr(chat_module, "rag_agent_service", fake_service)
    monkeypatch.setattr(chat_module, "new_request_id", lambda: "req-info-001")
    monkeypatch.setattr(
        chat_module,
        "new_turn_id",
        lambda _session_id: "session-001:info-turn-abc",
    )
    monkeypatch.setattr(chat_module, "trace_span", fake_trace_span)

    response = await chat_module.get_session_info("session-001")

    assert calls_span[0]["name"] == "chat_session_info_request"
    assert calls_span[0]["trace_context"] == {
        "session_id": "session-001",
        "request_id": "req-info-001",
        "turn_id": "session-001:info-turn-abc",
        "api": "chat_session_info",
    }
    assert calls_span[0]["input_data"]["session_id"] == "session-001"
    assert calls_span[0]["input_data"]["request_id"] == "req-info-001"
    assert calls_span[0]["input_data"]["turn_id"] == "session-001:info-turn-abc"
    assert spans[0].output_data == {
        "status": "success",
        "session_id": "session-001",
        "request_id": "req-info-001",
        "turn_id": "session-001:info-turn-abc",
    }
    assert response == SessionInfoResponse(
        session_id="session-001",
        message_count=len(history),
        history=history,
    )


@pytest.mark.asyncio
async def test_get_session_info_endpoint_records_error_output_on_get_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_get_session_history(session_id: str):
        del session_id
        raise RuntimeError("get info failed")

    calls_span, spans, fake_trace_span = _prepare_fake_span_hooks()
    fake_service = SimpleNamespace(get_session_history=fake_get_session_history)
    monkeypatch.setattr(chat_module, "rag_agent_service", fake_service)
    monkeypatch.setattr(chat_module, "new_request_id", lambda: "req-info-err-001")
    monkeypatch.setattr(
        chat_module,
        "new_turn_id",
        lambda _session_id: "session-001:info-turn-error",
    )
    monkeypatch.setattr(chat_module, "trace_span", fake_trace_span)

    with pytest.raises(HTTPException) as exc:
        await chat_module.get_session_info("session-001")

    assert exc.value.status_code == 500
    assert exc.value.detail == "get info failed"
    assert calls_span[0]["name"] == "chat_session_info_request"
    assert calls_span[0]["trace_context"] == {
        "session_id": "session-001",
        "request_id": "req-info-err-001",
        "turn_id": "session-001:info-turn-error",
        "api": "chat_session_info",
    }
    assert spans[0].output_data == {
        "status": "error",
        "session_id": "session-001",
        "request_id": "req-info-err-001",
        "turn_id": "session-001:info-turn-error",
    }
