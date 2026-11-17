"""Tests for structured query-pipeline diagnostics."""

from __future__ import annotations

import json
from copy import deepcopy
from io import StringIO

import pytest

from app.rag.query.debugger import (
    ConsoleQueryDebugger,
    MemoryQueryDebugger,
    NoOpQueryDebugger,
    sanitize_trace_data,
)
from app.rag.query.models import QueryTraceEvent


def test_noop_debugger_discards_events() -> None:
    assert NoOpQueryDebugger().emit(QueryTraceEvent(stage="analyze", status="started")) is None


def test_memory_debugger_keeps_events_in_emit_order() -> None:
    debugger = MemoryQueryDebugger()
    debugger.emit(QueryTraceEvent(stage="analyze", status="started"))
    debugger.emit(QueryTraceEvent(stage="analyze", status="completed"))

    assert tuple(event.status for event in debugger.events) == ("started", "completed")


def test_sanitizer_masks_credentials_and_only_truncates_preview_fields() -> None:
    data = {
        "api_key": "api-secret",
        "document": "x" * 100,
        "metadata_filter": "y" * 100,
    }

    sanitized = sanitize_trace_data(data, preview_length=12)

    assert sanitized == {
        "api_key": "***",
        "document": "x" * 12 + "…",
        "metadata_filter": "y" * 100,
    }


def test_sanitizer_recurses_without_matching_secret_text_or_partial_preview_keys() -> None:
    data = {
        "nested": [
            {"Token": "one", "ordinary": "contains secret but stays"},
            ( {"password": "two", "SECRET": "three"}, ),
        ],
        "Preview": "abcdef",
        "document_type": "abcdef",
        "CONTENT": "uvwxyz",
    }

    sanitized = sanitize_trace_data(data, preview_length=3)

    assert sanitized["nested"] == [
        {"Token": "***", "ordinary": "contains secret but stays"},
        ({"password": "***", "SECRET": "***"},),
    ]
    assert sanitized["Preview"] == "abc…"
    assert sanitized["CONTENT"] == "uvw…"
    assert sanitized["document_type"] == "abcdef"


def test_sanitizer_handles_cycles_depth_limits_and_shared_values() -> None:
    circular_dict: dict[str, object] = {}
    circular_dict["self"] = circular_dict
    circular_list: list[object] = []
    circular_list.append(circular_list)
    shared = {"value": "kept"}
    nested: dict[str, object] = {"next": {"next": {"next": "too-deep"}}}

    sanitized = sanitize_trace_data(
        {
            "dict": circular_dict,
            "list": circular_list,
            "left": shared,
            "right": shared,
            "nested": nested,
        },
        max_depth=2,
    )

    assert sanitized["dict"] == {"self": "<circular>"}
    assert sanitized["list"] == ["<circular>"]
    assert sanitized["left"] == {"value": "kept"}
    assert sanitized["right"] == {"value": "kept"}
    assert sanitized["nested"] == {"next": {"next": "<max-depth>"}}


def test_sanitizer_masks_normalized_credential_key_variants_without_auth_overreach() -> None:
    data = {
        "Api-Key": "one",
        "access token": "two",
        "PASSWORD": "three",
        "client_secret": "four",
        "Authorization": "five",
        "apikey": "six",
        "passwd": "seven",
        "credentials": "eight",
        "authorization_hint": "remains",
        "author": "remains",
        "nested": {"Credential": "nine"},
    }

    sanitized = sanitize_trace_data(data)

    assert {key: sanitized[key] for key in data if key not in {"authorization_hint", "author", "nested"}} == {
        "Api-Key": "***",
        "access token": "***",
        "PASSWORD": "***",
        "client_secret": "***",
        "Authorization": "***",
        "apikey": "***",
        "passwd": "***",
        "credentials": "***",
    }
    assert sanitized["authorization_hint"] == "remains"
    assert sanitized["author"] == "remains"
    assert sanitized["nested"] == {"Credential": "***"}


def test_sanitizer_and_memory_debugger_do_not_mutate_or_share_event_data() -> None:
    data = {"nested": {"items": ["first"]}}
    original = deepcopy(data)
    debugger = MemoryQueryDebugger()
    debugger.emit(QueryTraceEvent(stage="plan", status="completed", data=data))
    data["nested"]["items"].append("later")

    assert data == {"nested": {"items": ["first", "later"]}}
    assert original == {"nested": {"items": ["first"]}}
    assert debugger.events[0].data == original

    exposed = debugger.events[0].data
    exposed["nested"]["items"].append("consumer-change")
    assert debugger.events[0].data == original


def test_console_debugger_writes_sanitized_ndjson_without_closing_stream() -> None:
    stream = StringIO()
    debugger = ConsoleQueryDebugger(stream)
    debugger.emit(
        QueryTraceEvent(
            stage="rewrite",
            status="started",
            duration_ms=1.5,
            data={"text": "中文", "secret": "do-not-print"},
        )
    )
    debugger.emit(QueryTraceEvent(stage="rewrite", status="completed", data={"count": 2}))

    lines = stream.getvalue().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first == {
        "stage": "rewrite",
        "status": "started",
        "duration_ms": 1.5,
        "data": {"text": "中文", "secret": "***"},
    }
    assert second == {
        "stage": "rewrite",
        "status": "completed",
        "duration_ms": 0.0,
        "data": {"count": 2},
    }
    stream.write("still-open")
    assert stream.getvalue().endswith("still-open")


def test_console_debugger_serializes_non_finite_floats_as_json_null() -> None:
    stream = StringIO()
    ConsoleQueryDebugger(stream).emit(
        QueryTraceEvent(
            stage="execute",
            status="completed",
            duration_ms=float("nan"),
            data={"nan": float("nan"), "positive": float("inf"), "negative": -float("inf")},
        )
    )

    def reject_non_json_constant(value: str) -> None:
        raise ValueError(value)

    assert json.loads(stream.getvalue(), parse_constant=reject_non_json_constant) == {
        "stage": "execute",
        "status": "completed",
        "duration_ms": None,
        "data": {"nan": None, "positive": None, "negative": None},
    }


class ShortWritingStream:
    """A text stream whose write method deliberately accepts a short prefix."""

    def __init__(self, chunk_size: int) -> None:
        self.chunk_size = chunk_size
        self.value = ""

    def write(self, text: str) -> int:
        accepted = text[: self.chunk_size]
        self.value += accepted
        return len(accepted)


def test_console_debugger_retries_short_writes_until_the_event_is_complete() -> None:
    stream = ShortWritingStream(chunk_size=4)
    ConsoleQueryDebugger(stream).emit(QueryTraceEvent(stage="plan", status="started"))  # type: ignore[arg-type]

    assert json.loads(stream.value) == {
        "stage": "plan",
        "status": "started",
        "duration_ms": 0.0,
        "data": {},
    }


@pytest.mark.parametrize(
    ("preview_length", "expected", "error"),
    [
        (0, "…", None),
        (-1, None, "preview_length must be a non-negative integer"),
        (True, None, "preview_length must be a non-negative integer"),
        (1.5, None, "preview_length must be a non-negative integer"),
    ],
)
def test_sanitizer_validates_preview_length(
    preview_length: object,
    expected: str | None,
    error: str | None,
) -> None:
    if error is not None:
        with pytest.raises(ValueError, match=error):
            sanitize_trace_data({"preview": "abc"}, preview_length=preview_length)  # type: ignore[arg-type]
        return

    assert sanitize_trace_data({"preview": "abc"}, preview_length=preview_length) == {
        "preview": expected
    }


def test_console_debugger_uses_injected_preview_length() -> None:
    stream = StringIO()

    ConsoleQueryDebugger(stream, preview_length=3).emit(
        QueryTraceEvent(
            stage="retrieve",
            status="completed",
            data={"document": "abcdef"},
        )
    )

    assert json.loads(stream.getvalue())["data"]["document"] == "abc…"


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_console_debugger_rejects_invalid_preview_length(value: object) -> None:
    with pytest.raises(ValueError, match="preview_length must be a non-negative integer"):
        ConsoleQueryDebugger(StringIO(), preview_length=value)  # type: ignore[arg-type]
