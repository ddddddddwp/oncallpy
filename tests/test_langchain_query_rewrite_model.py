from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.rag.infrastructure.langchain_query_rewrite_model import (
    LangChainQueryRewriteModel,
    _QueryRewriteSchema,
)


class _StructuredModel:
    def __init__(self, response: object) -> None:
        self.response = response
        self.messages = None

    def invoke(self, messages):  # type: ignore[no-untyped-def]
        self.messages = messages
        return self.response


class _ChatModel:
    def __init__(self, response: object) -> None:
        self.structured = _StructuredModel(response)
        self.schemas: list[type[object]] = []
        self.methods: list[str | None] = []

    def with_structured_output(self, schema, *, method=None):  # type: ignore[no-untyped-def]
        self.schemas.append(schema)
        self.methods.append(method)
        return self.structured


def _response() -> _QueryRewriteSchema:
    return _QueryRewriteSchema(
        primary_text="查 0x3601",
        subqueries=["0x3601 的说明"],
        entity_candidates=[],
        confidence=0.9,
        reason="preserved identifier",
    )


def test_adapter_uses_one_strict_schema_and_stable_messages() -> None:
    chat = _ChatModel(_response())
    adapter = LangChainQueryRewriteModel(chat)

    result = adapter.generate({"b": "中文", "a": 1})

    assert chat.schemas == [_QueryRewriteSchema]
    assert chat.methods == ["function_calling"]
    assert result["primary_text"] == "查 0x3601"
    assert chat.structured.messages[0].content.startswith("Rewrite only for retrieval.")
    assert json.loads(chat.structured.messages[1].content) == {"a": 1, "b": "中文"}


def test_adapter_rejects_unexpected_structured_response_type() -> None:
    adapter = LangChainQueryRewriteModel(_ChatModel({"primary_text": "wrong"}))

    with pytest.raises(TypeError, match="unexpected type"):
        adapter.generate({})


@pytest.mark.parametrize(
    "payload",
    [
        {"confidence": True},
        {"confidence": float("inf")},
        {"confidence": float("nan")},
        {"primary_text": " "},
        {"subqueries": [" "]},
    ],
)
def test_schema_rejects_unsafe_values(payload: dict[str, object]) -> None:
    data = _response().model_dump()
    data.update(payload)

    with pytest.raises(ValidationError):
        _QueryRewriteSchema(**data)


@pytest.mark.parametrize("confidence", [0, 1, 0.0, 1.0, 0.25])
def test_schema_accepts_json_numbers_and_normalizes_confidence(confidence: int | float) -> None:
    data = _response().model_dump()
    data["confidence"] = confidence

    parsed = _QueryRewriteSchema(**data)

    assert parsed.confidence == float(confidence)
    assert type(parsed.confidence) is float
