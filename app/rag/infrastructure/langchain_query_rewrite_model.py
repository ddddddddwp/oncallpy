"""LangChain adapter for the optional structured query-rewrite model."""

from __future__ import annotations

import json
import math
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _EntityCandidateSchema(BaseModel):
    """Untrusted model output, bounded before it reaches query-domain code."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    field: str = Field(min_length=1, max_length=80)
    value: str = Field(min_length=1, max_length=160)
    evidence: str = Field(min_length=1, max_length=500)


class _QueryRewriteSchema(BaseModel):
    """The only response shape accepted from LangChain structured output."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    primary_text: str = Field(min_length=1, max_length=500)
    subqueries: list[str] = Field(default_factory=list, max_length=16)
    entity_candidates: list[_EntityCandidateSchema] = Field(default_factory=list, max_length=16)
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(max_length=500)

    @field_validator("primary_text", "subqueries", mode="after")
    @classmethod
    def _reject_blank_strings(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("must not be blank")
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, str) or not item.strip() or len(item) > 500:
                    raise ValueError("subqueries must contain bounded non-blank strings")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _strict_finite_confidence(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("confidence must be a finite JSON number")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("confidence must be a finite JSON number")
        return normalized


_SYSTEM_PROMPT = """Rewrite only for retrieval. Preserve every technical identifier exactly.
Return one primary query and at most max_subqueries focused subqueries.
Do not invent devices, manufacturers, models, alarms, register addresses, or function codes.
Entity candidates require an exact evidence substring from the supplied text or context.
Do not output database filters or retrieval settings."""


class LangChainQueryRewriteModel:
    """Adapt a LangChain chat model to the query module's small model protocol."""

    def __init__(self, chat_model: Any) -> None:
        self._structured_model = chat_model.with_structured_output(
            _QueryRewriteSchema,
            method="function_calling",
        )

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        response = self._structured_model.invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=serialized)]
        )
        if not isinstance(response, _QueryRewriteSchema):
            raise TypeError("structured query rewrite response has unexpected type")
        return response.model_dump()
