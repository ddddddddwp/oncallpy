"""Offline quality cases for deterministic analysis and rewrite decisions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.query.analyzer import DefaultQueryAnalyzer
from app.rag.query.models import QueryRequest, RewriteMode
from app.rag.query.rewrite_policy import DefaultRewritePolicy, RewritePolicySettings


@pytest.fixture(scope="module")
def quality_cases() -> list[dict[str, object]]:
    path = Path(__file__).with_name("fixtures") / "query_quality_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case_index", range(3))
def test_quality_cases_are_deterministic_and_offline(
    quality_cases: list[dict[str, object]], case_index: int
) -> None:
    case = quality_cases[case_index]
    request = QueryRequest(text=str(case["text"]), profile=str(case["profile"]), rewrite_mode=RewriteMode.AUTO)
    analysis = DefaultQueryAnalyzer().analyze(request)
    decision = DefaultRewritePolicy(RewritePolicySettings(llm_enabled=True)).decide(request, analysis)

    assert list(analysis.intents) == case["expected_intents"]
    expected_filter = case["expected_filter_subset"]
    assert isinstance(expected_filter, dict)
    assert expected_filter.items() <= analysis.inferred_filter.items()
    assert decision.use_llm is case["llm_expected"]
