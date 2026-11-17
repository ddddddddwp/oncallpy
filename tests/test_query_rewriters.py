"""Tests for deterministic and bounded query rewriting."""

from __future__ import annotations

import math

import pytest

from app.rag.models import RetrievalProfile
from app.rag.query.metadata_registry import MetadataFieldRegistry, MetadataFieldSpec
from app.rag.query.models import (
    EntityCandidate,
    QueryAnalysis,
    QueryRequest,
    QueryTurn,
    RewriteMode,
)
from app.rag.query.rewrite_policy import DefaultRewritePolicy, RewritePolicySettings
from app.rag.query.rewriters.llm_rewriter import LLMQueryRewriter, QueryRewriteValidationError
from app.rag.query.rewriters.rule_rewriter import RuleQueryRewriter


def make_analysis(**overrides: object) -> QueryAnalysis:
    values: dict[str, object] = {
        "normalized_text": "逆变器怎么排查？",
        "profile": RetrievalProfile.AIOPS,
        "confidence": 0.9,
    }
    values.update(overrides)
    return QueryAnalysis(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", (RewriteMode.OFF, RewriteMode.RULE, "unknown"))
def test_non_auto_modes_and_unknown_mode_do_not_use_llm(mode: RewriteMode | str) -> None:
    decision = DefaultRewritePolicy(RewritePolicySettings(llm_enabled=True)).decide(
        QueryRequest(text="排查", rewrite_mode=mode), make_analysis(confidence=0.1)
    )

    assert decision.use_llm is False


def test_auto_policy_uses_rules_for_simple_high_confidence_query() -> None:
    decision = DefaultRewritePolicy(RewritePolicySettings(llm_enabled=True)).decide(
        QueryRequest(text="逆变器怎么排查？"), make_analysis()
    )

    assert decision.use_llm is False
    assert decision.reasons == ()


def test_auto_policy_with_disabled_llm_keeps_simple_query_reason_free() -> None:
    decision = DefaultRewritePolicy(RewritePolicySettings(llm_enabled=False)).decide(
        QueryRequest(text="逆变器怎么排查？"), make_analysis()
    )

    assert decision.use_llm is False
    assert decision.reasons == ()


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    (
        ({"intents": ("troubleshooting", "verification")}, "multiple_intents"),
        ({"context_dependent": True}, "context_dependent"),
        ({"confidence": 0.74}, "low_confidence"),
        ({"complexity": "compound"}, "compound_query"),
    ),
)
def test_auto_policy_uses_llm_only_for_triggered_condition(
    overrides: dict[str, object], expected_reason: str
) -> None:
    decision = DefaultRewritePolicy(RewritePolicySettings(llm_enabled=True)).decide(
        QueryRequest(text="排查并验证", rewrite_mode=RewriteMode.AUTO), make_analysis(**overrides)
    )

    assert decision.use_llm is True
    assert decision.reasons == (expected_reason,)


def test_explicit_filter_does_not_auto_call_llm() -> None:
    decision = DefaultRewritePolicy(RewritePolicySettings(llm_enabled=True)).decide(
        QueryRequest(text="排查", metadata_filter={"device_type": "inverter"}),
        make_analysis(confidence=0.1),
    )

    assert decision.use_llm is False
    assert decision.reasons == ("explicit_metadata_filter",)


def test_explicit_llm_mode_reports_disabled_model() -> None:
    decision = DefaultRewritePolicy(RewritePolicySettings(llm_enabled=False)).decide(
        QueryRequest(text="排查", rewrite_mode=RewriteMode.LLM), make_analysis()
    )

    assert decision.use_llm is False
    assert decision.reasons == ("llm_disabled",)


def test_rule_rewriter_preserves_entities_without_generating_subqueries() -> None:
    request = QueryRequest(text="华为逆变器怎么排查？")
    analysis = make_analysis(
        entities={"manufacturer": "huawei", "device_type": "inverter", "duplicate": "huawei"}
    )

    rewrite = RuleQueryRewriter().rewrite(request, analysis)

    assert rewrite.primary_text == "逆变器怎么排查？"
    assert rewrite.subqueries == ()
    assert rewrite.preserved_terms == ("huawei", "inverter")
    assert rewrite.source == "rule"
    assert request.text == "华为逆变器怎么排查？"
    assert analysis.entities["manufacturer"] == "huawei"


def test_rule_rewriter_marks_off_source() -> None:
    rewrite = RuleQueryRewriter().rewrite(
        QueryRequest(text="原始问题", rewrite_mode=RewriteMode.OFF), make_analysis()
    )

    assert rewrite.source == "off"


def test_rule_rewriter_treats_invalid_mode_as_off() -> None:
    rewrite = RuleQueryRewriter().rewrite(
        QueryRequest(text="原始问题", rewrite_mode="invalid"), make_analysis()
    )

    assert rewrite.source == "off"


class FakeModel:
    def __init__(self, output: dict[str, object]) -> None:
        self.output = output
        self.payload: dict[str, object] | None = None

    def generate(self, payload: dict[str, object]) -> dict[str, object]:
        self.payload = payload
        return self.output


def test_llm_rewriter_rejects_oversized_payload_before_calling_model() -> None:
    model = FakeModel({"primary_text": "unused"})

    with pytest.raises(QueryRewriteValidationError, match="request.text"):
        LLMQueryRewriter(model).rewrite(QueryRequest(text="x" * 100_000), make_analysis())

    assert model.payload is None


@pytest.mark.parametrize(
    "analysis",
    (
        make_analysis(normalized_text="x" * 8_001),
        make_analysis(intents=("x",) * 9),
        make_analysis(entities={str(index): "value" for index in range(17)}),
    ),
)
def test_llm_rewriter_rejects_other_oversized_payload_parts_before_model_call(
    analysis: QueryAnalysis,
) -> None:
    model = FakeModel({"primary_text": "unused"})

    with pytest.raises(QueryRewriteValidationError):
        LLMQueryRewriter(model).rewrite(QueryRequest(text="query"), analysis)

    assert model.payload is None


def test_llm_rewriter_rejects_output_that_drops_protected_term() -> None:
    model = FakeModel(
        {
            "primary_text": "告警怎么处理",
            "subqueries": [],
            "entity_candidates": [],
            "confidence": 0.9,
            "reason": "rewrite",
        }
    )

    with pytest.raises(QueryRewriteValidationError, match="protected term"):
        LLMQueryRewriter(model, max_subqueries=2).rewrite(
            QueryRequest(text="SUN2000 告警怎么处理"), make_analysis(entities={"model": "SUN2000"})
        )


def test_llm_rewriter_bounds_context_and_isolates_payload_and_result() -> None:
    output: dict[str, object] = {
        "primary_text": "SUN2000 告警怎么处理",
        "subqueries": ["SUN2000 告警怎么处理", "SUN2000 告警怎么处理", "SUN2000 排查"],
        "entity_candidates": [
            {"field": "model", "value": "SUN2000", "evidence": "SUN2000"}
        ],
        "confidence": 0.8,
        "reason": "rewrite",
    }
    model = FakeModel(output)
    context = tuple(QueryTurn(role="user", content=f"{index}-" + "x" * 1200) for index in range(8))

    rewrite = LLMQueryRewriter(model, max_subqueries=3).rewrite(
        QueryRequest(text="SUN2000 告警怎么处理", conversation_context=context),
        make_analysis(entities={"model": "SUN2000"}),
    )
    assert model.payload is not None
    payload_context = model.payload["conversation_context"]
    assert isinstance(payload_context, list)
    assert [item["content"] for item in payload_context] == [turn.content[-1000:] for turn in context[-6:]]
    assert rewrite.subqueries == ("SUN2000 告警怎么处理", "SUN2000 排查")
    assert rewrite.entity_candidates[0].field == "model"

    output["subqueries"] = ["changed"]
    assert rewrite.subqueries == ("SUN2000 告警怎么处理", "SUN2000 排查")


def test_llm_rewriter_accepts_chinese_surface_forms_for_canonical_protected_terms() -> None:
    rewrite = LLMQueryRewriter(
        FakeModel({"primary_text": "华为逆变器告警排查", "subqueries": []})
    ).rewrite(
        QueryRequest(text="华为逆变器告警怎么处理"),
        make_analysis(entities={"manufacturer": "huawei", "device_type": "inverter"}),
    )

    assert rewrite.preserved_terms == ("huawei", "inverter")


def test_llm_rewriter_does_not_match_short_ascii_term_inside_larger_identifier() -> None:
    with pytest.raises(QueryRewriteValidationError, match="protected term"):
        LLMQueryRewriter(FakeModel({"primary_text": "地址 40312", "subqueries": []})).rewrite(
            QueryRequest(text="功能码 03 怎么读"),
            make_analysis(entities={"function_code": "03"}),
        )


def test_llm_rewriter_rejects_candidate_value_from_a_different_source_than_evidence() -> None:
    with pytest.raises(QueryRewriteValidationError, match="entity candidate"):
        LLMQueryRewriter(
            FakeModel(
                {
                    "primary_text": "huawei SUN2000",
                    "subqueries": [],
                    "entity_candidates": [
                        {"field": "model", "value": "huawei", "evidence": "SUN2000"}
                    ],
                }
            )
        ).rewrite(QueryRequest(text="huawei SUN2000"), make_analysis())


@pytest.mark.parametrize(
    ("text", "candidate"),
    (
        ("40312", {"field": "function_code", "value": "03", "evidence": "03"}),
        ("SUN2000", {"field": "model", "value": "SUN", "evidence": "SUN"}),
    ),
)
def test_llm_rewriter_rejects_short_ascii_candidate_substrings(
    text: str, candidate: dict[str, str]
) -> None:
    with pytest.raises(QueryRewriteValidationError, match="entity candidate"):
        LLMQueryRewriter(
            FakeModel({"primary_text": text, "subqueries": [], "entity_candidates": [candidate]})
        ).rewrite(QueryRequest(text=text), make_analysis())


def test_llm_rewriter_rejects_field_that_conflicts_with_evidence_inference() -> None:
    with pytest.raises(QueryRewriteValidationError, match="entity candidate"):
        LLMQueryRewriter(
            FakeModel(
                {
                    "primary_text": "huawei",
                    "subqueries": [],
                    "entity_candidates": [
                        {"field": "model", "value": "huawei", "evidence": "huawei"}
                    ],
                }
            )
        ).rewrite(QueryRequest(text="huawei"), make_analysis())


@pytest.mark.parametrize(
    ("text", "candidate"),
    (
        ("型号 SUN2000", {"field": "model", "value": "SUN2000", "evidence": "型号 SUN2000"}),
        ("告警名是 CPUHigh", {"field": "alert_name", "value": "CPUHigh", "evidence": "告警名是 CPUHigh"}),
        (
            "寄存器 40001",
            {"field": "register_address", "value": "40001", "evidence": "寄存器 40001"},
        ),
    ),
)
def test_llm_rewriter_accepts_bounded_value_inside_contextual_evidence(
    text: str, candidate: dict[str, str]
) -> None:
    rewrite = LLMQueryRewriter(
        FakeModel({"primary_text": text, "subqueries": [], "entity_candidates": [candidate]})
    ).rewrite(QueryRequest(text=text), make_analysis())

    assert rewrite.entity_candidates[0].value == candidate["value"]


@pytest.mark.parametrize(
    ("text", "candidate"),
    (
        (
            "华为 SUN2000 型号",
            {"field": "model", "value": "SUN2000", "evidence": "华为 SUN2000 型号"},
        ),
        (
            "华为设备告警 CPUHigh",
            {"field": "alert_name", "value": "CPUHigh", "evidence": "华为设备告警 CPUHigh"},
        ),
        (
            "华为逆变器寄存器 40001",
            {
                "field": "register_address",
                "value": "40001",
                "evidence": "华为逆变器寄存器 40001",
            },
        ),
    ),
)
def test_llm_rewriter_accepts_contextual_evidence_with_unrelated_entities(
    text: str, candidate: dict[str, str]
) -> None:
    rewrite = LLMQueryRewriter(
        FakeModel({"primary_text": text, "subqueries": [], "entity_candidates": [candidate]})
    ).rewrite(QueryRequest(text=text), make_analysis())

    assert rewrite.entity_candidates[0].value == candidate["value"]


def test_llm_rewriter_rejects_mixed_evidence_with_short_ascii_substring() -> None:
    candidate = {"field": "function_code", "value": "03", "evidence": "03 告警"}
    with pytest.raises(QueryRewriteValidationError, match="entity candidate"):
        LLMQueryRewriter(
            FakeModel({"primary_text": "地址 403 告警", "subqueries": [], "entity_candidates": [candidate]})
        ).rewrite(QueryRequest(text="地址 403 告警"), make_analysis())


def test_llm_rewriter_honors_candidate_profile_restrictions() -> None:
    registry = MetadataFieldRegistry(
        (MetadataFieldSpec("protocol_only", profiles=(RetrievalProfile.PROTOCOL,)),)
    )
    model = FakeModel(
        {
            "primary_text": "value",
            "subqueries": [],
            "entity_candidates": [
                {"field": "protocol_only", "value": "value", "evidence": "value"}
            ],
        }
    )
    rewriter = LLMQueryRewriter(model, registry=registry)

    with pytest.raises(QueryRewriteValidationError, match="entity candidate"):
        rewriter.rewrite(QueryRequest(text="value"), make_analysis(profile=RetrievalProfile.CHAT))

    assert rewriter.rewrite(
        QueryRequest(text="value"), make_analysis(profile=RetrievalProfile.PROTOCOL)
    ).entity_candidates[0].field == "protocol_only"


def test_llm_rewriter_rejects_entity_candidate_object_bypass() -> None:
    with pytest.raises(QueryRewriteValidationError, match="entity candidate"):
        LLMQueryRewriter(
            FakeModel(
                {
                    "primary_text": "SUN2000",
                    "subqueries": [],
                    "entity_candidates": [
                        EntityCandidate(field="model", value="SUN2000", evidence="x" * 501)
                    ],
                }
            )
        ).rewrite(QueryRequest(text="SUN2000"), make_analysis())


def test_llm_rewriter_canonicalizes_valid_candidate_before_returning_it() -> None:
    rewrite = LLMQueryRewriter(
        FakeModel(
            {
                "primary_text": "华为逆变器排查",
                "subqueries": [],
                "entity_candidates": [
                    {"field": "厂商", "value": " 华为 ", "evidence": " 华为 "}
                ],
            }
        ),
        registry=MetadataFieldRegistry((MetadataFieldSpec("manufacturer", aliases=("厂商",)),)),
    ).rewrite(QueryRequest(text="华为逆变器排查"), make_analysis())

    assert rewrite.entity_candidates[0].field == "manufacturer"
    assert rewrite.entity_candidates[0].value == "huawei"
    assert rewrite.entity_candidates[0].evidence == "华为"


@pytest.mark.parametrize("maximum", (True, 1.5, -1, 17))
def test_llm_rewriter_rejects_invalid_max_subqueries(maximum: object) -> None:
    with pytest.raises(ValueError, match="max_subqueries"):
        LLMQueryRewriter(FakeModel({"primary_text": "unused"}), max_subqueries=maximum)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "output, error",
    (
        ({"primary_text": "", "subqueries": []}, "primary_text"),
        ({"primary_text": "ok", "subqueries": ["x" * 501]}, "500"),
        ({"primary_text": "ok", "subqueries": ["a", "b"]}, "max_subqueries"),
        (
            {
                "primary_text": "ok",
                "subqueries": [],
                "entity_candidates": [{"field": "unknown", "value": "x", "evidence": "x"}],
            },
            "entity candidate",
        ),
        (
            {
                "primary_text": "ok",
                "subqueries": [],
                "entity_candidates": [
                    {"field": "model", "value": "x", "evidence": "x" * 501}
                ],
            },
            "entity candidate",
        ),
        (
            {
                "primary_text": "ok",
                "subqueries": [],
                "entity_candidates": [{"field": "model", "value": "x", "evidence": "absent"}],
            },
            "entity candidate",
        ),
        (
            {
                "primary_text": "SUN2000",
                "subqueries": [],
                "entity_candidates": [
                    {"field": "model", "value": "HALLUCINATED", "evidence": "SUN2000"}
                ],
            },
            "entity candidate",
        ),
        (
            {"primary_text": "ok", "subqueries": {"not", "ordered"}},
            "subqueries",
        ),
        (
            {"primary_text": "ok", "subqueries": (query for query in ("one",))},
            "subqueries",
        ),
        (
            {
                "primary_text": "ok",
                "subqueries": [],
                "entity_candidates": [{"field": "model", "value": "x", "evidence": "x"}] * 17,
            },
            "entity_candidates",
        ),
        (
            {
                "primary_text": "ok",
                "subqueries": [],
                "entity_candidates": [
                    {"field": "model", "value": "x" * 161, "evidence": "evidence"}
                ],
            },
            "entity candidate",
        ),
        ({"primary_text": "ok", "subqueries": [], "reason": "x" * 501}, "reason"),
    ),
)
def test_llm_rewriter_rejects_invalid_model_output(output: dict[str, object], error: str) -> None:
    with pytest.raises(QueryRewriteValidationError, match=error):
        LLMQueryRewriter(FakeModel(output), max_subqueries=1).rewrite(
            QueryRequest(text="evidence"), make_analysis()
        )


@pytest.mark.parametrize("confidence", (True, math.nan, math.inf, -0.1, 1.1))
def test_llm_rewriter_rejects_non_finite_or_out_of_range_confidence(confidence: object) -> None:
    with pytest.raises(QueryRewriteValidationError, match="confidence"):
        LLMQueryRewriter(FakeModel({"primary_text": "ok", "subqueries": [], "confidence": confidence})).rewrite(
            QueryRequest(text="query"), make_analysis()
        )


def test_llm_rewriter_accepts_taxonomy_normalized_candidate_from_chinese_evidence() -> None:
    rewrite = LLMQueryRewriter(
        FakeModel(
            {
                "primary_text": "华为逆变器排查",
                "subqueries": [],
                "entity_candidates": [
                    {"field": "manufacturer", "value": "huawei", "evidence": "华为"}
                ],
            }
        )
    ).rewrite(QueryRequest(text="华为逆变器排查"), make_analysis())

    assert rewrite.entity_candidates[0].value == "huawei"
