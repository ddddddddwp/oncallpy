"""Tests for deterministic query analysis and metadata safety rules."""

import pytest

from app.rag.models import RetrievalProfile
from app.rag.query.analyzer import DefaultQueryAnalyzer
from app.rag.query.metadata_registry import (
    DEFAULT_FIELD_SPECS,
    MetadataFieldRegistry,
    MetadataFieldSpec,
    default_metadata_registry,
)
from app.rag.query.models import EntityCandidate, QueryRequest, QueryTurn
from app.rag.query.query_router import DefaultQueryRouter


def test_aiops_analysis_derives_intents_taxonomy_and_runbook_filter() -> None:
    analysis = DefaultQueryAnalyzer().analyze(
        QueryRequest(text="华为逆变器告警怎么排查、处理并验证？", profile="aiops")
    )

    assert analysis.intents == ("troubleshooting", "resolution", "verification")
    assert analysis.inferred_filter["document_type"] == "runbook"
    assert analysis.inferred_filter["runbook_section"] == [
        "troubleshooting",
        "resolution",
        "verification",
    ]
    assert analysis.inferred_filter["device_type"] == "inverter"
    assert analysis.inferred_filter["manufacturer"] == "huawei"
    assert analysis.complexity == "compound"


def test_context_reference_marks_contextual_query() -> None:
    analysis = DefaultQueryAnalyzer().analyze(
        QueryRequest(text="它处理完以后怎么验证？", profile="aiops")
    )

    assert analysis.context_dependent is True
    assert analysis.complexity == "contextual"
    assert round(analysis.confidence, 2) == 0.35
    assert analysis.reasons == ("context_reference", "multiple_intents")


def test_unknown_profile_uses_chat_taxonomy_only() -> None:
    analysis = DefaultQueryAnalyzer().analyze(
        QueryRequest(text="华为逆变器说明", profile="unsupported")
    )

    assert analysis.profile.value == "chat"
    assert analysis.inferred_filter == {
        "device_type": "inverter",
        "manufacturer": "huawei",
    }


def test_registry_only_relaxes_relaxable_fields_and_keeps_strict_matches() -> None:
    registry = default_metadata_registry()
    metadata_filter = {
        "device_type": "inverter",
        "manufacturer": "huawei",
        "register_address": "40001",
    }

    assert registry.fallbacks(metadata_filter) == [
        metadata_filter,
        {"device_type": "inverter", "register_address": "40001"},
    ]
    assert registry.requires_strict_match(metadata_filter) is True


def test_registry_rejects_unsafe_or_unsupported_entity_candidates() -> None:
    registry = default_metadata_registry()
    text = "请检查华为逆变器告警"
    context = (QueryTurn(role="assistant", content="上一条是设备告警"),)

    assert registry.validate_candidate(
        EntityCandidate(field="unknown", value="value", evidence="value"), text, context
    ) is False
    assert registry.validate_candidate(
        EntityCandidate(field="manufacturer", value="huawei", evidence=""), text, context
    ) is False
    assert registry.validate_candidate(
        EntityCandidate(field="manufacturer", value="huawei", evidence="不存在"), text, context
    ) is False
    assert registry.validate_candidate(
        EntityCandidate(field="manufacturer", value="huawei", evidence="华为"), text, context
    ) is True
    assert registry.validate_candidate(
        EntityCandidate(field="alert_name", value="CPUHigh", evidence="CPUHigh"),
        text,
        (QueryTurn(role="assistant", content="告警名是 CPUHigh"),),
    ) is True
    assert registry.validate_candidate(
        EntityCandidate(field="manufacturer", value="huawei", evidence="content"), text, context
    ) is False
    assert registry.validate_candidate(
        EntityCandidate(field="manufacturer", value="huawei", evidence="assistant"), text, context
    ) is False
    assert registry.validate_candidate(
        EntityCandidate(field="manufacturer", value="huawei", evidence="第二行证据"),
        text,
        (QueryTurn(role="assistant", content="第一行\n第二行证据"),),
    ) is False


def test_default_registry_declares_all_filterable_metadata_boundaries() -> None:
    specs = {spec.name: spec for spec in DEFAULT_FIELD_SPECS}

    assert set(specs) == {
        "document_type",
        "runbook_section",
        "alert_name",
        "business_domain",
        "device_type",
        "manufacturer",
        "model",
        "protocol_type",
        "register_address",
        "register_name",
        "function_code",
        "data_type",
        "unit",
        "scale",
        "read_write",
        "route_key",
        "_source",
    }
    assert all(spec.filterable for spec in specs.values())
    assert {name for name, spec in specs.items() if spec.strict} == {
        "alert_name",
        "device_type",
        "model",
        "register_address",
        "function_code",
    }
    assert {name for name, spec in specs.items() if spec.relaxable} == {"manufacturer"}


def test_validate_filter_drops_unknown_and_empty_values_and_honors_alias_profiles() -> None:
    registry = MetadataFieldRegistry(
        (
            MetadataFieldSpec("manufacturer", aliases=("厂商",)),
            MetadataFieldSpec("register_address", profiles=(RetrievalProfile.PROTOCOL,)),
        )
    )

    assert registry.validate_filter(
        {
            "unknown": "value",
            "厂商": "huawei",
            "register_address": "40001",
        },
        RetrievalProfile.AIOPS,
    ) == {"manufacturer": "huawei"}
    assert registry.validate_filter(
        {"register_address": "40001"}, RetrievalProfile.PROTOCOL
    ) == {"register_address": "40001"}
    for value in ("", "   ", [], {}, False, 0, None):
        assert registry.validate_filter({"manufacturer": value}) == {}
    with pytest.raises(ValueError, match="multiple values"):
        registry.validate_filter({"manufacturer": "invt", "厂商": "huawei"})


def test_strict_and_relaxable_field_is_never_removed_from_fallbacks() -> None:
    registry = MetadataFieldRegistry((MetadataFieldSpec("target", strict=True, relaxable=True),))

    assert registry.fallbacks({"target": "value"}) == [{"target": "value"}]


def test_fallbacks_do_not_relax_falsey_manufacturer_values() -> None:
    registry = default_metadata_registry()

    for value in ("", "   ", [], {}, False, 0, None):
        metadata_filter = {"manufacturer": value, "device_type": "inverter"}
        assert registry.fallbacks(metadata_filter) == [metadata_filter]


def test_protocol_analysis_includes_protocol_metadata() -> None:
    analysis = DefaultQueryAnalyzer().analyze(
        QueryRequest(text="华为逆变器 Modbus 0x3601 功能码 03", profile="protocol")
    )

    assert analysis.inferred_filter["document_type"] == "protocol"
    assert analysis.inferred_filter["protocol_type"] == "modbus"
    assert analysis.inferred_filter["register_address"] == "0x3601"
    assert analysis.inferred_filter["function_code"] == "03"


def test_analyzer_normalizes_whitespace_and_marks_long_unstructured_query() -> None:
    normalized = DefaultQueryAnalyzer().analyze(QueryRequest(text="  华为\n逆变器\t说明  "))
    long_query = DefaultQueryAnalyzer().analyze(QueryRequest(text="问题 " * 50))

    assert normalized.normalized_text == "华为 逆变器 说明"
    assert long_query.confidence == 0.7
    assert long_query.reasons == ("long_unstructured_query",)


def test_protocol_analysis_and_router_preserve_multiline_field_boundaries() -> None:
    text = "名称: 电压\n单位: V"

    analysis = DefaultQueryAnalyzer().analyze(QueryRequest(text=text, profile="protocol"))
    router_filter = DefaultQueryRouter().infer_metadata_filter(text, RetrievalProfile.PROTOCOL)

    assert analysis.normalized_text == "名称: 电压 单位: V"
    assert analysis.inferred_filter["register_name"] == "电压"
    assert router_filter["register_name"] == "电压"


def test_registry_canonicalizes_aliases_for_candidates_fallbacks_and_strict_matches() -> None:
    registry = MetadataFieldRegistry(
        (
            MetadataFieldSpec("manufacturer", aliases=("厂商",), relaxable=True),
            MetadataFieldSpec("device_type", aliases=("设备",), strict=True),
        )
    )

    assert registry.validate_candidate(
        EntityCandidate(field="厂商", value="huawei", evidence="华为"), "华为逆变器"
    ) is True
    assert registry.fallbacks({"厂商": "huawei", "设备": "inverter"}) == [
        {"manufacturer": "huawei", "device_type": "inverter"},
        {"device_type": "inverter"},
    ]
    assert registry.requires_strict_match({"设备": "inverter"}) is True


def test_registry_returns_deeply_isolated_filter_and_fallback_values() -> None:
    registry = default_metadata_registry()
    source = {
        "runbook_section": ["troubleshooting"],
        "_source": {"paths": ["runbook.md"]},
        "manufacturer": "huawei",
    }

    validated = registry.validate_filter(source)
    validated["runbook_section"].append("resolution")
    validated["_source"]["paths"].append("changed.md")
    assert source == {
        "runbook_section": ["troubleshooting"],
        "_source": {"paths": ["runbook.md"]},
        "manufacturer": "huawei",
    }

    fallbacks = registry.fallbacks(source)
    fallbacks[0]["runbook_section"].append("resolution")
    fallbacks[1]["_source"]["paths"].append("changed.md")
    assert source["runbook_section"] == ["troubleshooting"]
    assert fallbacks[1]["runbook_section"] == ["troubleshooting"]
    assert fallbacks[0]["_source"]["paths"] == ["runbook.md"]


def test_router_uses_analyzer_registry_when_no_registry_is_explicitly_injected() -> None:
    registry = MetadataFieldRegistry((MetadataFieldSpec("vendor", relaxable=True),))
    router = DefaultQueryRouter(analyzer=DefaultQueryAnalyzer(registry))

    assert router.metadata_filter_fallbacks({"vendor": "acme"}) == [
        {"vendor": "acme"},
        {},
    ]


def test_router_rejects_registry_that_differs_from_its_analyzer() -> None:
    analyzer_registry = MetadataFieldRegistry((MetadataFieldSpec("vendor", relaxable=True),))
    other_registry = MetadataFieldRegistry((MetadataFieldSpec("vendor", relaxable=True),))
    analyzer = DefaultQueryAnalyzer(analyzer_registry)

    with pytest.raises(ValueError, match="analyzer.registry"):
        DefaultQueryRouter(analyzer=analyzer, registry=other_registry)

    router = DefaultQueryRouter(analyzer=analyzer, registry=analyzer_registry)
    assert router.metadata_filter_fallbacks({"vendor": "acme"}) == [
        {"vendor": "acme"},
        {},
    ]


def test_registry_rejects_duplicate_specs_and_cross_field_aliases() -> None:
    with pytest.raises(ValueError, match="names must be unique"):
        MetadataFieldRegistry((MetadataFieldSpec("vendor"), MetadataFieldSpec("vendor")))
    with pytest.raises(ValueError, match="maps to multiple fields"):
        MetadataFieldRegistry(
            (
                MetadataFieldSpec("vendor", aliases=("厂商",)),
                MetadataFieldSpec("manufacturer", aliases=("厂商",)),
            )
        )


def test_validate_filter_rejects_canonical_and_alias_even_when_one_is_empty() -> None:
    registry = MetadataFieldRegistry((MetadataFieldSpec("manufacturer", aliases=("厂商",)),))

    with pytest.raises(ValueError, match="multiple values"):
        registry.validate_filter({"manufacturer": "", "厂商": "huawei"})
