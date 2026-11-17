from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import pytest

from app.rag.models import Document, RetrievalProfile, SearchHit
from app.rag.query.cli import _json_value, _preview, main
from app.rag.query.models import (
    QueryAnalysis,
    QueryPlan,
    QueryRequest,
    QueryResult,
    QueryTraceEvent,
    RewriteDecision,
    RewrittenQuery,
)


def _plan(request: QueryRequest) -> QueryPlan:
    return QueryPlan(
        request=request,
        analysis=QueryAnalysis(request.text, RetrievalProfile(request.profile)),
        decision=RewriteDecision(False),
        rewrite=RewrittenQuery(request.text, source="off"),
        queries=(request.text,),
        metadata_filter=None,
        final_top_k=3,
        candidate_k=9,
        route_first=True,
        hybrid=False,
        rrf_k=60,
        max_attempts=8,
    )


class CapturingEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, QueryRequest]] = []

    def analyze(self, request: QueryRequest):
        self.calls.append(("analyze", request))
        return QueryAnalysis(request.text, RetrievalProfile(request.profile))

    def plan(self, request: QueryRequest):
        self.calls.append(("plan", request))
        return _plan(request)

    def query(self, request: QueryRequest):
        self.calls.append(("query", request))
        return QueryResult(
            [Document("a\npreview", {"_file_name": "a.md"})],
            _plan(request),
            trace=(QueryTraceEvent("execute", "completed", data={"attempts_used": 2}),),
        )


@pytest.mark.parametrize(
    "command,method", [("analyze", "analyze"), ("plan", "plan"), ("run", "query")]
)
def test_cli_calls_only_requested_engine_method(command, method, capsys) -> None:
    engine = CapturingEngine()
    assert main([command, "CPUHigh", "--format", "json"], engine) == 0
    assert [name for name, _ in engine.calls] == [method]
    assert engine.calls[0][1].rewrite_mode == "auto"
    assert json.loads(capsys.readouterr().out)


def test_plan_and_run_forward_all_explicit_request_options(capsys) -> None:
    engine = CapturingEngine()
    assert (
        main(
            ["plan", "CPUHigh", "--profile", "aiops", "--top-k", "4", "--rewrite-mode", "llm"],
            engine,
        )
        == 0
    )
    request = engine.calls[-1][1]
    assert (request.profile, request.top_k, request.rewrite_mode, request.debug) == (
        "aiops",
        4,
        "llm",
        False,
    )
    capsys.readouterr()
    assert (
        main(
            [
                "run",
                "register",
                "--profile",
                "protocol",
                "--top-k",
                "2",
                "--rewrite-mode",
                "off",
                "--debug",
            ],
            engine,
        )
        == 0
    )
    request = engine.calls[-1][1]
    assert (request.profile, request.top_k, request.rewrite_mode, request.debug) == (
        "protocol",
        2,
        "off",
        True,
    )


def test_cli_table_includes_plan_and_run_debug_fields(monkeypatch, capsys) -> None:
    monkeypatch.setenv("RAG_QUERY_DEBUG_PREVIEW_LENGTH", "1")
    assert main(["run", "CPUHigh", "--debug"], CapturingEngine()) == 0
    output = capsys.readouterr().out
    for key in (
        "stage",
        "profile",
        "rewrite_source",
        "primary_query",
        "queries",
        "filter",
        "route_first",
        "hybrid",
        "final_top_k",
        "candidate_k",
        "max_attempts",
        "degraded",
        "reasons",
        "error",
        "document_count",
        "attempt_budget",
        "attempts_used",
        "document[1].source",
        "trace[1].stage",
        "trace[1].status",
        "trace[1].duration_ms",
    ):
        assert f"{key}:" in output
    assert "document[1].preview: a" in output


def test_cli_run_error_has_nonzero_exit_but_writes_structured_payload(capsys) -> None:
    class ErrorEngine(CapturingEngine):
        def query(self, request):
            return QueryResult([], _plan(request), error="not available")

    assert main(["run", "CPUHigh", "--format", "json"], ErrorEngine()) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "not available"


@pytest.mark.parametrize("output_format", ["json", "table"])
def test_cli_resanitizes_structured_error(output_format, capsys) -> None:
    class ErrorEngine(CapturingEngine):
        def query(self, request):
            return QueryResult([], _plan(request), error="token=abcdef Bearer topsecret")

    assert main(["run", "CPUHigh", "--format", output_format], ErrorEngine()) == 1
    output = capsys.readouterr().out
    assert "abcdef" not in output and "topsecret" not in output


def test_cli_safe_failure_does_not_leak_secret(capsys) -> None:
    class Broken:
        def analyze(self, request):
            raise RuntimeError("token=abcdef Bearer topsecret")

    assert main(["analyze", "CPUHigh"], Broken()) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "abcdef" not in captured.err and "topsecret" not in captured.err


def test_cli_terminal_output_cannot_inject_lines_or_ansi(capsys) -> None:
    class UnsafeEngine(CapturingEngine):
        def analyze(self, request):
            return QueryAnalysis("ok\nforged\x1b[31m\u0085\u009b", RetrievalProfile.CHAT)

    assert main(["analyze", "x"], UnsafeEngine()) == 0
    output = capsys.readouterr().out
    assert "\x1b" not in output and "\u0085" not in output and "\u009b" not in output
    assert "\nforged" not in output

    class Broken:
        def analyze(self, request):
            raise RuntimeError("bad\nforged\x1b[31m\u0085\u009b")

    assert main(["analyze", "x"], Broken()) == 1
    error = capsys.readouterr().err
    assert error.count("\n") == 1
    assert "\x1b" not in error and "\u0085" not in error and "\u009b" not in error


def test_cli_sanitizes_controls_inside_structured_table_values(capsys) -> None:
    unsafe = "query\nforged\x1b[31m\u0085\u009b"
    request = QueryRequest("x")
    plan = _plan(request)
    from dataclasses import replace

    plan = replace(
        plan,
        queries=(unsafe,),
        metadata_filter={"unsafe": unsafe},
    )

    class UnsafeEngine(CapturingEngine):
        def query(self, incoming):
            return QueryResult([], plan, degradation_reasons=(unsafe,))

    assert main(["run", "x"], UnsafeEngine()) == 0
    output = capsys.readouterr().out
    assert "\x1b" not in output and "\u0085" not in output and "\u009b" not in output
    assert 'queries: ["query\\nforged\\u001b[31m' in output
    assert 'filter: {"unsafe": "query\\nforged\\u001b[31m' in output
    assert 'reasons: ["query\\nforged\\u001b[31m' in output


def test_json_converter_handles_dataclasses_tuple_enum_nan_and_cycle() -> None:
    payload = _json_value(
        {"profile": RetrievalProfile.CHAT, "values": (1, math.nan), "doc": Document("x")}
    )
    assert payload == {
        "profile": "chat",
        "values": [1, None],
        "doc": {"page_content": "x", "metadata": {}, "id": None},
    }
    circular: list[object] = []
    circular.append(circular)
    with pytest.raises(TypeError, match="list"):
        _json_value(circular)


def test_json_converter_serializes_search_hit_and_rejects_key_collision() -> None:
    hit = SearchHit(Document("x", id="1"), 0.5, "dense")
    assert _json_value(hit)["document"]["id"] == "1"
    with pytest.raises(TypeError, match="dict"):
        _json_value({1: "a", "1": "b"})


def test_preview_only_samples_a_bounded_prefix() -> None:
    class TrackingText(str):
        slices: list[slice] = []

        def __getitem__(self, key):
            if isinstance(key, slice):
                self.slices.append(key)
            return super().__getitem__(key)

    text = TrackingText("x" * 20_000_000)
    assert _preview(text, 12) == "x" * 12
    assert text.slices and text.slices[0].stop < 1_000


def test_cli_argparse_errors_and_help_do_not_need_engine(capsys) -> None:
    with pytest.raises(SystemExit) as empty:
        main(["analyze", "   "])
    assert empty.value.code == 2
    with pytest.raises(SystemExit) as top_k:
        main(["plan", "x", "--top-k", "0"])
    assert top_k.value.code == 2
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    assert "usage:" in capsys.readouterr().out

    for option in ("--top-k", "--rewrite-mode"):
        with pytest.raises(SystemExit) as analyze_option:
            main(["analyze", "x", option, "1" if option == "--top-k" else "off"])
        assert analyze_option.value.code == 2


def test_module_help_survives_invalid_preview_environment() -> None:
    environment = dict(os.environ, RAG_QUERY_DEBUG_PREVIEW_LENGTH="invalid")
    completed = subprocess.run(
        [sys.executable, "-m", "app.rag.query.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0
    assert "usage:" in completed.stdout


def test_injected_analyze_does_not_import_config(monkeypatch, capsys) -> None:
    import builtins

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "app.config":
            raise AssertionError("config must remain lazy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert main(["analyze", "x"], CapturingEngine()) == 0
    assert "stage: analyze" in capsys.readouterr().out


@pytest.mark.parametrize("attempts", [True, -1])
def test_cli_rejects_invalid_attempt_counts(attempts, capsys) -> None:
    class AttemptsEngine(CapturingEngine):
        def query(self, request):
            return QueryResult(
                [],
                _plan(request),
                trace=(QueryTraceEvent("execute", "completed", data={"attempts_used": attempts}),),
            )

    assert main(["run", "x"], AttemptsEngine()) == 0
    output = capsys.readouterr().out
    assert "attempt_budget: 8" in output
    assert "attempts_used: unavailable" in output
