from __future__ import annotations

from contextlib import contextmanager

from app.config import config
from app.rag.models import Document, RetrievalProfile, SearchHit, SearchQuery
from app.rag.query.engine import QueryEngine
from app.rag.query.models import (
    QueryAnalysis,
    QueryExecution,
    QueryPlan,
    QueryRequest,
    RewriteDecision,
    RewrittenQuery,
)
from app.rag.query.query_service import QueryService


class Analyzer:
    def analyze(self, request):
        return QueryAnalysis(request.text, RetrievalProfile.CHAT)


class Policy:
    def __init__(self, use_llm=False):
        self.use_llm = use_llm

    def decide(self, request, analysis):
        return RewriteDecision(self.use_llm)


class Rewriter:
    def __init__(self, source="rule", error=None):
        self.calls = 0
        self.source = source
        self.error = error

    def rewrite(self, request, analysis):
        self.calls += 1
        if self.error:
            raise self.error
        return RewrittenQuery(request.text, source=self.source)


class Planner:
    def plan(self, request, analysis, decision, rewrite):
        return QueryPlan(
            request,
            analysis,
            decision,
            rewrite,
            (rewrite.primary_text,),
            None,
            1,
            2,
            False,
            False,
            60,
            2,
        )


class Executor:
    def __init__(self, error=None):
        self.error = error

    def execute(self, plan, debugger=None):
        if self.error:
            raise self.error
        return QueryExecution([SearchHit(Document("text", {"nested": {"a": 1}}), 1.0, "dense")])


class Reranker:
    def __init__(self, error=None):
        self.error = error
        self.args = None

    def rerank(self, query, candidates, profile):
        self.args = (query, candidates, profile)
        if self.error:
            raise self.error
        return candidates


def make_engine(*, llm=None, policy=None, executor=None, reranker=None):
    return QueryEngine(
        Analyzer(),
        policy or Policy(),
        Rewriter(),
        llm,
        Planner(),
        executor or Executor(),
        reranker or Reranker(),
        None,
    )


def test_simple_query_never_calls_llm() -> None:
    llm = Rewriter("llm")
    result = make_engine(llm=llm).query(QueryRequest("hello", debug=True))
    assert llm.calls == 0
    assert result.documents[0].page_content == "text"
    assert [event.stage for event in result.trace][:2] == ["analyze", "analyze"]


def test_llm_failure_uses_rule_fallback_and_is_degraded() -> None:
    llm = Rewriter("llm", RuntimeError("bad model"))
    result = make_engine(llm=llm, policy=Policy(True)).query(QueryRequest("hello"))
    assert result.plan.rewrite.source == "fallback"
    assert result.degraded is True
    assert result.degradation_reasons == ("llm_rewrite_failed",)


def test_missing_llm_emits_failed_stage_before_rule_fallback() -> None:
    result = make_engine(policy=Policy(True)).query(QueryRequest("hello", debug=True))
    llm_events = [event.status for event in result.trace if event.stage == "llm_rewrite"]
    assert llm_events == ["started", "failed"]


def test_executor_failure_is_a_safe_empty_result() -> None:
    result = make_engine(executor=Executor(RuntimeError("down"))).query(QueryRequest("hello"))
    assert result.documents == []
    assert result.error == "down"
    assert result.degradation_reasons == ("executor_failed",)


def test_empty_and_sensitive_executor_errors_are_safely_summarized() -> None:
    empty = make_engine(executor=Executor(RuntimeError())).query(QueryRequest("hello"))
    leaked = make_engine(
        executor=Executor(RuntimeError("Bearer abc api_key=xyz password=hunter2"))
    ).query(QueryRequest("hello"))
    assert empty.error == "RuntimeError"
    assert leaked.error == "Bearer *** api_key=*** password=***"


def test_execution_reasons_are_in_trace_data() -> None:
    class Degraded(Executor):
        def execute(self, plan, debugger=None):
            return QueryExecution([], True, ["dense_search_failed"])

    result = make_engine(executor=Degraded()).query(QueryRequest("hello", debug=True))
    event = next(
        event for event in result.trace if event.stage == "execute" and event.status == "degraded"
    )
    assert event.data["degradation_reasons"] == ("dense_search_failed",)
    assert event.data["attempts_used"] == 0
    assert event.data["candidate_count"] == 0
    assert event.data["route_attempted"] is False
    assert event.data["route_matched"] is False


def test_external_execution_error_is_sanitized_and_marks_execute_failed() -> None:
    class ExternalError(Executor):
        def execute(self, plan, debugger=None):
            return QueryExecution(
                [],
                True,
                ["dense_search_failed"],
                error="access_token=abc client-secret:xyz postgres://user:hunter2@db/x",
            )

    result = make_engine(executor=ExternalError()).query(QueryRequest("hello", debug=True))
    events = [event for event in result.trace if event.stage == "execute"]
    assert [event.status for event in events] == ["started", "failed"]
    event = events[-1]
    assert event.data["error"] == "access_token=*** client-secret=*** postgres://user:***@db/x"
    assert result.error == event.data["error"]


def test_empty_external_execution_error_is_still_failed() -> None:
    class EmptyError(Executor):
        def execute(self, plan, debugger=None):
            return QueryExecution([], error="")

    result = make_engine(executor=EmptyError()).query(QueryRequest("hello", debug=True))
    assert result.error == "Error"
    assert result.degraded is True
    assert [event.status for event in result.trace if event.stage == "execute"] == [
        "started",
        "failed",
    ]


def test_malicious_reranker_container_fails_inside_rerank_stage() -> None:
    class BadSlice(list):
        def __getitem__(self, key):
            if isinstance(key, slice):
                raise RuntimeError("api-key=unsafe")
            return super().__getitem__(key)

    class BadReranker(Reranker):
        def rerank(self, query, candidates, profile):
            return BadSlice(candidates)

    result = make_engine(reranker=BadReranker()).query(QueryRequest("hello", debug=True))
    assert result.documents[0].page_content == "text"
    assert "reranker_failed" in result.degradation_reasons
    assert [event.status for event in result.trace if event.stage == "rerank"] == [
        "started",
        "failed",
    ]


def test_candidate_snapshot_failure_has_materialization_failed_trace() -> None:
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("passwd=unsafe")

    class BadCandidate(Executor):
        def execute(self, plan, debugger=None):
            return QueryExecution(
                [SearchHit(Document("text", {"bad": Uncopyable()}), 1.0, "dense")]
            )

    result = make_engine(executor=BadCandidate()).query(QueryRequest("hello", debug=True))
    statuses = [event.status for event in result.trace if event.stage == "result_materialization"]
    assert statuses == ["started", "failed"]
    assert result.error == "passwd=***"


def test_uncopyable_trace_and_candidate_become_safe_failures() -> None:
    class Uncopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("password=trace-secret")

    class Emits(Executor):
        def execute(self, plan, debugger=None):
            debugger.emit(
                __import__("app.rag.query.models", fromlist=["QueryTraceEvent"]).QueryTraceEvent(
                    "custom", "completed", data={"value": Uncopyable()}
                )
            )
            return QueryExecution(
                [SearchHit(Document("text", {"bad": Uncopyable()}), 1.0, "dense")]
            )

    result = make_engine(executor=Emits()).query(QueryRequest("hello", debug=True))
    custom = next(event for event in result.trace if event.stage == "custom")
    assert custom.data == {"trace_data_unavailable": True}
    assert result.documents == []
    assert "result_materialization_failed" in result.degradation_reasons
    assert result.error == "password=***"


def test_context_window_returns_isolated_document_and_slices() -> None:
    class Many(Executor):
        def execute(self, plan, debugger=None):
            return QueryExecution(
                [
                    SearchHit(
                        Document("one", {"_context_window": "window", "nested": {"a": 1}}, "id1"),
                        1,
                        "dense",
                    ),
                    SearchHit(Document("two", {}, "id2"), 1, "dense"),
                ]
            )

    engine = make_engine(executor=Many())
    result = engine.query(QueryRequest("hi"))
    assert len(result.documents) == 1
    assert result.documents[0].page_content == "window"
    assert result.documents[0].metadata["_original_text"] == "one"
    result.documents[0].metadata["nested"]["a"] = 9
    assert (
        engine.executor.execute(
            Planner().plan(
                QueryRequest("x"),
                QueryAnalysis("x", RetrievalProfile.CHAT),
                RewriteDecision(False),
                RewrittenQuery("x"),
            )
        )
        .candidates[0]
        .document.metadata["nested"]["a"]
        == 1
    )


def test_query_service_defaults_to_off_but_forwards_explicit_rewrite_mode() -> None:
    class CapturingEngine:
        def __init__(self):
            self.request = None

        def query(self, request):
            self.request = request
            plan = Planner().plan(
                request,
                QueryAnalysis(request.text, RetrievalProfile.CHAT),
                RewriteDecision(False),
                RewrittenQuery(request.text),
            )
            from app.rag.query.models import QueryResult

            return QueryResult([], plan)

    engine = CapturingEngine()
    QueryService(engine=engine).retrieve(SearchQuery("hello"))
    assert engine.request.rewrite_mode == "off"
    QueryService(engine=engine, rewrite_mode="auto").retrieve(SearchQuery("hello"))
    assert engine.request.rewrite_mode == "auto"


def test_query_service_span_normalizes_profile_and_default_top_k(monkeypatch) -> None:
    outputs = []

    @contextmanager
    def span_factory(name, input_data=None):
        outputs.append({"input": input_data})

        class Span:
            def set_output(self, output):
                outputs[-1]["output"] = output

        yield Span()

    class CapturingEngine:
        def query(self, request):
            from app.rag.query.models import QueryResult

            analysis = QueryAnalysis(request.text, RetrievalProfile.CHAT)
            plan = Planner().plan(
                request, analysis, RewriteDecision(False), RewrittenQuery(request.text)
            )
            return QueryResult([], plan)

    monkeypatch.setattr(config, "rag_top_k", 7)
    QueryService(engine=CapturingEngine(), trace_span_factory=span_factory).retrieve(
        SearchQuery("hello", profile="unknown", top_k=None)
    )
    assert outputs[0]["input"] == {"query": "hello", "profile": "chat", "top_k": 7}


def test_query_service_resanitizes_injected_engine_error() -> None:
    outputs = []

    @contextmanager
    def span_factory(name, input_data=None):
        class Span:
            def set_output(self, output):
                outputs.append(output)

        yield Span()

    class InjectedEngine:
        def query(self, request):
            from app.rag.query.models import QueryResult

            plan = Planner().plan(
                request,
                QueryAnalysis(request.text, RetrievalProfile.CHAT),
                RewriteDecision(False),
                RewrittenQuery(request.text),
            )
            return QueryResult([], plan, error="PASSWORD=hunter2")

    assert (
        QueryService(engine=InjectedEngine(), trace_span_factory=span_factory).retrieve(
            SearchQuery("hello")
        )
        == []
    )
    assert outputs[0]["message"] == "PASSWORD=***"


def test_query_service_treats_empty_injected_error_as_failure() -> None:
    outputs = []

    @contextmanager
    def span_factory(name, input_data=None):
        class Span:
            def set_output(self, output):
                outputs.append(output)

        yield Span()

    class InjectedEngine:
        def query(self, request):
            from app.rag.query.models import QueryResult

            plan = Planner().plan(
                request,
                QueryAnalysis(request.text, RetrievalProfile.CHAT),
                RewriteDecision(False),
                RewrittenQuery(request.text),
            )
            return QueryResult([], plan, error="")

    QueryService(engine=InjectedEngine(), trace_span_factory=span_factory).retrieve(
        SearchQuery("hello")
    )
    assert outputs[0] == {"status": "error", "message": "Error", "candidate_k": 2}
