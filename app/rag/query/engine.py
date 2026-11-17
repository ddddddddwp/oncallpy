"""The framework-neutral orchestration boundary of the query pipeline."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import replace

from app.rag.models import Document, SearchHit
from app.rag.query.contracts import (
    QueryAnalyzer,
    QueryDebugger,
    QueryExecutor,
    QueryPlanner,
    QueryRewriter,
    Reranker,
    RewritePolicy,
)
from app.rag.query.errors import safe_error_summary
from app.rag.query.models import (
    QueryAnalysis,
    QueryPlan,
    QueryRequest,
    QueryResult,
    QueryTraceEvent,
)


class _RecordingDebugger:
    """Always retain a safe local trace; a diagnostic sink must be non-fatal."""

    def __init__(self, downstream: QueryDebugger | None) -> None:
        self._downstream = downstream
        self._events: list[QueryTraceEvent] = []

    @property
    def events(self) -> tuple[QueryTraceEvent, ...]:
        return tuple(_safe_event_snapshot(event) for event in self._events)

    def emit(self, event: QueryTraceEvent) -> None:
        snapshot = _safe_event_snapshot(event)
        self._events.append(snapshot)
        if self._downstream is not None:
            try:
                self._downstream.emit(_safe_event_snapshot(snapshot))
            except Exception:
                pass


class QueryEngine:
    """Analyse, plan and execute retrieval without binding to application wiring."""

    def __init__(
        self,
        analyzer: QueryAnalyzer,
        rewrite_policy: RewritePolicy,
        rule_rewriter: QueryRewriter,
        llm_rewriter: QueryRewriter | None,
        planner: QueryPlanner,
        executor: QueryExecutor,
        reranker: Reranker,
        debugger: QueryDebugger | None,
    ) -> None:
        self.analyzer = analyzer
        self.rewrite_policy = rewrite_policy
        self.rule_rewriter = rule_rewriter
        self.llm_rewriter = llm_rewriter
        self.planner = planner
        self.executor = executor
        self.reranker = reranker
        self.debugger = debugger

    def analyze(self, request: QueryRequest) -> QueryAnalysis:
        recorder = _RecordingDebugger(self.debugger)
        return self._run("analyze", recorder, lambda: self.analyzer.analyze(request))

    def plan(self, request: QueryRequest) -> QueryPlan:
        recorder = _RecordingDebugger(self.debugger)
        return self._prepare(request, recorder)

    def query(self, request: QueryRequest) -> QueryResult:
        recorder = _RecordingDebugger(self.debugger)
        plan = self._prepare(request, recorder)
        reasons: list[str] = []
        if plan.rewrite.source == "fallback":
            reasons.append("llm_rewrite_failed")
        try:
            execution = self._execute(plan, recorder)
        except Exception as exc:
            reasons.append("executor_failed")
            return self._failed_result(request, plan, recorder, reasons, exc)
        reasons.extend(execution.degradation_reasons)
        if execution.error is not None:
            reasons.append("executor_failed")
            return QueryResult(
                documents=[],
                plan=plan,
                trace=recorder.events if request.debug else (),
                degraded=True,
                degradation_reasons=tuple(_unique(reasons)),
                error=safe_error_summary(execution.error),
            )
        try:
            fused = self._run(
                "result_materialization",
                recorder,
                lambda: deepcopy(execution.candidates),
            )
            try:
                reranked = self._run(
                    "rerank",
                    recorder,
                    lambda: self._materialize_reranked(
                        plan.rewrite.primary_text,
                        execution.candidates,
                        plan.analysis.profile,
                    ),
                )
            except Exception:
                reranked = fused
                reasons.append("reranker_failed")
            documents = self._context_documents(reranked[: plan.final_top_k], recorder)
        except Exception as exc:
            reasons.append("result_materialization_failed")
            return self._failed_result(request, plan, recorder, reasons, exc)
        all_reasons = _unique(reasons)
        return QueryResult(
            documents=documents,
            plan=plan,
            trace=recorder.events if request.debug else (),
            degraded=execution.degraded or bool(all_reasons),
            degradation_reasons=tuple(all_reasons),
        )

    def _prepare(self, request: QueryRequest, recorder: _RecordingDebugger) -> QueryPlan:
        analysis = self._run("analyze", recorder, lambda: self.analyzer.analyze(request))
        decision = self._run(
            "rewrite_policy",
            recorder,
            lambda: self.rewrite_policy.decide(request, analysis),
        )
        if decision.use_llm:
            try:
                rewrite = self._run(
                    "llm_rewrite",
                    recorder,
                    lambda: self._llm_rewrite(request, analysis),
                )
            except Exception:
                rewrite = self._run(
                    "rule_rewrite",
                    recorder,
                    lambda: self.rule_rewriter.rewrite(request, analysis),
                )
                rewrite = replace(rewrite, source="fallback")
        else:
            rewrite = self._run(
                "rule_rewrite",
                recorder,
                lambda: self.rule_rewriter.rewrite(request, analysis),
            )
        return self._run(
            "plan",
            recorder,
            lambda: self.planner.plan(request, analysis, decision, rewrite),
        )

    def _llm_rewrite(self, request: QueryRequest, analysis: QueryAnalysis):  # type: ignore[no-untyped-def]
        if self.llm_rewriter is None:
            raise RuntimeError("LLM rewriter is unavailable")
        return self.llm_rewriter.rewrite(request, analysis)

    def _materialize_reranked(self, query, candidates, profile):  # type: ignore[no-untyped-def]
        value = self.reranker.rerank(query, deepcopy(candidates), profile)
        if isinstance(value, list):
            value = value[:]
        return list(value)

    def _execute(self, plan: QueryPlan, recorder: _RecordingDebugger):
        recorder.emit(QueryTraceEvent(stage="execute", status="started"))
        started = time.monotonic()
        try:
            execution = self.executor.execute(deepcopy(plan), recorder)
        except Exception:
            recorder.emit(
                QueryTraceEvent(
                    stage="execute",
                    status="failed",
                    duration_ms=_elapsed_ms(started),
                )
            )
            raise
        error = safe_error_summary(execution.error) if execution.error is not None else None
        recorder.emit(
            QueryTraceEvent(
                stage="execute",
                status=("failed" if error else "degraded" if execution.degraded else "completed"),
                duration_ms=_elapsed_ms(started),
                data={
                    "degradation_reasons": tuple(execution.degradation_reasons),
                    "attempts_used": execution.attempts_used,
                    "candidate_count": len(execution.candidates),
                    "route_attempted": execution.route_attempted,
                    "route_matched": execution.route_matched,
                    "metadata_filter": execution.metadata_filter,
                    **({"error": error} if error else {}),
                },
            )
        )
        return execution

    @staticmethod
    def _failed_result(request, plan, recorder, reasons, exc):  # type: ignore[no-untyped-def]
        return QueryResult(
            documents=[],
            plan=plan,
            trace=recorder.events if request.debug else (),
            degraded=True,
            degradation_reasons=tuple(_unique(reasons)),
            error=safe_error_summary(exc),
        )

    def _context_documents(
        self, hits: list[SearchHit], recorder: _RecordingDebugger
    ) -> list[Document]:
        def build() -> list[Document]:
            result: list[Document] = []
            for hit in hits:
                original = hit.document.page_content
                metadata = deepcopy(hit.document.metadata)
                context = metadata.get("_context_window")
                if context:
                    metadata.setdefault("_original_text", original)
                    original = str(context)
                result.append(
                    Document(page_content=original, metadata=metadata, id=hit.document.id)
                )
            return result

        return self._run("context_window", recorder, build)

    @staticmethod
    def _run(stage: str, recorder: _RecordingDebugger, callback):  # type: ignore[no-untyped-def]
        recorder.emit(QueryTraceEvent(stage=stage, status="started"))
        started = time.monotonic()
        try:
            value = callback()
        except Exception:
            recorder.emit(
                QueryTraceEvent(
                    stage=stage,
                    status="failed",
                    duration_ms=max(0.0, (time.monotonic() - started) * 1000),
                )
            )
            raise
        recorder.emit(
            QueryTraceEvent(
                stage=stage,
                status="completed",
                duration_ms=max(0.0, (time.monotonic() - started) * 1000),
            )
        )
        return value


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _safe_event_snapshot(event: QueryTraceEvent) -> QueryTraceEvent:
    try:
        return deepcopy(event)
    except Exception:
        return QueryTraceEvent(
            stage=str(event.stage),
            status=event.status,
            duration_ms=event.duration_ms,
            data={"trace_data_unavailable": True},
        )


def _elapsed_ms(started: float) -> float:
    return max(0.0, (time.monotonic() - started) * 1000)
