"""Command-line inspection boundary for the independent query pipeline.

The module deliberately accepts an injected :class:`QueryEngine`: this keeps
``analyze`` and ``plan`` useful in offline tests and avoids constructing RAG
infrastructure until an actual command needs the default engine.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from enum import StrEnum
from typing import Any

from app.rag.query.engine import QueryEngine
from app.rag.query.errors import safe_error_summary
from app.rag.query.models import QueryPlan, QueryRequest, QueryResult, RewriteMode

_PROFILES = ("chat", "aiops", "protocol")
_REWRITE_MODES = tuple(mode.value for mode in RewriteMode)


def _non_blank(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("TEXT must not be empty")
    return value


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("top-k must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("top-k must be a positive integer")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-query")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("analyze", "plan", "run"):
        child = subparsers.add_parser(command)
        child.add_argument("text", metavar="TEXT", type=_non_blank)
        child.add_argument("--profile", choices=_PROFILES, default="chat")
        child.add_argument("--format", choices=("table", "json"), default="table")
        if command != "analyze":
            child.add_argument("--top-k", type=_positive_int, default=None)
            child.add_argument("--rewrite-mode", choices=_REWRITE_MODES, default="auto")
        if command == "run":
            child.add_argument("--debug", action="store_true")
    return parser


def main(argv: list[str] | None = None, engine: QueryEngine | None = None) -> int:
    """Run one analysis, planning, or retrieval command and return an exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    request = QueryRequest(
        text=args.text,
        profile=args.profile,
        top_k=getattr(args, "top_k", None),
        rewrite_mode=getattr(args, "rewrite_mode", RewriteMode.AUTO),
        debug=bool(getattr(args, "debug", False)),
    )
    try:
        if engine is None:
            # Importing the composition root only here makes --help and injected
            # offline engines independent from stores and LLM configuration.
            from app.rag.bootstrap import get_query_engine

            engine = get_query_engine()
        if args.command == "analyze":
            value = engine.analyze(request)
        elif args.command == "plan":
            value = engine.plan(request)
        else:
            value = engine.query(request)
        output = _render(value, args.command, args.format)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        summary = _terminal_text(safe_error_summary(exc))
        sys.stderr.write(f"查询命令失败: {summary}\n")
        return 1
    sys.stdout.write(output)
    return 1 if isinstance(value, QueryResult) and value.error is not None else 0


def _render(value: Any, command: str, output_format: str) -> str:
    if isinstance(value, QueryResult) and value.error is not None:
        value = replace(value, error=safe_error_summary(value.error))
    if output_format == "json":
        return json.dumps(_json_value(value), ensure_ascii=False, allow_nan=False) + "\n"
    if command == "analyze":
        return _render_analysis(value)
    if command == "plan":
        if not isinstance(value, QueryPlan):
            raise TypeError("plan command returned an unexpected value")
        return _render_plan(value)
    if not isinstance(value, QueryResult):
        raise TypeError("run command returned an unexpected value")
    return _render_result(value)


def _json_value(value: Any, active: set[int] | None = None) -> Any:
    """Convert supported domain values into strict JSON-compatible values."""
    active = set() if active is None else active
    if isinstance(value, StrEnum):
        return value.value
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return _with_cycle_guard(
            value,
            active,
            lambda: _json_mapping(value, active),
        )
    if isinstance(value, list | tuple):
        return _with_cycle_guard(
            value, active, lambda: [_json_value(item, active) for item in value]
        )
    if is_dataclass(value) and not isinstance(value, type):
        return _with_cycle_guard(
            value,
            active,
            lambda: {
                item.name: _json_value(getattr(value, item.name), active) for item in fields(value)
            },
        )
    raise TypeError(type(value).__name__)


def _json_mapping(value: Mapping[Any, Any], active: set[int]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, item in value.items():
        normalized_key = str(key)
        if normalized_key in converted:
            raise TypeError(type(value).__name__)
        converted[normalized_key] = _json_value(item, active)
    return converted


def _with_cycle_guard(value: Any, active: set[int], convert: Any) -> Any:
    identifier = id(value)
    if identifier in active:
        raise TypeError(type(value).__name__)
    active.add(identifier)
    try:
        return convert()
    finally:
        active.remove(identifier)


def _render_analysis(analysis: Any) -> str:
    return _table(
        (
            ("stage", "analyze"),
            ("profile", analysis.profile),
            ("normalized", analysis.normalized_text),
            ("intents", analysis.intents),
            ("filter", analysis.inferred_filter),
            ("complexity", analysis.complexity),
            ("confidence", analysis.confidence),
            ("context", analysis.context_dependent),
        )
    )


def _render_plan(plan: QueryPlan) -> str:
    return _table(_plan_rows(plan))


def _plan_rows(plan: QueryPlan) -> tuple[tuple[str, Any], ...]:
    return (
        ("stage", "plan"),
        ("profile", plan.analysis.profile),
        ("rewrite_source", plan.rewrite.source),
        ("primary_query", plan.rewrite.primary_text),
        ("queries", plan.queries),
        ("filter", plan.metadata_filter),
        ("route_first", plan.route_first),
        ("hybrid", plan.hybrid),
        ("final_top_k", plan.final_top_k),
        ("candidate_k", plan.candidate_k),
        ("max_attempts", plan.max_attempts),
    )


def _render_result(result: QueryResult) -> str:
    rows: list[tuple[str, Any]] = [("stage", "run"), *_plan_rows(result.plan)[1:]]
    rows.extend(
        (
            ("degraded", result.degraded),
            ("reasons", result.degradation_reasons),
            ("error", result.error),
            ("document_count", len(result.documents)),
        )
    )
    preview_length: int | None = None
    for index, document in enumerate(result.documents, 1):
        if preview_length is None:
            preview_length = _preview_length()
        rows.append((f"document[{index}].source", _document_source(document)))
        rows.append((f"document[{index}].preview", _preview(document.page_content, preview_length)))
    attempts = _attempts_used(result)
    rows.append(("attempt_budget", result.plan.max_attempts))
    rows.append(("attempts_used", attempts if attempts is not None else "unavailable"))
    for index, event in enumerate(result.trace, 1):
        rows.append((f"trace[{index}].stage", event.stage))
        rows.append((f"trace[{index}].status", event.status))
        rows.append((f"trace[{index}].duration_ms", event.duration_ms))
    return _table(tuple(rows))


def _attempts_used(result: QueryResult) -> int | None:
    for event in reversed(result.trace):
        value = event.data.get("attempts_used")
        if (
            event.stage == "execute"
            and not isinstance(value, bool)
            and isinstance(value, int)
            and value >= 0
        ):
            return value
    return None


def _document_source(document: Any) -> str:
    metadata = document.metadata or {}
    for key in ("_file_name", "_source", "source", "id"):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return str(document.id) if getattr(document, "id", None) else "-"


def _preview(text: Any, length: int) -> str:
    if length <= 0:
        return ""
    sample_length = max(length * 4, length + 32)
    if isinstance(text, str):
        sampled = text[:sample_length]
    else:
        sampled = str(text)[:sample_length]
    normalized = _terminal_text(sampled)
    return normalized[: max(0, length)]


def _preview_length() -> int:
    try:
        from app.config import config

        value = config.rag_query_debug_preview_length
    except Exception:
        return 240
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 240
    return value


def _table(rows: tuple[tuple[str, Any], ...]) -> str:
    return "".join(f"{key}: {_table_value(value)}\n" for key, value in rows)


def _table_value(value: Any) -> str:
    normalized = _json_value(value)
    if isinstance(normalized, (dict, list)):
        serialized = json.dumps(normalized, ensure_ascii=False, allow_nan=False, sort_keys=True)
        return _terminal_text(serialized)
    return _plain_text(str(normalized))


def _plain_text(value: str) -> str:
    """Keep one table row on one terminal line without changing JSON output."""
    return _terminal_text(value)


def _terminal_text(value: str) -> str:
    """Collapse terminal control characters, including C1 and ANSI ESC."""
    return re.sub(r"[\x00-\x1f\x7f-\x9f]+", " ", value).strip()


if __name__ == "__main__":
    raise SystemExit(main())
