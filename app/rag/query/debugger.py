"""Safe, structured debug output for the query pipeline."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, TextIO

from app.rag.query.models import QueryTraceEvent

MAX_TRACE_DEPTH = 64
_SENSITIVE_KEY_PARTS = ("apikey", "token", "password", "secret")
_SENSITIVE_KEY_NAMES = frozenset(
    {"authorization", "passwd", "credential", "credentials"}
)
_PREVIEW_KEYS = frozenset({"document", "content", "preview"})


def sanitize_trace_data(
    data: Mapping[str, Any],
    preview_length: int = 200,
    max_depth: int = MAX_TRACE_DEPTH,
) -> dict[str, Any]:
    """Return JSON-safe trace data with credentials redacted and previews bounded.

    Preview keys are matched case-insensitively but must otherwise be exact; for
    example, ``document_type`` is not treated as a document preview.
    """
    if not _is_non_negative_int(preview_length):
        raise ValueError("preview_length must be a non-negative integer")
    if not _is_non_negative_int(max_depth):
        raise ValueError("max_depth must be a non-negative integer")
    return _sanitize_mapping(data, preview_length, max_depth, path_ids={id(data)}, depth=0)


def _sanitize_mapping(
    data: Mapping[Any, Any],
    preview_length: int,
    max_depth: int,
    *,
    path_ids: set[int],
    depth: int,
) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in data.items():
        json_key = _json_key(key)
        key_name = key.casefold() if isinstance(key, str) else ""
        normalized_key = _normalize_key(key_name)
        if _is_sensitive_key(normalized_key):
            sanitized[json_key] = "***"
        else:
            sanitized[json_key] = _sanitize_value(
                value,
                preview_length,
                max_depth,
                path_ids=path_ids,
                depth=depth,
                truncate=isinstance(key, str) and key_name in _PREVIEW_KEYS,
            )
    return sanitized


def _sanitize_value(
    value: Any,
    preview_length: int,
    max_depth: int,
    *,
    path_ids: set[int],
    depth: int,
    truncate: bool = False,
) -> Any:
    if isinstance(value, Mapping | list | tuple):
        if id(value) in path_ids:
            return "<circular>"
        if depth >= max_depth:
            return "<max-depth>"
        path_ids.add(id(value))
        try:
            if isinstance(value, Mapping):
                return _sanitize_mapping(
                    value,
                    preview_length,
                    max_depth,
                    path_ids=path_ids,
                    depth=depth + 1,
                )
            if isinstance(value, list):
                return [
                    _sanitize_value(
                        item,
                        preview_length,
                        max_depth,
                        path_ids=path_ids,
                        depth=depth + 1,
                    )
                    for item in value
                ]
            return tuple(
                _sanitize_value(
                    item,
                    preview_length,
                    max_depth,
                    path_ids=path_ids,
                    depth=depth + 1,
                )
                for item in value
            )
        finally:
            path_ids.remove(id(value))
    if isinstance(value, str):
        if truncate and len(value) > preview_length:
            return value[:preview_length] + "…"
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, bool | int | float):
        return value
    return f"<unsupported trace value: {type(value).__name__}>"


def _json_key(key: Any) -> str:
    if isinstance(key, str | int | float | bool) or key is None:
        return str(key)
    return f"<unsupported trace key: {type(key).__name__}>"


def _is_non_negative_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _normalize_key(key: str) -> str:
    return key.replace("_", "").replace("-", "").replace(" ", "")


def _is_sensitive_key(normalized_key: str) -> bool:
    return any(part in normalized_key for part in _SENSITIVE_KEY_PARTS) or normalized_key in _SENSITIVE_KEY_NAMES


class NoOpQueryDebugger:
    """Discard query trace events when debugging is disabled."""

    def emit(self, event: QueryTraceEvent) -> None:
        return None


class MemoryQueryDebugger:
    """Keep immutable snapshots of trace events for in-process diagnostics."""

    def __init__(self) -> None:
        self._events: list[QueryTraceEvent] = []

    @property
    def events(self) -> tuple[QueryTraceEvent, ...]:
        """Expose copies so consumers cannot mutate stored snapshots."""
        return tuple(_snapshot_event(event) for event in self._events)

    def emit(self, event: QueryTraceEvent) -> None:
        self._events.append(_snapshot_event(event))


class ConsoleQueryDebugger:
    """Write one sanitized trace event per line to an injected text stream."""

    def __init__(self, stream: TextIO, preview_length: int = 200) -> None:
        if not _is_non_negative_int(preview_length):
            raise ValueError("preview_length must be a non-negative integer")
        self._stream = stream
        # The CLI composition root can inject rag_query_debug_preview_length
        # without making the default application bootstrap write to stderr.
        self._preview_length = preview_length

    def emit(self, event: QueryTraceEvent) -> None:
        payload = {
            "stage": event.stage,
            "status": event.status,
            "duration_ms": _finite_number_or_none(event.duration_ms),
            "data": sanitize_trace_data(event.data, preview_length=self._preview_length),
        }
        _write_all(self._stream, json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def _snapshot_event(event: QueryTraceEvent) -> QueryTraceEvent:
    return QueryTraceEvent(
        stage=event.stage,
        status=event.status,
        duration_ms=event.duration_ms,
        data=deepcopy(event.data),
    )


def _finite_number_or_none(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if not isinstance(value, float) or math.isfinite(value) else None


def _write_all(stream: TextIO, line: str) -> None:
    remaining = line
    while remaining:
        try:
            written = stream.write(remaining)
        except Exception as exc:
            raise OSError("query debug stream write failed") from exc
        if isinstance(written, bool) or not isinstance(written, int) or written <= 0:
            raise OSError("query debug stream write made no progress")
        if written > len(remaining):
            raise OSError("query debug stream reported an invalid write length")
        remaining = remaining[written:]
