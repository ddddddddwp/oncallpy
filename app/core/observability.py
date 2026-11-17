"""可选的 CozeLoop 观测层封装。"""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from uuid import uuid4
from typing import Any

from loguru import logger

from app.config import config


class _NoopSpan:
    """Fallback span，确保观测缺失时不影响业务主流程。"""

    def set_input(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def set_output(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None


TraceContext = dict[str, Any]


_cozeloop_trace_context: ContextVar[TraceContext | None] = ContextVar(
    "cozeloop_trace_context",
    default=None,
)


def new_turn_id(session_id: str) -> str:
    """为会话生成一轮请求的唯一 ID。"""

    return f"{session_id}:{uuid4().hex[:12]}"


def new_request_id() -> str:
    """为单次请求生成唯一 ID。"""

    return uuid4().hex


def get_trace_context() -> TraceContext:
    """读取当前协程链路上的 observability 上下文。"""

    return dict(_cozeloop_trace_context.get() or {})


def _normalize_trace_context(trace_context: Mapping[str, Any] | None) -> TraceContext:
    if not trace_context:
        return {}

    return {k: v for k, v in dict(trace_context).items() if v is not None}


def _attach_trace_tags(span: Any, trace_context: Mapping[str, Any]) -> None:
    """将上下文写入 span 的扩展字段。"""

    thread_id = trace_context.get("session_id")
    if thread_id is not None:
        _call_if_supported(span, "set_thread_id", str(thread_id))

    user_id = trace_context.get("user_id")
    if user_id is not None:
        _call_if_supported(span, "set_user_id", str(user_id))

    for key, value in trace_context.items():
        if value is None:
            continue
        _call_if_supported(span, "set_tag_item", key, str(value))


def _call_if_supported(obj: Any, method: str, *args: Any) -> None:
    if obj is None:
        return
    handler = getattr(obj, method, None)
    if callable(handler):
        try:
            handler(*args)
        except Exception as exc:  # pragma: no cover - SDK 行为差异
            logger.warning(f"CozeLoop 调用 {method} 失败: {exc}")


def _load_cozeloop_module():
    try:
        import importlib

        return importlib.import_module("cozeloop")
    except Exception as exc:  # pragma: no cover - 未安装依赖时不打断流程
        logger.warning(f"CozeLoop SDK 未安装或不可用: {exc}")
        return None


_cozeloop_module: Any | None = None
_cozeloop_client: Any | None = None


def initialize_cozeloop() -> bool:
    """初始化 CozeLoop（若配置完整且 SDK 可用）。"""

    global _cozeloop_client, _cozeloop_module
    if not config.cozeloop_workspace_id or not config.cozeloop_api_token:
        logger.warning("CozeLoop 配置缺失（COZELOOP_WORKSPACE_ID / COZELOOP_API_TOKEN），跳过观测初始化")
        _cozeloop_module = None
        _cozeloop_client = None
        return False

    module = _load_cozeloop_module()
    if module is None:
        _cozeloop_module = None
        _cozeloop_client = None
        return False

    client_factory = getattr(module, "new_client", None)
    if not callable(client_factory):
        logger.warning("CozeLoop SDK 未提供 new_client，跳过观测初始化")
        _cozeloop_module = None
        _cozeloop_client = None
        return False

    client_kwargs = {
        "workspace_id": config.cozeloop_workspace_id,
        "api_token": config.cozeloop_api_token,
    }
    if config.cozeloop_api_base_url:
        client_kwargs["api_base_url"] = config.cozeloop_api_base_url

    try:
        _cozeloop_client = client_factory(**client_kwargs)
    except Exception as exc:  # pragma: no cover - SDK 初始化失败时不影响业务
        logger.warning(f"CozeLoop 初始化失败: {exc}")
        _cozeloop_module = None
        _cozeloop_client = None
        return False

    _cozeloop_module = module
    logger.info("CozeLoop Trace 初始化完成")
    return True


def shutdown_cozeloop() -> None:
    """关闭 CozeLoop 会话。"""

    global _cozeloop_client, _cozeloop_module
    if _cozeloop_client is None and _cozeloop_module is None:
        return

    close_target = _cozeloop_client or _cozeloop_module
    close_handler = getattr(close_target, "close", None)
    if callable(close_handler):
        try:
            close_handler()
        except Exception as exc:  # pragma: no cover - 兼容性兜底
            logger.warning(f"CozeLoop close 调用失败: {exc}")
    _cozeloop_client = None
    _cozeloop_module = None


def is_cozeloop_enabled() -> bool:
    """检查 CozeLoop 是否可用。"""

    return _cozeloop_client is not None


@contextmanager
def trace_span(
    name: str,
    span_type: str = "custom",
    input_data: Any | None = None,
    set_output: Callable[[], Any] | Any | None = None,
    trace_context: Mapping[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """安全的 span 上报封装，SDK 缺失时返回空 span。"""

    merged_context: TraceContext = get_trace_context()
    merged_context.update(_normalize_trace_context(trace_context))

    merged_input: dict[str, Any]
    if isinstance(input_data, dict):
        merged_input = dict(merged_context)
        merged_input.update(input_data)
    elif input_data is not None:
        merged_input = dict(merged_context)
        merged_input["input"] = input_data
    else:
        merged_input = dict(merged_context)

    token = None
    span = _NoopSpan()
    try:
        if merged_context:
            token = _cozeloop_trace_context.set(merged_context)

        if is_cozeloop_enabled():
            span_creator = (
                getattr(_cozeloop_client, "start_span", None)
                if _cozeloop_client
                else None
            )
            if callable(span_creator):
                try:
                    span = span_creator(name, span_type)
                    _attach_trace_tags(span, merged_context)
                except Exception as exc:  # pragma: no cover - SDK 行为差异
                    logger.warning(f"CozeLoop span 创建失败（{name}）: {exc}")
            else:
                logger.warning("CozeLoop SDK 未提供 start_span，已跳过本次观测")

        _call_if_supported(span, "set_input", merged_input)

        if not is_cozeloop_enabled():
            yield span
            return

        yield span

        if callable(set_output):
            output_value = set_output()
        else:
            output_value = set_output
        _call_if_supported(span, "set_output", output_value)
    except Exception as exc:
        _call_if_supported(
            span,
            "set_output",
            {"status": "error", "message": str(exc)},
        )
        logger.warning(f"CozeLoop span 执行失败（{name}）: {exc}")
        raise
    finally:
        if token is not None:
            _cozeloop_trace_context.reset(token)
        _call_if_supported(span, "finish")


__all__ = [
    "initialize_cozeloop",
    "is_cozeloop_enabled",
    "get_trace_context",
    "new_request_id",
    "new_turn_id",
    "shutdown_cozeloop",
    "trace_span",
]
