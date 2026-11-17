"""Redis-backed LangGraph checkpoint helpers."""

# 本模块为 LangGraph 的 Redis checkpointer 提供统一封装：
# 创建 / 读取 / 删除 / 关闭，屏蔽不同 checkpointer 实现之间的 API 差异。

from __future__ import annotations

import inspect
from typing import Any

from loguru import logger

from app.config import config

try:
    # langgraph-checkpoint-redis 包提供的异步 Redis checkpointer
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
except ImportError:  # pragma: no cover - exercised by tests through monkeypatching.
    # 未安装依赖时降级为 None，在创建时抛出明确错误；测试可通过 monkeypatch 替换
    AsyncRedisSaver = None  # type: ignore[assignment]


async def create_redis_checkpointer(redis_url: str | None = None) -> Any:
    """Create and initialize the async Redis checkpointer used by LangGraph."""
    # 依赖未安装时立即报错，提示需要安装 langgraph-checkpoint-redis
    if AsyncRedisSaver is None:
        raise RuntimeError(
            "Redis checkpoint support is not installed. "
            "Install the langgraph-checkpoint-redis package."
        )

    # 优先使用显式传入的 URL，否则回退到全局配置中的 redis_url
    resolved_url = redis_url or config.redis_url
    try:
        saver = AsyncRedisSaver(redis_url=resolved_url)
        # asetup 完成底层连接与存储结构初始化，之后才能读写 checkpoint
        await saver.asetup()
        logger.info("Redis LangGraph checkpointer initialized")
        return saver
    except Exception as exc:
        # 包装原始异常，给出可操作的排查提示（检查 REDIS_URL 与 Redis 服务可用性）
        raise RuntimeError(
            "Failed to initialize Redis LangGraph checkpointer. "
            f"Check REDIS_URL and Redis availability. REDIS_URL={resolved_url!r}"
        ) from exc


async def get_latest_checkpoint(checkpointer: Any, config_dict: dict[str, Any]) -> Any:
    """Return the latest checkpoint data for a LangGraph config."""
    # 按 异步 -> 同步 的顺序探测可用接口，兼容不同版本的 checkpointer 实现：
    # 新版实现直接提供 aget，返回 checkpoint 数据本身
    if hasattr(checkpointer, "aget"):
        return await checkpointer.aget(config_dict)

    # 旧版异步实现提供 aget_tuple，返回 (checkpoint, metadata, ...) 元组，需要提取
    if hasattr(checkpointer, "aget_tuple"):
        checkpoint_tuple = await checkpointer.aget_tuple(config_dict)
        return _extract_checkpoint(checkpoint_tuple)

    # 同步实现的返回值可能是 awaitable，统一探测后再决定是否 await
    if hasattr(checkpointer, "get"):
        result = checkpointer.get(config_dict)
        if inspect.isawaitable(result):
            result = await result
        return _extract_checkpoint(result)

    if hasattr(checkpointer, "get_tuple"):
        result = checkpointer.get_tuple(config_dict)
        if inspect.isawaitable(result):
            result = await result
        return _extract_checkpoint(result)

    # 没有任何可读接口时，视为 checkpointer 不受支持
    raise RuntimeError("Configured LangGraph checkpointer cannot read checkpoints")


async def delete_checkpoint_thread(checkpointer: Any, thread_id: str) -> None:
    """Delete all checkpoints for a LangGraph thread."""
    # 按 thread_id 删除该会话的全部 checkpoint（对应 clear_session 场景），优先异步接口
    if hasattr(checkpointer, "adelete_thread"):
        await checkpointer.adelete_thread(thread_id)
        return

    # 同步删除接口的返回值同样可能是 awaitable，统一探测
    if hasattr(checkpointer, "delete_thread"):
        result = checkpointer.delete_thread(thread_id)
        if inspect.isawaitable(result):
            await result
        return

    raise RuntimeError("Configured LangGraph checkpointer cannot delete threads")


async def close_checkpointer(checkpointer: Any) -> None:
    """Close a checkpointer if the implementation exposes a close method."""
    # 优先异步 aclose，其次同步 close；两者都没有则直接跳过（该实现无需关闭）
    for method_name in ("aclose", "close"):
        close = getattr(checkpointer, method_name, None)
        if close is None:
            continue
        result = close()
        if inspect.isawaitable(result):
            await result
        return


def _extract_checkpoint(value: Any) -> Any:
    # get_tuple/get 的返回值可能是带 checkpoint 属性的包装对象
    if hasattr(value, "checkpoint"):
        return value.checkpoint
    # 也可能是 (checkpoint, metadata, parent_config) 元组，取第一个元素
    if isinstance(value, tuple) and value:
        return value[0]
    # 否则原样返回（已是 checkpoint 数据本身，或为 None）
    return value
