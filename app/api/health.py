"""健康检查接口。"""

import logging

import redis.asyncio as redis
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import config

router = APIRouter()
logger = logging.getLogger(__name__)

REDIS_TIMEOUT_SECONDS = 1.0


class _LazyMilvusManager:
    """Delay importing the Milvus client until the health check executes."""

    @staticmethod
    def health_check() -> bool:
        from app.rag.milvus_client import milvus_manager as concrete_manager

        return bool(concrete_manager.health_check())


milvus_manager = _LazyMilvusManager()


def _log_component_exception(component: str, exc: Exception) -> None:
    logger.warning("%s health check failed: %s", component, type(exc).__name__)


async def check_redis(redis_url: str) -> bool:
    """Ping Redis and close the short-lived health-check client."""
    client = redis.Redis.from_url(
        redis_url,
        socket_connect_timeout=REDIS_TIMEOUT_SECONDS,
        socket_timeout=REDIS_TIMEOUT_SECONDS,
    )
    try:
        return bool(await client.ping())
    except Exception as exc:
        _log_component_exception("Redis", exc)
        return False
    finally:
        await client.aclose()


def _dependency_component(available: bool) -> dict[str, str]:
    return {
        "status": "ok" if available else "error",
        "message": "available" if available else "unavailable",
    }


@router.get("/health")
async def health_check():
    """检查服务配置以及 Milvus、Redis 依赖。"""
    try:
        milvus_available = bool(milvus_manager.health_check())
    except Exception as exc:
        _log_component_exception("Milvus", exc)
        milvus_available = False

    try:
        redis_available = await check_redis(config.redis_url)
    except Exception as exc:
        _log_component_exception("Redis", exc)
        redis_available = False

    dependencies_available = milvus_available and redis_available
    status_code = 200 if dependencies_available else 503
    components = {
        "milvus": _dependency_component(milvus_available),
        "redis": _dependency_component(redis_available),
        "configuration": {
            "status": "ok" if config.app_name == "EnergyAgent" else "warning",
            "message": "loaded",
        },
    }

    return JSONResponse(
        status_code=status_code,
        content={
            "code": status_code,
            "message": "available" if dependencies_available else "unavailable",
            "data": {
                "service": config.app_name,
                "version": config.app_version,
                "status": "ok" if dependencies_available else "error",
                "components": components,
            },
        },
    )
