import importlib
import json
import logging
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

FAKE_REDIS_URL = "redis://:fake-password@redis.invalid:6379/0"
SECRET_EXCEPTION_TEXT = "fake-password must not leak"


@pytest.fixture
def health_module() -> ModuleType:
    return importlib.import_module("app.api.health")


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    health_module: ModuleType,
    app_name: str = "EnergyAgent",
) -> None:
    monkeypatch.setattr(
        health_module,
        "config",
        SimpleNamespace(
            app_name=app_name,
            app_version="test-version",
            redis_url=FAKE_REDIS_URL,
        ),
    )


def _payload(response) -> dict:
    return json.loads(response.body)


def test_importing_health_does_not_import_milvus_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "app.api.health", raising=False)
    monkeypatch.delitem(sys.modules, "app.rag.milvus_client", raising=False)

    importlib.import_module("app.api.health")

    assert "app.rag.milvus_client" not in sys.modules


def test_lazy_milvus_manager_delegates_and_propagates_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    health_module: ModuleType,
) -> None:
    error = RuntimeError(SECRET_EXCEPTION_TEXT)
    concrete_check = Mock(side_effect=error)
    fake_module = ModuleType("app.rag.milvus_client")
    fake_module.milvus_manager = SimpleNamespace(health_check=concrete_check)
    monkeypatch.setitem(sys.modules, "app.rag.milvus_client", fake_module)

    with pytest.raises(RuntimeError) as exc_info:
        health_module._LazyMilvusManager.health_check()

    assert exc_info.value is error
    concrete_check.assert_called_once_with()


@pytest.mark.asyncio
async def test_health_reports_all_components_when_dependencies_are_available(
    monkeypatch: pytest.MonkeyPatch,
    health_module: ModuleType,
) -> None:
    _configure(monkeypatch, health_module)
    monkeypatch.setattr(health_module.milvus_manager, "health_check", lambda: True)
    redis_check = AsyncMock(return_value=True)
    monkeypatch.setattr(health_module, "check_redis", redis_check)

    response = await health_module.health_check()
    payload = _payload(response)

    assert response.status_code == 200
    assert set(payload) == {"code", "message", "data"}
    assert payload["code"] == 200
    assert payload["message"] == "available"
    assert payload["data"]["status"] == "ok"
    assert payload["data"]["components"] == {
        "milvus": {"status": "ok", "message": "available"},
        "redis": {"status": "ok", "message": "available"},
        "configuration": {"status": "ok", "message": "loaded"},
    }
    assert FAKE_REDIS_URL not in response.body.decode()
    redis_check.assert_awaited_once_with(FAKE_REDIS_URL)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("milvus_available", "redis_available", "failed_component"),
    [
        (False, True, "milvus"),
        (True, False, "redis"),
    ],
)
async def test_health_is_unavailable_when_a_required_dependency_returns_false(
    monkeypatch: pytest.MonkeyPatch,
    health_module: ModuleType,
    milvus_available: bool,
    redis_available: bool,
    failed_component: str,
) -> None:
    _configure(monkeypatch, health_module)
    monkeypatch.setattr(
        health_module.milvus_manager,
        "health_check",
        lambda: milvus_available,
    )
    monkeypatch.setattr(
        health_module,
        "check_redis",
        AsyncMock(return_value=redis_available),
    )

    response = await health_module.health_check()
    payload = _payload(response)

    assert response.status_code == 503
    assert payload["code"] == 503
    assert payload["message"] == "unavailable"
    assert payload["data"]["status"] == "error"
    assert payload["data"]["components"][failed_component] == {
        "status": "error",
        "message": "unavailable",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_component", ["milvus", "redis"])
async def test_health_sanitizes_dependency_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    health_module: ModuleType,
    failed_component: str,
) -> None:
    _configure(monkeypatch, health_module)
    caplog.set_level(logging.WARNING, logger=health_module.__name__)

    if failed_component == "milvus":

        def raise_milvus_error() -> bool:
            raise RuntimeError(SECRET_EXCEPTION_TEXT)

        monkeypatch.setattr(
            health_module.milvus_manager,
            "health_check",
            raise_milvus_error,
        )
        monkeypatch.setattr(
            health_module,
            "check_redis",
            AsyncMock(return_value=True),
        )
    else:
        monkeypatch.setattr(health_module.milvus_manager, "health_check", lambda: True)
        monkeypatch.setattr(
            health_module,
            "check_redis",
            AsyncMock(side_effect=RuntimeError(SECRET_EXCEPTION_TEXT)),
        )

    response = await health_module.health_check()
    payload = _payload(response)

    assert response.status_code == 503
    assert payload["data"]["components"][failed_component] == {
        "status": "error",
        "message": "unavailable",
    }
    assert SECRET_EXCEPTION_TEXT not in response.body.decode()
    assert SECRET_EXCEPTION_TEXT not in caplog.text
    assert failed_component.capitalize() in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_non_energyagent_configuration_is_warning_but_does_not_fail_health(
    monkeypatch: pytest.MonkeyPatch,
    health_module: ModuleType,
) -> None:
    _configure(monkeypatch, health_module, app_name="AnotherService")
    monkeypatch.setattr(health_module.milvus_manager, "health_check", lambda: True)
    monkeypatch.setattr(
        health_module,
        "check_redis",
        AsyncMock(return_value=True),
    )

    response = await health_module.health_check()
    payload = _payload(response)

    assert response.status_code == 200
    assert payload["data"]["components"]["configuration"] == {
        "status": "warning",
        "message": "loaded",
    }


@pytest.mark.asyncio
async def test_check_redis_pings_with_short_timeouts_and_always_closes(
    monkeypatch: pytest.MonkeyPatch,
    health_module: ModuleType,
) -> None:
    client = SimpleNamespace(
        ping=AsyncMock(return_value=True),
        aclose=AsyncMock(),
    )
    from_url = Mock(return_value=client)
    monkeypatch.setattr(health_module.redis.Redis, "from_url", from_url)

    result = await health_module.check_redis(FAKE_REDIS_URL)

    assert result is True
    from_url.assert_called_once()
    args, kwargs = from_url.call_args
    assert args == (FAKE_REDIS_URL,)
    assert 0 < kwargs["socket_connect_timeout"] <= 2
    assert 0 < kwargs["socket_timeout"] <= 2
    client.ping.assert_awaited_once_with()
    client.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_check_redis_closes_and_sanitizes_a_ping_exception(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    health_module: ModuleType,
) -> None:
    client = SimpleNamespace(
        ping=AsyncMock(side_effect=RuntimeError(SECRET_EXCEPTION_TEXT)),
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(
        health_module.redis.Redis,
        "from_url",
        Mock(return_value=client),
    )
    caplog.set_level(logging.WARNING, logger=health_module.__name__)

    result = await health_module.check_redis(FAKE_REDIS_URL)

    assert result is False
    client.aclose.assert_awaited_once_with()
    assert SECRET_EXCEPTION_TEXT not in caplog.text
    assert FAKE_REDIS_URL not in caplog.text
    assert "Redis" in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_redis_construction_exception_is_sanitized_by_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    health_module: ModuleType,
) -> None:
    _configure(monkeypatch, health_module)
    monkeypatch.setattr(health_module.milvus_manager, "health_check", lambda: True)
    monkeypatch.setattr(
        health_module.redis.Redis,
        "from_url",
        Mock(side_effect=RuntimeError(SECRET_EXCEPTION_TEXT)),
    )
    caplog.set_level(logging.WARNING, logger=health_module.__name__)

    response = await health_module.health_check()
    payload = _payload(response)

    assert response.status_code == 503
    assert payload["data"]["components"]["redis"] == {
        "status": "error",
        "message": "unavailable",
    }
    assert SECRET_EXCEPTION_TEXT not in response.body.decode()
    assert SECRET_EXCEPTION_TEXT not in caplog.text
    assert "Redis" in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.asyncio
async def test_redis_close_exception_is_sanitized_by_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    health_module: ModuleType,
) -> None:
    _configure(monkeypatch, health_module)
    monkeypatch.setattr(health_module.milvus_manager, "health_check", lambda: True)
    client = SimpleNamespace(
        ping=AsyncMock(return_value=True),
        aclose=AsyncMock(side_effect=RuntimeError(SECRET_EXCEPTION_TEXT)),
    )
    monkeypatch.setattr(
        health_module.redis.Redis,
        "from_url",
        Mock(return_value=client),
    )
    caplog.set_level(logging.WARNING, logger=health_module.__name__)

    response = await health_module.health_check()
    payload = _payload(response)

    assert response.status_code == 503
    assert payload["data"]["components"]["redis"] == {
        "status": "error",
        "message": "unavailable",
    }
    assert SECRET_EXCEPTION_TEXT not in response.body.decode()
    assert SECRET_EXCEPTION_TEXT not in caplog.text
    assert "Redis" in caplog.text
    assert "RuntimeError" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError(SECRET_EXCEPTION_TEXT),
        ConnectionError(SECRET_EXCEPTION_TEXT),
    ],
    ids=["unexpected-exception", "connection-error"],
)
def test_real_milvus_health_check_logs_only_exception_type(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    monkeypatch.setenv("APP_NAME", "EnergyAgent")
    milvus_module = importlib.import_module("app.rag.milvus_client")
    manager = milvus_module.MilvusClientManager.__new__(milvus_module.MilvusClientManager)
    manager._client = object()
    monkeypatch.setattr(
        milvus_module.connections,
        "list_connections",
        Mock(side_effect=error),
    )
    caplog.set_level(logging.ERROR, logger=milvus_module.__name__)
    sink_id = milvus_module.logger.add(caplog.handler, format="{message}")

    try:
        result = manager.health_check()
    finally:
        milvus_module.logger.remove(sink_id)

    assert result is False
    assert SECRET_EXCEPTION_TEXT not in caplog.text
    assert "Milvus" in caplog.text
    assert type(error).__name__ in caplog.text
