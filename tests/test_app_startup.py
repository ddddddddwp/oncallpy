from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi import APIRouter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_isolated_logger_probe(tmp_path: Path, probe: str) -> dict[str, bool]:
    log_dir = tmp_path / "runtime-logs"
    fake_config = tmp_path / "config.yaml"
    fake_config.write_text(f"log_dir: {log_dir.as_posix()}\n", encoding="utf-8")
    environment = {
        "ENERGY_AGENT_CONFIG_FILES": str(fake_config),
        "HOME": str(tmp_path),
        "PYTHONPATH": str(PROJECT_ROOT),
    }

    assert Path(sys.executable).is_absolute()
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    assert set(payload.values()) <= {True, False}
    return payload


def test_importing_app_and_logger_module_does_not_configure_logging(tmp_path: Path) -> None:
    result = _run_isolated_logger_probe(
        tmp_path,
        """
import json
from pathlib import Path
import app

log_dir = Path.home() / "runtime-logs"
absent_after_app_import = not log_dir.exists()
import app.utils.logger
payload = {
    "log_dir_absent_after_app_import": absent_after_app_import,
    "log_dir_absent_after_logger_import": not log_dir.exists(),
}
print(json.dumps(payload, sort_keys=True))
""",
    )

    assert result == {
        "log_dir_absent_after_app_import": True,
        "log_dir_absent_after_logger_import": True,
    }


def test_setup_logger_configures_once_per_call_without_handler_growth(tmp_path: Path) -> None:
    result = _run_isolated_logger_probe(
        tmp_path,
        """
import json
from pathlib import Path
from loguru import logger
from app.utils.logger import setup_logger

log_dir = Path.home() / "runtime-logs"
absent_before_setup = not log_dir.exists()
setup_logger()
first_handler_count = len(logger._core.handlers)
exists_after_first_setup = log_dir.is_dir()
setup_logger()
second_handler_count = len(logger._core.handlers)
payload = {
    "log_dir_absent_before_setup": absent_before_setup,
    "log_dir_exists_after_first_setup": exists_after_first_setup,
    "handler_count_stable_after_second_setup": (
        first_handler_count == second_handler_count == 2
    ),
}
print(json.dumps(payload, sort_keys=True))
""",
    )

    assert result == {
        "log_dir_absent_before_setup": True,
        "log_dir_exists_after_first_setup": True,
        "handler_count_stable_after_second_setup": True,
    }


def _load_isolated_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "static").mkdir()

    fake_config_module = types.ModuleType("app.config")
    fake_config_module.config = types.SimpleNamespace(
        app_name="EnergyAgent",
        app_version="test-version",
        debug=False,
        host="127.0.0.1",
        port=9900,
        log_dir=str(tmp_path / "runtime-logs"),
    )
    monkeypatch.setitem(sys.modules, "app.config", fake_config_module)

    fake_api = types.ModuleType("app.api")
    for name in ("aiops", "chat", "file", "health", "oncall"):
        module = types.ModuleType(f"app.api.{name}")
        module.router = APIRouter()
        if name == "oncall":
            module.router.add_api_route(
                "/oncall-probe",
                lambda: {"status": "ok"},
                methods=["GET"],
            )
        setattr(fake_api, name, module)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    fake_api.aiops.aiops_service = None
    fake_api.aiops.multi_agent_aiops_service = None
    fake_api.chat.rag_agent_service = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "app.api", fake_api)

    fake_observability = types.ModuleType("app.core.observability")
    fake_observability.initialize_cozeloop = lambda: None
    fake_observability.shutdown_cozeloop = lambda: None

    @contextmanager
    def noop_trace_span(*args, **kwargs):
        _ = (args, kwargs)
        yield

    fake_observability.trace_span = noop_trace_span
    monkeypatch.setitem(sys.modules, "app.core.observability", fake_observability)

    fake_milvus = types.ModuleType("app.rag.milvus_client")
    fake_milvus.milvus_manager = types.SimpleNamespace(connect=lambda: None, close=lambda: None)
    monkeypatch.setitem(sys.modules, "app.rag.milvus_client", fake_milvus)

    script_path = PROJECT_ROOT / "app" / "main.py"
    spec = importlib.util.spec_from_file_location("test_isolated_app_main", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_registers_oncall_routes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_module = _load_isolated_main(tmp_path, monkeypatch)

    assert "/api/oncall-probe" in {route.path for route in main_module.app.routes}


@pytest.mark.asyncio
async def test_lifespan_initializes_logging_before_observability_and_startup_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_module = _load_isolated_main(tmp_path, monkeypatch)
    events: list[str] = []

    monkeypatch.setattr(main_module, "setup_logger", lambda: events.append("setup_logger"))
    monkeypatch.setattr(
        main_module,
        "initialize_cozeloop",
        lambda: events.append("initialize_cozeloop"),
    )
    monkeypatch.setattr(
        main_module,
        "shutdown_cozeloop",
        lambda: events.append("shutdown_cozeloop"),
    )

    @contextmanager
    def fake_trace_span(name: str, **kwargs):
        _ = kwargs
        events.append(f"trace_enter:{name}")
        try:
            yield
        finally:
            events.append(f"trace_exit:{name}")

    monkeypatch.setattr(main_module, "trace_span", fake_trace_span)
    monkeypatch.setattr(
        main_module,
        "logger",
        types.SimpleNamespace(info=lambda message: events.append(f"log:{message}")),
    )
    monkeypatch.setattr(
        main_module,
        "milvus_manager",
        types.SimpleNamespace(
            connect=lambda: events.append("milvus_connect"),
            close=lambda: events.append("milvus_close"),
        ),
    )

    async def initialize_checkpointer() -> None:
        events.append("checkpoint_initialize")

    async def cleanup_rag_agent() -> None:
        events.append("rag_cleanup")

    main_module.chat.rag_agent_service._initialize_checkpointer = initialize_checkpointer
    main_module.chat.rag_agent_service.cleanup = cleanup_rag_agent

    async with main_module.lifespan(main_module.app):
        events.append("lifespan_yield")

    assert events[0] == "setup_logger"
    assert events.count("setup_logger") == 1
    assert events.index("setup_logger") < events.index("initialize_cozeloop")
    assert events.index("setup_logger") < next(
        index for index, event in enumerate(events) if event.startswith("log:")
    )
    assert "milvus_connect" in events
    assert "milvus_close" in events
    assert events[-1] == "shutdown_cozeloop"
