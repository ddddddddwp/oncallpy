from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def load_stack_module():
    module_path = SCRIPTS_DIR / "energyagent_stack.py"
    spec = importlib.util.spec_from_file_location("energyagent_stack", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_legacy_stack_script_imports_as_repo_package() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib; "
                "canonical = importlib.import_module('scripts.energyagent_stack'); "
                "legacy = importlib.import_module('scripts.oncallpy_stack'); "
                "assert legacy.main is canonical.main; "
                "assert not hasattr(legacy, 'StackManager')"
            ),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_legacy_stack_script_loads_directly_under_arbitrary_module_name() -> None:
    module_path = SCRIPTS_DIR / "oncallpy_stack.py"
    spec = importlib.util.spec_from_file_location("arbitrary_legacy_stack", module_path)
    assert spec is not None
    assert spec.loader is not None
    legacy = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(legacy)

    assert callable(legacy.main)
    assert Path(legacy.main.__code__.co_filename).resolve() == SCRIPTS_DIR / "energyagent_stack.py"
    assert not hasattr(legacy, "StackManager")


def test_legacy_stack_script_loads_directly_under_dotted_arbitrary_name() -> None:
    module_path = SCRIPTS_DIR / "oncallpy_stack.py"
    spec = importlib.util.spec_from_file_location("tools.arbitrary_legacy_stack", module_path)
    assert spec is not None
    assert spec.loader is not None
    legacy = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(legacy)

    assert callable(legacy.main)
    assert Path(legacy.main.__code__.co_filename).resolve() == SCRIPTS_DIR / "energyagent_stack.py"
    assert not hasattr(legacy, "StackManager")


def test_canonical_stack_script_uses_energyagent_branding() -> None:
    script = (SCRIPTS_DIR / "energyagent_stack.py").read_text(encoding="utf-8").casefold()

    assert "energyagent" in script
    assert "oncallpy" not in script
    assert "superbizagent" not in script


class FakeSystem:
    def __init__(
        self,
        *,
        milvus_ready: bool = True,
        redis_ready: bool = True,
        redis_port_open: bool | None = None,
        redis_containers: list[str] | None = None,
    ) -> None:
        self.milvus_ready = milvus_ready
        self.redis_ping_ready = redis_ready
        self.redis_port_is_open = redis_ready if redis_port_open is None else redis_port_open
        self.redis_containers = redis_containers or ["1Panel-redis-VrI6"]
        self.actions: list[tuple[str, str]] = []

    def ensure_docker(self) -> None:
        self.actions.append(("check", "docker"))

    def milvus_stack_ready(self) -> bool:
        self.actions.append(("check", "milvus"))
        return self.milvus_ready

    def start_milvus_stack(self) -> None:
        self.actions.append(("start", "milvus"))
        self.milvus_ready = True

    def wait_for_milvus(self) -> None:
        self.actions.append(("wait", "milvus"))

    def redis_ready(self, redis_url: str) -> bool:
        self.actions.append(("check", "redis-ping"))
        return self.redis_ping_ready

    def redis_port_open(self, host: str, port: int) -> bool:
        self.actions.append(("check", f"redis:{host}:{port}"))
        return self.redis_port_is_open

    def redis_container_names(self) -> list[str]:
        self.actions.append(("list", "redis-containers"))
        return self.redis_containers

    def start_container(self, name: str) -> None:
        self.actions.append(("start", name))
        self.redis_ping_ready = True
        self.redis_port_is_open = True

    def wait_for_tcp(self, host: str, port: int, label: str) -> None:
        self.actions.append(("wait", f"{label}:{host}:{port}"))

    def wait_for_redis(self, redis_url: str) -> None:
        self.actions.append(("wait", "redis-ping"))
        if not self.redis_ping_ready:
            raise TimeoutError("redis authentication failed")

    def stop_service(self, service) -> None:
        self.actions.append(("stop", service.name))

    def start_service(self, service) -> None:
        self.actions.append(("start", service.name))

    def wait_for_service(self, service) -> None:
        self.actions.append(("wait", service.name))


def test_stack_manager_default_redis_url_has_no_credentials(monkeypatch) -> None:
    module = load_stack_module()
    monkeypatch.delenv("REDIS_URL", raising=False)

    manager = module.StackManager(project_root=Path("/repo"), system=FakeSystem())
    parsed = urlsplit(manager.redis_url)

    assert manager.redis_url == "redis://localhost:6379/0"
    assert manager.redis_url == module.DEFAULT_REDIS_URL
    assert parsed.username is None
    assert parsed.password is None
    assert "@" not in parsed.netloc


def test_stack_manager_reads_redis_url_from_project_dotenv(tmp_path, monkeypatch) -> None:
    module = load_stack_module()
    monkeypatch.delenv("REDIS_URL", raising=False)
    (tmp_path / ".env").write_text(
        "REDIS_URL=redis://:dotenv-secret@redis.local:6380/2\n",
        encoding="utf-8",
    )

    manager = module.StackManager(project_root=tmp_path, system=FakeSystem())

    assert manager.redis_url == "redis://:dotenv-secret@redis.local:6380/2"


def test_explicit_redis_url_overrides_environment_and_dotenv(tmp_path, monkeypatch) -> None:
    module = load_stack_module()
    monkeypatch.setenv("REDIS_URL", "redis://environment.local:6379/0")
    (tmp_path / ".env").write_text(
        "REDIS_URL=redis://dotenv.local:6379/0\n",
        encoding="utf-8",
    )

    manager = module.StackManager(
        project_root=tmp_path,
        system=FakeSystem(),
        redis_url="redis://explicit.local:6379/0",
    )

    assert manager.redis_url == "redis://explicit.local:6379/0"


def test_dependency_start_is_conditional() -> None:
    module = load_stack_module()
    system = FakeSystem(milvus_ready=True, redis_ready=True)
    manager = module.StackManager(project_root=Path("/repo"), system=system)

    manager.ensure_dependencies()

    assert ("start", "milvus") not in system.actions
    assert ("start", "1Panel-redis-VrI6") not in system.actions
    assert ("check", "docker") in system.actions
    assert ("check", "milvus") in system.actions
    assert ("check", "redis-ping") in system.actions


def test_missing_dependencies_are_started_without_restarting_healthy_ones() -> None:
    module = load_stack_module()
    system = FakeSystem(milvus_ready=False, redis_ready=False)
    manager = module.StackManager(project_root=Path("/repo"), system=system)

    manager.ensure_dependencies()

    assert ("start", "milvus") in system.actions
    assert ("wait", "milvus") in system.actions
    assert ("start", "1Panel-redis-VrI6") in system.actions
    assert ("wait", "redis:localhost:6379") in system.actions
    assert ("wait", "redis-ping") in system.actions


def test_open_redis_port_without_successful_ping_is_not_ready() -> None:
    module = load_stack_module()
    system = FakeSystem(redis_ready=False, redis_port_open=True)
    manager = module.StackManager(
        project_root=Path("/repo"),
        system=system,
        redis_url="redis://:wrong@localhost:6379/0",
    )

    with pytest.raises(TimeoutError, match="redis"):
        manager.ensure_dependencies()

    assert ("check", "redis-ping") in system.actions
    assert ("wait", "redis-ping") in system.actions


def test_ready_redis_uses_authenticated_url_without_exposing_it(capsys) -> None:
    module = load_stack_module()
    secret_url = "redis://:do-not-print@localhost:6379/0"
    system = FakeSystem(redis_ready=True, redis_port_open=True)
    manager = module.StackManager(
        project_root=Path("/repo"), system=system, redis_url=secret_url
    )

    manager.ensure_dependencies()

    assert secret_url not in capsys.readouterr().out


def test_restart_application_services_stops_before_starting() -> None:
    module = load_stack_module()
    system = FakeSystem()
    manager = module.StackManager(project_root=Path("/repo"), system=system)

    manager.restart_application_services()

    expected_names = ["mcp-monitor", "mcp-cls", "fastapi"]
    stop_actions = [action for action in system.actions if action[0] == "stop"]
    start_actions = [action for action in system.actions if action[0] == "start"]
    wait_actions = [action for action in system.actions if action[0] == "wait"]

    assert [name for _, name in stop_actions] == expected_names
    assert [name for _, name in start_actions] == expected_names
    assert [name for _, name in wait_actions] == expected_names
    assert system.actions.index(("stop", "fastapi")) < system.actions.index(
        ("start", "mcp-monitor")
    )


def test_stack_manager_places_service_logs_in_configured_log_dir() -> None:
    module = load_stack_module()
    manager = module.StackManager(
        project_root=Path("/repo"), system=FakeSystem(), log_dir="runtime-logs"
    )

    log_files = {service.name: service.log_file for service in manager.services}

    assert log_files == {
        "mcp-monitor": Path("runtime-logs") / "mcp_monitor.log",
        "mcp-cls": Path("runtime-logs") / "mcp_cls.log",
        "fastapi": Path("runtime-logs") / "server.log",
    }


def test_docker_system_resolves_and_creates_nested_service_log_dir(tmp_path, monkeypatch) -> None:
    module = load_stack_module()
    service = module.ServiceSpec(
        name="dummy",
        command=["python", "-c", "print('ready')"],
        pid_file="dummy.pid",
        log_file=Path("runtime-logs") / "dummy.log",
        pattern="dummy",
        port=1234,
    )
    system = module.DockerSystem(tmp_path)

    class DummyProcess:
        pid = 12345

    def fake_popen(*_args, **_kwargs):
        return DummyProcess()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)

    system.start_service(service)

    assert (tmp_path / "runtime-logs").is_dir()
    assert (tmp_path / "runtime-logs" / "dummy.log").exists()
    assert (tmp_path / "dummy.pid").read_text(encoding="utf-8") == "12345\n"
