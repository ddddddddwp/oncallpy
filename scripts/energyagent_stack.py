#!/usr/bin/env python3
"""Check dependencies and restart the EnergyAgent application services."""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

from dotenv import dotenv_values

DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_LOG_DIR = "runtime-logs"


def project_redis_url(project_root: Path) -> str | None:
    value = dotenv_values(project_root / ".env").get("REDIS_URL")
    return str(value).strip() if value else None


def _redis_command(*parts: str) -> bytes:
    encoded = [part.encode("utf-8") for part in parts]
    body = [f"*{len(encoded)}\r\n".encode()]
    for part in encoded:
        body.extend((f"${len(part)}\r\n".encode(), part, b"\r\n"))
    return b"".join(body)


def _read_redis_line(stream) -> bytes:
    response = stream.readline(4096)
    if not response.endswith(b"\r\n"):
        raise ConnectionError("incomplete Redis response")
    if response.startswith(b"-"):
        raise ConnectionError("Redis rejected the readiness command")
    return response


class ServiceSpec:
    def __init__(
        self,
        *,
        name: str,
        command: list[str],
        pid_file: str,
        log_file: str | Path,
        pattern: str,
        port: int,
        health_url: str | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.pid_file = pid_file
        self.log_file = Path(log_file)
        self.pattern = pattern
        self.port = port
        self.health_url = health_url


class StackManager:
    def __init__(
        self,
        *,
        project_root: Path,
        system: DockerSystem | None = None,
        redis_url: str | None = None,
        log_dir: str | Path | None = None,
    ) -> None:
        self.project_root = project_root
        self.system = system or DockerSystem(project_root)
        self.redis_url = (
            redis_url
            or os.environ.get("REDIS_URL")
            or project_redis_url(project_root)
            or DEFAULT_REDIS_URL
        )
        self.log_dir = Path(log_dir or os.environ.get("LOG_DIR") or DEFAULT_LOG_DIR).expanduser()
        self.services = [
            ServiceSpec(
                name="mcp-monitor",
                command=[".venv/bin/python", "mcp_servers/monitor_server.py"],
                pid_file="mcp_monitor.pid",
                log_file=self.log_dir / "mcp_monitor.log",
                pattern="mcp_servers/monitor_server.py",
                port=8004,
            ),
            ServiceSpec(
                name="mcp-cls",
                command=[".venv/bin/python", "mcp_servers/cls_server.py"],
                pid_file="mcp_cls.pid",
                log_file=self.log_dir / "mcp_cls.log",
                pattern="mcp_servers/cls_server.py",
                port=8003,
            ),
            ServiceSpec(
                name="fastapi",
                command=[
                    ".venv/bin/python",
                    "-m",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9900",
                    "--log-level",
                    "info",
                ],
                pid_file="server.pid",
                log_file=self.log_dir / "server.log",
                pattern="uvicorn app.main:app",
                port=9900,
                health_url="http://127.0.0.1:9900/health",
            ),
        ]

    def ensure_dependencies(self) -> None:
        self.system.ensure_docker()

        if self.system.milvus_stack_ready():
            print("milvus: ready")
        else:
            print("milvus: starting docker compose stack")
            self.system.start_milvus_stack()
            self.system.wait_for_milvus()

        redis_host, redis_port = self._redis_endpoint()
        if self.system.redis_ready(self.redis_url):
            print(f"redis: ready on {redis_host}:{redis_port}")
            return

        if not self.system.redis_port_open(redis_host, redis_port):
            redis_containers = self.system.redis_container_names()
            if not redis_containers:
                raise RuntimeError(
                    "redis: port is closed and no Docker container with name matching 'redis' was found"
                )

            container_name = redis_containers[0]
            print(f"redis: starting existing container {container_name}")
            self.system.start_container(container_name)
            self.system.wait_for_tcp(redis_host, redis_port, "redis")

        self.system.wait_for_redis(self.redis_url)
        print(f"redis: ready on {redis_host}:{redis_port}")

    def restart_application_services(self) -> None:
        for service in self.services:
            self.system.stop_service(service)

        for service in self.services:
            self.system.start_service(service)
            self.system.wait_for_service(service)

    def restart(self) -> None:
        self.ensure_dependencies()
        self.restart_application_services()

    def _redis_endpoint(self) -> tuple[str, int]:
        parsed = urlparse(self.redis_url)
        return parsed.hostname or "localhost", parsed.port or 6379


class DockerSystem:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root

    def ensure_docker(self) -> None:
        self._run(["docker", "info"])

    def milvus_stack_ready(self) -> bool:
        checks = {
            "milvus-etcd": True,
            "milvus-minio": True,
            "milvus-standalone": True,
            "milvus-attu": False,
        }
        return all(
            self._container_ready(name, require_health=health) for name, health in checks.items()
        )

    def start_milvus_stack(self) -> None:
        self._run(["docker", "compose", "-f", "vector-database.yml", "up", "-d"])

    def wait_for_milvus(self) -> None:
        self._wait_until(self.milvus_stack_ready, "milvus", timeout=180)

    def redis_port_open(self, host: str, port: int) -> bool:
        return self._port_open(host, port)

    def redis_ready(self, redis_url: str) -> bool:
        parsed = urlparse(redis_url)
        if parsed.scheme != "redis":
            return False
        host = parsed.hostname or "localhost"
        port = parsed.port or 6379
        username = unquote(parsed.username) if parsed.username else None
        password = unquote(parsed.password) if parsed.password else None
        database = parsed.path.lstrip("/") or "0"

        try:
            with socket.create_connection((host, port), timeout=1) as connection:
                with connection.makefile("rb") as stream:

                    def execute(expected: bytes, *parts: str) -> bool:
                        connection.sendall(_redis_command(*parts))
                        return _read_redis_line(stream) == expected

                    if password:
                        auth = (
                            ("AUTH", username, password)
                            if username
                            else ("AUTH", password)
                        )
                        if not execute(b"+OK\r\n", *auth):
                            return False
                    if database != "0" and not execute(
                        b"+OK\r\n", "SELECT", database
                    ):
                        return False
                    return execute(b"+PONG\r\n", "PING")
        except (ConnectionError, OSError, UnicodeError, ValueError):
            return False

    def redis_container_names(self) -> list[str]:
        result = self._run(
            ["docker", "ps", "-a", "--filter", "name=redis", "--format", "{{.Names}}"],
            capture_output=True,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def start_container(self, name: str) -> None:
        self._run(["docker", "start", name])

    def wait_for_tcp(self, host: str, port: int, label: str) -> None:
        self._wait_until(lambda: self._port_open(host, port), label, timeout=60)

    def wait_for_redis(self, redis_url: str) -> None:
        self._wait_until(lambda: self.redis_ready(redis_url), "redis", timeout=60)

    def stop_service(self, service: ServiceSpec) -> None:
        pid_path = self.project_root / service.pid_file
        pid = self._read_pid(pid_path)
        if pid is not None and self._pid_running(pid):
            print(f"{service.name}: stopping pid {pid}")
            self._terminate_pid(pid)

        self._run(["pkill", "-f", service.pattern], check=False)
        pid_path.unlink(missing_ok=True)

    def start_service(self, service: ServiceSpec) -> None:
        log_path = service.log_file
        if not log_path.is_absolute():
            log_path = self.project_root / log_path
        print(f"{service.name}: starting")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("ab")
        try:
            process = subprocess.Popen(
                service.command,
                cwd=self.project_root,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        finally:
            log_handle.close()
        (self.project_root / service.pid_file).write_text(f"{process.pid}\n", encoding="utf-8")

    def wait_for_service(self, service: ServiceSpec) -> None:
        if service.health_url:
            self._wait_until(lambda: self._http_ok(service.health_url), service.name, timeout=90)
            return
        self.wait_for_tcp("127.0.0.1", service.port, service.name)

    def _run(
        self,
        args: list[str],
        *,
        check: bool = True,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=self.project_root,
            check=check,
            text=True,
            capture_output=capture_output,
        )

    def _container_ready(self, name: str, *, require_health: bool) -> bool:
        result = self._run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}no-healthcheck{{end}}",
                name,
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            return False
        state, _, health = result.stdout.strip().partition(" ")
        if state != "running":
            return False
        return health == "healthy" if require_health else True

    def _port_open(self, host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            return False

    def _http_ok(self, url: str) -> bool:
        try:
            with urlopen(url, timeout=2) as response:
                return 200 <= response.status < 300
        except OSError:
            return False

    def _wait_until(self, predicate, label: str, *, timeout: int) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                print(f"{label}: ready")
                return
            time.sleep(1)
        raise TimeoutError(f"{label}: timed out after {timeout}s")

    def _read_pid(self, path: Path) -> int | None:
        try:
            text = path.read_text(encoding="utf-8").strip()
            return int(text) if text else None
        except (FileNotFoundError, ValueError):
            return None

    def _pid_running(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _terminate_pid(self, pid: int) -> None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if not self._pid_running(pid):
                return
            time.sleep(0.5)

        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            return


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("restart", "deps"),
        nargs="?",
        default="restart",
        help="restart EnergyAgent services after ensuring dependencies, or only ensure dependencies",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root; defaults to the parent of scripts/",
    )
    parser.add_argument(
        "--redis-url",
        default=None,
        help="Redis URL used to derive the host and port; defaults to REDIS_URL or project default",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="directory for service logs; defaults to LOG_DIR or runtime-logs",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    manager = StackManager(
        project_root=args.project_root.resolve(),
        redis_url=args.redis_url,
        log_dir=args.log_dir,
    )
    try:
        if args.action == "deps":
            manager.ensure_dependencies()
        else:
            manager.restart()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
