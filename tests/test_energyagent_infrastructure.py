from __future__ import annotations

import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = PROJECT_ROOT / "vector-database.yml"
DOCKERFILE_PATH = PROJECT_ROOT / "Dockerfile"
DOCKERIGNORE_PATH = PROJECT_ROOT / ".dockerignore"
REQUIRED_REDIS_PASSWORD = "${REDIS_PASSWORD:?REDIS_PASSWORD is required}"


def load_compose() -> dict[str, object]:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def test_compose_owns_password_protected_persistent_redis() -> None:
    compose = load_compose()
    services = compose["services"]
    redis = services["redis"]

    assert redis["image"] == "redis:8-alpine"
    assert redis["container_name"] == "energyagent-redis"
    assert redis["command"] == [
        "redis-server",
        "--appendonly",
        "yes",
        "--requirepass",
        REQUIRED_REDIS_PASSWORD,
    ]
    assert redis["environment"]["REDIS_PASSWORD"] == REQUIRED_REDIS_PASSWORD
    assert redis["ports"] == ["127.0.0.1:6379:6379"]
    assert redis["volumes"] == ["redis-data:/data"]
    assert redis["restart"] == "unless-stopped"
    assert "redis-data" in compose["volumes"]


def test_redis_healthcheck_uses_authenticated_ping() -> None:
    redis = load_compose()["services"]["redis"]
    healthcheck = redis["healthcheck"]
    test_command = " ".join(healthcheck["test"])

    assert 'REDISCLI_AUTH="$${REDIS_PASSWORD}"' in test_command
    assert "redis-cli ping" in test_command
    assert not re.search(r"(?:^|\s)-a(?:\s|$)", test_command)
    assert "ping" in test_command.casefold()
    assert healthcheck["interval"]
    assert healthcheck["timeout"]
    assert healthcheck["retries"] > 0


def test_compose_preserves_existing_milvus_topology() -> None:
    compose = load_compose()

    assert {"etcd", "minio", "standalone", "attu", "redis"} <= set(compose["services"])
    assert compose["services"]["minio"]["ports"] == ["9001:9001", "9000:9000"]
    assert compose["services"]["standalone"]["ports"] == ["19530:19530", "9091:9091"]
    assert compose["services"]["attu"]["ports"] == ["8000:3000"]
    assert compose["networks"]["default"]["name"] == "milvus"


def test_dockerfile_uses_uv_python_312_and_frozen_production_dependencies() -> None:
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
    lines = [line.strip() for line in dockerfile.splitlines() if line.strip()]

    assert lines[0].startswith("FROM ghcr.io/astral-sh/uv:python3.12")
    assert "COPY pyproject.toml uv.lock README.md ./" in lines
    assert "RUN uv sync --frozen --no-dev --no-install-project" in lines
    assert any(line.startswith("RUN uv sync --frozen --no-dev") for line in lines)
    assert "USER energyagent" in lines


def test_dockerfile_copies_only_required_runtime_sources() -> None:
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
    copy_lines = [
        line.strip() for line in dockerfile.splitlines() if line.strip().startswith("COPY ")
    ]

    assert "COPY app ./app" in copy_lines
    assert "COPY static ./static" in copy_lines
    assert "COPY config/energyagent.yaml ./config/energyagent.yaml" in copy_lines
    assert "COPY config ./config" not in copy_lines
    assert [line for line in copy_lines if line.startswith("COPY config/")] == [
        "COPY config/energyagent.yaml ./config/energyagent.yaml"
    ]
    assert not any(re.fullmatch(r"COPY\s+\.?/?\s+\.?/?", line) for line in copy_lines)
    assert ".env" not in dockerfile.casefold()
    assert "secret" not in dockerfile.casefold()


def test_dockerignore_is_default_deny_with_a_minimal_public_allowlist() -> None:
    lines = [
        line.strip()
        for line in DOCKERIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert lines[0] == "**"
    assert {line for line in lines if line.startswith("!")} == {
        "!.dockerignore",
        "!Dockerfile",
        "!pyproject.toml",
        "!uv.lock",
        "!README.md",
        "!app/",
        "!app/**",
        "!static/",
        "!static/**",
        "!config/",
        "!config/energyagent.yaml",
    }
    assert {
        ".env",
        ".env.*",
        "config/local*",
        ".energyagent/",
        ".git/",
        ".worktrees/",
        ".venv/",
        "**/__pycache__/",
        ".pytest_cache/",
        ".ruff_cache/",
        "htmlcov/",
        ".coverage*",
        "coverage.xml",
        "logs/",
        "runtime-logs/",
        "tests/",
    } <= set(lines)


def test_dockerfile_exposes_and_starts_energyagent_http_server() -> None:
    dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "EXPOSE 9900" in dockerfile
    assert (
        'CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", ' '"--port", "9900"]'
    ) in dockerfile
