# EnergyAgent Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Protect the current RAG baseline and establish EnergyAgent naming, layered configuration, secret-safe source configuration migration, compatibility helpers, atomic local persistence, Redis-owned local infrastructure, and expanded health reporting.

**Architecture:** Keep the existing `app` Python package and current RAG/Redis/Milvus behavior. Add small foundation modules under `app/core` and `app/domains/shared`, with tests that do not access real credentials or external providers. All real configuration migration is copy-first, fingerprint-verified, ignored by Git, and performed only after the migration tool passes isolated fixture tests.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic Settings, PyYAML, Redis, Docker Compose, pytest, Loguru.

---

## File Structure

- `app/config.py`: EnergyAgent defaults, layered YAML source, current and legacy provider profiles, compatibility environment aliases.
- `config/energyagent.yaml`: tracked non-secret defaults.
- `config/local.example.yaml`: tracked placeholder-only local override template.
- `app/core/config_migration.py`: secret-safe inventory, snapshot, fingerprint, dotenv merge, and known-key mapping logic.
- `scripts/migrate_oncall_config.py`: thin CLI over `app.core.config_migration`; never prints values.
- `app/core/compatibility.py`: legacy environment, state path, and local-file copy helpers.
- `app/domains/shared/atomic_json_repository.py`: atomic JSON persistence primitive for later alert, identity, and on-call packages.
- `scripts/energyagent_stack.py`: canonical stack CLI.
- `scripts/oncallpy_stack.py`: one-release forwarding shim.
- `Dockerfile`: EnergyAgent application image.
- `vector-database.yml`: add the Redis service and health checks used by EnergyAgent.
- `app/api/health.py`: component-based Milvus, Redis, and configuration health without secret disclosure.
- `tests/test_energyagent_branding.py`: product/package/UI/script naming contract.
- `tests/test_layered_config.py`: YAML/env precedence and provider profile tests.
- `tests/test_config_migration.py`: fixture-only migration, permissions, fingerprint, conflict, and redaction tests.
- `tests/test_compatibility.py`: legacy state and environment behavior.
- `tests/test_atomic_json_repository.py`: atomic write/rollback behavior.
- `tests/test_energyagent_infrastructure.py`: Dockerfile and Compose contract.
- `tests/test_health_components.py`: component health and redaction contract.

### Task 1: Verify and Protect the Existing RAG Baseline

**Files:**
- Verify: `app/rag/**`
- Verify: `app/services/**`
- Verify: `tests/test_document_splitter_runbook.py`
- Verify: `tests/test_llm_factory_and_observability.py`
- Verify: `tests/test_rag_ingestion_pipeline.py`
- Verify: `tests/test_rag_retrieval_service.py`
- Verify: `tests/test_redis_checkpoint.py`
- Verify: `tests/test_retrieval_fusion.py`
- Verify: `tests/test_sparse_index_service.py`
- Verify: `tests/test_vector_embedding_service.py`
- Verify: `tests/test_vector_store_filtering.py`

- [x] **Step 1: Record the exact pre-checkpoint worktree without secret contents**

Run:

```bash
git status --short
git diff --stat
git diff --cached --name-status
```

Expected: the existing RAG refactor and tests are visible; no secret file such as `.env` is tracked or staged.

- [x] **Step 2: Run the focused RAG and checkpoint suite**

Run:

```bash
uv run pytest \
  tests/test_document_splitter_runbook.py \
  tests/test_llm_factory_and_observability.py \
  tests/test_rag_ingestion_pipeline.py \
  tests/test_rag_retrieval_service.py \
  tests/test_redis_checkpoint.py \
  tests/test_retrieval_fusion.py \
  tests/test_sparse_index_service.py \
  tests/test_vector_embedding_service.py \
  tests/test_vector_store_filtering.py -q
```

Expected: PASS. If a test fails, invoke `systematic-debugging` and fix the existing baseline before creating the checkpoint.

- [x] **Step 3: Scan candidate checkpoint files for secret-bearing paths**

Run:

```bash
git status --short | rg -v '(^|/)(\.env|config/local\.yaml|runtime-logs|logs|volumes|uploads|graphify-out)(/|$)'
git check-ignore .env runtime-logs graphify-out
```

Expected: `.env`, runtime logs, and Graphify output are excluded from the checkpoint.

- [x] **Step 4: Create the RAG checkpoint commit**

Stage only the current RAG refactor, its compatibility shims, config/test changes, and `.gitignore`; do not stage agent state, runtime output, docs ignored by policy, or secrets.

```bash
git add .gitignore app tests
git diff --cached --check
git commit -m "feat(rag): checkpoint retrieval and indexing refactor"
```

Expected: one checkpoint commit; the working tree may still contain ignored/untracked tooling and Graphify artifacts.

### Task 2: Establish the EnergyAgent Branding Contract

**Files:**
- Create: `tests/test_energyagent_branding.py`
- Modify: `pyproject.toml:1-6`
- Modify: `app/config.py:20-24`
- Modify: `app/__init__.py:1-7`
- Modify: `app/main.py:60-76`
- Modify: `static/index.html`
- Modify: `static/app.js:1-3`
- Modify: `Makefile:1-35,95-105`
- Modify: `README.md:1-45`
- Modify: `start-windows.bat`
- Modify: `stop-windows.bat`

- [x] **Step 1: Write the failing branding test**

```python
from pathlib import Path
import tomllib

from app.config import Settings


ROOT = Path(__file__).resolve().parents[1]


def test_energyagent_is_the_canonical_product_name() -> None:
    settings = Settings(_env_file=None)
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    html = (ROOT / "static/index.html").read_text(encoding="utf-8")
    javascript = (ROOT / "static/app.js").read_text(encoding="utf-8")

    assert settings.app_name == "EnergyAgent"
    assert metadata["project"]["name"] == "energy-agent"
    assert metadata["project"]["description"].startswith("EnergyAgent")
    assert "EnergyAgent" in html
    assert "class EnergyAgentApp" in javascript


def test_product_surfaces_do_not_present_legacy_branding() -> None:
    paths = [
        ROOT / "app/__init__.py",
        ROOT / "app/main.py",
        ROOT / "static/index.html",
        ROOT / "Makefile",
        ROOT / "README.md",
        ROOT / "start-windows.bat",
        ROOT / "stop-windows.bat",
    ]
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "SuperBizAgent" not in text
    assert "oncallpy" not in text.lower()
```

- [x] **Step 2: Run the branding test to verify it fails**

Run:

```bash
uv run pytest tests/test_energyagent_branding.py -q
```

Expected: FAIL because settings, package metadata, and UI still use legacy names.

- [x] **Step 3: Apply the canonical product replacements**

Use these exact canonical values:

```toml
[project]
name = "energy-agent"
description = "EnergyAgent 智能能源运维平台 - 支持 RAG、AIOps、告警与值班协同"
```

```python
# app/config.py
app_name: str = "EnergyAgent"

# app/main.py
description="EnergyAgent 智能能源运维系统"
```

```javascript
// static/app.js
class EnergyAgentApp {
```

Rename the matching constructor call from `new SuperBizAgentApp(...)` to `new EnergyAgentApp(...)`. Replace user-visible `SuperBizAgent` and product-name `oncallpy` strings in the listed files with `EnergyAgent`. Do not replace on-call domain language or API paths.

- [x] **Step 4: Run the branding contract**

Run:

```bash
uv run pytest tests/test_energyagent_branding.py -q
```

Expected: PASS.

- [x] **Step 5: Commit branding**

```bash
git add pyproject.toml app/config.py app/__init__.py app/main.py static/index.html static/app.js Makefile README.md start-windows.bat stop-windows.bat tests/test_energyagent_branding.py
git commit -m "feat: establish EnergyAgent branding"
```

### Task 3: Add Layered Typed Configuration and Provider Profiles

**Files:**
- Create: `config/energyagent.yaml`
- Create: `config/local.example.yaml`
- Create: `tests/test_layered_config.py`
- Modify: `app/config.py`
- Modify: `.gitignore`
- Modify: `pyproject.toml`

- [x] **Step 1: Write failing layered configuration tests**

```python
import os
from pathlib import Path

from app.config import Settings


def test_yaml_layers_then_environment_take_precedence(tmp_path: Path, monkeypatch) -> None:
    defaults = tmp_path / "energyagent.yaml"
    local = tmp_path / "local.yaml"
    defaults.write_text("app_name: EnergyAgent\nport: 9900\n", encoding="utf-8")
    local.write_text("port: 9910\n", encoding="utf-8")
    monkeypatch.setenv("ENERGY_AGENT_CONFIG_FILES", os.pathsep.join((str(defaults), str(local))))
    monkeypatch.setenv("PORT", "9920")

    settings = Settings(_env_file=None)

    assert settings.app_name == "EnergyAgent"
    assert settings.port == 9920


def test_current_and_oncall_legacy_profiles_coexist(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "current-key")
    monkeypatch.setenv("ONCALL_LEGACY_DEEPSEEK_API_KEY", "legacy-key")
    monkeypatch.setenv("COZELOOP_API_TOKEN", "current-token")
    monkeypatch.setenv("ONCALL_LEGACY_COZELOOP_API_TOKEN", "legacy-token")

    settings = Settings(_env_file=None)

    assert settings.provider_profile("current").deepseek_api_key == "current-key"
    assert settings.provider_profile("oncall_legacy").deepseek_api_key == "legacy-key"
    assert settings.provider_profile("current").cozeloop_api_token == "current-token"
    assert settings.provider_profile("oncall_legacy").cozeloop_api_token == "legacy-token"
```

- [x] **Step 2: Run the layered configuration tests to verify failure**

Run:

```bash
uv run pytest tests/test_layered_config.py -q
```

Expected: FAIL because `yaml_files`, provider profiles, and legacy fields do not exist.

- [x] **Step 3: Implement deterministic YAML settings input**

Add `pyyaml>=6.0.2` to project dependencies. Add these types and helpers to `app/config.py` while preserving every existing setting field:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic.fields import FieldInfo
from pydantic_settings import PydanticBaseSettingsSource


def _merge_dict(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_files(paths: tuple[Path, ...]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            continue
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"configuration root must be a mapping: {path}")
        merged = _merge_dict(merged, loaded)
    return merged


@dataclass(frozen=True)
class ProviderProfile:
    deepseek_api_key: str
    cozeloop_api_token: str
```

Add a settings source after environment and dotenv sources so their values retain higher precedence:

```python
class LayeredYamlSettingsSource(PydanticBaseSettingsSource):
    default_paths: ClassVar[tuple[Path, ...]] = (
        Path("config/energyagent.yaml"),
        Path("config/local.yaml"),
    )

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        self._data = _load_yaml_files(self._paths())

    def _paths(self) -> tuple[Path, ...]:
        configured = os.getenv("ENERGY_AGENT_CONFIG_FILES", "")
        if not configured:
            return self.default_paths
        return tuple(Path(item) for item in configured.split(os.pathsep) if item)

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._data)


class Settings(BaseSettings):
    oncall_legacy_deepseek_api_key: str = ""
    oncall_legacy_cozeloop_api_token: str = ""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            LayeredYamlSettingsSource(settings_cls),
            file_secret_settings,
        )

    def provider_profile(self, name: str) -> ProviderProfile:
        if name == "current":
            return ProviderProfile(self.deepseek_api_key, self.cozeloop_api_token)
        if name == "oncall_legacy":
            return ProviderProfile(
                self.oncall_legacy_deepseek_api_key,
                self.oncall_legacy_cozeloop_api_token,
            )
        raise KeyError(f"unknown provider profile: {name}")
```

The tracked YAML files contain placeholders and non-secret defaults only:

```yaml
# config/energyagent.yaml
app_name: EnergyAgent
app_version: 1.2.1
host: 0.0.0.0
port: 9900
log_dir: runtime-logs
milvus_host: localhost
milvus_port: 19530
milvus_collection_name: biz
```

```yaml
# config/local.example.yaml
# Copy to config/local.yaml for machine-specific, non-secret overrides.
host: 127.0.0.1
port: 9900
```

Add `.energyagent/` and `config/local.yaml` to `.gitignore`.

- [x] **Step 4: Run configuration regression tests**

Run:

```bash
uv run pytest tests/test_layered_config.py tests/test_llm_factory_and_observability.py tests/test_redis_checkpoint.py -q
```

Expected: PASS; existing flat setting consumers continue to work.

- [x] **Step 5: Commit layered configuration**

```bash
git add app/config.py config/energyagent.yaml config/local.example.yaml .gitignore pyproject.toml uv.lock tests/test_layered_config.py
git commit -m "feat(config): add layered EnergyAgent settings"
```

### Task 4: Build and Test Secret-Safe Configuration Migration

**Files:**
- Create: `app/core/config_migration.py`
- Create: `scripts/migrate_oncall_config.py`
- Create: `tests/test_config_migration.py`

- [x] **Step 1: Write failing migration tests using fake secrets**

```python
from pathlib import Path
import json
import stat

from app.core.config_migration import MigrationRequest, migrate_configuration


def test_migration_preserves_target_keys_and_profiles_conflicts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / ".env.local").write_text(
        "DEEPSEEK_API_KEY=source-deep\nCOZELOOP_API_TOKEN=source-loop\nMEM0_BASE_URL=http://mem0\n",
        encoding="utf-8",
    )
    (source / "config.yaml").write_text("milvus:\n  address: 127.0.0.1:19530\n", encoding="utf-8")
    (target / ".env").write_text(
        "DEEPSEEK_API_KEY=target-deep\nCOZELOOP_API_TOKEN=target-loop\nDASHSCOPE_API_KEY=target-dash\n",
        encoding="utf-8",
    )

    result = migrate_configuration(
        MigrationRequest(source_root=source, target_root=target, timestamp="20260720T120000Z")
    )
    merged = (target / ".env").read_text(encoding="utf-8")

    assert "DEEPSEEK_API_KEY=target-deep" in merged
    assert "ONCALL_LEGACY_DEEPSEEK_API_KEY=source-deep" in merged
    assert "COZELOOP_API_TOKEN=target-loop" in merged
    assert "ONCALL_LEGACY_COZELOOP_API_TOKEN=source-loop" in merged
    assert "DASHSCOPE_API_KEY=target-dash" in merged
    assert "MEM0_BASE_URL=http://mem0" in merged
    assert "source-deep" not in json.dumps(result.manifest)
    assert stat.S_IMODE((target / ".env").stat().st_mode) == 0o600


def test_migration_vault_is_private_and_manifest_uses_fingerprints(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    (source / ".env.local").write_text("QWEN_API_KEY=fake-qwen\n", encoding="utf-8")
    (target / ".env").write_text("APP_NAME=EnergyAgent\n", encoding="utf-8")

    result = migrate_configuration(
        MigrationRequest(source_root=source, target_root=target, timestamp="20260720T120000Z")
    )

    assert stat.S_IMODE(result.vault_dir.stat().st_mode) == 0o700
    manifest = json.loads((result.vault_dir / "source-manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"][0]["sha256"]
    assert "fake-qwen" not in json.dumps(manifest)
```

- [x] **Step 2: Run migration tests to verify failure**

Run:

```bash
uv run pytest tests/test_config_migration.py -q
```

Expected: FAIL because the migration module does not exist.

- [x] **Step 3: Implement migration primitives**

Implement these public types and function in `app/core/config_migration.py`:

```python
@dataclass(frozen=True)
class MigrationRequest:
    source_root: Path
    target_root: Path
    timestamp: str
    seed_values: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MigrationResult:
    vault_dir: Path
    backup_path: Path
    manifest: dict[str, object]
    imported_keys: tuple[str, ...]
    conflicted_keys: tuple[str, ...]


def migrate_configuration(request: MigrationRequest) -> MigrationResult:
    """Snapshot source/target config, merge dotenv keys, and replace target .env atomically."""
```

The implementation must:

```python
CONFLICT_ALIASES = {
    "DEEPSEEK_API_KEY": "ONCALL_LEGACY_DEEPSEEK_API_KEY",
    "COZELOOP_API_TOKEN": "ONCALL_LEGACY_COZELOOP_API_TOKEN",
}
SNAPSHOT_PATHS = (
    ".env.local",
    ".env.example",
    "config.yaml",
    "manifest/docker/docker-compose.yml",
    "manifest/observability",
)
```

- Parse dotenv lines without evaluating shell expressions.
- Preserve comments and target-only values when writing the target file.
- Import equal shared keys only once.
- Write conflicting source values under `CONFLICT_ALIASES`.
- Import source-only dotenv keys unchanged.
- Add each missing `seed_values` entry to the target dotenv before replacement. The CLI seeds the currently effective `REDIS_URL` so removing the legacy code default cannot disconnect existing checkpoints.
- Parse the password from the effective `REDIS_URL`; when present, store it as local-only `REDIS_PASSWORD` for Docker Compose without printing it.
- Store nested YAML key names and fingerprints in the manifest; do not copy YAML secret values into tracked config.
- Copy source files and `target-before/.env` into `.energyagent/migration/<timestamp>`.
- Create the vault with `0700` and candidate secret files with `0600`.
- Write `.env.tmp`, fsync it, chmod it to `0600`, and replace `.env` with `os.replace`.
- Return key names and conflict names only.

The CLI is only argument parsing and sanitized reporting:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate local OnCallAgent configuration into EnergyAgent")
    parser.add_argument("--source", type=Path, default=Path("/root/code/oncall/OnCallAgent"))
    parser.add_argument("--target", type=Path, default=Path.cwd())
    args = parser.parse_args()
    settings = Settings()
    seed_values = {"REDIS_URL": settings.redis_url}
    if password := urlparse(settings.redis_url).password:
        seed_values["REDIS_PASSWORD"] = password
    result = migrate_configuration(
        MigrationRequest(
            source_root=args.source.resolve(),
            target_root=args.target.resolve(),
            timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            seed_values=seed_values,
        )
    )
    print(f"vault={result.vault_dir}")
    print(f"imported_keys={len(result.imported_keys)}")
    print(f"conflicted_keys={','.join(result.conflicted_keys)}")
    return 0
```

- [x] **Step 4: Run migration and redaction tests**

Run:

```bash
uv run pytest tests/test_config_migration.py -q
```

Expected: PASS, with no fake secret values in captured output or manifests.

- [x] **Step 5: Commit the migration tool**

```bash
git add app/core/config_migration.py tests/test_config_migration.py
git add -f scripts/migrate_oncall_config.py
git commit -m "feat(config): add secret-safe OnCall migration"
```

### Task 5: Add Legacy Compatibility and Canonical Stack Script

**Files:**
- Create: `app/core/compatibility.py`
- Create: `tests/test_compatibility.py`
- Create: `scripts/energyagent_stack.py`
- Modify: `scripts/oncallpy_stack.py`
- Modify: `tests/test_oncallpy_stack_script.py`

- [x] **Step 1: Write failing compatibility tests**

```python
from pathlib import Path

from app.core.compatibility import copy_legacy_state, resolve_compat_env


def test_new_environment_name_wins_over_legacy(monkeypatch) -> None:
    monkeypatch.setenv("ENERGY_AGENT_ALERT_STATE_FILE", "/new/alerts.json")
    monkeypatch.setenv("SUPER_BIZ_AGENT_ALERT_STATE_FILE", "/old/alerts.json")
    assert resolve_compat_env(
        "ENERGY_AGENT_ALERT_STATE_FILE",
        "SUPER_BIZ_AGENT_ALERT_STATE_FILE",
        "/default/alerts.json",
    ) == "/new/alerts.json"


def test_legacy_state_is_copied_and_verified(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.json"
    current = tmp_path / "current.json"
    legacy.write_text('{"alerts": [{"id": "a-1"}]}', encoding="utf-8")

    result = copy_legacy_state(legacy, current)

    assert result.copied is True
    assert result.sha256
    assert current.read_bytes() == legacy.read_bytes()
    assert legacy.exists()
```

- [x] **Step 2: Run compatibility tests to verify failure**

Run:

```bash
uv run pytest tests/test_compatibility.py -q
```

Expected: FAIL because compatibility helpers do not exist.

- [x] **Step 3: Implement environment and copy-first state compatibility**

```python
@dataclass(frozen=True)
class StateCopyResult:
    copied: bool
    source: Path
    destination: Path
    sha256: str


def resolve_compat_env(current: str, legacy: str, default: str) -> str:
    if value := os.getenv(current):
        return value
    if value := os.getenv(legacy):
        logger.warning("legacy environment variable in use: {}", legacy)
        return value
    return default


def copy_legacy_state(source: Path, destination: Path) -> StateCopyResult:
    if destination.exists() or not source.exists():
        return StateCopyResult(False, source, destination, "")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest != hashlib.sha256(source.read_bytes()).hexdigest():
        temporary.unlink(missing_ok=True)
        raise OSError("legacy state checksum mismatch")
    os.replace(temporary, destination)
    return StateCopyResult(True, source, destination, digest)
```

- [x] **Step 4: Create the canonical stack script and forwarding shim**

Copy the implementation of `scripts/oncallpy_stack.py` to `scripts/energyagent_stack.py`, change its module docstring and CLI descriptions to EnergyAgent, and replace the old script with this forwarding shim:

```python
#!/usr/bin/env python3
"""Compatibility entry point for the renamed EnergyAgent stack manager."""

from energyagent_stack import main


if __name__ == "__main__":
    raise SystemExit(main())
```

Update `tests/test_oncallpy_stack_script.py` to load `energyagent_stack.py` as the canonical implementation and add a subprocess-free import assertion for the shim.

- [x] **Step 5: Run compatibility and stack tests**

Run:

```bash
uv run pytest tests/test_compatibility.py tests/test_oncallpy_stack_script.py -q
```

Expected: PASS.

- [x] **Step 6: Commit compatibility helpers**

```bash
git add app/core/compatibility.py tests/test_compatibility.py tests/test_oncallpy_stack_script.py
git add -f scripts/energyagent_stack.py scripts/oncallpy_stack.py
git commit -m "feat: add EnergyAgent compatibility layer"
```

### Task 6: Add Atomic JSON Repository Foundation

**Files:**
- Create: `app/domains/__init__.py`
- Create: `app/domains/shared/__init__.py`
- Create: `app/domains/shared/atomic_json_repository.py`
- Create: `tests/test_atomic_json_repository.py`

- [x] **Step 1: Write failing atomic persistence tests**

```python
from pathlib import Path
import json

import pytest

from app.domains.shared.atomic_json_repository import AtomicJsonRepository


def test_repository_round_trips_json_and_creates_private_file(tmp_path: Path) -> None:
    path = tmp_path / "state" / "alerts.json"
    repository = AtomicJsonRepository(path, default_factory=lambda: {"alerts": []})

    repository.save({"alerts": [{"id": "a-1"}]})

    assert repository.load() == {"alerts": [{"id": "a-1"}]}
    assert path.stat().st_mode & 0o777 == 0o600


def test_failed_replace_preserves_previous_valid_state(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "state.json"
    repository = AtomicJsonRepository(path, default_factory=dict)
    repository.save({"version": 1})

    def fail_replace(_source, _destination):
        raise OSError("disk failure")

    monkeypatch.setattr("app.domains.shared.atomic_json_repository.os.replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        repository.save({"version": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"version": 1}
```

- [x] **Step 2: Run the repository tests to verify failure**

Run:

```bash
uv run pytest tests/test_atomic_json_repository.py -q
```

Expected: FAIL because the repository module does not exist.

- [x] **Step 3: Implement the atomic repository**

```python
T = TypeVar("T")


class AtomicJsonRepository(Generic[T]):
    def __init__(self, path: Path, default_factory: Callable[[], T]) -> None:
        self.path = path
        self.default_factory = default_factory

    def load(self) -> T:
        if not self.path.exists():
            return self.default_factory()
        return cast(T, json.loads(self.path.read_text(encoding="utf-8")))

    def save(self, value: T) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, self.path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
```

- [x] **Step 4: Run atomic persistence tests**

Run:

```bash
uv run pytest tests/test_atomic_json_repository.py -q
```

Expected: PASS.

- [x] **Step 5: Commit persistence foundation**

```bash
git add app/domains tests/test_atomic_json_repository.py
git commit -m "feat(storage): add atomic JSON repository"
```

### Task 7: Add EnergyAgent Application Container and Redis Compose Ownership

**Files:**
- Create: `Dockerfile`
- Create: `tests/test_energyagent_infrastructure.py`
- Modify: `vector-database.yml`

- [x] **Step 1: Write failing infrastructure tests**

```python
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_compose_owns_redis_with_healthcheck() -> None:
    compose = yaml.safe_load((ROOT / "vector-database.yml").read_text(encoding="utf-8"))
    redis = compose["services"]["redis"]
    assert redis["image"].startswith("redis:")
    assert redis["healthcheck"]["test"][-1] == "ping"
    assert "6379:6379" in redis["ports"]


def test_application_dockerfile_runs_energyagent_api() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "uv sync --frozen --no-dev" in dockerfile
    assert 'CMD ["uv", "run", "uvicorn", "app.main:app"' in dockerfile
    assert "COPY .env" not in dockerfile
```

- [x] **Step 2: Run infrastructure tests to verify failure**

Run:

```bash
uv run pytest tests/test_energyagent_infrastructure.py -q
```

Expected: FAIL because Redis and the application Dockerfile do not exist.

- [x] **Step 3: Add Redis service and application image**

Add this service to `vector-database.yml`:

```yaml
  redis:
    container_name: energyagent-redis
    image: redis:8-alpine
    command: ["redis-server", "--appendonly", "yes", "--requirepass", "${REDIS_PASSWORD:?REDIS_PASSWORD is required}"]
    ports:
      - "6379:6379"
    volumes:
      - ./volumes/redis:/data
    healthcheck:
      test: ["CMD", "redis-cli", "-a", "${REDIS_PASSWORD:?REDIS_PASSWORD is required}", "ping"]
      interval: 5s
      timeout: 3s
      retries: 20
    restart: unless-stopped
```

Create the Dockerfile:

```dockerfile
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app ./app
COPY static ./static
COPY config ./config

EXPOSE 9900
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9900"]
```

- [x] **Step 4: Validate tests and Compose syntax**

Run:

```bash
uv run pytest tests/test_energyagent_infrastructure.py -q
REDIS_PASSWORD=test-only docker compose -f vector-database.yml config --quiet
```

Expected: PASS; Compose validation emits no secret values.

- [x] **Step 5: Commit infrastructure**

```bash
git add Dockerfile vector-database.yml tests/test_energyagent_infrastructure.py
git commit -m "feat(ops): containerize EnergyAgent and own Redis"
```

### Task 8: Expand Health Reporting Without Secret Disclosure

**Files:**
- Create: `tests/test_health_components.py`
- Modify: `app/api/health.py`

- [x] **Step 1: Write failing component health tests**

```python
import json

import pytest


@pytest.mark.asyncio
async def test_health_reports_milvus_redis_and_configuration(monkeypatch) -> None:
    import app.api.health as health

    monkeypatch.setattr(health.milvus_manager, "health_check", lambda: True)

    async def redis_ok(_url: str) -> bool:
        return True

    monkeypatch.setattr(health, "check_redis", redis_ok)
    response = await health.health_check()
    payload = json.loads(response.body)

    assert response.status_code == 200
    assert payload["data"]["components"]["milvus"]["status"] == "ok"
    assert payload["data"]["components"]["redis"]["status"] == "ok"
    assert payload["data"]["components"]["configuration"]["status"] == "ok"


@pytest.mark.asyncio
async def test_health_never_returns_redis_credentials(monkeypatch) -> None:
    import app.api.health as health

    monkeypatch.setattr(health.config, "redis_url", "redis://:top-secret@localhost:6379/0")
    monkeypatch.setattr(health.milvus_manager, "health_check", lambda: True)

    async def redis_failed(_url: str) -> bool:
        return False

    monkeypatch.setattr(health, "check_redis", redis_failed)
    response = await health.health_check()

    assert b"top-secret" not in response.body
```

- [x] **Step 2: Run health tests to verify failure**

Run:

```bash
uv run pytest tests/test_health_components.py -q
```

Expected: FAIL because component health and `check_redis` do not exist.

- [x] **Step 3: Implement Redis and component health**

```python
async def check_redis(redis_url: str) -> bool:
    from redis.asyncio import Redis

    client = Redis.from_url(redis_url)
    try:
        return bool(await client.ping())
    finally:
        await client.aclose()


def component(status: str, message: str) -> dict[str, str]:
    return {"status": status, "message": message}
```

Refactor `health_check` to build:

```python
components = {
    "milvus": component("ok" if milvus_ok else "error", "available" if milvus_ok else "unavailable"),
    "redis": component("ok" if redis_ok else "error", "available" if redis_ok else "unavailable"),
    "configuration": component(
        "ok" if config.app_name == "EnergyAgent" else "warning",
        "loaded",
    ),
}
```

Return HTTP `200` only when Milvus and Redis are available; otherwise return `503`. Log exception classes and component names, not URLs or credential-bearing exception strings.

- [x] **Step 4: Run health and existing API tests**

Run:

```bash
uv run pytest tests/test_health_components.py tests/test_llm_factory_and_observability.py tests/test_redis_checkpoint.py -q
```

Expected: PASS.

- [x] **Step 5: Commit health reporting**

```bash
git add app/api/health.py tests/test_health_components.py
git commit -m "feat(health): report EnergyAgent dependencies"
```

### Task 9: Execute the Real Local Configuration Migration and Verify Foundation

**Files:**
- Modify: `app/config.py`
- Modify: `app/__init__.py`
- Modify: `app/main.py`
- Modify: `app/rag/rag_metadata_builder.py`
- Modify: `app/utils/logger.py`
- Modify: `scripts/energyagent_stack.py`
- Modify: `scripts/migrate_oncall_config.py`
- Modify: `tests/test_config_migration.py`
- Modify: `tests/test_energyagent_compatibility.py`
- Modify: `tests/test_redis_checkpoint.py`
- Modify: `tests/test_rag_ingestion_pipeline.py`
- Local only: `.env`
- Local only: `.energyagent/migration/<timestamp>/**`
- Update: `docs/superpowers/plans/2026-07-20-energyagent-foundation.md` checkboxes

- [x] **Step 1: Harden the real-migration entry point before integration**

Write failing tests proving that migration seeds are independent of the caller's
CWD and ambient settings, CLI failures print only the exception class, and the
application Redis default contains no credential. Then remove the legacy
credential-bearing default, keep its normal application fallback passwordless,
and have the migration CLI generate a fresh high-entropy `REDIS_PASSWORD` plus
matching percent-encoded local `REDIS_URL` when those target keys are missing.
The generated credential must never be printed or hard-coded. Catch migration
failures without a traceback or exception message.
Also prove that the CLI forces the active `APP_NAME` to `EnergyAgent` while
preserving any displaced target/source value under a deterministic
`ONCALL_LEGACY_APP_NAME` alias. The canonical update must be idempotent and use
the same atomic `.env` replacement path as secret migration.

Run:

```bash
uv run pytest tests/test_config_migration.py tests/test_redis_checkpoint.py -q
```

Expected: PASS without reading a real configuration file or displaying a value.

- [ ] **Step 2: Integrate the reviewed branch and confirm private destinations**

Fast-forward the actual `/root/code/oncallpy` checkout to the reviewed Foundation
head. Preserve unrelated untracked user files. Tighten the existing target
`.env` to mode `0600` without reading it; if `config/local.yaml` exists, tighten
it to `0600` as well.

Run:

```bash
git check-ignore .env config/local.yaml .energyagent/migration/probe
git ls-files .env config/local.yaml .energyagent
```

Expected: all three destinations are ignored; `git ls-files` prints nothing.

- [ ] **Step 3: Run the real migration without displaying values**

Run:

```bash
umask 077
uv run python scripts/migrate_oncall_config.py \
  --source /root/code/oncall/OnCallAgent \
  --target /root/code/oncallpy
```

Expected output contains only vault path, imported key count, and conflict key names. It never contains a value.

- [ ] **Step 4: Verify permissions, tracking status, and Settings load**

Run:

```bash
stat -c '%a %n' .env .energyagent
git check-ignore .env .energyagent
uv run python -c 'from app.config import Settings; s=Settings(); print(s.app_name, s.provider_profile("current").deepseek_api_key != "", s.provider_profile("oncall_legacy").deepseek_api_key != "")'
```

Expected: `.env` is `600`, `.energyagent` is `700`, both are ignored, and the final command prints only `EnergyAgent True True` or equivalent booleans.

- [ ] **Step 5: Run the complete test and quality suite**

Run:

```bash
uv run pytest -q
uv run ruff check app tests
git diff --check
```

Expected: PASS with no secret values in output.

- [ ] **Step 6: Update the plan checkboxes and commit verification metadata**

```bash
git add -f docs/superpowers/plans/2026-07-20-energyagent-foundation.md
git commit -m "docs: complete EnergyAgent foundation plan"
```

Expected: the plan records completed tasks; no local secret or migration-vault file is committed.
