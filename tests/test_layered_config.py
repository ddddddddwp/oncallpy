import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ENV_NAMES = (
    "ENERGY_AGENT_CONFIG_FILES",
    "APP_NAME",
    "APP_VERSION",
    "PORT",
    "DEEPSEEK_API_KEY",
    "ONCALL_LEGACY_DEEPSEEK_API_KEY",
    "COZELOOP_API_TOKEN",
    "ONCALL_LEGACY_COZELOOP_API_TOKEN",
    "MILVUS_DATABASE_NAME",
    "RAG_QUERY_REWRITE_MODE",
    "RAG_QUERY_LLM_ENABLED",
    "RAG_QUERY_LLM_MODEL",
    "RAG_QUERY_LLM_TIMEOUT_SECONDS",
    "RAG_QUERY_REWRITE_CONFIDENCE_THRESHOLD",
    "RAG_QUERY_MAX_SUBQUERIES",
    "RAG_QUERY_MAX_RETRIEVAL_ATTEMPTS",
    "RAG_QUERY_DEBUG_PREVIEW_LENGTH",
)


@pytest.fixture(autouse=True)
def _clear_config_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in CONFIG_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_default_yaml_paths_are_anchored_to_project_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    settings = Settings(_env_file=None)

    assert settings.app_name == "EnergyAgent"
    assert settings.app_version == "1.2.1"


def test_yaml_layers_then_environment_take_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = tmp_path / "energyagent.yaml"
    local = tmp_path / "local.yaml"
    defaults.write_text("app_name: EnergyAgent\nport: 9900\n", encoding="utf-8")
    local.write_text("port: 9910\n", encoding="utf-8")
    monkeypatch.setenv("ENERGY_AGENT_CONFIG_FILES", os.pathsep.join((str(defaults), str(local))))
    monkeypatch.setenv("PORT", "9920")

    settings = Settings(_env_file=None)

    assert settings.app_name == "EnergyAgent"
    assert settings.port == 9920


def test_milvus_database_name_can_be_selected_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MILVUS_DATABASE_NAME", "oncall_agent")

    settings = Settings(_env_file=None)

    assert settings.milvus_database_name == "oncall_agent"


def test_query_pipeline_settings_have_safe_defaults_and_normalize_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_QUERY_REWRITE_MODE", "  UNKNOWN ")
    monkeypatch.setenv("RAG_QUERY_LLM_ENABLED", "true")
    monkeypatch.setenv("RAG_QUERY_LLM_MODEL", "  rewrite-model  ")

    settings = Settings(_env_file=None)

    assert settings.rag_query_rewrite_mode == "off"
    assert settings.rag_query_llm_enabled is True
    assert settings.rag_query_llm_model == "rewrite-model"
    assert settings.rag_query_llm_timeout_seconds == 3.0
    assert settings.rag_query_rewrite_confidence_threshold == 0.75
    assert settings.rag_query_max_subqueries == 2
    assert settings.rag_query_max_retrieval_attempts == 8
    assert settings.rag_query_debug_preview_length == 240


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("RAG_QUERY_LLM_TIMEOUT_SECONDS", "0"),
        ("RAG_QUERY_LLM_TIMEOUT_SECONDS", "nan"),
        ("RAG_QUERY_LLM_TIMEOUT_SECONDS", "inf"),
        ("RAG_QUERY_LLM_TIMEOUT_SECONDS", "-inf"),
        ("RAG_QUERY_REWRITE_CONFIDENCE_THRESHOLD", "1.1"),
        ("RAG_QUERY_MAX_SUBQUERIES", "9"),
        ("RAG_QUERY_MAX_RETRIEVAL_ATTEMPTS", "0"),
        ("RAG_QUERY_DEBUG_PREVIEW_LENGTH", "-1"),
    ],
)
def test_query_pipeline_settings_reject_invalid_boundaries(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_query_llm_timeout_rejects_boolean_init_value() -> None:
    with pytest.raises(ValueError):
        Settings(rag_query_llm_timeout_seconds=True, _env_file=None)


def test_later_yaml_file_overrides_earlier_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = tmp_path / "energyagent.yaml"
    local = tmp_path / "local.yaml"
    defaults.write_text("app_name: Earlier\nport: 9900\n", encoding="utf-8")
    local.write_text("app_name: Later\nport: 9910\n", encoding="utf-8")
    monkeypatch.setenv("ENERGY_AGENT_CONFIG_FILES", os.pathsep.join((str(defaults), str(local))))

    settings = Settings(_env_file=None)

    assert settings.app_name == "Later"
    assert settings.port == 9910


def test_init_environment_dotenv_and_yaml_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = tmp_path / "energyagent.yaml"
    dotenv = tmp_path / ".env.test"
    defaults.write_text("port: 9901\n", encoding="utf-8")
    dotenv.write_text("PORT=9902\n", encoding="utf-8")
    monkeypatch.setenv("ENERGY_AGENT_CONFIG_FILES", str(defaults))
    monkeypatch.setenv("PORT", "9903")

    assert Settings(port=9904, _env_file=dotenv).port == 9904
    assert Settings(_env_file=dotenv).port == 9903

    monkeypatch.delenv("PORT")
    assert Settings(_env_file=dotenv).port == 9902
    assert Settings(_env_file=None).port == 9901


def test_current_and_oncall_legacy_profiles_coexist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "current-key")
    monkeypatch.setenv("ONCALL_LEGACY_DEEPSEEK_API_KEY", "legacy-key")
    monkeypatch.setenv("COZELOOP_API_TOKEN", "current-token")
    monkeypatch.setenv("ONCALL_LEGACY_COZELOOP_API_TOKEN", "legacy-token")

    settings = Settings(_env_file=None)

    assert settings.provider_profile("current").deepseek_api_key == "current-key"
    assert settings.provider_profile("oncall_legacy").deepseek_api_key == "legacy-key"
    assert settings.provider_profile("current").cozeloop_api_token == "current-token"
    assert settings.provider_profile("oncall_legacy").cozeloop_api_token == "legacy-token"


def test_unknown_provider_profile_raises_key_error() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(KeyError, match="unknown provider profile: retired"):
        settings.provider_profile("retired")


def test_yaml_root_must_be_a_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    monkeypatch.setenv("ENERGY_AGENT_CONFIG_FILES", str(invalid))

    with pytest.raises(
        ValueError,
        match=re.escape(f"configuration root must be a mapping: {invalid}"),
    ):
        Settings(_env_file=None)


def _secret_like_nonempty_fields(value: Any, prefix: str = "") -> list[str]:
    findings: list[str] = []
    if not isinstance(value, Mapping):
        return findings

    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        normalized_key = str(key).casefold()
        if any(marker in normalized_key for marker in ("api_key", "token", "secret", "password")):
            if child not in (None, ""):
                findings.append(path)
        findings.extend(_secret_like_nonempty_fields(child, path))
    return findings


@pytest.mark.parametrize(
    "relative_path",
    ("config/energyagent.yaml", "config/local.example.yaml"),
)
def test_tracked_yaml_contains_no_nonempty_secret_like_fields(relative_path: str) -> None:
    path = PROJECT_ROOT / relative_path
    assert path.is_file(), f"missing tracked configuration file: {relative_path}"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert isinstance(data, Mapping)
    assert _secret_like_nonempty_fields(data) == []
