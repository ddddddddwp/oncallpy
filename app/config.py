"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

import math
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


@dataclass(frozen=True)
class ProviderProfile:
    """一组可独立选择的提供方凭证。"""

    deepseek_api_key: str
    cozeloop_api_token: str


def _merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并映射，后出现的值优先。"""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_dict(current, value)
        else:
            merged[key] = value
    return merged


def _load_yaml_files(paths: Iterable[Path]) -> dict[str, Any]:
    """按顺序加载并合并存在的 YAML 配置文件。"""
    data: dict[str, Any] = {}
    for path in paths:
        if not path.exists():
            continue
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError(f"configuration root must be a mapping: {path}")
        data = _merge_dict(data, loaded)
    return data


class LayeredYamlSettingsSource(PydanticBaseSettingsSource):
    """缓存按顺序合并的 EnergyAgent YAML 配置。"""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        configured_files = os.getenv("ENERGY_AGENT_CONFIG_FILES")
        if configured_files:
            # Explicit relative paths intentionally resolve from the caller's CWD.
            paths = tuple(Path(path) for path in configured_files.split(os.pathsep) if path)
        else:
            config_dir = Path(__file__).resolve().parents[1] / "config"
            paths = (config_dir / "energyagent.yaml", config_dir / "local.yaml")
        self._data = _load_yaml_files(paths)

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        """返回单个字段的缓存值。"""
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        """返回缓存的 YAML 配置。"""
        return dict(self._data)


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """定义从高到低的配置优先级。"""
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            LayeredYamlSettingsSource(settings_cls),
            file_secret_settings,
        )

    # 应用配置
    app_name: str = "EnergyAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900
    log_dir: str = "runtime-logs"

    # Redis checkpoint 配置
    redis_url: str = "redis://localhost:6379/0"

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）

    # 聊天模型提供方
    chat_provider: str = "minimax"

    # MiniMax 配置（聊天模型）
    minimax_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    minimax_base_url: str = "https://api.minimax.io/v1"
    minimax_model: str = "MiniMax-M2"

    # DeepSeek 配置（聊天模型，OpenAI 兼容接口）
    deepseek_api_key: str = ""
    oncall_legacy_deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-v4-flash"

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒
    milvus_database_name: str = "default"
    milvus_collection_name: str = "biz"

    # RAG 配置
    rag_top_k: int = 3
    rag_aiops_top_k: int = 8
    rag_model: str = "MiniMax-M2"  # 默认与 MiniMax 聊天模型对齐
    rag_schema_version: int = 1
    rag_vector_metric_type: str = "L2"
    rag_vector_index_type: str = "IVF_FLAT"
    rag_vector_nlist: int = 128
    rag_vector_nprobe: int = 10
    rag_sentence_window_enabled: bool = False
    rag_sentence_window_size: int = 2
    rag_sentence_min_unit_chars: int = 4
    rag_route_retrieval_enabled: bool = True
    rag_route_top_k: int = 2
    rag_hybrid_enabled: bool = True
    rag_hybrid_profiles: str = "aiops,protocol"
    rag_sparse_index_path: str = "data/rag_sparse.sqlite"
    rag_sparse_top_k: int = 20
    rag_rrf_k: int = 60
    rag_query_rewrite_mode: str = "off"
    rag_query_llm_enabled: bool = False
    rag_query_llm_model: str = ""
    rag_query_llm_timeout_seconds: float = 3.0
    rag_query_rewrite_confidence_threshold: float = 0.75
    rag_query_max_subqueries: int = 2
    rag_query_max_retrieval_attempts: int = 8
    rag_query_debug_preview_length: int = 240

    @field_validator("rag_query_rewrite_mode", mode="before")
    @classmethod
    def _normalize_query_rewrite_mode(cls, value: object) -> str:
        normalized = value.strip().lower() if isinstance(value, str) else ""
        return normalized if normalized in {"off", "rule", "auto", "llm"} else "off"

    @field_validator("rag_query_llm_model", mode="before")
    @classmethod
    def _strip_query_llm_model(cls, value: object) -> str:
        return value.strip() if isinstance(value, str) else ""

    @field_validator("rag_query_llm_timeout_seconds", mode="before")
    @classmethod
    def _validate_query_llm_timeout(cls, value: object) -> object:
        if isinstance(value, bool):
            raise ValueError("rag_query_llm_timeout_seconds must be positive and finite")
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return value
        if not math.isfinite(numeric) or numeric <= 0:
            raise ValueError("rag_query_llm_timeout_seconds must be positive")
        return value

    @field_validator("rag_query_rewrite_confidence_threshold")
    @classmethod
    def _validate_query_confidence_threshold(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("rag_query_rewrite_confidence_threshold must be from 0 to 1")
        return value

    @field_validator("rag_query_max_subqueries")
    @classmethod
    def _validate_query_max_subqueries(cls, value: int) -> int:
        if not 0 <= value <= 8:
            raise ValueError("rag_query_max_subqueries must be from 0 to 8")
        return value

    @field_validator("rag_query_max_retrieval_attempts")
    @classmethod
    def _validate_query_max_attempts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("rag_query_max_retrieval_attempts must be positive")
        return value

    @field_validator("rag_query_debug_preview_length")
    @classmethod
    def _validate_query_preview_length(cls, value: int) -> int:
        if value < 0:
            raise ValueError("rag_query_debug_preview_length must be non-negative")
        return value

    # CozeLoop 观测配置（可选）
    cozeloop_workspace_id: str = ""
    cozeloop_api_token: str = ""
    oncall_legacy_cozeloop_api_token: str = ""
    cozeloop_api_base_url: str = ""

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://localhost:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://localhost:8004/mcp"

    @property
    def mcp_servers(self) -> dict[str, dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            },
        }

    def provider_profile(self, name: str) -> ProviderProfile:
        """获取当前或兼容旧版 OnCall 的提供方凭证。"""
        if name == "current":
            return ProviderProfile(
                deepseek_api_key=self.deepseek_api_key,
                cozeloop_api_token=self.cozeloop_api_token,
            )
        if name == "oncall_legacy":
            return ProviderProfile(
                deepseek_api_key=self.oncall_legacy_deepseek_api_key,
                cozeloop_api_token=self.oncall_legacy_cozeloop_api_token,
            )
        raise KeyError(f"unknown provider profile: {name}")


# 全局配置实例
config = Settings()
