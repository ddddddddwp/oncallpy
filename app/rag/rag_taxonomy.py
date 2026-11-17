"""Device taxonomy helpers for RAG metadata and retrieval."""

# taxonomy 将同一设备/厂商的多种写法收敛为 canonical 值，保证索引与查询 filter 能相等匹配。
from __future__ import annotations

from typing import Any

DEVICE_TYPE_METADATA_KEYS = (
    "device_type",
    "equipment_type",
    "asset_type",
    "设备类型",
    "设备类别",
)
MANUFACTURER_METADATA_KEYS = (
    "manufacturer",
    "vendor",
    "vendor_name",
    "maker",
    "厂家",
    "厂商",
)

DEVICE_TYPE_ALIASES = {
    # 字典 key 是最终写入 metadata 的稳定值，tuple 收录中英文及历史命名。
    "inverter": ("inverter", "pv_inverter", "photovoltaic_inverter", "逆变器"),
    "meter": ("meter", "ammeter", "electric_meter", "电表", "计量表"),
    "charger_gun": ("charger_gun", "charging_gun", "charge_gun", "充电枪"),
    "battery": ("battery", "battery_pack", "ess_battery", "电池", "电池簇"),
}
MANUFACTURER_ALIASES = {
    "invt": ("invt", "英威腾", "英威腾电气", "英威腾逆变器"),
    "huawei": ("huawei", "hua_wei", "华为"),
    "sungrow": ("sungrow", "阳光", "阳光电源"),
    "growatt": ("growatt", "古瑞瓦特"),
    "goodwe": ("goodwe", "固德威"),
    "ginlong": ("ginlong", "solis", "锦浪"),
    "tbea": ("tbea", "特变电工"),
}


def normalize_taxonomy_metadata(metadata: dict[str, Any]) -> None:
    """Normalize device taxonomy aliases onto filterable metadata keys."""
    # 允许输入使用任一 alias key，但输出统一写回 device_type/manufacturer。
    device_type = _first_metadata_value(metadata, DEVICE_TYPE_METADATA_KEYS)
    if device_type:
        metadata["device_type"] = normalize_device_type(device_type)

    manufacturer = _first_metadata_value(metadata, MANUFACTURER_METADATA_KEYS)
    if manufacturer:
        metadata["manufacturer"] = normalize_manufacturer(manufacturer)


def infer_taxonomy_from_text(text: str) -> dict[str, str]:
    """Infer device taxonomy filters from a user query."""
    # 查询推断不接受未知值，避免把任意词误变成必须命中的过滤条件。
    result: dict[str, str] = {}
    device_type = _match_alias(text, DEVICE_TYPE_ALIASES, allow_unknown=False)
    if device_type:
        result["device_type"] = device_type

    manufacturer = _match_alias(text, MANUFACTURER_ALIASES, allow_unknown=False)
    if manufacturer:
        result["manufacturer"] = manufacturer

    return result


def infer_taxonomy_from_document(
    file_name: str,
    metadata: dict[str, Any] | None = None,
    content: str = "",
) -> dict[str, str]:
    """Infer taxonomy from stable document context, not only chunk text."""
    # 文档推断组合文件名、结构 metadata 与有限正文，比仅看单个短 chunk 更稳定。
    metadata = metadata or {}
    context_parts = [file_name]
    for key in (
        "manufacturer",
        "device_type",
        "model",
        "h1",
        "h2",
        "h3",
        "heading_path",
        "route_key",
        "alert_name",
        "_file_name",
    ):
        value = metadata.get(key)
        if value:
            context_parts.append(str(value))
    # 正文限制 1000 字，兼顾早期关键信息与规则扫描成本/误匹配范围。
    if content:
        context_parts.append(content[:1000])
    return infer_taxonomy_from_text(" ".join(context_parts))


def normalize_device_type(value: Any) -> str:
    # 写入侧允许保留未收录的规范化值，避免新设备类型被无声丢弃。
    return _match_alias(value, DEVICE_TYPE_ALIASES, allow_unknown=True)


def normalize_manufacturer(value: Any) -> str:
    return _match_alias(value, MANUFACTURER_ALIASES, allow_unknown=True)


def route_key_with_taxonomy(metadata: dict[str, Any], base_key: str) -> str:
    """Prefix a route key with device taxonomy when available."""
    base = str(base_key).strip()
    # 前缀顺序固定为设备类型/厂商，使 route_key 可读且可重复生成。
    prefix_parts = [
        str(metadata[key]).strip() for key in ("device_type", "manufacturer") if metadata.get(key)
    ]
    if not prefix_parts:
        return base

    prefix = "/".join(prefix_parts)
    # 已含相同前缀时保持幂等，避免多次标准化产生 inverter/huawei/inverter/huawei/...。
    if base.startswith(f"{prefix}/") or base == prefix:
        return base
    if not base:
        return prefix
    return f"{prefix}/{base}"


def _first_metadata_value(metadata: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = metadata.get(key)
        if str(value or "").strip():
            return value
    return None


def _match_alias(
    value: Any,
    aliases: dict[str, tuple[str, ...]],
    *,
    allow_unknown: bool,
) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        return ""

    # 先比较 canonical，再检查 alias 的完全匹配或包含关系，最终返回 canonical key。
    for canonical, alias_values in aliases.items():
        if normalized == canonical:
            return canonical
        for alias in alias_values:
            normalized_alias = _normalize_text(alias)
            if normalized == normalized_alias or normalized_alias in normalized:
                return canonical

    # 写入标准化可保留未知 taxonomy；查询推断则返回空，二者风险策略不同。
    return normalized if allow_unknown else ""


def _normalize_text(value: Any) -> str:
    # 消除大小写与常见分隔符差异，让 hua-wei、hua wei 等形式可落到同一比较空间。
    return str(value).strip().lower().replace(" ", "").replace("-", "_").replace("/", "_")
