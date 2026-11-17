"""Protocol document metadata extraction helpers."""

# 协议文档字段来源多样（中英文表头、frontmatter、自由文本），本模块统一映射为稳定过滤键。
from __future__ import annotations

import re
from typing import Any


PROTOCOL_METADATA_KEYS = (
    "protocol_type",
    "协议类型",
    "通信协议",
)
# 每组 alias 按优先顺序查找，兼容历史英文 key 与中文资料中的常见列名。
REGISTER_ADDRESS_KEYS = (
    "register_address",
    "address",
    "addr",
    "寄存器地址",
    "地址",
)
REGISTER_NAME_KEYS = (
    "register_name",
    "name",
    "寄存器名称",
    "名称",
)
FUNCTION_CODE_KEYS = (
    "function_code",
    "fc",
    "功能码",
)
DATA_TYPE_KEYS = (
    "data_type",
    "type",
    "数据类型",
    "类型",
)
UNIT_KEYS = ("unit", "单位")
SCALE_KEYS = ("scale", "倍率", "系数")
READ_WRITE_KEYS = ("read_write", "rw", "access", "读写", "权限")
DESCRIPTION_KEYS = ("description", "desc", "说明", "描述")

HEADER_KEY_MAP = {
    # Markdown 表头先映射到 canonical key，后续索引和查询只依赖这一套字段名。
    **{key: "register_address" for key in REGISTER_ADDRESS_KEYS},
    **{key: "register_name" for key in REGISTER_NAME_KEYS},
    **{key: "function_code" for key in FUNCTION_CODE_KEYS},
    **{key: "data_type" for key in DATA_TYPE_KEYS},
    **{key: "unit" for key in UNIT_KEYS},
    **{key: "scale" for key in SCALE_KEYS},
    **{key: "read_write" for key in READ_WRITE_KEYS},
    **{key: "description" for key in DESCRIPTION_KEYS},
}


def normalize_protocol_metadata(
    metadata: dict[str, Any],
    *,
    file_name: str = "",
    content: str = "",
) -> None:
    """Normalize protocol/register metadata onto filterable keys."""
    # 文件名和标题比任意 chunk 正文更稳定，先用于建立文档级协议上下文。
    stable_context = " ".join(
        str(value)
        for value in (
            file_name,
            metadata.get("h1", ""),
            metadata.get("h2", ""),
            metadata.get("h3", ""),
            metadata.get("_file_name", ""),
        )
        if value
    )
    context = " ".join(value for value in (stable_context, content) if value)

    # 合并优先级：稳定上下文 < 当前正文 < 显式 metadata；越明确的来源最后覆盖。
    inferred = infer_protocol_metadata_from_text(stable_context)
    inferred.update(infer_protocol_metadata_from_text(content))
    explicit = _metadata_aliases(metadata)
    inferred.update({key: value for key, value in explicit.items() if value})

    # 只补缺失字段，不覆盖调用方已经提供的 canonical metadata。
    for key, value in inferred.items():
        if value and not metadata.get(key):
            metadata[key] = value

    # 检出协议特征时仅把普通 document 升级为 protocol，不覆盖更具体的已有类型。
    if _looks_like_protocol(metadata, context):
        document_type = str(metadata.get("document_type", "")).strip()
        if not document_type or document_type == "document":
            metadata["document_type"] = "protocol"


def infer_protocol_metadata_from_text(text: str) -> dict[str, str]:
    """Infer protocol filter fields from free text or a user query."""
    # 该函数也用于用户 query，因此只做确定性规则抽取，不依赖外部模型。
    result: dict[str, str] = {}
    protocol_type = _infer_protocol_type(text)
    if protocol_type:
        result["protocol_type"] = protocol_type

    register_address = extract_register_address(text)
    if register_address:
        result["register_address"] = register_address

    function_code = _extract_function_code(text)
    if function_code:
        result["function_code"] = function_code

    data_type = _extract_data_type(text)
    if data_type:
        result["data_type"] = data_type

    # 结构相似的“标签: 值”字段共用一条抽取与规范化流水线。
    for key, pattern in (
        ("register_name", r"(?:寄存器名称|名称|name)\s*[:：]\s*([^\n\r|]+)"),
        ("unit", r"(?:单位|unit)\s*[:：]\s*([^\n\r|]+)"),
        ("scale", r"(?:倍率|系数|scale)\s*[:：]\s*([^\n\r|]+)"),
        ("read_write", r"(?:读写|权限|access|rw)\s*[:：]\s*([^\n\r|]+)"),
        ("description", r"(?:说明|描述|description|desc)\s*[:：]\s*([^\n\r|]+)"),
    ):
        value = _extract_labeled_value(text, pattern)
        normalized = _normalize_protocol_value(key, value)
        if normalized:
            result[key] = normalized

    return result


def extract_markdown_protocol_table_rows(content: str) -> list[tuple[str, dict[str, Any]]]:
    """Extract one protocol row document per Markdown register table row."""
    rows: list[tuple[str, dict[str, Any]]] = []
    lines = content.splitlines()
    h1 = ""
    h2 = ""
    index = 0

    # 单次线性扫描同时跟踪最近的 h1/h2，并发现“表头 + 分隔行”结构。
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("# "):
            h1 = stripped[2:].strip()
        elif stripped.startswith("## "):
            h2 = stripped[3:].strip()

        # Markdown 表必须紧跟由 --- 组成的分隔行；普通带竖线正文不会被当作表格。
        if (
            stripped.startswith("|")
            and index + 1 < len(lines)
            and _is_markdown_separator(lines[index + 1])
        ):
            # 先保留原始表头用于兼容映射，再把每列变为 canonical metadata key。
            headers = [_clean_cell(cell) for cell in _split_table_row(stripped)]
            canonical_headers = [_canonical_header(header) for header in headers]
            row_index = index + 2
            # 连续的 | 行属于当前表格；zip(strict=False) 容忍资料中缺列或多列的行。
            while row_index < len(lines) and lines[row_index].strip().startswith("|"):
                values = [_clean_cell(cell) for cell in _split_table_row(lines[row_index])]
                metadata: dict[str, Any] = {
                    canonical: value
                    for canonical, value in zip(canonical_headers, values, strict=False)
                    if canonical and value
                }
                # 至少包含地址或功能码才建立独立协议 chunk，避免把普通 Markdown 表全部重复索引。
                if _is_protocol_row(metadata):
                    if h1:
                        metadata["h1"] = h1
                    if h2:
                        metadata["h2"] = h2
                    metadata["document_type"] = "protocol"
                    normalize_protocol_metadata(metadata, content=" ".join(values))
                    rows.append((_protocol_row_text(metadata), metadata))
                row_index += 1
            # 当前表已由内层循环完整消费，外层直接跳到下一段非表格内容。
            index = row_index
            continue

        index += 1

    return rows


def _metadata_aliases(metadata: dict[str, Any]) -> dict[str, str]:
    # 从多个别名中选第一个非空值，再按目标字段做格式标准化。
    aliases = {
        "protocol_type": _first_metadata_value(metadata, PROTOCOL_METADATA_KEYS),
        "register_address": _first_metadata_value(metadata, REGISTER_ADDRESS_KEYS),
        "register_name": _first_metadata_value(metadata, REGISTER_NAME_KEYS),
        "function_code": _first_metadata_value(metadata, FUNCTION_CODE_KEYS),
        "data_type": _first_metadata_value(metadata, DATA_TYPE_KEYS),
        "unit": _first_metadata_value(metadata, UNIT_KEYS),
        "scale": _first_metadata_value(metadata, SCALE_KEYS),
        "read_write": _first_metadata_value(metadata, READ_WRITE_KEYS),
        "description": _first_metadata_value(metadata, DESCRIPTION_KEYS),
    }
    result: dict[str, str] = {}
    for key, value in aliases.items():
        normalized = _normalize_protocol_value(key, value)
        if normalized:
            result[key] = normalized
    return result


def _looks_like_protocol(metadata: dict[str, Any], context: str) -> bool:
    if any(metadata.get(key) for key in ("register_address", "protocol_type", "function_code")):
        return True
    normalized = context.lower()
    return any(keyword in normalized for keyword in ("寄存器", "register", "modbus", "协议"))


def extract_register_address(text: Any) -> str:
    value = str(text or "")
    # 十六进制地址特征最明确，优先于可能与普通数字混淆的十进制地址。
    hex_match = re.search(r"\b0x[0-9a-fA-F]+\b", value)
    if hex_match:
        return hex_match.group(0).lower()

    # 十进制仅接受带地址标签且为 4~6 位的数字，降低误把年份/数量当寄存器的概率。
    labeled_match = re.search(r"(?:寄存器地址|地址|addr(?:ess)?)\s*[:：]?\s*(\d{4,6})", value, re.I)
    if labeled_match:
        return labeled_match.group(1)

    return ""


def _extract_function_code(text: Any) -> str:
    value = str(text or "")
    match = re.search(r"(?:功能码|function\s*code|fc)\s*[:：]?\s*(?:0x)?([0-9a-fA-F]{1,2})", value, re.I)
    if not match:
        return ""
    # 去掉可选 0x 后补齐两位小写十六进制，保证 filter 比较格式一致。
    return match.group(1).zfill(2).lower()


def _extract_data_type(text: Any) -> str:
    value = str(text or "")
    match = re.search(r"\b(u?int(?:8|16|32|64)|float(?:32|64)?|double|bool|bit)\b", value, re.I)
    return match.group(1).lower() if match else ""


def _extract_labeled_value(text: Any, pattern: str) -> str:
    match = re.search(pattern, str(text or ""), flags=re.I)
    return match.group(1).strip() if match else ""


def _infer_protocol_type(text: Any) -> str:
    normalized = str(text or "").lower()
    if "modbus" in normalized:
        return "modbus"
    if "canopen" in normalized:
        return "canopen"
    if re.search(r"\bcan\b", normalized):
        return "can"
    return ""


def _first_metadata_value(metadata: dict[str, Any], keys: tuple[str, ...]) -> Any:
    # keys 的声明顺序就是优先级，canonical 英文键通常位于别名列表首位。
    for key in keys:
        value = metadata.get(key)
        if str(value or "").strip():
            return value
    return None


def _normalize_protocol_value(key: str, value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    # 不同字段有不同的比较语义：协议/类型小写、权限大写、功能码定长。
    if key == "protocol_type":
        return _infer_protocol_type(raw) or raw.lower()
    if key == "register_address":
        return extract_register_address(raw) or raw.lower()
    if key == "function_code":
        return raw.replace("0x", "").replace("0X", "").zfill(2).lower()
    if key == "data_type":
        return raw.lower()
    if key == "read_write":
        return raw.upper()
    return raw


def _canonical_header(header: str) -> str:
    # 同时尝试原文与空格转下划线后的 lowercase 形式，覆盖中英文表头习惯。
    normalized = header.strip().lower().replace(" ", "_")
    return HEADER_KEY_MAP.get(header) or HEADER_KEY_MAP.get(normalized, "")


def _is_markdown_separator(line: str) -> bool:
    cells = _split_table_row(line.strip())
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return stripped.split("|") if stripped else []


def _clean_cell(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _is_protocol_row(metadata: dict[str, Any]) -> bool:
    return bool(metadata.get("register_address") or metadata.get("function_code"))


def _protocol_row_text(metadata: dict[str, Any]) -> str:
    # 行级 chunk 使用固定中文标签顺序，既适合精确检索，也便于最终上下文阅读。
    labels = (
        ("register_address", "地址"),
        ("register_name", "名称"),
        ("function_code", "功能码"),
        ("data_type", "数据类型"),
        ("unit", "单位"),
        ("scale", "倍率"),
        ("read_write", "读写"),
        ("description", "说明"),
    )
    parts = [f"{label}: {metadata[key]}" for key, label in labels if metadata.get(key)]
    return "\n".join(parts)
