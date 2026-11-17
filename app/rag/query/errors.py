"""Bounded, redacted error summaries for query boundaries."""

from __future__ import annotations

import re

_MAX_ERROR_LENGTH = 300
_BEARER = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_SENSITIVE_SEGMENT = (
    r"access[_-]?token|client[_-]?secret|api[_-]?key|password|passwd|credential|token|secret"
)
_SENSITIVE_KEY = rf"(?:[a-z0-9]+[_-])*(?:{_SENSITIVE_SEGMENT})(?:[_-][a-z0-9]+)*"
_ASSIGNMENT = re.compile(rf"(?i)\b({_SENSITIVE_KEY})\s*[:=]\s*[^\s,;]+")
_URI_PASSWORD = re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://[^\s/:@]+:)[^\s/@]+(@)")
_DOUBLE_QUOTED_KEY = re.compile(rf'(?i)("{_SENSITIVE_KEY}")\s*:\s*"(?:\\.|[^"\\])*"')
_SINGLE_QUOTED_KEY = re.compile(rf"(?i)('{_SENSITIVE_KEY}')\s*:\s*'(?:\\.|[^'\\])*'")
_QUOTED_ASSIGNMENT = re.compile(rf"(?i)\b({_SENSITIVE_KEY})\s*[:=]\s*([\"'])(?:\\.|(?!\2).)*\2")
_AUTHORIZATION = re.compile(r"(?i)\b(?:proxy[-_])?authorization\s*[:=][^\r\n]*")
_DOUBLE_QUOTED_AUTHORIZATION = re.compile(
    r'(?i)("(?:proxy[-_])?authorization")\s*:\s*"(?:\\.|[^"\\])*"'
)
_SINGLE_QUOTED_AUTHORIZATION = re.compile(
    r"(?i)('(?:proxy[-_])?authorization')\s*:\s*'(?:\\.|[^'\\])*'"
)


def safe_error_summary(error: object) -> str:
    """Return a useful message without leaking common credential formats."""
    try:
        message = str(error).strip()
    except Exception:
        return type(error).__name__ or "Error"
    if not message:
        message = type(error).__name__ if isinstance(error, BaseException) else "Error"
    message = _DOUBLE_QUOTED_AUTHORIZATION.sub(r'\1: "***"', message)
    message = _SINGLE_QUOTED_AUTHORIZATION.sub(r"\1: '***'", message)
    message = _AUTHORIZATION.sub("Authorization=***", message)
    message = _DOUBLE_QUOTED_KEY.sub(r'\1: "***"', message)
    message = _SINGLE_QUOTED_KEY.sub(r"\1: '***'", message)
    message = _QUOTED_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=***", message)
    message = _BEARER.sub("Bearer ***", message)
    message = _ASSIGNMENT.sub(lambda match: f"{match.group(1)}=***", message)
    message = _URI_PASSWORD.sub(r"\1***\2", message)
    if len(message) > _MAX_ERROR_LENGTH:
        message = message[: _MAX_ERROR_LENGTH - 1] + "…"
    return message
