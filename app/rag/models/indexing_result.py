"""Summary returned by directory indexing."""

# 目录索引会逐文件隔离失败，因此结果对象同时保存总计数与每个失败文件的原因。
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class IndexingResult:
    """Indexing counters and failures with the legacy serialization contract."""

    # success 表示整批没有失败；success_count 与 skipped_count 分开，便于判断实际写入量。
    success: bool = False
    directory_path: str = ""
    total_files: int = 0
    success_count: int = 0
    skipped_count: int = 0
    fail_count: int = 0
    deleted_count: int = 0
    inserted_chunk_count: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    error_message: str = ""
    # default_factory 确保每次索引都有独立的失败字典，避免 dataclass 实例间共享可变对象。
    failed_files: dict[str, str] = field(default_factory=dict)

    def increment_success_count(self) -> None:
        """Increase the number of indexed files."""
        self.success_count += 1

    def increment_skipped_count(self) -> None:
        """Increase the number of unchanged files."""
        self.skipped_count += 1

    def increment_fail_count(self) -> None:
        """Increase the number of failed files."""
        self.fail_count += 1

    def add_failed_file(self, file_path: str, error: str) -> None:
        """Record a failed file and its error."""
        self.failed_files[file_path] = error

    def get_duration_ms(self) -> int:
        """Return elapsed indexing time in milliseconds."""
        # 未结束或尚未开始时没有完整区间，保持旧 API 的 0 毫秒语义。
        if self.start_time and self.end_time:
            return int((self.end_time - self.start_time).total_seconds() * 1000)
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize using the existing API response keys."""
        # 显式列出键名，维持已有 HTTP/调用方依赖的序列化契约。
        return {
            "success": self.success,
            "directory_path": self.directory_path,
            "total_files": self.total_files,
            "success_count": self.success_count,
            "skipped_count": self.skipped_count,
            "fail_count": self.fail_count,
            "deleted_count": self.deleted_count,
            "inserted_chunk_count": self.inserted_chunk_count,
            "duration_ms": self.get_duration_ms(),
            "error_message": self.error_message,
            "failed_files": self.failed_files,
        }
