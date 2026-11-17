"""Atomic JSON persistence for small domain state documents."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar, cast

T = TypeVar("T")


class AtomicJsonRepository(Generic[T]):
    """Load and atomically replace a single JSON document."""

    def __init__(self, path: Path, default_factory: Callable[[], T]) -> None:
        self._path = path
        self._default_factory = default_factory

    def load(self) -> T:
        try:
            serialized = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return self._default_factory()
        return cast(T, json.loads(serialized))

    def save(self, value: T) -> None:
        serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            dir=self._path.parent,
        )
        temporary_path: Path | None = Path(temporary_name)
        try:
            os.fchmod(temporary_descriptor, 0o600)
            with os.fdopen(
                temporary_descriptor,
                "w",
                encoding="utf-8",
                newline="\n",
            ) as temporary_file:
                temporary_descriptor = -1
                temporary_file.write(serialized)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())

            os.replace(temporary_path, self._path)
            temporary_path = None
            self._fsync_parent()
        finally:
            if temporary_descriptor >= 0:
                os.close(temporary_descriptor)
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _fsync_parent(self) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        descriptor = os.open(self._path.parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
