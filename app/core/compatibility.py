"""Compatibility helpers for legacy EnergyAgent configuration and state."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StateCopyResult:
    copied: bool
    source: Path
    destination: Path
    sha256: str


def resolve_compat_env(current: str, legacy: str, default: str) -> str:
    """Prefer the current value, warning only when a legacy value is used."""
    if current:
        return current
    if legacy:
        warnings.warn(
            "SUPER_BIZ_AGENT_* environment variables are deprecated; use ENERGY_AGENT_* instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return legacy
    return default


def _sha256_file(path: Path) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("checksum target must be a regular file")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _source_is_stable(before: os.stat_result, after: os.stat_result) -> bool:
    return (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )


def _destination_status(destination: Path) -> bool:
    """Return whether the destination exists, rejecting symlinks."""
    try:
        destination_stat = destination.lstat()
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(destination_stat.st_mode):
        raise ValueError("destination must not be a symlink")
    return True


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_legacy_state(source: Path, destination: Path) -> StateCopyResult:
    """Copy a regular legacy state file once without removing the source."""
    result = StateCopyResult(False, source, destination, "")
    if _destination_status(destination):
        return result

    try:
        source_lstat = source.lstat()
    except FileNotFoundError:
        return result
    if stat.S_ISLNK(source_lstat.st_mode) or not stat.S_ISREG(source_lstat.st_mode):
        raise ValueError("legacy state source must be a regular file and not a symlink")

    destination.parent.mkdir(parents=True, exist_ok=True)
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW

    source_descriptor = os.open(source, source_flags)
    temporary_path: Path | None = None
    try:
        source_before = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(source_before.st_mode)
            or source_before.st_dev != source_lstat.st_dev
            or source_before.st_ino != source_lstat.st_ino
        ):
            raise ValueError("legacy state source must remain a regular file")

        temporary_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary_path = Path(temporary_name)
        source_digest = hashlib.sha256()
        try:
            os.fchmod(temporary_descriptor, 0o600)
            with os.fdopen(temporary_descriptor, "wb") as temporary_handle:
                temporary_descriptor = -1
                with os.fdopen(source_descriptor, "rb", closefd=False) as source_handle:
                    for chunk in iter(lambda: source_handle.read(1024 * 1024), b""):
                        temporary_handle.write(chunk)
                        source_digest.update(chunk)
                temporary_handle.flush()
                os.fsync(temporary_handle.fileno())
        finally:
            if temporary_descriptor >= 0:
                os.close(temporary_descriptor)

        source_after = os.fstat(source_descriptor)
        if not _source_is_stable(source_before, source_after):
            raise OSError("legacy state source changed while it was being copied")

        digest = source_digest.hexdigest()
        if _sha256_file(temporary_path) != digest:
            raise OSError("legacy state checksum mismatch")

        try:
            os.link(temporary_path, destination, follow_symlinks=False)
        except FileExistsError:
            return result
        temporary_path.unlink()
        temporary_path = None
        _fsync_directory(destination.parent)
        return StateCopyResult(True, source, destination, digest)
    finally:
        os.close(source_descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
