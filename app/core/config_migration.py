"""Secret-safe, one-way migration of legacy OnCall configuration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from urllib.parse import quote, unquote_to_bytes, urlparse

import yaml
from dotenv.parser import parse_stream

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

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UNSAFE_COMPOSE_DOLLAR_TOKEN_RE = re.compile(r"\$(?=\$|\{|[A-Za-z_])")
_INVALID_PERCENT_ESCAPE_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_SAFE_TIMESTAMP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_UNQUOTED_VALUE_RE = re.compile(r"^[A-Za-z0-9_./:+-]*$")
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_CONFLICT_ALIASES_CASEFOLD = {key.casefold(): value for key, value in CONFLICT_ALIASES.items()}


@dataclass(frozen=True)
class MigrationRequest:
    source_root: Path
    target_root: Path
    timestamp: str
    seed_values: Mapping[str, str] = field(default_factory=dict)
    canonical_values: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MigrationResult:
    vault_dir: Path
    backup_path: Path
    manifest: dict[str, object]
    imported_keys: tuple[str, ...]
    conflicted_keys: tuple[str, ...]


def _require_directory(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")


def _validate_timestamp(timestamp: str) -> None:
    if timestamp in {"", ".", ".."} or _SAFE_TIMESTAMP_RE.fullmatch(timestamp) is None:
        raise ValueError("timestamp must be a safe filename component")


def _validate_no_symlinks(path: Path, resolved_root: Path | None = None) -> None:
    if path.is_symlink():
        raise ValueError(f"snapshot path contains a symbolic link: {path}")
    if not path.exists():
        return
    if resolved_root is not None and not path.resolve(strict=True).is_relative_to(resolved_root):
        raise ValueError(f"snapshot path escapes source root: {path}")
    if path.is_dir():
        for child in path.iterdir():
            _validate_no_symlinks(child, resolved_root)


def _validate_source_snapshot_path(source_root: Path, relative_path: Path) -> None:
    resolved_root = source_root.resolve(strict=True)
    candidate = source_root
    for component in relative_path.parts:
        candidate /= component
        if candidate.is_symlink():
            raise ValueError(f"snapshot path contains a symbolic link: {candidate}")
        if not candidate.exists():
            return
        if not candidate.resolve(strict=True).is_relative_to(resolved_root):
            raise ValueError(f"snapshot path escapes source root: {candidate}")
    _validate_no_symlinks(candidate, resolved_root)


def _validate_inputs(request: MigrationRequest) -> None:
    _validate_timestamp(request.timestamp)
    _require_directory(request.source_root, "source root")
    _require_directory(request.target_root, "target root")

    for relative_path in SNAPSHOT_PATHS:
        _validate_source_snapshot_path(request.source_root, Path(relative_path))
    _validate_no_symlinks(request.target_root / ".env")

    private_root = request.target_root / ".energyagent"
    migration_root = private_root / "migration"
    for path in (private_root, migration_root):
        if path.is_symlink():
            raise ValueError(f"migration path must not be a symbolic link: {path}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_private_directory(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"private directory must not be a symbolic link: {path}")
    if path.exists():
        if not path.is_dir():
            raise NotADirectoryError(f"private directory path is not a directory: {path}")
    else:
        if not path.parent.exists():
            _ensure_private_directory(path.parent)
        path.mkdir(mode=0o700)
        _fsync_directory(path.parent)
    path.chmod(0o700)


def _open_readonly_no_follow(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags)


def _open_source_root(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(path, flags)
    except OSError as error:
        if error.errno in {getattr(os, "ELOOP", 40), getattr(os, "ENOTDIR", 20)}:
            raise ValueError(f"source root contains a symbolic link: {path}") from error
        raise


def _open_source_relative(root_descriptor: int, relative_path: Path) -> int | None:
    current_descriptor = os.dup(root_descriptor)
    try:
        for index, component in enumerate(relative_path.parts):
            try:
                component_stat = os.stat(
                    component,
                    dir_fd=current_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                os.close(current_descriptor)
                return None
            if stat.S_ISLNK(component_stat.st_mode):
                raise ValueError(f"snapshot path contains a symbolic link: {relative_path}")

            flags = os.O_RDONLY
            if index < len(relative_path.parts) - 1 and hasattr(os, "O_DIRECTORY"):
                flags |= os.O_DIRECTORY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                next_descriptor = os.open(component, flags, dir_fd=current_descriptor)
            except OSError as error:
                if error.errno in {getattr(os, "ELOOP", 40), getattr(os, "ENOTDIR", 20)}:
                    raise ValueError(
                        f"snapshot path contains a symbolic link: {relative_path}"
                    ) from error
                raise
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        return current_descriptor
    except BaseException:
        os.close(current_descriptor)
        raise


def _open_source_child(directory_descriptor: int, name: str, display_path: Path) -> int:
    child_stat = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if stat.S_ISLNK(child_stat.st_mode):
        raise ValueError(f"snapshot path contains a symbolic link: {display_path / name}")
    flags = os.O_RDONLY
    if stat.S_ISDIR(child_stat.st_mode) and hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        return os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as error:
        if error.errno in {getattr(os, "ELOOP", 40), getattr(os, "ENOTDIR", 20)}:
            raise ValueError(
                f"snapshot path contains a symbolic link: {display_path / name}"
            ) from error
        raise


def _read_source_descriptor(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    return b"".join(chunks)


def _read_private_source(path: Path) -> bytes:
    descriptor = _open_readonly_no_follow(path)
    with os.fdopen(descriptor, "rb") as stream:
        return stream.read()


def _write_private_file(path: Path, content: bytes) -> None:
    _ensure_private_directory(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _yaml_key_paths(value: object, prefix: str = "") -> set[str]:
    paths: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            paths.add(path)
            paths.update(_yaml_key_paths(child, path))
    elif isinstance(value, list):
        for child in value:
            paths.update(_yaml_key_paths(child, prefix))
    return paths


def _snapshot_metadata(relative_path: str, destination: Path, content: bytes) -> dict[str, object]:
    suffix = destination.suffix.casefold()
    kind = "yaml" if suffix in _YAML_SUFFIXES else "file"
    metadata: dict[str, object] = {
        "path": relative_path,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "mode": "0600",
        "kind": kind,
    }
    if kind == "yaml":
        try:
            parsed = yaml.safe_load(content.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError):
            parsed = None
        metadata["yaml_keys"] = sorted(_yaml_key_paths(parsed))
    return metadata


def _copy_snapshot_path(
    source_descriptor: int,
    source_display_path: Path,
    destination: Path,
    vault_relative_path: Path,
    entries: list[dict[str, object]],
) -> None:
    source_stat = os.fstat(source_descriptor)
    if stat.S_ISDIR(source_stat.st_mode):
        _ensure_private_directory(destination)
        for child_name in sorted(os.listdir(source_descriptor)):
            child_descriptor = _open_source_child(
                source_descriptor,
                child_name,
                source_display_path,
            )
            try:
                _copy_snapshot_path(
                    child_descriptor,
                    source_display_path / child_name,
                    destination / child_name,
                    vault_relative_path / child_name,
                    entries,
                )
            finally:
                os.close(child_descriptor)
        return
    if not stat.S_ISREG(source_stat.st_mode):
        raise ValueError(f"snapshot path is not a regular file or directory: {source_display_path}")

    content = _read_source_descriptor(source_descriptor)
    _write_private_file(destination, content)
    entries.append(_snapshot_metadata(vault_relative_path.as_posix(), destination, content))


def _create_vault(request: MigrationRequest) -> tuple[Path, Path, dict[str, object]]:
    private_root = request.target_root / ".energyagent"
    migration_root = private_root / "migration"
    vault_dir = migration_root / request.timestamp
    _ensure_private_directory(private_root)
    _ensure_private_directory(migration_root)
    if vault_dir.exists() or vault_dir.is_symlink():
        raise FileExistsError(f"migration vault already exists: {vault_dir}")
    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=migration_root))
    staging_dir.chmod(0o700)
    published = False
    try:
        entries: list[dict[str, object]] = []
        source_root_descriptor = _open_source_root(request.source_root)
        try:
            for relative_path_text in SNAPSHOT_PATHS:
                relative_path = Path(relative_path_text)
                source_descriptor = _open_source_relative(source_root_descriptor, relative_path)
                if source_descriptor is None:
                    continue
                try:
                    _copy_snapshot_path(
                        source_descriptor,
                        request.source_root / relative_path,
                        staging_dir / "source" / relative_path,
                        Path("source") / relative_path,
                        entries,
                    )
                finally:
                    os.close(source_descriptor)
        finally:
            os.close(source_root_descriptor)

        staging_backup_path = staging_dir / "target-before" / ".env"
        target_env = request.target_root / ".env"
        if target_env.exists():
            target_descriptor = _open_readonly_no_follow(target_env)
            try:
                _copy_snapshot_path(
                    target_descriptor,
                    target_env,
                    staging_backup_path,
                    Path("target-before") / ".env",
                    entries,
                )
            finally:
                os.close(target_descriptor)

        entries.sort(key=lambda entry: str(entry["path"]))
        manifest: dict[str, object] = {"files": entries}
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _write_private_file(staging_dir / "manifest.json", manifest_bytes)
        for current_root, _directories, _files in os.walk(staging_dir, topdown=False):
            _fsync_directory(Path(current_root))
        os.replace(staging_dir, vault_dir)
        published = True
        _fsync_directory(migration_root)
    except BaseException:
        if not published and staging_dir.exists():
            shutil.rmtree(staging_dir)
            _fsync_directory(migration_root)
        raise

    return vault_dir, vault_dir / "target-before" / ".env", manifest


def _dotenv_values(text: str) -> dict[str, tuple[str, str]]:
    values: dict[str, tuple[str, str]] = {}
    for binding in parse_stream(StringIO(text)):
        if binding.error or binding.key is None or binding.value is None:
            continue
        folded_key = binding.key.casefold()
        values.pop(folded_key, None)
        values[folded_key] = (binding.key, binding.value)
    return values


def _serialize_dotenv_value(value: str) -> str:
    if _SAFE_UNQUOTED_VALUE_RE.fullmatch(value) is not None:
        return value
    if "\\'" in value:
        if _UNSAFE_COMPOSE_DOLLAR_TOKEN_RE.search(value) is not None:
            raise ValueError("dotenv value cannot be represented safely for Python and Compose")
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    escaped = value.replace("'", "\\'")
    return f"'{escaped}'"


def redis_password_from_url(redis_url: str) -> str | None:
    if any(character in redis_url for character in ("\x00", "\r", "\n")):
        raise ValueError("Redis password URL contains an unsafe control character")

    try:
        encoded_password = urlparse(redis_url).password
    except ValueError as error:
        raise ValueError("Redis password URL is invalid") from error
    if encoded_password is None:
        return None
    if _INVALID_PERCENT_ESCAPE_RE.search(encoded_password) is not None:
        raise ValueError("Redis password contains an invalid percent escape")
    try:
        password = unquote_to_bytes(encoded_password).decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Redis password is not valid UTF-8") from error
    if any(character in password for character in ("\x00", "\r", "\n")):
        raise ValueError("Redis password contains an unsafe control character")
    return password


def _append_assignment(text: str, key: str, value: str) -> str:
    if text and not text.endswith(("\n", "\r")):
        text += "\n"
    return f"{text}{key}={_serialize_dotenv_value(value)}\n"


def _available_alias(
    base_alias: str,
    source_value: str,
    current_values: Mapping[str, tuple[str, str]],
) -> tuple[str, bool]:
    candidate = base_alias
    suffix = 2
    while candidate.casefold() in current_values:
        if current_values[candidate.casefold()][1] == source_value:
            return candidate, False
        candidate = f"{base_alias}_{suffix}"
        suffix += 1
    return candidate, True


def _derived_seed_values(seed_values: Mapping[str, str]) -> dict[str, str]:
    seeds: dict[str, str] = {}
    for key, value in seed_values.items():
        if _KEY_RE.fullmatch(key) is None:
            raise ValueError(f"invalid seed key: {key}")
        if not isinstance(value, str):
            raise TypeError(f"seed value for {key} must be a string")
        if "\n" in value or "\r" in value:
            raise ValueError(f"seed value for {key} must be one line")
        seeds[key] = value

    folded_keys = {key.casefold(): key for key in seeds}
    redis_url_key = folded_keys.get("redis_url")
    if redis_url_key is not None and "redis_password" not in folded_keys:
        password = redis_password_from_url(seeds[redis_url_key])
        if password is not None:
            seeds["REDIS_PASSWORD"] = password
    return seeds


def _redis_url_with_password(redis_url: str, password: str) -> str:
    try:
        parsed = urlparse(redis_url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise ValueError("Redis seed URL is invalid") from error
    if hostname is None:
        raise ValueError("Redis seed URL must include a hostname")

    host = f"[{hostname}]" if ":" in hostname else hostname
    port_suffix = f":{port}" if port is not None else ""
    encoded_password = quote(password, safe="")
    netloc = f":{encoded_password}@{host}{port_suffix}"
    return parsed._replace(netloc=netloc).geturl()


def _reconciled_seed_values(
    seed_values: Mapping[str, str],
    current_values: Mapping[str, tuple[str, str]],
) -> dict[str, str]:
    seeds = _derived_seed_values(seed_values)
    replace_existing_keys: set[str] = set()
    redis_password_key = {key.casefold(): key for key in seeds}.get("redis_password")
    redis_url_key = {key.casefold(): key for key in seeds}.get("redis_url")
    current_password = current_values.get("redis_password")
    current_url = current_values.get("redis_url")
    current_url_value = current_url[1] if current_url is not None else None
    needs_legacy_url_alias = False

    if current_password is not None:
        if redis_password_key is None:
            redis_password_key = "REDIS_PASSWORD"
        seeds[redis_password_key] = current_password[1]

        if current_url_value is not None:
            desired_url = _redis_url_with_password(
                current_url_value,
                current_password[1],
            )
            if desired_url != current_url_value:
                if redis_url_key is None:
                    redis_url_key = "REDIS_URL"
                seeds[redis_url_key] = desired_url
                replace_existing_keys.add(redis_url_key.casefold())
                needs_legacy_url_alias = (
                    redis_password_from_url(current_url_value) is None
                )
        elif redis_url_key is not None:
            seeds[redis_url_key] = _redis_url_with_password(
                seeds[redis_url_key],
                current_password[1],
            )

    elif current_url is not None:
        current_url_password = redis_password_from_url(current_url_value)
        if current_url_password is not None:
            if redis_password_key is None:
                redis_password_key = "REDIS_PASSWORD"
            seeds[redis_password_key] = current_url_password
            desired_url = _redis_url_with_password(current_url_value, current_url_password)
            if redis_url_key is None:
                redis_url_key = "REDIS_URL"
            seeds[redis_url_key] = desired_url
        elif redis_password_key is not None:
            generated_password = seeds[redis_password_key]
            desired_url = _redis_url_with_password(current_url_value, generated_password)
            if redis_url_key is None:
                redis_url_key = "REDIS_URL"
            seeds[redis_url_key] = desired_url
            replace_existing_keys.add(redis_url_key.casefold())
            needs_legacy_url_alias = True

    if needs_legacy_url_alias and redis_url_key is not None and current_url_value is not None:
        reconciled_url = seeds.pop(redis_url_key)
        legacy_key = "ONCALL_LEGACY_REDIS_URL"
        existing_legacy = current_values.get(legacy_key.casefold())
        if existing_legacy is None or existing_legacy[1] != current_url_value:
            seeds[legacy_key] = current_url_value
        seeds[redis_url_key] = reconciled_url

    return {
        key: value
        for key, value in seeds.items()
        if key.casefold() not in current_values
        or key.casefold() in replace_existing_keys
    }


def _validated_canonical_values(canonical_values: Mapping[str, str]) -> dict[str, str]:
    values: dict[str, str] = {}
    folded_keys: set[str] = set()
    for key, value in canonical_values.items():
        if not isinstance(key, str) or _KEY_RE.fullmatch(key) is None:
            raise ValueError("invalid canonical key")
        folded_key = key.casefold()
        if folded_key in folded_keys:
            raise ValueError("duplicate canonical key")
        if not isinstance(value, str):
            raise TypeError(f"canonical value for {key} must be a string")
        if "\n" in value or "\r" in value:
            raise ValueError(f"canonical value for {key} must be one line")
        _serialize_dotenv_value(value)
        values[key] = value
        folded_keys.add(folded_key)
    return values


def _merge_dotenv(
    target_text: str,
    source_text: str,
    seed_values: Mapping[str, str],
    canonical_values: Mapping[str, str],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    current_values = _dotenv_values(target_text)
    source_values = _dotenv_values(source_text)
    imported_keys: list[str] = []
    imported_folded_keys: set[str] = set()
    conflicted_keys: list[str] = []
    merged = target_text

    def record_imported_key(key: str) -> None:
        folded_key = key.casefold()
        if folded_key in imported_folded_keys:
            return
        imported_folded_keys.add(folded_key)
        imported_keys.append(key)

    for folded_key, (key, source_value) in source_values.items():
        if folded_key not in current_values:
            merged = _append_assignment(merged, key, source_value)
            current_values[folded_key] = (key, source_value)
            record_imported_key(key)
            continue
        if current_values[folded_key][1] == source_value:
            continue

        conflicted_keys.append(key)
        base_alias = _CONFLICT_ALIASES_CASEFOLD.get(folded_key, f"ONCALL_LEGACY_{key}")
        alias, should_append = _available_alias(base_alias, source_value, current_values)
        if should_append:
            merged = _append_assignment(merged, alias, source_value)
            current_values[alias.casefold()] = (alias, source_value)
            record_imported_key(alias)

    for key, value in _reconciled_seed_values(seed_values, current_values).items():
        folded_key = key.casefold()
        if folded_key in current_values and current_values[folded_key][1] == value:
            continue
        merged = _append_assignment(merged, key, value)
        current_values[folded_key] = (key, value)
        record_imported_key(key)

    for key, canonical_value in _validated_canonical_values(canonical_values).items():
        folded_key = key.casefold()
        current = current_values.get(folded_key)
        if current is not None and current[1] == canonical_value:
            continue
        if current is not None:
            base_alias = _CONFLICT_ALIASES_CASEFOLD.get(
                folded_key,
                f"ONCALL_LEGACY_{key}",
            )
            alias, should_append = _available_alias(
                base_alias,
                current[1],
                current_values,
            )
            if should_append:
                merged = _append_assignment(merged, alias, current[1])
                current_values[alias.casefold()] = (alias, current[1])
                record_imported_key(alias)
        merged = _append_assignment(merged, key, canonical_value)
        current_values[folded_key] = (key, canonical_value)
        record_imported_key(key)

    return merged, tuple(imported_keys), tuple(conflicted_keys)


def _read_snapshot_text(path: Path) -> str:
    if not path.exists():
        return ""
    return _read_private_source(path).decode("utf-8")


def _write_durable_temporary_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    created = True
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def _restore_target_env(target_root: Path, original_content: bytes | None) -> None:
    target_env = target_root / ".env"
    if original_content is None:
        target_env.unlink(missing_ok=True)
        _fsync_directory(target_root)
        return

    restore_path = target_root / ".env.restore.tmp"
    restore_ready = False
    try:
        _write_durable_temporary_file(restore_path, original_content)
        restore_ready = True
        os.replace(restore_path, target_env)
        restore_ready = False
        _fsync_directory(target_root)
    except BaseException:
        if restore_ready:
            restore_path.unlink(missing_ok=True)
        raise


def _replace_target_env(target_root: Path, content: str, original_content: bytes | None) -> None:
    target_env = target_root / ".env"
    temporary_env = target_root / ".env.tmp"
    temporary_ready = False
    try:
        _write_durable_temporary_file(temporary_env, content.encode("utf-8"))
        temporary_ready = True
        os.replace(temporary_env, target_env)
        temporary_ready = False
        try:
            _fsync_directory(target_root)
        except BaseException as publish_error:
            _restore_target_env(target_root, original_content)
            raise publish_error
    except BaseException:
        if temporary_ready:
            temporary_env.unlink(missing_ok=True)
        raise


def migrate_configuration(request: MigrationRequest) -> MigrationResult:
    """Snapshot source/target config, merge dotenv keys, and replace target .env atomically."""

    _validate_inputs(request)
    vault_dir, backup_path, manifest = _create_vault(request)
    source_text = _read_snapshot_text(vault_dir / "source" / ".env.local")
    original_target = _read_private_source(backup_path) if backup_path.exists() else None
    target_text = original_target.decode("utf-8") if original_target is not None else ""
    merged, imported_keys, conflicted_keys = _merge_dotenv(
        target_text,
        source_text,
        request.seed_values,
        request.canonical_values,
    )
    _replace_target_env(request.target_root, merged, original_target)
    return MigrationResult(
        vault_dir=vault_dir,
        backup_path=backup_path,
        manifest=manifest,
        imported_keys=imported_keys,
        conflicted_keys=conflicted_keys,
    )
