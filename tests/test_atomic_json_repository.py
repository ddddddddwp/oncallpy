from __future__ import annotations

import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from app.domains.shared import atomic_json_repository as repository_module
from app.domains.shared.atomic_json_repository import AtomicJsonRepository


def test_round_trip_is_unicode_deterministic_and_private(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "state.json"
    repository = AtomicJsonRepository[dict[str, object]](path, dict)
    value = {"zeta": [3, 1], "alpha": "能源"}

    repository.save(value)

    assert repository.load() == value
    assert path.read_text(encoding="utf-8") == (
        '{\n  "alpha": "能源",\n  "zeta": [\n    3,\n    1\n  ]\n}\n'
    )
    assert path.stat().st_mode & 0o777 == 0o600


def test_missing_load_returns_a_fresh_default_each_time(tmp_path: Path) -> None:
    repository = AtomicJsonRepository[list[str]](tmp_path / "missing.json", list)

    first = repository.load()
    first.append("changed")
    second = repository.load()

    assert first == ["changed"]
    assert second == []
    assert second is not first


def test_replace_failure_preserves_previous_value_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    repository = AtomicJsonRepository[dict[str, int]](path, dict)
    repository.save({"version": 1})
    previous_text = path.read_text(encoding="utf-8")
    temporary_paths: list[Path] = []

    def fail_replace(temporary: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        temporary_path = Path(temporary)
        temporary_paths.append(temporary_path)
        assert Path(target) == path
        assert temporary_path.parent == path.parent
        assert temporary_path.stat().st_mode & 0o777 == 0o600
        raise OSError("replace failed")

    monkeypatch.setattr(repository_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        repository.save({"version": 2})

    assert path.read_text(encoding="utf-8") == previous_text
    assert temporary_paths and all(not temporary.exists() for temporary in temporary_paths)
    assert sorted(tmp_path.iterdir()) == [path]


def test_serialization_failure_preserves_previous_value(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    repository = AtomicJsonRepository[dict[str, object]](path, dict)
    repository.save({"version": 1})
    previous_text = path.read_text(encoding="utf-8")

    with pytest.raises(TypeError):
        repository.save({"unsupported": object()})

    assert path.read_text(encoding="utf-8") == previous_text
    assert sorted(tmp_path.iterdir()) == [path]


def test_each_save_uses_a_unique_private_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "state.json"
    repository = AtomicJsonRepository[dict[str, int]](path, dict)
    temporary_paths: list[Path] = []

    def record_and_fail(temporary: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        temporary_path = Path(temporary)
        temporary_paths.append(temporary_path)
        assert Path(target) == path
        assert temporary_path.parent == path.parent
        assert temporary_path.stat().st_mode & 0o777 == 0o600
        raise OSError("replace failed")

    monkeypatch.setattr(repository_module.os, "replace", record_and_fail)

    for version in range(2):
        with pytest.raises(OSError, match="replace failed"):
            repository.save({"version": version})

    assert len(set(temporary_paths)) == 2
    assert all(not temporary.exists() for temporary in temporary_paths)
    assert not list(tmp_path.iterdir())


def test_sequential_and_concurrent_writes_leave_valid_json(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    repository = AtomicJsonRepository[dict[str, int]](path, dict)

    for version in range(5):
        repository.save({"version": version})
        assert json.loads(path.read_text(encoding="utf-8")) == {"version": version}

    values = [{"version": version} for version in range(5, 25)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(repository.save, values))

    assert json.loads(path.read_text(encoding="utf-8")) in values


def test_invalid_json_propagates_json_decode_error(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("not-json\n", encoding="utf-8")
    repository = AtomicJsonRepository[dict[str, object]](path, dict)

    with pytest.raises(json.JSONDecodeError, match="Expecting value"):
        repository.load()


def test_directory_destination_is_rejected_and_temp_is_cleaned(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.mkdir()
    repository = AtomicJsonRepository[dict[str, int]](path, dict)

    with pytest.raises(OSError):
        repository.save({"version": 1})

    assert path.is_dir()
    assert sorted(tmp_path.iterdir()) == [path]


def test_successful_save_fsyncs_file_and_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "nested" / "state.json"
    repository = AtomicJsonRepository[dict[str, int]](path, dict)
    real_fsync = os.fsync
    fsynced_modes: list[int] = []

    def record_fsync(descriptor: int) -> None:
        fsynced_modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(repository_module.os, "fsync", record_fsync)

    repository.save({"version": 1})

    assert any(stat.S_ISREG(mode) for mode in fsynced_modes)
    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)
