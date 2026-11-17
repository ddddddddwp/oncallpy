from __future__ import annotations

import hashlib
import importlib
import os
import stat
import warnings
from pathlib import Path

import pytest

from app.core import compatibility
from app.core.compatibility import copy_legacy_state, resolve_compat_env


@pytest.mark.parametrize(
    ("legacy_name", "current_name"),
    [
        ("app.services.document_splitter_service", "app.rag.document_splitter_service"),
        ("app.services.rag_deduplicator", "app.rag.rag_deduplicator"),
        ("app.services.rag_document_loader", "app.rag.rag_document_loader"),
        ("app.services.rag_ingestion_models", "app.rag.rag_ingestion_models"),
        ("app.services.rag_metadata_builder", "app.rag.rag_metadata_builder"),
        ("app.services.vector_embedding_service", "app.rag.vector_embedding_service"),
        ("app.services.vector_search_service", "app.rag.vector_search_service"),
        ("app.services.vector_store_manager", "app.rag.vector_store_manager"),
    ],
)
def test_legacy_service_module_is_the_current_rag_module(
    legacy_name: str,
    current_name: str,
) -> None:
    assert importlib.import_module(legacy_name) is importlib.import_module(current_name)


@pytest.mark.parametrize(
    ("module_name", "public_names"),
    [
        ("app.rag.document_splitter_service", ("DocumentSplitterService", "document_splitter_service")),
        ("app.rag.rag_document_loader", ("RagDocumentLoader",)),
        ("app.rag.rag_metadata_builder", ("RagMetadataBuilder",)),
        ("app.rag.rag_deduplicator", ("RagDeduplicator",)),
        ("app.rag.vector_embedding_service", ("DashScopeEmbeddings", "vector_embedding_service")),
        ("app.rag.vector_store_manager", ("VectorStoreManager", "vector_store_manager")),
    ],
)
def test_legacy_rag_public_objects_remain_importable(
    module_name: str,
    public_names: tuple[str, ...],
) -> None:
    module = importlib.import_module(module_name)

    for public_name in public_names:
        assert getattr(module, public_name) is not None


def test_rag_package_exposes_modular_api_lazily() -> None:
    import app.rag as rag

    assert rag.Document.__module__ == "app.rag.models.document"
    assert rag.SearchQuery.__module__ == "app.rag.models.search_query"
    assert rag.SearchHit.__module__ == "app.rag.models.search_hit"
    assert rag.IndexingResult.__module__ == "app.rag.models.indexing_result"
    assert rag.IndexingService.__module__ == "app.rag.indexing.indexing_service"
    assert rag.QueryService.__module__ == "app.rag.query.query_service"


def test_current_compat_env_wins_without_warning() -> None:
    with warnings.catch_warnings(record=True) as recorded:
        value = resolve_compat_env("current-value", "legacy-value", "default-value")

    assert value == "current-value"
    assert not recorded


def test_legacy_compat_env_warns_without_disclosing_value() -> None:
    legacy_value = "legacy-secret-value"

    with pytest.warns(DeprecationWarning) as recorded:
        value = resolve_compat_env("", legacy_value, "default-value")

    assert value == legacy_value
    assert legacy_value not in str(recorded[0].message)
    assert "SUPER_BIZ_AGENT" in str(recorded[0].message)


def test_empty_compat_env_values_fall_back_to_default() -> None:
    with warnings.catch_warnings(record=True) as recorded:
        value = resolve_compat_env("", "", "default-value")

    assert value == "default-value"
    assert not recorded


def test_whitespace_compat_env_value_counts_as_nonempty() -> None:
    with warnings.catch_warnings(record=True) as recorded:
        value = resolve_compat_env(" ", "legacy-value", "default-value")

    assert value == " "
    assert not recorded


def test_copy_legacy_state_is_byte_exact_private_and_preserves_source(tmp_path: Path) -> None:
    payload = b'{"alerts":["alpha","beta"],"enabled":true}\n'
    source = tmp_path / "legacy" / "state.json"
    source.parent.mkdir()
    source.write_bytes(payload)
    destination = tmp_path / "nested" / "energyagent" / "state.json"

    result = copy_legacy_state(source, destination)

    assert result.copied is True
    assert result.source == source
    assert result.destination == destination
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert source.read_bytes() == payload
    assert destination.read_bytes() == payload
    assert destination.stat().st_mode & 0o777 == 0o600


def test_successful_copy_fsyncs_destination_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "legacy.json"
    destination = tmp_path / "nested" / "current.json"
    source.write_bytes(b"legacy")
    real_fsync = os.fsync
    fsynced_modes: list[int] = []

    def record_fsync(descriptor: int) -> None:
        fsynced_modes.append(os.fstat(descriptor).st_mode)
        real_fsync(descriptor)

    monkeypatch.setattr(compatibility.os, "fsync", record_fsync)

    result = copy_legacy_state(source, destination)

    assert result.copied is True
    assert any(stat.S_ISDIR(mode) for mode in fsynced_modes)


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "legacy.json"
    destination = tmp_path / "current.json"
    source.write_bytes(b"legacy")
    destination.write_bytes(b"current")

    result = copy_legacy_state(source, destination)

    assert result.copied is False
    assert result.sha256 == ""
    assert source.read_bytes() == b"legacy"
    assert destination.read_bytes() == b"current"


@pytest.mark.parametrize("race_kind", ["file", "symlink"])
def test_publish_race_never_overwrites_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, race_kind: str
) -> None:
    source = tmp_path / "legacy.json"
    destination = tmp_path / "current.json"
    winner_target = tmp_path / "winner.json"
    source.write_bytes(b"legacy")
    winner_target.write_bytes(b"race-winner")
    real_link = os.link
    real_replace = os.replace

    def install_race_winner(target: Path) -> None:
        if race_kind == "symlink":
            target.symlink_to(winner_target)
        else:
            target.write_bytes(b"race-winner")

    def racing_link(
        temporary: str | os.PathLike[str],
        target: str | os.PathLike[str],
        *,
        follow_symlinks: bool = True,
    ) -> None:
        install_race_winner(Path(target))
        real_link(temporary, target, follow_symlinks=follow_symlinks)

    def racing_replace(temporary: str | os.PathLike[str], target: str | os.PathLike[str]) -> None:
        install_race_winner(Path(target))
        real_replace(temporary, target)

    monkeypatch.setattr(compatibility.os, "link", racing_link)
    monkeypatch.setattr(compatibility.os, "replace", racing_replace)

    result = copy_legacy_state(source, destination)

    assert result.copied is False
    assert result.sha256 == ""
    assert source.read_bytes() == b"legacy"
    if race_kind == "symlink":
        assert destination.is_symlink()
        assert destination.readlink() == winner_target
        assert winner_target.read_bytes() == b"race-winner"
    else:
        assert destination.read_bytes() == b"race-winner"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_missing_source_is_not_copied(tmp_path: Path) -> None:
    source = tmp_path / "missing.json"
    destination = tmp_path / "current.json"

    result = copy_legacy_state(source, destination)

    assert result.copied is False
    assert result.source == source
    assert result.destination == destination
    assert result.sha256 == ""
    assert not destination.exists()


def test_publish_failure_preserves_race_winner_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "legacy.json"
    destination = tmp_path / "current.json"
    source.write_bytes(b"legacy")

    def fail_publish(
        _temporary: str | os.PathLike[str],
        target: str | os.PathLike[str],
        *,
        follow_symlinks: bool,
    ) -> None:
        assert follow_symlinks is False
        Path(target).write_bytes(b"race-winner")
        raise OSError("publish failed")

    monkeypatch.setattr(compatibility.os, "link", fail_publish)

    with pytest.raises(OSError, match="publish failed"):
        copy_legacy_state(source, destination)

    assert destination.read_bytes() == b"race-winner"
    assert sorted(tmp_path.iterdir()) == [destination, source]


def test_checksum_mismatch_refuses_to_replace_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "legacy.json"
    destination = tmp_path / "current.json"
    source.write_bytes(b"legacy")
    monkeypatch.setattr(compatibility, "_sha256_file", lambda _path: "0" * 64)

    with pytest.raises(OSError, match="checksum"):
        copy_legacy_state(source, destination)

    assert not destination.exists()
    assert sorted(tmp_path.iterdir()) == [source]


@pytest.mark.parametrize("source_kind", ["symlink", "directory"])
def test_non_regular_legacy_source_is_rejected(tmp_path: Path, source_kind: str) -> None:
    regular_file = tmp_path / "regular.json"
    regular_file.write_bytes(b"legacy")
    source = tmp_path / "legacy.json"
    if source_kind == "symlink":
        source.symlink_to(regular_file)
    else:
        source.mkdir()

    with pytest.raises(ValueError, match="regular file"):
        copy_legacy_state(source, tmp_path / "current.json")


def test_destination_symlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "legacy.json"
    source.write_bytes(b"legacy")
    target = tmp_path / "target.json"
    target.write_bytes(b"current")
    destination = tmp_path / "current.json"
    destination.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        copy_legacy_state(source, destination)

    assert target.read_bytes() == b"current"


def test_copy_uses_a_unique_private_temporary_file_per_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "legacy.json"
    destination = tmp_path / "current.json"
    source.write_bytes(b"legacy")
    temporary_paths: list[Path] = []

    def record_and_fail(
        temporary: str | os.PathLike[str],
        _target: str | os.PathLike[str],
        *,
        follow_symlinks: bool,
    ) -> None:
        assert follow_symlinks is False
        temporary_path = Path(temporary)
        temporary_paths.append(temporary_path)
        assert temporary_path.stat().st_mode & 0o777 == 0o600
        raise OSError("publish failed")

    monkeypatch.setattr(compatibility.os, "link", record_and_fail)

    for _ in range(2):
        with pytest.raises(OSError, match="publish failed"):
            copy_legacy_state(source, destination)

    assert len(set(temporary_paths)) == 2
    assert all(not path.exists() for path in temporary_paths)
