import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import types
from io import StringIO
from pathlib import Path
from urllib.parse import quote

import pytest
from dotenv import dotenv_values

from app.core import config_migration
from app.core.config_migration import (
    CONFLICT_ALIASES,
    SNAPSHOT_PATHS,
    MigrationRequest,
    migrate_configuration,
)

TIMESTAMP = "20260720T120000Z"


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source_root.mkdir()
    target_root.mkdir()
    return source_root, target_root


def _request(
    source_root: Path,
    target_root: Path,
    *,
    timestamp: str = TIMESTAMP,
    seed_values: dict[str, str] | None = None,
    canonical_values: dict[str, str] | None = None,
) -> MigrationRequest:
    return MigrationRequest(
        source_root=source_root,
        target_root=target_root,
        timestamp=timestamp,
        seed_values=seed_values or {},
        canonical_values=canonical_values or {},
    )


def _redis_seed_pair(password: str) -> dict[str, str]:
    encoded_password = quote(password, safe="")
    return {
        "REDIS_PASSWORD": password,
        "REDIS_URL": f"redis://:{encoded_password}@localhost:6379/0",
    }


def test_migration_preserves_target_and_aliases_source_conflicts(tmp_path: Path) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text(
        "DEEPSEEK_API_KEY=fake-source-deepseek\n"
        "COZELOOP_API_TOKEN=fake-source-cozeloop\n"
        "SOURCE_ONLY=fake-source-only\n"
        "SHARED_EQUAL=fake-shared\n",
        encoding="utf-8",
    )
    (source_root / "config.yaml").write_text(
        "provider:\n  token: fake-yaml-token\n",
        encoding="utf-8",
    )
    (target_root / ".env").write_text(
        "# preserve this target comment\n"
        "DEEPSEEK_API_KEY=fake-target-deepseek\n"
        "COZELOOP_API_TOKEN=fake-target-cozeloop\n"
        "TARGET_ONLY=fake-target-only\n"
        "SHARED_EQUAL=fake-shared\n",
        encoding="utf-8",
    )

    result = migrate_configuration(_request(source_root, target_root))

    migrated = (target_root / ".env").read_text(encoding="utf-8")
    assert migrated.startswith("# preserve this target comment\n")
    assert "DEEPSEEK_API_KEY=fake-target-deepseek\n" in migrated
    assert "COZELOOP_API_TOKEN=fake-target-cozeloop\n" in migrated
    assert "ONCALL_LEGACY_DEEPSEEK_API_KEY=fake-source-deepseek\n" in migrated
    assert "ONCALL_LEGACY_COZELOOP_API_TOKEN=fake-source-cozeloop\n" in migrated
    assert "SOURCE_ONLY=fake-source-only\n" in migrated
    assert "TARGET_ONLY=fake-target-only\n" in migrated
    assert migrated.count("SHARED_EQUAL=") == 1
    assert result.conflicted_keys == ("DEEPSEEK_API_KEY", "COZELOOP_API_TOKEN")


def test_public_constants_match_the_legacy_contract() -> None:
    assert CONFLICT_ALIASES == {
        "DEEPSEEK_API_KEY": "ONCALL_LEGACY_DEEPSEEK_API_KEY",
        "COZELOOP_API_TOKEN": "ONCALL_LEGACY_COZELOOP_API_TOKEN",
    }
    assert SNAPSHOT_PATHS == (
        ".env.local",
        ".env.example",
        "config.yaml",
        "manifest/docker/docker-compose.yml",
        "manifest/observability",
    )


def test_generic_conflict_uses_an_unoccupied_deterministic_alias(tmp_path: Path) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("GENERIC_KEY=fake-source-generic\n", encoding="utf-8")
    (target_root / ".env").write_text(
        "GENERIC_KEY=fake-target-generic\n" "ONCALL_LEGACY_GENERIC_KEY=fake-existing-alias\n",
        encoding="utf-8",
    )

    result = migrate_configuration(_request(source_root, target_root))

    migrated = (target_root / ".env").read_text(encoding="utf-8")
    assert "ONCALL_LEGACY_GENERIC_KEY=fake-existing-alias\n" in migrated
    assert "ONCALL_LEGACY_GENERIC_KEY_2=fake-source-generic\n" in migrated
    assert result.conflicted_keys == ("GENERIC_KEY",)


def test_canonical_app_name_preserves_equal_legacy_target_and_source_value(
    tmp_path: Path,
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("APP_NAME=LegacyShared\n", encoding="utf-8")
    (target_root / ".env").write_text("APP_NAME=LegacyShared\n", encoding="utf-8")

    result = migrate_configuration(
        _request(
            source_root,
            target_root,
            canonical_values={"APP_NAME": "EnergyAgent"},
        )
    )

    migrated = (target_root / ".env").read_text(encoding="utf-8")
    parsed = dotenv_values(stream=StringIO(migrated), interpolate=False)
    assert parsed["APP_NAME"] == "EnergyAgent"
    assert parsed["ONCALL_LEGACY_APP_NAME"] == "LegacyShared"
    assert [line for line in migrated.splitlines() if line.startswith("APP_NAME=")] == [
        "APP_NAME=LegacyShared",
        "APP_NAME=EnergyAgent",
    ]
    assert result.imported_keys == ("ONCALL_LEGACY_APP_NAME", "APP_NAME")
    assert result.conflicted_keys == ()


def test_canonical_app_name_preserves_distinct_source_and_target_with_alias_collision(
    tmp_path: Path,
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("APP_NAME=LegacySource\n", encoding="utf-8")
    (target_root / ".env").write_text(
        "APP_NAME=LegacyTarget\n" "ONCALL_LEGACY_APP_NAME=OccupiedAlias\n",
        encoding="utf-8",
    )

    result = migrate_configuration(
        _request(
            source_root,
            target_root,
            canonical_values={"APP_NAME": "EnergyAgent"},
        )
    )

    parsed = dotenv_values(target_root / ".env", interpolate=False)
    assert parsed["APP_NAME"] == "EnergyAgent"
    assert parsed["ONCALL_LEGACY_APP_NAME"] == "OccupiedAlias"
    assert parsed["ONCALL_LEGACY_APP_NAME_2"] == "LegacySource"
    assert parsed["ONCALL_LEGACY_APP_NAME_3"] == "LegacyTarget"
    assert result.imported_keys == (
        "ONCALL_LEGACY_APP_NAME_2",
        "ONCALL_LEGACY_APP_NAME_3",
        "APP_NAME",
    )
    assert result.conflicted_keys == ("APP_NAME",)


def test_canonical_app_name_is_added_without_legacy_alias_when_missing(tmp_path: Path) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("SOURCE_ONLY=FakeSource\n", encoding="utf-8")
    (target_root / ".env").write_text("TARGET_ONLY=FakeTarget\n", encoding="utf-8")

    result = migrate_configuration(
        _request(
            source_root,
            target_root,
            canonical_values={"APP_NAME": "EnergyAgent"},
        )
    )

    parsed = dotenv_values(target_root / ".env", interpolate=False)
    assert parsed["APP_NAME"] == "EnergyAgent"
    assert "ONCALL_LEGACY_APP_NAME" not in parsed
    assert result.imported_keys == ("SOURCE_ONLY", "APP_NAME")
    assert result.conflicted_keys == ()


def test_source_only_app_name_and_canonical_override_report_unique_logical_keys(
    tmp_path: Path,
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("APP_NAME=LegacySource\n", encoding="utf-8")

    result = migrate_configuration(
        _request(
            source_root,
            target_root,
            canonical_values={"APP_NAME": "EnergyAgent"},
        )
    )

    parsed = dotenv_values(target_root / ".env", interpolate=False)
    assert parsed["APP_NAME"] == "EnergyAgent"
    assert parsed["ONCALL_LEGACY_APP_NAME"] == "LegacySource"
    assert result.imported_keys == ("APP_NAME", "ONCALL_LEGACY_APP_NAME")
    assert len({key.casefold() for key in result.imported_keys}) == len(result.imported_keys)


def test_already_canonical_app_name_is_idempotent_across_migrations(tmp_path: Path) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("APP_NAME=LegacySource\n", encoding="utf-8")
    (target_root / ".env").write_text("APP_NAME=LegacyTarget\n", encoding="utf-8")
    canonical_values = {"APP_NAME": "EnergyAgent"}

    first = migrate_configuration(
        _request(source_root, target_root, canonical_values=canonical_values)
    )
    after_first = (target_root / ".env").read_text(encoding="utf-8")
    second = migrate_configuration(
        _request(
            source_root,
            target_root,
            timestamp="20260720T120001Z",
            canonical_values=canonical_values,
        )
    )
    after_second = (target_root / ".env").read_text(encoding="utf-8")

    parsed = dotenv_values(stream=StringIO(after_second), interpolate=False)
    assert parsed["APP_NAME"] == "EnergyAgent"
    assert parsed["ONCALL_LEGACY_APP_NAME"] == "LegacySource"
    assert parsed["ONCALL_LEGACY_APP_NAME_2"] == "LegacyTarget"
    assert after_second == after_first
    assert first.imported_keys == (
        "ONCALL_LEGACY_APP_NAME",
        "ONCALL_LEGACY_APP_NAME_2",
        "APP_NAME",
    )
    assert second.imported_keys == ()
    assert first.conflicted_keys == second.conflicted_keys == ("APP_NAME",)


@pytest.mark.parametrize(
    ("canonical_values", "hidden_value"),
    (
        ({"INVALID-KEY": "HiddenBrandValue"}, "HiddenBrandValue"),
        ({"APP_NAME": "HiddenBrandValue\nINJECTED=Fake"}, "HiddenBrandValue"),
    ),
)
def test_unsafe_canonical_assignment_is_rejected_without_value_disclosure(
    tmp_path: Path,
    canonical_values: dict[str, str],
    hidden_value: str,
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("", encoding="utf-8")
    original = b"TARGET_ONLY=FakeTarget\n"
    (target_root / ".env").write_bytes(original)

    with pytest.raises((TypeError, ValueError)) as exc_info:
        migrate_configuration(
            _request(
                source_root,
                target_root,
                canonical_values=canonical_values,
            )
        )

    assert hidden_value not in str(exc_info.value)
    assert (target_root / ".env").read_bytes() == original


def test_casefold_duplicate_canonical_keys_are_rejected_without_value_disclosure(
    tmp_path: Path,
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("", encoding="utf-8")
    original = b"TARGET_ONLY=FakeTarget\n"
    (target_root / ".env").write_bytes(original)
    hidden_values = ("HiddenFirstBrand", "HiddenSecondBrand")

    with pytest.raises(ValueError) as exc_info:
        migrate_configuration(
            _request(
                source_root,
                target_root,
                canonical_values={
                    "APP_NAME": hidden_values[0],
                    "app_name": hidden_values[1],
                },
            )
        )

    assert str(exc_info.value) == "duplicate canonical key"
    for hidden_value in hidden_values:
        assert hidden_value not in str(exc_info.value)
    assert (target_root / ".env").read_bytes() == original


def test_dotenv_merge_matches_consumer_grammar_and_casefolds_effective_keys(
    tmp_path: Path,
) -> None:
    source_root, target_root = _roots(tmp_path)
    source_text = (
        '  deepseek_api_key = "fake-source-old"\n'
        'DEEPSEEK_API_KEY = "fake-source-effective\nsecond ${UNEXPANDED}"\n'
        "duplicate = fake-source-duplicate\n"
        'SOURCE_ONLY = "fake-source ${UNEXPANDED}"\n'
    )
    target_text = (
        "# preserve target bytes exactly\n"
        ' DEEPSEEK_API_KEY = "fake-target-old"\n'
        "deepseek_api_key = fake-target-effective\n"
        "DUPLICATE=fake-target-old-duplicate\n"
        "DUPLICATE = fake-target-effective-duplicate\n"
        " oncall_legacy_deepseek_api_key = fake-existing-alias\n"
    )
    (source_root / ".env.local").write_text(source_text, encoding="utf-8")
    (target_root / ".env").write_text(target_text, encoding="utf-8")

    result = migrate_configuration(_request(source_root, target_root))

    migrated = (target_root / ".env").read_text(encoding="utf-8")
    parsed = {
        key.casefold(): value
        for key, value in dotenv_values(stream=StringIO(migrated), interpolate=False).items()
        if value is not None
    }
    assert migrated.startswith(target_text)
    assert parsed["deepseek_api_key"] == "fake-target-effective"
    assert parsed["duplicate"] == "fake-target-effective-duplicate"
    assert parsed["oncall_legacy_deepseek_api_key"] == "fake-existing-alias"
    assert parsed["oncall_legacy_deepseek_api_key_2"] == (
        "fake-source-effective\nsecond ${UNEXPANDED}"
    )
    assert parsed["oncall_legacy_duplicate"] == "fake-source-duplicate"
    assert parsed["source_only"] == "fake-source ${UNEXPANDED}"
    assert result.conflicted_keys == ("DEEPSEEK_API_KEY", "duplicate")


def test_manifest_contains_only_non_secret_metadata_and_yaml_key_paths(tmp_path: Path) -> None:
    source_root, target_root = _roots(tmp_path)
    fake_values = (
        "fake-dotenv-secret",
        "fake-nested-password",
        "fake-service-host",
        "fake-target-secret",
    )
    (source_root / ".env.local").write_text(f"SOURCE_SECRET={fake_values[0]}\n", encoding="utf-8")
    (source_root / "config.yaml").write_text(
        "database:\n"
        f"  password: {fake_values[1]}\n"
        f"  host: {fake_values[2]}\n"
        "features:\n"
        "  enabled: true\n",
        encoding="utf-8",
    )
    (target_root / ".env").write_text(f"TARGET_SECRET={fake_values[3]}\n", encoding="utf-8")

    result = migrate_configuration(_request(source_root, target_root))

    manifest_text = json.dumps(result.manifest, sort_keys=True)
    disk_manifest_text = (result.vault_dir / "manifest.json").read_text(encoding="utf-8")
    for fake_value in fake_values:
        assert fake_value not in manifest_text
        assert fake_value not in disk_manifest_text
    assert json.loads(disk_manifest_text) == result.manifest
    assert set(result.manifest) == {"files"}

    entries = result.manifest["files"]
    assert isinstance(entries, list)
    assert entries
    for entry in entries:
        assert set(entry) <= {"path", "size", "sha256", "mode", "kind", "yaml_keys"}
        assert {"path", "size", "sha256", "mode", "kind"} <= set(entry)
        assert not Path(entry["path"]).is_absolute()
        assert len(entry["sha256"]) == 64

    yaml_entry = next(entry for entry in entries if entry["path"] == "source/config.yaml")
    assert yaml_entry["yaml_keys"] == [
        "database",
        "database.host",
        "database.password",
        "features",
        "features.enabled",
    ]
    assert set(result.imported_keys) == {"SOURCE_SECRET"}
    assert result.conflicted_keys == ()


def test_snapshots_all_declared_paths_with_private_permissions(tmp_path: Path) -> None:
    source_root, target_root = _roots(tmp_path)
    fixture_contents = {
        ".env.local": "LOCAL_ONLY=fake-local\n",
        ".env.example": "EXAMPLE_ONLY=fake-example\n",
        "config.yaml": "service:\n  password: fake-config-password\n",
        "manifest/docker/docker-compose.yml": "services:\n  api:\n    image: fake-image\n",
        "manifest/observability/prometheus/prometheus.yml": "scrape_configs: []\n",
        "manifest/observability/grafana/dashboards/main.json": '{"fake": true}\n',
    }
    for relative_path, content in fixture_contents.items():
        path = source_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    target_before = b"TARGET_ONLY=fake-target-before\n"
    (target_root / ".env").write_bytes(target_before)

    result = migrate_configuration(_request(source_root, target_root))

    assert result.vault_dir == target_root / ".energyagent" / "migration" / TIMESTAMP
    assert result.backup_path == result.vault_dir / "target-before" / ".env"
    assert result.backup_path.read_bytes() == target_before
    for relative_path, content in fixture_contents.items():
        source_path = source_root / relative_path
        snapshot_path = result.vault_dir / "source" / relative_path
        assert source_path.read_text(encoding="utf-8") == content
        assert snapshot_path.read_text(encoding="utf-8") == content
    assert "EXAMPLE_ONLY=" not in (target_root / ".env").read_text(encoding="utf-8")

    private_directories = [
        target_root / ".energyagent",
        target_root / ".energyagent" / "migration",
        result.vault_dir,
        *(path for path in result.vault_dir.rglob("*") if path.is_dir()),
    ]
    for path in private_directories:
        assert stat.S_IMODE(path.stat().st_mode) == 0o700
    for path in (path for path in result.vault_dir.rglob("*") if path.is_file()):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE((target_root / ".env").stat().st_mode) == 0o600


def test_missing_seed_values_are_appended_and_redis_password_is_derived(tmp_path: Path) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("", encoding="utf-8")
    (target_root / ".env").write_text(
        "# target stays first\nEXISTING_SEED=fake-existing-target\n",
        encoding="utf-8",
    )
    seed_values = {
        "REDIS_URL": "redis://:fake-redis-password@redis.invalid:6379/0",
        "EXPLICIT_SEED": "fake-explicit-seed",
        "EXISTING_SEED": "fake-seed-must-not-replace",
    }

    result = migrate_configuration(_request(source_root, target_root, seed_values=seed_values))

    migrated = (target_root / ".env").read_text(encoding="utf-8")
    parsed = dotenv_values(stream=StringIO(migrated), interpolate=False)
    assert migrated.startswith("# target stays first\n")
    assert "EXISTING_SEED=fake-existing-target\n" in migrated
    assert "fake-seed-must-not-replace" not in migrated
    assert parsed["REDIS_URL"] == "redis://:fake-redis-password@redis.invalid:6379/0"
    assert parsed["REDIS_PASSWORD"] == "fake-redis-password"
    assert parsed["EXPLICIT_SEED"] == "fake-explicit-seed"
    assert set(result.imported_keys) == {"REDIS_URL", "REDIS_PASSWORD", "EXPLICIT_SEED"}


def test_redis_seed_pair_fills_both_missing_values_with_encoded_password(tmp_path: Path) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("", encoding="utf-8")
    (target_root / ".env").write_text("# target\n", encoding="utf-8")
    password = "fake generated/+?#'\\tail"
    seed_values = _redis_seed_pair(password)

    result = migrate_configuration(_request(source_root, target_root, seed_values=seed_values))

    parsed = dotenv_values(target_root / ".env", interpolate=False)
    assert parsed["REDIS_PASSWORD"] == password
    assert parsed["REDIS_URL"] == seed_values["REDIS_URL"]
    assert config_migration.redis_password_from_url(parsed["REDIS_URL"]) == password
    assert result.imported_keys == ("REDIS_PASSWORD", "REDIS_URL")


def test_redis_seed_url_uses_source_password_when_url_is_missing(tmp_path: Path) -> None:
    source_root, target_root = _roots(tmp_path)
    existing_password = "fake-source-redis-password"
    generated_password = "fake-generated-redis-password"
    (source_root / ".env.local").write_text(
        f"REDIS_PASSWORD={existing_password}\n",
        encoding="utf-8",
    )
    (target_root / ".env").write_text("# target\n", encoding="utf-8")

    migrate_configuration(
        _request(
            source_root,
            target_root,
            seed_values=_redis_seed_pair(generated_password),
        )
    )

    parsed = dotenv_values(target_root / ".env", interpolate=False)
    assert parsed["REDIS_PASSWORD"] == existing_password
    assert config_migration.redis_password_from_url(parsed["REDIS_URL"]) == existing_password
    assert generated_password not in (target_root / ".env").read_text(encoding="utf-8")


def test_redis_seed_password_uses_source_authenticated_url_when_missing(tmp_path: Path) -> None:
    source_root, target_root = _roots(tmp_path)
    existing_password = "fake existing/+ password"
    existing_url = _redis_seed_pair(existing_password)["REDIS_URL"]
    generated_password = "fake-generated-redis-password"
    (source_root / ".env.local").write_text(f"REDIS_URL={existing_url}\n", encoding="utf-8")
    (target_root / ".env").write_text("# target\n", encoding="utf-8")

    migrate_configuration(
        _request(
            source_root,
            target_root,
            seed_values=_redis_seed_pair(generated_password),
        )
    )

    parsed = dotenv_values(target_root / ".env", interpolate=False)
    assert parsed["REDIS_URL"] == existing_url
    assert parsed["REDIS_PASSWORD"] == existing_password
    assert generated_password not in (target_root / ".env").read_text(encoding="utf-8")


def test_redis_seed_pair_preserves_existing_values(tmp_path: Path) -> None:
    source_root, target_root = _roots(tmp_path)
    existing_password = "fake-existing-redis-password"
    existing_url = _redis_seed_pair(existing_password)["REDIS_URL"]
    original = f"REDIS_PASSWORD={existing_password}\nREDIS_URL={existing_url}\n"
    (source_root / ".env.local").write_text("", encoding="utf-8")
    (target_root / ".env").write_text(original, encoding="utf-8")

    result = migrate_configuration(
        _request(
            source_root,
            target_root,
            seed_values=_redis_seed_pair("fake-generated-redis-password"),
        )
    )

    assert (target_root / ".env").read_text(encoding="utf-8") == original
    assert result.imported_keys == ()


def test_redis_seed_pair_preserves_passwordless_url_and_adds_generated_password(
    tmp_path: Path,
) -> None:
    source_root, target_root = _roots(tmp_path)
    passwordless_url = "redis://localhost:6379/0"
    generated_password = "fake-generated-redis-password"
    (source_root / ".env.local").write_text("", encoding="utf-8")
    (target_root / ".env").write_text(f"REDIS_URL={passwordless_url}\n", encoding="utf-8")

    result = migrate_configuration(
        _request(
            source_root,
            target_root,
            seed_values=_redis_seed_pair(generated_password),
        )
    )

    parsed = dotenv_values(target_root / ".env", interpolate=False)
    assert parsed["REDIS_URL"] == _redis_seed_pair(generated_password)["REDIS_URL"]
    assert parsed["REDIS_PASSWORD"] == generated_password
    assert parsed["ONCALL_LEGACY_REDIS_URL"] == passwordless_url
    assert result.imported_keys == (
        "REDIS_PASSWORD",
        "ONCALL_LEGACY_REDIS_URL",
        "REDIS_URL",
    )


def test_redis_seed_pair_uses_current_password_to_authenticate_current_passwordless_url(
    tmp_path: Path,
) -> None:
    source_root, target_root = _roots(tmp_path)
    current_password = "fake-current-redis-password"
    passwordless_url = "redis://localhost:6379/0"
    generated_password = "fake-generated-redis-password"
    (source_root / ".env.local").write_text("", encoding="utf-8")
    (target_root / ".env").write_text(
        f"REDIS_PASSWORD={current_password}\nREDIS_URL={passwordless_url}\n",
        encoding="utf-8",
    )

    result = migrate_configuration(
        _request(
            source_root,
            target_root,
            seed_values=_redis_seed_pair(generated_password),
        )
    )

    parsed = dotenv_values(target_root / ".env", interpolate=False)
    assert parsed["REDIS_PASSWORD"] == current_password
    assert parsed["REDIS_URL"] == _redis_seed_pair(current_password)["REDIS_URL"]
    assert parsed["ONCALL_LEGACY_REDIS_URL"] == passwordless_url
    assert result.imported_keys == (
        "ONCALL_LEGACY_REDIS_URL",
        "REDIS_URL",
    )


def test_repeated_redis_seed_migration_keeps_first_pair_unchanged(tmp_path: Path) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("", encoding="utf-8")
    (target_root / ".env").write_text("# target\n", encoding="utf-8")
    first_password = "fake-first-generated-password"

    migrate_configuration(
        _request(source_root, target_root, seed_values=_redis_seed_pair(first_password))
    )
    after_first = (target_root / ".env").read_text(encoding="utf-8")
    second = migrate_configuration(
        _request(
            source_root,
            target_root,
            timestamp="20260720T120001Z",
            seed_values=_redis_seed_pair("fake-second-generated-password"),
        )
    )

    parsed = dotenv_values(target_root / ".env", interpolate=False)
    assert (target_root / ".env").read_text(encoding="utf-8") == after_first
    assert parsed["REDIS_PASSWORD"] == first_password
    assert config_migration.redis_password_from_url(parsed["REDIS_URL"]) == first_password
    assert second.imported_keys == ()


@pytest.mark.parametrize(
    ("encoded", "decoded"),
    (
        ("fake%40value", "fake@value"),
        ("fake%23value", "fake#value"),
        ("fake%24value", "fake$value"),
        ("fake%20value", "fake value"),
        ("fake%27quote%5Ctail", "fake'quote\\tail"),
    ),
)
def test_redis_password_helper_strictly_percent_decodes_utf8(encoded: str, decoded: str) -> None:
    redis_url = f"redis://:{encoded}@redis.invalid:6379/0"

    assert config_migration.redis_password_from_url(redis_url) == decoded


@pytest.mark.parametrize("encoded", ("fake%", "fake%2", "fake%GG", "%FF", "%00", "%0A"))
def test_redis_password_helper_rejects_invalid_or_unsafe_encoding(encoded: str) -> None:
    redis_url = f"redis://:{encoded}@redis.invalid:6379/0"

    with pytest.raises(ValueError, match="Redis password"):
        config_migration.redis_password_from_url(redis_url)


@pytest.mark.parametrize("control", ("\x00", "\r", "\n"))
def test_redis_password_helper_rejects_literal_control_characters(control: str) -> None:
    redis_url = f"redis://:fake{control}password@redis.invalid:6379/0"

    with pytest.raises(ValueError, match="Redis password"):
        config_migration.redis_password_from_url(redis_url)


def test_derived_redis_password_roundtrips_through_safe_dotenv_serialization(
    tmp_path: Path,
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("", encoding="utf-8")
    (target_root / ".env").write_text("# target\n", encoding="utf-8")
    encoded_password = "fake%40%23%24%20%27%5Ctail"
    decoded_password = "fake@#$ '\\tail"
    redis_url = f"redis://:{encoded_password}@redis.invalid:6379/0"

    migrate_configuration(_request(source_root, target_root, seed_values={"REDIS_URL": redis_url}))

    migrated = (target_root / ".env").read_text(encoding="utf-8")
    parsed = dotenv_values(stream=StringIO(migrated), interpolate=False)
    assert parsed["REDIS_URL"] == redis_url
    assert parsed["REDIS_PASSWORD"] == decoded_password


@pytest.fixture(scope="module")
def docker_compose_command() -> tuple[str, str]:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is unavailable")
    availability = subprocess.run(
        (docker, "compose", "version"),
        capture_output=True,
        text=True,
        check=False,
    )
    if availability.returncode != 0:
        pytest.skip("Docker Compose is unavailable")
    return docker, "compose"


def _run_compose_config(
    tmp_path: Path,
    docker_compose_command: tuple[str, str],
    env_file: Path,
    *,
    environment: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    compose_file = tmp_path / "compose.yaml"
    compose_file.write_text(
        "services:\n"
        "  probe:\n"
        "    image: scratch\n"
        "    env_file:\n"
        f"      - {json.dumps(str(env_file))}\n",
        encoding="utf-8",
    )
    fake_home = tmp_path / "fake-compose-home"
    fake_home.mkdir(exist_ok=True)
    process_environment = {
        "HOME": str(fake_home),
        "PATH": os.defpath,
        **(environment or {}),
    }
    return subprocess.run(
        (
            *docker_compose_command,
            "--env-file",
            str(env_file),
            "-f",
            str(compose_file),
            "config",
            "--format",
            "json",
        ),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=check,
        env=process_environment,
    )


def _compose_service_password(
    tmp_path: Path,
    docker_compose_command: tuple[str, str],
    env_file: Path,
    *,
    environment: dict[str, str] | None = None,
) -> str:
    compose_result = _run_compose_config(
        tmp_path,
        docker_compose_command,
        env_file,
        environment=environment,
    )
    compose_config = json.loads(compose_result.stdout)
    return compose_config["services"]["probe"]["environment"]["REDIS_PASSWORD"]


@pytest.mark.parametrize(
    ("encoded_password", "decoded_password"),
    (
        ("fake%5Cvalue", "fake\\value"),
        ("fake%27value", "fake'value"),
        ("fake%24value", "fake$value"),
        ("fake%27%24VAR", "fake'$VAR"),
        ("fake%20space%23hash", "fake space#hash"),
        ("fake%5C%5C%27combined%24", "fake\\\\'combined$"),
        ("fake%5C%27inert%24-", "fake\\'inert$-"),
        (
            "fake%5C%5C%5C%27quoted%22value%24",
            r"""fake\\\'quoted"value$""",
        ),
    ),
)
def test_migrated_redis_password_matches_python_and_compose_consumers(
    tmp_path: Path,
    docker_compose_command: tuple[str, str],
    encoded_password: str,
    decoded_password: str,
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("", encoding="utf-8")
    (target_root / ".env").write_text("# fake target\n", encoding="utf-8")
    redis_url = f"redis://:{encoded_password}@redis.invalid:6379/0"

    migrate_configuration(_request(source_root, target_root, seed_values={"REDIS_URL": redis_url}))

    migrated_env = target_root / ".env"
    parsed = dotenv_values(dotenv_path=migrated_env, interpolate=False)
    assert parsed["REDIS_PASSWORD"] == decoded_password

    compose_password = _compose_service_password(
        tmp_path,
        docker_compose_command,
        migrated_env,
    )

    assert compose_password.replace("$$", "$") == decoded_password


@pytest.mark.parametrize("value", ("fake\\'$VAR", "fake\\'${HOME}"))
def test_serializer_rejects_compose_interpolation_in_double_quoted_fallback(
    value: str,
) -> None:
    with pytest.raises(ValueError, match="cannot be represented safely") as raised:
        config_migration._serialize_dotenv_value(value)

    assert value not in str(raised.value)


@pytest.mark.parametrize(
    ("expression", "variable_name", "fake_value"),
    (
        ("$VAR", "VAR", "fake-compose-var"),
        ("${HOME}", "HOME", "fake-compose-home-value"),
    ),
)
def test_compose_oracle_interpolates_unsafe_double_quoted_values(
    tmp_path: Path,
    docker_compose_command: tuple[str, str],
    expression: str,
    variable_name: str,
    fake_value: str,
) -> None:
    original = f"fake\\'{expression}"
    escaped = original.replace("\\", "\\\\").replace('"', '\\"')
    env_file = tmp_path / "unsafe.env"
    env_file.write_text(f'REDIS_PASSWORD="{escaped}"\n', encoding="utf-8")

    compose_password = _compose_service_password(
        tmp_path,
        docker_compose_command,
        env_file,
        environment={variable_name: fake_value},
    )

    assert compose_password == f"fake\\'{fake_value}"
    assert compose_password != original


@pytest.mark.parametrize(
    "dollar_token",
    ("$$", "${9VAR}", "${VAR:=x}", "${VAR/foo/bar}", "${VAR"),
)
def test_migration_rejects_every_compose_dollar_token_in_double_quoted_fallback(
    tmp_path: Path,
    docker_compose_command: tuple[str, str],
    dollar_token: str,
) -> None:
    value = f"fake\\'{dollar_token}"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    oracle_env = tmp_path / "unsafe-token.env"
    oracle_env.write_text(f'REDIS_PASSWORD="{escaped}"\n', encoding="utf-8")

    compose_result = _run_compose_config(
        tmp_path,
        docker_compose_command,
        oracle_env,
        environment={"VAR": "fake-compose-var"},
        check=False,
    )
    if compose_result.returncode == 0:
        compose_config = json.loads(compose_result.stdout)
        compose_password = compose_config["services"]["probe"]["environment"]["REDIS_PASSWORD"]
        assert compose_password.replace("$$", "$") != value
    else:
        assert compose_result.stderr

    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be represented safely") as raised:
        migrate_configuration(
            _request(
                source_root,
                target_root,
                seed_values={"REDIS_PASSWORD": value},
            )
        )

    assert str(raised.value) == ("dotenv value cannot be represented safely for Python and Compose")
    assert value not in str(raised.value)


def test_atomic_replace_failure_preserves_target_bytes_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("SOURCE_ONLY=fake-source-only\n", encoding="utf-8")
    original = b"# exact original bytes\r\nTARGET_ONLY=fake-target\r\n"
    (target_root / ".env").write_bytes(original)

    original_replace = config_migration.os.replace

    def fail_replace(source: Path | str, target: Path | str) -> None:
        if Path(target) == target_root / ".env":
            raise OSError("fake atomic replace failure")
        original_replace(source, target)

    monkeypatch.setattr(config_migration.os, "replace", fail_replace)

    with pytest.raises(OSError, match="fake atomic replace failure"):
        migrate_configuration(_request(source_root, target_root))

    assert (target_root / ".env").read_bytes() == original
    assert not (target_root / ".env.tmp").exists()


def test_dotenv_text_is_never_executed_and_target_non_assignments_are_preserved(
    tmp_path: Path,
) -> None:
    source_root, target_root = _roots(tmp_path)
    sentinel = tmp_path / "must-not-exist"
    target_original = (
        "# literal target comment\n"
        f"not an assignment $(touch {sentinel})\n"
        "export TARGET_EXPORT=fake-target-export\n"
        "TARGET_LITERAL='$(printf fake-target-shell)'\n"
    )
    (target_root / ".env").write_text(target_original, encoding="utf-8")
    (source_root / ".env.local").write_text(
        f"SOURCE_LITERAL=$(touch {sentinel})\n"
        'export SOURCE_EXPORT="fake-source-export; printf ignored"\n'
        "invalid-key=fake-invalid\n",
        encoding="utf-8",
    )

    migrate_configuration(_request(source_root, target_root))

    migrated = (target_root / ".env").read_text(encoding="utf-8")
    parsed = dotenv_values(stream=StringIO(migrated), interpolate=False)
    assert migrated.startswith(target_original)
    assert parsed["SOURCE_LITERAL"] == f"$(touch {sentinel})"
    assert parsed["SOURCE_EXPORT"] == "fake-source-export; printf ignored"
    assert parsed["invalid-key"] == "fake-invalid"
    assert not sentinel.exists()


def test_snapshot_rejects_symlinks_without_copying_external_content(tmp_path: Path) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("SOURCE_ONLY=fake-source\n", encoding="utf-8")
    observability = source_root / "manifest" / "observability"
    observability.mkdir(parents=True)
    external = tmp_path / "outside-secret"
    external.write_text("fake-external-secret-must-not-copy", encoding="utf-8")
    (observability / "external-link").symlink_to(external)
    target_env = target_root / ".env"
    target_env.write_text("TARGET_ONLY=fake-target\n", encoding="utf-8")

    with pytest.raises(ValueError, match="symbolic link"):
        migrate_configuration(_request(source_root, target_root))

    assert target_env.read_text(encoding="utf-8") == "TARGET_ONLY=fake-target\n"
    assert not (target_root / ".energyagent" / "migration" / TIMESTAMP).exists()


def test_snapshot_rejects_intermediate_symlink_that_escapes_source_root(tmp_path: Path) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("SOURCE_ONLY=fake-source\n", encoding="utf-8")
    outside_manifest = tmp_path / "outside-manifest"
    outside_observability = outside_manifest / "observability"
    outside_observability.mkdir(parents=True)
    outside_secret = outside_observability / "external-secret"
    outside_secret.write_text("fake-external-secret-must-not-copy", encoding="utf-8")
    (source_root / "manifest").symlink_to(outside_manifest, target_is_directory=True)
    target_env = target_root / ".env"
    original = b"TARGET_ONLY=fake-target\n"
    target_env.write_bytes(original)

    with pytest.raises(ValueError, match="symbolic link"):
        migrate_configuration(_request(source_root, target_root))

    assert target_env.read_bytes() == original
    assert outside_secret.read_text(encoding="utf-8") == "fake-external-secret-must-not-copy"
    assert not (target_root / ".energyagent" / "migration" / TIMESTAMP).exists()


@pytest.mark.parametrize("swap_level", ("root", "component", "leaf"))
def test_source_swap_after_validation_never_copies_outside_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, swap_level: str
) -> None:
    source_root, target_root = _roots(tmp_path)
    source_env = source_root / ".env.local"
    source_env.write_text("SOURCE_ONLY=fake-source\n", encoding="utf-8")
    observability = source_root / "manifest" / "observability"
    observability.mkdir(parents=True)
    (observability / "inside").write_text("fake-inside", encoding="utf-8")
    outside_root = tmp_path / "outside-root"
    outside_root.mkdir()
    (outside_root / ".env.local").write_text("OUTSIDE_ONLY=fake-outside\n", encoding="utf-8")
    outside_manifest = outside_root / "manifest"
    (outside_manifest / "observability").mkdir(parents=True)
    (outside_manifest / "observability" / "outside").write_text(
        "fake-outside-must-not-copy", encoding="utf-8"
    )
    outside_leaf = outside_root / "outside-leaf"
    outside_leaf.write_text("OUTSIDE_ONLY=fake-outside-leaf\n", encoding="utf-8")
    target_env = target_root / ".env"
    original = b"TARGET_ONLY=fake-target\n"
    target_env.write_bytes(original)
    original_validate = config_migration._validate_inputs

    def validate_then_swap(request: MigrationRequest) -> None:
        original_validate(request)
        if swap_level == "root":
            source_root.rename(tmp_path / "held-source-root")
            source_root.symlink_to(outside_root, target_is_directory=True)
        elif swap_level == "component":
            manifest = source_root / "manifest"
            manifest.rename(source_root / "held-manifest")
            manifest.symlink_to(outside_manifest, target_is_directory=True)
        else:
            source_env.rename(source_root / ".env.local-held")
            source_env.symlink_to(outside_leaf)

    monkeypatch.setattr(config_migration, "_validate_inputs", validate_then_swap)

    with pytest.raises(ValueError, match="symbolic link"):
        migrate_configuration(_request(source_root, target_root))

    assert target_env.read_bytes() == original
    assert not (target_root / ".energyagent" / "migration" / TIMESTAMP).exists()


def test_successful_replace_has_no_fallible_post_replace_chmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("SOURCE_ONLY=fake-source\n", encoding="utf-8")
    target_env = target_root / ".env"
    target_env.write_text("TARGET_ONLY=fake-target\n", encoding="utf-8")
    original_chmod = Path.chmod
    post_replace_calls: list[Path] = []

    def reject_target_chmod(path: Path, *args: object, **kwargs: object) -> None:
        if path == target_env:
            post_replace_calls.append(path)
            raise OSError("post-replace chmod must not run")
        original_chmod(path, *args, **kwargs)

    monkeypatch.setattr(Path, "chmod", reject_target_chmod)

    migrate_configuration(_request(source_root, target_root))

    assert post_replace_calls == []
    assert "SOURCE_ONLY=fake-source\n" in target_env.read_text(encoding="utf-8")
    assert stat.S_IMODE(target_env.stat().st_mode) == 0o600


def test_copy_fault_removes_staging_and_never_publishes_partial_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("SOURCE_ONLY=fake-source\n", encoding="utf-8")
    target_env = target_root / ".env"
    original = b"TARGET_ONLY=fake-target\n"
    target_env.write_bytes(original)

    def fail_copy(*_args: object, **_kwargs: object) -> None:
        raise OSError("fake snapshot copy fault")

    monkeypatch.setattr(config_migration, "_copy_snapshot_path", fail_copy)

    with pytest.raises(OSError, match="snapshot copy fault"):
        migrate_configuration(_request(source_root, target_root))

    migration_root = target_root / ".energyagent" / "migration"
    assert target_env.read_bytes() == original
    assert not (migration_root / TIMESTAMP).exists()
    assert list(migration_root.glob(".staging-*")) == []


@pytest.mark.parametrize("fault", ("write", "fsync"))
def test_manifest_fault_removes_staging_and_never_publishes_partial_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("SOURCE_ONLY=fake-source\n", encoding="utf-8")
    target_env = target_root / ".env"
    original = b"TARGET_ONLY=fake-target\n"
    target_env.write_bytes(original)
    if fault == "write":
        original_write = config_migration._write_private_file

        def fail_manifest_write(path: Path, content: bytes) -> None:
            if path.name == "manifest.json":
                raise OSError("fake manifest write fault")
            original_write(path, content)

        monkeypatch.setattr(config_migration, "_write_private_file", fail_manifest_write)
    else:
        original_fsync = config_migration.os.fsync

        def fail_manifest_fsync(descriptor: int) -> None:
            try:
                descriptor_path = config_migration.os.readlink(f"/proc/self/fd/{descriptor}")
            except OSError:
                descriptor_path = ""
            if descriptor_path.endswith("/manifest.json"):
                raise OSError("fake manifest fsync fault")
            original_fsync(descriptor)

        monkeypatch.setattr(config_migration.os, "fsync", fail_manifest_fsync)

    with pytest.raises(OSError, match=f"manifest {fault} fault"):
        migrate_configuration(_request(source_root, target_root))

    migration_root = target_root / ".energyagent" / "migration"
    assert target_env.read_bytes() == original
    assert not (migration_root / TIMESTAMP).exists()
    assert list(migration_root.glob(".staging-*")) == []


def test_staging_directory_fsync_fault_never_publishes_vault(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("SOURCE_ONLY=fake-source\n", encoding="utf-8")
    target_env = target_root / ".env"
    original = b"TARGET_ONLY=fake-target\n"
    target_env.write_bytes(original)

    def fail_staging_fsync(path: Path) -> None:
        if path.name.startswith(".staging-"):
            raise OSError("fake staging directory fsync fault")

    monkeypatch.setattr(config_migration, "_fsync_directory", fail_staging_fsync, raising=False)

    with pytest.raises(OSError, match="staging directory fsync fault"):
        migrate_configuration(_request(source_root, target_root))

    migration_root = target_root / ".energyagent" / "migration"
    assert target_env.read_bytes() == original
    assert not (migration_root / TIMESTAMP).exists()
    assert list(migration_root.glob(".staging-*")) == []


def test_vault_rename_fault_cleans_staging_and_preserves_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("SOURCE_ONLY=fake-source\n", encoding="utf-8")
    target_env = target_root / ".env"
    original = b"TARGET_ONLY=fake-target\n"
    target_env.write_bytes(original)
    final_vault = target_root / ".energyagent" / "migration" / TIMESTAMP
    original_replace = config_migration.os.replace

    def fail_vault_rename(source: Path | str, target: Path | str) -> None:
        if Path(target) == final_vault and Path(source).name.startswith(".staging-"):
            raise OSError("fake vault rename fault")
        original_replace(source, target)

    monkeypatch.setattr(config_migration.os, "replace", fail_vault_rename)

    with pytest.raises(OSError, match="vault rename fault"):
        migrate_configuration(_request(source_root, target_root))

    assert target_env.read_bytes() == original
    assert not final_vault.exists()
    assert list(final_vault.parent.glob(".staging-*")) == []


def test_migration_root_fsync_fault_keeps_complete_vault_and_preserves_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("SOURCE_ONLY=fake-source\n", encoding="utf-8")
    target_env = target_root / ".env"
    original = b"TARGET_ONLY=fake-target\n"
    target_env.write_bytes(original)
    migration_root = target_root / ".energyagent" / "migration"

    def fail_migration_root_fsync(path: Path) -> None:
        if path == migration_root:
            raise OSError("fake migration root fsync fault")

    monkeypatch.setattr(
        config_migration, "_fsync_directory", fail_migration_root_fsync, raising=False
    )

    with pytest.raises(OSError, match="migration root fsync fault"):
        migrate_configuration(_request(source_root, target_root))

    final_vault = migration_root / TIMESTAMP
    assert target_env.read_bytes() == original
    assert (final_vault / "manifest.json").is_file()
    assert list(migration_root.glob(".staging-*")) == []


def test_target_directory_fsync_fault_restores_original_bytes_and_cleans_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text("SOURCE_ONLY=fake-source\n", encoding="utf-8")
    target_env = target_root / ".env"
    original = b"# exact original\r\nTARGET_ONLY=fake-target\r\n"
    target_env.write_bytes(original)
    original_fsync_directory = getattr(config_migration, "_fsync_directory", lambda _path: None)
    target_faults = 0

    def fail_target_fsync_once(path: Path) -> None:
        nonlocal target_faults
        if path == target_root and target_env.read_bytes() != original and target_faults == 0:
            target_faults += 1
            raise OSError("fake target directory fsync fault")
        original_fsync_directory(path)

    monkeypatch.setattr(config_migration, "_fsync_directory", fail_target_fsync_once, raising=False)

    with pytest.raises(OSError, match="target directory fsync fault"):
        migrate_configuration(_request(source_root, target_root))

    assert target_faults == 1
    assert target_env.read_bytes() == original
    assert not (target_root / ".env.tmp").exists()
    assert not (target_root / ".env.restore.tmp").exists()
    assert (target_root / ".energyagent" / "migration" / TIMESTAMP / "manifest.json").is_file()


def test_missing_roots_raise_clear_errors(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    target_root.mkdir()

    with pytest.raises(FileNotFoundError, match="source root"):
        migrate_configuration(_request(source_root, target_root))

    source_root.mkdir()
    target_root.rmdir()
    with pytest.raises(FileNotFoundError, match="target root"):
        migrate_configuration(_request(source_root, target_root))


@pytest.mark.parametrize("timestamp", ("", ".", "..", "../escape", "nested/name", r"nested\name"))
def test_timestamp_must_be_a_safe_filename_component(tmp_path: Path, timestamp: str) -> None:
    source_root, target_root = _roots(tmp_path)

    with pytest.raises(ValueError, match="timestamp"):
        migrate_configuration(_request(source_root, target_root, timestamp=timestamp))

    assert not (target_root / ".energyagent").exists()


def _load_cli_module(project_root: Path) -> types.ModuleType:
    script_path = project_root / "scripts" / "migrate_oncall_config.py"
    spec = importlib.util.spec_from_file_location("test_migrate_oncall_config_cli", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_import_is_side_effect_free_and_stdout_never_contains_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    monkeypatch.delitem(sys.modules, "app.config", raising=False)

    cli = _load_cli_module(project_root)

    assert "app.config" not in sys.modules

    class FakeSettings:
        def __init__(self) -> None:
            raise AssertionError("CLI seed must not construct Settings")

    fake_config = types.ModuleType("app.config")
    fake_config.Settings = FakeSettings
    monkeypatch.setitem(sys.modules, "app.config", fake_config)
    monkeypatch.setenv("REDIS_URL", "redis://:fake-ambient-password@redis.invalid:6379/9")
    monkeypatch.setenv(
        "ENERGY_AGENT_CONFIG_FILES",
        str(tmp_path / "fake-ambient-config.yaml"),
    )
    monkeypatch.setattr(cli, "_utc_timestamp", lambda: TIMESTAMP)
    generated_password = "fake-generated-cli-password"
    monkeypatch.setattr(cli.secrets, "token_urlsafe", lambda length: generated_password)
    source_root, target_root = _roots(tmp_path)
    (source_root / ".env.local").write_text(
        "DEEPSEEK_API_KEY=fake-cli-source-key\n" "APP_NAME=LegacyCliBrand\n",
        encoding="utf-8",
    )
    (target_root / ".env").write_text("DEEPSEEK_API_KEY=fake-cli-target-key\n", encoding="utf-8")

    exit_code = cli.main(["--source", str(source_root), "--target", str(target_root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.splitlines() == [
        f"vault={target_root / '.energyagent' / 'migration' / TIMESTAMP}",
        "imported_keys=5",
        "conflicted_keys=DEEPSEEK_API_KEY",
    ]
    for fake_value in (
        "fake-ambient-password",
        "fake-cli-source-key",
        "fake-cli-target-key",
        "LegacyCliBrand",
        generated_password,
    ):
        assert fake_value not in captured.out
        assert fake_value not in captured.err
    assert captured.err == ""


def test_cli_seed_is_safe_and_independent_of_environment_and_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli_module(Path(__file__).resolve().parents[1])

    class FakeSettings:
        def __init__(self) -> None:
            raise AssertionError("CLI seed must not construct Settings")

    fake_config = types.ModuleType("app.config")
    fake_config.Settings = FakeSettings
    monkeypatch.setitem(sys.modules, "app.config", fake_config)
    monkeypatch.setenv("REDIS_URL", "redis://:fake-ambient-password@redis.invalid:6379/9")
    monkeypatch.setenv(
        "ENERGY_AGENT_CONFIG_FILES",
        str(tmp_path / "fake-ambient-config.yaml"),
    )
    target_root = tmp_path / "target"
    target_root.mkdir()
    different_cwd = tmp_path / "different-cwd"
    different_cwd.mkdir()
    monkeypatch.chdir(different_cwd)
    generated_password = "fake generated/+?#'\\tail"
    requested_lengths: list[int] = []

    def fake_token_urlsafe(length: int) -> str:
        requested_lengths.append(length)
        return generated_password

    monkeypatch.setattr(cli.secrets, "token_urlsafe", fake_token_urlsafe)

    first_seeds = cli._redis_seed_values(target_root)
    second_seeds = cli._redis_seed_values(target_root)

    expected = _redis_seed_pair(generated_password)
    assert first_seeds == expected
    assert second_seeds == expected
    assert config_migration.redis_password_from_url(first_seeds["REDIS_URL"]) == generated_password
    assert requested_lengths == [32, 32]
    assert all(length >= 32 for length in requested_lengths)
    captured = capsys.readouterr()
    assert generated_password not in captured.out
    assert generated_password not in captured.err


def test_cli_main_passes_target_to_seed_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli_module(Path(__file__).resolve().parents[1])
    source_root, target_root = _roots(tmp_path)
    seed_targets: list[Path] = []
    requests: list[MigrationRequest] = []

    def fake_seed_values(target: Path) -> dict[str, str]:
        seed_targets.append(target)
        return _redis_seed_pair("fake-cli-generated-password")

    def fake_migrate(request: MigrationRequest) -> types.SimpleNamespace:
        requests.append(request)
        return types.SimpleNamespace(
            vault_dir=target_root / ".energyagent" / "migration" / TIMESTAMP,
            imported_keys=(),
            conflicted_keys=(),
        )

    monkeypatch.setattr(cli, "_redis_seed_values", fake_seed_values)
    monkeypatch.setattr(cli, "migrate_configuration", fake_migrate)
    monkeypatch.setattr(cli, "_utc_timestamp", lambda: TIMESTAMP)

    exit_code = cli.main(["--source", str(source_root), "--target", str(target_root)])

    assert exit_code == 0
    assert seed_targets == [target_root]
    assert requests[0].target_root == target_root
    assert requests[0].seed_values == _redis_seed_pair("fake-cli-generated-password")
    assert requests[0].canonical_values == {"APP_NAME": "EnergyAgent"}
    assert capsys.readouterr().out.splitlines() == [
        f"vault={target_root / '.energyagent' / 'migration' / TIMESTAMP}",
        "imported_keys=0",
        "conflicted_keys=",
    ]


def test_cli_preserves_existing_target_redis_url_without_printing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli_module(Path(__file__).resolve().parents[1])

    class FakeSettings:
        def __init__(self) -> None:
            raise AssertionError("CLI seed must not construct Settings")

    fake_config = types.ModuleType("app.config")
    fake_config.Settings = FakeSettings
    monkeypatch.setitem(sys.modules, "app.config", fake_config)
    source_root, target_root = _roots(tmp_path)
    existing_url = "redis://:fake-target-password@redis.invalid:6379/7"
    (source_root / ".env.local").write_text("", encoding="utf-8")
    (target_root / ".env").write_text(f"REDIS_URL={existing_url}\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_utc_timestamp", lambda: TIMESTAMP)
    generated_password = "fake-generated-cli-password"
    monkeypatch.setattr(cli.secrets, "token_urlsafe", lambda length: generated_password)

    exit_code = cli.main(["--source", str(source_root), "--target", str(target_root)])

    captured = capsys.readouterr()
    assert exit_code == 0
    parsed = dotenv_values(target_root / ".env", interpolate=False)
    assert parsed["REDIS_URL"] == existing_url
    assert parsed["REDIS_PASSWORD"] == "fake-target-password"
    assert "fake-target-password" not in captured.out
    assert generated_password not in captured.out
    assert generated_password not in captured.err
    assert captured.err == ""


def test_cli_sanitizes_unexpected_migration_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli_module(Path(__file__).resolve().parents[1])
    source_root, target_root = _roots(tmp_path)
    secret = "fake-migration-secret"
    monkeypatch.setattr(
        cli,
        "_redis_seed_values",
        lambda target=None: _redis_seed_pair("fake-generated-redis-password"),
    )
    monkeypatch.setattr(
        cli,
        "migrate_configuration",
        lambda request: (_ for _ in ()).throw(RuntimeError(secret)),
    )

    exit_code = cli.main(["--source", str(source_root), "--target", str(target_root)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "migration failed: RuntimeError\n"
    assert secret not in captured.out
    assert secret not in captured.err


def test_cli_sanitizes_seed_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _load_cli_module(Path(__file__).resolve().parents[1])
    source_root, target_root = _roots(tmp_path)
    secret = "fake-seed-secret"

    def fail_seed(length: int) -> str:
        raise LookupError(secret)

    monkeypatch.setattr(cli.secrets, "token_urlsafe", fail_seed)

    exit_code = cli.main(["--source", str(source_root), "--target", str(target_root)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "migration failed: LookupError\n"
    assert secret not in captured.out
    assert secret not in captured.err


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(23)])
def test_cli_does_not_swallow_process_control_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    cli = _load_cli_module(Path(__file__).resolve().parents[1])
    source_root, target_root = _roots(tmp_path)
    monkeypatch.setattr(
        cli,
        "_redis_seed_values",
        lambda target=None: _redis_seed_pair("fake-generated-redis-password"),
    )
    monkeypatch.setattr(
        cli,
        "migrate_configuration",
        lambda request: (_ for _ in ()).throw(error),
    )

    with pytest.raises(type(error)):
        cli.main(["--source", str(source_root), "--target", str(target_root)])
