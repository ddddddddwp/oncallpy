"""Command-line entry point for the secret-safe OnCall configuration migration."""

from __future__ import annotations

import argparse
import secrets
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

from app.core.config_migration import (
    MigrationRequest,
    migrate_configuration,
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _redis_seed_values(target_root: Path) -> dict[str, str]:
    """Return a newly generated Redis credential pair without inspecting config."""
    _ = target_root
    password = secrets.token_urlsafe(32)
    encoded_password = quote(password, safe="")
    return {
        "REDIS_PASSWORD": password,
        "REDIS_URL": f"redis://:{encoded_password}@localhost:6379/0",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Migrate legacy OnCall configuration")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/root/code/oncall/OnCallAgent"),
        help="legacy OnCall source root",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path.cwd(),
        help="EnergyAgent target root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = migrate_configuration(
            MigrationRequest(
                source_root=args.source,
                target_root=args.target,
                timestamp=_utc_timestamp(),
                seed_values=_redis_seed_values(args.target),
                canonical_values={"APP_NAME": "EnergyAgent"},
            )
        )
    except Exception as exc:
        print(f"migration failed: {type(exc).__name__}", file=sys.stderr)
        return 1

    print(f"vault={result.vault_dir}")
    print(f"imported_keys={len(result.imported_keys)}")
    print(f"conflicted_keys={','.join(result.conflicted_keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
