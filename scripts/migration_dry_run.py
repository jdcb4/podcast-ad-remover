#!/usr/bin/env python3
"""Dry-run Podcast Ad Remover database migrations against a copy of a DB."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings
from app.infra.database import init_db


@dataclass(frozen=True)
class MigrationDryRunResult:
    source_db: Path
    copied_db: Path
    schema_versions: list[str]
    table_names: list[str]


def _read_schema_versions(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        try:
            rows = conn.execute("SELECT version FROM schema_migrations ORDER BY version").fetchall()
        except sqlite3.OperationalError:
            return []
    return [row[0] for row in rows]


def _read_table_names(db_path: Path) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    return [row[0] for row in rows]


def run_migration_dry_run(source_db: Path, data_dir: Path) -> MigrationDryRunResult:
    source_db = source_db.resolve()
    data_dir = data_dir.resolve()

    if not source_db.exists():
        raise FileNotFoundError(f"Database not found: {source_db}")
    if not source_db.is_file():
        raise ValueError(f"Database path is not a file: {source_db}")

    copied_db = data_dir / "db" / "podcasts.db"
    copied_db.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_db, copied_db)

    original_data_dir = settings.DATA_DIR
    try:
        settings.DATA_DIR = str(data_dir)
        Path(settings.PODCASTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.FEEDS_DIR).mkdir(parents=True, exist_ok=True)
        Path(settings.MODELS_DIR).mkdir(parents=True, exist_ok=True)
        init_db()
    finally:
        settings.DATA_DIR = original_data_dir

    return MigrationDryRunResult(
        source_db=source_db,
        copied_db=copied_db,
        schema_versions=_read_schema_versions(copied_db),
        table_names=_read_table_names(copied_db),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Copy a podcasts.db file to a temporary data directory and run startup migrations there."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(settings.DB_PATH),
        help="Path to the source podcasts.db. Defaults to the configured DATA_DIR database.",
    )
    parser.add_argument(
        "--keep-copy",
        type=Path,
        help="Optional data directory where the migrated dry-run copy should be kept for inspection.",
    )
    args = parser.parse_args(argv)

    if args.keep_copy:
        args.keep_copy.mkdir(parents=True, exist_ok=True)
        result = run_migration_dry_run(args.db_path, args.keep_copy)
        print(f"Dry-run migration succeeded. Migrated copy kept at: {result.copied_db}")
    else:
        with tempfile.TemporaryDirectory(prefix="podcast-ad-remover-migration-") as temp_dir:
            result = run_migration_dry_run(args.db_path, Path(temp_dir))
            print("Dry-run migration succeeded against a temporary copy.")

    print(f"Source database: {result.source_db}")
    print(f"Tables after migration: {', '.join(result.table_names)}")
    if result.schema_versions:
        print(f"Formal migrations applied: {', '.join(result.schema_versions)}")
    else:
        print("Formal migrations applied: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
