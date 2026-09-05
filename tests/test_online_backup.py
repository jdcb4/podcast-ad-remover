import sqlite3
from pathlib import Path

import pytest

from app.core.config import settings
from app.infra.backup import backup_database
from app.infra.database import init_db
from scripts.migration_dry_run import run_migration_dry_run


def test_snapshot_includes_committed_wal_and_restores(isolated_data_dir, tmp_path):
    init_db()
    live = sqlite3.connect(settings.DB_PATH)
    try:
        live.execute('PRAGMA wal_autocheckpoint=0')
        live.execute('CREATE TABLE backup_probe (value TEXT)')
        live.execute("INSERT INTO backup_probe VALUES ('committed in WAL')")
        live.commit()
        snapshot = backup_database(settings.DB_PATH, tmp_path / 'snapshot.db')
        dry = run_migration_dry_run(Path(settings.DB_PATH), tmp_path / 'dry')
        for path in (snapshot, dry.copied_db):
            with sqlite3.connect(path) as restored:
                assert restored.execute('SELECT value FROM backup_probe').fetchone()[0] == 'committed in WAL'
                assert restored.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        with pytest.raises(FileExistsError):
            backup_database(settings.DB_PATH, snapshot)
        with pytest.raises(ValueError):
            backup_database(settings.DB_PATH, settings.DB_PATH)
    finally:
        live.close()


def test_startup_snapshot_precedes_legacy_schema_changes(isolated_data_dir):
    with sqlite3.connect(settings.DB_PATH) as conn:
        conn.execute('CREATE TABLE pre_upgrade_marker (id INTEGER)')
    init_db()
    backup = next((Path(settings.DATA_DIR) / 'backups').glob('*.db'))
    with sqlite3.connect(backup) as conn:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert names == {'pre_upgrade_marker'}
