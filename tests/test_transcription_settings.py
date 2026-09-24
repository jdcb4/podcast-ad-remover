import sqlite3
import sys
from types import SimpleNamespace

import pytest

from app.core import transcription_settings as ts
from app.core.ai_services import Transcriber
from app.infra.database import init_db, get_db_connection, CUDA_SETTINGS_MIGRATION


def test_cpu_precision_capabilities_and_fallback(monkeypatch):
    monkeypatch.setattr(ts, "supported_compute_types", lambda: ["float32", "int8"])
    assert ts.cpu_compute_type({}) == "float32"
    assert ts.cpu_compute_type({"whisper_compute_type": "int8"}) == "int8"
    assert ts.cpu_compute_type({"whisper_compute_type": "float16"}) == "float32"
    assert ts.provenance_matches({}, {})
    assert not ts.provenance_matches({}, {"whisper_compute_type": "int8"})


def test_precision_change_reloads_model(monkeypatch):
    monkeypatch.setattr(ts, "supported_compute_types", lambda: ["float32", "int8"])
    calls = []
    def model(name, **kwargs):
        calls.append((name, kwargs)); return object()
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=model))
    transcriber = Transcriber()
    transcriber.load_model({"whisper_model": "tiny"})
    transcriber.load_model({"whisper_model": "tiny"})
    transcriber.load_model({"whisper_model": "tiny", "whisper_compute_type": "int8"})
    assert [call[1]["compute_type"] for call in calls] == ["float32", "int8"]


def test_migration_is_backed_up_and_idempotent(isolated_data_dir, monkeypatch):
    from app.infra import database
    with monkeypatch.context() as patch:
        patch.setattr(database, "FORMAL_MIGRATIONS", [(v, sql) for v, sql in database.FORMAL_MIGRATIONS if v != CUDA_SETTINGS_MIGRATION])
        init_db()
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET whisper_model='small'")
        conn.commit()
    init_db(); init_db()
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM app_settings").fetchone()
        assert (row['whisper_device'], row['whisper_compute_type'], row['whisper_model']) == ('cpu', 'float32', 'small')
        assert conn.execute("SELECT count(*) FROM schema_migrations WHERE version=?", (CUDA_SETTINGS_MIGRATION,)).fetchone()[0] == 1
    assert list((isolated_data_dir / 'db' / 'backups').glob('*.db')) or list(isolated_data_dir.rglob('*before-migration*.db'))
