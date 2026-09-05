"""Consistent SQLite snapshots, including committed WAL transactions."""
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path


def backup_database(source: str | Path, destination: str | Path) -> Path:
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("Backup destination must differ from the source database")
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwriting a database, including a previous recovery point.
    with destination.open('xb'):
        pass
    fd, temporary = tempfile.mkstemp(prefix='.snapshot-', dir=destination.parent)
    os.close(fd)
    try:
        with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as src:
            with closing(sqlite3.connect(temporary)) as dst:
                src.backup(dst, pages=256, sleep=0.05)
                if dst.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise RuntimeError('Snapshot integrity check failed')
        with open(temporary, 'r+b') as snapshot:
            os.fsync(snapshot.fileno())
        os.replace(temporary, destination)
        return destination
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        Path(temporary).unlink(missing_ok=True)
