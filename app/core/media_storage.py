"""Processed audio placement, stable URL mapping and resumable copy migration.

local_filename remains a logical legacy pathname so existing feeds never change.
media_files maps its relative URL key to the configured, identity-checked volume.
No network filesystem is used for SQLite, locks, transcripts or working files.
"""
import asyncio
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from uuid import uuid4

from filelock import FileLock, Timeout
from app.core.config import settings
from app.infra.database import get_db_connection


class StorageUnavailable(RuntimeError):
    pass


ACTIVE = {'waiting', 'copying', 'cleaning', 'paused', 'error'}


def lock():
    return FileLock(str(Path(settings.DATA_DIR) / '.media-storage.lock'), timeout=1)


def safe_path(root, key):
    parts = PurePosixPath(key).parts
    if not parts or key.startswith('/') or any(p in {'.', '..'} or ':' in p or '\\' in p for p in parts):
        raise ValueError('Invalid audio storage path')
    root = Path(root).resolve()
    path = root.joinpath(*parts)
    if path.resolve() != path or not path.is_relative_to(root):
        raise ValueError('Audio storage aliases are not allowed')
    return path


def key_for(path):
    root = Path(settings.PODCASTS_DIR).resolve()
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError('Audio storage aliases are not allowed')
    return path.relative_to(root).as_posix()


def state(conn=None):
    if conn is None:
        with get_db_connection() as db:
            return state(db)
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='media_storage'").fetchone():
        return {'volume_id': None, 'status': 'idle', 'operation': None, 'error': None, 'backup_path': None}
    row = conn.execute('SELECT * FROM media_storage WHERE id=1').fetchone()
    return dict(row)


def volume(check_write=False):
    if not settings.MEDIA_DIR:
        raise StorageUnavailable('Set MEDIA_DIR and mount the destination before enabling separate audio storage.')
    root = Path(settings.MEDIA_DIR).absolute()
    if root.resolve() != root or not root.is_dir():
        raise StorageUnavailable('The configured media directory is unavailable or is an alias. Check the mount.')
    data = Path(settings.PODCASTS_DIR).resolve()
    if root == Path(settings.DATA_DIR).resolve() or root.is_relative_to(data) or data.is_relative_to(root):
        raise StorageUnavailable('Media storage must be separate from the existing podcast directory.')
    current = state()
    marker = root / '.par-media-volume'
    if current['volume_id']:
        if not marker.is_file() or marker.is_symlink() or marker.read_text().strip() != current['volume_id']:
            raise StorageUnavailable('Media volume identity does not match. Restore the correct mount; no local fallback is used.')
    if check_write:
        fd, name = tempfile.mkstemp(prefix='.par-probe-', dir=root)
        try:
            os.write(fd, b'PAR'); os.fsync(fd)
        finally:
            os.close(fd); Path(name).unlink()
    return root


def enable():
    with lock():
        root = volume(check_write=True)
        current = state()
        if current['volume_id']:
            return
        marker = root / '.par-media-volume'
        if marker.exists() or any(root.iterdir()):
            raise StorageUnavailable('Use an empty dedicated media directory. An existing volume must be restored with its matching appdata.')
        identity = uuid4().hex
        with marker.open('x', encoding='utf-8') as f:
            f.write(identity); f.flush(); os.fsync(f.fileno())
        with get_db_connection() as conn:
            conn.execute('UPDATE media_storage SET volume_id=? WHERE id=1', (identity,)); conn.commit()


def processing_blocked(conn):
    current = state(conn)
    if current['status'] in ACTIVE:
        return True
    if settings.MEDIA_DIR or current['volume_id']:
        if not current['volume_id']:
            return True
        try:
            volume()
        except (OSError, ValueError, StorageUnavailable):
            return True
    return False


def resolve(key):
    legacy = safe_path(settings.PODCASTS_DIR, key)
    # Some lightweight route tests use an empty database. Never create schema here.
    if not Path(settings.DB_PATH).is_file():
        return legacy
    with get_db_connection() as conn:
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='media_files'").fetchone():
            return legacy
        row = conn.execute('SELECT 1 FROM media_files WHERE path=?', (key,)).fetchone()
    return safe_path(volume() / 'audio', key) if row else legacy


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def copy_audio(source, *, retain=True):
    """Caller holds the storage lock. Commit mapping only after verified final copy."""
    root = volume()
    key = key_for(source)
    destination = safe_path(root / 'audio', key)
    size, checksum = source.stat().st_size, digest(source)
    if shutil.disk_usage(root).free < size + settings.MIN_FREE_SPACE_BYTES:
        raise StorageUnavailable('Insufficient free space on media storage.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != size or digest(destination) != checksum:
            raise StorageUnavailable('Destination file differs from the original; nothing was overwritten.')
    else:
        temporary = destination.with_name(destination.name + '.par-part')
        if temporary.is_symlink():
            raise StorageUnavailable('Unsafe temporary audio path')
        try:
            with source.open('rb') as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
                dst.flush(); os.fsync(dst.fileno())
            if temporary.stat().st_size != size or digest(temporary) != checksum:
                raise StorageUnavailable('Audio copy verification failed; original retained.')
            volume()  # Fail closed if the mount changed while copying.
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    with get_db_connection() as conn:
        conn.execute('INSERT OR REPLACE INTO media_files(path,size,sha256,original_retained) VALUES (?,?,?,?)',
                     (key, size, checksum, int(retain)))
        conn.commit()
    return key


def stage_publication(path):
    """Keep the working output until the episode's normal fenced commit succeeds."""
    current = state()
    if not settings.MEDIA_DIR and not current['volume_id']:
        return False
    if not current['volume_id']:
        raise StorageUnavailable('Enable the mounted media directory in System > Storage first.')
    with lock():
        copy_audio(Path(path), retain=True)
    return True


def remove_original(key):
    with get_db_connection() as conn:
        row = conn.execute('SELECT * FROM media_files WHERE path=?', (key,)).fetchone()
    if not row:
        return
    destination = safe_path(volume() / 'audio', key)
    if not destination.is_file() or destination.stat().st_size != row['size'] or digest(destination) != row['sha256']:
        raise StorageUnavailable('Media verification failed; original copies have been kept.')
    source = safe_path(settings.PODCASTS_DIR, key)
    if source.exists():
        if source.stat().st_size != row['size'] or digest(source) != row['sha256']:
            raise StorageUnavailable('Original changed since migration; it has been kept.')
        source.unlink()
    with get_db_connection() as conn:
        conn.execute('UPDATE media_files SET original_retained=0 WHERE path=?', (key,)); conn.commit()


def inventory():
    """Published current audio and immutable older revisions, never retry inputs."""
    with get_db_connection() as conn:
        rows = conn.execute("SELECT local_filename FROM episodes WHERE local_filename IS NOT NULL AND status != 'ignored'").fetchall()
        known = {r[0] for r in conn.execute('SELECT path FROM media_files')}
    candidates = {Path(r[0]) for r in rows}
    root = Path(settings.PODCASTS_DIR)
    for marker in root.glob('*/episode-*/attempt-*/published.json'):
        candidates.add(marker.parent / 'processed.mp3')
    # Pre-attempt layouts published processed.mp3 directly in the episode directory.
    candidates.update(root.glob('*/*/processed.mp3'))
    result = []
    for path in sorted(candidates):
        key = key_for(path)
        if key in known:
            continue
        if not path.is_file():
            raise StorageUnavailable(f'A published source file is missing: {key}. Restore it or remove the unavailable episode before migrating.')
        result.append((key, path.stat().st_size))
    return result


def preview():
    root = volume(check_write=True)
    files = inventory()
    total = sum(size for _, size in files)
    free = shutil.disk_usage(root).free
    return {'count': len(files), 'bytes': total, 'free': free,
            'enough': free >= total + settings.MIN_FREE_SPACE_BYTES}


def start(operation='copy'):
    if operation not in {'copy', 'cleanup'}:
        raise ValueError('Unknown storage operation')
    with lock(), get_db_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        current = state(conn)
        if current['status'] in ACTIVE:
            raise StorageUnavailable('Finish or cancel the current storage operation first.')
        if not current['volume_id']:
            raise StorageUnavailable('Enable the destination first.')
        volume(check_write=True)
        conn.execute('DELETE FROM media_migration_items')
        conn.execute("UPDATE media_storage SET status='waiting', operation=?, error=NULL, backup_path=NULL WHERE id=1", (operation,))
        conn.commit()


def control(action):
    with get_db_connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        current = state(conn)
        if action == 'pause' and current['status'] in {'waiting', 'copying', 'cleaning'}:
            conn.execute("UPDATE media_storage SET status='paused' WHERE id=1")
        elif action == 'resume' and current['status'] in {'paused', 'error'}:
            # Preserve the manifest and completed entries; re-check destination on the next step.
            next_state = ('cleaning' if current['operation'] == 'cleanup' else 'copying') if current['backup_path'] else 'waiting'
            conn.execute('UPDATE media_storage SET status=?, error=NULL WHERE id=1', (next_state,))
        elif action == 'cancel' and current['status'] in ACTIVE:
            conn.execute("UPDATE media_storage SET status='cancelled', error=NULL WHERE id=1")
        else:
            raise ValueError('That storage action is not available now.')
        conn.commit()


def status():
    current = state()
    with get_db_connection() as conn:
        rows = conn.execute('SELECT done,COUNT(*) AS count,COALESCE(SUM(size),0) AS bytes FROM media_migration_items GROUP BY done').fetchall()
        current['retained'] = conn.execute('SELECT COUNT(*) FROM media_files WHERE original_retained=1').fetchone()[0]
    current.update(total=sum(r['count'] for r in rows), completed=sum(r['count'] for r in rows if r['done']),
                   total_bytes=sum(r['bytes'] for r in rows), completed_bytes=sum(r['bytes'] for r in rows if r['done']),
                   configured=bool(settings.MEDIA_DIR), destination=settings.MEDIA_DIR or 'Existing appdata storage', appdata=settings.DATA_DIR)
    try:
        if settings.MEDIA_DIR or current['volume_id']:
            volume()
        current['availability_error'] = None
    except (OSError, ValueError, StorageUnavailable) as exc:
        current['availability_error'] = str(exc)
    return current


def step():
    """One durable item per tick. File lock serializes CLI and all web workers."""
    try:
        with lock():
            current = state()
            if current['status'] not in {'waiting', 'copying', 'cleaning'}:
                return
            try:
                volume(check_write=True)
                if current['status'] == 'waiting':
                    with get_db_connection() as conn:
                        if conn.execute("SELECT 1 FROM jobs WHERE status='running'").fetchone():
                            return
                    from app.infra.backup import backup_database
                    backup = Path(settings.DATA_DIR) / 'backups' / f'before-media-{uuid4().hex}.db'
                    backup_database(settings.DB_PATH, backup)
                    if current['operation'] == 'copy':
                        files = inventory()
                        if shutil.disk_usage(volume()).free < sum(s for _, s in files) + settings.MIN_FREE_SPACE_BYTES:
                            raise StorageUnavailable('Insufficient space for the complete migration.')
                    else:
                        with get_db_connection() as conn:
                            files = [(r['path'], r['size']) for r in conn.execute('SELECT * FROM media_files WHERE original_retained=1')]
                    with get_db_connection() as conn:
                        conn.executemany('INSERT INTO media_migration_items(path,size) VALUES (?,?)', files)
                        # A concurrent Pause/Cancel must not be overwritten.
                        conn.execute("UPDATE media_storage SET backup_path=?, status=CASE WHEN status='waiting' THEN ? ELSE status END WHERE id=1",
                                     (str(backup), 'copying' if current['operation'] == 'copy' else 'cleaning'))
                        conn.commit()
                    return
                with get_db_connection() as conn:
                    item = conn.execute('SELECT * FROM media_migration_items WHERE done=0 ORDER BY path LIMIT 1').fetchone()
                if item:
                    if current['operation'] == 'copy':
                        copy_audio(safe_path(settings.PODCASTS_DIR, item['path']))
                    else:
                        remove_original(item['path'])
                    with get_db_connection() as conn:
                        conn.execute('UPDATE media_migration_items SET done=1 WHERE path=?', (item['path'],)); conn.commit()
                else:
                    with get_db_connection() as conn:
                        conn.execute("UPDATE media_storage SET status='complete' WHERE id=1 AND status IN ('copying','cleaning')"); conn.commit()
            except (OSError, ValueError, StorageUnavailable) as exc:
                with get_db_connection() as conn:
                    conn.execute("UPDATE media_storage SET status='error',error=? WHERE id=1 AND status NOT IN ('cancelled','paused')", (str(exc),)); conn.commit()
    except Timeout:
        return


async def worker():
    while True:
        try:
            await asyncio.to_thread(step)
        except Exception:
            import logging
            logging.getLogger(__name__).exception('Storage maintenance tick failed; will retry')
        await asyncio.sleep(1)


def delete_media(prefix, conn):
    """Called under the episode deletion transaction. No nested write connection."""
    if state(conn)['status'] in ACTIVE:
        raise StorageUnavailable('Audio deletion is paused during storage migration.')
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='media_files'").fetchone():
        return
    rows = conn.execute('SELECT path FROM media_files').fetchall()
    keys = [r[0] for r in rows if r[0].startswith(prefix.rstrip('/') + '/')]
    if keys:
        root = volume() / 'audio'
        for key in keys:
            safe_path(root, key).unlink(missing_ok=True)
            conn.execute('DELETE FROM media_files WHERE path=?', (key,))
