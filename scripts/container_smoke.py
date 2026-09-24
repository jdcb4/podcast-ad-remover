"""Offline smoke check for a disposable container; never use against live /data."""
import json
import importlib.metadata
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    with tempfile.TemporaryDirectory(prefix='podcast-smoke-') as temporary:
        os.environ.update(DATA_DIR=temporary, PROCESSOR_ENABLED='false', BASE_URL='http://127.0.0.1:8199')
        from app.infra.database import init_db, get_db_connection
        from app.infra.backup import backup_database
        from app.core.audio import AudioProcessor
        from app.core.config import settings
        from faster_whisper import WhisperModel  # Validate native runtime imports without downloading models.
        assert not any((dist.metadata.get('Name') or '').lower().startswith('nvidia-')
                       for dist in importlib.metadata.distributions()), 'CPU image unexpectedly bundles NVIDIA packages'
        init_db()
        with get_db_connection() as conn:
            assert conn.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        backup_database(settings.DB_PATH, Path(temporary) / 'snapshot.db')
        subprocess.run([sys.executable, 'scripts/publish_pending_feeds.py'], cwd=root, check=True, timeout=30)
        audio = Path(temporary) / 'input.m4a'
        output = Path(temporary) / 'output.mp3'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:duration=2','-c:a','aac',str(audio)], check=True, timeout=30)
        AudioProcessor.remove_segments(str(audio), str(output), [])
        assert 1.8 < AudioProcessor.get_duration(str(output)) < 2.2
        server = subprocess.Popen([sys.executable,'-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8199'], cwd=root)
        try:
            for _ in range(60):
                if server.poll() is not None:
                    raise RuntimeError('Application exited during startup')
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8199/health', timeout=1) as response:
                        assert json.load(response)['status'] == 'healthy'
                    break
                except OSError:
                    time.sleep(0.25)
            else:
                raise RuntimeError('Application did not become healthy')
            with urllib.request.urlopen('http://127.0.0.1:8199/', timeout=5) as response:
                assert response.status == 200 and b'Podcast' in response.read()
            with urllib.request.urlopen('http://127.0.0.1:8199/static/js/episodes.js', timeout=5) as response:
                assert response.status == 200
            from app.core.cuda_runtime import state, installed
            from app.core.cuda_setup import preferences
            if settings.CUDA_SETUP:
                # This variant deliberately runs without --gpus and without network access.
                for _ in range(150):
                    status = state()
                    if status.get('phase') == 'failed':
                        break
                    time.sleep(0.25)
                else:
                    raise AssertionError('Missing-GPU setup did not fail promptly')
                assert 'NVIDIA GPU is not accessible' in status['message'], status
                assert status['effective_device'] == 'cpu'
            assert preferences()['whisper_device'] == 'cpu'
            assert not installed()
            assert not list((Path(temporary) / 'runtimes' / 'cuda').glob('staging-*'))
        finally:
            server.terminate()
            server.wait(timeout=10)
        print('PASS: native imports, migration, backup, AAC-to-MP3, startup, health, dashboard and assets')


if __name__ == '__main__':
    main()
