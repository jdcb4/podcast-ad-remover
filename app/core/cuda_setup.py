"""Shared startup/manual setup, with optimistic protection of newer user choices."""
import logging
import threading

from filelock import Timeout
from app.core import cuda_runtime as cuda
from app.core.config import settings
from app.infra.database import get_db_connection

logger = logging.getLogger(__name__)
PREFERENCE_KEYS = ('whisper_model', 'whisper_device', 'whisper_compute_type',
                   'whisper_cuda_compute_type', 'whisper_cpu_threads', 'ffmpeg_threads')


def preferences():
    with get_db_connection() as conn:
        return dict(conn.execute('SELECT * FROM app_settings WHERE id=1').fetchone())


def config_key(runtime):
    return {key: runtime.get(key) for key in ('whisper_model', 'whisper_cpu_threads', 'whisper_cuda_compute_type')} | {'bundle': cuda.MANIFEST['id']}


def ready(runtime):
    value = cuda.state()
    return (cuda.installed() and not value.get('disabled', True)
            and value.get('validated_config') == config_key(runtime))


def validate(runtime):
    from app.core.cuda_client import probe
    cuda.update_state(phase='validating', probe_requested=True,
                      message='Waiting for GPU access, then testing the selected model. A missing model will be downloaded.')
    result = probe({key: runtime.get(key) for key in PREFERENCE_KEYS})
    runtime['whisper_cuda_compute_type'] = result['precision']
    cuda.update_state(phase='ready', disabled=False, message='GPU validation passed.',
                      supported=result['supported'], validated_config=config_key(runtime))


def _perform(guard, runtime, generation, activate, automatic):
    try:
        cuda.update_state(phase='checking', message='Checking host driver and Docker GPU access.', probe_requested=True)
        cuda.install_bundle()
        validate(runtime)
        if activate:
            with get_db_connection() as conn:
                conn.execute('''UPDATE app_settings SET whisper_device='cuda', whisper_model=?,
                    whisper_compute_type=?, whisper_cuda_compute_type=?, cuda_bootstrap_done=1
                    WHERE id=1 AND whisper_choice_version=?''',
                    (runtime['whisper_model'], runtime['whisper_compute_type'], runtime['whisper_cuda_compute_type'], generation))
                conn.commit()
    except Exception as exc:
        logger.warning('CUDA setup failed; CPU remains available: %s', exc)
        cuda.update_state(phase='failed', disabled=True, message=str(exc)[:1000],
                          effective_device='cpu', effective_compute_type='float32')
    finally:
        try:
            cuda.update_state(probe_requested=False)
        finally:
            guard.release()


def start_setup(*, automatic=False, proposed=None):
    guard = cuda.lock('setup')
    try:
        guard.acquire()
    except Timeout:
        return False
    try:
        with get_db_connection() as conn:
            conn.execute('BEGIN IMMEDIATE')
            current = dict(conn.execute('SELECT * FROM app_settings WHERE id=1').fetchone())
            generation = current['whisper_choice_version']
            initial = automatic and not current['cuda_bootstrap_done'] and generation == 0
            if automatic and not (current['whisper_device'] == 'cuda' or (settings.CUDA_SETUP and initial)):
                guard.release()
                return False
            if not automatic:
                generation += 1
                conn.execute('UPDATE app_settings SET whisper_choice_version=?, cuda_bootstrap_done=1 WHERE id=1', (generation,))
            conn.commit()
        runtime = {key: current[key] for key in PREFERENCE_KEYS}
        if proposed:
            runtime.update({key: proposed[key] for key in PREFERENCE_KEYS if key in proposed})
        # First activation chooses a supported GPU default, not an assumed float16.
        if initial or (not automatic and current['whisper_device'] == 'cpu' and not proposed):
            runtime['whisper_cuda_compute_type'] = None
        cuda.update_state(phase='checking', disabled=True, message='GPU setup queued.', probe_requested=True)
        threading.Thread(target=_perform, args=(guard, runtime, generation, True, automatic), daemon=True).start()
        return True
    except BaseException:
        guard.release()
        raise


def startup():
    # Reset interrupted status, then revalidate requested GPU configurations.
    cuda.update_state(probe_requested=False, disabled=True, effective_device='cpu', effective_compute_type='float32')
    if not start_setup(automatic=True):
        cuda.update_state(phase='idle', message='CPU selected. GPU setup is optional.')
