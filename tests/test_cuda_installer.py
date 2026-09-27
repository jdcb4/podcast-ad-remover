import hashlib
import io
import json
import zipfile
from types import SimpleNamespace

import httpx
import pytest

from app.core import cuda_runtime as cuda
from app.infra.database import init_db


@pytest.fixture
def installer(isolated_data_dir, monkeypatch):
    init_db()
    monkeypatch.setattr(cuda, "prerequisites", lambda: None)
    monkeypatch.setattr(cuda.shutil, "disk_usage", lambda path: SimpleNamespace(free=10 * 1024**3))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('nvidia/cublas/lib/a.so', b'fixture')
        z.writestr('nvidia/cudnn/lib/b.so', b'fixture')
    data = archive.getvalue()
    manifest = {'id': 'test-bundle', 'packages': [{'url': 'https://files.pythonhosted.org/test.whl', 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}]}
    monkeypatch.setattr(cuda, 'MANIFEST', manifest)
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=data)))
    monkeypatch.setattr(cuda.httpx, 'Client', lambda **kwargs: client)
    return manifest


def test_verified_install_reuse_and_environment(installer, monkeypatch):
    cuda.install_bundle()
    assert cuda.installed()
    monkeypatch.setattr(cuda.httpx, 'Client', lambda **kwargs: pytest.fail('Unexpected repeat download'))
    cuda.install_bundle()
    assert str(cuda.bundle() / 'nvidia/cudnn/lib') in cuda.worker_environment()['LD_LIBRARY_PATH']


def test_checksum_failure_never_activates(installer):
    installer['packages'][0]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checksum'):
        cuda.install_bundle()
    assert not cuda.installed()
    assert not list(cuda.root().glob('staging-*'))


def test_disk_failure_does_not_download(installer, monkeypatch):
    monkeypatch.setattr(cuda.shutil, 'disk_usage', lambda path: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match='6 GiB'):
        cuda.install_bundle()


def test_corruption_is_detected_before_reuse(installer):
    cuda.install_bundle()
    (cuda.bundle() / 'nvidia/cublas/lib/a.so').write_bytes(b'changed')
    assert not cuda.installed(verify=True)


def test_damaged_inventory_is_treated_as_uninstalled(installer):
    cuda.install_bundle()
    (cuda.bundle() / 'inventory.json').write_text('{"missing": {}}')
    assert not cuda.installed()


def test_archive_path_escape_is_rejected(tmp_path):
    p = tmp_path / 'bad.whl'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('../outside', 'bad')
    with pytest.raises(ValueError, match='Unsafe'):
        cuda.extract_wheel(p, tmp_path / 'destination')
    assert not (tmp_path / 'outside').exists()


def test_cpu_platform_rejected_before_driver_access(monkeypatch):
    monkeypatch.setattr(cuda.platform, 'machine', lambda: 'aarch64')
    with pytest.raises(RuntimeError, match='x86-64'):
        cuda.prerequisites()
