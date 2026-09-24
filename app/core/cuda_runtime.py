"""Optional, pinned NVIDIA libraries. Never installs packages into the application."""
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import httpx
from filelock import FileLock

from app.core.config import settings
from app.infra.database import get_db_connection

MANIFEST = json.loads(Path(__file__).with_name("cuda_bundle.json").read_text())


def root():
    return Path(settings.DATA_DIR) / "runtimes" / "cuda"


def bundle():
    return root() / MANIFEST["id"]


def lock(name, timeout=0):
    root().mkdir(parents=True, exist_ok=True)
    return FileLock(str(root() / (name + ".lock")), timeout=timeout, thread_local=False)


def state():
    with get_db_connection() as conn:
        row = conn.execute("SELECT state_json FROM cuda_runtime WHERE id=1").fetchone()
    return json.loads(row[0]) if row else {}


def update_state(**changes):
    with get_db_connection() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT state_json FROM cuda_runtime WHERE id=1").fetchone()
        value = json.loads(row[0]) if row else {}
        value.update(changes)
        conn.execute("INSERT OR REPLACE INTO cuda_runtime(id,state_json) VALUES(1,?)", (json.dumps(value),))
        conn.commit()


def prerequisites():
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise RuntimeError("GPU setup requires a Linux x86-64 container (Linux Docker or Windows Docker Desktop with WSL2).")
    if importlib.metadata.version("ctranslate2") != MANIFEST["ctranslate2"]:
        raise RuntimeError("This application needs a matching reviewed CUDA bundle for its transcription engine.")
    code = "import ctypes; d=ctypes.CDLL('libcuda.so.1'); n=ctypes.c_int(); assert d.cuInit(0)==0; assert d.cuDeviceGetCount(ctypes.byref(n))==0 and n.value>0"
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("NVIDIA driver probe timed out. Check the host driver and Docker GPU access.") from exc
    if result.returncode:
        raise RuntimeError("NVIDIA GPU is not accessible. Install the host driver and enable Docker GPU access; then recreate the container.")


def installed(verify=False):
    try:
        marker = json.loads((bundle() / "installed.json").read_text())
        if marker != MANIFEST or not all((bundle() / "nvidia" / name / "lib").is_dir() for name in ("cublas", "cudnn")):
            return False
        inventory = json.loads((bundle() / "inventory.json").read_text())
        if not inventory:
            return False
        for name, expected in inventory.items():
            file = bundle() / name
            if not file.resolve().is_relative_to(bundle().resolve()) or file.stat().st_size != expected['size']:
                return False
            if verify:
                with file.open('rb') as stream:
                    if hashlib.file_digest(stream, 'sha256').hexdigest() != expected['sha256']:
                        return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def worker_environment():
    env = os.environ.copy()
    libraries = [str(bundle() / "nvidia" / name / "lib") for name in ("cublas", "cudnn")]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(libraries + [env.get("LD_LIBRARY_PATH", "")])
    env["DATA_DIR"] = str(settings.DATA_DIR)
    return env


def extract_wheel(wheel, destination):
    """Extract only inert wheel files; no setup scripts or arbitrary pip resolution."""
    with zipfile.ZipFile(wheel) as archive:
        total = 0
        for item in archive.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()) or "\\" in item.filename:
                raise ValueError("Unsafe path in CUDA package")
            if (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Unexpected symbolic link in CUDA package")
            total += item.file_size
            if total > 5 * 1024**3:
                raise ValueError("CUDA package exceeds extraction limit")
        archive.extractall(destination)


def install_bundle():
    """Caller owns setup.lock. A failed download cannot replace a working bundle."""
    prerequisites()
    if installed(verify=True):
        return
    root().mkdir(parents=True, exist_ok=True)
    for abandoned in root().glob("staging-*"):
        if abandoned.is_dir() and not abandoned.is_symlink() and abandoned.resolve().parent == root().resolve():
            shutil.rmtree(abandoned)
    if shutil.disk_usage(root()).free < 6 * 1024**3:
        raise RuntimeError("CUDA setup needs at least 6 GiB free in persistent storage.")
    update_state(phase="downloading", message="Downloading optional NVIDIA libraries (about 1.3 GB).", downloaded=0)
    downloaded = 0
    deadline = time.monotonic() + 3600
    with tempfile.TemporaryDirectory(prefix="staging-", dir=root()) as temporary:
        stage = Path(temporary)
        payload = stage / "payload"
        payload.mkdir()
        with httpx.Client(timeout=60, follow_redirects=False) as client:
            for package in MANIFEST["packages"]:
                if not package["url"].startswith("https://files.pythonhosted.org/"):
                    raise ValueError("Untrusted CUDA package location")
                wheel = stage / "package.whl"
                digest = hashlib.sha256()
                size = 0
                with client.stream("GET", package["url"]) as response, wheel.open("wb") as output:
                    response.raise_for_status()
                    last_update = 0
                    for data in response.iter_bytes(1024 * 1024):
                        if time.monotonic() > deadline:
                            raise TimeoutError("CUDA download exceeded one hour; retry setup.")
                        size += len(data)
                        if size > package["size"]:
                            raise ValueError("CUDA download exceeds expected size")
                        output.write(data); digest.update(data)
                        if time.monotonic() - last_update > 1:
                            update_state(downloaded=downloaded + size)
                            last_update = time.monotonic()
                if size != package["size"] or digest.hexdigest() != package["sha256"]:
                    raise ValueError("CUDA package checksum mismatch; download discarded")
                extract_wheel(wheel, payload)
                wheel.unlink()
                downloaded += size
        inventory = {}
        for file in payload.rglob('*'):
            if file.is_file():
                with file.open('rb') as stream:
                    inventory[str(file.relative_to(payload))] = {'size': file.stat().st_size, 'sha256': hashlib.file_digest(stream, 'sha256').hexdigest()}
        (payload / "inventory.json").write_text(json.dumps(inventory))
        (payload / "installed.json").write_text(json.dumps(MANIFEST))
        # An incomplete bundle is not active. Preserve it for diagnosis rather than overwrite.
        if bundle().exists():
            bundle().rename(root() / (MANIFEST["id"] + ".incomplete-" + str(time.time_ns())))
        payload.rename(bundle())
    update_state(downloaded=downloaded, bundle=MANIFEST["id"])
