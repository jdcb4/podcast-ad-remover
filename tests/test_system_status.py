from pathlib import Path

from app.core.config import settings
from app.core.system_status import _storage_breakdown


def write_bytes(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_storage_breakdown_reports_data_categories(isolated_data_dir):
    write_bytes(isolated_data_dir / "podcasts" / "show" / "episode" / "audio.mp3", 10)
    write_bytes(isolated_data_dir / "models" / "whisper.bin", 20)
    write_bytes(isolated_data_dir / "feeds" / "show.xml", 30)
    write_bytes(isolated_data_dir / "db" / "podcasts.db", 40)
    write_bytes(isolated_data_dir / "backups" / "backup.db", 50)
    write_bytes(isolated_data_dir / "app.log", 60)

    breakdown = {item["key"]: item for item in _storage_breakdown()}

    assert breakdown["podcasts"]["bytes"] == 10
    assert breakdown["models"]["bytes"] == 20
    assert breakdown["feeds"]["bytes"] == 30
    assert breakdown["database"]["bytes"] == 40
    assert breakdown["backups"]["bytes"] == 50
    assert breakdown["logs"]["bytes"] == 60
    assert breakdown["podcasts"]["display"] == "10 B"
    assert settings.DATA_DIR == str(isolated_data_dir)


def test_queue_template_renders_and_refreshes_storage_breakdown():
    source = Path("app/web/templates/admin/queue.html").read_text(encoding="utf-8")

    assert "data-storage-breakdown" in source
    assert "operation_status.disk.breakdown" in source
    assert "function escapeHtml" in source
