from pathlib import Path

import pytest

from app.core.config import settings
from app.core.processor import Processor


@pytest.mark.parametrize("suffix", ["", ".", "show", "show/..", "show/."])
def test_episode_cleanup_rejects_storage_and_subscription_roots(isolated_data_dir, suffix):
    root = Path(settings.PODCASTS_DIR)
    sibling = root / "show" / "keep" / "processed.mp3"
    sibling.parent.mkdir(parents=True)
    sibling.write_bytes(b"published")
    assert not Processor()._remove_episode_directory(str(root / suffix), "clean")
    assert sibling.read_bytes() == b"published"


def test_subscription_cleanup_rejects_nested_paths(isolated_data_dir):
    path = Path(settings.PODCASTS_DIR) / "show" / "episode"
    path.mkdir(parents=True)
    assert not Processor()._remove_subscription_directory("show/episode")
    assert path.exists()
