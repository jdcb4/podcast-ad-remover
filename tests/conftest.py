import os

import pytest

from app.core.config import settings


@pytest.fixture()
def isolated_data_dir(tmp_path):
    original_data_dir = settings.DATA_DIR
    settings.DATA_DIR = str(tmp_path)
    os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
    os.makedirs(settings.PODCASTS_DIR, exist_ok=True)
    os.makedirs(settings.FEEDS_DIR, exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    yield tmp_path
    settings.DATA_DIR = original_data_dir
