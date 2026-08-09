from app.core.config import Settings


def test_processor_is_enabled_by_default():
    assert Settings(_env_file=None).PROCESSOR_ENABLED is True


def test_processor_can_be_disabled_from_environment(monkeypatch):
    monkeypatch.setenv("PROCESSOR_ENABLED", "false")

    assert Settings(_env_file=None).PROCESSOR_ENABLED is False
