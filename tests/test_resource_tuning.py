from app.core.audio import AudioProcessor


def test_audio_thread_args_only_apply_positive_values():
    assert AudioProcessor._thread_args(0) == []
    assert AudioProcessor._thread_args(None) == []
    assert AudioProcessor._thread_args(3) == ["-threads", "3"]
    assert AudioProcessor._thread_args(100) == ["-threads", "64"]
