from app.core.audio import AudioProcessor


def test_calculate_keep_segments_sorts_clamps_and_merges_remove_segments():
    keep_segments = AudioProcessor._calculate_keep_segments(
        100.0,
        [
            {"start": 20, "end": 30},
            {"start": 10, "end": 15},
            {"start": 14, "end": 25},
            {"start": 90, "end": 120},
            {"start": -5, "end": 2},
            {"start": 50, "end": 50},
            {"start": "bad", "end": 70},
            {"end": 80},
        ],
    )

    assert keep_segments == [(2.0, 10.0), (30.0, 90.0)]


def test_calculate_keep_segments_keeps_whole_file_when_nothing_valid_is_removed():
    assert AudioProcessor._calculate_keep_segments(42.0, []) == [(0.0, 42.0)]
    assert AudioProcessor._calculate_keep_segments(42.0, [{"start": 5, "end": 5}]) == [(0.0, 42.0)]


def test_remove_segments_uses_calculated_keep_segments_in_filter(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd

    monkeypatch.setattr(AudioProcessor, "get_duration", staticmethod(lambda path: 100.0))
    monkeypatch.setattr("app.core.audio.subprocess.run", fake_run)

    AudioProcessor.remove_segments(
        "input.mp3",
        "output.mp3",
        [
            {"start": 20, "end": 30},
            {"start": 10, "end": 15},
            {"start": 14, "end": 25},
            {"start": 90, "end": 120},
        ],
        ffmpeg_threads=2,
    )

    filter_index = captured["cmd"].index("-filter_complex") + 1
    filter_complex = captured["cmd"][filter_index]

    assert "atrim=start=0.0:end=10.0" in filter_complex
    assert "atrim=start=30.0:end=90.0" in filter_complex
    assert "concat=n=2:v=0:a=1" in filter_complex
    assert "-threads" in captured["cmd"]
    assert "2" in captured["cmd"]
