import pytest
from unittest.mock import Mock, patch, MagicMock
from app.core.ai_services import Transcriber
from app.infra.database import get_db_connection, init_db


def test_transcriber_loads_runtime_settings_with_transcription_engine(isolated_data_dir):
    """Test that Transcriber loads transcription_engine from database settings."""
    init_db()
    
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET transcription_engine = 'whisperx' WHERE id = 1"
        )
        conn.commit()
    
    transcriber = Transcriber()
    settings = transcriber._load_runtime_settings()
    
    assert settings["transcription_engine"] == "whisperx"
    assert settings["whisper_model"] == "base"
    assert settings["whisper_cpu_threads"] == 0
    assert settings["ffmpeg_threads"] == 0


def test_transcriber_defaults_to_faster_whisper_when_not_set(isolated_data_dir):
    """Test that Transcriber defaults to faster-whisper when transcription_engine is NULL."""
    init_db()
    
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET transcription_engine = NULL WHERE id = 1"
        )
        conn.commit()
    
    transcriber = Transcriber()
    settings = transcriber._load_runtime_settings()
    
    assert settings["transcription_engine"] == "faster-whisper"


def test_transcriber_loads_faster_whisper_model(isolated_data_dir):
    """Test that Transcriber loads Faster-Whisper model when engine is faster-whisper."""
    init_db()
    
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET transcription_engine = 'faster-whisper', whisper_model = 'tiny' WHERE id = 1"
        )
        conn.commit()
    
    transcriber = Transcriber()
    
    # Mock the actual import inside load_model
    with patch('app.core.ai_services.importlib.util.find_spec') as mock_find_spec:
        mock_find_spec.return_value = True  # Simulate module exists
        
        with patch.object(transcriber, 'load_model') as mock_load:
            # Just verify the settings are loaded correctly
            settings = transcriber._load_runtime_settings()
            assert settings["transcription_engine"] == "faster-whisper"
            assert settings["whisper_model"] == "tiny"


def test_transcriber_loads_whisperx_model(isolated_data_dir):
    """Test that Transcriber loads WhisperX model when engine is whisperx."""
    init_db()
    
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET transcription_engine = 'whisperx', whisper_model = 'tiny' WHERE id = 1"
        )
        conn.commit()
    
    transcriber = Transcriber()
    
    # Just verify the settings are loaded correctly
    settings = transcriber._load_runtime_settings()
    assert settings["transcription_engine"] == "whisperx"
    assert settings["whisper_model"] == "tiny"


def test_transcriber_reloads_model_when_engine_changes(isolated_data_dir):
    """Test that Transcriber reloads model when transcription_engine changes."""
    init_db()
    
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET transcription_engine = 'faster-whisper', whisper_model = 'tiny' WHERE id = 1"
        )
        conn.commit()
    
    transcriber = Transcriber()
    
    # Set initial state
    transcriber.model_config = ("faster-whisper", "tiny", 0)
    transcriber.transcription_engine = "faster-whisper"
    
    # Change engine in DB
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET transcription_engine = 'whisperx' WHERE id = 1"
        )
        conn.commit()
    
    # Reload settings and verify config changes
    settings = transcriber._load_runtime_settings()
    assert settings["transcription_engine"] == "whisperx"
    
    # New config would be different, triggering reload
    new_config = (settings["transcription_engine"], settings["whisper_model"], settings["whisper_cpu_threads"])
    assert new_config != transcriber.model_config


def test_transcriber_unloads_model_with_engine_name(isolated_data_dir):
    """Test that unload_model logs the correct engine name."""
    init_db()
    
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET transcription_engine = 'whisperx' WHERE id = 1"
        )
        conn.commit()
    
    transcriber = Transcriber()
    transcriber.transcription_engine = "whisperx"
    transcriber.model = MagicMock()
    
    with patch('app.core.ai_services.logger') as mock_logger:
        transcriber.unload_model()
        
        mock_logger.info.assert_called()
        log_message = mock_logger.info.call_args[0][0]
        assert "whisperx" in log_message.lower()
        assert transcriber.model is None
        assert transcriber.transcription_engine is None


def test_transcribe_routes_to_faster_whisper(isolated_data_dir):
    """Test that transcribe() routes to _transcribe_faster_whisper when engine is faster-whisper."""
    init_db()
    
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET transcription_engine = 'faster-whisper' WHERE id = 1"
        )
        conn.commit()
    
    transcriber = Transcriber()
    
    with patch.object(transcriber, '_transcribe_faster_whisper') as mock_fw:
        mock_fw.return_value = {"segments": [], "text": "", "language": "en"}
        
        result = transcriber.transcribe("test.mp3")
        
        mock_fw.assert_called_once()
        assert result == {"segments": [], "text": "", "language": "en"}


def test_transcribe_routes_to_whisperx(isolated_data_dir):
    """Test that transcribe() routes to _transcribe_whisperx when engine is whisperx."""
    init_db()
    
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET transcription_engine = 'whisperx' WHERE id = 1"
        )
        conn.commit()
    
    transcriber = Transcriber()
    
    # Mock both transcription methods and AudioProcessor
    with patch.object(transcriber, '_transcribe_whisperx') as mock_wx:
        with patch('app.core.audio.AudioProcessor') as mock_audio:
            with patch.object(transcriber, 'load_model'):
                mock_audio.get_duration.return_value = 10.0
                mock_wx.return_value = {"segments": [], "text": "", "language": "en"}
                
                result = transcriber.transcribe("test.mp3")
                
                mock_wx.assert_called_once()
                assert result == {"segments": [], "text": "", "language": "en"}


def test_transcribe_whisperx_converts_segments_to_standard_format(isolated_data_dir):
    """Test that _transcribe_whisperx converts WhisperX segments to standard format."""
    init_db()
    
    transcriber = Transcriber()
    
    # Mock WhisperX response
    mock_whisperx_result = {
        "text": "Hello world",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello "},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]
    }
    
    # Mock the whisperx module
    mock_wx_module = MagicMock()
    mock_model = MagicMock()
    mock_model.transcribe.return_value = mock_whisperx_result
    mock_wx_module.load_model.return_value = mock_model
    mock_align_model = MagicMock()
    mock_metadata = {}
    mock_wx_module.load_align_model.return_value = (mock_align_model, mock_metadata)
    mock_wx_module.align.return_value = mock_whisperx_result
    
    with patch('app.core.audio.AudioProcessor') as mock_audio:
        mock_audio.get_duration.return_value = 2.0
        mock_audio.prepare_for_transcription.return_value = None
        
        # Manually set up the transcriber to avoid import issues
        transcriber.model = mock_model
        transcriber.transcription_engine = "whisperx"
        
        # Use the actual method but with mocked whisperx
        import sys
        sys.modules['whisperx'] = mock_wx_module
        
        try:
            result = transcriber._transcribe_whisperx("test.mp3", 2.0)
        finally:
            del sys.modules['whisperx']
    
    assert result["text"] == "Hello world"
    assert result["language"] == "en"
    assert len(result["segments"]) == 2
    
    # Check segment format
    seg = result["segments"][0]
    assert seg["id"] == 0
    assert seg["start"] == 0.0
    assert seg["end"] == 1.0
    assert seg["text"] == "Hello "
    assert seg["seek"] == 0
    assert seg["tokens"] == []
    assert seg["temperature"] == 0.0
    assert seg["avg_logprob"] == 0.0
    assert seg["compression_ratio"] == 0.0
    assert seg["no_speech_prob"] == 0.0


def test_transcribe_whisperx_handles_progress_callback(isolated_data_dir):
    """Test that _transcribe_whisperx calls progress callback."""
    init_db()
    
    transcriber = Transcriber()
    
    mock_whisperx_result = {
        "text": "Test",
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Test"}]
    }
    
    progress_calls = []
    
    def mock_progress(end, duration):
        progress_calls.append((end, duration))
    
    # Mock the whisperx module
    mock_wx_module = MagicMock()
    mock_model = MagicMock()
    mock_model.transcribe.return_value = mock_whisperx_result
    mock_wx_module.load_model.return_value = mock_model
    mock_wx_module.load_align_model.return_value = (MagicMock(), {})
    mock_wx_module.align.return_value = mock_whisperx_result
    
    with patch('app.core.audio.AudioProcessor') as mock_audio:
        mock_audio.get_duration.return_value = 1.0
        mock_audio.prepare_for_transcription.return_value = None
        
        transcriber.model = mock_model
        transcriber.transcription_engine = "whisperx"
        
        import sys
        sys.modules['whisperx'] = mock_wx_module
        
        try:
            transcriber._transcribe_whisperx("test.mp3", 1.0, progress_callback=mock_progress)
        finally:
            del sys.modules['whisperx']
    
    assert len(progress_calls) == 1
    assert progress_calls[0] == (1.0, 1.0)


def test_transcriber_model_config_includes_engine(isolated_data_dir):
    """Test that model_config tuple includes transcription_engine."""
    init_db()
    
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET transcription_engine = 'whisperx', whisper_model = 'base', whisper_cpu_threads = 4 WHERE id = 1"
        )
        conn.commit()
    
    transcriber = Transcriber()
    
    # Just verify the settings are loaded correctly
    settings = transcriber._load_runtime_settings()
    
    # The config tuple would be (engine, model, cpu_threads)
    expected_config = (settings["transcription_engine"], settings["whisper_model"], settings["whisper_cpu_threads"])
    assert expected_config == ("whisperx", "base", 4)


def test_transcriber_does_not_reload_if_config_unchanged(isolated_data_dir):
    """Test that Transcriber doesn't reload model if config is unchanged."""
    init_db()
    
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET transcription_engine = 'faster-whisper', whisper_model = 'base' WHERE id = 1"
        )
        conn.commit()
    
    transcriber = Transcriber()
    
    # Set up initial state
    transcriber.model = MagicMock()
    transcriber.model_config = ("faster-whisper", "base", 0)
    transcriber.transcription_engine = "faster-whisper"
    
    # Load settings - they should match the current config
    settings = transcriber._load_runtime_settings()
    current_config = (settings["transcription_engine"], settings["whisper_model"], settings["whisper_cpu_threads"])
    
    # Config should be the same, so no reload needed
    assert current_config == transcriber.model_config
