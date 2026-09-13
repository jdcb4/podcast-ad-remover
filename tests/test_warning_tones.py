import itertools
import json
import shutil
import struct
import subprocess
import wave

import pytest

from app.core.audio import AudioProcessor
from app.core.model_defaults import MODEL_DEFAULTS
from app.core.warning_tones import tone_path
from app.infra.database import get_db_connection, init_db
from tests.test_timeline_http import client


def test_model_defaults_upgrade_preserves_custom_settings(isolated_data_dir):
    init_db()
    columns = {'gemini': 'ai_model_cascade', 'openrouter': 'openrouter_model',
               'openai': 'openai_model', 'anthropic': 'anthropic_model',
               'gemini_tts': 'gemini_tts_model_cascade'}
    with get_db_connection() as conn:
        row = conn.execute('SELECT * FROM app_settings').fetchone()
        for provider, column in columns.items():
            assert json.loads(row[column]) == MODEL_DEFAULTS[provider]
        assert all(row[f'warning_tone_{part}'] == 0 for part in ('start', 'middle', 'end'))
        assert all(row[f'warning_tone_{part}_style'] == 'wooden' for part in ('start', 'middle', 'end'))
        conn.execute("DELETE FROM schema_migrations WHERE version = '20260913_0017_model_defaults'")
        conn.execute("UPDATE app_settings SET openai_model = ?, anthropic_model = 'my-custom-model'", ('["gpt-4o"]',))
        conn.commit()
    init_db()
    with get_db_connection() as conn:
        row = conn.execute('SELECT * FROM app_settings').fetchone()
        assert json.loads(row['openai_model']) == MODEL_DEFAULTS['openai']
        assert row['anthropic_model'] == 'my-custom-model'
        conn.execute("UPDATE app_settings SET openai_model = 'gpt-4o'")
        conn.commit()
    init_db()
    with get_db_connection() as conn:
        assert conn.execute('SELECT openai_model FROM app_settings').fetchone()[0] == 'gpt-4o'


def test_tone_settings_save_fixed_sound_and_old_form_compatibility(client):
    url = '/admin/global-subscription-settings/update'
    assert client.post(url, data={'warning_tones_present': 'true', 'warning_tone_start': 'true',
                                 'warning_tone_end': 'true', 'warning_tone_start_style': 'warm'},
                       follow_redirects=False).status_code == 303
    page = client.get('/admin/global-subscription-settings')
    assert page.status_code == 200
    assert '<audio controls' not in page.text
    for position in ('start', 'middle', 'end'):
        assert f'name="warning_tone_{position}"' in page.text
        assert f'warning_tone_{position}_style' not in page.text
    assert client.post(url, data={}, follow_redirects=False).status_code == 303
    with get_db_connection() as conn:
        row = conn.execute('SELECT * FROM app_settings').fetchone()
        assert (row['warning_tone_start'], row['warning_tone_middle'], row['warning_tone_end']) == (1, 0, 1)
        assert row['warning_tone_start_style'] == 'wooden'
    assert client.post(url, data={'warning_tones_present': 'true', 'warning_tone_start_style': '../bad'},
                       follow_redirects=False).status_code == 303
    assert client.post(url, data={'warning_tones_present': 'true', 'warning_tone_start_style': 'clear'},
                       follow_redirects=False).status_code == 303
    with get_db_connection() as conn:
        row = conn.execute('SELECT * FROM app_settings').fetchone()
        assert all(row[f'warning_tone_{part}'] == 0 for part in ('start', 'middle', 'end'))


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg required')
@pytest.mark.parametrize('start,middle,end', list(itertools.product((False, True), repeat=3)))
def test_actual_audio_marks_only_removed_intervals(tmp_path, start, middle, end):
    source, output = tmp_path / 'source.wav', tmp_path / 'output.mp3'
    with wave.open(str(source), 'wb') as handle:
        handle.setparams((1, 2, 22050, 0, 'NONE', 'not compressed'))
        handle.writeframes(b'\0\0' * 22050 * 12)
    cuts = [{'start': a, 'end': b} for a, b in [(0, 1), (3, 4), (4, 5), (8, 9), (11, 12)]]
    options = dict(start=start, middle=middle, end=end, start_style='soft', middle_style='soft', end_style='soft')
    AudioProcessor.remove_segments(str(source), str(output), cuts, warning_tones=options)
    edge, low = [AudioProcessor.get_duration(str(tone_path(kind))) for kind in ('start', 'middle')]
    assert AudioProcessor.get_duration(str(output)) == pytest.approx(7 + edge * (start + end) + 2 * low * middle, abs=0.1)
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output), '-f', 's16le',
                              '-ac', '1', '-ar', '22050', '-'], capture_output=True, check=True).stdout
    samples = struct.unpack(f'<{len(decoded)//2}h', decoded)
    # Verify audible energy at each cue and silence inside retained source intervals.
    def peak(t):
        return max(abs(x) for x in samples[int(t * 22050):int((t + .04) * 22050)])
    cursor = 0
    for enabled, duration, retained in [(start, edge, 2), (middle, low, 3), (middle, low, 2), (end, edge, 0)]:
        if enabled:
            assert peak(cursor + .08) > 100
            cursor += duration
        if retained:
            assert peak(cursor + .5) < 10
            cursor += retained
    # Enabled edge switches do not add bookends when only a middle cut exists.
    AudioProcessor.remove_segments(str(source), str(output), [{'start': 3, 'end': 5}], warning_tones=options)
    assert AudioProcessor.get_duration(str(output)) == pytest.approx(10 + low * middle, abs=.1)
    AudioProcessor.remove_segments(str(source), str(output), [], warning_tones=options)
    assert AudioProcessor.get_duration(str(output)) == pytest.approx(12, abs=.1)
