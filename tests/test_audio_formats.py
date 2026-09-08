import shutil
import subprocess

import pytest

from app.core.audio import AudioProcessor


@pytest.mark.skipif(not shutil.which('ffmpeg'), reason='FFmpeg required for real audio checks')
@pytest.mark.parametrize('extension,codec', [('mp3', 'libmp3lame'), ('m4a', 'aac'), ('ogg', 'libopus')])
@pytest.mark.parametrize('cuts', [[], [{'start': 1, 'end': 3}]])
def test_real_audio_always_publishes_playable_mp3(tmp_path, extension, codec, cuts):
    source, output = tmp_path / ('input.' + extension), tmp_path / 'processed.mp3'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=4',
                    '-c:a', codec, str(source)], check=True, capture_output=True, timeout=30)
    AudioProcessor.remove_segments(str(source), str(output), cuts)
    duration = AudioProcessor.get_duration(str(output))
    assert abs(duration - (2 if cuts else 4)) < 0.2
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                             '-show_entries', 'stream=codec_name', '-of', 'csv=p=0', str(output)],
                            check=True, capture_output=True, text=True, timeout=30)
    assert result.stdout.strip() == 'mp3'
