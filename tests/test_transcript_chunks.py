"""Chunk merging fixtures exercise speech crossing seams without model downloads."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.core.ai_services import Transcriber
from app.core.audio import AudioProcessor
from app.core.timeline import prepare_timeline


def segment(start, end, text):
    return SimpleNamespace(start=start, end=end, text=text, seek=0, tokens=[],
                           temperature=0, avg_logprob=0, compression_ratio=1,
                           no_speech_prob=0)


@pytest.fixture
def transcribe_chunks(tmp_path, monkeypatch):
    def run(chunks, duration=1300, fail_at=None):
        original = copy.deepcopy(chunks)
        paths = [tmp_path / f'chunk-{i}.wav' for i in range(len(chunks))]
        audio = str(tmp_path / 'original.mp3')
        def prepare(source, destination, **kwargs):
            assert source == audio
            (tmp_path / 'original.mp3.clean.wav').touch()
        def create(*args, **kwargs):
            for path in paths:
                path.touch()
            return [str(p) for p in paths]
        calls = []
        def transcribe(path, **kwargs):
            i = paths.index(Path(path))
            calls.append(i)
            def generate():
                if i == fail_at:
                    raise RuntimeError('Synthetic transcription failure')
                yield from chunks[i]
            return generate(), SimpleNamespace(language='en')
        monkeypatch.setattr(AudioProcessor, 'prepare_for_transcription', prepare)
        monkeypatch.setattr(AudioProcessor, 'create_audio_chunks', create)
        worker = Transcriber()
        worker.model = SimpleNamespace(transcribe=transcribe)
        progress = []
        try:
            result = worker._transcribe_chunked(audio, duration, lambda *p: progress.append(p))
            assert calls == list(range(len(chunks)))
            assert progress or not result['segments']
            assert chunks == original
            assert [s['id'] for s in result['segments']] == list(range(len(result['segments'])))
            units, _ = prepare_timeline(result, duration)
            assert all(u['end'] > u['start'] for u in units)
            assert units[0]['start'] == 0 and units[-1]['end'] == duration
            assert all(a['end'] == b['start'] for a, b in zip(units, units[1:]))
            return result
        finally:
            assert not any(p.exists() for p in paths)
            assert not (tmp_path / 'original.mp3.clean.wav').exists()
    return run


def test_nested_seam_retains_both_alternatives_as_one_item(transcribe_chunks):
    result = transcribe_chunks([
        [segment(0, 5, 'Opening.'), segment(1188, 1200, 'Full boundary sentence.')],
        [segment(11.44, 15.52, 'Short alternative.'), segment(20, 25, 'Following discussion.')]])
    merged = result['segments'][1]
    assert (merged['start'], merged['end']) == (1188, 1200)
    assert merged['text'] == 'Full boundary sentence.\nShort alternative.'
    assert [s['text'] for s in merged['source_segments']] == ['Full boundary sentence.', 'Short alternative.']
    assert [s['chunk_index'] for s in merged['source_segments']] == [0, 1]
    assert len(result['segments']) == 3


def test_segment_starting_before_seam_keeps_unique_speech_after_it(transcribe_chunks):
    # Previously the second chunk's entire 1187-1196 segment was dropped because
    # its start precedes the 1190 ownership seam. Its ending is unique speech.
    result = transcribe_chunks([
        [segment(1184, 1192, 'Earlier sentence.')],
        [segment(7, 16, 'Alternative with a unique ending.'), segment(18, 23, 'Next sentence.')]])
    merged = result['segments'][0]
    assert (merged['start'], merged['end']) == (1184, 1196)
    assert 'unique ending' in merged['text']
    assert result['segments'][1]['start'] == 1198


def test_ownership_excludes_overlap_context_and_keeps_exact_seam_start(transcribe_chunks):
    result = transcribe_chunks([
        [segment(1185, 1190, 'Before.'), segment(1190, 1200, 'Old context only.')],
        [segment(0, 10, 'New context only.'), segment(10, 15, 'After.')]])
    assert [s['text'] for s in result['segments']] == ['Before.', 'After.']
    assert all('source_segments' not in s for s in result['segments'])


def test_cross_chunk_ordering_and_chains_keep_full_connected_interval(transcribe_chunks):
    result = transcribe_chunks([
        [segment(1188, 1200, 'Long alternative.')],
        [segment(7, 13, 'Starts earlier.'), segment(13, 20, 'Continues.'),
         segment(20, 24, 'Touches but does not overlap.')]])
    assert result['segments'][0]['text'] == 'Starts earlier.\nLong alternative.\nContinues.'
    assert (result['segments'][0]['start'], result['segments'][0]['end']) == (1187, 1200)
    assert len(result['segments']) == 2


def test_multiple_seams_and_short_final_chunk(transcribe_chunks):
    result = transcribe_chunks([
        [segment(1188, 1200, 'First seam.')],
        [segment(11, 15, 'First alternative.'), segment(1188, 1200, 'Second seam.')],
        [segment(11, 15, 'Second alternative.'), segment(18, 30, 'Ending.')]], duration=2390)
    assert [(s['start'], s['end']) for s in result['segments']] == [(1188, 1200), (2368, 2390)]
    assert all(text in result['text'] for text in ['First seam.', 'First alternative.', 'Second seam.', 'Second alternative.', 'Ending.'])


def test_chunk_failure_cleans_all_temporary_audio(transcribe_chunks):
    with pytest.raises(RuntimeError, match='Synthetic transcription failure'):
        transcribe_chunks([[segment(0, 5, 'Opening.')], []], fail_at=1)


def test_empty_chunks_leave_the_full_duration_as_gap(transcribe_chunks):
    result = transcribe_chunks([[], []])
    assert result['segments'] == [] and result['text'] == ''
