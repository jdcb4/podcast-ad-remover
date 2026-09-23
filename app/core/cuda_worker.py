"""Private JSON-lines worker; launched only with the selected CUDA library paths."""
import ctypes
import json
import os
import signal
import sys
import tempfile
import wave
from pathlib import Path


def emit(value):
    print(json.dumps(value), flush=True)


def main():
    # Linux containers include Docker Desktop/WSL2. Do not leave orphan GPU allocations.
    parent = os.getppid()
    if sys.platform == 'linux':
        if ctypes.CDLL(None).prctl(1, signal.SIGKILL) != 0:
            raise RuntimeError('Cannot establish GPU worker parent-death protection')
        if os.getppid() != parent or parent == 1:
            return
    from app.core.ai_services import Transcriber
    from app.core.transcription_settings import supported_compute_types
    transcriber = Transcriber()
    for line in sys.stdin:
        try:
            command = json.loads(line)
            runtime = command['runtime']
            supported = supported_compute_types('cuda')
            precision = runtime.get('whisper_cuda_compute_type')
            if command['action'] == 'probe' and not precision:
                precision = 'float16' if 'float16' in supported else 'float32'
            if precision not in supported:
                raise ValueError('Selected GPU precision is unsupported on this GPU')
            runtime.update(_execution_device='cuda', _execution_precision=precision)
            if command['action'] == 'probe':
                transcriber.load_model(runtime)
                with tempfile.TemporaryDirectory(prefix='cuda-probe-') as tmp:
                    audio = str(Path(tmp) / 'silence.wav')
                    with wave.open(audio, 'wb') as wav:
                        wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
                        wav.writeframes(b'\0' * 32000)
                    segments, _ = transcriber.model.transcribe(audio, beam_size=1)
                    list(segments)  # Inference is lazy: loading alone is not a GPU test.
                result = {'supported': supported, 'precision': precision}
            elif command['action'] == 'transcribe':
                result = transcriber._transcribe_local(command['audio'],
                    lambda current, total: emit({'progress': [current, total]}), runtime)
            else:
                raise ValueError('Unknown GPU worker action')
            emit({'result': result})
        except Exception as exc:
            emit({'error': str(exc)[:1000]})


if __name__ == '__main__':
    main()
