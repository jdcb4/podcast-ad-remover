import sys
import wave
import os
import subprocess
import urllib.request
from piper.voice import PiperVoice

# Resolve model path: check Docker path, then relative path
# We prefer /data/models/piper (persistent)
DOCKER_MODEL_DIR = "/data/models/piper"
LOCAL_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "models", "piper")

MODEL_JSON = "en_GB-cori-high.onnx.json"
MODEL_URL_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/cori/high/"
MODEL_JSON = "en_GB-cori-high.onnx.json"
MODEL_URL_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/cori/high/"
# Models provided by: https://huggingface.co/rhasspy/piper-voices
# Current default: en_GB-cori-high
DEFAULT_MODEL = "en_GB-cori-high.onnx"

MODEL_NAME = DEFAULT_MODEL

if os.path.exists("/data"):
    # Likely inside Docker
    TARGET_DIR = DOCKER_MODEL_DIR
else:
    # Likely local dev
    TARGET_DIR = LOCAL_MODEL_DIR

os.makedirs(TARGET_DIR, exist_ok=True)
MODEL_PATH = os.path.join(TARGET_DIR, MODEL_NAME)
CONFIG_PATH = os.path.join(TARGET_DIR, MODEL_JSON)

def download_if_missing():
    """Download the Piper model if it doesn't exist."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
        print(f"Model not found at {MODEL_PATH}. Downloading...", file=sys.stderr)
        try:
            # Construct URL based on filename? No, that's hard. 
            # For now, ONLY support auto-download for the default Cori model.
            # If user picks another model (in future), they must ensure it exists or we need a mapping.
            if MODEL_NAME == "en_GB-cori-high.onnx":
                 model_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/cori/high/en_GB-cori-high.onnx"
                 json_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/cori/high/en_GB-cori-high.onnx.json"
                 
                 if not os.path.exists(MODEL_PATH):
                    print(f"Downloading {model_url}...", file=sys.stderr)
                    urllib.request.urlretrieve(model_url, MODEL_PATH)
                
                 if not os.path.exists(CONFIG_PATH):
                    print(f"Downloading {json_url}...", file=sys.stderr)
                    urllib.request.urlretrieve(json_url, CONFIG_PATH)
                    
                 print("Download complete.", file=sys.stderr)
            else:
                 print(f"Model {MODEL_NAME} not found and no download URL known.", file=sys.stderr)
                 
        except Exception as e:
            print(f"Failed to download model: {e}", file=sys.stderr)
            sys.exit(1)

def generate_tts(output_path):
    # Read text from stdin
    text = sys.stdin.read()
    if not text:
        print("No text received on stdin", file=sys.stderr)
        sys.exit(1)
        
    try:
        download_if_missing()
        
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Piper model not found at {MODEL_PATH}")

        # Load voice
        voice = PiperVoice.load(MODEL_PATH, config_path=CONFIG_PATH)
        
        # Synthesize to temp wav
        temp_wav = output_path + ".wav"
        with wave.open(temp_wav, "wb") as wav_file:
            voice.synthesize(text, wav_file)
            
            # Append 0.5s of silence to prevent clipping
            # 22050Hz * 1 channel * 2 bytes/sample * 0.5s
            # Piper typically outputs 22050Hz, 16-bit mono
            framerate = wav_file.getframerate() or 22050 
            nchannels = wav_file.getnchannels() or 1
            sampwidth = wav_file.getsampwidth() or 2
            
            silence_duration = 0.5 # seconds
            num_frames = int(framerate * silence_duration)
            silence_data = b'\x00' * num_frames * nchannels * sampwidth
            wav_file.writeframes(silence_data)
            
        # Convert to MP3 (or whatever extension is requested)
        if output_path.endswith(".mp3"):
            subprocess.run(
                ["ffmpeg", "-y", "-i", temp_wav, "-codec:a", "libmp3lame", "-qscale:a", "2", output_path],
                check=True,
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            os.remove(temp_wav)
        else:
            # If not mp3, just rename if extension matches wav, else warn
            if output_path.endswith(".wav"):
                os.rename(temp_wav, output_path)
            else:
                 # Fallback convert
                subprocess.run(
                    ["ffmpeg", "-y", "-i", temp_wav, output_path],
                    check=True,
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
                os.remove(temp_wav)
            
        print(f"Successfully saved Piper TTS to {output_path}")

    except Exception as e:
        print(f"Piper TTS failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Piper TTS Worker")
    parser.add_argument("output_path", help="Path to save the generated audio file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Name of the Piper model to use (default: %(default)s)")
    
    args = parser.parse_args()
    
    output_path = args.output_path
    MODEL_NAME = args.model
    MODEL_JSON = MODEL_NAME + ".json"
    
    # Update paths based on selected model
    MODEL_PATH = os.path.join(TARGET_DIR, MODEL_NAME)
    CONFIG_PATH = os.path.join(TARGET_DIR, MODEL_JSON)
    
    # Ensure directory exists
    dirname = os.path.dirname(output_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    
    generate_tts(output_path)
