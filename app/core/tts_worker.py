import sys
import os
import argparse
import logging

# Simple worker script for Piper TTS
# Designed to be run as a separate process to avoid stability issues

def main():
    parser = argparse.ArgumentParser(description="Piper TTS Worker")
    parser.add_argument("output_path", nargs="?", help="Path to save the output WAV")
    parser.add_argument("--model", required=True, help="Model filename (in models/piper/) or absolute path")
    parser.add_argument("--models-dir", help="Base models directory")
    parser.add_argument("--check", action="store_true", help="Only check if model loads")
    
    args = parser.parse_args()
    
    try:
        from piper import PiperVoice
    except ImportError:
        print("Error: piper-tts not installed", file=sys.stderr)
        sys.exit(1)
        
    # Resolve model path
    model_path = args.model
    if not os.path.isabs(model_path) and args.models_dir:
        # Try finding it in models/piper/
        path1 = os.path.join(args.models_dir, "piper", args.model)
        path2 = os.path.join(args.models_dir, "models", "piper", args.model) # fallback
        if os.path.exists(path1):
            model_path = path1
        elif os.path.exists(path2):
            model_path = path2
            
    config_path = model_path + ".json"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)
        
    try:
        # Load voice
        voice = PiperVoice.load(model_path, config_path=config_path, use_cuda=False)
        
        if args.check:
            # Just testing the load
            sys.exit(0)
            
        if not args.output_path:
            print("Error: Output path required", file=sys.stderr)
            sys.exit(1)
            
        # Read text from stdin
        text = sys.stdin.read().strip()
        if not text:
            sys.exit(0)
            
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
        
        # Synthesize to WAV
        import wave
        with wave.open(args.output_path, "wb") as wav_file:
            voice.synthesize(text, wav_file)
            
        sys.exit(0)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
