"""Small, locally synthesized removal cues; independent of speech providers."""

import math
import struct
import wave
from pathlib import Path


TONE_STYLES = {
    "soft": {"label": "Soft chime", "notes": (440, 554.37, 659.25), "low": 220},
    "warm": {"label": "Warm mellow", "notes": (330, 440, 550), "low": 165},
    "clear": {"label": "Clear signal", "notes": (523.25, 659.25, 783.99), "low": 261.63},
    "bell": {"label": "Gentle bell", "notes": (523.25, 783.99), "low": 220},
    "sonar": {"label": "Sonar pair", "notes": (440, 880), "low": 180},
    "wooden": {"label": "Wooden notes", "notes": (392, 523.25), "low": 196},
}
ASSET_DIR = Path(__file__).resolve().parents[1] / "web" / "static" / "audio" / "warning-tones"


def tone_path(kind: str) -> Path:
    if kind not in ("start", "end", "middle"):
        raise ValueError("Unknown warning tone")
    return ASSET_DIR / f"wooden-{kind}.wav"


def generate_assets():
    """Regenerate the bundled previews and processing assets with the standard library."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    rate = 22050
    for style, spec in TONE_STYLES.items():
        for kind in ("start", "end", "middle"):
            notes = spec["notes"] if kind != "middle" else (spec["low"],)
            if kind == "end":
                notes = tuple(reversed(notes))
            duration = 0.18 if kind != "middle" else 0.20
            samples = [0] * int(rate * 0.025)
            for index, frequency in enumerate(notes):
                count = int(rate * duration)
                for i in range(count):
                    # Smooth attack/release avoid clicks; peak is approximately -18 dBFS.
                    envelope = min(1, i / (rate * 0.025), (count - 1 - i) / (rate * 0.04))
                    phase = 2 * math.pi * frequency * i / rate
                    signal = math.sin(phase)
                    if style == "bell":
                        signal = (signal + .25 * math.sin(phase * 2.76)) / 1.25
                        envelope *= math.exp(-3 * i / count)
                    elif style == "wooden":
                        signal = (signal + .3 * math.sin(phase * 3)) / 1.3
                        envelope *= math.exp(-5 * i / count)
                    elif style == "sonar":
                        envelope *= math.sin(math.pi * i / count)
                    samples.append(round(32767 * 0.125 * envelope * signal))
                if index < len(notes) - 1:
                    samples.extend([0] * int(rate * 0.055))
            samples.extend([0] * int(rate * 0.025))
            with wave.open(str(ASSET_DIR / f"{style}-{kind}.wav"), "wb") as output:
                output.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                output.writeframes(struct.pack(f"<{len(samples)}h", *samples))


if __name__ == "__main__":
    generate_assets()
