"""Opt-in live YouTube smoke test; intentionally excluded from deterministic verification."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.youtube import (
    discover_youtube_entries,
    download_youtube_audio,
    resolve_youtube_source,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve public YouTube sources and download one audio fixture")
    parser.add_argument("--channel", required=True, help="Public YouTube channel URL")
    parser.add_argument("--playlist", required=True, help="Public explicit YouTube playlist URL")
    parser.add_argument("--download-video", required=True, help="Public individual video URL used only as an audio fixture")
    args = parser.parse_args()

    channel = resolve_youtube_source(args.channel)
    playlist = resolve_youtube_source(args.playlist)
    if channel.source_type != "youtube_channel":
        raise RuntimeError("--channel did not resolve as a channel")
    if playlist.source_type != "youtube_playlist":
        raise RuntimeError("--playlist did not resolve as a playlist")

    channel_entries = discover_youtube_entries(channel.source_type, channel.canonical_url)
    playlist_entries = discover_youtube_entries(playlist.source_type, playlist.canonical_url)
    if not channel_entries.entries or not playlist_entries.entries:
        raise RuntimeError("One of the live sources returned no entries")

    with tempfile.TemporaryDirectory(prefix="podcast-ad-remover-youtube-") as temp_dir:
        output = Path(download_youtube_audio(args.download_video, temp_dir))
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError("The live audio-only download did not produce a non-empty file")
        print(
            f"YouTube smoke test passed: channel={channel.external_id}, "
            f"playlist={playlist.external_id}, audio={output.name} ({output.stat().st_size} bytes)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
