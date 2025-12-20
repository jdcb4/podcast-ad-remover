import subprocess
import logging
import os
from typing import List, Dict

logger = logging.getLogger(__name__)

class AudioProcessor:
    @staticmethod
    def get_duration(file_path: str) -> float:
        """Get duration in seconds using ffprobe."""
        cmd = [
            "ffprobe", 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", 
            file_path
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception as e:
            logger.error(f"Failed to get duration: {e}")
            return 0.0

    @staticmethod
    def remove_segments(input_path: str, output_path: str, remove_segments: List[Dict[str, float]]):
        """
        Remove specified segments from audio.
        Logic: Calculate 'keep' segments and concatenate them.
        """
        if not remove_segments:
            # Just copy if no ads
            logger.info("No ads to remove, copying file.")
            subprocess.run(["ffmpeg", "-y", "-i", input_path, "-c", "copy", output_path], check=True)
            return

        total_duration = AudioProcessor.get_duration(input_path)
        keep_segments = []
        current_time = 0.0
        
        # Sort segments by start time
        sorted_segments = sorted(remove_segments, key=lambda x: x['start'])
        
        for seg in sorted_segments:
            start = seg['start']
            end = seg['end']
            
            if start > current_time:
                keep_segments.append((current_time, start))
            
            current_time = max(current_time, end)
            
        if current_time < total_duration:
            keep_segments.append((current_time, total_duration))
            
        logger.info(f"Keeping segments: {keep_segments}")
        
        # Construct filter complex
        # [0:a]atrim=start=0:end=10,asetpts=PTS-STARTPTS[a0];
        # [0:a]atrim=start=20:end=30,asetpts=PTS-STARTPTS[a1];
        # [a0][a1]concat=n=2:v=0:a=1[out]
        
        filter_parts = []
        concat_inputs = []
        
        for i, (start, end) in enumerate(keep_segments):
            filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]")
            concat_inputs.append(f"[a{i}]")
            
        filter_str = ";".join(filter_parts)
        concat_str = "".join(concat_inputs) + f"concat=n={len(keep_segments)}:v=0:a=1[out]"
        full_filter = f"{filter_str};{concat_str}"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter_complex", full_filter,
            "-map", "[out]",
            output_path
        ]
        
        logger.info("Running FFmpeg...")
        subprocess.run(cmd, check=True)

    @staticmethod
    def prepend_audio(main_audio_path: str, intro_audio_path: str, output_path: str):
        """Prepend intro audio to main audio."""
        logger.info(f"Prepending {intro_audio_path} to {main_audio_path}...")
        
        # We need to ensure formats are compatible. Simplest is to re-encode both to a common format or use filter complex.
        # [0:a][1:a]concat=n=2:v=0:a=1[out]
        # input 0 is intro, input 1 is main
        
        cmd = [
            "ffmpeg", "-y",
            "-i", intro_audio_path,
            "-i", main_audio_path,
            "-filter_complex", "[0:a]aformat=sample_rates=44100:channel_layouts=stereo[a0];[1:a]aformat=sample_rates=44100:channel_layouts=stereo[a1];[a0][a1]concat=n=2:v=0:a=1[out]",
            "-map", "[out]",
            output_path
        ]
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully prepended audio to {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to prepend audio: {e}")
            raise
            raise
            
    @staticmethod
    def concat_files(output_path: str, input_paths: List[str]):
        """Concatenate multiple audio files."""
        if not input_paths:
            return
            
        logger.info(f"Concatenating {len(input_paths)} files to {output_path}...")
        
        # Build filter complex
        # [0:a]aformat=...[a0];[1:a]aformat=...[a1];[a0][a1]concat=n=2:v=0:a=1[out]
        
        filter_parts = []
        concat_inputs = []
        cmd = ["ffmpeg", "-y"]
        
        for i, path in enumerate(input_paths):
            cmd.extend(["-i", path])
            filter_parts.append(f"[{i}:a]aformat=sample_rates=44100:channel_layouts=stereo[a{i}]")
            concat_inputs.append(f"[a{i}]")
            
        filter_str = ";".join(filter_parts)
        concat_str = "".join(concat_inputs) + f"concat=n={len(input_paths)}:v=0:a=1[out]"
        full_filter = f"{filter_str};{concat_str}"
        
        cmd.extend([
            "-filter_complex", full_filter,
            "-map", "[out]",
            output_path
        ])
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully concatenated files to {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to concatenate files: {e}")
            raise
