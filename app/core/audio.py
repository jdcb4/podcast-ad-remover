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
        
        # [a0][a1]concat=n=2:v=0:a=1[out_concat];[out_concat]aformat=...[out]
        
        filter_parts = []
        concat_inputs = []
        
        for i, (start, end) in enumerate(keep_segments):
            # Add aformat to ensure consistent sample rate and layout for concat
            filter_parts.append(f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS,aformat=sample_rates=44100:channel_layouts=stereo[a{i}]")
            concat_inputs.append(f"[a{i}]")
            
        filter_str = ";".join(filter_parts)
        # Output to intermediate [out_concat], then force format/padding before encoder
        # asetnsamples=n=1152 ensures standard MP3 frame boundaries
        # sample_fmts=s16p ensures we use signed 16-bit planar integers (avoiding float padding issues)
        concat_str = "".join(concat_inputs) + f"concat=n={len(keep_segments)}:v=0:a=1[out_concat]"
        format_str = f"[out_concat]asetnsamples=n=1152,aformat=sample_rates=44100:channel_layouts=stereo:sample_fmts=s16p[out]"
        full_filter = f"{filter_str};{concat_str};{format_str}"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-filter_complex", full_filter,
            "-map", "[out]",
            "-c:a", "libmp3lame",
            "-q:a", "2",
            output_path
        ]
        
        logger.info("Running FFmpeg...")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("FFmpeg completed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed with exit code {e.returncode}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            raise Exception(f"FFmpeg failed: {e.stderr}") from e

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
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Successfully concatenated files to {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to concatenate files: {e.returncode}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            raise Exception(f"FFmpeg concatenation failed: {e.stderr}") from e
    @staticmethod
    def prepare_for_transcription(input_path: str, output_path: str):
        """
        Extract a clean, normalized audio stream for transcription.
        Uses 16kHz mono as preferred by Whisper. Removes all other streams (video, cover art).
        """
        logger.info(f"Preparing clean audio for transcription: {output_path}")
        
        # -map 0:a:0 selects ONLY the first audio stream
        # -ar 16000 sets sample rate to 16kHz
        # -ac 1 sets to mono
        # -vn removes all video/attached images
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-map", "0:a:0",
            "-ar", "16000",
            "-ac", "1",
            "-vn",
            output_path
        ]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info("Normalized audio prepared.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to prepare audio for transcription: {e.stderr}")
            raise Exception(f"Audio preparation failed: {e.stderr}") from e
    @staticmethod
    def create_audio_chunks(input_path: str, chunk_duration: float, overlap: float) -> List[str]:
        """
        Split audio into overlapping chunks.
        Returns list of paths to chunk files.
        """
        total_duration = AudioProcessor.get_duration(input_path)
        logger.info(f"Splitting {input_path} (Total: {total_duration:.2f}s) into {chunk_duration}s chunks with {overlap}s overlap")
        
        chunks = []
        start = 0.0
        chunk_idx = 0
        
        while start < total_duration:
            output_chunk = f"{input_path}.chunk_{chunk_idx:03d}.wav"
            # ffmpeg -ss start -t duration
            # We use a slightly longer duration to ensure we don't miss anything
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-t", str(chunk_duration),
                "-i", input_path,
                "-c", "copy",
                output_chunk
            ]
            
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                chunks.append(output_chunk)
                logger.info(f"Created chunk {chunk_idx}: {start}s to {start + chunk_duration}s")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create chunk {chunk_idx}: {e.stderr}")
                # Cleanup already created chunks
                for c in chunks:
                    if os.path.exists(c):
                        os.remove(c)
                raise Exception(f"Audio chunking failed: {e.stderr}") from e
                
            start += (chunk_duration - overlap)
            chunk_idx += 1
            
            # If the remaining duration is less than the overlap, we are done
            if start >= total_duration - overlap:
                break
                
        return chunks
