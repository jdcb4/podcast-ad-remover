import base64
import json
import logging
import asyncio
import os
import sys
import gc
import importlib.util
import wave
from typing import List, Dict
from app.core.config import settings
import httpx

logger = logging.getLogger(__name__)


def piper_tts_available() -> bool:
    if os.getenv("TTS_ENABLED", "1").lower() in {"0", "false", "no", "off"}:
        return False
    return importlib.util.find_spec("piper") is not None


class RateLimitError(Exception):
    """Custom exception for API rate limit errors with retry timing info."""

    def __init__(self, message: str, is_daily_limit: bool = True, provider: str = "gemini"):
        super().__init__(message)
        self.is_daily_limit = is_daily_limit  # True = wait until midnight PT, False = short retry
        self.provider = provider
        self.original_message = message

    def get_next_retry_time(self):
        """Calculate appropriate retry time based on limit type."""
        from datetime import datetime, timedelta
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            # Fallback for older Python
            import pytz
            ZoneInfo = lambda tz: pytz.timezone(tz)

        if self.is_daily_limit:
            # Daily limit: retry at midnight Pacific Time + 5 min buffer
            pacific = ZoneInfo('America/Los_Angeles')
            now_pt = datetime.now(pacific)
            # Next midnight
            midnight_pt = now_pt.replace(hour=0, minute=5, second=0, microsecond=0)
            if midnight_pt <= now_pt:
                midnight_pt += timedelta(days=1)
            # Convert to naive UTC for database storage
            return midnight_pt.astimezone(ZoneInfo('UTC')).replace(tzinfo=None)
        else:
            # Per-minute limit: short 2-minute retry
            return datetime.now() + timedelta(minutes=2)

class Transcriber:
    def __init__(self):
        self.model = None
        self.model_config = None

    def _load_runtime_settings(self) -> Dict:
        runtime = {
            "whisper_model": "base",
            "whisper_cpu_threads": 0,
            "ffmpeg_threads": 0,
        }
        try:
            from app.infra.database import get_db_connection
            with get_db_connection() as conn:
                row = conn.execute("""
                    SELECT whisper_model, whisper_cpu_threads, ffmpeg_threads
                    FROM app_settings WHERE id = 1
                """).fetchone()
                if row:
                    runtime["whisper_model"] = row["whisper_model"] or "base"
                    runtime["whisper_cpu_threads"] = int(row["whisper_cpu_threads"] or 0)
                    runtime["ffmpeg_threads"] = int(row["ffmpeg_threads"] or 0)
        except Exception as e:
            logger.warning(f"Failed to fetch runtime settings, using defaults: {e}")
        return runtime

    def unload_model(self):
        if self.model is None:
            return
        logger.info("Unloading Faster-Whisper model from memory.")
        self.model = None
        self.model_config = None
        gc.collect()

    def load_model(self, runtime_settings: Dict | None = None):
        runtime_settings = runtime_settings or self._load_runtime_settings()
        idx = runtime_settings.get("whisper_model") or "base"
        cpu_threads = int(runtime_settings.get("whisper_cpu_threads") or 0)
        desired_config = (idx, cpu_threads)
        if self.model and self.model_config != desired_config:
            logger.info("Whisper settings changed; reloading Faster-Whisper model.")
            self.unload_model()

        if not self.model:
            # Use float32 for maximum compatibility and stability on CPU (especially ARM64)
            compute_type = "float32"
            logger.info(f"Loading Faster-Whisper model: {idx} (Download Root: {settings.MODELS_DIR})")
            logger.info(f"Using {compute_type} compute type for optimization.")
            if cpu_threads > 0:
                logger.info(f"Limiting Faster-Whisper CPU threads to {cpu_threads}.")

            import time
            start_load = time.time()

            model_kwargs = {
                "device": "cpu",
                "compute_type": compute_type,
                "download_root": settings.MODELS_DIR,
            }
            if cpu_threads > 0:
                model_kwargs["cpu_threads"] = cpu_threads

            # Use float32 for stability
            import faster_whisper
            self.model = faster_whisper.WhisperModel(idx, **model_kwargs)
            self.model_config = desired_config

            load_duration = time.time() - start_load
            logger.info(f"Model loaded in {load_duration:.2f}s")

    def transcribe(self, audio_path: str, progress_callback=None) -> Dict:
        from app.core.audio import AudioProcessor

        runtime_settings = self._load_runtime_settings()
        self.load_model(runtime_settings)
        ffmpeg_threads = int(runtime_settings.get("ffmpeg_threads") or 0)

        # Get total duration for progress calculation
        audio_duration = AudioProcessor.get_duration(audio_path)
        logger.info(f"Transcribing {audio_path} (Duration: {audio_duration:.2f}s)...")

        # Determine if we should use chunked transcription
        # Threshold: 20 minutes (1200 seconds)
        chunk_threshold = 1200.0
        if audio_duration > chunk_threshold:
            logger.info("File exceeds duration threshold. Using chunked transcription.")
            return self._transcribe_chunked(audio_path, audio_duration, progress_callback, ffmpeg_threads=ffmpeg_threads)

        # Prepare clean audio for transcription to avoid crashes with multi-stream files (MJPEG etc)
        # We use a temporary file for the clean audio
        clean_audio_path = audio_path + ".clean.wav"
        AudioProcessor.prepare_for_transcription(audio_path, clean_audio_path, ffmpeg_threads=ffmpeg_threads)

        try:
            # faster-whisper returns a generator
            # We transcribe the CLEAN audio path
            segments_generator, info = self.model.transcribe(
                clean_audio_path,
                beam_size=5
            )

            logger.info(f"Detected language: {info.language} with probability {info.language_probability}")

            segments_result = []

            # Helper to convert segment to dict
            def segment_to_dict(seg):
                return {
                    "id": seg.id,
                    "seek": seg.seek,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "tokens": seg.tokens,
                    "temperature": seg.temperature,
                    "avg_logprob": seg.avg_logprob,
                    "compression_ratio": seg.compression_ratio,
                    "no_speech_prob": seg.no_speech_prob
                }

            # Iterate generator
            for segment in segments_generator:
                if progress_callback:
                    # Progress based on segment end time
                    progress_callback(segment.end, audio_duration)

                # logger.info(f"Segment: {segment.start:.2f}s - {segment.end:.2f}s")
                segments_result.append(segment_to_dict(segment))

            result = {
                "text": "".join([s['text'] for s in segments_result]),
                "segments": segments_result,
                "language": info.language
            }

            logger.info(f"Transcription complete. Found {len(segments_result)} segments.")

            return result
        finally:
            # Clean up temporary audio file
            if os.path.exists(clean_audio_path):
                try:
                    os.remove(clean_audio_path)
                    logger.info("Cleaned up temporary transcription audio.")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp audio: {e}")

    def _transcribe_chunked(self, audio_path: str, total_duration: float, progress_callback=None, ffmpeg_threads: int = 0) -> Dict:
        from app.core.audio import AudioProcessor

        # Chunk settings
        chunk_duration = 1200.0 # 20 mins
        overlap = 20.0 # 20s overlap

        # Stage 1: Normalize original audio (same as single file logic)
        clean_audio_path = audio_path + ".clean.wav"
        AudioProcessor.prepare_for_transcription(audio_path, clean_audio_path, ffmpeg_threads=ffmpeg_threads)

        chunk_paths = []
        try:
            # Stage 2: Create chunks
            chunk_paths = AudioProcessor.create_audio_chunks(clean_audio_path, chunk_duration, overlap, ffmpeg_threads=ffmpeg_threads)
            logger.info(f"Created {len(chunk_paths)} chunks for processing.")

            all_segments = []

            # Stage 3: Process each chunk
            for i, chunk_path in enumerate(chunk_paths):
                logger.info(f"Processing chunk {i+1}/{len(chunk_paths)}: {chunk_path}")

                # Global start time for this chunk
                # Start: (n) * (C - O)
                global_start_time = i * (chunk_duration - overlap)

                # Define merge boundaries for this chunk
                # We keep segments that START within [Boundary-Start, Boundary-End]
                # Boundary-Start: global_start_time + overlap/2 (except first chunk)
                # Boundary-End: global_start_time + chunk_duration - overlap/2 (except last chunk)

                merge_start = global_start_time + (overlap / 2.0) if i > 0 else 0.0
                merge_end = global_start_time + chunk_duration - (overlap / 2.0) if i < (len(chunk_paths) - 1) else total_duration + 1.0

                logger.debug(f"Chunk {i} boundaries: {merge_start:.2f}s to {merge_end:.2f}s")

                # Transcribe chunk
                segments_generator, info = self.model.transcribe(chunk_path, beam_size=5)

                chunk_segments_count = 0
                for segment in segments_generator:
                    # Globalize segment timestamps
                    seg_start = segment.start + global_start_time
                    seg_end = segment.end + global_start_time

                    # Filter based on merge boundaries
                    if seg_start >= merge_start and seg_start < merge_end:
                        # Convert to dict and update timestamps
                        seg_dict = {
                            "id": len(all_segments), # New ID for merged list
                            "seek": segment.seek, # seek is relative to chunk, maybe not useful merged
                            "start": seg_start,
                            "end": seg_end,
                            "text": segment.text,
                            "tokens": segment.tokens,
                            "temperature": segment.temperature,
                            "avg_logprob": segment.avg_logprob,
                            "compression_ratio": segment.compression_ratio,
                            "no_speech_prob": segment.no_speech_prob
                        }
                        all_segments.append(seg_dict)
                        chunk_segments_count += 1

                        # Trigger overall progress callback
                        if progress_callback:
                            progress_callback(seg_end, total_duration)

                logger.info(f"Chunk {i} complete. Added {chunk_segments_count} segments.")

            # Final result
            result = {
                "text": "".join([s['text'] for s in all_segments]),
                "segments": all_segments,
                "language": "en" # Default or detected from first chunk?
            }

            logger.info(f"Chunked transcription complete. Found {len(all_segments)} total segments.")
            return result

        finally:
            # Cleanup chunks and normalized file
            for p in chunk_paths:
                if os.path.exists(p):
                    os.remove(p)
            if os.path.exists(clean_audio_path):
                os.remove(clean_audio_path)
            logger.info("Cleaned up temporary chunk files.")


class LLMProvider:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def list_models(self) -> List[str]:
        raise NotImplementedError

    def test_connection(self) -> Dict:
        try:
            # Simple hello world test
            response = self.generate("Say hello")
            return {"status": "ok", "response": response[:100]}
        except Exception as e:
            return {"status": "error", "error": str(e)}

class OpenAIProvider(LLMProvider):
    RATE_LIMIT_PATTERNS = (
        'resource_exhausted',
        'quota exceeded',
        'rate limit',
        '429',
        'too many requests',
        'resourceexhausted',
    )

    def __init__(
        self,
        api_key: str | List[str],
        models: List[str],
        base_url: str = None,
        provider_name: str = "OpenAI/Compatible",
        model_prefixes: tuple[str, ...] | None = ("gpt-", "o1-", "chatgpt-"),
        rate_limit_provider: str = "openai",
    ):
        import openai
        self.openai = openai
        self.api_keys = api_key if isinstance(api_key, list) else [api_key]
        self.current_key_idx = 0
        self.models = models
        self.base_url = base_url
        self.provider_name = provider_name
        self.model_prefixes = model_prefixes
        self.rate_limit_provider = rate_limit_provider
        self.is_openrouter = base_url and "openrouter" in base_url
        self._init_client()

    def _init_client(self):
        key = self.api_keys[self.current_key_idx]
        masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "***"
        logger.info(f"{self.provider_name}: Initializing client with key #{self.current_key_idx + 1} ({masked})")
        self.client = self.openai.OpenAI(api_key=key, base_url=self.base_url)

    def _rotate_key(self) -> bool:
        if self.current_key_idx + 1 < len(self.api_keys):
            self.current_key_idx += 1
            logger.warning(f"{self.provider_name}: Rate limit hit. Rotating to key #{self.current_key_idx + 1}...")
            self._init_client()
            return True
        return False

    def _is_rate_limit(self, error: Exception) -> bool:
        error_str = str(error).lower()
        return any(pattern in error_str for pattern in self.RATE_LIMIT_PATTERNS)

    def generate(self, prompt: str) -> str:
        last_error = None

        while True:
            all_rate_limited = True

            for model in self.models:
                try:
                    logger.info(f"{self.provider_name}: Using model {model} with key #{self.current_key_idx + 1}...")
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.choices[0].message.content or ""
                except Exception as e:
                    logger.warning(f"{self.provider_name} model {model} failed: {e}")
                    last_error = e
                    if not self._is_rate_limit(e):
                        all_rate_limited = False

            if all_rate_limited and self._rotate_key():
                continue

            if all_rate_limited:
                raise RateLimitError(
                    f"{self.provider_name} rate limit exceeded on all keys and models. Last error: {last_error}",
                    is_daily_limit=True,
                    provider=self.rate_limit_provider,
                )

            raise Exception(f"All {self.provider_name} models failed. Last error: {last_error}")

    def list_models(self) -> List[str]:
        try:
            models = self.client.models.list()
            model_ids = []
            for model in models.data:
                model_id = model.id
                if model_id.startswith("models/"):
                    model_id = model_id.replace("models/", "", 1)
                model_ids.append(model_id)

            if self.model_prefixes is None:
                return sorted(model_ids)
            return sorted([m for m in model_ids if m.startswith(self.model_prefixes)])
        except Exception as e:
            logger.error(f"{self.provider_name}: Failed to list models: {e}")
            return []

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, models: List[str]):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.models = models

    def generate(self, prompt: str) -> str:
        last_error = None
        for model in self.models:
            try:
                logger.info(f"Anthropic: Using model {model}...")
                response = self.client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                last_error = e
        raise Exception(f"All models failed. Last error: {last_error}")

    def list_models(self) -> List[str]:
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]

class AdDetector:
    GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GEMINI_REST_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    # Default Gemini Cascade
    DEFAULT_GEMINI_MODELS = [
        'gemini-3.5-flash',
        'gemini-3-flash',
        'gemini-3.1-flash-lite',
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite',
    ]
    DEFAULT_OPENROUTER_MODELS = [
        'google/gemini-3.5-flash',
        'google/gemini-3-flash',
        'google/gemini-3.1-flash-lite',
        'google/gemini-2.5-flash',
        'google/gemini-2.5-flash-lite',
    ]
    DEFAULT_GEMINI_TTS_MODELS = [
        'gemini-3.1-flash-tts-preview',
        'gemini-2.5-flash-preview-tts',
    ]
    GEMINI_TTS_VOICES = {'Orus', 'Enceladus', 'Laomedeia'}

    def __init__(self):
        self.settings = self._load_settings()

    def _load_settings(self):
        from app.infra.database import get_db_connection
        try:
            with get_db_connection() as conn:
                row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
                if row: return dict(row)
        except Exception:
            pass
        return {}

    def _parse_model_setting(self, value: str, default: List[str]) -> List[str]:
        """Helper to parse DB setting which might be a JSON list or a single string"""
        if not value: return default
        try:
            # Try parsing as JSON list
            parsed = json.loads(value)
            if isinstance(parsed, list): return parsed
            return [str(parsed)] # Single JSON value?
        except json.JSONDecodeError:
            # Fallback: Treat as simple string (legacy)
            return [value]

    def _get_gemini_api_keys(self) -> List[str]:
        api_keys = []

        db_keys_json = self.settings.get('gemini_api_keys')
        if db_keys_json:
            try:
                parsed = json.loads(db_keys_json)
                if isinstance(parsed, list):
                    api_keys.extend([k.strip() for k in parsed if isinstance(k, str) and k.strip()])
            except Exception:
                pass

        legacy_key = self.settings.get('gemini_api_key')
        if legacy_key and legacy_key not in api_keys:
            api_keys.append(legacy_key)

        if settings.GEMINI_API_KEY:
            env_keys = [k.strip() for k in settings.GEMINI_API_KEY.split(',') if k.strip()]
            for key in env_keys:
                if key not in api_keys:
                    api_keys.append(key)

        return api_keys

    def create_provider(self, provider_type: str, api_key: str = None, model: str = None, openrouter_key: str = None) -> LLMProvider:
        """Factory to create a provider instance."""
        explicit_api_key = api_key

        # Resolve keys (DB Overrides Env)
        if not api_key:
            db_key = None
            if provider_type == 'gemini':
                # For Gemini, try the new gemini_api_keys (JSON array) first
                db_keys_json = self.settings.get('gemini_api_keys')
                if db_keys_json:
                    try:
                        parsed = json.loads(db_keys_json)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            # Return first key for compatibility, but full list used below
                            db_key = parsed[0]
                    except:
                        pass
                # Fallback to legacy single key field
                if not db_key:
                    db_key = self.settings.get('gemini_api_key')
            elif provider_type == 'openai': db_key = self.settings.get('openai_api_key')
            elif provider_type == 'anthropic': db_key = self.settings.get('anthropic_api_key')
            elif provider_type == 'openrouter': db_key = self.settings.get('openrouter_api_key')

            # 2. Try Env second
            env_key = None
            if provider_type == 'gemini': env_key = settings.GEMINI_API_KEY
            elif provider_type == 'openai': env_key = settings.OPENAI_API_KEY
            elif provider_type == 'anthropic': env_key = settings.ANTHROPIC_API_KEY
            elif provider_type == 'openrouter': env_key = settings.OPENROUTER_API_KEY

            # Priority: DB > Env
            api_key = db_key if db_key else env_key

        if not api_key:
             raise ValueError(f"No API key found for {provider_type} (Check Admin Settings or Environment Variables)")

        # Resolve models (handle passed 'model' arg or DB settings)
        # If explicit 'model' arg is passed (e.g. from test tool), wrap it in list
        if model:
            # Check if it looks like a JSON list
            try:
                parsed = json.loads(model)
                if isinstance(parsed, list): models_list = parsed
                else: models_list = [model]
            except:
                models_list = [model]
        else:
            # Load from DB
            if provider_type == 'openai':
                models_list = self._parse_model_setting(self.settings.get('openai_model'), ['gpt-4o'])
            elif provider_type == 'anthropic':
                models_list = self._parse_model_setting(self.settings.get('anthropic_model'), ['claude-3-5-sonnet-20241022'])
            elif provider_type == 'openrouter':
                models_list = self._parse_model_setting(self.settings.get('openrouter_model'), self.DEFAULT_OPENROUTER_MODELS)
            else: # Gemini
                models_list = self._parse_model_setting(self.settings.get('ai_model_cascade'), self.DEFAULT_GEMINI_MODELS)

        if provider_type == 'openai':
            return OpenAIProvider(api_key, models_list, provider_name="OpenAI", rate_limit_provider="openai")

        elif provider_type == 'anthropic':
            return AnthropicProvider(api_key, models_list)

        elif provider_type == 'openrouter':
            return OpenAIProvider(
                api_key,
                models_list,
                base_url="https://openrouter.ai/api/v1",
                provider_name="OpenRouter",
                model_prefixes=None,
                rate_limit_provider="openrouter",
            )

        else: # Gemini
            # For Gemini, build the full keys list from all sources
            api_keys = []

            if explicit_api_key:
                api_keys.extend([k.strip() for k in explicit_api_key.split(',') if k.strip()])

            # 1. DB keys (gemini_api_keys JSON array)
            if not explicit_api_key:
                db_keys_json = self.settings.get('gemini_api_keys')
                if db_keys_json:
                    try:
                        parsed = json.loads(db_keys_json)
                        if isinstance(parsed, list):
                            api_keys.extend([k for k in parsed if k and k.strip()])
                    except:
                        pass

                # 2. Legacy single key from DB
                legacy_key = self.settings.get('gemini_api_key')
                if legacy_key and legacy_key not in api_keys:
                    api_keys.append(legacy_key)

                # 3. Environment variable (can be comma-separated)
                if settings.GEMINI_API_KEY:
                    env_keys = [k.strip() for k in settings.GEMINI_API_KEY.split(',') if k.strip()]
                    for k in env_keys:
                        if k not in api_keys:
                            api_keys.append(k)

            # If still no keys, use the api_key that was resolved above
            if not api_keys:
                api_keys = [api_key]

            return OpenAIProvider(
                api_keys,
                models_list,
                base_url=self.GEMINI_OPENAI_BASE_URL,
                provider_name="Gemini",
                model_prefixes=("gemini-",),
                rate_limit_provider="gemini",
            )

    def _get_provider(self) -> LLMProvider:
        # Use current settings
        provider_type = self.settings.get('active_ai_provider', 'gemini')
        return self.create_provider(provider_type)

    def detect_ads(self, transcript: Dict, options: Dict = None, whitelist_mode: bool = False) -> List[Dict[str, float]]:
        self.settings = self._load_settings()
        if not options:
            options = {
                "remove_ads": True, "remove_promos": True, "remove_intros": False, "remove_outros": False, "custom_instructions": None
            }

        # Prepare transcript text
        text_data = ""
        for seg in transcript['segments']:
            text_data += f"[{seg['start']:.2f}-{seg['end']:.2f}] {seg['text']}\n"

        # Build Prompt
        prompt = self._build_ad_prompt(options, text_data, whitelist_mode=whitelist_mode)

        # Execute
        try:
            provider = self._get_provider()
            response_text = provider.generate(prompt)
            raw_segments = self._parse_ad_response(response_text)

            # Whitelist mode: return ALL segments (including Content) for processor to invert
            if whitelist_mode:
                logger.info(f"Whitelist mode: returning all {len(raw_segments)} segments (including Content)")
                return raw_segments

            # Blacklist mode (default): Filter to only include requested types and exclude 'Content'
            removable_labels = []
            if options.get("remove_ads"): removable_labels.append("Ad")
            if options.get("remove_promos"):
                removable_labels.extend(["Promo", "Cross-promotion"])
            if options.get("remove_intros"): removable_labels.append("Intro")
            if options.get("remove_outros"): removable_labels.append("Outro")

            filtered = []
            for s in raw_segments:
                label = s.get('label', 'Ad')
                if label in removable_labels:
                    filtered.append(s)
                else:
                    logger.info(f"Skipping segment labeled '{label}' (Reason: {s.get('reason')})")

            return filtered
        except Exception as e:
            logger.error(f"Ad detection failed: {e}")
            raise e

    def generate_summary(self, transcript: Dict, podcast_name: str, episode_title: str, pub_date: str, subscription_settings: Dict = None) -> str:
        self.settings = self._load_settings()
        text_data = ""
        for seg in transcript['segments']:
            text_data += f"{seg['text']} "

        # Build targets list from subscription settings
        targets = []
        if subscription_settings:
            if subscription_settings.get('remove_ads'):
                targets.append(self.settings.get('ad_target_sponsor') or 'Sponsor messages, Ad reads')
            if subscription_settings.get('remove_promos'):
                targets.append(self.settings.get('ad_target_promo') or 'Cross-promotions, plugs for other shows')
            if subscription_settings.get('remove_intros'):
                targets.append(self.settings.get('ad_target_intro') or 'Intro music, opening segments')
            if subscription_settings.get('remove_outros'):
                targets.append(self.settings.get('ad_target_outro') or 'Outro music, closing segments')

        targets_text = "\n".join(targets) if targets else "None"

        # Build Prompt (use default if None or empty in database)
        db_template = self.settings.get('summary_prompt_template')
        template = db_template if db_template else """
        You are a smart assistant. Write a short 2-3 sentence summary of this podcast episode.
        The summary must:
        1. NOT mention the podcast name, episode title, or date.
        2. DO NOT summarize anything relating to {targets}.
        3. Start immediately with "This episode includes".
        4. Briefly summarize key topics.
        Transcript Context: {transcript_context}
        """

        # Ensure template is a string (defensive)
        if template is None:
            template = "Summarize this: {transcript_context}"


        try:
             prompt = template.format(transcript_context=text_data[:100000], targets=targets_text)
        except KeyError:
             prompt = template # Fallback

        try:
            provider = self._get_provider()
            return provider.generate(prompt).strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Welcome to {podcast_name}. Today's episode is {episode_title}."

    # --- Helpers ---
    def _build_ad_prompt(self, options, transcript_text, whitelist_mode: bool = False):
        # Fetch targets with safety defaults
        targets = []
        if options.get("remove_ads"):
            targets.append(self.settings.get('ad_target_sponsor') or 'Sponsor messages')
        if options.get("remove_promos"):
            targets.append(self.settings.get('ad_target_promo') or 'Promos')
        if options.get("remove_intros"):
            targets.append(self.settings.get('ad_target_intro') or 'Intro')
        if options.get("remove_outros"):
            targets.append(self.settings.get('ad_target_outro') or 'Outro')

        default_base = """
        Identify segments in the transcript that match the Targets.
        Targets: {targets}
        {custom_instr}
        Return a JSON array of objects with "start", "end", "label" (Ad/Promo/Intro/Outro), and "reason" (brief explanation).
        Example: [{{"start": 0.0, "end": 10.0, "label": "Ad", "reason": "Sponsor read for XYZ"}}]
        """

        base = self.settings.get('ad_prompt_base') or default_base

        custom = f"Custom: {options.get('custom_instructions')}" if options.get('custom_instructions') else ""

        # Whitelist mode: append instructions to also label Content segments
        whitelist_addendum = ""
        if whitelist_mode:
            whitelist_addendum = """
IMPORTANT: You MUST also label ALL substantive speech/content segments with label "Content".
Every segment of the transcript must be classified. Use "Content" for any segment containing substantive speech, interviews, reporting, or discussion that is NOT an ad, promo, intro, or outro.
Non-speech segments (music, jingles, silence) should NOT be labeled as Content.
Example: [{"start": 10.0, "end": 300.0, "label": "Content", "reason": "Main discussion segment"}]
"""

        # Use manual replacement instead of .format() to avoid breaking on JSON examples
        try:
            prompt = base.replace("{targets}", "\n".join(targets)).replace("{custom_instr}", custom)
            return prompt + whitelist_addendum + "\n\nTranscript:\n" + transcript_text
        except Exception as e:
             logger.warning(f"Prompt formatting failed: {e}")
             return base + whitelist_addendum + "\n\nTranscript:\n" + transcript_text

    def _parse_ad_response(self, text: str):
        text = text.strip()
        # Common markdown cleanup
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return self._normalize_ad_segments(json.loads(text))
        except json.JSONDecodeError:
            # Try to find JSON array pattern
            import re
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    return self._normalize_ad_segments(json.loads(match.group(0)))
                except: pass

            logger.error(f"Failed to parse JSON response: {text[:200]}...")
            return []

    def _normalize_ad_segments(self, payload) -> List[Dict[str, float]]:
        if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
            payload = payload["segments"]

        if not isinstance(payload, list):
            logger.warning("Ad detector response was not a JSON array.")
            return []

        normalized = []
        for item in payload:
            if not isinstance(item, dict):
                continue

            try:
                start = float(item["start"])
                end = float(item["end"])
            except (KeyError, TypeError, ValueError):
                logger.warning(f"Skipping ad segment with invalid timestamps: {item}")
                continue

            if start < 0:
                start = 0.0
            if end <= start:
                logger.warning(f"Skipping ad segment with non-positive duration: {item}")
                continue

            normalized.append({
                "start": start,
                "end": end,
                "label": str(item.get("label") or "Ad"),
                "reason": str(item.get("reason") or ""),
            })

        return normalized

    # Static method to list Gemini models
    @staticmethod
    def list_gemini_models():
        # Priority: DB (gemini_api_keys) > DB (gemini_api_key) > Env
        api_key = None
        try:
            from app.infra.database import get_db_connection
            with get_db_connection() as conn:
                row = conn.execute("SELECT gemini_api_keys, gemini_api_key FROM app_settings WHERE id = 1").fetchone()
                if row:
                    # Try new multi-key field first
                    if row['gemini_api_keys']:
                        try:
                            parsed = json.loads(row['gemini_api_keys'])
                            if isinstance(parsed, list) and len(parsed) > 0:
                                api_key = parsed[0]
                        except:
                            pass
                    # Fallback to legacy single key
                    if not api_key and row['gemini_api_key']:
                        api_key = row['gemini_api_key']
        except: pass

        # Fallback to env
        if not api_key:
            api_key = settings.GEMINI_API_KEY

        if not api_key:
            return []

        # Handle env variable with multiple comma-separated keys (use first one)
        if ',' in api_key:
            api_key = api_key.split(',')[0].strip()

        try:
            return OpenAIProvider(
                api_key,
                [],
                base_url=AdDetector.GEMINI_OPENAI_BASE_URL,
                provider_name="Gemini",
                model_prefixes=("gemini-",),
                rate_limit_provider="gemini",
            ).list_models()
        except Exception as e:
            logger.error(f"Failed to list Gemini models: {e}")
            return []

    def has_valid_config(self) -> bool:
        """Check if any API provider is configured via DB or Env."""
        # Check Gemini - both new multi-key and legacy single-key fields
        gemini_keys_json = self.settings.get('gemini_api_keys')
        if gemini_keys_json:
            try:
                parsed = json.loads(gemini_keys_json)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return True
            except:
                pass
        if self.settings.get('gemini_api_key') or settings.GEMINI_API_KEY:
            return True

        # Check others
        s = self.settings
        if s.get('openai_api_key') or settings.OPENAI_API_KEY: return True
        if s.get('anthropic_api_key') or settings.ANTHROPIC_API_KEY: return True
        if s.get('openrouter_api_key') or settings.OPENROUTER_API_KEY: return True

        return False

    def _clean_tts_text(self, text: str) -> str:
        text = text or ""
        chars_to_remove = ['"', '*', '_', '#', '\u201c', '\u201d', '\u2018', '\u2019']
        for char in chars_to_remove:
            text = text.replace(char, '')
        return text.strip()

    def _extract_gemini_tts_audio(self, payload: Dict) -> bytes:
        candidates = payload.get("candidates") or []
        for candidate in candidates:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                inline_data = part.get("inlineData") or part.get("inline_data") or {}
                data = inline_data.get("data")
                if data:
                    if isinstance(data, bytes):
                        return data
                    return base64.b64decode(data)
        raise RuntimeError("Gemini TTS response did not contain audio data.")

    def _write_pcm_wav(self, output_path: str, pcm_audio: bytes):
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with wave.open(output_path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(pcm_audio)

    async def _generate_gemini_tts(self, text: str, output_path: str):
        api_keys = self._get_gemini_api_keys()
        if not api_keys:
            raise RuntimeError("Gemini TTS selected but no Gemini API key is configured.")

        models = self._parse_model_setting(
            self.settings.get("gemini_tts_model_cascade"),
            self.DEFAULT_GEMINI_TTS_MODELS,
        )
        models = [model for model in models if model]
        if not models:
            models = self.DEFAULT_GEMINI_TTS_MODELS

        voice = self.settings.get("gemini_tts_voice") or "Orus"
        if voice not in self.GEMINI_TTS_VOICES:
            logger.warning(f"Unknown Gemini TTS voice '{voice}', falling back to Orus.")
            voice = "Orus"

        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": voice}
                    }
                },
            },
        }

        last_error = None
        async with httpx.AsyncClient(timeout=120) as client:
            for model in models:
                url = f"{self.GEMINI_REST_BASE_URL}/models/{model}:generateContent"
                for key_idx, api_key in enumerate(api_keys):
                    try:
                        logger.info(f"Generating TTS with Gemini model {model}, key #{key_idx + 1}.")
                        response = await client.post(
                            url,
                            headers={
                                "x-goog-api-key": api_key,
                                "Content-Type": "application/json",
                            },
                            json=payload,
                        )
                        if response.status_code >= 400:
                            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:500]}")

                        audio = self._extract_gemini_tts_audio(response.json())
                        self._write_pcm_wav(output_path, audio)
                        logger.info("Gemini TTS generation completed.")
                        return
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Gemini TTS model {model} failed with key #{key_idx + 1}: {e}")

        raise RuntimeError(f"All Gemini TTS models failed. Last error: {last_error}")

    async def validate_tts(self):
        """
        Check if TTS service is available and model is ready.
        """
        self.settings = self._load_settings()
        tts_provider = self.settings.get("tts_provider") or "piper"
        if tts_provider == "gemini":
            if not self._get_gemini_api_keys():
                raise RuntimeError("Gemini TTS is selected but no Gemini API key is configured.")
            models = self._parse_model_setting(
                self.settings.get("gemini_tts_model_cascade"),
                self.DEFAULT_GEMINI_TTS_MODELS,
            )
            if not models:
                raise RuntimeError("Gemini TTS is selected but no TTS models are configured.")
            logger.info("Gemini TTS validation skipped live API call to avoid consuming speech quota.")
            return True

        if not piper_tts_available():
            raise RuntimeError("Piper TTS is not installed or is disabled in this image.")

        try:
             # Fetch configured voice model
            piper_model_file = "en_GB-cori-high.onnx"
            try:
                from app.infra.database import get_db_connection
                with get_db_connection() as conn:
                    row = conn.execute("SELECT piper_model FROM app_settings WHERE id = 1").fetchone()
                    if row and row['piper_model']:
                        piper_model_file = row['piper_model']
            except: pass

            # Ensure model exists
            await self._ensure_piper_model(piper_model_file)

            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "tts_worker.py"))

            proc = await asyncio.create_subprocess_exec(
                sys.executable, script_path, "--check",
                "--model", piper_model_file,
                "--models-dir", settings.MODELS_DIR,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                error_msg = stderr.decode().strip()
                logger.error(f"TTS Validation Failed: {error_msg}")
                raise Exception(f"TTS Health Check Failed: {error_msg}")

            logger.info("TTS Validation Passed.")
            return True

        except Exception as e:
            logger.error(f"TTS Validation Error: {e}")
            raise e

    async def generate_audio(self, text: str, output_path: str):
        """
        Generate TTS audio using the configured TTS provider.
        """
        self.settings = self._load_settings()
        text = self._clean_tts_text(text)
        if not text:
            raise RuntimeError("Cannot generate TTS for empty text.")

        tts_provider = self.settings.get("tts_provider") or "piper"
        if tts_provider == "gemini":
            logger.info("Generating TTS (Gemini)...")
            await self._generate_gemini_tts(text, output_path)
            return

        if not piper_tts_available():
            raise RuntimeError("Piper TTS is not installed or is disabled in this image.")

        try:
            logger.info("Generating TTS (Piper in subprocess)...")

            # Clean text for TTS (remove markdown artifacts and quotes)
            # TTS engines often struggle or speak "asterisk" or "quote" aloud
            chars_to_remove = ['"', '*', '“', '”', '‘', '’', '_', '#']
            for char in chars_to_remove:
                text = text.replace(char, '')

            # Fetch configured voice model
            piper_model_file = "en_GB-cori-high.onnx"
            try:
                from app.infra.database import get_db_connection
                with get_db_connection() as conn:
                    row = conn.execute("SELECT piper_model FROM app_settings WHERE id = 1").fetchone()
                    if row and row['piper_model']:
                        piper_model_file = row['piper_model']
            except Exception as e:
                logger.warning(f"Failed to fetch piper setting, using default: {e}")

            # Ensure model exists
            await self._ensure_piper_model(piper_model_file)

            # Resolve absolute path to the worker script
            script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "tts_worker.py"))

            # Run the worker script
            proc = await asyncio.create_subprocess_exec(
                sys.executable, script_path, output_path,
                "--model", piper_model_file,
                "--models-dir", settings.MODELS_DIR,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await proc.communicate(input=text.encode())

            if proc.returncode != 0:
                logger.error(f"TTS worker failed: {stderr.decode()}")
                raise Exception(f"TTS worker failed with exit code {proc.returncode}")

            logger.info("TTS generation completed.")

        except Exception as e:
            logger.error(f"TTS failed: {e}")
            raise e

    async def _ensure_piper_model(self, model_filename: str):
        """Ensures the piper model and its config exist locally."""
        model_dir = os.path.join(settings.MODELS_DIR, "piper")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, model_filename)
        config_path = model_path + ".json"

        if os.path.exists(model_path) and os.path.exists(config_path):
            return model_path

        logger.info(f"Piper model {model_filename} not found locally. Attempting download from HuggingFace...")

        # Base URLs for Piper models on HuggingFace
        # We try to infer the path: lang/lang_REGION/voice/quality/filename
        # Example: en/en_US/amy/medium/en_US-amy-medium.onnx

        parts = model_filename.replace('.onnx', '').split('-')
        if len(parts) >= 3:
            lang_region = parts[0]
            lang = lang_region.split('_')[0]
            voice = parts[1]
            quality = parts[2]

            remote_base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{lang}/{lang_region}/{voice}/{quality}/{model_filename}"
        else:
            # Fallback for non-standard names?
            # Most common ones are like en_GB-cori-high
            logger.warning(f"Could not infer path for {model_filename}, trying direct link fallback")
            # We don't really have a direct link without voices.json, but let's try a common ones
            # For now, let's just fail if we can't infer it, or better, download voices.json
            raise Exception(f"Piper model {model_filename} not found and cannot infer download URL. Please download it manually to {model_dir}")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Download ONNX
            logger.info(f"Downloading {model_filename} from {remote_base}...")
            async with client.stream("GET", remote_base) as response:
                if response.status_code != 200:
                    raise Exception(f"Failed to download Piper model: HTTP {response.status_code}")
                with open(model_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

            # Download JSON
            logger.info(f"Downloading {model_filename}.json...")
            async with client.stream("GET", remote_base + ".json") as response:
                if response.status_code != 200:
                    # Clean up partial ONNX if config fails?
                    if os.path.exists(model_path): os.remove(model_path)
                    raise Exception(f"Failed to download Piper config: HTTP {response.status_code}")
                with open(config_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)

        logger.info(f"Piper model {model_filename} downloaded successfully.")
        return model_path

