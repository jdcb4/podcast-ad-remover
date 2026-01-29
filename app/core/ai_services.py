import faster_whisper
from google import genai
import json
import logging
import asyncio
import os
import sys
from typing import List, Dict
from app.core.config import settings
import httpx

logger = logging.getLogger(__name__)

class Transcriber:
    def __init__(self):
        self.model = None

    def load_model(self):
        if not self.model:
            # Check DB for whisper model
            idx = "base"
            try:
                from app.infra.database import get_db_connection
                with get_db_connection() as conn:
                    row = conn.execute("SELECT whisper_model FROM app_settings WHERE id = 1").fetchone()
                    if row and row['whisper_model']:
                        idx = row['whisper_model']
            except Exception as e:
                logger.warning(f"Failed to fetch whisper setting, using default: {e}")
                
            # Use float32 for maximum compatibility and stability on CPU (especially ARM64)
            compute_type = "float32"
            logger.info(f"Loading Faster-Whisper model: {idx} (Download Root: {settings.MODELS_DIR})")
            logger.info(f"Using {compute_type} compute type for optimization.")
            
            import time
            start_load = time.time()
            
            # Use float32 for stability
            self.model = faster_whisper.WhisperModel(
                idx, 
                device="cpu", 
                compute_type=compute_type, 
                download_root=settings.MODELS_DIR
            )
            
            load_duration = time.time() - start_load
            logger.info(f"Model loaded in {load_duration:.2f}s")

    def transcribe(self, audio_path: str, progress_callback=None) -> Dict:
        from app.core.audio import AudioProcessor
        
        self.load_model()
        
        # Get total duration for progress calculation
        audio_duration = AudioProcessor.get_duration(audio_path)
        logger.info(f"Transcribing {audio_path} (Duration: {audio_duration:.2f}s)...")
        
        # Determine if we should use chunked transcription
        # Threshold: 20 minutes (1200 seconds)
        chunk_threshold = 1200.0
        if audio_duration > chunk_threshold:
            logger.info("File exceeds duration threshold. Using chunked transcription.")
            return self._transcribe_chunked(audio_path, audio_duration, progress_callback)
            
        # Prepare clean audio for transcription to avoid crashes with multi-stream files (MJPEG etc)
        # We use a temporary file for the clean audio
        clean_audio_path = audio_path + ".clean.wav"
        AudioProcessor.prepare_for_transcription(audio_path, clean_audio_path)
        
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

    def _transcribe_chunked(self, audio_path: str, total_duration: float, progress_callback=None) -> Dict:
        from app.core.audio import AudioProcessor
        
        # Chunk settings
        chunk_duration = 1200.0 # 20 mins
        overlap = 20.0 # 20s overlap
        
        # Stage 1: Normalize original audio (same as single file logic)
        clean_audio_path = audio_path + ".clean.wav"
        AudioProcessor.prepare_for_transcription(audio_path, clean_audio_path)
        
        chunk_paths = []
        try:
            # Stage 2: Create chunks
            chunk_paths = AudioProcessor.create_audio_chunks(clean_audio_path, chunk_duration, overlap)
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

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model_cascade: List[str]):
        self.client = genai.Client(api_key=api_key)
        self.cascade = model_cascade

    def generate(self, prompt: str) -> str:
        last_error = None
        for model_name in self.cascade:
            try:
                logger.info(f"Gemini: Trying model {model_name}...")
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                logger.warning(f"Gemini model {model_name} failed: {e}")
                last_error = e
        raise Exception(f"All Gemini models failed. Last error: {last_error}")
        
    def list_models(self) -> List[str]:
        models = []
        try:
            # New SDK lists usually return objects with .name (full resource name)
            for m in self.client.models.list():
                # We want mainly text generation models. 
                # The filtering attributes in new SDK might differ.
                # For now getting all and stripping prefix.
                name = m.name
                if name.startswith('models/'):
                    name = name.replace('models/', '')
                models.append(name)
        except Exception as e:
            logger.error(f"Gemini list models failed: {e}")
        return models

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, models: List[str], base_url: str = None):
        import openai
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.models = models
        self.is_openrouter = base_url and "openrouter" in base_url

    def generate(self, prompt: str) -> str:
        last_error = None
        for model in self.models:
            try:
                logger.info(f"OpenAI/Compatible: Using model {model}...")
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                last_error = e
        raise Exception(f"All models failed. Last error: {last_error}")
        
    def list_models(self) -> List[str]:
        try:
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]
            
            if self.is_openrouter:
                 return sorted(model_ids)
            else:
                return sorted([m for m in model_ids if m.startswith(("gpt-", "o1-", "chatgpt-"))])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
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
    # Default Gemini Cascade
    DEFAULT_GEMINI_MODELS = [
        'gemini-2.5-pro',
        'gemini-2.5-flash',
        'gemini-2.5-flash-lite',
        'gemini-2.0-flash',
        'gemini-2.0-flash-lite'
    ]

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

    def create_provider(self, provider_type: str, api_key: str = None, model: str = None, openrouter_key: str = None) -> LLMProvider:
        """Factory to create a provider instance."""
        
        # Resolve keys
        # Resolve keys (DB Overrides Env)
        if not api_key:
            # 1. Try DB first (passed via create_provider or from self.settings lookup if we did that earlier)
            # Actually, create_provider is called with args, but let's check self.settings if api_key is None/params not sufficient
            
            db_key = None
            if provider_type == 'gemini': db_key = self.settings.get('gemini_api_key')
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
                models_list = self._parse_model_setting(self.settings.get('openrouter_model'), ['google/gemini-2.0-flash-001'])
            else: # Gemini
                models_list = self._parse_model_setting(self.settings.get('ai_model_cascade'), self.DEFAULT_GEMINI_MODELS)
                
        if provider_type == 'openai':
            return OpenAIProvider(api_key, models_list)
            
        elif provider_type == 'anthropic':
            return AnthropicProvider(api_key, models_list)
            
        elif provider_type == 'openrouter':
            return OpenAIProvider(api_key, models_list, base_url="https://openrouter.ai/api/v1")
            
        else: # Gemini
            # For Gemini, we might have passed models_list if 'model' arg was used,
            # otherwise we default.
            return GeminiProvider(api_key, models_list)

    def _get_provider(self) -> LLMProvider:
        # Use current settings
        provider_type = self.settings.get('active_ai_provider', 'gemini')
        return self.create_provider(provider_type)

    def detect_ads(self, transcript: Dict, options: Dict = None) -> List[Dict[str, float]]:
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
        prompt = self._build_ad_prompt(options, text_data)

        # Execute
        try:
            provider = self._get_provider()
            response_text = provider.generate(prompt)
            return self._parse_ad_response(response_text)
        except Exception as e:
            logger.error(f"Ad detection failed: {e}")
            raise e

    def generate_summary(self, transcript: Dict, podcast_name: str, episode_title: str, pub_date: str) -> str:
        self.settings = self._load_settings()
        text_data = ""
        for seg in transcript['segments']:
            text_data += f"{seg['text']} "
            
        # Build Prompt (use default if None or empty in database)
        db_template = self.settings.get('summary_prompt_template')
        template = db_template if db_template else """
        You are a smart assistant. Write a short 2-3 sentence summary of this podcast episode.
        The summary must:
        1. NOT mention the podcast name, episode title, or date.
        2. Start immediately with "This episode includes".
        3. Briefly summarize key topics.
        Transcript Context: {transcript_context}
        """

        # Ensure template is a string (defensive)
        if template is None:
            template = "Summarize this: {transcript_context}"

        
        try:
             prompt = template.format(transcript_context=text_data[:100000])
        except KeyError:
             prompt = template # Fallback
        
        try:
            provider = self._get_provider()
            return provider.generate(prompt).strip()
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            return f"Welcome to {podcast_name}. Today's episode is {episode_title}."

    # --- Helpers ---
    def _build_ad_prompt(self, options, transcript_text):
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
        
        try:
            return base.format(targets="\n".join(targets), custom_instr=custom) + "\n\nTranscript:\n" + transcript_text
        except Exception as e:
             logger.warning(f"Prompt formatting failed: {e}")
             return base + "\n\nTranscript:\n" + transcript_text

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
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON array pattern
            import re
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except: pass
            
            logger.error(f"Failed to parse JSON response: {text[:200]}...")
            return []

    # Static method to list Gemini models
    @staticmethod
    def list_gemini_models():
        # Priority: DB > Env
        api_key = settings.GEMINI_API_KEY
        try:
            from app.infra.database import get_db_connection
            with get_db_connection() as conn:
                row = conn.execute("SELECT gemini_api_key FROM app_settings WHERE id = 1").fetchone()
                if row and row['gemini_api_key']:
                    api_key = row['gemini_api_key']
        except: pass

        if not api_key:
            return []
            
        try:
            genai.configure(api_key=api_key)
            models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    models.append(m.name.replace('models/', ''))
            return models
        except Exception as e:
            logger.error(f"Failed to list Gemini models: {e}")
            return []
            
    def has_valid_config(self) -> bool:
        """Check if any API provider is configured via DB or Env."""
        # Check Gemini
        if self.settings.get('gemini_api_key') or settings.GEMINI_API_KEY: return True
        
        # Check others
        s = self.settings
        if s.get('openai_api_key') or settings.OPENAI_API_KEY: return True
        if s.get('anthropic_api_key') or settings.ANTHROPIC_API_KEY: return True
        if s.get('openrouter_api_key') or settings.OPENROUTER_API_KEY: return True
        
        return False

    async def validate_tts(self):
        """
        Check if TTS service is available and model is ready.
        """
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
        Generate TTS audio using local system TTS (offline).
        Uses app/core/tts_worker.py in a separate process to avoid stability issues.
        """
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

