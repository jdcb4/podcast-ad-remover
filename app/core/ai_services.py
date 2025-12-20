import whisper
from google import genai
import json
import logging
import asyncio
import os
import sys
from typing import List, Dict
from app.core.config import settings

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
                
            logger.info(f"Loading Whisper model: {idx}")
            self.model = whisper.load_model(idx, download_root=settings.MODELS_DIR)

    def transcribe(self, audio_path: str) -> Dict:
        self.load_model()
        logger.info(f"Transcribing {audio_path}...")
        # verbose=True prints to stdout, which Docker captures
        result = self.model.transcribe(audio_path, verbose=True)
        return result

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
        text_data = ""
        for seg in transcript['segments']:
            text_data += f"{seg['text']} "
            
        # Build Prompt (Simplified)
        template = self.settings.get('summary_prompt_template', """
        You are a smart assistant. Write a short 2-3 sentence summary of this podcast episode.
        The summary must:
        1. NOT mention the podcast name, episode title, or date.
        2. Start immediately with "This episode includes".
        3. Briefly summarize key topics.
        Transcript Context: {transcript_context}
        """)
        
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
        if not settings.GEMINI_API_KEY:
            return []
            
        try:
            # Check if we are using the new SDK or old one based on imports
            # The file imports 'from google import genai' which is the new SDK style
            # but creates Client. It doesn't typically have global configure.
            # Assuming mixed usage or valid previously. We just guard the key here.
            genai.configure(api_key=settings.GEMINI_API_KEY)
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

    async def generate_audio(self, text: str, output_path: str):
        """
        Generate TTS audio using local system TTS (offline).
        Uses app/core/tts_worker.py in a separate process to avoid stability issues.
        """
        try:
            logger.info("Generating TTS (Piper in subprocess)...")
            
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
            
            # Resolve absolute path to the worker script
            script_path = os.path.join(os.path.dirname(__file__), "tts_worker.py")
            
            # Run the worker script
            proc = await asyncio.create_subprocess_exec(
                sys.executable, script_path, output_path, "--model", piper_model_file,
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

