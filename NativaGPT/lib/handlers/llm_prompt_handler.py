import base64
import json
import os
import time
import requests
import re
from io import StringIO, BytesIO
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any, Tuple
from NativaGPT.lib.coloring_logger import logger
from datetime import datetime
import uuid

from NativaGPT.lib.config_manager import ConfigManager

load_dotenv()

API_KEY = os.getenv("API_KEY")

try:
    from PIL import Image
except ImportError:
    Image = None


class LLMPromptHandler:
    """
    LLM Prompt Handler v3.4 - SYSTEM PROMPT OVERRIDE SUPPORT
    """

    _THINKING_PATTERNS = [
        re.compile(r"<think\s*>.*?</think\s*>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<thinking\s*>.*?</thinking\s*>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<thought\s*>.*?</thought\s*>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<analysis\s*>.*?</analysis\s*>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<reasoning\s*>.*?</reasoning\s*>", re.DOTALL | re.IGNORECASE),
    ]
    _JSON_PATTERN = re.compile(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})")
    _NEWLINE_PATTERN = re.compile(r"\n\s*\n\s*\n")
    _WHITESPACE_PATTERN = re.compile(r"^\s+|\s+$")
    _COMMENT_PATTERN = re.compile(r"//.*|/\*.*?\*/", re.DOTALL)
    _TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")
    _SINGLE_QUOTE_PATTERN = re.compile(r"'([^']*)'")

    MAX_IMAGE_SIZE_MB = 10
    WARN_IMAGE_SIZE_KB = 500

    __slots__ = (
        "config",
        "endpoint",
        "model",
        "vision_model",
        "temperature",
        "max_tokens",
        "channel_id",
        "thread_id",
        "image_dir",
        "generated_images_dir",
        "setup_prompt",
        "session",
        "executor",
        "config_manager",
        "_enhanced_prompt_cache",
    )

    def __init__(self, config):
        self.config = config
        llm_config = config.get("llm_config", {})

        self.endpoint = llm_config.get("endpoint", "")
        self.vision_model = llm_config.get("vision_model", "llava:latest")
        self.temperature = llm_config.get("temperature", 0.1)
        self.max_tokens = llm_config.get("max_tokens", 2000)
        self.channel_id = llm_config.get("channel_id", "")

        self.thread_id = llm_config.get("thread_id", "")
        if not self.thread_id:
            self.thread_id = str(uuid.uuid4())

        self.image_dir = "/tmp/nativa_vlm_images"
        self.generated_images_dir = "/tmp/nativa_generated_images"
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.generated_images_dir, exist_ok=True)

        self.setup_prompt = llm_config.get("model_config", {}).get("setup_prompt", "")

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=20, max_retries=3, pool_block=False
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.executor = ThreadPoolExecutor(max_workers=4)
        self.config_manager = ConfigManager(config)
        self._enhanced_prompt_cache = {}

        logger.info(f"LLM Handler was initialized. Endpoint: {self.endpoint}")

    def _handle_ndjson_responses(self, response) -> Dict:
        """Parses NDJSON stream."""
        content_parts = []
        final_message = None
        run_id = None
        model_name = "unknown"
        response_metadata = {}

        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    event_type = event.get("type")
                    if event_type == "start":
                        run_id = event.get("run_id")
                    elif event_type == "token":
                        content_parts.append(event.get("content", ""))
                    elif event_type == "message":
                        message_data = event.get("content", {})
                        final_message = message_data.get("content", "")
                        response_metadata = message_data.get("response_metadata", {})
                        model_name = response_metadata.get("model_name", "unknown")
                    elif event_type == "done":
                        logger.info(f"Stream completed: {run_id}")
                        break
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.error(f"Error in NDJSON streaming: {e}")

        response_text = final_message or "".join(content_parts)

        if not response_text:
            return {"error": "No content received", "success": False}

        return {
            "response": response_text,
            "run_id": run_id,
            "model": model_name,
            "total_duration": 0,
            "response_metadata": response_metadata,
            "success": True,
        }

    # 🔑 ALTERAÇÃO PRINCIPAL: Adicionado argumento 'system_instruction'
    def send_to_llm(
        self, prompt: str, images: List[str] = None, system_instruction: str = None
    ) -> Dict:
        """
        Sends request to LLM. Supports overriding the system prompt.
        """
        request_start = time.time()
        files_to_close = []

        try:
            # Disable cache when system_instruction is provided (for VLM mode)
            # Check for None explicitly, not just falsy (empty string should also bypass cache)
            if system_instruction is not None:
                enhanced_prompt = self._build_enhanced_prompt(
                    prompt, system_instruction
                )
            else:
                prompt_key = hash(prompt + (system_instruction or ""))
                enhanced_prompt = self._enhanced_prompt_cache.get(prompt_key)

                if not enhanced_prompt:
                    enhanced_prompt = self._build_enhanced_prompt(
                        prompt, system_instruction
                    )
                    if len(self._enhanced_prompt_cache) < 100:
                        self._enhanced_prompt_cache[prompt_key] = enhanced_prompt

            base_data = {
                "channel_id": self.channel_id,
                "thread_id": self.thread_id,
                "user_info": "{}",
                "message": enhanced_prompt,
            }

            headers = {"x-api-key": API_KEY}
            files = None

            if images:
                image_prep_start = time.time()
                files = self._prepare_images_optimized(images)
                files_to_close = [f[1][1] for f in files]
                image_prep_time = (time.time() - image_prep_start) * 1000
                logger.info(
                    f"[IMG] Prepared {len(files)} images in {image_prep_time:.1f}ms"
                )

            try:
                # Use longer timeout for images
                timeout_val = 60 if files else 10

                if files:
                    response = self.session.post(
                        self.endpoint,
                        headers=headers,
                        data=base_data,
                        files=files,
                        stream=True,
                        timeout=timeout_val,
                    )
                else:
                    response = self.session.post(
                        self.endpoint,
                        headers=headers,
                        data=base_data,
                        stream=True,
                        timeout=timeout_val,
                    )

                response.raise_for_status()
                response_data = self._handle_ndjson_responses(response)

            finally:
                for fh in files_to_close:
                    try:
                        fh.close()
                    except:
                        pass

            if not response_data.get("success", True):
                return response_data

            request_time = (time.time() - request_start) * 1000
            logger.info(f"[LLM] Total request time: {request_time:.1f}ms")

            return self.process_llm_response(response_data, images)

        except requests.exceptions.RequestException as e:
            self._log_request_error(e)
            return {"error": f"LLM API request failed: {str(e)}", "success": False}

    # 🔑 ALTERAÇÃO PRINCIPAL: Lógica de override do prompt
    def _build_enhanced_prompt(
        self, user_prompt: str, override_system: str = None
    ) -> str:
        """Build prompt efficiently with optional override."""
        parts = []

        # Se houver override (mesmo que string vazia), usa-o. Senão, usa o self.setup_prompt (padrão)
        if override_system is not None:
            system_prompt_to_use = override_system
        else:
            system_prompt_to_use = self.setup_prompt

        # Só adiciona system prompt se não for vazio
        if system_prompt_to_use:
            parts.append(system_prompt_to_use + "\n\n")

        # Se houver override (modo VLM), não adicionamos os cabeçalhos padrão de User Request
        if override_system is not None:
            # Em modo VLM, apenas adicionamos o prompt do utilizador, sem headers
            parts.append(f"{user_prompt}\n")
        else:
            # Modo normal
            parts.append(
                "=" * 50 + "\nUSER REQUEST\n" + "=" * 50 + "\n\n" + user_prompt + "\n\n"
            )
            parts.append(
                "=" * 50
                + "\nRESPONSE FORMAT\n"
                + "=" * 50
                + """
Respond with:
1. A clear text description/analysis of what you're doing and why
2. Use the available functions when you need to execute commands
"""
            )

        return "".join(parts)

    def _prepare_images_optimized(
        self, images: List[str]
    ) -> List[Tuple[str, Tuple[str, Any, str]]]:
        """Prepara múltiplas imagens (Lista de Tuplos)."""
        if not images:
            return []
        files = []
        max_size = self.MAX_IMAGE_SIZE_MB * 1024 * 1024

        def load_image_fast(path):
            if not os.path.exists(path):
                return None
            try:
                if os.path.getsize(path) > max_size:
                    return None
                fh = open(path, "rb")
                mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
                return ("image", (os.path.basename(path), fh, mime))
            except:
                return None

        futures = {self.executor.submit(load_image_fast, p): p for p in images}
        for future in as_completed(futures):
            res = future.result()
            if res:
                files.append(res)

        logger.info(f"[IMG] Loaded {len(files)}/{len(images)} images successfully")
        return files

    # ... (MÉTODOS AUXILIARES: _log_request_error, process_llm_response, etc. - MANTÊM-SE IGUAIS) ...
    def _log_request_error(self, e):
        logger.error(f"Req failed: {e}")

    def process_llm_response(self, response_data, images=None):
        try:
            txt = response_data.get("response", "")
            clean = self._clean_response_text(txt)
            jsons = self._extract_json_commands(clean)
            return {"text_content": clean, "json_strings": jsons, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _clean_response_text(self, text):
        if not text:
            return text
        for p in self._THINKING_PATTERNS:
            text = p.sub("", text)
        return self._NEWLINE_PATTERN.sub("\n\n", text).strip()

    @lru_cache(maxsize=256)
    def _is_command_json_cached(self, s):
        try:
            return isinstance(json.loads(s), dict) and "command" in json.loads(s)
        except:
            return False

    def _extract_json_commands(self, text):
        cmds = []
        for m in self._JSON_PATTERN.findall(text):
            if self._is_command_json_cached(m):
                cmds.append(m)
        return cmds

    def save_base64_image(self, b64, ext="png"):
        return None

    def send_output_to_llm(self, out):
        return self.send_to_llm(json.dumps(out))

    def cleanup(self):
        pass

    def __del__(self):
        try:
            self.session.close()
            self.executor.shutdown(wait=False)
        except:
            pass
