"""
LLM Prompt Handler v4.2 - Unified Single VLM Model

Features:
- Single VLM model handles text and images
- Auto-downloads model if not available (Ollama)
- Works with modern VLMs: llama3.2, moondream, minivlm, gpt-4o, etc.
- Unified interface for Ollama local models and external API

Usage Examples:
===============

# Single VLM model config:
# {
#     "llm_config": {
#         "backend": "ollama",
#         "ollama_endpoint": "http://localhost:11434",
#         "model": "llama3.2",
#         "temperature": 0.1,
#         "max_tokens": 2000
#     }
# }

# API backend:
# {
#     "llm_config": {
#         "backend": "api",
#         "endpoint": "https://api.openai.com/v1/chat/completions",
#         "api_key": "your-key",
#         "model": "gpt-4o"
#     }
# }

# Code usage:
handler = LLMPromptHandler(config)

# Text only
result = handler.send_to_llm("Hello!")

# With images - same method
result = handler.send_to_llm("Describe this", images=["/path/image.jpg"])
"""

import base64
import json
import os
import time
import requests
import re
from requests.adapters import HTTPAdapter
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from typing import Dict, List, Optional, Any, Tuple
from NativaGPT.lib.coloring_logger import logger

load_dotenv()

API_KEY = os.getenv("API_KEY", "")


class LLMPromptHandler:
    """
    LLM Prompt Handler v4.0 - Unified Ollama/API Backend

    Supports:
    - Ollama local models (text and vision)
    - External LLM APIs with image support
    - Automatic image encoding and optimization
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

    MAX_IMAGE_SIZE_MB = 10
    MAX_IMAGE_DIMENSION = 1024
    JPEG_QUALITY = 75

    __slots__ = (
        "config",
        "backend",
        "endpoint",
        "api_key",
        "model",
        "temperature",
        "max_tokens",
        "system_prompt",
        "session",
        "executor",
        "image_dir",
    )

    def __init__(self, config):
        self.config = config
        llm_config = config.get("llm_config", {})

        self.backend = llm_config.get("backend", "ollama")

        if self.backend == "ollama":
            self.endpoint = llm_config.get("ollama_endpoint", "http://localhost:11434")
        else:
            self.endpoint = llm_config.get("endpoint", "")

        self.api_key = llm_config.get("api_key", API_KEY)
        self.model = llm_config.get("model", "llama3.2")
        self.temperature = llm_config.get("temperature", 0.1)
        self.max_tokens = llm_config.get("max_tokens", 2000)
        self.system_prompt = llm_config.get("model_config", {}).get("setup_prompt", "")

        self.image_dir = "/tmp/nativa_vlm_images"
        os.makedirs(self.image_dir, exist_ok=True)

        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=10, pool_maxsize=20, max_retries=3, pool_block=False
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info(f"LLM Handler v4.2 initialized")
        logger.info(f"Backend: {self.backend}")
        logger.info(f"Endpoint: {self.endpoint}")
        logger.info(f"Model: {self.model} (handles text + images)")

        if self.backend == "ollama":
            self._ensure_ollama_model()

    def send_to_llm(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Send request to LLM backend.

        Args:
            prompt: The user prompt
            images: List of image paths or base64 strings
            system_prompt: Override system prompt (None = use default)

        Returns:
            Dict with 'text_content', 'json_strings', 'success', and optionally 'raw_response'
        """
        if self.backend == "ollama":
            return self._send_to_ollama(prompt, images, system_prompt)
        else:
            return self._send_to_api(prompt, images, system_prompt)

    def send_to_vlm(
        self,
        image_path: str,
        prompt: str = "Describe this image in detail.",
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Send image to Vision Language Model for analysis.

        Args:
            image_path: Path to image file (from ROS capture, camera, or local file)
            prompt: Question/instruction about the image
            system_prompt: Optional system prompt override

        Returns:
            Dict with 'text_content', 'success'

        Example:
            result = handler.send_to_vlm("/tmp/captured_image.jpg", "What's in this scene?")
        """
        logger.info(f"[VLM] Analyzing image: {image_path}")

        if self.backend == "ollama":
            return self._send_to_ollama_vlm(image_path, prompt, system_prompt)
        else:
            return self._send_to_api_vlm(image_path, prompt, system_prompt)

    def send_images_to_llm(
        self,
        images: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """
        Simplified multi-image sending to LLM/VLM.

        Args:
            images: List of image paths (from ROS, camera, or files)
            prompt: Question/instruction about the images
            system_prompt: Optional system prompt override

        Returns:
            Dict with 'text_content', 'json_strings', 'success'

        Example:
            result = handler.send_images_to_llm(
                ["/path/img1.jpg", "/path/img2.jpg"],
                "Compare these images and describe differences"
            )
        """
        return self.send_to_llm(prompt, images=images, system_prompt=system_prompt)

    def _send_to_ollama(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """Send request to Ollama API - same model handles text + images."""
        request_start = time.time()

        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            }

            if system_prompt is not None:
                payload["system"] = system_prompt
            elif self.system_prompt:
                payload["system"] = self.system_prompt

            if images:
                base64_images = self._prepare_images_base64(images)
                if base64_images:
                    payload["images"] = base64_images
                    logger.info(
                        f"[Ollama] Sending {len(base64_images)} images with model {self.model}"
                    )

            response = self.session.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=180,
            )

            response.raise_for_status()
            response_data = response.json()

            result = self._handle_ollama_response(response_data)
            request_time = (time.time() - request_start) * 1000
            logger.info(f"[Ollama] Request completed in {request_time:.1f}ms")

            return self.process_llm_response(result, images)

        except requests.exceptions.RequestException as e:
            logger.error(f"[Ollama] Request failed: {e}")
            return {"error": f"Ollama request failed: {str(e)}", "success": False}
        except Exception as e:
            logger.error(f"[Ollama] Unexpected error: {e}")
            return {"error": f"Unexpected error: {str(e)}", "success": False}

    def _ensure_ollama_model(self):
        """Check if model exists in Ollama, pull if missing."""
        try:
            response = self.session.get(f"{self.endpoint}/api/tags", timeout=10)
            response.raise_for_status()
            tags = response.json()
            model_names = [m["name"] for m in tags.get("models", [])]

            model_short = self.model.split(":")[0] if ":" in self.model else self.model
            if any(model_short in name for name in model_names):
                logger.info(f"[Ollama] Model '{self.model}' is available")
                return True

            logger.info(f"[Ollama] Pulling model '{self.model}'...")
            pull_response = self.session.post(
                f"{self.endpoint}/api/pull",
                json={"name": self.model},
                stream=True,
                timeout=600,
            )
            for line in pull_response.iter_lines():
                if line:
                    logger.info(f"[Ollama] {line.decode('utf-8')}")
            logger.info(f"[Ollama] Model '{self.model}' pulled successfully")
            return True
        except Exception as e:
            logger.error(f"[Ollama] Model check/pull failed: {e}")
            return False

    def _send_to_ollama_vlm(
        self,
        image_path: str,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """Send image to Ollama - uses same model for text + images."""
        return self._send_to_ollama(
            prompt, images=[image_path], system_prompt=system_prompt
        )

    def _send_to_api(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """Send request to external API."""
        request_start = time.time()
        files_to_close = []

        try:
            enhanced_prompt = self._build_prompt(prompt, system_prompt)

            base_data = {
                "message": enhanced_prompt,
            }

            if self.api_key:
                headers = {
                    "x-api-key": self.api_key,
                    "Authorization": f"Bearer {self.api_key}",
                }
            else:
                headers = {}

            files = None
            if images:
                files = self._prepare_images_multipart(images)
                files_to_close = [f[1][1] for f in files] if files else []
                logger.info(f"[API] Prepared {len(files) if files else 0} images")

            timeout = 120 if images else 60

            try:
                if files:
                    response = self.session.post(
                        self.endpoint,
                        headers=headers,
                        data=base_data,
                        files=files,
                        stream=True,
                        timeout=timeout,
                    )
                else:
                    response = self.session.post(
                        self.endpoint,
                        headers=headers,
                        json=base_data,
                        stream=True,
                        timeout=timeout,
                    )

                response.raise_for_status()
                response_data = self._handle_api_response(response)

            finally:
                for fh in files_to_close:
                    try:
                        fh.close()
                    except:
                        pass

            if not response_data.get("success", True):
                return response_data

            request_time = (time.time() - request_start) * 1000
            logger.info(f"[API] Request completed in {request_time:.1f}ms")

            return self.process_llm_response(response_data, images)

        except requests.exceptions.RequestException as e:
            logger.error(f"[API] Request failed: {e}")
            return {"error": f"API request failed: {str(e)}", "success": False}
        except Exception as e:
            logger.error(f"[API] Unexpected error: {e}")
            return {"error": str(e), "success": False}

    def _send_to_api_vlm(
        self,
        image_path: str,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        """Send image to API - uses same model for text + images."""
        return self._send_to_api(
            prompt, images=[image_path], system_prompt=system_prompt
        )

    def _handle_ollama_response(self, response: dict) -> dict:
        """Handle Ollama API response."""
        try:
            response_text = response.get("response", "")
            return {
                "response": response_text,
                "model": response.get("model", self.model),
                "total_duration": response.get("total_duration", 0),
                "eval_count": response.get("eval_count", 0),
                "prompt_eval_count": response.get("prompt_eval_count", 0),
                "success": True,
                "raw_response": response,
            }
        except Exception as e:
            logger.error(f"Error handling Ollama response: {e}")
            return {"error": str(e), "success": False}

    def _handle_api_response(self, response) -> Dict:
        """Parse NDJSON streaming response from API."""
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
                        break
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.error(f"Error in API streaming: {e}")

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

    def _build_prompt(
        self, user_prompt: str, override_system: Optional[str] = None
    ) -> str:
        """Build the complete prompt."""
        parts = []

        system_to_use = (
            override_system if override_system is not None else self.system_prompt
        )

        if system_to_use:
            parts.append(system_to_use + "\n\n")

        if override_system is not None:
            parts.append(f"{user_prompt}\n")
        else:
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

    def _load_image_as_base64(self, image_path: str) -> Optional[str]:
        """Load image file and return base64 encoded string."""
        try:
            if not os.path.exists(image_path):
                logger.error(f"[IMG] Image not found: {image_path}")
                return None

            max_size = self.MAX_IMAGE_SIZE_MB * 1024 * 1024
            if os.path.getsize(image_path) > max_size:
                logger.warning(f"[IMG] Image too large: {image_path}")
                return None

            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        except Exception as e:
            logger.error(f"[IMG] Error loading image: {e}")
            return None

    def _prepare_images_base64(self, images: List[str]) -> List[str]:
        """Prepare images as base64 strings for Ollama."""
        base64_images = []

        def load_image(path_or_b64):
            if os.path.exists(path_or_b64):
                return self._load_image_as_base64(path_or_b64)
            elif len(path_or_b64) > 100:
                return path_or_b64
            return None

        futures = {self.executor.submit(load_image, img): img for img in images}
        for future in as_completed(futures):
            result = future.result()
            if result:
                base64_images.append(result)

        logger.info(
            f"[IMG] Prepared {len(base64_images)}/{len(images)} images for Ollama"
        )
        return base64_images

    def _prepare_images_multipart(
        self, images: List[str]
    ) -> List[Tuple[str, Tuple[str, Any, str]]]:
        """Prepare images for multipart form upload."""
        if not images:
            return []

        files = []
        max_size = self.MAX_IMAGE_SIZE_MB * 1024 * 1024

        def load_image(path):
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

        futures = {self.executor.submit(load_image, p): p for p in images}
        for future in as_completed(futures):
            res = future.result()
            if res:
                files.append(res)

        logger.info(f"[IMG] Prepared {len(files)}/{len(images)} images for upload")
        return files

    def process_llm_response(self, response_data: Dict, images=None) -> Dict:
        """Process LLM response and extract text and JSON commands."""
        try:
            txt = response_data.get("response", "")
            clean = self._clean_response_text(txt)
            jsons = self._extract_json_commands(clean)
            return {
                "text_content": clean,
                "json_strings": jsons,
                "success": True,
                "raw_response": response_data.get("raw_response"),
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def _clean_response_text(self, text: str) -> str:
        """Clean response text by removing thinking tags."""
        if not text:
            return text
        for pattern in self._THINKING_PATTERNS:
            text = pattern.sub("", text)
        return self._NEWLINE_PATTERN.sub("\n\n", text).strip()

    @lru_cache(maxsize=256)
    def _is_command_json_cached(self, s: str) -> bool:
        """Check if string is a command JSON."""
        try:
            parsed = json.loads(s)
            return isinstance(parsed, dict) and "command" in parsed
        except:
            return False

    def _extract_json_commands(self, text: str) -> List[str]:
        """Extract JSON command strings from text."""
        commands = []
        for match in self._JSON_PATTERN.findall(text):
            if self._is_command_json_cached(match):
                commands.append(match)
        return commands

    def send_output_to_llm(self, output: Dict) -> Dict:
        """Send command output to LLM for processing."""
        return self.send_to_llm(json.dumps(output))

    def cleanup(self):
        """Cleanup resources."""
        try:
            self.session.close()
            self.executor.shutdown(wait=False)
        except:
            pass

    def __del__(self):
        self.cleanup()
