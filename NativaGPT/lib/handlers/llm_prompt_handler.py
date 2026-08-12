"""Client for talking to an LLM over a generic OpenAI-compatible Chat Completions API.

This module intentionally speaks the widely-implemented
``POST {base_url}/chat/completions`` wire format instead of any
proprietary/vendor-specific protocol. That format is implemented natively by
OpenAI, and is also exposed (or emulated) by most self-hosted inference
servers, including the ones this project already ships launch scripts for
under ``NativaGPT/bash/`` (Ollama's ``/v1`` endpoint, LM Studio, KoboldCpp).
Pointing ``llm_config.base_url`` at any of those - or at OpenAI itself, with
an API key - is enough to make ``LLMPromptHandler`` work, with no code
changes.
"""

import base64
import json
import mimetypes
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from NativaGPT.lib.coloring_logger import logger

# Load variables from a local .env file (e.g. LLM_API_KEY) into the process
# environment, if one is present. See .env.example for the expected keys.
load_dotenv()

try:
    from PIL import Image
except ImportError:  # Pillow is an optional dependency of this module.
    Image = None


class LLMPromptHandler:
    """Generic client for OpenAI-compatible Chat Completions APIs.

    Builds an OpenAI-style ``messages`` array from a user prompt (plus an
    optional system prompt override and optional images), sends it to
    ``{base_url}/chat/completions``, and returns the parsed assistant reply
    along with any JSON "command" blocks found inside it.

    Both streaming (Server-Sent Events) and non-streaming responses are
    supported, selected via the ``stream`` config flag. Multimodal prompts
    are supported by attaching images as base64 ``image_url`` content
    blocks, per the OpenAI vision message format.

    Attributes:
        config: The full application config dict passed at construction.
        base_url: Root URL of the OpenAI-compatible API (no trailing slash).
        model: Model name used for text-only requests.
        vision_model: Model name used when images are attached to the prompt.
        temperature: Sampling temperature sent with every request.
        max_tokens: Maximum tokens requested per completion.
        stream: Whether to use SSE streaming (True) or a single blocking
            request (False).
        timeout: Request timeout, in seconds.
        api_key: API key read from the environment variable named by
            ``llm_config.api_key_env`` (empty string if unset, e.g. for
            servers like local Ollama that don't require authentication).
    """

    # Tags some models wrap their internal "thinking" in; stripped from the
    # final response before it's shown to the user or scanned for commands.
    _THINKING_PATTERNS = [
        re.compile(r"<think\s*>.*?</think\s*>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<thinking\s*>.*?</thinking\s*>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<thought\s*>.*?</thought\s*>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<analysis\s*>.*?</analysis\s*>", re.DOTALL | re.IGNORECASE),
        re.compile(r"<reasoning\s*>.*?</reasoning\s*>", re.DOTALL | re.IGNORECASE),
    ]
    # Matches a single (possibly one-level-nested) JSON object embedded in text.
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
        "base_url",
        "model",
        "vision_model",
        "temperature",
        "max_tokens",
        "stream",
        "timeout",
        "api_key",
        "image_dir",
        "generated_images_dir",
        "setup_prompt",
        "session",
        "executor",
        "_enhanced_prompt_cache",
    )

    def __init__(self, config: Dict[str, Any]):
        """Initialize the handler from the ``llm_config`` section of the app config.

        Args:
            config: Full application config dict (as loaded by
                ``ConfigManager``). Only the ``llm_config`` sub-dict is read
                here; every field has a safe default so the handler works
                out of the box against a local Ollama server.
        """
        self.config = config
        llm_config = config.get("llm_config", {})

        # Root of the OpenAI-compatible API, e.g. "https://api.openai.com/v1"
        # or "http://localhost:11434/v1" for a local Ollama server. Defaults
        # to local Ollama so the project runs with zero config and no API
        # key out of the box (see NativaGPT/bash/launch_llm_ollama.sh).
        self.base_url = llm_config.get("base_url", "http://localhost:11434/v1").rstrip("/")
        self.model = llm_config.get("model", "deepseek-r1:latest")
        self.vision_model = llm_config.get("vision_model", "llava:latest")
        self.temperature = llm_config.get("temperature", 0.1)
        self.max_tokens = llm_config.get("max_tokens", 2000)
        self.stream = llm_config.get("stream", True)
        self.timeout = llm_config.get("timeout", 60)

        # The API key itself is never stored in config/version control - only
        # the *name* of the environment variable that holds it is. Defaults
        # to LLM_API_KEY; see .env.example.
        api_key_env = llm_config.get("api_key_env", "LLM_API_KEY")
        self.api_key = os.getenv(api_key_env, "")

        self.image_dir = "/tmp/nativa_vlm_images"
        self.generated_images_dir = "/tmp/nativa_generated_images"
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.generated_images_dir, exist_ok=True)

        self.setup_prompt = llm_config.get("model_config", {}).get("setup_prompt", "")

        # A pooled/retrying session avoids reconnecting on every request.
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=20, max_retries=3, pool_block=False
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Used to encode multiple images in parallel (see _prepare_image_content_blocks).
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._enhanced_prompt_cache = {}

        logger.info(
            f"LLM Handler initialized. base_url={self.base_url} "
            f"model={self.model} stream={self.stream}"
        )

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_enhanced_prompt(
        self, user_prompt: str, override_system: Optional[str] = None
    ) -> Tuple[str, str]:
        """Build the (system, user) message text pair for a chat request.

        Args:
            user_prompt: The raw user-facing prompt text.
            override_system: If not None, replaces the configured
                ``setup_prompt`` as the system message (used by callers such
                as VLM/image-description flows that need a different system
                instruction for a single call). An empty string is a valid
                override meaning "no system message".

        Returns:
            A ``(system_content, user_content)`` tuple. ``system_content`` is
            ``""`` when there is no system message to send.
        """
        if override_system is not None:
            system_prompt_to_use = override_system
        else:
            system_prompt_to_use = self.setup_prompt

        if override_system is not None:
            # Override mode (e.g. VLM): send the user prompt as-is, without
            # the standard "USER REQUEST" / "RESPONSE FORMAT" scaffolding
            # below, since the caller is asking a narrower question.
            user_content = f"{user_prompt}\n"
        else:
            user_content = (
                "=" * 50
                + "\nUSER REQUEST\n"
                + "=" * 50
                + "\n\n"
                + user_prompt
                + "\n\n"
                + "=" * 50
                + "\nRESPONSE FORMAT\n"
                + "=" * 50
                + """
Respond with:
1. A clear text description/analysis of what you're doing and why
2. Use the available functions when you need to execute commands
"""
            )

        return system_prompt_to_use, user_content

    def _prepare_image_content_blocks(self, images: List[str]) -> List[Dict[str, Any]]:
        """Load and base64-encode local images into OpenAI ``image_url`` content blocks.

        Images are loaded in parallel via ``self.executor``. Any image that
        is missing or larger than ``MAX_IMAGE_SIZE_MB`` is silently skipped
        (and logged) rather than failing the whole request.

        Args:
            images: List of local image file paths.

        Returns:
            A list of ``{"type": "image_url", "image_url": {"url": "data:<mime>;base64,<data>"}}``
            dicts. Order matches completion order of the parallel loads, not
            necessarily the input order.
        """
        if not images:
            return []
        max_size = self.MAX_IMAGE_SIZE_MB * 1024 * 1024

        def encode_one(path: str) -> Optional[Dict[str, Any]]:
            if not os.path.exists(path) or os.path.getsize(path) > max_size:
                return None
            try:
                mime, _ = mimetypes.guess_type(path)
                mime = mime or "image/jpeg"
                with open(path, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode("ascii")
                return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            except Exception:
                return None

        blocks: List[Dict[str, Any]] = []
        futures = {self.executor.submit(encode_one, p): p for p in images}
        for future in as_completed(futures):
            result = future.result()
            if result:
                blocks.append(result)

        logger.info(f"[IMG] Encoded {len(blocks)}/{len(images)} images successfully")
        return blocks

    def _build_messages(
        self,
        prompt: str,
        images: Optional[List[str]],
        system_instruction: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Assemble the OpenAI ``messages`` array and pick which model to use.

        Args:
            prompt: The user-facing prompt text.
            images: Optional list of local image file paths for a multimodal
                request. When provided, the user message content becomes a
                list of content blocks (text + image_url) instead of a
                plain string, and ``self.vision_model`` is selected.
            system_instruction: Optional override for the system prompt, see
                ``_build_enhanced_prompt``.

        Returns:
            Tuple of ``(messages, model)`` ready to drop into a Chat
            Completions request payload.
        """
        system_content, user_content = self._build_enhanced_prompt(prompt, system_instruction)

        messages: List[Dict[str, Any]] = []
        if system_content:
            messages.append({"role": "system", "content": system_content})

        if images:
            content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": user_content}]
            content_blocks.extend(self._prepare_image_content_blocks(images))
            messages.append({"role": "user", "content": content_blocks})
            model = self.vision_model
        else:
            messages.append({"role": "user", "content": user_content})
            model = self.model

        return messages, model

    def _headers(self) -> Dict[str, str]:
        """Build the request headers, including auth only if an API key is set."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _send_streaming_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST the request with ``stream=True`` and consume the SSE response.

        Parses ``data: {...}`` lines as they arrive and accumulates
        ``choices[0].delta.content`` from each chunk, per the standard
        OpenAI streaming Chat Completions format. Stops at the terminating
        ``data: [DONE]`` line.

        Args:
            payload: The Chat Completions request body (already has
                ``"stream": True``).

        Returns:
            Dict with ``response`` (the concatenated text), ``model``, and
            ``success``, or ``{"error": ..., "success": False}`` if no
            content was received.
        """
        url = f"{self.base_url}/chat/completions"
        content_parts: List[str] = []
        model_name = payload.get("model", "unknown")

        response = self.session.post(
            url, headers=self._headers(), json=payload, stream=True, timeout=self.timeout
        )
        response.raise_for_status()

        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                model_name = event.get("model", model_name)
                choices = event.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    content_parts.append(delta.get("content", "") or "")
        except Exception as e:
            logger.error(f"Error while streaming SSE response: {e}")

        response_text = "".join(content_parts)
        if not response_text:
            return {"error": "No content received", "success": False}
        return {"response": response_text, "model": model_name, "success": True}

    def _send_nonstreaming_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST the request with ``stream=False`` and read the single JSON response.

        Args:
            payload: The Chat Completions request body (already has
                ``"stream": False``).

        Returns:
            Dict with ``response`` (the message content), ``model``, and
            ``success``, or ``{"error": ..., "success": False}`` if the
            response doesn't match the expected shape.
        """
        url = f"{self.base_url}/chat/completions"
        response = self.session.post(
            url, headers=self._headers(), json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            return {"error": f"Unexpected response shape: {e}", "success": False}
        return {"response": content, "model": data.get("model", "unknown"), "success": True}

    def send_to_llm(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request to the configured OpenAI-compatible endpoint.

        This is the main entry point used by the rest of the package
        (``NativaMCPWrapper``, the main ``nativa.py`` loop, ``MCPClient``).
        Its signature and return shape are considered a stable internal
        contract - callers only need ``text_content``/``json_strings``/
        ``success`` from the returned dict.

        Args:
            prompt: The user-facing prompt text.
            images: Optional list of local image file paths for multimodal
                input. When provided, ``self.vision_model`` is used instead
                of ``self.model``.
            system_instruction: Optional override for the configured setup
                prompt (see ``_build_enhanced_prompt``).

        Returns:
            On success, a dict with ``text_content`` (cleaned response
            text), ``json_strings`` (any embedded JSON command blocks found
            in it), and ``success: True``. On failure, a dict with
            ``error`` and ``success: False``.
        """
        request_start = time.time()
        try:
            # Cache the built prompt/messages for repeated identical calls,
            # but never cache when a per-call system override is supplied
            # (e.g. VLM mode), since the override changes the result.
            cache_key = None
            if system_instruction is None:
                cache_key = hash(prompt)
                cached = self._enhanced_prompt_cache.get(cache_key)
                if cached is not None:
                    messages, model = cached
                else:
                    messages, model = self._build_messages(prompt, images, system_instruction)
                    if len(self._enhanced_prompt_cache) < 100:
                        self._enhanced_prompt_cache[cache_key] = (messages, model)
            else:
                messages, model = self._build_messages(prompt, images, system_instruction)

            payload = {
                "model": model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": self.stream,
            }

            response_data = (
                self._send_streaming_request(payload)
                if self.stream
                else self._send_nonstreaming_request(payload)
            )

            if not response_data.get("success", True):
                return response_data

            request_time = (time.time() - request_start) * 1000
            logger.info(f"[LLM] Total request time: {request_time:.1f}ms")

            return self.process_llm_response(response_data, images)

        except requests.exceptions.RequestException as e:
            self._log_request_error(e)
            return {"error": f"LLM API request failed: {str(e)}", "success": False}

    # ------------------------------------------------------------------
    # Response post-processing
    # ------------------------------------------------------------------

    def _log_request_error(self, e: Exception) -> None:
        """Log a failed HTTP request against the LLM endpoint."""
        logger.error(f"Req failed: {e}")

    def process_llm_response(
        self, response_data: Dict[str, Any], images: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Clean the raw response text and extract any embedded JSON commands.

        Args:
            response_data: The dict returned by ``_send_streaming_request``
                or ``_send_nonstreaming_request`` (must contain ``response``).
            images: Unused here; kept for interface symmetry with callers
                that may want to know which images were part of the request.

        Returns:
            Dict with ``text_content``, ``json_strings``, and ``success``,
            or ``{"error": ..., "success": False}`` on failure.
        """
        try:
            txt = response_data.get("response", "")
            clean = self._clean_response_text(txt)
            jsons = self._extract_json_commands(clean)
            return {"text_content": clean, "json_strings": jsons, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _clean_response_text(self, text: str) -> str:
        """Strip model "thinking" tags and collapse excess blank lines."""
        if not text:
            return text
        for pattern in self._THINKING_PATTERNS:
            text = pattern.sub("", text)
        return self._NEWLINE_PATTERN.sub("\n\n", text).strip()

    @lru_cache(maxsize=256)
    def _is_command_json_cached(self, s: str) -> bool:
        """Return True if ``s`` parses as a JSON object containing a "command" key."""
        try:
            return isinstance(json.loads(s), dict) and "command" in json.loads(s)
        except Exception:
            return False

    def _extract_json_commands(self, text: str) -> List[str]:
        """Find all JSON-object-looking substrings in ``text`` that look like commands."""
        cmds = []
        for m in self._JSON_PATTERN.findall(text):
            if self._is_command_json_cached(m):
                cmds.append(m)
        return cmds

    def save_base64_image(self, b64: str, ext: str = "png") -> Optional[str]:
        """Reserved for future image-generation support; currently a no-op stub."""
        return None

    def send_output_to_llm(self, out: Any) -> Dict[str, Any]:
        """Send the JSON-serialized result of a previously executed command back to the LLM.

        Args:
            out: Any JSON-serializable object (typically a command execution
                result dict) to hand back to the model for it to narrate.

        Returns:
            Same shape as ``send_to_llm``.
        """
        return self.send_to_llm(json.dumps(out))

    def cleanup(self) -> None:
        """Placeholder for future explicit resource cleanup (currently a no-op)."""
        pass

    def __del__(self):
        """Best-effort cleanup of the HTTP session and thread pool on GC."""
        try:
            self.session.close()
            self.executor.shutdown(wait=False)
        except Exception:
            pass
