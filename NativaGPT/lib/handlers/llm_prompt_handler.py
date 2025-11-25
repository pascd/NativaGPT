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

# Se necessário, ajuste este caminho de importação:
from NativaGPT.lib.config_manager import ConfigManager

load_dotenv()

# Assume que a chave da API está configurada nas variáveis de ambiente
API_KEY = os.getenv('API_KEY')

try:
    from PIL import Image
except ImportError:
    Image = None


class LLMPromptHandler:
    """
    LLM Prompt Handler v3.2 - MULTI-IMAGE SUPPORT (API-Specific Fix)
    Key optimizations for image handling:
    - Supports sending multiple images under the same 'image' field (API requirement).
    - Lazy file opening.
    - File size checking before loading.
    - Parallel file loading.
    """

    # Class-level compiled regex patterns
    _THINKING_PATTERNS = [
        re.compile(r'<think\s*>.*?</think\s*>', re.DOTALL | re.IGNORECASE),
        re.compile(r'<thinking\s*>.*?</thinking\s*>', re.DOTALL | re.IGNORECASE),
        re.compile(r'<thought\s*>.*?</thought\s*>', re.DOTALL | re.IGNORECASE),
        re.compile(r'<analysis\s*>.*?</analysis\s*>', re.DOTALL | re.IGNORECASE),
        re.compile(r'<reasoning\s*>.*?</reasoning\s*>', re.DOTALL | re.IGNORECASE),
    ]
    _JSON_PATTERN = re.compile(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})')
    _NEWLINE_PATTERN = re.compile(r'\n\s*\n\s*\n')
    _WHITESPACE_PATTERN = re.compile(r'^\s+|\s+$')
    _COMMENT_PATTERN = re.compile(r'//.*|/\*.*?\*/', re.DOTALL)
    _TRAILING_COMMA_PATTERN = re.compile(r',\s*([}\]])')
    _SINGLE_QUOTE_PATTERN = re.compile(r"'([^']*)'")

    # Image size limits
    MAX_IMAGE_SIZE_MB = 10
    WARN_IMAGE_SIZE_KB = 500

    # ... (O __slots__ e __init__ permanecem inalterados)
    __slots__ = (
        'config', 'endpoint', 'model', 'vision_model', 'temperature',
        'max_tokens', 'channel_id', 'thread_id', 'image_dir',
        'generated_images_dir', 'setup_prompt', 'session', 'executor',
        'config_manager', '_enhanced_prompt_cache'
    )

    def __init__(self, config):
        self.config = config
        llm_config = config.get("llm_config", {})
        self.endpoint = llm_config.get("endpoint", "")
        self.model = llm_config.get("model", "deepseek-r1:latest")
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
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.config_manager = ConfigManager(config)
        self._enhanced_prompt_cache = {}
        logger.info(f"LLM Handler was initialized. Endpoint: {self.endpoint}, Model: {self.model}")

    def _handle_ndjson_responses(self, response) -> Dict:
        """Optimized NDJSON streaming parser."""
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
            "response_metadata": response_metadata
        }

    def send_to_llm(self, prompt: str, images: List[str] = None) -> Dict:
        """
        OPTIMIZED: Send to LLM with fast image handling and multi-file support.
        """
        request_start = time.time()
        files_to_close = [] # Lista para manter os handles abertos dos ficheiros

        try:
            # Build prompt (with caching)
            prompt_key = hash(prompt)
            enhanced_prompt = self._enhanced_prompt_cache.get(prompt_key)
            if not enhanced_prompt:
                enhanced_prompt = self._build_enhanced_prompt(prompt)
                if len(self._enhanced_prompt_cache) < 100:
                    self._enhanced_prompt_cache[prompt_key] = enhanced_prompt

            base_data = {
                "channel_id": self.channel_id,
                "thread_id": self.thread_id,
                "user_info": "{}",
                "message": enhanced_prompt,
            }

            headers = {'x-api-key': API_KEY}

            # OPTIMIZED: Fast image preparation
            files = None
            if images:
                image_prep_start = time.time()
                # 🔑 MODIFICAÇÃO: _prepare_images_optimized agora devolve uma lista de tuplos
                files = self._prepare_images_optimized(images)

                # Guarda os handles dos ficheiros para fechar no bloco finally
                # O handle é o segundo elemento do tuplo interno (name, handle, mime)
                files_to_close = [f[1][1] for f in files]

                image_prep_time = (time.time() - image_prep_start) * 1000
                logger.info(f"[IMG] Prepared {len(files)} images in {image_prep_time:.1f}ms")

            try:
                # 🔑 MODIFICAÇÃO: Passamos a lista 'files' diretamente para requests
                if files:
                    response = self.session.post(
                        self.endpoint,
                        headers=headers,
                        data=base_data,
                        files=files, # 'files' é agora uma lista, o formato correto para múltiplas imagens no mesmo campo.
                        stream=True,
                        timeout=30  # Increased timeout for image uploads
                    )
                else:
                    response = self.session.post(
                        self.endpoint,
                        headers=headers,
                        data=base_data,
                        stream=True,
                        timeout=10
                    )

                response.raise_for_status()
                response_data = self._handle_ndjson_responses(response)

            finally:
                # 🔑 CORREÇÃO CRÍTICA: Fechar todos os handles abertos
                for file_handle in files_to_close:
                    try:
                        file_handle.close()
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


    def _log_request_error(self, e: requests.exceptions.RequestException) -> None:
        """Efficient error logging."""
        logger.error(f"LLM API request failed: {e}")

        if not (hasattr(e, 'response') and e.response is not None):
            return

        logger.error(f"Response status: {e.response.status_code}")
        logger.error(f"Response headers: {e.response.headers}")

        try:
            logger.error(f"Response body: {json.dumps(e.response.json(), indent=2)}")
        except (ValueError, AttributeError):
            logger.error(f"Response body (raw): {e.response.text[:500]}")

    def _build_enhanced_prompt(self, user_prompt: str) -> str:
        """Build prompt efficiently."""
        parts = []

        if self.setup_prompt:
            parts.append(self.setup_prompt)
            parts.append("\n\n")

        parts.extend([
            "=" * 50,
            "\nUSER REQUEST\n",
            "=" * 50,
            "\n",
            user_prompt,
            "\n\n",
            "=" * 50,
            "\nRESPONSE FORMAT\n",
            "=" * 50,
            """
Respond with:
1. A clear text description/analysis of what you're doing and why
2. Use the available functions when you need to execute commands

The functions are provided separately and you can call them as needed.
"""
        ])

        return ''.join(parts)


    def _prepare_images_optimized(self, images: List[str]) -> List[Tuple[str, Tuple[str, Any, str]]]:
        """
        CORRIGIDO: Prepara múltiplas imagens para upload, usando a chave 'image'
        para todos os ficheiros, no formato List[Tuple] exigido por requests.
        """
        if not images:
            return []

        # 'files' é uma lista de tuplos: [ ('campo', (nome, handle, mime)), ... ]
        files: List[Tuple[str, Tuple[str, Any, str]]] = []
        max_size_bytes = self.MAX_IMAGE_SIZE_MB * 1024 * 1024
        warn_size_bytes = self.WARN_IMAGE_SIZE_KB * 1024

        def load_image_fast(path):
            if not os.path.exists(path):
                logger.warn(f"[IMG] Image not found: {path}")
                return None

            try:
                # Check file size BEFORE loading
                file_size = os.path.getsize(path)

                if file_size > max_size_bytes:
                    logger.warn(
                        f"[IMG] Skipping large image {os.path.basename(path)}: "
                        f"{file_size / 1024 / 1024:.1f}MB"
                    )
                    return None

                if file_size > warn_size_bytes:
                    logger.info(
                        f"[IMG] Loading large image {os.path.basename(path)}: "
                        f"{file_size / 1024:.1f}KB"
                    )

                # Open file in binary mode
                load_start = time.time()
                file_handle = open(path, 'rb')
                load_time = (time.time() - load_start) * 1000

                if load_time > 50:
                    logger.warn(f"[IMG] Slow file read: {load_time:.1f}ms for {os.path.basename(path)}")

                # Determina o MIME type
                if path.lower().endswith('.png'):
                    mime_type = 'image/png'
                elif path.lower().endswith(('.jpg', '.jpeg')):
                    mime_type = 'image/jpeg'
                else:
                    mime_type = 'application/octet-stream'


                # FORMATO CORRIGIDO: Retorna o tuplo completo necessário para a lista 'files'
                return (
                    'image', # <--- Nome do campo da API (MUST be 'image')
                    (
                        os.path.basename(path), # Nome do ficheiro
                        file_handle,            # Handle do ficheiro (conteúdo binário)
                        mime_type               # Mime type
                    )
                )

            except Exception as e:
                logger.error(f"[IMG] Failed to load {path}: {e}")
                return None

        # Parallel loading with thread pool
        futures = {
            self.executor.submit(load_image_fast, p): p
            for p in images
        }

        # Collect results as they complete
        loaded_count = 0
        for future in as_completed(futures):
            result = future.result()
            if result:
                files.append(result) # Adiciona o tuplo de ficheiro à lista
                loaded_count += 1

        logger.info(f"[IMG] Loaded {loaded_count}/{len(images)} images successfully")
        return files


    # ... (O resto da classe process_llm_response, _clean_response_text, etc., permanece inalterado)

    def process_llm_response(self, response_data: Dict, images: List[str] = None) -> Dict:
        """Optimized response processing."""
        try:
            response_text = response_data.get("response", "") or ""
            cleaned_text = self._clean_response_text(response_text)
            json_strings = self._extract_json_commands(cleaned_text)

            saved_images = []
            if "images" in response_data:
                futures = [
                    self.executor.submit(self.save_base64_image, img_data, "png")
                    for img_data in response_data["images"]
                ]

                for future in as_completed(futures):
                    image_path = future.result()
                    if image_path:
                        saved_images.append(image_path)

            return {
                "text_content": cleaned_text,
                "json_strings": json_strings,
                "saved_images": saved_images,
                "model_used": response_data.get("model", "unknown"),
                "processing_time": response_data.get("total_duration", 0) / 1e9,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error processing LLM response: {e}")
            return {"error": f"Response processing error: {str(e)}", "success": False}

    def _clean_response_text(self, text: str) -> str:
        """Fast text cleaning."""
        if not text:
            return text

        for pattern in self._THINKING_PATTERNS:
            text = pattern.sub('', text)

        text = self._NEWLINE_PATTERN.sub('\n\n', text)
        return text.strip()

    @lru_cache(maxsize=256)
    def _is_command_json_cached(self, json_str: str) -> bool:
        """Cached command JSON validation."""
        try:
            obj = json.loads(json_str)
            if isinstance(obj, dict):
                if 'command' in obj:
                    return True
                func = obj.get('function')
                if isinstance(func, dict) and 'command' in func:
                    return True
            return False
        except:
            return False

    def _extract_json_commands(self, text: str) -> List[str]:
        """Extract JSON commands efficiently."""
        json_commands = []

        try:
            matches = self._JSON_PATTERN.findall(text)

            for js in matches:
                try:
                    json.loads(js)
                    if self._is_command_json_cached(js):
                        json_commands.append(js)
                except json.JSONDecodeError:
                    fixed = self._fix_json_syntax(js)
                    if fixed and self._is_command_json_cached(fixed):
                        json_commands.append(fixed)
        except Exception as e:
            logger.error(f"Error extracting JSON commands: {e}")

        return json_commands

    def _fix_json_syntax(self, json_str: str) -> Optional[str]:
        """Fast JSON syntax fixes."""
        try:
            json_str = self._COMMENT_PATTERN.sub('', json_str)
            json_str = self._TRAILING_COMMA_PATTERN.sub(r'\1', json_str)
            json_str = self._SINGLE_QUOTE_PATTERN.sub(r'"\1"', json_str)

            json.loads(json_str)
            return json_str
        except:
            return None

    def save_base64_image(self, base64_string: str, extension: str = "png") -> Optional[str]:
        """Optimized base64 image saving."""
        try:
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',', 1)[1]

            image_data = base64.b64decode(base64_string)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            image_path = os.path.join(self.generated_images_dir, f"generated_image_{ts}.{extension}")

            with open(image_path, "wb") as f:
                f.write(image_data)

            logger.info(f"Saved image to {image_path}")
            return image_path
        except Exception as e:
            logger.error(f"Error saving base64 image: {e}")
            return None

    def send_output_to_llm(self, commands_output: Dict[str, List[Dict[str, Any]]]) -> Dict:
        """Optimized output formatting and sending."""
        try:
            parts = [
                "### Command Execution Results\n\n",
                "The following commands were executed. Analyze the results and provide guidance.\n"
            ]

            images_to_attach = []
            pointcloud_files = []

            for command, results in commands_output.items():
                parts.append(f"\n**Command:** `{command}`\n")

                for idx, result in enumerate(results, 1):
                    if len(results) > 1:
                        parts.append(f"  Execution #{idx}:\n")

                    self._format_result(result, parts, images_to_attach, pointcloud_files)

            parts.append("\n**Instructions:** Review the results above. ")
            if images_to_attach:
                parts.append(f"Attached {len(images_to_attach)} image(s). ")
            if pointcloud_files:
                parts.append(f"Point clouds: {', '.join(os.path.basename(p) for p in pointcloud_files)}. ")

            parts.extend([
                "\n\nProvide analysis. ",
                "If errors occurred, suggest fixes. ",
                "If data was retrieved, summarize findings. ",
                "Provide next commands if needed (JSON format)."
            ])

            feedback_prompt = ''.join(parts)

            logger.info(f"Sending feedback with {len(images_to_attach)} images")
            return self.send_to_llm(feedback_prompt, images=images_to_attach)

        except Exception as e:
            logger.error(f"Error sending outputs to LLM: {e}")
            return {"error": f"Failed to send outputs: {str(e)}", "success": False}

    def _format_result(self, result: Dict, parts: List[str],
                       images: List[str], pointclouds: List[str]) -> None:
        """Format a single result efficiently."""
        returncode = result.get('returncode', 'N/A')
        running = result.get('running', False)
        output_info = result.get('output_info', {})

        status = f"Running (PID: {result.get('pid')})" if running else "Completed"
        parts.extend([
            f"  - Status: {status}\n",
            f"  - Return Code: {returncode}\n"
        ])

        ros_topic_data = output_info.get('ros_topic_data')
        if ros_topic_data:
            parts.extend([
                "  - **ROS Topic Read**\n",
                f"    - Topic: `{output_info.get('topic_name')}`\n",
                f"    - Type: `{output_info.get('message_type')}`\n",
                f"    - Timestamp: {output_info.get('timestamp')}\n",
            ])

        output_type = output_info.get('type', 'text')
        files = output_info.get('files', [])
        data = output_info.get('data', '')
        has_error = output_info.get('has_error', False)

        if has_error and not ros_topic_data:
            stderr = result.get('stderr', '').strip()
            if stderr:
                parts.append(f"  - **Error:**\n```\n{stderr[:1000]}\n```\n")

        if output_type == 'image':
            self._format_image_output(output_info, ros_topic_data, files, parts, images)
        elif output_type == 'pointcloud':
            self._format_pointcloud_output(output_info, ros_topic_data, files, parts, pointclouds)
        elif output_type in ['json', 'structured']:
            self._format_json_output(data, ros_topic_data, parts)
        elif output_type == 'text':
            self._format_text_output(result, data, ros_topic_data, parts)

        parts.append("\n")

    def _format_image_output(self, output_info, ros_topic_data, files, parts, images):
        """Format image output."""
        if ros_topic_data:
            parts.append("  - **Image from ROS Topic**\n")
        else:
            parts.append(f"  - Images: {len(files)}\n")

        for img_path in files:
            if os.path.exists(img_path):
                images.append(img_path)
                size_kb = os.path.getsize(img_path) / 1024
                parts.append(f"    - {os.path.basename(img_path)} ({size_kb:.1f}KB)\n")
            else:
                parts.append(f"    - {img_path} (not found)\n")

    def _format_pointcloud_output(self, output_info, ros_topic_data, files, parts, pointclouds):
        """Format pointcloud output."""
        for pc_path in files:
            if os.path.exists(pc_path):
                pointclouds.append(pc_path)
                size_mb = os.path.getsize(pc_path) / (1024 * 1024)
                parts.append(f"    - {os.path.basename(pc_path)} ({size_mb:.2f}MB)\n")

    def _format_json_output(self, data, ros_topic_data, parts):
        """Format JSON output."""
        json_str = self._format_json_data(data)
        parts.append(f"```json\n{json_str}\n```\n")

    def _format_text_output(self, result, data, ros_topic_data, parts):
        """Format text output."""
        stdout = result.get('stdout', '').strip()
        if stdout:
            display = stdout[:1500] + "\n... (truncated)" if len(stdout) > 1500 else stdout
            parts.append(f"  - Output:\n```\n{display}\n```\n")

    def _format_json_data(self, data: Any) -> str:
        """Fast JSON formatting."""
        try:
            if isinstance(data, dict):
                json_str = json.dumps(data, indent=2, ensure_ascii=False)
            elif isinstance(data, str) and data.strip().startswith('{'):
                json_str = json.dumps(json.loads(data), indent=2, ensure_ascii=False)
            else:
                json_str = str(data)

            return json_str[:2000] + "\n..." if len(json_str) > 2000 else json_str
        except:
            return str(data)[:1000]

    def cleanup(self):
        """Cleanup images."""
        try:
            def cleanup_dir(directory):
                if not os.path.exists(directory):
                    return

                for f in os.listdir(directory):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            os.remove(os.path.join(directory, f))
                        except:
                            pass

            futures = [
                self.executor.submit(cleanup_dir, self.image_dir),
                self.executor.submit(cleanup_dir, self.generated_images_dir)
            ]

            for future in as_completed(futures):
                future.result()

            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Resource cleanup."""
        try:
            self.cleanup()
            self.executor.shutdown(wait=False)
            self.session.close()
        except:
            pass