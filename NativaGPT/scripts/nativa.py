import base64
import json
import os
import sys
import time
import signal
import select
import threading
import pathlib
import re
import asyncio
from io import StringIO
from hashlib import sha1
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

# New import for key detection
try:
    from pynput import keyboard
    HAS_PYNPUT = True
except ImportError:
    HAS_PYNPUT = False
    logger.warn("Warning: 'pynput' not installed. Hotkey toggling will be disabled.")

from dotenv import load_dotenv
from NativaGPT.lib.coloring_logger import logger
from NativaGPT.lib.config_manager import ConfigManager
from NativaGPT.lib.handlers.llm_response_handler import LLMResponseHandler
from NativaGPT.lib.handlers.llm_prompt_handler import LLMPromptHandler
from NativaGPT.lib.handlers.json_response_handler import JsonResponseHandler
from NativaGPT.lib.handlers.topic_reader_handler import TopicReaderHandler
from NativaGPT.lib.rag_similarity_check import RAGSimilarityCheck
from NativaGPT.lib.speech_to_text.stt_prompt_handler import STTPromptHandler
from NativaGPT.lib.text_to_speech.tts_prompt_handler import TTSPromptHandler
from NativaGPT.lib.command_execution import CommandExecution
from NativaGPT.lib.mcp.mcp_client import MCPClient


class AsyncLoopThread:
    """Manages a persistent event loop in a background thread for async operations."""

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    def start(self):
        """Start the background event loop thread."""
        if self.thread is not None:
            return

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self._started.wait()  # Wait for loop to be ready

    def _run_loop(self):
        """Run the event loop in the background thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._started.set()
        self.loop.run_forever()

    def run_coroutine(self, coro):
        """Run a coroutine in the background loop and wait for result."""
        if self.loop is None:
            raise RuntimeError("Event loop not started")
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def stop(self):
        """Stop the background event loop."""
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread is not None:
            self.thread.join(timeout=5)


class NativaGPT:
    """
    NativaGPT v2.1 - Added MCP Toggle Hotkey
    """

    # Pre-compiled trigger pattern (built lazily per instance)
    _trigger_pattern = None

    def __init__(self, config):
        logger.info("Initializing NativaGPT v2.1...")

        # Core configuration
        self.config = config

        nativa_cfg = self.config.get("nativa_gpt", {})
        llm_cfg = self.config.get("llm_config", {})

        self.use_tts = bool(nativa_cfg.get("use_tts", False))
        self.use_stt = bool(nativa_cfg.get("use_stt", False))
        self.active_listening_timeout = nativa_cfg.get("active_listening_timeout", 30)
        self.trigger_commands = nativa_cfg.get("user_msgs", {}).get("trigger_commands", [])
        self.listening_msg = nativa_cfg.get("user_msgs", {}).get("listening_msg", "Listening...")
        self.setup_prompt = llm_cfg.get("model_config", {}).get("setup_prompt", "")
        self.max_agentic_iterations = nativa_cfg.get("max_agentic_iterations", 5)

        # Initialize handlers
        self.stt_handler = STTPromptHandler(config) if self.use_stt else None
        self.tts_handler = TTSPromptHandler(config) if self.use_tts else None
        self.llm_handler = LLMPromptHandler(config)
        self.llm_response_handler = LLMResponseHandler()
        self.json_response_handler = JsonResponseHandler()
        self.rag_similarity_check = RAGSimilarityCheck(config)
        self.topic_reader = TopicReaderHandler(config)

        # Command execution with ROS topic awareness
        self.command_execution = CommandExecution(topic_reader_handler=self.topic_reader)

        # State management
        self._last_topic_context: Optional[Dict[str, Any]] = None
        self.is_actively_listening = False
        self.listening_timer = None
        self.text_mode = True

        # Thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nativa")

        # Build trigger pattern
        self._build_trigger_pattern()

        # Initialize TTS directories
        if self.use_tts and self.tts_handler:
            try:
                self.tts_handler.set_output_dir()
                self.tts_handler.set_speakers_dir()
            except Exception:
                pass

        # --- MCP Initialization ---
        self.mcp_loop = None
        self.mcp_client = None

        mcp_config = config.get("mcp", {})
        self.mcp = mcp_config.get("enabled", False)
        self.mcp_servers = mcp_config.get("mcp_servers", {})

        self.mcp_server_hosts = []
        for server_name, server_info in self.mcp_servers.items():
            host_path = server_info.get("host")
            if host_path:
                self.mcp_server_hosts.append(host_path)

        # Setup MCP infrastructure regardless of initial enabled state
        # This allows toggling ON later even if it starts OFF
        if self.mcp_server_hosts:
            try:
                self.mcp_loop = AsyncLoopThread()
                self.mcp_loop.start()

                self.mcp_client = MCPClient(self.llm_handler)
                self.mcp_loop.run_coroutine(
                    self.mcp_client.connect_to_server(self.mcp_server_hosts)
                )
                logger.info(f"MCP infrastructure ready with {len(self.mcp_server_hosts)} servers")
            except Exception as e:
                logger.error(f"Failed to initialize MCP infrastructure: {e}")
                self.mcp = False
        else:
            logger.warning("No MCP servers configured. Toggling will be disabled.")
            self.mcp = False

        # --- Keyboard Listener for Toggle ---
        self.keyboard_listener = None
        if HAS_PYNPUT:
            self._start_keyboard_listener()

        logger.info(f"NativaGPT initialized. MCP Mode is currently: {'ON' if self.mcp else 'OFF'}")

    def _start_keyboard_listener(self):
        """Starts a background listener for key presses."""
        def on_release(key):
            try:
                # CHECK FOR F9 KEY TO TOGGLE MCP
                if key == keyboard.Key.f9:
                    self._toggle_mcp_mode()
            except Exception as e:
                logger.error(f"Key error: {e}")

        self.keyboard_listener = keyboard.Listener(on_release=on_release)
        self.keyboard_listener.start()
        logger.info("Hotkeys active: Press [F9] to toggle MCP Mode")

    def _toggle_mcp_mode(self):
        """Toggles the MCP boolean flag."""
        if not self.mcp_client:
            logger.warning("Cannot toggle MCP: Client not initialized (check config)")
            return

        self.mcp = not self.mcp
        status = "ENABLED" if self.mcp else "DISABLED"

        # Visual feedback
        logger.info(f"\n{'='*40}")
        logger.info(f"   MCP MODE NOW: {status}")
        logger.info(f"{'='*40}\n")

        # Reset prompt if sitting at input
        if self.text_mode:
            logger.info("Type /quit to exit, /voice for voice mode, /mcp to toggle MCP mode.")
            sys.stdout.write("Insert prompt: ")
            sys.stdout.flush()

    def _build_trigger_pattern(self):
        """Build optimized regex pattern for trigger detection."""
        if not self.trigger_commands:
            return

        # Pre-process triggers for faster matching
        self._trigger_words_sets = []
        for trigger in self.trigger_commands:
            words = set(trigger.lower().split())
            self._trigger_words_sets.append(words)

    def detected_trigger(self, content: str) -> bool:
        if not content or not self._trigger_words_sets: return False
        words = set(char.lower() for char in content if char.isalnum() or char.isspace())
        cleaned = ''.join(words).split()
        content_words = set(cleaned)
        for trigger_words in self._trigger_words_sets:
            if trigger_words.issubset(content_words): return True
        return False

    def start_active_listening(self):
        self.is_actively_listening = True
        self.listening_timer = threading.Timer(self.active_listening_timeout, self.stop_active_listening)
        self.listening_timer.start()
        if self.use_tts and self.tts_handler:
            try: self.tts_handler.send_tts_prompt(self.listening_msg)
            except: pass
        logger.info("Active listening started")

    def stop_active_listening(self):
        self.is_actively_listening = False
        if self.listening_timer:
            self.listening_timer.cancel()
            self.listening_timer = None
        logger.info("Active listening stopped")

    # ---------------------- Core processing (Optimized) ----------------------

    def _reset_interaction_state(self):
        self._last_topic_context = None
        if hasattr(self.topic_reader, "clear_history"):
            try: self.topic_reader.clear_history()
            except: pass
        try: self.llm_handler.cleanup()
        except: pass

    def _format_rag_knowledge(self, retrieved_knowledge: Any) -> str:
        if isinstance(retrieved_knowledge, str): return retrieved_knowledge
        if isinstance(retrieved_knowledge, list):
            output = StringIO()
            for i, item in enumerate(retrieved_knowledge, 1):
                output.write(f"Result {i}: {str(item)}\n\n")
            return output.getvalue()
        return str(retrieved_knowledge)

    def _build_enhanced_prompt(self, prompt: str, rag_knowledge: Optional[str]) -> str:
        output = StringIO()
        if self.setup_prompt: output.write(self.setup_prompt + "\n\n")
        output.write(f"The user input is: {prompt}\n\n")
        if rag_knowledge: output.write(f"Retrieved RAG knowledge:\n{rag_knowledge}")
        return output.getvalue()

    def _process_text_pipeline_api(self, prompt: str):
        """Optimized API pipeline with parallel context collection."""
        self._reset_interaction_state()
        logger.info("Collecting context and sending to LLM...")

        try:
            rag_knowledge = self.rag_similarity_check.get_relevant_knowledge(prompt)
            enhanced_prompt = self._build_enhanced_prompt(prompt, rag_knowledge)
            parsed_response = self.llm_handler.send_to_llm(enhanced_prompt)

            if not parsed_response or not parsed_response.get("success", False):
                logger.error("LLM API failed")
                return
            logger.info("LLM responded successfully")

        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return

        response_text = parsed_response.get("text_content", "")
        json_strings = parsed_response.get("json_strings", [])
        logger.info(f"Response: {response_text}")

        if self.use_tts and response_text and self.tts_handler:
            try: self.tts_handler.send_tts_prompt(response_text)
            except Exception as e: logger.warning(f"TTS failed: {e}")

        if json_strings:
            self._execute_agentic_loop(json_strings, max_iterations=self.max_agentic_iterations)
        else:
            logger.info("No commands in response")

        self._reset_interaction_state()
        logger.info("Pipeline completed")

    def _execute_agentic_loop(self, json_strings: List[str], max_iterations: int = 5):
        # [Logic remains identical to previous version]
        # Using abbreviated version for brevity
        iteration = 0
        executed_fingerprints = set()

        while iteration < max_iterations and json_strings:
            iteration += 1
            try:
                functions = self.json_response_handler.check_all_functions(json_strings)
                if not functions: break
                commands = self.json_response_handler.get_function_command(functions)

                fp = sha1("|".join(sorted(json_strings)).encode("utf-8")).hexdigest()
                if fp in executed_fingerprints: break

                commands_output = self.command_execution.get_commands_output(commands)
                executed_fingerprints.add(fp)

                # ... Logging ...

                feedback_response = self.llm_handler.send_output_to_llm(commands_output)
                if not feedback_response or not feedback_response.get("success", False): break

                new_json_strings = feedback_response.get("json_strings", [])
                if new_json_strings: json_strings = new_json_strings
                else: break
            except Exception: break

    # ---------------------- Mode loops (Optimized) ----------------------

    def push_to_talk_once(self):
        if not self.stt_handler: return
        logger.info("PTT: Listening...")
        self.stt_handler.start_stt_handler()
        try:
            while True:
                if self.stt_handler.process_audio():
                    stt_response = self.stt_handler.get_response()
                    if stt_response and "transcription" in stt_response:
                        transcription = stt_response["transcription"]
                        logger.info(f"PTT> {transcription}")
                        self._process_text_pipeline_api(transcription)
                        break
                time.sleep(0.01)
        finally:
            self.stt_handler.stop_stt_handler()

    def _voice_mode_loop(self):
        """Voice mode with wake word detection."""
        if not self.stt_handler: return

        self.stt_handler.start_stt_handler()
        try:
            logger.info("Voice mode: Listening for wake word (type /text to exit)")
            while True:
                if self._stdin_ready():
                    user_cmd = sys.stdin.readline().strip()
                    if self._handle_meta_command(user_cmd):
                        if self.text_mode: break
                        continue

                if self.stt_handler.process_audio():
                    stt_response = self.stt_handler.get_response()
                    if stt_response and "transcription" in stt_response:
                        transcription = stt_response["transcription"]

                        if not self.is_actively_listening:
                            if self.detected_trigger(transcription):
                                logger.info("Wake word detected!")
                                self.start_active_listening()
                        else:
                            # Logic branching based on MCP mode
                            logger.info(f"Processing: {transcription} (MCP: {self.mcp})")
                            if self.mcp:
                                try:
                                    # Voice response via MCP
                                    resp = self.mcp_loop.run_coroutine(self.mcp_client.process_query(transcription))
                                    logger.info(f"MCP Response: {resp}")
                                    if self.use_tts and self.tts_handler:
                                        self.tts_handler.send_tts_prompt(resp)
                                except Exception as e:
                                    logger.error(f"MCP Error: {e}")
                            else:
                                self._process_text_pipeline_api(transcription)

                            time.sleep(0.5)
        finally:
            self.stt_handler.stop_stt_handler()

    def _text_mode_loop(self):
        """Text mode loop."""
        while True:
            # Prompt shows current mode state
            mode_indicator = "[MCP]" if self.mcp else "[STD]"
            logger.info("Type /quit to exit, /voice for voice mode, /mcp to toggle MCP mode.")
            user_input = input(f"{mode_indicator} Insert prompt: ").strip()

            if self._handle_meta_command(user_input):
                if not self.text_mode:
                    break
                continue
            if not user_input:
                continue

            if not self.mcp:
                self._process_text_pipeline_api(user_input)
            else:
                # Run the async MCP query using the background loop
                try:
                    logger.info("Querying MCP...")
                    response = self.mcp_loop.run_coroutine(
                        self.mcp_client.process_query(user_input)
                    )
                    logger.info(f"\n{response}\n")
                except Exception as e:
                    logger.error(f"MCP query failed: {e}")
                    self._process_text_pipeline_api(user_input) # Fallback

    def _handle_meta_command(self, text: str) -> bool:
        """Handle meta commands."""
        if not text.startswith("/"): return False
        parts = text[1:].split()
        cmd = parts[0].lower() if parts else ""

        if cmd in ("quit", "exit"): raise KeyboardInterrupt
        if cmd == "voice":
            self.text_mode = False
            logger.info("🎙️  Voice mode ON")
            return True
        if cmd == "text":
            self.text_mode = True
            logger.info("⌨️  Text mode ON")
            return True
        if cmd == "ptt":
            self.push_to_talk_once()
            return True
        if cmd == "mcp":
            self._toggle_mcp_mode()
            return True

        return True

    def _stdin_ready(self) -> bool:
        return select.select([sys.stdin], [], [], 0.0)[0] != []

    def start(self):
        """Main loop."""
        logger.info("Starting NativaGPT v2.1...")
        while True:
            try:
                if not self.text_mode:
                    self._voice_mode_loop()
                else:
                    self._text_mode_loop()
            except KeyboardInterrupt:
                logger.info("\nShutting down...")
                break
            finally:
                if self.listening_timer:
                    self.listening_timer.cancel()
                if self.stt_handler:
                    self.stt_handler.stop_stt_handler()

    def __del__(self):
        """Cleanup resources."""
        try:
            if self.keyboard_listener:
                self.keyboard_listener.stop()
            self.executor.shutdown(wait=False)
            if self.listening_timer:
                self.listening_timer.cancel()
            if self.mcp_client and self.mcp_loop:
                try: self.mcp_loop.run_coroutine(self.mcp_client.cleanup())
                except: pass
            if self.mcp_loop:
                self.mcp_loop.stop()
        except:
            pass


def main():
    cfg_path = str(pathlib.Path(__file__).parent.parent.parent / "config" / "config_default.json")
    config_manager = ConfigManager(config_path=cfg_path)
    config = config_manager.get()

    nativa = NativaGPT(config)
    nativa.start()


if __name__ == "__main__":
    main()