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
    NativaGPT v2.0 - Performance Optimized
    - Pre-compiled regex patterns
    - Cached config values
    - StringIO for efficient string building
    - Parallel RAG and topic collection
    - Optimized agentic loop with fingerprint tracking
    - Reduced memory allocations
    - Thread pooling for I/O operations
    """

    # Pre-compiled trigger pattern (built lazily per instance)
    _trigger_pattern = None

    def __init__(self, config):
        logger.info("Initializing NativaGPT v2.0...")

        # Core configuration
        self.config = config

        # Cache frequently accessed config values
        self._cache_config_values()

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
        self._last_commands_fingerprint = None

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

        # Initialize MCP client if enabled - FIXED CONFIG READING
        self.mcp_loop = None
        self.mcp_client = None

        mcp_config = config.get("mcp", {})
        self.mcp = mcp_config.get("enabled", False)
        self.mcp_servers = mcp_config.get("mcp_servers", {})

        self.mcp_server_hosts = []
        for server_name, server_info in self.mcp_servers.items():
            host_path = server_info.get("host")
            if host_path:
                logger.info(f"Found MCP server '{server_name}': {host_path}")
                self.mcp_server_hosts.append(host_path)

        if self.mcp:
            if not self.mcp_server_hosts:
                logger.warning("MCP enabled but no server hosts found in config")
            else:
                logger.info(f"Initializing MCP with {len(self.mcp_server_hosts)} server(s)")
                try:
                    # Start background event loop for MCP
                    self.mcp_loop = AsyncLoopThread()
                    self.mcp_loop.start()

                    # Create and connect MCP client
                    self.mcp_client = MCPClient()
                    self.mcp_loop.run_coroutine(
                        self.mcp_client.connect_to_server(self.mcp_server_hosts)
                    )
                    logger.info("MCP client connected successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize MCP: {e}")
                    import traceback
                    traceback.print_exc()
                    self.mcp = False
                    if self.mcp_loop:
                        self.mcp_loop.stop()
                        self.mcp_loop = None
        else:
            logger.info("MCP is disabled in config")

        logger.info("NativaGPT v2.0 initialized successfully")

    def _cache_config_values(self):
        """Cache frequently accessed configuration values."""
        nativa_cfg = self.config.get("nativa_gpt", {})
        llm_cfg = self.config.get("llm_config", {})

        self.use_tts = bool(nativa_cfg.get("use_tts", False))
        self.use_stt = bool(nativa_cfg.get("use_stt", False))
        self.active_listening_timeout = nativa_cfg.get("active_listening_timeout", 30)
        self.trigger_commands = nativa_cfg.get("user_msgs", {}).get("trigger_commands", [])
        self.listening_msg = nativa_cfg.get("user_msgs", {}).get("listening_msg", "Listening...")
        self.setup_prompt = llm_cfg.get("model_config", {}).get("setup_prompt", "")
        self.max_agentic_iterations = nativa_cfg.get("max_agentic_iterations", 5)

    def _build_trigger_pattern(self):
        """Build optimized regex pattern for trigger detection."""
        if not self.trigger_commands:
            return

        # Pre-process triggers for faster matching
        self._trigger_words_sets = []
        for trigger in self.trigger_commands:
            words = set(trigger.lower().split())
            self._trigger_words_sets.append(words)

    # ---------------------- Topic helpers (Optimized) ----------------------

    def _collect_topic_context(self) -> Dict[str, Any]:
        """Collect topic context with error handling."""
        try:
            payload = self.topic_reader.process_all_topics()
            self._last_topic_context = payload
            return payload
        except Exception as e:
            logger.warning(f"Topic collection failed: {e}")
            return {"generated_at": None, "items": [], "tmp_dir": None}

    def _format_topic_context_for_llm(self, payload: Dict[str, Any], max_items: int = 12) -> Tuple[str, List[str]]:
        """Format topic context using StringIO for efficiency."""
        items = payload.get("items", [])[:max_items]
        output = StringIO()
        image_paths = []

        hdr = payload.get("generated_at", "unknown time")
        output.write(f"### Live robot/topic context (generated {hdr})\n")
        output.write("Use this fresh context for reasoning. Images will be attached when available.\n\n")

        if not items:
            output.write("- (no recent topic items)\n")
            return output.getvalue(), image_paths

        for it in items:
            src = it.get("source")
            name = it.get("name")
            modality = it.get("modality")
            ts = it.get("timestamp")
            hints = it.get("analysis_hints", {})

            if modality == "image":
                p = it.get("data")
                if p and os.path.exists(p):
                    image_paths.append(p)
                    output.write(f"- [{ts}] ({src}) {name} → IMAGE attached: {os.path.basename(p)}\n")
                else:
                    output.write(f"- [{ts}] ({src}) {name} → IMAGE (missing)\n")

            elif modality == "text":
                txt = str(it.get("data", "")).strip()
                txt = txt[:400] + "…" if len(txt) > 400 else txt
                output.write(f"- [{ts}] ({src}) {name} → TEXT: {txt}\n")

            elif modality == "structured":
                try:
                    compact = json.dumps(it.get("data", {}), separators=(",", ":"), ensure_ascii=False)
                except Exception:
                    compact = str(it.get("data"))
                compact = compact[:400] + "…" if len(compact) > 400 else compact
                output.write(f"- [{ts}] ({src}) {name} → STRUCT: {compact}\n")

            else:
                output.write(f"- [{ts}] ({src}) {name} → {modality}\n")

            if hints:
                relevant_hints = {k: v for k, v in hints.items()
                                if v and k in ("analysis_type", "detection_objects", "vlm_analysis", "vla_analysis")}
                if relevant_hints:
                    try:
                        htxt = json.dumps(relevant_hints, ensure_ascii=False)
                        output.write(f"  ↳ hints: {htxt}\n")
                    except:
                        pass

        return output.getvalue(), image_paths

    # ---------------------- Wake word logic (Optimized) ----------------------

    def detected_trigger(self, content: str) -> bool:
        """Optimized trigger detection using set intersection."""
        if not content or not self._trigger_words_sets:
            return False

        # Clean and tokenize once
        words = set(char.lower() for char in content if char.isalnum() or char.isspace())
        cleaned = ''.join(words).split()
        content_words = set(cleaned)

        # Check if any trigger matches (all words present)
        for trigger_words in self._trigger_words_sets:
            if trigger_words.issubset(content_words):
                return True

        return False

    def start_active_listening(self):
        """Start listening mode with timeout."""
        self.is_actively_listening = True
        self.listening_timer = threading.Timer(self.active_listening_timeout, self.stop_active_listening)
        self.listening_timer.start()

        if self.use_tts and self.tts_handler:
            try:
                self.tts_handler.send_tts_prompt(self.listening_msg)
            except Exception:
                pass

        logger.info("Active listening started")

    def stop_active_listening(self):
        """Stop listening mode."""
        self.is_actively_listening = False
        if self.listening_timer:
            self.listening_timer.cancel()
            self.listening_timer = None
        logger.info("Active listening stopped")

    # ---------------------- Core processing (Optimized) ----------------------

    def _reset_interaction_state(self):
        """Clear state for independent interactions."""
        self._last_topic_context = None

        # Clear topic history
        if hasattr(self.topic_reader, "clear_history"):
            try:
                self.topic_reader.clear_history()
            except Exception:
                pass

        # Cleanup LLM handler
        try:
            self.llm_handler.cleanup()
        except Exception:
            pass

    def _format_rag_knowledge(self, retrieved_knowledge: Any) -> str:
        """Efficiently format RAG knowledge."""
        if isinstance(retrieved_knowledge, str):
            return retrieved_knowledge

        if isinstance(retrieved_knowledge, list):
            output = StringIO()
            for i, item in enumerate(retrieved_knowledge, 1):
                if isinstance(item, tuple) and len(item) >= 3:
                    entry = item[0]
                    if isinstance(entry, dict):
                        output.write(f"Result {i}: {json.dumps(entry, indent=2)}\n\n")
                    else:
                        output.write(f"Result {i}: {str(entry)}\n\n")
                else:
                    output.write(f"Result {i}: {str(item)}\n\n")
            return output.getvalue()

        return str(retrieved_knowledge)

    def _build_enhanced_prompt(self, prompt: str, topic_context: str, rag_knowledge: Optional[str]) -> str:
        """Build prompt efficiently with StringIO."""
        output = StringIO()

        if self.setup_prompt:
            output.write(self.setup_prompt)
            output.write("\n\n")

        output.write("The user input is: ")
        output.write(prompt)
        output.write("\n\n")

        if topic_context:
            output.write(topic_context)
            output.write("\n\n")

        if rag_knowledge:
            output.write("Retrieved RAG knowledge:\n")
            output.write(rag_knowledge)

        return output.getvalue()

    def _collect_context_parallel(self, prompt: str) -> Tuple[Dict[str, Any], str, List[str], Optional[str]]:
        """Collect topic context and RAG knowledge in parallel."""
        topic_future = self.executor.submit(self._collect_topic_context)
        rag_future = self.executor.submit(self.rag_similarity_check.retrieve, prompt)

        # Wait for both
        topic_payload = topic_future.result()
        retrieved_knowledge = rag_future.result()

        # Format results
        topic_context_text, topic_images = self._format_topic_context_for_llm(topic_payload)
        rag_knowledge_text = self._format_rag_knowledge(retrieved_knowledge) if retrieved_knowledge else None

        return topic_payload, topic_context_text, topic_images, rag_knowledge_text

    def _process_text_pipeline_api(self, prompt: str):
        """Optimized API pipeline with parallel context collection."""
        self._reset_interaction_state()

        logger.info("Collecting context and sending to LLM...")

        try:
            # Parallel context collection
            topic_payload, topic_context, topic_images, rag_knowledge = self._collect_context_parallel(prompt)

            # Build enhanced prompt
            enhanced_prompt = self._build_enhanced_prompt(prompt, topic_context, rag_knowledge)

            # Send to LLM
            parsed_response = self.llm_handler.send_to_llm(enhanced_prompt, images=topic_images)

            if not parsed_response or not parsed_response.get("success", False):
                logger.error("LLM API failed")
                return

            logger.info("LLM responded successfully")

        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Stage 2: Process response
        response_text = parsed_response.get("text_content", "")
        json_strings = parsed_response.get("json_strings", [])

        logger.info(f"Response: {response_text}")
        logger.info(f"Commands: {len(json_strings)}")

        # TTS output
        if self.use_tts and response_text and self.tts_handler:
            try:
                self.tts_handler.send_tts_prompt(response_text)
            except Exception as e:
                logger.warning(f"TTS failed: {e}")

        # Execute agentic loop
        if json_strings:
            self._execute_agentic_loop(json_strings, max_iterations=self.max_agentic_iterations)
        else:
            logger.info("No commands in response")

        self._reset_interaction_state()
        logger.info("Pipeline completed")

    def _execute_agentic_loop(self, json_strings: List[str], max_iterations: int = 5):
        """
        Optimized agentic loop with better duplicate detection and fingerprint tracking.
        """
        iteration = 0
        executed_fingerprints = set()  # Track all executed command sets

        while iteration < max_iterations and json_strings:
            iteration += 1
            logger.info(f"=== Iteration {iteration}/{max_iterations} ===")

            try:
                # Extract functions
                functions = self.json_response_handler.check_all_functions(json_strings)

                if not functions:
                    logger.info("No executable functions - stopping")
                    break

                commands = self.json_response_handler.get_function_command(functions)
                logger.info(f"Executing {len(commands)} command(s)")

                # Create fingerprint
                fp = sha1("|".join(sorted(json_strings)).encode("utf-8")).hexdigest()

                # Check if already executed
                if fp in executed_fingerprints:
                    logger.warning(f"Duplicate commands detected - stopping")
                    logger.debug(f"Fingerprint: {fp[:16]}...")
                    break

                # Execute commands
                commands_output = self.command_execution.get_commands_output(
                    commands,
                    wait_for_errors_seconds=3.0,
                    timeout=30.0,
                    detach_on_no_error=False
                )

                # Mark as executed
                executed_fingerprints.add(fp)
                self._last_commands_fingerprint = fp

                # Log results
                for cmd, results in commands_output.items():
                    for result in results:
                        output_info = result.get('output_info', {})
                        rc = result.get('returncode', 'N/A')
                        output_type = output_info.get('type', 'text')
                        files = len(output_info.get('files', []))
                        logger.info(f"  {cmd[:60]}... → rc={rc}, type={output_type}, files={files}")

                # Send to LLM for analysis
                logger.info(f"Sending outputs to LLM (iteration {iteration})...")
                feedback_response = self.llm_handler.send_output_to_llm(commands_output)

                if not feedback_response or not feedback_response.get("success", False):
                    logger.warning("Failed to get feedback - stopping")
                    break

                # Get analysis
                feedback_text = feedback_response.get("text_content", "")
                logger.info(f"Analysis: {feedback_text}")

                # TTS feedback
                if self.use_tts and feedback_text and self.tts_handler:
                    try:
                        self.tts_handler.send_tts_prompt(feedback_text)
                    except Exception as e:
                        logger.warning(f"TTS failed: {e}")

                # Check for new commands
                new_json_strings = feedback_response.get("json_strings", [])

                if new_json_strings:
                    # Check if these are truly new
                    new_fp = sha1("|".join(sorted(new_json_strings)).encode("utf-8")).hexdigest()

                    if new_fp in executed_fingerprints:
                        logger.info("LLM returned already-executed commands - stopping")
                        logger.debug(f"Repeated fingerprint: {new_fp[:16]}...")
                        break
                    else:
                        logger.info(f"LLM provided {len(new_json_strings)} new command(s) - continuing")
                        json_strings = new_json_strings
                else:
                    logger.info("No more commands - completing loop")
                    break

            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                import traceback
                traceback.print_exc()
                break

        if iteration >= max_iterations:
            logger.warning(f"Max iterations ({max_iterations}) reached")

        logger.info(f"Loop completed: {iteration} iteration(s), {len(executed_fingerprints)} unique command sets")

    # ---------------------- Mode loops (Optimized) ----------------------

    def push_to_talk_once(self):
        """Single push-to-talk interaction."""
        if not self.stt_handler:
            logger.error("STT not available")
            return

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
            logger.info("PTT: Done")

    def _voice_mode_loop(self):
        """Voice mode with wake word detection."""
        if not self.stt_handler:
            logger.error("STT not available")
            return

        self.stt_handler.start_stt_handler()
        try:
            logger.info("Voice mode: Listening for wake word (type /text to exit)")
            while True:
                if self._stdin_ready():
                    user_cmd = sys.stdin.readline().strip()
                    if self._handle_meta_command(user_cmd):
                        if self.text_mode:
                            break
                        continue

                if self.stt_handler.process_audio():
                    stt_response = self.stt_handler.get_response()
                    if stt_response and "transcription" in stt_response:
                        transcription = stt_response["transcription"]
                        logger.info(f"Heard: {transcription}")

                        if not self.is_actively_listening:
                            if self.detected_trigger(transcription):
                                logger.info("Wake word detected!")
                                self.start_active_listening()
                        else:
                            if not self.detected_trigger(transcription):

                                self._process_text_pipeline_api(transcription)

                                time.sleep(0.5)
        finally:
            self.stt_handler.stop_stt_handler()

    def _text_mode_loop(self):
        """Text mode loop."""
        while True:
            user_input = input("Insert prompt: ").strip()
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
                    response = self.mcp_loop.run_coroutine(
                        self.mcp_client.process_query(user_input)
                    )
                    print(f"\n{response}")
                except Exception as e:
                    logger.error(f"MCP query failed: {e}")
                    import traceback
                    traceback.print_exc()

    def _handle_meta_command(self, text: str) -> bool:
        """Handle meta commands."""
        if not text.startswith("/"):
            return False

        parts = text[1:].split()
        cmd = parts[0].lower() if parts else ""
        arg = parts[1].lower() if len(parts) > 1 else ""

        if cmd in ("quit", "exit"):
            raise KeyboardInterrupt

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

        if cmd == "backend":
            self._switch_backend(arg)
            return True

        return True

    def _stdin_ready(self) -> bool:
        """Check if stdin has data."""
        return select.select([sys.stdin], [], [], 0.0)[0] != []

    def start(self):
        """Main loop."""
        logger.info("Starting NativaGPT v2.0...")
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
                    self.listening_timer = None
                if self.stt_handler:
                    self.stt_handler.stop_stt_handler()

    def __del__(self):
        """Cleanup resources."""
        try:
            self.executor.shutdown(wait=False)
            if self.listening_timer:
                self.listening_timer.cancel()

            # Cleanup MCP
            if self.mcp_client and self.mcp_loop:
                try:
                    self.mcp_loop.run_coroutine(self.mcp_client.cleanup())
                except:
                    pass

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