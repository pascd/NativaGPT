#!/usr/bin/env python3
"""
NativaGPT v3.0 - Main Entry Point

Features:
- Unified Ollama/API backend support
- MCP tool integration
- Voice mode (STT/TTS) support
- Text and voice interaction modes
"""

import os
import sys
import time
import signal
import select
import threading
import pathlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional
from io import StringIO

from dotenv import load_dotenv

from NativaGPT.lib.coloring_logger import logger
from NativaGPT.lib.config_manager import ConfigManager
from NativaGPT.lib.handlers.llm_prompt_handler import LLMPromptHandler
from NativaGPT.lib.handlers.llm_response_handler import LLMResponseHandler
from NativaGPT.lib.handlers.json_response_handler import JsonResponseHandler
from NativaGPT.lib.rag_similarity_check import RAGSimilarityCheck
from NativaGPT.lib.command_execution import CommandExecution
from NativaGPT.lib.mcp.mcp_client import MCPClient

load_dotenv()


class AsyncLoopThread:
    """Background event loop for async operations."""

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    def start(self):
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self._started.wait()

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._started.set()
        self.loop.run_forever()

    def run_coroutine(self, coro):
        if self.loop is None:
            raise RuntimeError("Event loop not started")
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def stop(self):
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread is not None:
            self.thread.join(timeout=5)


class NativaGPT:
    """
    NativaGPT v3.0 - AI Assistant with Tool Integration

    Supports:
    - Ollama local models (text and vision)
    - External LLM APIs
    - MCP tool servers
    - Voice input/output (optional)
    """

    def __init__(self, config: Dict[str, Any]):
        logger.info("Initializing NativaGPT v3.0...")

        self.config = config
        nativa_cfg = config.get("nativa_gpt", {})
        llm_cfg = config.get("llm_config", {})

        self.active_listening_timeout = nativa_cfg.get("active_listening_timeout", 30)
        self.trigger_commands = nativa_cfg.get("user_msgs", {}).get(
            "trigger_commands", []
        )
        self.listening_msg = nativa_cfg.get("user_msgs", {}).get(
            "listening_msg", "Listening..."
        )
        self.setup_prompt = llm_cfg.get("model_config", {}).get("setup_prompt", "")
        self.max_agentic_iterations = nativa_cfg.get("max_agentic_iterations", 5)

        voice_cfg = config.get("voice", {})
        self.use_tts = voice_cfg.get("enabled", False) and voice_cfg.get("tts", {}).get(
            "api_url"
        )
        self.use_stt = voice_cfg.get("enabled", False) and voice_cfg.get("stt", {}).get(
            "api_url"
        )

        self.llm_handler = LLMPromptHandler(config)
        self.llm_response_handler = LLMResponseHandler()
        self.json_response_handler = JsonResponseHandler()
        self.rag_similarity_check = RAGSimilarityCheck(config)

        try:
            from NativaGPT.lib.handlers.topic_reader_handler import TopicReaderHandler

            self.topic_reader = TopicReaderHandler(config)
            self.command_execution = CommandExecution(
                topic_reader_handler=self.topic_reader
            )
        except ImportError:
            self.topic_reader = None
            self.command_execution = CommandExecution()

        self._last_topic_context: Optional[Dict[str, Any]] = None
        self.is_actively_listening = False
        self.listening_timer = None
        self.text_mode = True

        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nativa")
        self._build_trigger_pattern()

        self.stt_handler = None
        self.tts_handler = None

        self.mcp_loop = None
        self.mcp_client = None

        mcp_config = config.get("mcp", {})
        self.mcp_enabled = mcp_config.get("enabled", False)
        mcp_servers = mcp_config.get("mcp_servers", {})

        self.mcp_server_hosts = []
        for server_name, server_info in mcp_servers.items():
            host_path = server_info.get("host")
            if host_path:
                host_path = os.path.expanduser(host_path)
                self.mcp_server_hosts.append(host_path)

        if self.mcp_server_hosts:
            try:
                self.mcp_loop = AsyncLoopThread()
                self.mcp_loop.start()

                self.mcp_client = MCPClient(self.llm_handler)
                self.mcp_loop.run_coroutine(
                    self.mcp_client.connect_to_server(self.mcp_server_hosts)
                )
                logger.info(f"✓ MCP ready with {len(self.mcp_server_hosts)} server(s)")
            except Exception as e:
                logger.error(f"MCP init failed: {e}")
                self.mcp_enabled = False
        else:
            self.mcp_enabled = False

        logger.info(f"NativaGPT v3.0 ready!")
        logger.info(
            f"Backend: {llm_cfg.get('backend', 'ollama')} | MCP: {'ON' if self.mcp_enabled else 'OFF'}"
        )
        if self.use_stt:
            logger.info("Voice: STT enabled")
        if self.use_tts:
            logger.info("Voice: TTS enabled")

    def _build_trigger_pattern(self):
        if not self.trigger_commands:
            return
        self._trigger_words_sets = [
            set(t.lower().split()) for t in self.trigger_commands
        ]

    def detected_trigger(self, content: str) -> bool:
        if (
            not content
            or not hasattr(self, "_trigger_words_sets")
            or not self._trigger_words_sets
        ):
            return False
        words = set(
            "".join(
                c.lower() if c.isalnum() or c.isspace() else " " for c in content
            ).split()
        )
        return any(t.issubset(words) for t in self._trigger_words_sets)

    def start_active_listening(self):
        self.is_actively_listening = True
        self.listening_timer = threading.Timer(
            self.active_listening_timeout, self.stop_active_listening
        )
        self.listening_timer.start()

        if self.tts_handler and self.listening_msg:
            try:
                self.tts_handler.send_tts_prompt(self.listening_msg)
            except Exception as e:
                logger.warning(f"TTS failed: {e}")

        logger.info("Active listening started")

    def stop_active_listening(self):
        self.is_actively_listening = False
        if self.listening_timer:
            self.listening_timer.cancel()
            self.listening_timer = None
        logger.info("Active listening stopped")

    def _ensure_tts_handler(self) -> bool:
        if self.tts_handler is not None:
            return True
        if not self.use_tts:
            return False
        try:
            from NativaGPT.lib.text_to_speech.tts_prompt_handler import TTSPromptHandler

            self.tts_handler = TTSPromptHandler(self.config)
            logger.info("✓ TTS handler created")
            return True
        except Exception as e:
            logger.error(f"TTS handler failed: {e}")
            self.use_tts = False
            return False

    def _ensure_stt_handler(self) -> bool:
        if self.stt_handler is not None:
            return True
        if not self.use_stt:
            return False
        try:
            from NativaGPT.lib.speech_to_text.stt_prompt_handler import STTPromptHandler

            self.stt_handler = STTPromptHandler(self.config)
            logger.info("✓ STT handler created")
            return True
        except Exception as e:
            logger.error(f"STT handler failed: {e}")
            self.use_stt = False
            return False

    def _reset_interaction_state(self):
        self._last_topic_context = None
        if self.topic_reader and hasattr(self.topic_reader, "clear_history"):
            try:
                self.topic_reader.clear_history()
            except:
                pass
        try:
            self.llm_handler.cleanup()
        except:
            pass

    def _format_rag_knowledge(self, retrieved_knowledge: Any) -> str:
        if isinstance(retrieved_knowledge, str):
            return retrieved_knowledge
        if isinstance(retrieved_knowledge, list):
            output = StringIO()
            for i, item in enumerate(retrieved_knowledge, 1):
                if isinstance(item, tuple) and len(item) >= 3:
                    entry, score, text = item[0], item[1], item[2]
                    output.write(f"Result {i} (relevance: {score:.3f}):\n")
                    output.write(f"{text[:500]}...\n\n")
                else:
                    output.write(f"Result {i}: {str(item)[:500]}\n\n")
            return output.getvalue()
        return str(retrieved_knowledge)

    def _build_enhanced_prompt(self, prompt: str, rag_knowledge: Optional[str]) -> str:
        output = StringIO()
        if self.setup_prompt:
            output.write(self.setup_prompt + "\n\n")
        output.write(f"The user input is: {prompt}\n\n")
        if rag_knowledge:
            output.write(f"Retrieved RAG knowledge:\n{rag_knowledge}")
        return output.getvalue()

    def _process_text_pipeline(self, prompt: str):
        """Standard LLM pipeline without MCP tools."""
        self._reset_interaction_state()
        logger.info("Processing with standard LLM pipeline...")

        try:
            rag_knowledge = self.rag_similarity_check.retrieve(prompt)
            enhanced_prompt = self._build_enhanced_prompt(
                prompt, self._format_rag_knowledge(rag_knowledge)
            )
            parsed_response = self.llm_handler.send_to_llm(enhanced_prompt)

            if not parsed_response or not parsed_response.get("success", False):
                logger.error("LLM request failed")
                return

        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return

        response_text = parsed_response.get("text_content", "")
        json_strings = parsed_response.get("json_strings", [])

        if response_text:
            logger.info(f"Response: {response_text}")

        if self.use_tts and response_text and self._ensure_tts_handler():
            try:
                self.tts_handler.send_tts_prompt(response_text)
            except Exception as e:
                logger.warning(f"TTS failed: {e}")

        if json_strings:
            self._execute_agentic_loop(
                json_strings, max_iterations=self.max_agentic_iterations
            )

        self._reset_interaction_state()

    def _execute_agentic_loop(self, json_strings: list, max_iterations: int = 5):
        iteration = 0
        executed_fingerprints = set()

        while iteration < max_iterations and json_strings:
            iteration += 1
            try:
                functions = self.json_response_handler.check_all_functions(json_strings)
                if not functions:
                    break
                commands = self.json_response_handler.get_function_command(functions)

                from hashlib import sha1

                fp = sha1("|".join(sorted(json_strings)).encode()).hexdigest()
                if fp in executed_fingerprints:
                    break

                commands_output = self.command_execution.get_commands_output(commands)
                executed_fingerprints.add(fp)

                logger.info(
                    f"Iteration {iteration}: Executed {len(commands)} command(s)"
                )

                feedback_response = self.llm_handler.send_output_to_llm(commands_output)
                if not feedback_response or not feedback_response.get("success", False):
                    break

                new_json_strings = feedback_response.get("json_strings", [])
                if new_json_strings:
                    json_strings = new_json_strings
                else:
                    break
            except Exception as e:
                logger.error(f"Agentic loop error: {e}")
                break

    def _voice_mode_loop(self):
        if not self.use_stt:
            logger.warning("Voice mode requested but STT is disabled")
            self.text_mode = True
            return

        if not self._ensure_stt_handler():
            logger.error("Cannot enter voice mode without STT")
            self.text_mode = True
            return

        try:
            self.stt_handler.start_stt_handler()
            logger.info("🎤 Voice mode active. Listening for wake word...")

            while not self.text_mode:
                if self._stdin_ready():
                    user_cmd = sys.stdin.readline().strip()
                    if self._handle_meta_command(user_cmd):
                        if self.text_mode:
                            break
                        continue

                if self.stt_handler.process_audio():
                    stt_response = self.stt_handler.get_response()
                    if stt_response and "transcription" in stt_response:
                        transcription = stt_response["transcription"].strip()
                        if transcription:
                            logger.info(f"🎤 Heard: {transcription}")

                            if not self.is_actively_listening and self.detected_trigger(
                                transcription
                            ):
                                logger.info("✓ Wake word detected!")
                                self.start_active_listening()

                            if self.is_actively_listening or not self.trigger_commands:
                                mode = "MCP" if self.mcp_enabled else "Standard"
                                logger.info(f"Processing [{mode}]: {transcription}")

                                if self.mcp_enabled and self.mcp_client:
                                    try:
                                        resp = self.mcp_loop.run_coroutine(
                                            self.mcp_client.process_query(transcription)
                                        )
                                        logger.info(f"Response: {resp}")
                                        if self.use_tts and self._ensure_tts_handler():
                                            self.tts_handler.send_tts_prompt(resp)
                                    except Exception as e:
                                        logger.error(f"MCP error: {e}")
                                else:
                                    self._process_text_pipeline(transcription)

                time.sleep(0.01)

        except Exception as e:
            logger.error(f"Voice mode error: {e}")
        finally:
            if self.stt_handler:
                try:
                    self.stt_handler.stop_stt_handler()
                except:
                    pass
            logger.info("Exited voice mode")

    def _text_mode_loop(self):
        while True:
            mode = "[MCP]" if self.mcp_enabled else "[STD]"
            logger.info("Commands: /quit | /voice | /mcp | /help")

            try:
                user_input = input(f"{mode} > ").strip()
            except EOFError:
                raise KeyboardInterrupt
            except KeyboardInterrupt:
                raise

            if self._handle_meta_command(user_input):
                if not self.text_mode:
                    break
                continue

            if not user_input:
                continue

            if not self.mcp_enabled:
                self._process_text_pipeline(user_input)
            else:
                try:
                    logger.info("Processing with MCP...")
                    response = self.mcp_loop.run_coroutine(
                        self.mcp_client.process_query(user_input)
                    )
                    logger.info(f"\n{response}\n")

                    if self.use_tts and self._ensure_tts_handler():
                        try:
                            self.tts_handler.send_tts_prompt(response)
                        except Exception as e:
                            logger.warning(f"TTS failed: {e}")

                except Exception as e:
                    logger.error(f"MCP error: {e}")
                    logger.info("Falling back to standard mode...")
                    self._process_text_pipeline(user_input)

    def _handle_meta_command(self, text: str) -> bool:
        if not text.startswith("/"):
            return False

        cmd = text[1:].strip().lower()
        if not cmd:
            return True

        if cmd in ("quit", "exit", "q"):
            raise KeyboardInterrupt

        if cmd == "voice":
            if not self.use_stt:
                logger.warning("Voice mode disabled (set voice.enabled: true)")
                return True
            self.text_mode = False
            logger.info("Switching to VOICE mode")
            return True

        if cmd == "text":
            self.text_mode = True
            logger.info("Switched to TEXT mode")
            return True

        if cmd in ("mcp", "tool", "tools"):
            if self.mcp_client:
                self.mcp_enabled = not self.mcp_enabled
                logger.info(f"MCP: {'ENABLED' if self.mcp_enabled else 'DISABLED'}")
            else:
                logger.warning("MCP not available")
            return True

        if cmd in ("help", "h", "?"):
            logger.info(
                """
╔════════════════════════════════════════════════════╗
║             NativaGPT v3.0 Commands                ║
╚════════════════════════════════════════════════════╝

  /quit          - Exit
  /voice         - Switch to voice mode
  /text          - Switch to text mode  
  /mcp           - Toggle MCP tools on/off
  /help          - Show this help

Status:
  Backend: """
                + self.llm_handler.backend
                + """
  MCP: """
                + ("ON" if self.mcp_enabled else "OFF")
                + """
  Voice: """
                + ("STT+TTS" if self.use_stt and self.use_tts else "OFF")
                + """
"""
            )
            return True

        logger.warning(f"Unknown command: /{cmd}")
        return True

    def _stdin_ready(self) -> bool:
        return select.select([sys.stdin], [], [], 0.0)[0] != []

    def start(self):
        logger.info("=" * 50)
        logger.info("Starting NativaGPT v3.0")
        logger.info("=" * 50)

        try:
            while True:
                try:
                    if not self.text_mode and self.use_stt:
                        self._voice_mode_loop()
                    else:
                        self._text_mode_loop()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.error(f"Loop error: {e}")
                    time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\nShutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up...")

        if self.stt_handler:
            try:
                self.stt_handler.stop_stt_handler()
            except:
                pass

        try:
            self.executor.shutdown(wait=False)
        except:
            pass

        if self.listening_timer:
            try:
                self.listening_timer.cancel()
            except:
                pass

        if self.mcp_client and self.mcp_loop:
            try:
                self.mcp_loop.run_coroutine(self.mcp_client.cleanup())
            except:
                pass

        if self.mcp_loop:
            try:
                self.mcp_loop.stop()
            except:
                pass

        logger.info("✓ Cleanup completed")


def main():
    script_dir = pathlib.Path(__file__).parent.parent.parent
    cfg_path = script_dir / "config" / "config_default.json"

    if not cfg_path.exists():
        logger.error(f"Config not found: {cfg_path}")
        sys.exit(1)

    logger.info(f"Loading config: {cfg_path}")
    config_manager = ConfigManager(config_path=str(cfg_path))
    config = config_manager.get()

    nativa = NativaGPT(config)
    nativa.start()


if __name__ == "__main__":
    main()
