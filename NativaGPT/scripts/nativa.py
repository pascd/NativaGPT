#!/usr/bin/env python3
"""
NativaGPT v2.2 - Fixed Version
Main entry point with proper lazy initialization
"""
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
        self._started.wait()

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
    NativaGPT v2.2 - Fixed Mode Handling

    CRITICAL FIXES:
    - ✅ STT/TTS handlers are NOT created during __init__
    - ✅ Handlers only created when entering their respective modes
    - ✅ Proper config validation before handler creation
    - ✅ Clean mode switching with status messages
    """

    _trigger_pattern = None

    def __init__(self, config):
        logger.info("Initializing NativaGPT v2.2 (FIXED VERSION)...")

        # Core configuration
        self.config = config

        nativa_cfg = self.config.get("nativa_gpt", {})
        llm_cfg = self.config.get("llm_config", {})

        # Check config for TTS/STT availability (NOT creating handlers yet!)
        self.use_tts = bool(nativa_cfg.get("use_tts", False))
        self.use_stt = bool(nativa_cfg.get("use_stt", False))

        self.active_listening_timeout = nativa_cfg.get("active_listening_timeout", 30)
        self.trigger_commands = nativa_cfg.get("user_msgs", {}).get("trigger_commands", [])
        self.listening_msg = nativa_cfg.get("user_msgs", {}).get("listening_msg", "Listening...")
        self.setup_prompt = llm_cfg.get("model_config", {}).get("setup_prompt", "")
        self.max_agentic_iterations = nativa_cfg.get("max_agentic_iterations", 5)

        # Initialize core handlers (these are always needed)
        self.llm_handler = LLMPromptHandler(config)
        self.llm_response_handler = LLMResponseHandler()
        self.json_response_handler = JsonResponseHandler()
        self.rag_similarity_check = RAGSimilarityCheck(config)
        self.topic_reader = TopicReaderHandler(config)

        # Command execution
        self.command_execution = CommandExecution(topic_reader_handler=self.topic_reader)

        # State management
        self._last_topic_context: Optional[Dict[str, Any]] = None
        self.is_actively_listening = False
        self.listening_timer = None
        self.text_mode = True

        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nativa")

        # Build trigger pattern
        self._build_trigger_pattern()

        # ================================================================
        # CRITICAL FIX: DO NOT CREATE STT/TTS HANDLERS HERE!
        # They will be created lazily when needed
        # ================================================================
        self.stt_handler = None
        self.tts_handler = None

        # Log configuration status
        if self.use_tts:
            logger.info("✓ TTS enabled in config (will initialize on demand)")
        else:
            logger.info("✗ TTS disabled in config")

        if self.use_stt:
            logger.info("✓ STT enabled in config (will initialize when entering voice mode)")
        else:
            logger.info("✗ STT disabled in config")

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
                # Expand home directory
                host_path = os.path.expanduser(host_path)
                self.mcp_server_hosts.append(host_path)

        # Setup MCP infrastructure
        if self.mcp_server_hosts:
            try:
                self.mcp_loop = AsyncLoopThread()
                self.mcp_loop.start()

                self.mcp_client = MCPClient(self.llm_handler)
                self.mcp_loop.run_coroutine(
                    self.mcp_client.connect_to_server(self.mcp_server_hosts)
                )
                logger.info(f"✓ MCP infrastructure ready with {len(self.mcp_server_hosts)} server(s)")
            except Exception as e:
                logger.error(f"Failed to initialize MCP infrastructure: {e}")
                import traceback
                traceback.print_exc()
                self.mcp = False
        else:
            logger.info("ℹ No MCP servers configured")
            self.mcp = False

        logger.info(f"NativaGPT v2.2 initialized successfully!")
        logger.info(f"Mode: {'MCP' if self.mcp else 'Standard'} | Input: {'Text' if self.text_mode else 'Voice'}")

    def _toggle_mcp_mode(self):
        """Toggle MCP on/off."""
        if not self.mcp_client:
            logger.warning("⚠ Cannot toggle MCP: No servers configured")
            return

        self.mcp = not self.mcp
        status = "ENABLED" if self.mcp else "DISABLED"

        logger.info(f"\n{'='*50}")
        logger.info(f"   MCP MODE: {status}")
        logger.info(f"{'='*50}")

    def _build_trigger_pattern(self):
        """Build optimized regex pattern for trigger detection."""
        if not self.trigger_commands:
            return

        self._trigger_words_sets = []
        for trigger in self.trigger_commands:
            words = set(trigger.lower().split())
            self._trigger_words_sets.append(words)

    def detected_trigger(self, content: str) -> bool:
        if not content or not hasattr(self, '_trigger_words_sets') or not self._trigger_words_sets:
            return False
        words = set(char.lower() for char in content if char.isalnum() or char.isspace())
        cleaned = ''.join(words).split()
        content_words = set(cleaned)
        for trigger_words in self._trigger_words_sets:
            if trigger_words.issubset(content_words):
                return True
        return False

    def start_active_listening(self):
        self.is_actively_listening = True
        self.listening_timer = threading.Timer(
            self.active_listening_timeout,
            self.stop_active_listening
        )
        self.listening_timer.start()

        # Use TTS if available
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

    # ====================== LAZY HANDLER CREATION ======================

    def _ensure_tts_handler(self):
        """Create TTS handler on-demand."""
        if self.tts_handler is not None:
            return True

        if not self.use_tts:
            logger.warning("⚠ TTS is disabled in config")
            return False

        try:
            from NativaGPT.lib.text_to_speech.tts_prompt_handler import TTSPromptHandler
            logger.info("Creating TTS handler...")
            self.tts_handler = TTSPromptHandler(self.config)
            logger.info("✓ TTS handler created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create TTS handler: {e}")
            self.use_tts = False
            return False

    def _ensure_stt_handler(self):
        """Create STT handler on-demand."""
        if self.stt_handler is not None:
            return True

        if not self.use_stt:
            logger.warning("⚠ STT is disabled in config")
            return False

        try:
            from NativaGPT.lib.speech_to_text.stt_prompt_handler import STTPromptHandler
            logger.info("Creating STT handler for voice mode...")
            self.stt_handler = STTPromptHandler(self.config)
            logger.info("✓ STT handler created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create STT handler: {e}")
            import traceback
            traceback.print_exc()
            self.use_stt = False
            return False

    # ====================== CORE PROCESSING ======================

    def _reset_interaction_state(self):
        self._last_topic_context = None
        if hasattr(self.topic_reader, "clear_history"):
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
                output.write(f"Result {i}: {str(item)}\n\n")
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

    def _process_text_pipeline_api(self, prompt: str):
        """Standard LLM pipeline."""
        self._reset_interaction_state()
        logger.info("Processing with standard LLM pipeline...")

        try:
            rag_knowledge = self.rag_similarity_check.retrieve(prompt)
            enhanced_prompt = self._build_enhanced_prompt(prompt, rag_knowledge)
            parsed_response = self.llm_handler.send_to_llm(enhanced_prompt)

            if not parsed_response or not parsed_response.get("success", False):
                logger.error("LLM API failed")
                return

        except Exception as e:
            logger.error(f"LLM request failed: {e}")
            return

        response_text = parsed_response.get("text_content", "")
        json_strings = parsed_response.get("json_strings", [])

        if response_text:
            logger.info(f"Response: {response_text}")

        # Use TTS if enabled and available
        if self.use_tts and response_text:
            if self._ensure_tts_handler():
                try:
                    self.tts_handler.send_tts_prompt(response_text)
                except Exception as e:
                    logger.warning(f"TTS failed: {e}")

        if json_strings:
            self._execute_agentic_loop(json_strings, max_iterations=self.max_agentic_iterations)

        self._reset_interaction_state()

    def _execute_agentic_loop(self, json_strings: List[str], max_iterations: int = 5):
        """Execute commands with feedback loop."""
        iteration = 0
        executed_fingerprints = set()

        while iteration < max_iterations and json_strings:
            iteration += 1
            try:
                functions = self.json_response_handler.check_all_functions(json_strings)
                if not functions:
                    break
                commands = self.json_response_handler.get_function_command(functions)

                fp = sha1("|".join(sorted(json_strings)).encode("utf-8")).hexdigest()
                if fp in executed_fingerprints:
                    break

                commands_output = self.command_execution.get_commands_output(commands)
                executed_fingerprints.add(fp)

                logger.info(f"Iteration {iteration}: Executed {len(commands)} command(s)")

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

    # ====================== MODE LOOPS ======================

    def _voice_mode_loop(self):
        """Voice mode with lazy STT initialization."""
        if not self.use_stt:
            logger.warning("⚠ Voice mode requested but STT is disabled in config")
            self.text_mode = True
            return

        # Create STT handler NOW (not during __init__)
        if not self._ensure_stt_handler():
            logger.error("Cannot enter voice mode without STT")
            self.text_mode = True
            return

        # Start listening
        try:
            self.stt_handler.start_stt_handler()
            logger.info("🎤 Voice mode active. Listening for wake word...")
            logger.info("Commands: /text (exit voice) | /mcp (toggle tools)")

            while not self.text_mode:
                # Check for keyboard commands
                if self._stdin_ready():
                    user_cmd = sys.stdin.readline().strip()
                    if self._handle_meta_command(user_cmd):
                        if self.text_mode:
                            break
                        continue

                # Process audio
                if self.stt_handler.process_audio():
                    stt_response = self.stt_handler.get_response()
                    if stt_response and "transcription" in stt_response:
                        transcription = stt_response["transcription"].strip()
                        if transcription:
                            logger.info(f"🎤 Heard: {transcription}")

                            # Check for wake word
                            if not self.is_actively_listening and self.detected_trigger(transcription):
                                logger.info("✓ Wake word detected!")
                                self.start_active_listening()

                            # Process if listening
                            if self.is_actively_listening or not self.trigger_commands:
                                mode = "MCP" if self.mcp else "Standard"
                                logger.info(f"Processing in {mode} mode: {transcription}")

                                if self.mcp and self.mcp_client:
                                    # MCP mode
                                    try:
                                        resp = self.mcp_loop.run_coroutine(
                                            self.mcp_client.process_query(transcription)
                                        )
                                        logger.info(f"MCP Response: {resp}")

                                        # TTS response
                                        if self.use_tts and self._ensure_tts_handler():
                                            self.tts_handler.send_tts_prompt(resp)
                                    except Exception as e:
                                        logger.error(f"MCP Error: {e}")
                                        import traceback
                                        traceback.print_exc()
                                else:
                                    # Standard mode
                                    self._process_text_pipeline_api(transcription)

                time.sleep(0.01)

        except Exception as e:
            logger.error(f"Error in voice mode: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.stt_handler:
                try:
                    self.stt_handler.stop_stt_handler()
                except Exception as e:
                    logger.error(f"Error stopping STT: {e}")
            logger.info("Exited voice mode")

    def _text_mode_loop(self):
        """Text mode loop."""
        while True:
            mode_indicator = "[MCP]" if self.mcp else "[STD]"
            logger.info("Commands: /quit | /voice | /text | /mcp | /mode [mcp|standard|voice|text] | /help")

            try:
                user_input = input(f"{mode_indicator} Insert prompt: ").strip()
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

            if not self.mcp:
                # Standard LLM mode
                self._process_text_pipeline_api(user_input)
            else:
                # MCP tool mode
                try:
                    logger.info("Querying MCP...")
                    response = self.mcp_loop.run_coroutine(
                        self.mcp_client.process_query(user_input)
                    )
                    logger.info(f"\n{response}\n")

                    # TTS if available
                    if self.use_tts and self._ensure_tts_handler():
                        try:
                            self.tts_handler.send_tts_prompt(response)
                        except Exception as e:
                            logger.warning(f"TTS failed: {e}")

                except Exception as e:
                    logger.error(f"MCP query failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback
                    logger.info("Falling back to standard mode...")
                    self._process_text_pipeline_api(user_input)

    def _handle_meta_command(self, text: str) -> bool:
        """Handle meta commands."""
        if not text.startswith("/"):
            return False

        cmd = text[1:].strip().lower()
        if not cmd:
            return True

        if cmd in ("quit", "exit", "q"):
            raise KeyboardInterrupt

        if cmd == "voice":
            if not self.use_stt:
                logger.warning("⚠ Voice mode disabled in config (set use_stt: true)")
                return True
            self.text_mode = False
            logger.info("✓ Switching to VOICE mode")
            return True

        if cmd == "text":
            self.text_mode = True
            logger.info("✓ Switched to TEXT mode")
            return True

        if cmd in ("mcp", "tool", "tools"):
            self._toggle_mcp_mode()
            return True

        # Unified /mode command
        if cmd.startswith("mode "):
            mode = cmd[5:].strip().lower()

            if mode == "mcp":
                if self.mcp_client:
                    self.mcp = True
                    logger.info("✓ MCP TOOL MODE ENABLED")
                else:
                    logger.warning("⚠ MCP not available (no servers configured)")

            elif mode in ("standard", "std"):
                self.mcp = False
                logger.info("✓ STANDARD LLM MODE ENABLED")

            elif mode == "voice":
                if self.use_stt:
                    self.text_mode = False
                    logger.info("✓ VOICE MODE ENABLED (will activate next)")
                else:
                    logger.warning("⚠ Voice mode disabled (set use_stt: true in config)")

            elif mode == "text":
                self.text_mode = True
                logger.info("✓ TEXT MODE ENABLED")

            else:
                logger.info("Available: /mode mcp | standard | voice | text")
            return True

        if cmd in ("help", "h", "?"):
            logger.info(f"""
╔══════════════════════════════════════════════════════╗
║             NativaGPT v2.2 - Commands                ║
╚══════════════════════════════════════════════════════╝

Mode Switching:
  /mode mcp          - Enable MCP tools (ROS, weather, etc.)
  /mode standard     - Enable standard LLM mode
  /mode voice        - Enable voice input/output
  /mode text         - Enable keyboard input

Quick Commands:
  /quit              - Exit NativaGPT
  /voice             - Switch to voice mode
  /text              - Switch to text mode
  /mcp               - Toggle MCP on/off
  /help              - Show this help

Current Status:
  STT (voice):  {'✓ enabled' if self.use_stt else '✗ disabled'}
  TTS (speech): {'✓ enabled' if self.use_tts else '✗ disabled'}
  MCP (tools):  {'✓ enabled' if self.mcp else '✗ disabled'}
  Mode:         {'voice' if not self.text_mode else 'text'}

Version: 2.2 (Fixed)
            """)
            return True

        logger.warning(f"⚠ Unknown command: /{cmd}")
        logger.info("Type /help for available commands")
        return True

    def _stdin_ready(self) -> bool:
        """Check if stdin has data ready."""
        return select.select([sys.stdin], [], [], 0.0)[0] != []

    # ====================== MAIN LOOP ======================

    def start(self):
        """Main loop."""
        logger.info("="*60)
        logger.info("Starting NativaGPT v2.2 (Fixed Version)")
        logger.info("="*60)
        logger.info(f"STT:  {'✓ enabled' if self.use_stt else '✗ disabled'}")
        logger.info(f"TTS:  {'✓ enabled' if self.use_tts else '✗ disabled'}")
        logger.info(f"MCP:  {'✓ enabled' if self.mcp else '✗ disabled'}")
        logger.info(f"Mode: {'text' if self.text_mode else 'voice'}")
        logger.info("="*60)
        logger.info("Type /help for available commands")
        logger.info("")

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
                    logger.error(f"Error in main loop: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)

        except KeyboardInterrupt:
            logger.info("\n")
            logger.info("="*60)
            logger.info("Shutting down gracefully...")
            logger.info("="*60)
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up resources...")

        # Stop STT if running
        if self.stt_handler:
            try:
                self.stt_handler.stop_stt_handler()
            except:
                pass

        # Cleanup executors
        try:
            self.executor.shutdown(wait=False)
        except:
            pass

        # Cancel timers
        if self.listening_timer:
            try:
                self.listening_timer.cancel()
            except:
                pass

        # Cleanup MCP
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

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass


def main():
    """Main entry point."""
    # Get config path
    script_dir = pathlib.Path(__file__).parent.parent.parent
    cfg_path = script_dir / "config" / "config_default.json"

    if not cfg_path.exists():
        logger.error(f"Config file not found: {cfg_path}")
        sys.exit(1)

    logger.info(f"Loading config from: {cfg_path}")

    config_manager = ConfigManager(config_path=str(cfg_path))
    config = config_manager.get()

    nativa = NativaGPT(config)
    nativa.start()


if __name__ == "__main__":
    main()