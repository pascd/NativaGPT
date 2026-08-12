#!/usr/bin/env python3
"""Interactive NativaGPT application entry point.

Defines the ``NativaGPT`` orchestrator, which wires together the LLM, MCP
(tool-calling), and RAG subsystems into a single interactive, text-based
assistant.

Also defines ``AsyncLoopThread``, a small helper that runs a persistent
asyncio event loop on a background thread so the (async) MCP client can be
driven from the otherwise synchronous main loop.
"""
import base64
import json
import os
import sys
import time
import signal
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
    """Runs a persistent asyncio event loop on a dedicated background thread.

    This lets synchronous code (the main interaction loop in ``NativaGPT``)
    submit coroutines (e.g. MCP client calls) to a long-lived event loop and
    block until they complete, without needing the whole application to be
    async.
    """

    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._started = threading.Event()

    def start(self):
        """Start the background thread and its event loop, if not already running.

        Blocks until the event loop has been created and is ready to accept
        coroutines. Calling this more than once is a no-op.
        """
        if self.thread is not None:
            return

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        self._started.wait()

    def _run_loop(self):
        """Create the event loop on this thread and run it until stopped."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self._started.set()
        self.loop.run_forever()

    def run_coroutine(self, coro):
        """Schedule a coroutine on the background loop and wait for its result.

        Args:
            coro: The coroutine object to execute.

        Returns:
            The value returned by the coroutine.

        Raises:
            RuntimeError: If the background loop has not been started yet.
        """
        if self.loop is None:
            raise RuntimeError("Event loop not started")
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()

    def stop(self):
        """Stop the background event loop and join its thread.

        Waits up to 5 seconds for the thread to exit.
        """
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread is not None:
            self.thread.join(timeout=5)


class NativaGPT:
    """Main interactive orchestrator for the NativaGPT assistant.

    Coordinates the LLM prompt/response handlers, MCP tool-calling client,
    RAG similarity check, and the agentic command-execution loop, and
    drives them through a keyboard-input interaction loop. Mode switching
    (standard/MCP) is handled via in-session ``/`` meta commands, see
    ``_handle_meta_command``.
    """

    def __init__(self, config):
        """Initialize handlers, state, and (optionally) the MCP client.

        Reads assistant behavior from the ``nativa_gpt`` and ``llm_config``
        sections of ``config``, eagerly creates the always-needed handlers
        (LLM, RAG, topic reader, command execution), and — if any MCP
        server hosts are configured — starts a background event loop and
        connects to them.

        Args:
            config: Parsed application configuration (as returned by
                ``ConfigManager.get()``) containing the ``nativa_gpt``,
                ``llm_config`` and ``mcp`` sections used to configure this
                instance.
        """
        logger.info("Initializing NativaGPT v2.2 (FIXED VERSION)...")

        # Core configuration
        self.config = config

        nativa_cfg = self.config.get("nativa_gpt", {})
        llm_cfg = self.config.get("llm_config", {})

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

        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nativa")

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
        logger.info(f"Mode: {'MCP' if self.mcp else 'Standard'}")

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

    # ====================== CORE PROCESSING ======================

    def _reset_interaction_state(self):
        """Clear per-interaction state between (or before/after) LLM turns.

        Drops the cached topic context, clears the topic reader's history
        (if supported), and cleans up the LLM handler's internal state.
        Errors from either sub-cleanup are silently ignored so a reset
        never blocks the main loop.
        """
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
        """Render RAG-retrieved knowledge into a single prompt-ready string.

        Args:
            retrieved_knowledge: The value returned by
                ``RAGSimilarityCheck.retrieve`` — a string, a list of
                result items, or any other value.

        Returns:
            ``retrieved_knowledge`` unchanged if it is already a string;
            otherwise a numbered "Result N: ..." listing if it is a list;
            otherwise its ``str()`` representation.
        """
        if isinstance(retrieved_knowledge, str):
            return retrieved_knowledge
        if isinstance(retrieved_knowledge, list):
            output = StringIO()
            for i, item in enumerate(retrieved_knowledge, 1):
                output.write(f"Result {i}: {str(item)}\n\n")
            return output.getvalue()
        return str(retrieved_knowledge)

    def _build_enhanced_prompt(self, prompt: str, rag_knowledge: Optional[str]) -> str:
        """Assemble the final prompt sent to the LLM.

        Combines the configured setup/system prompt, the raw user input,
        and any retrieved RAG knowledge into a single text block.

        Args:
            prompt: The raw user input (typed or transcribed).
            rag_knowledge: Formatted RAG knowledge to append, or None/empty
                to omit it.

        Returns:
            The composed prompt string to send to the LLM.
        """
        output = StringIO()
        if self.setup_prompt:
            output.write(self.setup_prompt + "\n\n")
        output.write(f"The user input is: {prompt}\n\n")
        if rag_knowledge:
            output.write(f"Retrieved RAG knowledge:\n{rag_knowledge}")
        return output.getvalue()

    def _process_text_pipeline_api(self, prompt: str):
        """Run the standard (non-MCP) LLM pipeline for one user turn.

        Retrieves RAG knowledge for ``prompt``, builds the enhanced prompt,
        sends it to the LLM, logs the response text, and — if the LLM
        returned any embedded JSON function calls — hands them off to
        ``_execute_agentic_loop`` for execution. Interaction
        state is reset both before and after processing. Any failure to
        reach the LLM is logged and aborts this turn without raising.

        Args:
            prompt: The raw user input (typed or transcribed) to process.
        """
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

        if json_strings:
            self._execute_agentic_loop(json_strings, max_iterations=self.max_agentic_iterations)

        self._reset_interaction_state()

    def _execute_agentic_loop(self, json_strings: List[str], max_iterations: int = 5):
        """Execute LLM-requested function calls and feed their output back to the LLM.

        On each iteration: parses ``json_strings`` into callable functions,
        runs them via ``CommandExecution``, and sends the resulting output
        back to the LLM (``send_output_to_llm``). If the LLM responds with
        further JSON function calls, the loop repeats with those; it stops
        when there are no more functions to call, the LLM call fails, or
        ``max_iterations`` is reached. A SHA1 fingerprint of the current
        batch of JSON strings is tracked to break out early if the same
        exact set of commands would be executed again (avoiding infinite
        repeat loops).

        Args:
            json_strings: JSON-encoded function-call strings extracted from
                the LLM's response.
            max_iterations: Maximum number of execute/feedback round-trips
                to perform before giving up. Defaults to 5.
        """
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

    # ====================== MAIN INTERACTION LOOP ======================

    def _text_mode_loop(self):
        """Run the keyboard-input interaction loop.

        Repeatedly prompts for a line of input, dispatches ``/`` meta
        commands via ``_handle_meta_command``, and routes any other input
        to either the MCP client (if MCP mode is on, with a fallback to the
        standard pipeline on failure) or the standard LLM pipeline
        (``_process_text_pipeline_api``). Raises ``KeyboardInterrupt`` on
        EOF (e.g. Ctrl+D) so the outer ``start`` loop can shut down
        gracefully.
        """
        while True:
            mode_indicator = "[MCP]" if self.mcp else "[STD]"
            logger.info("Commands: /quit | /mcp | /mode [mcp|standard] | /help")

            try:
                user_input = input(f"{mode_indicator} Insert prompt: ").strip()
            except EOFError:
                raise KeyboardInterrupt
            except KeyboardInterrupt:
                raise

            if self._handle_meta_command(user_input):
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

                except Exception as e:
                    logger.error(f"MCP query failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fallback
                    logger.info("Falling back to standard mode...")
                    self._process_text_pipeline_api(user_input)

    def _handle_meta_command(self, text: str) -> bool:
        """Parse and execute a leading-slash meta command, if present.

        Recognized commands include ``/quit`` (``/exit``, ``/q``),
        ``/mcp`` (``/tool``, ``/tools``), ``/mode <mcp|standard>``, and
        ``/help`` (``/h``, ``/?``). Unknown ``/`` commands are logged as a
        warning but still consumed (treated as handled).

        Args:
            text: The raw line of input from the user.

        Returns:
            True if ``text`` started with ``/`` and was handled as a meta
            command (regardless of whether the command was recognized);
            False if ``text`` is regular input to be processed normally.

        Raises:
            KeyboardInterrupt: If the command is ``/quit``, ``/exit`` or
                ``/q``, to signal the main loop to shut down.
        """
        if not text.startswith("/"):
            return False

        cmd = text[1:].strip().lower()
        if not cmd:
            return True

        if cmd in ("quit", "exit", "q"):
            raise KeyboardInterrupt

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

            else:
                logger.info("Available: /mode mcp | standard")
            return True

        if cmd in ("help", "h", "?"):
            logger.info(f"""
╔══════════════════════════════════════════════════════╗
║             NativaGPT v2.2 - Commands                ║
╚══════════════════════════════════════════════════════╝

Mode Switching:
  /mode mcp          - Enable MCP tools (ROS, weather, etc.)
  /mode standard     - Enable standard LLM mode

Quick Commands:
  /quit              - Exit NativaGPT
  /mcp               - Toggle MCP on/off
  /help              - Show this help

Current Status:
  MCP (tools):  {'✓ enabled' if self.mcp else '✗ disabled'}

Version: 2.2 (Fixed)
            """)
            return True

        logger.warning(f"⚠ Unknown command: /{cmd}")
        logger.info("Type /help for available commands")
        return True

    # ====================== MAIN LOOP ======================

    def start(self):
        """Run the assistant until interrupted, then clean up.

        Logs the current MCP configuration, then repeatedly runs the
        keyboard-input interaction loop (``_text_mode_loop``), restarting
        it if it returns. Non-fatal exceptions from the loop are logged and
        the outer loop retries after a short pause. A ``KeyboardInterrupt``
        (from ``/quit`` or Ctrl+C/Ctrl+D) breaks out of the loop and
        triggers a graceful shutdown via ``cleanup`` in the ``finally``
        block.
        """
        logger.info("="*60)
        logger.info("Starting NativaGPT v2.2 (Fixed Version)")
        logger.info("="*60)
        logger.info(f"MCP:  {'✓ enabled' if self.mcp else '✗ disabled'}")
        logger.info("="*60)
        logger.info("Type /help for available commands")
        logger.info("")

        try:
            while True:
                try:
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
        """Release resources held by this instance.

        Shuts down the thread pool executor without waiting, and cleans up
        and stops the MCP client/event loop if they were initialized. All
        sub-steps are best-effort: any exception raised during a step is
        swallowed so cleanup always completes.
        """
        logger.info("Cleaning up resources...")

        # Cleanup executors
        try:
            self.executor.shutdown(wait=False)
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
    """CLI entry point: load config and run the interactive assistant.

    Locates ``config/config_default.json`` relative to the package root,
    exits with status 1 if it is missing, otherwise builds a
    ``ConfigManager``, constructs ``NativaGPT`` from the resulting config,
    and starts its main loop.
    """
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