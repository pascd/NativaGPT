#!/usr/bin/env python3
"""
NativaGPT REST API Service

Provides HTTP API for NativaGPT with tool integration.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys
import os
import signal
from typing import Optional, Dict, Any
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from NativaGPT.scripts.nativa_mcp_wrapper import NativaMCPWrapper

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

wrapper: Optional[NativaMCPWrapper] = None
executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="nativa_api")

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 300


def initialize_wrapper(config_path: Optional[str] = None) -> bool:
    """Initialize NativaMCPWrapper with configuration."""
    global wrapper
    try:
        if config_path is None:
            script_path = Path(__file__).resolve()
            possible_paths = [
                script_path.parent.parent.parent / "config" / "config_default.json",
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break

        if not config_path or not os.path.exists(config_path):
            logger.error(f"Config not found: {config_path}")
            return False

        logger.info(f"Loading config: {config_path}")
        wrapper = NativaMCPWrapper(config_path=config_path, max_history=10)

        logger.info(f"✓ Initialized with {wrapper.get_loaded_tools_count()} servers")
        return True

    except Exception as e:
        logger.error(f"Init failed: {e}")
        import traceback

        traceback.print_exc()
        return False


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "service": "NativaGPT REST API",
            "version": "3.0.0",
            "status": "running",
            "initialized": wrapper is not None,
            "endpoints": {
                "health": "GET /health",
                "status": "GET /status",
                "chat": "POST /chat",
                "history": "GET /history | DELETE /history",
                "tools": "GET /tools",
            },
        }
    ), 200


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify(
        {
            "status": "healthy",
            "initialized": wrapper is not None,
            "servers": wrapper.list_server_names() if wrapper else [],
        }
    ), 200


@app.route("/status", methods=["GET"])
def get_status():
    if wrapper is None:
        return jsonify({"error": "Not initialized"}), 503

    return jsonify(
        {
            "initialized": True,
            "servers": wrapper.list_server_names(),
            "server_count": len(wrapper.list_server_names()),
            "tool_count": wrapper.get_loaded_tools_count(),
            "history_length": wrapper.get_history_length(),
        }
    ), 200


def process_chat(message: str, clear_history: bool = False) -> Dict[str, Any]:
    """Process chat message in thread pool."""
    if wrapper is None:
        return {"success": False, "error": "Not initialized"}

    try:
        if clear_history:
            wrapper.clear_history()

        result = wrapper.ask(message)
        return {
            "success": True,
            "response": result.get("response", ""),
            "tools_called": result.get("tools_called", []),
            "history_length": wrapper.get_history_length(),
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"success": False, "error": str(e)}


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.

    Body: {"message": "...", "clear_history": false, "timeout": 120}
    """
    if wrapper is None:
        return jsonify({"error": "Not initialized"}), 503

    start_time = time.time()

    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({"error": "Missing message"}), 400

        message = data["message"]
        if not message or not message.strip():
            return jsonify({"error": "Empty message"}), 400

        clear_history = data.get("clear_history", False)
        timeout = min(data.get("timeout", DEFAULT_TIMEOUT), MAX_TIMEOUT)

        future = executor.submit(process_chat, message, clear_history)

        try:
            result = future.result(timeout=timeout)
            result["processing_time"] = round(time.time() - start_time, 2)
            status = 200 if result.get("success") else 500
            return jsonify(result), status

        except FuturesTimeoutError:
            return jsonify(
                {
                    "error": "Request timed out",
                    "processing_time": round(time.time() - start_time, 2),
                }
            ), 504

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def get_history():
    if wrapper is None:
        return jsonify({"error": "Not initialized"}), 503

    return jsonify(
        {
            "history": wrapper.conversation_history,
            "length": wrapper.get_history_length(),
        }
    ), 200


@app.route("/history", methods=["DELETE"])
def clear_history():
    if wrapper is None:
        return jsonify({"error": "Not initialized"}), 503

    wrapper.clear_history()
    return jsonify({"success": True, "message": "History cleared"}), 200


@app.route("/tools", methods=["GET"])
def list_tools():
    if wrapper is None:
        return jsonify({"error": "Not initialized"}), 503

    return jsonify(
        {
            "servers": wrapper.list_server_names(),
            "tool_count": wrapper.get_loaded_tools_count(),
        }
    ), 200


@app.route("/shutdown", methods=["POST"])
def shutdown():
    global wrapper
    try:
        if wrapper:
            wrapper.shutdown()
            wrapper = None

        executor.shutdown(wait=False)
        return jsonify({"success": True, "message": "Shutdown complete"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal error"}), 500


def signal_handler(sig, frame):
    logger.info("Shutdown signal received")
    global wrapper
    if wrapper:
        try:
            wrapper.shutdown()
        except:
            pass
    executor.shutdown(wait=False)
    sys.exit(0)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="NativaGPT REST API")
    parser.add_argument("--config", type=str, help="Config file path")
    parser.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5555, help="Port (default: 5555)")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    global DEFAULT_TIMEOUT
    DEFAULT_TIMEOUT = args.timeout

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 50)
    logger.info("  NativaGPT REST API v3.0")
    logger.info("=" * 50)

    if not initialize_wrapper(args.config):
        sys.exit(1)

    logger.info(f"Starting on {args.host}:{args.port}")
    logger.info(f"Timeout: {DEFAULT_TIMEOUT}s")

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if wrapper:
            wrapper.shutdown()
        executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
