#!/usr/bin/env python3
"""Flask REST API exposing NativaGPT's MCP wrapper over HTTP.

Wraps a single global ``NativaMCPWrapper`` instance behind HTTP endpoints
(chat, history, tool listing, status, health, shutdown) so that
non-Python clients — such as ROS nodes — can drive NativaGPT's LLM/MCP
pipeline over a local network connection instead of embedding it directly.
Requires Python 3.10+.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import sys
import os
import signal
from typing import Dict, Any, Optional
from pathlib import Path
import time

# Add NativaGPT to path if needed
nativagpt_path = os.path.expanduser("~/Documents/uv-projects/NativaGPT")
if nativagpt_path not in sys.path:
    sys.path.insert(0, nativagpt_path)

from NativaGPT.scripts.nativa_mcp_wrapper import NativaMCPWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global wrapper instance
wrapper: Optional[NativaMCPWrapper] = None

# Configuration
DEFAULT_REQUEST_TIMEOUT = 120  # seconds - 2 minutes for LLM processing
MAX_REQUEST_TIMEOUT = 180  # seconds - 3 minutes maximum


def initialize_wrapper(config_path: str = None):
    """Create the global ``wrapper`` instance from a config file.

    If ``config_path`` is not given, searches a few candidate locations
    relative to this script for ``config/config_default.json`` and uses
    the first one that exists.

    Args:
        config_path: Explicit path to a config JSON file, or None to
            auto-detect ``config_default.json``.

    Returns:
        True if the wrapper was created successfully; False if no config
        file could be found/read or wrapper construction raised.
    """
    global wrapper
    try:
        if config_path is None:
            # Try multiple possible locations
            script_path = Path(__file__).resolve()

            possible_paths = [
                script_path.parent.parent.parent / "config" / "config_default.json",
                script_path.parent.parent / "config" / "config_default.json",
                script_path.parent / "config" / "config_default.json",
            ]

            # Find first existing config
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    logger.info(f"Found config at: {config_path}")
                    break

            if config_path is None:
                logger.error(f"Config not found. Tried locations:")
                for path in possible_paths:
                    logger.error(f"  - {path}")
                return False

        if not os.path.exists(config_path):
            logger.error(f"Config file not found: {config_path}")
            return False

        logger.info(f"Initializing NativaMCPWrapper with config: {config_path}")
        wrapper = NativaMCPWrapper(config_path=config_path, max_history=10)

        logger.info("✓ NativaMCPWrapper initialized successfully")
        logger.info(f"✓ Loaded {wrapper.get_loaded_tools_count()} MCP servers")
        logger.info(f"✓ Connected servers: {', '.join(wrapper.list_server_names())}")
        return True

    except Exception as e:
        logger.error(f"✗ Failed to initialize NativaMCPWrapper: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route("/", methods=["GET"])
def index():
    """Root endpoint listing service metadata and available endpoints.

    Returns:
        A 200 JSON response describing the service name/version/status,
        whether the wrapper is initialized, a map of available endpoints,
        and the configured request timeouts.
    """
    return jsonify({
        "service": "NativaGPT REST API",
        "version": "1.0.0",
        "status": "running",
        "initialized": wrapper is not None,
        "endpoints": {
            "health": "GET /health",
            "status": "GET /status",
            "chat": "POST /chat",
            "history": "GET /history",
            "clear_history": "DELETE /history",
            "tools": "GET /tools",
            "shutdown": "POST /shutdown"
        },
        "timeout": {
            "default": DEFAULT_REQUEST_TIMEOUT,
            "max": MAX_REQUEST_TIMEOUT
        }
    }), 200


@app.route("/health", methods=["GET"])
def health_check():
    """Lightweight health check endpoint for uptime/readiness probes.

    Returns:
        A 200 JSON response with a static "healthy" status, whether the
        wrapper is initialized, and the list of currently connected MCP
        server names (empty if not initialized).
    """
    return jsonify({
        "status": "healthy",
        "service": "NativaGPT",
        "initialized": wrapper is not None,
        "servers": wrapper.list_server_names() if wrapper else [],
        "default_timeout": DEFAULT_REQUEST_TIMEOUT,
        "max_timeout": MAX_REQUEST_TIMEOUT,
    }), 200


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.

    Expected JSON body:
    {
        "message": "user message here",
        "clear_history": false,  # optional
        "timeout": 120  # optional, max processing time
    }

    Returns:
    {
        "success": true,
        "response": "assistant response",
        "tools_called": [...],
        "raw_result": ...,
        "history_length": 5,
        "processing_time": 2.5
    }
    """
    if wrapper is None:
        return jsonify({"success": False, "error": "Service not initialized"}), 503

    start_time = time.time()

    try:
        data = request.json
        if not data or "message" not in data:
            return jsonify({
                "success": False,
                "error": 'Missing "message" field in request body',
            }), 400

        user_message = data["message"]
        clear_history = data.get("clear_history", False)

        # Validate message
        if not user_message or not user_message.strip():
            return jsonify({
                "success": False,
                "error": "Empty message not allowed"
            }), 400

        # Clear history if requested
        if clear_history:
            wrapper.clear_history()
            logger.info("Conversation history cleared")

        # Process the message
        logger.info(f"Processing message: {user_message[:50]}...")

        # Call wrapper directly - it has internal async handling
        result = wrapper.ask(user_message)

        processing_time = time.time() - start_time

        # Build response
        response_data = {
            "success": True,
            "response": result.get("response", ""),
            "tools_called": result.get("tools_called", []),
            "raw_result": result.get("raw_result"),
            "history_length": wrapper.get_history_length(),
            "processing_time": round(processing_time, 2),
        }

        logger.info(
            f"Response generated in {processing_time:.2f}s (tools: {len(result.get('tools_called', []))})"
        )
        return jsonify(response_data), 200

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing chat request after {processing_time:.2f}s: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": str(e),
            "processing_time": round(processing_time, 2)
        }), 500


@app.route("/history", methods=["GET"])
def get_history():
    """Return the full conversation history and its length/cap.

    Returns:
        A 200 JSON response with ``history`` (list of exchanges),
        ``history_length``, and ``max_history`` on success; 503 if the
        wrapper is not initialized; 500 with an error message on failure.
    """
    if wrapper is None:
        return jsonify({"success": False, "error": "Service not initialized"}), 503

    try:
        return jsonify({
            "success": True,
            "history": wrapper.conversation_history,
            "history_length": wrapper.get_history_length(),
            "max_history": wrapper.max_history_length,
        }), 200
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/history", methods=["DELETE"])
def clear_history():
    """Clear the wrapper's conversation history.

    Returns:
        A 200 JSON success response on success; 503 if the wrapper is not
        initialized; 500 with an error message on failure.
    """
    if wrapper is None:
        return jsonify({"success": False, "error": "Service not initialized"}), 503

    try:
        wrapper.clear_history()
        logger.info("History cleared via API")
        return jsonify({"success": True, "message": "History cleared"}), 200
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/tools", methods=["GET"])
def list_tools():
    """List connected MCP server names and the number of loaded tools.

    Returns:
        A 200 JSON response with ``servers`` and ``tool_count`` on
        success; 503 if the wrapper is not initialized; 500 with an error
        message on failure.
    """
    if wrapper is None:
        return jsonify({"success": False, "error": "Service not initialized"}), 503

    try:
        return jsonify({
            "success": True,
            "servers": wrapper.list_server_names(),
            "tool_count": wrapper.get_loaded_tools_count(),
        }), 200
    except Exception as e:
        logger.error(f"Error listing tools: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/status", methods=["GET"])
def get_status():
    """Return a detailed status snapshot of the running service.

    Returns:
        A 200 JSON response with a ``status`` object covering
        initialization state, connected servers/tool count, conversation
        history length/cap, and configured timeouts; 503 if the wrapper
        is not initialized; 500 with an error message on failure.
    """
    if wrapper is None:
        return jsonify({
            "success": False,
            "error": "Service not initialized"
        }), 503

    try:
        return jsonify({
            "success": True,
            "status": {
                "initialized": wrapper is not None,
                "servers": wrapper.list_server_names(),
                "server_count": len(wrapper.list_server_names()),
                "tool_count": wrapper.get_loaded_tools_count(),
                "history_length": wrapper.get_history_length(),
                "max_history": wrapper.max_history_length,
                "timeouts": {
                    "default": DEFAULT_REQUEST_TIMEOUT,
                    "max": MAX_REQUEST_TIMEOUT
                }
            }
        }), 200
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """Shut down the MCP wrapper and (if possible) the Flask dev server.

    Shuts down and clears the global ``wrapper``, then attempts to invoke
    Werkzeug's ``werkzeug.server.shutdown`` function from the request
    environment (only available when running the Flask development
    server outside threaded/reloader edge cases). If that function is
    unavailable, the process itself is not stopped and the caller is
    advised to press Ctrl+C.

    Returns:
        A 200 JSON success response describing whether the shutdown
        signal was actually sent to the server; 500 with an error message
        on failure.
    """
    global wrapper
    try:
        if wrapper:
            logger.info("Shutting down NativaMCPWrapper...")
            wrapper.shutdown()
            wrapper = None
            logger.info("✓ Wrapper shutdown complete")

        logger.info("Service shutdown complete")

        # Shutdown Flask
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            return jsonify({
                "success": True,
                "message": "Shutdown signal sent (press Ctrl+C to stop)",
            }), 200

        func()
        return jsonify({
            "success": True,
            "message": "Service shutdown initiated"
        }), 200

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    """Flask error handler for unmatched routes (404 Not Found).

    Args:
        e: The exception raised by Flask/Werkzeug for the missing route.

    Returns:
        A 404 JSON error response pointing the caller to ``GET /``.
    """
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "message": "Visit GET / for available endpoints"
    }), 404


@app.errorhandler(504)
def handle_timeout(e):
    """Flask error handler for gateway timeouts (504 Gateway Timeout).

    Args:
        e: The exception raised for the timed-out request.

    Returns:
        A 504 JSON error response indicating the request took too long.
    """
    return jsonify({
        "success": False,
        "error": "Request timed out",
        "message": "The request took too long to process."
    }), 504


@app.errorhandler(500)
def handle_internal_error(e):
    """Flask error handler for uncaught server errors (500 Internal Server Error).

    Args:
        e: The exception that caused the 500 response.

    Returns:
        A 500 JSON error response including the exception message.
    """
    logger.error(f"Internal server error: {e}")
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "message": str(e)
    }), 500


def signal_handler(sig, frame):
    """OS signal handler that shuts down the wrapper and exits the process.

    Registered for SIGINT/SIGTERM in ``main`` so that Ctrl+C or a service
    manager stop request cleanly releases MCP connections before exiting.

    Args:
        sig: The signal number received.
        frame: The current stack frame (unused; required by the signal
            handler signature).
    """
    logger.info("\n🛑 Received shutdown signal")
    global wrapper
    if wrapper:
        try:
            wrapper.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    sys.exit(0)


def main():
    """CLI entry point: parse args, initialize the wrapper, and run the Flask server.

    Supports ``--config``, ``--host``, ``--port``, ``--timeout`` and
    ``--debug`` command-line options. Registers SIGINT/SIGTERM handlers,
    initializes the global wrapper (exiting with status 1 on failure),
    then runs the Flask app in threaded mode until interrupted, shutting
    down the wrapper in the ``finally`` block regardless of outcome.
    """
    import argparse

    parser = argparse.ArgumentParser(description="NativaGPT REST Service")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config JSON file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=5555, help="Port to bind to (default: 5555)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Default request timeout in seconds (default: 120)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Update timeout if specified
    global DEFAULT_REQUEST_TIMEOUT
    DEFAULT_REQUEST_TIMEOUT = args.timeout
    logger.info(f"Request timeout set to {args.timeout}s")

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize wrapper
    logger.info("=" * 50)
    logger.info("  NativaGPT REST Service")
    logger.info("=" * 50)

    if not initialize_wrapper(args.config):
        logger.error("Failed to initialize. Exiting.")
        sys.exit(1)

    # Start Flask server
    logger.info(f"Starting service on {args.host}:{args.port}")
    logger.info(f"Request timeout: {DEFAULT_REQUEST_TIMEOUT}s (max: {MAX_REQUEST_TIMEOUT}s)")
    logger.info(f"\nEndpoints available:")
    logger.info(f"  - GET    /              (Service info)")
    logger.info(f"  - GET    /health        (Health check)")
    logger.info(f"  - GET    /status        (Detailed status)")
    logger.info(f"  - POST   /chat          (Send message)")
    logger.info(f"  - GET    /history       (Get history)")
    logger.info(f"  - DELETE /history       (Clear history)")
    logger.info(f"  - GET    /tools         (List tools)")
    logger.info(f"  - POST   /shutdown      (Shutdown service)")
    logger.info("")
    logger.info(f"Open in browser: http://{args.host}:{args.port}/")
    logger.info("")

    try:
        # Use threaded mode for better concurrent handling
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True,
            use_reloader=False,
        )
    except KeyboardInterrupt:
        logger.info("\nReceived interrupt signal")
    finally:
        if wrapper:
            try:
                wrapper.shutdown()
            except Exception as e:
                logger.error(f"Error during final shutdown: {e}")
        logger.info("Service stopped")


if __name__ == "__main__":
    main()