#!/usr/bin/env python3
"""
Generic MCP Server for NativaGPT Functions
Loads functions from JSON configuration files and exposes them as MCP tools.
"""

import os
import sys
import json
import subprocess
import re
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
# Disable logging to prevent stdout interference with MCP JSON protocol
# from NativaGPT.lib.coloring_logger import logger


def get_server_name(config_path: str) -> str:
    """Derive an MCP server name from a config file's path.

    Args:
        config_path: Path to the JSON function-definitions file.

    Returns:
        A "nativa-<name>" server name derived from the file stem, with any
        "_functions" suffix stripped and underscores turned into hyphens.
    """
    path = Path(config_path)
    name = path.stem.replace("_functions", "").replace("_", "-")
    return f"nativa-{name}" if name else "nativa-functions"


def load_functions(config_path: str) -> list[dict]:
    """Load a list of function definitions from a JSON config file.

    Args:
        config_path: Path to the JSON file containing a list of function
            definitions.

    Returns:
        The parsed list of function definitions, or an empty list if the
        file is missing, invalid, or does not contain a JSON list.
    """
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        # logger.error(f"Error loading {config_path}: {e}")
        print(f"Error loading {config_path}: {e}", file=sys.stderr)
        return []


def get_args_from_command(command: str) -> list[str]:
    """Extract argument names from `{placeholder}` tokens in a command template.

    Args:
        command: A command string that may contain `{arg_name}` placeholders.

    Returns:
        The list of placeholder names found, in order of appearance.
    """
    return re.findall(r"\{(\w+)\}", command)


def create_tool_function(
    name: str,
    description: str,
    command: str,
    execution: str,
    location: str,
    arg_names: list[str],
):
    """Build an async MCP tool callable that runs a shell/ROS command template.

    The returned function fills `{placeholder}` values in `command` from the
    arguments it receives at call time (or appends them verbatim if the
    command has no placeholders), optionally `cd`s into `location` first,
    and executes the result as a shell command - directly for "shell", or
    after sourcing the corresponding ROS 1/ROS 2 setup script for
    "ros"/"ros2" execution types.

    Args:
        name: Tool name (informational only; not used in the function body).
        description: Tool description (informational only; not used in the
            function body).
        command: Command template, possibly containing `{arg_name}`
            placeholders.
        execution: Execution mode - "shell", "ros", or "ros2".
        location: Optional working directory to `cd` into before running
            the command.
        arg_names: Names of the placeholders present in `command`, in
            order.

    Returns:
        An async callable `tool_impl(args) -> str` suitable for registration
        as an MCP tool.
    """
    has_placeholders = bool(arg_names)

    async def tool_impl(args: Any = None) -> str:
        """Run the templated command with `args` substituted in and return its output."""
        try:
            cmd = command

            # Handle args - can be dict (from MCP) or string or None
            if args is None:
                parsed = []
            elif isinstance(args, dict):
                # MCP passes args as {"arg_name": value, ...} or {"args": "values..."}
                if "args" in args:
                    # Some MCP clients pass {"args": "value1 value2..."}
                    args_str = args["args"]
                    parsed = args_str.split() if args_str else []
                else:
                    # Pass as {"velx": "0.5", ...}
                    parsed = [str(args.get(arg, "")) for arg in arg_names]
            else:
                # Direct string argument
                parsed = str(args).split() if args else []

            # If command has placeholders, parse args to fill them
            if has_placeholders:
                for i, arg in enumerate(arg_names):
                    if i < len(parsed):
                        cmd = cmd.replace(f"{{{arg}}}", parsed[i])
            else:
                # No placeholders - just execute the command as-is
                if parsed:
                    cmd = f"{cmd} {' '.join(parsed)}"

            # Add location if specified
            if location:
                cmd = f"cd {location} && {cmd}"

            timeout = 10  # Reduced timeout for faster feedback

            if execution == "shell":
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=timeout
                )
                output = result.stdout.strip()
                if not output:
                    output = result.stderr.strip()
                if not output:
                    # For commands that might have succeeded silently, check return code
                    if result.returncode == 0:
                        output = f"Command executed successfully (exit code: {result.returncode})"
                    else:
                        output = f"Command failed with exit code: {result.returncode}"
                return output

            elif execution in ["ros", "ros2"]:
                ros_distro = "noetic" if execution == "ros" else "humble"
                ros_setup = f"/opt/ros/{ros_distro}/setup.bash"

                if os.path.exists(ros_setup):
                    full_cmd = f"bash -c 'source {ros_setup} && {cmd}'"
                else:
                    full_cmd = cmd

                result = subprocess.run(
                    full_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                output = result.stdout.strip()
                if not output:
                    output = result.stderr.strip()
                if not output:
                    # For commands that might have succeeded silently, check return code
                    if result.returncode == 0:
                        output = f"Command executed successfully (exit code: {result.returncode})"
                    else:
                        output = f"Command failed with exit code: {result.returncode}"
                return output

            else:
                return f"Unsupported execution type: {execution}"

        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    return tool_impl

def sanitize_tool_name(name: str) -> str:
    """Sanitize a function name into a valid MCP tool identifier.

    Args:
        name: The raw function name.

    Returns:
        A lowercase name containing only letters, digits, underscores,
        hyphens, and dots, with no leading/trailing or repeated
        underscores.
    """
    # Replace invalid characters with underscore
    sanitized = re.sub(r'[^A-Za-z0-9_\-\.]', '_', name)
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized.lower()

def create_server(config_path: str) -> FastMCP:
    """Build a FastMCP server whose tools are generated from a JSON config file.

    Args:
        config_path: Path to a JSON file containing a list of function
            definitions (each with name/description/command/execution/
            location).

    Returns:
        A FastMCP server instance with one tool registered per successfully
        parsed function definition; definitions that fail to register are
        skipped with a logged warning.
    """
    server_name = get_server_name(config_path)
    mcp = FastMCP(server_name)

    functions = load_functions(config_path)

    if not functions:
        # logger.warning(f"No functions in {config_path}")
        print(f"No functions in {config_path}", file=sys.stderr)
        return mcp

    # logger.info(f"Creating server '{server_name}' with {len(functions)} functions")
    print(
        f"Creating server '{server_name}' with {len(functions)} functions",
        file=sys.stderr,
    )

    for func_def in functions:
        func_info = func_def.get("function", func_def)
        name = func_info.get("name", "unknown")
        description = func_info.get("description", "")
        command = func_info.get("command", "")
        execution = func_info.get("execution", "shell")
        location = func_info.get("location", "")

        try:
            arg_names = get_args_from_command(command)
            tool_func = create_tool_function(
                name, description, command, execution, location, arg_names
            )

            # ✅ Sanitize tool name
            tool_name = sanitize_tool_name(name)

            # Avisa se o nome foi alterado
            if tool_name != name.lower().replace(" ", "_"):
                print(
                    f"⚠️  Tool name sanitized: '{name}' → '{tool_name}'",
                    file=sys.stderr
                )

            # Create tool using FastMCP's tool decorator properly
            decorated_tool = mcp.tool(name=tool_name, description=description)(
                tool_func
            )

            print(f"  ✓ {name}", file=sys.stderr)

        except Exception as e:
            print(f"  ✗ {name}: {e}", file=sys.stderr)

    return mcp


def main():
    """CLI entry point: load a function-definitions JSON file (from argv or a default path) and run the MCP server over stdio."""
    if len(sys.argv) < 2:
        config_dir = Path(__file__).resolve().parents[3] / "config" / "functions"
        if config_dir.exists():
            json_files = list(config_dir.glob("*_functions.json"))
            if json_files:
                config_path = str(json_files[0])
                print(f"Using: {config_path}")
            else:
                print("No JSON files found")
                sys.exit(1)
        else:
            print("Usage: python mcp_server_generic.py <config.json>")
            sys.exit(1)
    else:
        config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print(f"File not found: {config_path}")
        sys.exit(1)

    mcp = create_server(config_path)

    print(f"\nStarting server: {get_server_name(config_path)}", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
