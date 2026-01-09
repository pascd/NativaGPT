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
from NativaGPT.lib.coloring_logger import logger


def get_server_name(config_path: str) -> str:
    """Extract server name from config file path."""
    path = Path(config_path)
    name = path.stem.replace("_functions", "").replace("_", "-")
    return f"nativa-{name}" if name else "nativa-functions"


def load_functions(config_path: str) -> list[dict]:
    """Load function definitions from JSON file."""
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        logger.error(f"Error loading {config_path}: {e}")
        return []


def get_args_from_command(command: str) -> list[str]:
    """Extract argument names from command placeholders like {topic_name}."""
    return re.findall(r"\{(\w+)\}", command)


def create_tool_function(
    name: str,
    description: str,
    command: str,
    execution: str,
    location: str,
    arg_names: list[str],
):
    """Create an async tool function."""
    has_placeholders = bool(arg_names)

    async def tool_impl(args: str = "") -> str:
        """Execute the command and return output."""
        try:
            cmd = command

            # If command has placeholders, parse args to fill them
            if has_placeholders:
                parsed = args.split() if args else []
                for i, arg in enumerate(arg_names):
                    if i < len(parsed):
                        cmd = cmd.replace(f"{{{arg}}}", parsed[i])
            else:
                # No placeholders - just execute the command as-is
                if args:
                    cmd = f"{cmd} {args}"

            # Add location if specified
            if location:
                cmd = f"cd {location} && {cmd}"

            timeout = 30

            if execution == "shell":
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=timeout
                )
                output = (
                    result.stdout.strip()
                    if result.stdout.strip()
                    else result.stderr.strip()
                )
                return output if output else "Command executed (no output)"

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
                output = (
                    result.stdout.strip()
                    if result.stdout.strip()
                    else result.stderr.strip()
                )
                return output if output else "Command executed (no output)"

            else:
                return f"Unsupported execution type: {execution}"

        except subprocess.TimeoutExpired:
            return "Error: Command timed out"
        except Exception as e:
            return f"Error: {str(e)}"

    return tool_impl


def create_server(config_path: str) -> FastMCP:
    """Create MCP server from JSON config."""
    server_name = get_server_name(config_path)
    mcp = FastMCP(server_name)

    functions = load_functions(config_path)

    if not functions:
        logger.warning(f"No functions in {config_path}")
        return mcp

    logger.info(f"Creating server '{server_name}' with {len(functions)} functions")

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

            # Sanitize tool name for MCP
            tool_name = name.lower().replace(" ", "_").replace("-", "_")

            # Create tool using FastMCP's tool decorator properly
            # The function name must match what we want the tool to be called
            decorated_tool = mcp.tool(name=tool_name, description=description)(
                tool_func
            )

            logger.info(f"  ✓ {name}")

        except Exception as e:
            logger.error(f"  ✗ {name}: {e}")

    return mcp


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        config_dir = Path("/home/pedro/Documents/git-repos/NativaGPT/config/functions")
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

    print(f"\nStarting server: {get_server_name(config_path)}")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
