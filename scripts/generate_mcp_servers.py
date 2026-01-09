#!/usr/bin/env python3
"""
Generate MCP server launchers for all JSON function files.
"""

import os
import json
from pathlib import Path

FUNCTIONS_DIR = Path("/home/pedro/Documents/git-repos/NativaGPT/config/functions")
OUTPUT_DIR = Path("/home/pedro/Documents/git-repos/NativaGPT/NativaGPT/lib/mcp/servers")


def get_server_name(config_path: Path) -> str:
    """Generate server name from config file."""
    return config_path.stem.replace("_functions", "").replace("_", "-")


def generate_launcher(config_path: Path, server_name: str) -> str:
    """Generate a launcher script for an MCP server."""
    return f'''#!/usr/bin/env python3
# Auto-generated launcher for {config_path.name}
import sys
sys.path.insert(0, "{str(config_path.parent.parent.parent)}")

from NativaGPT.lib.mcp.mcp_server_generic import create_server

if __name__ == "__main__":
    mcp = create_server("{str(config_path)}")
    mcp.run(transport='stdio')
'''


def main():
    """Generate launchers for all function files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = list(FUNCTIONS_DIR.glob("*_functions.json"))

    if not json_files:
        print("No function JSON files found")
        return

    print(f"Generating MCP server configurations for {len(json_files)} files:\n")

    for config_path in json_files:
        server_name = get_server_name(config_path)

        # Generate launcher script
        launcher_path = OUTPUT_DIR / f"launch_{server_name}.py"
        launcher_content = generate_launcher(config_path, server_name)

        with open(launcher_path, "w") as f:
            f.write(launcher_content)

        os.chmod(launcher_path, 0o755)

        # Count functions
        with open(config_path, "r") as f:
            data = json.load(f)
            func_count = len(data) if isinstance(data, list) else 0

        print(f"  ✓ {config_path.name}")
        print(f"    → Server: nativa-{server_name}")
        print(f"    → Functions: {func_count}")
        print(f"    → Launcher: {launcher_path}\n")

    # Generate example config update
    print("\n" + "=" * 60)
    print("Add these servers to your config/config_default.json:")
    print("=" * 60 + "\n")

    for config_path in json_files:
        server_name = get_server_name(config_path)
        print(f'''  "{server_name}_mcp": {{
    "host": "{OUTPUT_DIR / f"launch_{server_name}.py"}"
  }},''')


if __name__ == "__main__":
    main()
