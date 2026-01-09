#!/usr/bin/env python3
# Auto-generated launcher for native_functions.json
import sys
sys.path.insert(0, "/home/pedro/Documents/git-repos/NativaGPT")

from NativaGPT.lib.mcp.mcp_server_generic import create_server

if __name__ == "__main__":
    mcp = create_server("/home/pedro/Documents/git-repos/NativaGPT/config/functions/native_functions.json")
    mcp.run(transport='stdio')
