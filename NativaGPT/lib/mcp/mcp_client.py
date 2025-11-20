import asyncio
import json
import os
import sys
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from bytez import Bytez
from dotenv import load_dotenv

load_dotenv()

BYTEZ_KEY = os.getenv("BYTEZ_API_KEY") or os.getenv("BYTEZ_KEY") or "c5767be58c1ff8b1dde99ff333f6c4b3"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")


class MCPClient:
    def __init__(self):
        # Store multiple server sessions
        self.servers: List[Dict[str, Any]] = []
        self.exit_stack = AsyncExitStack()

        # Bytez setup (exactly like the working example)
        self.sdk = Bytez(BYTEZ_KEY)
        if OPENAI_KEY:
            self.model = self.sdk.model("openai/gpt-4o", OPENAI_KEY)
        else:
            self.model = self.sdk.model("openai/gpt-4o")

    async def connect_to_server(self, server_script_list: List[str]):
        """Connect to multiple MCP servers (python or node)."""

        for server_script_path in server_script_list:

            # Validate file exists
            if not os.path.exists(server_script_path):
                print(f"[ERROR] Server script not found: {server_script_path}")
                continue

            is_python = server_script_path.endswith(".py")
            is_js = server_script_path.endswith(".js")
            if not (is_python or is_js):
                print(f"Skipping invalid server script: {server_script_path}")
                continue

            print(f"[INFO] Connecting to server: {server_script_path}")
            try:
                command = "python" if is_python else "node"

                # Get current environment (includes ROS if sourced)
                env = os.environ.copy()

                server_params = StdioServerParameters(
                    command=command,
                    args=[server_script_path],
                    env=env  # Changed from env=None - pass the environment!
                )

                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                stdio, write = stdio_transport

                session = await self.exit_stack.enter_async_context(
                    ClientSession(stdio, write)
                )
                await session.initialize()

                # List tools for this server
                response = await session.list_tools()
                tools = response.tools

                # Store server info
                server_name = os.path.basename(server_script_path)
                self.servers.append({
                    "name": server_name,
                    "session": session,
                    "stdio": stdio,
                    "write": write,
                    "tools": tools
                })

                print(f"\nConnected to '{server_name}' with tools: {[tool.name for tool in tools]}")

            except Exception as e:
                print(f"Failed to connect to {server_script_path}: {e}")
                continue

        if not self.servers:
            raise RuntimeError("No MCP servers could be connected")

        total_tools = sum(len(server["tools"]) for server in self.servers)
        print(f"\nTotal servers: {len(self.servers)}, Total tools: {total_tools}")

    def _get_all_tools(self) -> List[Any]:
        """Get flattened list of all tools from all servers."""
        all_tools = []
        for server in self.servers:
            all_tools.extend(server["tools"])
        return all_tools

    def _find_tool_server(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Find which server has the specified tool."""
        for server in self.servers:
            for tool in server["tools"]:
                if tool.name == tool_name:
                    return server
        return None

    def _run_model(self, messages: List[Dict[str, str]]) -> str:
        """
        Call Bytez model with given messages and return plain text.
        Handles both (output, error) and result.output/result.error styles.
        """
        result = self.model.run(messages)

        # Old style: (output, error)
        if isinstance(result, tuple) and len(result) == 2:
            output, error = result
        else:
            # New style: object with .output / .error
            output = getattr(result, "output", result)
            error = getattr(result, "error", None)

        if error:
            return f"Model error: {error}"

        # Normalise to string
        if isinstance(output, dict) and "content" in output:
            return str(output["content"])
        return str(output)

    @staticmethod
    def _extract_tool_text(result: Any) -> str:
        """Turn an MCP call_tool result into readable text."""
        try:
            if hasattr(result, "content") and result.content:
                parts = []
                for item in result.content:
                    if getattr(item, "type", None) == "text" and getattr(item, "text", None):
                        parts.append(item.text)
                    elif hasattr(item, "text") and item.text:
                        parts.append(item.text)
                if parts:
                    return "\n".join(parts)
            return str(result)
        except Exception:
            return "No result"

    def _build_tool_spec_prompt(self) -> str:
        """Build a JSON description of all tools from all servers for the LLM."""
        tools_info = []
        all_tools = self._get_all_tools()

        for tool in all_tools:
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            })

        tools_json = json.dumps(tools_info, indent=2)

        system_prompt = f"""
You are a tool-using assistant. You have access to the following tools,
described as a JSON array of objects (name, description, input_schema):

{tools_json}

For each user query, you must decide ONE of two actions:

1) Answer directly with your own knowledge:
   Respond EXACTLY as a single JSON object:
   {{
     "action": "final_answer",
     "answer": "<your answer in natural language>"
   }}

2) Call one of the tools above:
   Respond EXACTLY as a single JSON object:
   {{
     "action": "call_tool",
     "tool": "<tool name>",
     "args": {{ ... arguments matching the input_schema ... }}
   }}

Rules:
- The JSON must be valid. No trailing commas.
- Do NOT include any extra keys.
- Do NOT add any text before or after the JSON.
- 'tool' must be one of the tool names listed above.
"""
        return system_prompt.strip()

    async def process_query(self, query: str) -> str:
        """
        Let the LLM decide whether to call MCP tools from any connected server.
        """
        if not self.servers:
            return "Error: No MCP servers connected."

        # 1) PLAN: ask LLM if it wants to call a tool
        system_prompt = self._build_tool_spec_prompt()
        plan_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        plan_raw = self._run_model(plan_messages)

        # Try to parse JSON plan
        try:
            plan = self._parse_plan_json(plan_raw)
        except json.JSONDecodeError:
            return plan_raw

        action = plan.get("action")

        # Get list of all valid tool names
        all_tool_names = [tool.name for tool in self._get_all_tools()]

        # Handle case where model puts tool name as action (common mistake)
        if action not in ["final_answer", "call_tool"]:
            # Check if action is actually a tool name
            if action in all_tool_names:
                # Model put tool name as action, fix it
                plan = {
                    "action": "call_tool",
                    "tool": action,
                    "args": plan.get("args", {})
                }
                action = "call_tool"
                print(f"[DEBUG] Fixed plan: model used tool name '{plan['tool']}' as action")
            # Check if there's a "tool" field - if so, it's trying to call a tool
            elif "tool" in plan or "args" in plan:
                plan = {
                    "action": "call_tool",
                    "tool": plan.get("tool", action),
                    "args": plan.get("args", {})
                }
                action = "call_tool"
            else:
                # Unknown format, return as-is
                return f"Unexpected plan from model: {json.dumps(plan, indent=2)}"

        # 2A) No tool: final answer directly
        if action == "final_answer":
            answer = plan.get("answer", "")
            return answer or "(no answer provided)"

        # 2B) Tool call
        if action == "call_tool":
            tool_name = plan.get("tool")
            args = plan.get("args", {})

            if not tool_name:
                return f"Tool plan missing 'tool' field: {plan_raw}"

            # Find which server has this tool
            server = self._find_tool_server(tool_name)
            if not server:
                return f"Tool '{tool_name}' not found in any connected server"

            # Call MCP tool on the appropriate server
            try:
                print(f"[INFO] Calling tool: {tool_name} with args: {args}")
                result = await server["session"].call_tool(tool_name, args)
                tool_output = self._extract_tool_text(result)
                print(f"[INFO] Tool completed successfully")
            except Exception as e:
                import traceback
                print(f"[ERROR] Tool execution failed:\n{traceback.format_exc()}")
                return f"Error calling tool '{tool_name}' on server '{server['name']}': {e}"

            # 3) Ask LLM to summarize tool output
            answer_system_prompt = """
    You are a helpful assistant. The user asked a question, and an external tool
    was called to get relevant data. Your job is to explain the tool output in
    clear, concise natural language for the user.
    """
            answer_messages = [
                {"role": "system", "content": answer_system_prompt.strip()},
                {
                    "role": "user",
                    "content": (
                        f"User question:\n{query}\n\n"
                        f"Tool used: {tool_name}\n"
                        f"Tool arguments: {json.dumps(args)}\n\n"
                        f"Tool raw output:\n{tool_output}\n\n"
                        "Please summarize this for the user."
                    ),
                },
            ]
            final_answer = self._run_model(answer_messages)
            return final_answer

        # 2C) Unknown action
        return f"Unexpected plan from model: {plan_raw}"

    @staticmethod
    def _parse_plan_json(plan_raw: str):
        """Parse the model's plan as JSON, handling code blocks."""
        plan_raw = plan_raw.strip()

        # Remove code block fences
        if plan_raw.startswith("```"):
            lines = plan_raw.splitlines()
            lines = [ln for ln in lines if not ln.strip().startswith("```")]
            plan_raw = "\n".join(lines).strip()

        # Extract first {...} block
        start = plan_raw.find("{")
        end = plan_raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = plan_raw[start : end + 1]
        else:
            json_str = plan_raw

        return json.loads(json_str)

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <server_script1> [server_script2] ...")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1:])

        # Test query
        response = await client.process_query("What's the weather in Florida?")
        print(f"\nResponse: {response}")

    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())