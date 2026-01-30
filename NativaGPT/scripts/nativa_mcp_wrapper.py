import asyncio
import json
import os
import sys
import glob
from typing import List, Dict, Any, Optional, Callable, Set
from contextlib import AsyncExitStack
from pathlib import Path

# MCP Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# NativaGPT Imports
from NativaGPT.lib.handlers.llm_prompt_handler import LLMPromptHandler
from NativaGPT.lib.config_manager import ConfigManager


class NativaMCPWrapper:
    def __init__(
        self,
        config_path: str = "config/config_default.json",
        additional_mcp_servers: Optional[Dict[str, Dict]] = None,
        additional_json_paths: Optional[List[str]] = None,
        max_history: int = 10,
        auto_initialize: bool = True,
    ):
        """
        Initialize Nativa MCP Wrapper with flexible configuration.

        Args:
            config_path: Path to config file (default: config/config_default.json)
            additional_mcp_servers: Extra MCP servers to add beyond config
            additional_json_paths: Extra function JSON paths to load beyond database_folder
            max_history: Maximum conversation exchanges to keep (default: 10)
            auto_initialize: Automatically initialize MCP servers on startup (default: True)
        """
        manager = ConfigManager(config_path)
        self.config = manager.get()

        # Store additional servers and JSON paths
        self.additional_mcp_servers = additional_mcp_servers or {}
        self.additional_json_paths = additional_json_paths or []

        # Conversation memory settings
        self.max_history_length = max_history or 10

        # 1. Path to your mcp_server_generic.py
        # Defaulting to the lib/mcp directory
        self.generic_server_script = os.path.join(
            os.path.dirname(__file__), "..", "lib", "mcp", "mcp_server_generic.py"
        )

        mcp_section = self.config.get("mcp", {})
        self.mcp_enabled = mcp_section.get("enabled", True)

        # Merge configured servers with additional servers
        configured_servers = mcp_section.get("mcp_servers", {})
        self.server_configs = {**configured_servers, **self.additional_mcp_servers}

        # Path where function JSONs are stored (from your config_default.json)
        self.functions_dir = self.config.get("nativa_gpt", {}).get(
            "database_folder", ""
        )

        self.llm_handler = LLMPromptHandler(config=self.config)
        self.system_context = (
            self.config.get("llm_config", {})
            .get("model_config", {})
            .get("setup_prompt", "")
        )

        self.hmi_status_context: Dict[str, str] = {}
        self.custom_local_tools: Dict[str, Callable] = {}
        self.exit_stack = AsyncExitStack()
        self.sessions: List[ClientSession] = []
        self._connected_server_names: Set[str] = set()
        self._is_initialized = False

        # Conversation memory for context retention
        self.conversation_history: List[Dict[str, str]] = []

        # Auto-initialize MCP servers
        if auto_initialize and self.mcp_enabled:
            print("[NativaMCP] Auto-initializing MCP servers...", file=sys.stderr)
            self._initialize_sync()
            print(
                f"[NativaMCP] ✓ Initialized {len(self.sessions)} MCP servers",
                file=sys.stderr,
            )

    def _initialize_sync(self):
        """Synchronously initialize MCP servers during __init__."""
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run initialization
            loop.run_until_complete(self._initialize_mcp())

        except Exception as e:
            print(
                f"[NativaMCP] Error during auto-initialization: {e}", file=sys.stderr
            )
            import traceback

            traceback.print_exc()

    async def _initialize_mcp(self):
        """Initializes both static servers and dynamic generic servers."""
        if self._is_initialized or not self.mcp_enabled:
            return

        print("[NativaMCP] Loading MCP servers...", file=sys.stderr)

        # 1. Load Static Servers (including additional ones)
        for name, data in self.server_configs.items():
            await self._connect_to_server(name, data)

        # 2. Collect JSON files from multiple sources
        all_json_files = []

        # 2a. From database_folder (original behavior)
        if os.path.exists(self.functions_dir):
            json_files = glob.glob(os.path.join(self.functions_dir, "*_functions.json"))
            all_json_files.extend(json_files)
            print(
                f"[NativaMCP] Found {len(json_files)} function files in database_folder",
                file=sys.stderr,
            )

        # 2b. From additional JSON paths
        for json_path in self.additional_json_paths:
            if os.path.exists(json_path):
                all_json_files.append(json_path)
            elif os.path.isdir(json_path):
                # If it's a directory, look for *_functions.json files
                dir_json_files = glob.glob(os.path.join(json_path, "*_functions.json"))
                all_json_files.extend(dir_json_files)
                print(
                    f"[NativaMCP] Found {len(dir_json_files)} function files in {json_path}",
                    file=sys.stderr,
                )

        # 3. Spawn Generic Servers for all JSON files
        for json_path in all_json_files:
            server_name = f"generic-{Path(json_path).stem}"
            generic_config = {
                "command": "python3",
                "path": self.generic_server_script,
                "args": [json_path],
            }
            await self._connect_to_server(server_name, generic_config)

        self._is_initialized = True
        print(
            f"[NativaMCP] Initialization complete - {len(self._connected_server_names)} servers active",
            file=sys.stderr,
        )

    async def _connect_to_server(self, name: str, server_data: Dict):
        """Helper to establish a connection to any MCP server."""
        if name in self._connected_server_names:
            return

        path = server_data.get("host") or server_data.get("path")
        if not path:
            return

        # Build execution arguments
        cmd = server_data.get("command") or (
            "python3" if path.endswith(".py") else "node"
        )
        # Combine script path with any extra args (like the JSON path)
        args = [path] + server_data.get("args", [])

        params = StdioServerParameters(
            command=cmd, args=args, env={**os.environ, **server_data.get("env", {})}
        )

        try:
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(params)
            )
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            await session.initialize()

            self.sessions.append(session)
            self._connected_server_names.add(name)
            print(f"[MCP] ✓ Started server: {name}", file=sys.stderr)
        except Exception as e:
            print(f"[MCP] ✗ Failed to start {name}: {e}", file=sys.stderr)

    def ask(self, query: str) -> Dict[str, Any]:
        """Entry point for HMI calls. Returns both tools and response."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._process(query))

    async def _process(self, query: str) -> Dict[str, Any]:
        await self._initialize_mcp()

        # Gather tools from all active MCP sessions
        all_tools = []
        for session in self.sessions:
            resp = await session.list_tools()
            for t in resp.tools:
                all_tools.append(
                    {
                        "name": t.name,
                        "desc": t.description,
                        "input_schema": t.inputSchema,
                    }
                )

        for name, func in self.custom_local_tools.items():
            all_tools.append({"name": name, "desc": func.__doc__ or "Local tool"})

        status_str = "\n".join(
            [f"- {k}: {v}" for k, v in self.hmi_status_context.items()]
        )

        # Get conversation context for memory
        conversation_context = self.get_conversation_context()

        # Structure prompt for tool use with conversation memory
        prompt = (
            f"SYSTEM ROLE: {self.system_context}\n\n"
            f"HMI CONTEXT:\n{status_str}\n\n"
            f"CONVERSATION HISTORY:\n{conversation_context}\n\n"
            f"AVAILABLE TOOLS:\n{json.dumps(all_tools, indent=2)}\n\n"
            "INSTRUCTIONS:\n"
            "1. Consider conversation context and previous interactions when responding\n"
            '2. To use a tool, reply ONLY with: {"action": "call_tool", "tool": "NAME", "args": {}}\n'
            '3. If answering the user directly, reply with: {"answer": "MESSAGE"}\n'
            "4. Reference previous tool usage and responses as needed for context\n\n"
            f"USER QUERY: {query}"
        )

        response = self.llm_handler.send_to_llm(prompt)
        processed_response = await self._handle_logic(
            response.get("text_content", "{}"), query, all_tools or []
        )

        # Save to conversation history
        self.add_to_history(query, processed_response)

        return processed_response

    async def _handle_logic(
        self, content: str, original_query: str, all_tools: List[Dict] = []
    ) -> Dict[str, Any]:
        cleaned = self._clean_json(content)
        tools_called = []

        try:
            data = json.loads(cleaned)
            is_tool = isinstance(data, dict) and ("tool" in data)

            if is_tool:
                tool_name = data["tool"]
                args = data.get("args", {})
                tools_called.append({"name": tool_name, "args": args})

                # Execution
                if tool_name in self.custom_local_tools:
                    result = self.custom_local_tools[tool_name](**args)
                else:
                    result = await self._execute_mcp_tool(tool_name, args)

                # Summarize result
                summary_prompt = (
                    f"User asked: {original_query}\n"
                    f"Tool called: {tool_name} with args {args}\n"
                    f"Tool result: {result}\n"
                    "Task: Provide a clear, concise answer to the user about what happened."
                )
                final = self.llm_handler.send_to_llm(summary_prompt)
                clean_answer = final.get("text_content", str(result)).strip()

                return {
                    "tools_called": tools_called,
                    "response": clean_answer,
                    "raw_result": result,
                }

            # Direct answer - no tools called
            direct_answer = data.get("answer", content).strip()
            return {
                "tools_called": tools_called,
                "response": direct_answer,
                "raw_result": None,
            }
        except Exception as e:
            return {
                "tools_called": tools_called,
                "response": f"Error processing response: {str(e)}",
                "raw_result": None,
            }

    async def _execute_mcp_tool(self, tool_name: str, args: Dict):
        for session in self.sessions:
            tools = await session.list_tools()
            if any(t.name == tool_name for t in tools.tools):
                result = await session.call_tool(tool_name, args)
                if hasattr(result, "content"):
                    return str(result.content)
                return str(result)
        return f"Tool '{tool_name}' not found."

    def _clean_json(self, text: str) -> str:
        text = text.replace("```json", "").replace("```", "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            return text[start : end + 1]
        return text

    def shutdown(self):
        """Clean shutdown of all MCP connections."""
        if self._is_initialized:
            asyncio.run(self.exit_stack.aclose())

    def add_mcp_server(self, name: str, server_config: Dict[str, Any]):
        """
        Dynamically add a new MCP server configuration.

        Args:
            name: Server name
            server_config: Server configuration dict with command, path, args, env, etc.

        Example:
            wrapper.add_mcp_server("weather", {
                "command": "python3",
                "path": "/path/to/weather_server.py",
                "args": ["--api-key", "key123"]
            })
        """
        self.additional_mcp_servers[name] = server_config
        self.server_configs[name] = server_config

    def add_function_json(self, json_path: str):
        """
        Dynamically add a new function JSON file to load.

        Args:
            json_path: Path to a JSON file or directory containing *_functions.json files

        Example:
            wrapper.add_function_json("/path/to/custom_functions.json")
            wrapper.add_function_json("/path/to/functions_directory/")
        """
        if json_path not in self.additional_json_paths:
            self.additional_json_paths.append(json_path)

    def get_loaded_tools_count(self) -> int:
        """Get count of currently loaded tools (requires initialization)."""
        if not self._is_initialized:
            return 0
        return len(self.sessions)

    def list_server_names(self) -> List[str]:
        """Get list of connected MCP server names."""
        return list(self._connected_server_names)

    def add_to_history(self, user_query: str, response: Dict[str, Any]):
        """Add conversation exchange to history."""
        self.conversation_history.append(
            {
                "user": user_query,
                "assistant": response.get("response", ""),
                "tools_called": response.get("tools_called", []),
                "timestamp": self._get_timestamp(),
            }
        )

        # Keep only recent exchanges
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[
                -self.max_history_length :
            ]

    def get_conversation_context(self) -> str:
        """Get formatted conversation history for prompt context."""
        if not self.conversation_history:
            return "No previous conversation."

        context_lines = ["Recent conversation history:"]
        for i, exchange in enumerate(self.conversation_history, 1):
            context_lines.append(f"{i}. User: {exchange['user']}")
            tools_called = exchange.get("tools_called", [])
            if tools_called:
                tools = ", ".join(
                    [
                        tool.get("name", "unknown")
                        if isinstance(tool, dict)
                        else str(tool)
                        for tool in tools_called
                    ]
                )
                context_lines.append(f"   Tools used: {tools}")
            assistant_response = exchange.get("assistant", "")
            context_lines.append(f"   Assistant: {assistant_response[:200]}...")
            context_lines.append("")

        return "\n".join(context_lines)

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("🗑️ Conversation history cleared.", file=sys.stderr)

    def get_history_length(self) -> int:
        """Get current conversation history length."""
        return len(self.conversation_history)

    def _get_timestamp(self) -> str:
        """Get current timestamp for history entries."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def usage_examples():
        """Show usage examples for the flexible NativaMCPWrapper."""
        examples = """
        === NATIVA MCP WRAPPER USAGE EXAMPLES ===

        1. Basic Usage (uses config_default.json):
           wrapper = NativaMCPWrapper()
           result = wrapper.ask("What tools are available?")

        2. Custom Config File:
           wrapper = NativaMCPWrapper("config/my_config.json")
           result = wrapper.ask("List available tools")

        3. Add Additional MCP Server:
           wrapper = NativaMCPWrapper(
               additional_mcp_servers={
                   "weather": {
                       "command": "python3",
                       "path": "/path/to/weather_server.py",
                       "args": ["--api-key", "key123"]
                   }
               }
           )

        4. Add Custom Function JSON:
           wrapper = NativaMCPWrapper(
               additional_json_paths=[
                   "/path/to/custom_functions.json",
                   "/path/to/more_functions/"
               ]
           )

        5. Mix Everything:
           wrapper = NativaMCPWrapper(
               config_path="config/production.json",
               additional_mcp_servers={
                   "weather": {...},
                   "database": {...}
               },
               additional_json_paths=[
                   "/opt/custom_tools/functions.json",
                   "/opt/robotic_tools/"
               ]
           )

        6. Dynamic Addition:
           wrapper = NativaMCPWrapper()
           wrapper.add_mcp_server("weather", {...})
           wrapper.add_function_json("/path/to/tools.json")

           print(f"Servers: {wrapper.list_server_names()}")
           print(f"Tools loaded: {wrapper.get_loaded_tools_count()}")

        7. Conversation Memory:
           wrapper = NativaMCPWrapper(max_history=10)

           result1 = wrapper.ask("Hi, I'm working on robotics project")
           result2 = wrapper.ask("What tools can help with my project?")

           print(f"History length: {wrapper.get_history_length()}")
           wrapper.clear_history()
        """
        print(examples, file=sys.stderr)