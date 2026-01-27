import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Callable
from contextlib import AsyncExitStack

# MCP Imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# NativaGPT Imports
from NativaGPT.lib.handlers.llm_prompt_handler import LLMPromptHandler
from NativaGPT.lib.config_manager import ConfigManager

class NativaMCPWrapper:
    def __init__(self, config_path: str = "config/config_default.json", system_context: str = ""):
        manager = ConfigManager(config_path)
        self.config = manager.get()

        # 1. Configuration Access
        mcp_section = self.config.get("mcp", {})
        self.mcp_enabled = mcp_section.get("enabled", True)
        self.server_configs = mcp_section.get("mcp_servers", {})

        # 2. LLM Handler & Context Setup
        self.llm_handler = LLMPromptHandler(config=self.config)

        # Load default setup prompt from config if none provided
        config_prompt = self.config.get("llm_config", {}).get("model_config", {}).get("setup_prompt", "")
        self.system_context = system_context or config_prompt

        # HMI Dynamic Context (e.g., "Temperature: 45C", "Mode: Auto")
        self.hmi_status_context: Dict[str, str] = {}
        self.custom_local_tools: Dict[str, Callable] = {}

        self.exit_stack = AsyncExitStack()
        self.sessions: List[ClientSession] = []
        self._is_initialized = False

    def update_hmi_status(self, key: str, value: str):
        """Add live machine data that the LLM should know about."""
        self.hmi_status_context[key] = value

    def set_system_prompt(self, context: str):
        self.system_context = context

    def register_local_tool(self, name: str, func: Callable):
        """Register a Python function for the LLM to call locally."""
        self.custom_local_tools[name] = func

    async def _initialize_mcp(self):
        if self._is_initialized or not self.mcp_enabled:
            return

        for name, server_data in self.server_configs.items():
            path = server_data.get("host") or server_data.get("path")
            if not path: continue

            cmd = server_data.get("command") or ("python" if path.endswith(".py") else "node")

            params = StdioServerParameters(
                command=cmd,
                args=[path],
                env={**os.environ, **server_data.get("env", {})}
            )

            try:
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
                stdio, write = stdio_transport
                session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                await session.initialize()
                self.sessions.append(session)
                print(f"[Info] Connected to MCP server: {name}")
            except Exception as e:
                print(f"[Error] Failed to connect to {name}: {e}")

        self._is_initialized = True

    def ask(self, query: str) -> str:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._process(query))

    async def _process(self, query: str) -> str:
        await self._initialize_mcp()

        # Build tools list
        all_tools = []
        for session in self.sessions:
            resp = await session.list_tools()
            all_tools.extend([{"name": t.name, "desc": t.description} for t in resp.tools])
        for name, func in self.custom_local_tools.items():
            all_tools.append({"name": name, "desc": func.__doc__ or "Local tool"})

        # Build enhanced prompt with HMI Status
        status_str = "\n".join([f"- {k}: {v}" for k, v in self.hmi_status_context.items()])

        prompt = (
            f"SYSTEM ROLE: {self.system_context}\n\n"
            f"CURRENT HMI STATUS:\n{status_str}\n\n"
            f"AVAILABLE TOOLS: {json.dumps(all_tools)}\n\n"
            "INSTRUCTIONS:\n"
            "1. If you need to use a tool, respond ONLY with JSON: "
            "{\"action\": \"call_tool\", \"tool\": \"NAME\", \"args\": {}}\n"
            "2. If you can answer directly, respond with: "
            "{\"answer\": \"YOUR MESSAGE\"}\n\n"
            f"USER QUERY: {query}"
        )

        response = self.llm_handler.send_to_llm(prompt)
        return await self._handle_logic(response.get("text_content", "{}"), query)

    async def _handle_logic(self, content: str, original_query: str) -> str:
        if not content.strip(): return "No response from system."

        cleaned = self._clean_json(content)
        try:
            data = json.loads(cleaned)

            # IMPROVED DETECTION: Check for 'action' OR just presence of 'tool'
            is_tool = isinstance(data, dict) and (
                data.get("action") == "call_tool" or ("tool" in data and "args" in data)
            )

            if is_tool:
                tool_name = data["tool"]
                args = data.get("args", {})

                print(f"[Executing] {tool_name} with {args}")

                if tool_name in self.custom_local_tools:
                    result = self.custom_local_tools[tool_name](**args)
                else:
                    result = await self._execute_mcp_tool(tool_name, args)

                # Natural language summary pass
                summary_prompt = (
                    f"User asked: {original_query}\n"
                    f"Tool execution result: {result}\n"
                    "Task: Provide a natural language summary for the HMI display."
                )
                final = self.llm_handler.send_to_llm(summary_prompt)
                return final.get("text_content", str(result)).strip()

            return data.get("answer", content).strip()

        except (json.JSONDecodeError, KeyError):
            return cleaned.strip()

    async def _execute_mcp_tool(self, tool_name: str, args: Dict):
        for session in self.sessions:
            tools = await session.list_tools()
            if any(t.name == tool_name for t in tools.tools):
                result = await session.call_tool(tool_name, args)
                # Extract text from MCP Tool Result
                if hasattr(result, 'content'):
                    return " ".join([c.text for c in result.content if hasattr(c, 'text')])
                return str(result)
        return f"Tool '{tool_name}' not found."

    def _clean_json(self, text: str) -> str:
        text = text.replace("```json", "").replace("```", "").strip()
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end+1]
        return text

    def shutdown(self):
        if self._is_initialized:
            asyncio.run(self.exit_stack.aclose())