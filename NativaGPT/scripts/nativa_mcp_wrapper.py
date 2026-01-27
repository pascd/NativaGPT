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
    def __init__(self, config_path: str = "config/config.yaml", system_context: str = ""):
        # 1. Load the manager and immediately extract the data dictionary
        # This fixes the TypeError because self.config becomes a dict, not the manager object.
        manager = ConfigManager(config_path)
        self.config = manager.get()

        # 2. Now self.config is a dict, so .get(key, default) works as expected
        self.server_configs = self.config.get("mcp_servers", [])

        # 3. Initialize the Nativa LLM Handler (it will now receive a dict)
        self.llm_handler = LLMPromptHandler(config=self.config)

        # 4. HMI Customization State
        self.system_context = system_context
        self.custom_local_tools: Dict[str, Callable] = {}

        # 5. MCP State
        self.exit_stack = AsyncExitStack()
        self.sessions: List[ClientSession] = []
        self._is_initialized = False

    def set_system_prompt(self, context: str):
        """Update the HMI persona or rules at runtime."""
        self.system_context = context

    def register_local_tool(self, name: str, func: Callable):
        """Register a standard Python function to be visible to the LLM."""
        self.custom_local_tools[name] = func

    async def _initialize_mcp(self):
        if self._is_initialized: return

        # Ensure server_configs is iterable
        if not self.server_configs:
            return

        for server in self.server_configs:
            path = server.get("path")
            # Avoid passing extra arguments to .get() if it's strictly 1 arg
            cmd = server.get("command") if server.get("command") else ("python" if path.endswith(".py") else "node")

            params = StdioServerParameters(
                command=cmd,
                args=[path],
                env={**os.environ, **server.get("env", {})}
            )

            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(params))
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

            await session.initialize()
            self.sessions.append(session)

        self._is_initialized = True

    def ask(self, query: str) -> str:
        """HMI Entry point: Blocks until response is ready."""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._process(query))

    async def _process(self, query: str) -> str:
        await self._initialize_mcp()

        # Gather MCP tools + Local Tools
        all_tools = []
        for session in self.sessions:
            resp = await session.list_tools()
            all_tools.extend([{"name": t.name, "desc": t.description, "source": "mcp"} for t in resp.tools])

        for name, func in self.custom_local_tools.items():
            all_tools.append({"name": name, "desc": func.__doc__ or "Local HMI function", "source": "local"})

        # Build Dynamic Prompt
        prompt = (
            f"SYSTEM: {self.system_context}\n"
            f"INSTRUCTIONS: Respond in JSON format with 'action', 'tool', and 'args' OR 'answer'.\n"
            f"AVAILABLE_TOOLS: {json.dumps(all_tools)}\n"
            f"USER_QUERY: {query}"
        )

        response = self.llm_handler.send_to_llm(prompt)
        return await self._handle_logic(response.get("text_content", "{}"), query)

    async def _handle_logic(self, content: str, original_query: str) -> str:
        if not content or not content.strip():
            return "Peço desculpa, não consegui processar essa informação."

        cleaned_content = self._clean_json(content)

        try:
            # Tenta tratar como comando (JSON)
            data = json.loads(cleaned_content)

            if isinstance(data, dict) and "action" in data and data["action"] == "call_tool":
                tool_name = data["tool"]
                args = data.get("args", {})

                if tool_name in self.custom_local_tools:
                    result = self.custom_local_tools[tool_name](**args)
                else:
                    result = await self._execute_mcp_tool(tool_name, args)

                # Segunda passagem para naturalizar a resposta da ferramenta
                summary_prompt = f"User: {original_query}\nTool Result: {result}\nTask: Provide a concise HMI update."
                final = self.llm_handler.send_to_llm(summary_prompt)

                # Retorna apenas o conteúdo de texto da resposta final
                return final.get("text_content", str(result)).strip()

            # Se for JSON mas tiver a chave 'answer'
            return data.get("answer", content).strip()

        except (json.JSONDecodeError, TypeError):
            # Se não for JSON, o modelo provavelmente respondeu em texto direto.
            # Retornamos o conteúdo limpo de tags markdown.
            return cleaned_content.strip()

    async def _execute_mcp_tool(self, tool_name: str, args: Dict):
        for session in self.sessions:
            tools = await session.list_tools()
            if any(t.name == tool_name for t in tools.tools):
                return await session.call_tool(tool_name, args)
        return f"Tool '{tool_name}' not found in any MCP server."

    def _clean_json(self, text: str) -> str:
        """Remove blocos de código e espaços desnecessários."""
        # Remove tags de markdown
        text = text.replace("```json", "").replace("```", "").strip()

        # Se houver texto explicativo antes/depois do JSON, tentamos isolar o {}
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and start < end:
            return text[start:end+1]

        return text

    def shutdown(self):
        if self._is_initialized:
            asyncio.run(self.exit_stack.aclose())