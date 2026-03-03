"""
MCP Client v3.0 - Model Context Protocol Client with VLM Support

Features:
- Multi-server support
- Automatic VLM invocation for image analysis
- Better error handling and self-healing
- Unified LLM/VLM backend interface
"""

import asyncio
import json
import os
import sys
import re
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
from NativaGPT.lib.handlers.llm_prompt_handler import LLMPromptHandler
from NativaGPT.lib.coloring_logger import logger

load_dotenv()


class MCPClient:
    """
    MCP Client v3.0 - Model Context Protocol Client with VLM Support
    """

    def __init__(self, llm_handler: Optional[LLMPromptHandler] = None):
        self.servers: List[Dict[str, Any]] = []
        self.exit_stack = AsyncExitStack()
        self.llm_handler = llm_handler
        logger.info("MCP Client v3.0 initialized")

    async def connect_to_server(self, server_script_list: List[str]):
        """Connect to multiple MCP servers."""
        for server_script_path in server_script_list:
            if not os.path.exists(server_script_path):
                logger.error(f"Server script not found: {server_script_path}")
                continue

            is_python = server_script_path.endswith(".py")
            is_js = server_script_path.endswith(".js")
            is_bash = server_script_path.endswith(".sh")

            if not (is_python or is_js or is_bash):
                logger.warning(f"Skipping invalid server script: {server_script_path}")
                continue

            logger.info(f"Connecting to server: {server_script_path}")
            try:
                command = "bash" if is_bash else ("node" if is_js else sys.executable)

                env = os.environ.copy()
                ros_vars = ["ROS_DISTRO", "ROS_ROOT", "ROS_MASTER_URI", "PYTHONPATH"]
                missing = [v for v in ros_vars[:2] if v not in env]
                if missing:
                    logger.debug(f"Missing ROS env vars: {missing}")

                server_params = StdioServerParameters(
                    command=command,
                    args=[server_script_path],
                    env=env,
                )

                stdio_transport = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                stdio, write = stdio_transport

                session = await self.exit_stack.enter_async_context(
                    ClientSession(stdio, write)
                )
                await session.initialize()

                response = await session.list_tools()
                tools = response.tools

                server_name = os.path.basename(server_script_path)
                self.servers.append(
                    {
                        "name": server_name,
                        "session": session,
                        "stdio": stdio,
                        "write": write,
                        "tools": tools,
                    }
                )

                logger.info(f"✓ Connected to '{server_name}' with {len(tools)} tools")
                for tool in tools:
                    desc = tool.description[:50] if tool.description else ""
                    logger.debug(f"  - {tool.name}: {desc}...")

            except Exception as e:
                logger.error(f"Failed to connect to {server_script_path}: {e}")
                continue

        if not self.servers:
            raise RuntimeError("No MCP servers could be connected")

        total_tools = sum(len(s["tools"]) for s in self.servers)
        logger.info(f"Total: {len(self.servers)} servers, {total_tools} tools")

    def _get_all_tools(self) -> List[Any]:
        all_tools = []
        for server in self.servers:
            all_tools.extend(server["tools"])
        return all_tools

    def _find_tool_server(self, tool_name: str) -> Optional[Dict[str, Any]]:
        for server in self.servers:
            for tool in server["tools"]:
                if tool.name == tool_name:
                    return server
        return None

    def _run_llm(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call LLM handler with optional images."""
        if not self.llm_handler:
            return "Error: LLM handler not initialized"

        try:
            response = self.llm_handler.send_to_llm(
                prompt,
                images=images,
                system_prompt=system_prompt,
            )

            if not response.get("success", False):
                return f"Model error: {response.get('error')}"

            return response.get("text_content", "") or "Empty response"

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"Error: {str(e)}"

    def _run_vlm(self, image_path: str, prompt: str) -> str:
        """Call VLM for image analysis."""
        if not self.llm_handler:
            return "Error: LLM handler not initialized"

        try:
            response = self.llm_handler.send_to_vlm(image_path, prompt)

            if not response.get("success", False):
                return f"VLM error: {response.get('error')}"

            return response.get("text_content", "") or "Empty VLM response"

        except Exception as e:
            logger.error(f"VLM error: {e}")
            return f"Error: {str(e)}"

    @staticmethod
    def _extract_tool_text(result: Any) -> str:
        """Extract text from MCP tool result."""
        try:
            if hasattr(result, "content") and result.content:
                parts = []
                for item in result.content:
                    if getattr(item, "type", None) == "text" and getattr(
                        item, "text", None
                    ):
                        parts.append(item.text)
                    elif hasattr(item, "text") and item.text:
                        parts.append(item.text)
                if parts:
                    return "\n".join(parts)
            return str(result)
        except Exception:
            return "No result"

    def _extract_images_from_tool_output(self, tool_output: str) -> List[str]:
        """Extract image paths from tool output."""
        images = []

        try:
            data = json.loads(tool_output)
            if isinstance(data, dict):
                if "image_path" in data:
                    path = data["image_path"]
                    if path and os.path.exists(path):
                        images.append(path)
                        logger.info(f"Found image: {path}")

                if "files" in data and isinstance(data["files"], list):
                    for f in data["files"]:
                        if f and os.path.exists(f) and self._is_image_file(f):
                            images.append(f)
                            logger.info(f"Found file: {f}")
        except json.JSONDecodeError:
            pattern = r"([/\w\-\.]+\.(?:jpg|jpeg|png|gif|bmp|webp))"
            matches = re.findall(pattern, tool_output, re.IGNORECASE)
            for match in matches:
                if os.path.exists(match):
                    images.append(match)
                    logger.info(f"Found image: {match}")

        return images

    @staticmethod
    def _is_image_file(path: str) -> bool:
        exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"}
        _, ext = os.path.splitext(path.lower())
        return ext in exts

    def _build_tool_spec_prompt(self) -> str:
        """Build tool specification prompt for LLM."""
        tools_info = []
        all_tools = self._get_all_tools()

        for tool in all_tools:
            tools_info.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema,
                }
            )

        tools_json = json.dumps(tools_info, indent=2)

        return f"""You are a tool-using assistant with access to tools.

Available tools:
{tools_json}

RESPOND WITH ONLY ONE OF THESE JSON FORMATS:

1) Direct answer (no tool needed):
{{"action": "final_answer", "answer": "your answer here"}}

2) Call a tool:
{{"action": "call_tool", "tool": "exact_tool_name", "args": {{...}}}}

RULES:
- Output ONLY valid JSON (no other text)
- Tool name must match exactly from the list
- For ROS topics, use full names like "/camera/image_raw"
- When handling images, describe what you see directly in your answer
"""

    async def process_query(self, query: str, max_iterations: int = 3) -> str:
        """
        Process user query with tool calling and VLM integration.
        """
        if not self.servers:
            return "Error: No MCP servers connected"

        if not self.llm_handler:
            return "Error: LLM handler not initialized"

        original_query = query
        last_error = ""

        for iteration in range(1, max_iterations + 1):
            logger.info(f"Query iteration {iteration}/{max_iterations}")

            system_prompt = self._build_tool_spec_prompt()
            plan_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

            plan_raw = self._run_llm(plan_messages)
            logger.debug(f"LLM response: {plan_raw[:200]}...")

            plan = self._extract_json_plan(plan_raw)

            if plan is None:
                if iteration == 1:
                    return plan_raw
                query = self._build_correction_query(
                    original_query, "Invalid JSON", plan_raw, {}
                )
                continue

            action = plan.get("action")
            all_tool_names = [t.name for t in self._get_all_tools()]

            if action not in ["final_answer", "call_tool"]:
                if action in all_tool_names:
                    plan = {
                        "action": "call_tool",
                        "tool": action,
                        "args": plan.get("args", {}),
                    }
                    action = "call_tool"
                elif "tool" in plan:
                    plan = {
                        "action": "call_tool",
                        "tool": plan.get("tool"),
                        "args": plan.get("args", {}),
                    }
                    action = "call_tool"
                else:
                    if iteration == 1:
                        return plan_raw
                    query = self._build_correction_query(
                        original_query, "Invalid action", plan_raw, {}
                    )
                    continue

            if action == "final_answer":
                answer = plan.get("answer", "")
                return answer or "(no answer)"

            if action == "call_tool":
                tool_name = plan.get("tool")
                args = plan.get("args", {})

                if not tool_name:
                    return f"Missing tool name in plan: {plan_raw}"

                server = self._find_tool_server(tool_name)
                if not server:
                    error_msg = f"Tool '{tool_name}' not found"
                    if iteration < max_iterations:
                        query = self._build_correction_query(
                            original_query, error_msg, "", {}
                        )
                        continue
                    return error_msg

                try:
                    logger.info(f"Calling tool: {tool_name} with args: {args}")
                    result = await server["session"].call_tool(tool_name, args)
                    tool_output = self._extract_tool_text(result)
                    logger.info(f"Tool output: {tool_output[:200]}...")

                    images = self._extract_images_from_tool_output(tool_output)

                    if images:
                        logger.info(f"[VLM] Analyzing {len(images)} image(s)")

                        vlm_prompt = f"User asked: {original_query}\n\nAnalyze this image and provide a detailed response."

                        image_path = images[0]
                        final_answer = self._run_vlm(image_path, vlm_prompt)

                        if (
                            "error" in final_answer.lower()
                            and "VLM" not in final_answer
                        ):
                            final_answer = self._run_llm(
                                f"User asked: {original_query}\n\nTool result: {tool_output}\n\nAn image was captured but analysis failed. Describe what happened.",
                                images=images,
                            )

                        return final_answer
                    else:
                        answer_prompt = f"""The user asked: {original_query}

Tool '{tool_name}' returned:
{tool_output}

Summarize this result clearly for the user. If there's an error, explain what went wrong."""

                        final_answer = self._run_llm(answer_prompt)
                        return final_answer

                except Exception as e:
                    error_msg = f"Tool error: {str(e)}"
                    logger.error(error_msg)

                    if iteration < max_iterations:
                        query = self._build_correction_query(
                            original_query, error_msg, "", {}
                        )
                        continue
                    return error_msg

        return f"Could not complete request after {max_iterations} attempts. Last error: {last_error}"

    def _extract_json_plan(self, text: str) -> Optional[Dict]:
        """Extract JSON plan from LLM response."""
        try:
            text = text.strip()

            if text.startswith("```"):
                lines = text.splitlines()
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()

            text = (
                text.replace('"', '"')
                .replace('"', '"')
                .replace("'", "'")
                .replace("'", "'")
            )

            start = text.find("{")
            end = text.rfind("}")

            if start != -1 and end != -1 and end > start:
                json_str = text[start : end + 1]
                return json.loads(json_str)

            return None

        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse error: {e}")
            return None

    def _build_correction_query(
        self, original: str, error: str, last_response: str, verification: Dict
    ) -> str:
        """Build correction prompt for retry."""
        return f"""Previous attempt failed.
Original request: {original}
Error: {error}
Previous response: {last_response[:500] if last_response else "None"}

Please try again with a corrected plan. Output ONLY valid JSON."""

    async def cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up MCP client...")
        await self.exit_stack.aclose()
        if self.llm_handler:
            self.llm_handler.cleanup()


async def main():
    """Test the MCP client."""
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <server_script> [server_script2 ...]")
        sys.exit(1)

    from NativaGPT.lib.config_manager import ConfigManager

    config_path = "config/config_default.json"
    config_manager = ConfigManager(config_path)
    config = config_manager.get()

    llm_handler = LLMPromptHandler(config)
    client = MCPClient(llm_handler)

    try:
        await client.connect_to_server(sys.argv[1:])

        queries = [
            "List available ROS topics",
            "Capture an image from the camera",
            "What's the weather?",
        ]

        for query in queries:
            print(f"\n{'=' * 60}")
            print(f"Query: {query}")
            print("=" * 60)
            response = await client.process_query(query)
            print(f"Response: {response}")

    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
