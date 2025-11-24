import asyncio
import json
import os
import sys
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
    Enhanced MCP Client v2.0

    Key features:
    - Multi-server support
    - Automatic image attachment from tool outputs
    - Better error handling
    - ROS command execution support
    """

    def __init__(self, llm_handler: Optional[LLMPromptHandler] = None):
        # Store multiple server sessions
        self.servers: List[Dict[str, Any]] = []
        self.exit_stack = AsyncExitStack()

        self.llm_handler = llm_handler
        logger.info("MCP Client initialized with NativaGPT LLM handler")

    async def connect_to_server(self, server_script_list: List[str]):
        """Connect to multiple MCP servers (python or node)."""

        for server_script_path in server_script_list:

            # Validate file exists
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
                # Determine command based on file type
                if is_bash:
                    command = "bash"
                elif is_python:
                    command = "python"
                else:  # is_js
                    command = "node"

                # CRITICAL: Pass full environment including ROS variables
                env = os.environ.copy()

                # Ensure key ROS variables are present
                ros_vars = ['ROS_DISTRO', 'ROS_ROOT', 'ROS_PACKAGE_PATH',
                           'ROS_MASTER_URI', 'PYTHONPATH', 'LD_LIBRARY_PATH',
                           'CMAKE_PREFIX_PATH', 'PKG_CONFIG_PATH', 'PATH']

                missing_ros = []
                for var in ['ROS_DISTRO', 'ROS_ROOT']:
                    if var not in env:
                        missing_ros.append(var)

                if missing_ros:
                    logger.warning(f"Missing ROS environment variables: {missing_ros}")
                    logger.warning("Did you source ROS? Run: source /opt/ros/noetic/setup.bash")

                server_params = StdioServerParameters(
                    command=command,
                    args=[server_script_path],
                    env=env  # Pass full environment to subprocess
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

                logger.info(f"✓ Connected to '{server_name}' with {len(tools)} tools:")
                for tool in tools:
                    logger.info(f"  - {tool.name}: {tool.description[:60]}...")

            except Exception as e:
                logger.error(f"Failed to connect to {server_script_path}: {e}")
                continue

        if not self.servers:
            raise RuntimeError("No MCP servers could be connected")

        total_tools = sum(len(server["tools"]) for server in self.servers)
        logger.info(f"\n✓ Total servers: {len(self.servers)}, Total tools: {total_tools}")

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

    def _run_model(self, messages: List[Dict[str, str]], images: List[str] = None) -> str:
        """
        Call NativaGPT's LLM handler with given messages and return plain text.
        """
        try:
            # Convert messages to a single prompt
            prompt_parts = []
            for msg in messages:
                role = msg['role']
                content = msg['content']

                if role == 'system':
                    prompt_parts.append(f"{content}\n\n")
                elif role == 'user':
                    prompt_parts.append(f"User: {content}\n")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}\n")

            combined_prompt = "".join(prompt_parts)

            # Call the LLM handler
            response = self.llm_handler.send_to_llm(combined_prompt, images=images)

            if not response.get("success", False):
                error_msg = response.get("error", "Unknown error")
                return f"Model error: {error_msg}"

            # Extract text content
            text_content = response.get("text_content", "")

            if not text_content:
                return "Model returned empty response"

            return text_content

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"LLM call failed:\n{error_details}")
            return f"Model error: {str(e)}"

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

    def _extract_images_from_tool_output(self, tool_output: str) -> List[str]:
        """
        Extract image paths from tool output.
        Looks for JSON with 'image_path' or common image file paths.
        """
        images = []

        # Try to parse as JSON first
        try:
            data = json.loads(tool_output)
            if isinstance(data, dict):
                # Look for image_path key
                if 'image_path' in data:
                    img_path = data['image_path']
                    if img_path and os.path.exists(img_path):
                        images.append(img_path)
                        logger.info(f"Found image from JSON: {img_path}")

                # Look for files array
                if 'files' in data and isinstance(data['files'], list):
                    for f in data['files']:
                        if f and os.path.exists(f) and self._is_image_file(f):
                            images.append(f)
                            logger.info(f"Found image from files: {f}")
        except json.JSONDecodeError:
            # Not JSON, try regex for file paths
            import re
            # Match common image paths
            image_pattern = r'([/\w\-\.]+\.(?:jpg|jpeg|png|gif|bmp|webp))'
            matches = re.findall(image_pattern, tool_output, re.IGNORECASE)

            for match in matches:
                if os.path.exists(match):
                    images.append(match)
                    logger.info(f"Found image from regex: {match}")

        return images

    @staticmethod
    def _is_image_file(path: str) -> bool:
        """Check if file is an image based on extension."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
        _, ext = os.path.splitext(path.lower())
        return ext in image_extensions

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
You are a tool-using assistant with access to powerful tools including ROS commands, image capture, and more.

Available tools (JSON format):
{tools_json}

For each user query, respond with ONE of these actions:

1) Answer directly:
   {{
     "action": "final_answer",
     "answer": "<your natural language answer>"
   }}

2) Call a tool:
   {{
     "action": "call_tool",
     "tool": "<exact tool name from list>",
     "args": {{ <arguments matching input_schema> }}
   }}

IMPORTANT RULES:
- Output ONLY valid JSON (no text before/after, no trailing commas)
- 'tool' must exactly match a name from the list above
- 'args' must match the tool's input_schema
- For ROS topics: use full topic names like "/camera/color/image_raw"
- For ROS commands: use complete commands like "rostopic list" or "rosnode info /node_name"

EXAMPLES:
User: "Show me what the camera sees"
Response: {{"action": "call_tool", "tool": "capture_and_analyze_image", "args": {{"topic_name": "/camera/color/image_raw"}}}}

User: "List all ROS topics"
Response: {{"action": "call_tool", "tool": "list_topics", "args": {{}}}}

User: "What's 2+2?"
Response: {{"action": "final_answer", "answer": "4"}}
"""
        return system_prompt.strip()

    async def process_query(self, query: str) -> str:
        """
        Process user query with automatic tool selection and image attachment.
        """
        if not self.servers:
            return "Error: No MCP servers connected."

        # 1) PLAN: ask LLM if it wants to call a tool
        system_prompt = self._build_tool_spec_prompt()
        plan_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        logger.info("Planning action...")
        plan_raw = self._run_model(plan_messages)

        # Try to parse JSON plan
        try:
            plan = self._parse_plan_json(plan_raw)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse plan as JSON: {e}")
            return plan_raw

        action = plan.get("action")

        # Get list of all valid tool names
        all_tool_names = [tool.name for tool in self._get_all_tools()]

        # Handle common mistake where model uses tool name as action
        if action not in ["final_answer", "call_tool"]:
            if action in all_tool_names:
                logger.info(f"Fixed plan: model used tool name '{action}' as action")
                plan = {
                    "action": "call_tool",
                    "tool": action,
                    "args": plan.get("args", {})
                }
                action = "call_tool"
            elif "tool" in plan or "args" in plan:
                plan = {
                    "action": "call_tool",
                    "tool": plan.get("tool", action),
                    "args": plan.get("args", {})
                }
                action = "call_tool"
            else:
                return f"Unexpected plan from model: {json.dumps(plan, indent=2)}"

        # 2A) Direct answer
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

            # Call MCP tool
            try:
                logger.info(f"Calling tool: {tool_name}")
                logger.info(f"Arguments: {json.dumps(args, indent=2)}")

                result = await server["session"].call_tool(tool_name, args)
                tool_output = self._extract_tool_text(result)

                logger.info(f"Tool output length: {len(tool_output)} chars")

                # CRITICAL: Extract images from tool output
                images_to_send = self._extract_images_from_tool_output(tool_output)

                if images_to_send:
                    logger.info(f"✓ Found {len(images_to_send)} image(s) to attach to LLM")
                else:
                    logger.info("No images found in tool output")

            except Exception as e:
                import traceback
                logger.error(f"Tool execution failed:\n{traceback.format_exc()}")
                return f"Error calling tool '{tool_name}': {e}"

            # 3) Ask LLM to summarize tool output WITH IMAGES
            answer_system_prompt = """
You are a helpful assistant analyzing tool outputs. The user asked a question and an external tool provided data.

Your job:
1. If images are attached, describe what you see in them
2. Explain the tool's findings in clear, natural language
3. Answer the user's original question based on the data

Be specific and informative.
"""
            answer_messages = [
                {"role": "system", "content": answer_system_prompt.strip()},
                {
                    "role": "user",
                    "content": (
                        f"User question: {query}\n\n"
                        f"Tool used: {tool_name}\n"
                        f"Tool arguments: {json.dumps(args)}\n\n"
                        f"Tool output:\n{tool_output}\n\n"
                        "Please analyze and explain this for the user."
                    ),
                },
            ]

            # CRITICAL: Send WITH images if available
            logger.info("Generating final answer with LLM...")
            if images_to_send:
                logger.info(f"Attaching {len(images_to_send)} images to LLM context")
                final_answer = self._run_model(answer_messages, images=images_to_send)
            else:
                final_answer = self._run_model(answer_messages)

            return final_answer

        # Unknown action
        return f"Unexpected plan from model: {plan_raw}"

    @staticmethod
    def _parse_plan_json(plan_raw: str):
        """Parse the model's plan as JSON, handling code blocks."""
        plan_raw = plan_raw.strip()

        # Remove markdown code blocks
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
        """Cleanup resources."""
        logger.info("Cleaning up MCP client...")
        await self.exit_stack.aclose()
        if self.llm_handler:
            self.llm_handler.cleanup()


async def main():
    """Test the MCP client."""
    if len(sys.argv) < 2:
        print("Usage: python mcp_client.py <server_script1> [server_script2] ...")
        sys.exit(1)

    # Create LLM handler
    from NativaGPT.lib.config_manager import ConfigManager
    config_path = "config/config_default.json"
    config_manager = ConfigManager(config_path)
    config = config_manager.get()

    from NativaGPT.lib.handlers.llm_prompt_handler import LLMPromptHandler
    llm_handler = LLMPromptHandler(config)

    client = MCPClient(llm_handler)
    try:
        await client.connect_to_server(sys.argv[1:])

        # Test queries
        queries = [
            "List all available ROS topics",
            "Show me what the robot's camera sees",
            "What's the weather in Florida?"
        ]

        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            response = await client.process_query(query)
            print(f"Response: {response}")

    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())