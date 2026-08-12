"""MCP client for NativaGPT.

Connects to one or more Model Context Protocol (MCP) servers, asks the
configured LLM to plan whether to answer directly or invoke a tool, executes
the chosen tool, and drives a self-correcting retry loop that verifies
answers and extracts images from tool output for multimodal follow-up.
"""

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
    """Client that connects to one or more MCP servers and drives an LLM-based tool-use loop.

    Maintains sessions to multiple MCP servers, asks the configured LLM
    handler to plan whether to answer directly or call a tool, executes the
    chosen tool, extracts any images referenced in the tool output for
    multimodal follow-up, and retries with self-correction when a tool call
    fails or the LLM's answer doesn't verifiably satisfy the original
    request.
    """

    def __init__(self, llm_handler: Optional[LLMPromptHandler] = None):
        """Initialize the client with no server connections yet.

        Args:
            llm_handler: Handler used to send prompts (and images) to the
                underlying LLM. If omitted, methods that need it return an
                error string instead of raising.
        """
        # Store multiple server sessions
        self.servers: List[Dict[str, Any]] = []
        self.exit_stack = AsyncExitStack()

        self.llm_handler = llm_handler
        logger.info("MCP Client initialized with NativaGPT LLM handler")

    async def connect_to_server(self, server_script_list: List[str]):
        """Launch and connect to each MCP server script, registering its tools.

        For every script, spawns the appropriate interpreter (bash/python
        via `sys.executable`/node) as an MCP stdio subprocess, forwarding the
        current environment (including ROS variables) so ROS-based servers
        work correctly. Scripts that don't exist or have an unsupported
        extension are skipped with a logged warning/error; a server that
        fails to connect is likewise skipped and does not abort the rest.

        Args:
            server_script_list: Paths to MCP server scripts (.py, .js, or
                .sh).

        Raises:
            RuntimeError: If none of the given servers could be connected.
        """

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
                    # Use the same interpreter running NativaGPT (venv-safe)
                    command = sys.executable
                else:  # is_js
                    command = "node"

                # CRITICAL: Pass full environment including ROS variables
                env = os.environ.copy()

                # Ensure key ROS variables are present
                ros_vars = [
                    "ROS_DISTRO",
                    "ROS_ROOT",
                    "ROS_PACKAGE_PATH",
                    "ROS_MASTER_URI",
                    "PYTHONPATH",
                    "LD_LIBRARY_PATH",
                    "CMAKE_PREFIX_PATH",
                    "PKG_CONFIG_PATH",
                    "PATH",
                ]

                missing_ros = []
                for var in ["ROS_DISTRO", "ROS_ROOT"]:
                    if var not in env:
                        missing_ros.append(var)

                if missing_ros:
                    logger.warning(f"Missing ROS environment variables: {missing_ros}")
                    logger.warning(
                        "Did you source ROS? Run: source /opt/ros/noetic/setup.bash"
                    )

                server_params = StdioServerParameters(
                    command=command,
                    args=[server_script_path],
                    env=env,  # Pass full environment to subprocess
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
                self.servers.append(
                    {
                        "name": server_name,
                        "session": session,
                        "stdio": stdio,
                        "write": write,
                        "tools": tools,
                    }
                )

                logger.info(f"✓ Connected to '{server_name}' with {len(tools)} tools:")
                for tool in tools:
                    desc = tool.description or ""
                    logger.info(f"  - {tool.name}: {desc[:60]}...")

            except Exception as e:
                logger.error(f"Failed to connect to {server_script_path}: {e}")
                continue

        if not self.servers:
            raise RuntimeError("No MCP servers could be connected")

        total_tools = sum(len(server["tools"]) for server in self.servers)
        logger.info(f"\nTotal servers: {len(self.servers)}, Total tools: {total_tools}")

    def _get_all_tools(self) -> List[Any]:
        """Return a flattened list of tools from every connected server."""
        all_tools = []
        for server in self.servers:
            all_tools.extend(server["tools"])
        return all_tools

    def _find_tool_server(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Return the server dict exposing `tool_name`, or None if not found."""
        for server in self.servers:
            for tool in server["tools"]:
                if tool.name == tool_name:
                    return server
        return None

    def _run_model(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List[str]] = None,
        system_instruction: Optional[str] = None,
    ) -> str:
        """Flatten a chat-style message list into one prompt and send it to the LLM handler."""
        try:
            prompt_parts = []
            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                # Se houver override, ignoramos mensagens de sistema antigas
                if role == "system" and system_instruction:
                    continue

                if role == "system":
                    prompt_parts.append(f"{content}\n\n")
                elif role == "user":
                    prompt_parts.append(f"User: {content}\n")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}\n")

            combined_prompt = "".join(prompt_parts)

            # Passa o system_instruction para o handler (vazio string se não fornecido)
            final_system_instruction = system_instruction if system_instruction else ""
            final_images = images if images is not None else []

            if not self.llm_handler:
                return "Error: LLM handler not initialized"

            response = self.llm_handler.send_to_llm(
                combined_prompt,
                images=final_images,
                system_instruction=final_system_instruction,
            )

            if not response.get("success", False):
                return f"Model error: {response.get('error')}"

            return response.get("text_content", "") or "Model returned empty response"

        except Exception as e:
            return f"Model error: {str(e)}"

    @staticmethod
    def _extract_tool_text(result: Any) -> str:
        """Join the text content items of an MCP call_tool result into one string."""
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
        """Extract existing image file paths from tool output.

        Tries to parse the output as JSON and read an `image_path` or
        `files` field first; if it isn't JSON, falls back to regex-matching
        image file paths in the raw text. Only paths that exist on disk are
        returned.
        """
        images = []

        # Try to parse as JSON first
        try:
            data = json.loads(tool_output)
            if isinstance(data, dict):
                # Look for image_path key
                if "image_path" in data:
                    img_path = data["image_path"]
                    if img_path and os.path.exists(img_path):
                        images.append(img_path)
                        logger.info(f"Found image from JSON: {img_path}")

                # Look for files array
                if "files" in data and isinstance(data["files"], list):
                    for f in data["files"]:
                        if f and os.path.exists(f) and self._is_image_file(f):
                            images.append(f)
                            logger.info(f"Found image from files: {f}")
        except json.JSONDecodeError:
            # Not JSON, try regex for file paths
            import re

            # Match common image paths
            image_pattern = r"([/\w\-\.]+\.(?:jpg|jpeg|png|gif|bmp|webp))"
            matches = re.findall(image_pattern, tool_output, re.IGNORECASE)

            for match in matches:
                if os.path.exists(match):
                    images.append(match)
                    logger.info(f"Found image from regex: {match}")

        return images

    @staticmethod
    def _is_image_file(path: str) -> bool:
        """Return True if `path`'s extension matches a known image format."""
        image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
        }
        _, ext = os.path.splitext(path.lower())
        return ext in image_extensions

    def _build_tool_spec_prompt(self) -> str:
        """Build the system prompt listing all tools and the expected JSON plan format."""
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

        system_prompt = f"""
You are a tool-using assistant with access to powerful tools.

Available tools (JSON format):
{tools_json}

For each user query, respond with ONE of these actions:

1) Answer directly (for explanations, conversations, and image descriptions):
   {{
     "action": "final_answer",
     "answer": "<your natural language answer>"
   }}

2) Call a tool (for executing commands and getting data):
   {{
     "action": "call_tool",
     "tool": "<exact tool name from list>",
     "args": {{ <arguments matching input_schema> }}
   }}

IMPORTANT RULES:
- Output ONLY valid JSON (no text before/after, no trailing commas)
- 'tool' must exactly match a name from the list above
- 'args' must match the tool's input_schema
- Commands with {{placeholder}} need those arguments filled in
- For ROS topics: use full topic names like "/camera/color/image_raw"

EXAMPLES:
User: "What is 2+2?"
Response: {{"action": "final_answer", "answer": "4"}}

User: "Check disk space"
Response: {{"action": "call_tool", "tool": "check_disk", "args": {{}}}}

User: "What do you see in the image?"
Response: {{"action": "final_answer", "answer": "The image shows a robotic arm with a blue gripper."}}
"""
        return system_prompt.strip()

    async def process_query(
        self, query: str, max_correction_iterations: int = 3
    ) -> str:
        """Answer a query by planning and possibly executing an MCP tool, retrying with self-correction on failure.

        Each iteration asks the LLM to produce a JSON plan (`final_answer`
        or `call_tool`) given the available tools, then either returns the
        direct answer or invokes the chosen tool on whichever connected
        server exposes it. Tool output is inspected for images (which are
        then described by the LLM using vision input) and for error
        indicators; on iterations after the first, generated answers are
        also checked against the original query via an LLM-based
        verification pass. Whenever a plan is malformed, a tool is missing,
        a tool call raises, or verification fails, the loop builds a
        correction prompt describing the previous failure and retries, up
        to `max_correction_iterations` times.

        Args:
            query: The user's natural-language request.
            max_correction_iterations: Maximum number of plan/execute/verify
                attempts before giving up.

        Returns:
            The final natural-language answer, or a message describing why
            the request could not be fulfilled after exhausting all
            iterations. Failures are returned as strings rather than
            raised.
        """
        logger.info(f"MCP process_query ENTRY - query: {query[:100]}...")

        if not self.servers:
            logger.error("No MCP servers connected")
            return "Error: No MCP servers connected."

        if not self.llm_handler:
            logger.error("LLM handler not initialized")
            return "Error: LLM handler not initialized"

        logger.info(f"MCP servers available: {[s['name'] for s in self.servers]}")

        original_query = query
        last_error = ""
        iteration = 0

        while iteration < max_correction_iterations:
            iteration += 1
            logger.info(
                f"MCP Query Iteration {iteration}/{max_correction_iterations}: {query[:100]}..."
            )

            # 1) PLAN: ask LLM if it wants to call a tool
            system_prompt = self._build_tool_spec_prompt()
            plan_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

            logger.info("Planning action...")
            plan_raw = self._run_model(plan_messages)

            logger.info(f"LLM response length: {len(plan_raw)} chars")
            logger.info(f"LLM response preview: {plan_raw[:200]}...")

            # ALWAYS check for executable JSON in the response first
            forced_plan = self._force_execute_json_plan(plan_raw)
            if forced_plan:
                logger.info(
                    "🎯 EXECUTING JSON plan found in LLM response - bypassing normal extraction"
                )
                plan = forced_plan
            else:
                # Try to parse JSON plan normally
                plan = self._extract_json_plan(plan_raw)

                logger.info(f"Normal extraction result: {plan}")

                if plan is None:
                    logger.warning("Could not extract valid JSON plan from response")
                    if iteration == 1:
                        # No valid plan found - return as direct answer
                        logger.info("No tool plan found, returning as direct answer")
                        return plan_raw
                    else:
                        # On retry, ask for proper JSON
                        query = self._build_correction_query(
                            original_query,
                            'Response was not valid JSON format - must include {"action": "call_tool" or "final_answer", ...}',
                            plan_raw,
                            {
                                "fulfilled": False,
                                "reason": "Response was not valid JSON format",
                            },
                        )
                        continue

            logger.info(
                f"Successfully extracted plan: action={plan.get('action')}, tool={plan.get('tool', 'N/A')}"
            )

            action = plan.get("action")

            # Get list of all valid tool names
            all_tool_names = [tool.name for tool in self._get_all_tools()]

            # Handle common mistake where model uses tool name as action
            if action not in ["final_answer", "call_tool"]:
                if action in all_tool_names:
                    logger.info(
                        f"Fixed plan: model used tool name '{action}' as action"
                    )
                    plan = {
                        "action": "call_tool",
                        "tool": action,
                        "args": plan.get("args", {}),
                    }
                    action = "call_tool"
                elif "tool" in plan or "args" in plan:
                    plan = {
                        "action": "call_tool",
                        "tool": plan.get("tool", action),
                        "args": plan.get("args", {}),
                    }
                    action = "call_tool"
                else:
                    # Not a valid plan - return as answer
                    if iteration == 1:
                        return plan_raw
                    query = self._build_correction_query(
                        original_query,
                        "Invalid plan format",
                        plan_raw,
                        {"fulfilled": False, "reason": "Invalid plan format"},
                    )
                    continue

            # 2A) Direct answer
            if action == "final_answer":
                answer = plan.get("answer", "")
                # If this is not the first iteration, verify the answer addresses the original query
                if iteration > 1:
                    verification = self._verify_answer_fulfills_request(
                        original_query, answer
                    )
                    if verification["fulfilled"]:
                        return answer
                    else:
                        logger.info(
                            f"Answer doesn't fully fulfill request: {verification['reason']}"
                        )
                        query = self._build_correction_query(
                            original_query, last_error, answer, verification
                        )
                        continue
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
                    error_msg = f"Tool '{tool_name}' not found in any connected server"
                    last_error = error_msg
                    # Try to get an alternative tool
                    if iteration < max_correction_iterations:
                        query = self._build_correction_query(
                            original_query,
                            error_msg,
                            "",
                            {"fulfilled": False, "reason": "Tool not found"},
                        )
                        logger.info(f"Tool not found, attempting correction...")
                        continue
                    return error_msg

                # Call MCP tool
                try:
                    logger.info(f"Calling tool: {tool_name}")
                    logger.info(f"Arguments: {json.dumps(args, indent=2)}")

                    result = await server["session"].call_tool(tool_name, args)
                    tool_output = self._extract_tool_text(result)

                    logger.info(f"Tool output: {tool_output[:200]}...")
                    logger.info(f"Tool output length: {len(tool_output)} chars")

                    # Check if tool execution had an error
                    is_error, error_reason = self._check_tool_error(
                        tool_output, tool_name
                    )

                    # CRITICAL: Extract images from tool output
                    images_to_send = self._extract_images_from_tool_output(tool_output)

                    logger.info("Generating final answer with LLM...")

                    if images_to_send:
                        user_content = f"Vejo uma imagem. Descreve o que está visível nesta imagem, em português: {query}"

                        answer_messages = [
                            {
                                "role": "user",
                                "content": user_content,
                            }
                        ]

                        logger.info(
                            f"Attaching {len(images_to_send)} images to LLM context (Visual Mode)"
                        )

                        final_answer = self._run_model(
                            answer_messages,
                            images=images_to_send,
                            system_instruction="",
                        )

                        # Verify the visual answer fulfills the request
                        if iteration > 1 or is_error:
                            verification = self._verify_answer_fulfills_request(
                                original_query, final_answer
                            )
                            if not verification["fulfilled"]:
                                logger.info(
                                    f"Visual answer doesn't fulfill request: {verification['reason']}"
                                )
                                last_error = (
                                    tool_output
                                    if is_error
                                    else "Incomplete visual analysis"
                                )
                                if iteration < max_correction_iterations:
                                    query = self._build_correction_query(
                                        original_query,
                                        last_error,
                                        final_answer,
                                        verification,
                                    )
                                    continue

                    else:
                        # TEXT MODE - Explain the result or error
                        answer_system_prompt = """
You are a helpful assistant. The user asked a question and a tool provided the text result below.
Summarize the result clearly in Portuguese.

IMPORTANT: If the tool result shows an error, timeout, or unexpected output, explain what went wrong
and suggest what might need to be fixed.
                        """
                        answer_messages = [
                            {"role": "system", "content": answer_system_prompt.strip()},
                            {
                                "role": "user",
                                "content": (
                                    f"User Question: {query}\n"
                                    f"Tool Result: {tool_output}\n\n"
                                    "Explain this result to the user. If there's an error, explain what went wrong."
                                ),
                            },
                        ]
                        final_answer = self._run_model(answer_messages)

                        # Check if tool result was successful and fulfills request
                        if iteration == 1 and not is_error:
                            # First attempt with successful result - verify it answers the user
                            verification = self._verify_answer_fulfills_request(
                                original_query, final_answer
                            )
                            if verification["fulfilled"]:
                                return final_answer
                            else:
                                logger.info(
                                    f"Tool result doesn't fulfill request: {verification['reason']}"
                                )
                                last_error = tool_output
                                if iteration < max_correction_iterations:
                                    query = self._build_correction_query(
                                        original_query,
                                        last_error,
                                        final_answer,
                                        verification,
                                    )
                                    continue
                        elif is_error:
                            # Tool returned an error - need correction
                            logger.info(f"Tool error detected: {tool_output[:200]}...")
                            last_error = tool_output
                            if iteration < max_correction_iterations:
                                verification = {
                                    "fulfilled": False,
                                    "reason": f"Tool error: {tool_output[:100]}...",
                                }
                                query = self._build_correction_query(
                                    original_query,
                                    last_error,
                                    final_answer,
                                    verification,
                                )
                                continue

                    return final_answer

                except Exception as e:
                    error_msg = f"Error calling tool '{tool_name}': {str(e)}"
                    logger.error(error_msg)
                    last_error = str(e)

                    if iteration < max_correction_iterations:
                        # Try to get a corrected plan
                        query = self._build_correction_query(
                            original_query,
                            last_error,
                            "",
                            {"fulfilled": False, "reason": f"Exception: {str(e)}"},
                        )
                        logger.info(f"Tool exception, attempting correction...")
                        continue
                    return error_msg

            # Unknown action
            return f"Unexpected plan from model: {plan_raw}"

        # Max iterations reached
        return (
            f"Could not fulfill request after {max_correction_iterations} attempts.\n"
            f"Last error: {last_error}\n"
            f"Original query: {original_query}\n\n"
            "Please check the tool configuration or try a different approach."
        )

    def _check_tool_error(self, tool_output: str, tool_name: str) -> tuple[bool, str]:
        """Heuristically detect an error in tool output via keyword matching; returns (is_error, reason)."""
        output_lower = tool_output.lower().strip()

        # Check for explicit error indicators
        error_patterns = [
            ("error", "Output contains 'error'"),
            ("failed", "Output contains 'failed'"),
            ("exception", "Output contains 'exception'"),
            ("cannot", "Output contains 'cannot'"),
            ("unable", "Output contains 'unable'"),
            ("not found", "Output contains 'not found'"),
            ("no such", "Output contains 'no such'"),
            ("invalid", "Output contains 'invalid'"),
            ("permission denied", "Output contains 'permission denied'"),
            ("connection refused", "Output contains 'connection refused'"),
            ("service unavailable", "Output contains 'service unavailable'"),
            ("timed out", "Output contains 'timed out'"),
            ("timeout", "Output contains 'timeout'"),
        ]

        for pattern, reason in error_patterns:
            if pattern in output_lower:
                return True, reason

        # Check for empty or very short responses that might indicate failure
        if len(output_lower) < 3:
            return True, f"Output is empty or too short ({len(output_lower)} chars)"

        # Check for specific success indicators
        success_patterns = [
            "command executed",
            "success",
            "done",
            "published",
            "ok",
            "true",
        ]
        for pattern in success_patterns:
            if pattern in output_lower:
                return False, "Success indicator found"

        # If output looks like command output but no errors, it's likely successful
        # ROS commands often return minimal output on success
        return False, ""

    def _verify_answer_fulfills_request(self, original_query: str, answer: str) -> dict:
        """Ask the LLM whether `answer` fulfills `original_query`; returns a fulfilled/reason/confidence dict (defaults to fulfilled=True if verification itself fails)."""
        try:
            if not self.llm_handler:
                return {
                    "fulfilled": True,
                    "reason": "No LLM handler, assuming success",
                    "confidence": 0.5,
                }

            verification_prompt = f"""
You are a verification assistant. Determine if the answer fulfills the user's request.

Original User Request: {original_query}

Actual Answer: {answer}

Respond with ONLY a JSON object (no other text):
{{
    "fulfilled": true/false,
    "reason": "brief explanation of why it does or doesn't fulfill the request",
    "confidence": 0.0-1.0
}}
            """

            response = self.llm_handler.send_to_llm(
                verification_prompt,
                system_instruction="You are a verification assistant. Output ONLY valid JSON.",
            )

            if response.get("success"):
                text = response.get("text_content", "")
                # Extract JSON from response
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1:
                    json_str = text[start : end + 1]
                    result = json.loads(json_str)
                    return {
                        "fulfilled": result.get("fulfilled", False),
                        "reason": result.get("reason", "Unknown"),
                        "confidence": result.get("confidence", 0.5),
                    }

            return {
                "fulfilled": True,
                "reason": "Could not verify, assuming success",
                "confidence": 0.5,
            }

        except Exception as e:
            logger.warning(f"Verification failed: {e}")
            return {
                "fulfilled": True,
                "reason": "Verification error, assuming success",
                "confidence": 0.5,
            }

    def _build_correction_query(
        self, original_query: str, last_error: str, last_answer: str, verification: dict
    ) -> str:
        """Build a follow-up prompt asking the LLM to analyze the previous failure and produce a corrected JSON plan."""
        last_answer_preview = (
            last_answer[:500] if last_answer else "No answer generated"
        )

        correction_prompt = f"""
The previous attempt to answer this request failed or was incomplete:

ORIGINAL USER REQUEST: {original_query}

PREVIOUS ERROR/ISSUE: {last_error}

VERIFICATION RESULT: {verification["reason"]}

PREVIOUS ANSWER (if any): {last_answer_preview}

Please analyze what went wrong and provide a corrected plan. 
Output ONLY a JSON object:
{{
    "action": "call_tool" or "final_answer",
    "tool": "correct_tool_name" (only if action is call_tool),
    "args": {{ corrected_arguments }},
    "answer": "direct answer" (only if action is final_answer),
    "analysis": "brief explanation of what was wrong and how you're fixing it"
}}

Think about:
1. What specifically failed or was incomplete?
2. What tool should be called instead or with different parameters?
3. How can we ensure the request is fulfilled this time?
        """
        return correction_prompt

    @staticmethod
    def _parse_plan_json(plan_raw: str):
        """Parse the model's plan as JSON, stripping markdown code fences first."""
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

    def _extract_json_plan(self, plan_raw: str):
        """Extract a JSON plan dict from a raw LLM response, tolerating code fences and curly quotes; returns None if none found."""
        try:
            # First try direct parsing
            return self._parse_plan_json(plan_raw)
        except (json.JSONDecodeError, ValueError):
            pass

        try:
            plan_raw = plan_raw.strip()

            # Normalize curly quotes to straight quotes (common Unicode quotes)
            replacements = [
                ('"', '"'),
                ('"', '"'),  # curly double quotes
                ("'", "'"),
                ("'", "'"),  # curly single quotes
                ('"', '"'),
                ('"', '"'),  # another variant
            ]
            for old, new in replacements:
                plan_raw = plan_raw.replace(old, new)

            # Remove markdown code blocks
            if plan_raw.startswith("```"):
                lines = plan_raw.splitlines()
                lines = [ln for ln in lines if not ln.strip().startswith("```")]
                plan_raw = "\n".join(lines).strip()

            # Find JSON object - look for "action" key regardless of quote style
            # First, try to find {"action
            start = plan_raw.find('{"action')
            if start == -1:
                # Try finding just {action without quotes (unlikely but possible)
                start = plan_raw.find("{action")

            if start == -1:
                # Last resort: find first { and try to parse from there
                brace_start = plan_raw.find("{")
                if brace_start != -1:
                    # Look for "action" or 'action' nearby
                    after_brace = plan_raw[brace_start : brace_start + 100]
                    if '"action' in after_brace or "'action" in after_brace:
                        start = brace_start

            if start != -1:
                # Find the matching closing brace
                depth = 0
                end = start
                for i in range(start, len(plan_raw)):
                    if plan_raw[i] == "{":
                        depth += 1
                    elif plan_raw[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break

                if depth == 0:
                    json_str = plan_raw[start:end]
                    # Try to parse it
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try replacing curly quotes and retry
                        for old, new in [('"', '"'), ('"', '"')]:
                            json_str_fixed = json_str.replace(old, new)
                            try:
                                return json.loads(json_str_fixed)
                            except json.JSONDecodeError:
                                continue

            return None
        except Exception as e:
            logger.warning(f"Failed to extract JSON plan: {e}")
            return None

    def _force_execute_json_plan(self, response_text: str):
        """Scan the response for any balanced {...} block that parses as a valid call_tool/final_answer plan.

        Last-resort fallback used when normal extraction fails to isolate a
        plan; attempts simple fixes (trailing commas, smart quotes) before
        giving up on a block.

        Returns:
            The first valid plan dict found, or None if none did.
        """
        try:
            # Look for JSON blocks and execute them directly
            import re

            # Find all JSON-like blocks (handle nested braces)
            # Look for { followed by balanced braces
            json_blocks = []
            i = 0
            while i < len(response_text):
                if response_text[i] == "{":
                    # Start of potential JSON block
                    depth = 0
                    start = i
                    for j in range(i, len(response_text)):
                        if response_text[j] == "{":
                            depth += 1
                        elif response_text[j] == "}":
                            depth -= 1
                            if depth == 0:
                                # Found matching closing brace
                                json_blocks.append(response_text[start : j + 1])
                                i = j + 1
                                break
                    else:
                        # No matching closing brace found
                        i += 1
                else:
                    i += 1

            logger.info(f"Found {len(json_blocks)} potential JSON blocks in response")

            for block in json_blocks:
                logger.info(f"Testing JSON block: {block[:100]}...")

                try:
                    # Try to parse as JSON
                    plan = json.loads(block)

                    # Check if it's a valid plan
                    action = plan.get("action")
                    if action in ["call_tool", "final_answer"]:
                        logger.info(f"Found executable JSON plan: {block[:100]}...")
                        return plan

                except json.JSONDecodeError:
                    # Try fixing common JSON issues
                    block_fixed = block

                    # Remove trailing commas before }
                    block_fixed = re.sub(r",\s*\}", "}", block_fixed)
                    block_fixed = re.sub(r",\s*\]", "]", block_fixed)

                    # Normalize quotes - handle all variations
                    block_fixed = block_fixed.replace('"', '"').replace('"', '"')
                    block_fixed = block_fixed.replace(""", "'").replace(""", "'")
                    block_fixed = block_fixed.replace('"', '"').replace('"', '"')

                    logger.info(f"Trying fixed block: {block_fixed[:100]}...")

                    try:
                        plan = json.loads(block_fixed)
                        action = plan.get("action")
                        if action in ["call_tool", "final_answer"]:
                            logger.info(
                                f"Found executable JSON plan after fixing: {block_fixed[:100]}..."
                            )
                            return plan
                    except json.JSONDecodeError as e2:
                        logger.info(f"Block still invalid: {e2}")
                        continue

            logger.info("No executable JSON plans found")
            return None

        except Exception as e:
            logger.warning(f"Failed to force execute JSON plan: {e}")
            return None

    async def cleanup(self):
        """Close all MCP server sessions/subprocesses and clean up the LLM handler."""
        logger.info("Cleaning up MCP client...")
        await self.exit_stack.aclose()
        if self.llm_handler:
            self.llm_handler.cleanup()


async def main():
    """Manual smoke test: connect to servers given as CLI args and run sample queries through process_query."""
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
            "What's the weather in Florida?",
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
