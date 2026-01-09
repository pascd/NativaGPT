#!/usr/bin/env python3
"""
Test script for enhanced JSON detection in LLM responses.
Tests various formats and edge cases to ensure robust command extraction.
"""

import sys
import os
import json

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from NativaGPT.lib.handlers.llm_response_handler import LLMResponseHandler
from NativaGPT.lib.handlers.json_response_handler import JsonResponseHandler
from NativaGPT.lib.coloring_logger import logger


def test_json_detection():
    """Test various JSON detection scenarios."""

    llm_handler = LLMResponseHandler()
    json_handler = JsonResponseHandler()

    test_cases = [
        # Test case 1: Standard JSON in code block
        {
            "name": "Standard JSON in code block",
            "response": """Here's the command to execute:

```json
{"command": "ls -la", "execution": "shell", "location": "/home/user"}
```

This will list all files.""",
        },
        # Test case 2: JSON without code block
        {
            "name": "JSON without code block",
            "response": """I'll execute this command for you: {"command": "echo 'Hello World'", "execution": "shell", "location": ""}""",
        },
        # Test case 3: Multiple JSON objects
        {
            "name": "Multiple JSON objects",
            "response": """Here are two commands:
First: {"command": "pwd", "execution": "shell", "location": "/tmp"}
Second: {"command": "date", "execution": "shell", "location": ""}""",
        },
        # Test case 4: Alternative field names
        {
            "name": "Alternative field names",
            "response": """{"action": "ls -la", "type": "shell", "path": "/home/user"}""",
        },
        # Test case 5: Nested function structure
        {
            "name": "Nested function structure",
            "response": """{"function": {"command": "echo 'test'", "execution": "shell", "location": ""}}""",
        },
        # Test case 6: Malformed JSON with think tags
        {
            "name": "Malformed JSON with think tags",
            "response": """<think>I need to execute a command</think>
Let me run this for you:
```json
{"command": "ps aux", "execution": "shell", "location": ""}
```
Done.""",
        },
        # Test case 7: JSON array
        {
            "name": "JSON array of commands",
            "response": """[{"command": "echo 'first'", "execution": "shell", "location": ""}, {"command": "echo 'second'", "execution": "shell", "location": ""}]""",
        },
        # Test case 8: Missing fields (should get defaults)
        {
            "name": "Missing fields with defaults",
            "response": """{"command": "echo 'test'"}""",
        },
        # Test case 9: Mixed text and JSON
        {
            "name": "Mixed text and JSON",
            "response": """I'll help you with that task. First, let me check the current directory: {"command": "pwd", "execution": "shell", "location": ""}. Then I'll list the files: {"command": "ls -la", "execution": "shell", "location": ""}. Let me know if you need anything else!""",
        },
        # Test case 10: Complex nested JSON
        {
            "name": "Complex nested JSON",
            "response": """{"tool": {"call": "move_turtle", "args": "0 2 0 0 0 0", "executor": "ros", "working_dir": "/tmp"}}""",
        },
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n--- Test {i}: {test_case['name']} ---")

        try:
            # Extract JSON using LLM handler
            extracted = llm_handler.extract_json_str(test_case["response"])

            logger.info(f"Found {len(extracted['json_strings'])} JSON strings")
            logger.info(f"Text content: '{extracted['text_content'][:100]}...'")

            # Process JSON through JsonResponseHandler
            commands = json_handler.check_all_functions(extracted)

            logger.info(f"Processed {len(commands)} commands")

            for j, cmd in enumerate(commands):
                logger.info(
                    f"  Command {j + 1}: {cmd['command']} (execution: {cmd['execution']}, location: '{cmd['location']}')"
                )

            results.append(
                {
                    "test_name": test_case["name"],
                    "success": True,
                    "json_count": len(extracted["json_strings"]),
                    "command_count": len(commands),
                    "commands": commands,
                }
            )

        except Exception as e:
            logger.error(f"Test failed: {e}")
            results.append(
                {"test_name": test_case["name"], "success": False, "error": str(e)}
            )

    # Summary
    logger.info("\n=== TEST SUMMARY ===")
    passed = sum(1 for r in results if r["success"])
    total = len(results)

    logger.info(f"Passed: {passed}/{total}")

    for result in results:
        status = "✅" if result["success"] else "❌"
        logger.info(f"{status} {result['test_name']}")
        if not result["success"]:
            logger.info(f"   Error: {result['error']}")
        else:
            logger.info(
                f"   JSON: {result['json_count']}, Commands: {result['command_count']}"
            )

    return results


if __name__ == "__main__":
    test_json_detection()
