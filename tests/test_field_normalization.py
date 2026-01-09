#!/usr/bin/env python3
"""
Debug test for field normalization in JSON response handler.
"""

import sys
import os
import json

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from NativaGPT.lib.handlers.json_response_handler import JsonResponseHandler
from NativaGPT.lib.coloring_logger import logger


def test_field_normalization():
    """Test field normalization specifically."""

    handler = JsonResponseHandler()

    test_cases = [
        # Test case 1: Alternative field names
        {
            "name": "Alternative field names",
            "json": {"action": "ls -la", "type": "shell", "path": "/home/user"},
        },
        # Test case 2: Missing fields with defaults
        {"name": "Missing fields with defaults", "json": {"command": "echo 'test'"}},
        # Test case 3: Complex nested structure
        {
            "name": "Complex nested structure",
            "json": {
                "tool": {
                    "call": "move_turtle",
                    "args": "0 2 0 0 0 0",
                    "executor": "ros",
                    "working_dir": "/tmp",
                }
            },
        },
    ]

    for test_case in test_cases:
        logger.info(f"\n--- Testing: {test_case['name']} ---")

        # Test normalization
        normalized = handler._normalize_field_names(test_case["json"])
        logger.info(f"Original: {test_case['json']}")
        logger.info(f"Normalized: {normalized}")

        # Test function object normalization
        function_obj = handler._normalize_function_object(test_case["json"])
        logger.info(f"Function object: {function_obj}")

        # Test both together
        final = handler._normalize_field_names(function_obj)
        logger.info(f"Final: {final}")

        # Check if it has required fields
        required_fields = ["command", "execution", "location"]
        missing_fields = [
            field for field in required_fields if field not in final or not final[field]
        ]
        logger.info(f"Missing fields: {missing_fields}")


if __name__ == "__main__":
    test_field_normalization()
