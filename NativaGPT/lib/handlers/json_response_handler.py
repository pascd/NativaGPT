"""Normalization and formatting of JSON command payloads from LLM responses.

Provides :class:`JsonResponseHandler`, which normalizes the many possible
shapes an LLM-emitted command JSON object can take (varying field names and
nesting) into a consistent ``{"command", "execution", "location"}``
structure, and prepares them for execution/display.
"""

import os
import json
import sys

from NativaGPT.lib.coloring_logger import logger


class JsonResponseHandler:
    """Normalizes and validates JSON command payloads extracted from LLM responses.

    Accepts command JSON in a variety of shapes and field-naming
    conventions and converts them into a consistent structure of
    ``command``, ``execution``, and ``location`` fields, ready to be
    executed or displayed.

    Attributes:
        execution_list: Indicative mapping of known execution/executor
            names (e.g. ``"shell"``, ``"python"``, ``"ros2 launch"``) to an
            optional command prefix. Not currently used to prefix commands.
    """

    # Mapeamento conhecido (apenas indicativo; hoje não é necessário prefixar executores)
    execution_list = {
        "shell": "",
        "bash": "",
        "python": "python",
        "ros": "",
        "ros1": "",
        "ros2": "",
        "ros1 launch": "roslaunch",
        "ros2 launch": "ros2 launch",
        "other": "",
    }

    def __init__(self):
        """Initialize the handler and log its creation."""
        logger.info("Created JsonResponseHandler instance.")

    def _normalize_function_object(self, fn_obj):
        """Unwrap a nested command object into a flat command dict.

        Enhanced normalization that accepts multiple command structures:
          {'command': '...', 'execution': 'shell', 'location': '/tmp'}
          {'function': {'command': '...', ...}}
          {'action': '...', 'type': 'shell', 'path': '/tmp'}
          {'tool': '...', 'method': 'shell', 'directory': '/tmp'}
          {'call': '...', 'executor': 'shell', 'working_dir': '/tmp'}

        Args:
            fn_obj: The candidate command object (typically a dict, but
                passed through unchanged for other types).

        Returns:
            The unwrapped ``dict`` (e.g. the value of a nested
            ``"function"``/``"action"``/``"tool"``/``"call"`` key) if one
            was found, otherwise ``fn_obj`` unchanged.
        """
        if isinstance(fn_obj, dict):
            # Handle nested function structure
            if "function" in fn_obj and isinstance(fn_obj["function"], dict):
                return fn_obj["function"]

            # Handle nested action structure
            if "action" in fn_obj and isinstance(fn_obj["action"], dict):
                return fn_obj["action"]

            # Handle nested tool structure
            if "tool" in fn_obj and isinstance(fn_obj["tool"], dict):
                return fn_obj["tool"]

            # Handle nested call structure
            if "call" in fn_obj and isinstance(fn_obj["call"], dict):
                return fn_obj["call"]

        return fn_obj

    def _normalize_field_names(self, function_data):
        """Normalize various field names to standard format.

        Handles multiple naming conventions for command fields, mapping
        aliases such as ``cmd``/``action``/``tool``/``call``/``execute``/
        ``run`` to ``"command"``, ``executor``/``type``/``method``/
        ``shell``/``runtime`` to ``"execution"``, and
        ``path``/``directory``/``dir``/``working_dir``/``folder`` to
        ``"location"`` (case-insensitive). Existing standard fields are
        left untouched.

        Args:
            function_data: The command dict whose field names should be
                normalized. Non-dict input is returned unchanged.

        Returns:
            A copy of ``function_data`` with standard field names added
            where a recognized alias was found.
        """
        if not isinstance(function_data, dict):
            return function_data

        # Field name mappings (case-insensitive)
        field_mappings = {
            "command": ["command", "cmd", "action", "tool", "call", "execute", "run"],
            "execution": [
                "execution",
                "executor",
                "type",
                "method",
                "shell",
                "runtime",
            ],
            "location": [
                "location",
                "path",
                "directory",
                "dir",
                "working_dir",
                "working_directory",
                "folder",
            ],
        }

        normalized = function_data.copy()

        # Normalize each field
        for standard_field, possible_names in field_mappings.items():
            if standard_field not in normalized:
                for possible_name in possible_names:
                    # Case-insensitive search
                    for key, value in function_data.items():
                        if key.lower() == possible_name.lower():
                            normalized[standard_field] = value
                            break
                    if standard_field in normalized:
                        break

        return normalized

    def _format_with_location(self, command: str, location: str) -> str:
        """Prefix a command with a directory change when a location is given.

        'location' representa directório de execução.
        Para garantir portabilidade, prefixamos 'cd <dir> && <command>' quando location existir.

        Args:
            command: The command string to run.
            location: The working directory the command should run in, or
                empty/``None`` if not applicable.

        Returns:
            str: ``command`` unchanged if ``location`` is empty or ``command``
            already starts with ``"cd "``, otherwise ``"cd <location> && <command>"``.
        """
        location = (location or "").strip()
        if not location:
            return command.strip()
        # Evita duplicar se já vier com 'cd'
        if command.strip().startswith("cd "):
            return command.strip()
        return f"cd {location} && {command}".strip()

    # Return all functions and executions for each one
    def check_all_functions(self, response):
        """Normalize and validate all command JSON strings in a response.

        Iterates over the JSON strings/dicts in ``response``, normalizes
        each one's structure and field names, fills in default
        ``execution`` (``"shell"``) and ``location`` (``""``) values when
        missing, and splits ``&&``-joined command sequences into separate
        entries. Items missing the ``command`` field (after defaulting) are
        skipped and logged as errors.

        Args:
            response: Either a list of JSON strings/dicts, or a dict
                containing a ``"json_strings"`` key with such a list.

        Returns:
            list: A list of normalized command dicts, each with
            ``"command"``, ``"execution"``, ``"location"``, ``"raw_data"``,
            and ``"is_sequence"`` keys. Returns an empty list on error or if
            the response format is unrecognized.
        """
        try:
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response content: {response}")

            if isinstance(response, list):
                json_strings = response
            elif isinstance(response, dict) and "json_strings" in response:
                json_strings = response["json_strings"]
            else:
                logger.error(f"Unexpected response format: {type(response)}")
                return []

            commands_list = []
            logger.debug(f"Found {len(json_strings)} JSON strings")

            for i, json_str in enumerate(json_strings):
                logger.debug(f"Processing JSON string {i}: {json_str}")
                try:
                    function_data = (
                        json_str if isinstance(json_str, dict) else json.loads(json_str)
                    )
                    function_data = self._normalize_function_object(function_data)
                    function_data = self._normalize_field_names(function_data)

                    # Check for required fields with more flexible validation
                    required_fields = ["command", "execution", "location"]
                    missing_fields = []

                    for field in required_fields:
                        if field not in function_data or not function_data[field]:
                            missing_fields.append(field)

                    # If missing fields, try to infer defaults
                    if missing_fields:
                        logger.debug(
                            f"Missing fields {missing_fields}, attempting to infer defaults"
                        )

                        # Default execution to 'shell' if missing
                        if "execution" in missing_fields:
                            function_data["execution"] = "shell"
                            missing_fields.remove("execution")
                            logger.debug("Defaulted execution to 'shell'")

                        # Default location to current directory if missing
                        if "location" in missing_fields:
                            function_data["location"] = ""
                            missing_fields.remove("location")
                            logger.debug("Defaulted location to current directory")

                    if not missing_fields:
                        command = str(function_data.get("command", ""))
                        location = str(function_data.get("location", ""))
                        execution = str(function_data.get("execution", "")).lower()

                        logger.debug(
                            f"Extracted - command: '{command}', location: '{location}', execution: '{execution}'"
                        )

                        formatted_command = self._format_with_location(
                            command, location
                        )

                        logger.info(
                            f"Processed {execution} command: {formatted_command}"
                        )

                        # Permite sequências '&&' como comandos separados
                        command_sequence = [
                            c.strip()
                            for c in formatted_command.split("&&")
                            if c.strip()
                        ]

                        if len(command_sequence) > 1:
                            for seq_cmd in command_sequence:
                                commands_list.append(
                                    {
                                        "command": seq_cmd,
                                        "execution": execution,
                                        "location": location,
                                        "raw_data": function_data,
                                        "is_sequence": True,
                                    }
                                )
                        else:
                            commands_list.append(
                                {
                                    "command": formatted_command,
                                    "execution": execution,
                                    "location": location,
                                    "raw_data": function_data,
                                    "is_sequence": False,
                                }
                            )
                    else:
                        logger.error(
                            f"Missing mandatory fields {missing_fields}: {json_str}"
                        )

                except json.JSONDecodeError as json_e:
                    logger.error(f"JSON parsing error for item {i}: {json_e}")
                    continue
                except Exception as item_e:
                    logger.error(f"Error processing item {i}: {item_e}")
                    continue

            logger.debug(f"Final commands_list: {commands_list}")
            logger.info(f"Successfully processed {len(commands_list)} commands")
            return commands_list

        except Exception as e:
            logger.error(f"Error processing functions: {e}")
            import traceback

            traceback.print_exc()
            return []

    def get_function_command(self, functions):
        """Devolve lista de strings de comando executáveis (shell).

        Args:
            functions: List of normalized command dicts (as returned by
                :meth:`check_all_functions`), each expected to have a
                ``"command"`` key and optionally ``"execution"``.

        Returns:
            list: A list of executable command strings (empty list on
            error or if ``functions`` is falsy).
        """
        try:
            commands_list = []
            if functions:
                for function in functions:
                    execution_type = (function.get("execution") or "").lower()
                    command = function.get("command") or ""
                    # Hoje não adicionamos prefixos específicos (ros/ros2/python),
                    # o comando já vem completo; manter simples/robusto.
                    formatted_command = command.strip()
                    logger.info(f"Processed command: {formatted_command}")
                    commands_list.append(formatted_command)
            return commands_list
        except Exception as e:
            logger.error(f"Error formatting commands: {e}")
            return []

    def parse_command_output(self, output):
        """Format a command's output as a human-readable, unescaped string.

        Serializes ``output`` to indented JSON, then decodes unicode
        escape sequences (e.g. turning literal ``\\n`` into real newlines)
        and strips a surrounding pair of quotes if present.

        Args:
            output: The value to format (typically a command's raw output,
                which may be a string, dict, list, etc.).

        Returns:
            str: The formatted, unescaped output string.
        """
        json_string = json.dumps(output, indent=2)
        formatted_output = json_string.encode("utf-8").decode("unicode_escape")
        if formatted_output.startswith('"') and formatted_output.endswith('"'):
            formatted_output = formatted_output[1:-1]
        return formatted_output
