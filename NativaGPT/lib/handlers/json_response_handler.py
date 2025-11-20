import os
import json
import sys

from NativaGPT.lib.coloring_logger import logger

class JsonResponseHandler:

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
        "other": ""
    }

    def __init__(self):
        logger.info("Created JsonResponseHandler instance.")

    def _normalize_function_object(self, fn_obj):
        """
        Aceita tanto estruturas 'planas' como aninhadas:
          {'command': '...', 'execution': 'shell', 'location': '/tmp'}
          {'function': {'command': '...', ...}}
        """
        if isinstance(fn_obj, dict) and "function" in fn_obj and isinstance(fn_obj["function"], dict):
            return fn_obj["function"]
        return fn_obj

    def _format_with_location(self, command: str, location: str) -> str:
        """
        'location' representa directório de execução.
        Para garantir portabilidade, prefixamos 'cd <dir> && <command>' quando location existir.
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
        try:
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response content: {response}")

            if isinstance(response, list):
                json_strings = response
            elif isinstance(response, dict) and 'json_strings' in response:
                json_strings = response['json_strings']
            else:
                logger.error(f"Unexpected response format: {type(response)}")
                return []

            commands_list = []
            logger.debug(f"Found {len(json_strings)} JSON strings")

            for i, json_str in enumerate(json_strings):
                logger.debug(f"Processing JSON string {i}: {json_str}")
                try:
                    function_data = json_str if isinstance(json_str, dict) else json.loads(json_str)
                    function_data = self._normalize_function_object(function_data)

                    required_fields = ['command', 'execution', 'location']
                    missing_fields = [field for field in required_fields if field not in function_data]

                    if not missing_fields:
                        command = str(function_data.get('command', ''))
                        location = str(function_data.get('location', ''))
                        execution = str(function_data.get('execution', '')).lower()

                        logger.debug(f"Extracted - command: '{command}', location: '{location}', execution: '{execution}'")

                        formatted_command = self._format_with_location(command, location)

                        logger.info(f"Processed {execution} command: {formatted_command}")

                        # Permite sequências '&&' como comandos separados
                        command_sequence = [c.strip() for c in formatted_command.split("&&") if c.strip()]

                        if len(command_sequence) > 1:
                            for seq_cmd in command_sequence:
                                commands_list.append({
                                    'command': seq_cmd,
                                    'execution': execution,
                                    'location': location,
                                    'raw_data': function_data,
                                    'is_sequence': True
                                })
                        else:
                            commands_list.append({
                                'command': formatted_command,
                                'execution': execution,
                                'location': location,
                                'raw_data': function_data,
                                'is_sequence': False
                            })
                    else:
                        logger.error(f"Missing mandatory fields {missing_fields}: {json_str}")

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
        """Devolve lista de strings de comando executáveis (shell)."""
        try:
            commands_list = []
            if functions:
                for function in functions:
                    execution_type = (function.get('execution') or '').lower()
                    command = function.get('command') or ''
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
        json_string = json.dumps(output, indent=2)
        formatted_output = json_string.encode('utf-8').decode('unicode_escape')
        if formatted_output.startswith('"') and formatted_output.endswith('"'):
            formatted_output = formatted_output[1:-1]
        return formatted_output