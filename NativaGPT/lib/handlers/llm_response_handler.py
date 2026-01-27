import os
import json
import sys
import re

from NativaGPT.lib.coloring_logger import logger


class LLMResponseHandler:
    def __init__(self):
        logger.info("Initializing LLM Response Handler module.")

    def extract_json_str(self, llm_response):
        """
        Enhanced function to extract JSON strings from LLM responses.
        Handles multiple formats including code blocks, malformed JSON, and nested structures.
        :param llm_response: is the response from an LLM request. Accepts JSON strings and normal strings containing JSON strings.
        :return: returns the json string of the response and the text in seperated variables.
        """
        # First, sanitize the response to remove think tags
        sanitized_response = self.sanitize_response_text(llm_response)

        # Enhanced JSON extraction with multiple patterns
        json_strings = []
        positions = []

        # Method 1: Extract JSON from code blocks (```json...``` or ```...```)
        code_block_pattern = r"```(?:json)?\s*(\{[^}]*\}|\[[^\]]*\])\s*```"
        code_matches = re.finditer(
            code_block_pattern, sanitized_response, re.IGNORECASE | re.DOTALL
        )
        for match in code_matches:
            json_content = match.group(1).strip()
            if self._validate_and_add_json(
                json_content, json_strings, positions, match.start(), match.end()
            ):
                logger.debug(f"Found JSON in code block: {json_content[:100]}...")

        # Method 2: Enhanced pattern for JSON objects and arrays (handles nested structures better)
        json_pattern = r"(?:^|\s|\n)(\{[^}]*\}|\[[^\]]*\])(?:$|\s|\n|[,.!?])"
        json_matches = re.finditer(
            json_pattern, sanitized_response, re.MULTILINE | re.DOTALL
        )

        for match in json_matches:
            # Extract the JSON part (group 1)
            raw_json = match.group(1).strip()

            # Skip if this JSON was already found in a code block
            already_found = False
            for existing_json in json_strings:
                if raw_json == existing_json:
                    already_found = True
                    break

            if not already_found and self._validate_and_add_json(
                raw_json, json_strings, positions, match.start(), match.end()
            ):
                logger.debug(f"Found JSON with enhanced pattern: {raw_json[:100]}...")

        # Method 3: Fallback - look for JSON-like structures more broadly
        if not json_strings:
            broad_pattern = r"(\{[^}]*\}|\[[^\]]*\])"
            broad_matches = re.finditer(broad_pattern, sanitized_response)
            for match in broad_matches:
                raw_json = match.group().strip()
                if self._validate_and_add_json(
                    raw_json, json_strings, positions, match.start(), match.end()
                ):
                    logger.debug(f"Found JSON with broad pattern: {raw_json[:100]}...")

        # Method 4: Try to parse the entire response as JSON if no JSON was found
        if not json_strings:
            try:
                json.loads(sanitized_response.strip())
                json_strings.append(sanitized_response.strip())
                positions.append((0, len(sanitized_response)))
                logger.debug("Entire response is valid JSON")
            except json.JSONDecodeError:
                pass

        # Extract regular text by removing JSON portions
        text_parts = []
        last_end = 0

        # Sort positions by start index
        sorted_positions = sorted(positions, key=lambda x: x[0])

        for start, end in sorted_positions:
            # Add text before the JSON
            if start > last_end:
                text = sanitized_response[last_end:start].strip()
                if text:
                    text_parts.append(text)
            last_end = end

        # Check if there is text after the JSON strings
        if last_end < len(sanitized_response):
            text = sanitized_response[last_end:].strip()
            if text:
                text_parts.append(text)

        logger.info(
            f"Extracted {len(json_strings)} JSON strings and {len(text_parts)} text parts"
        )

        return {
            "json_strings": json_strings,
            "text_content": " ".join(text_parts),
        }

    def _validate_and_add_json(self, json_str, json_strings, positions, start, end):
        """
        Helper method to validate JSON and add to results if valid.
        Returns True if JSON was added, False otherwise.
        """
        try:
            # Parse to validate
            parsed = json.loads(json_str)

            # Additional validation: check if it looks like a command structure
            if isinstance(parsed, dict):
                # Check for common command fields (case-insensitive)
                command_fields = ["command", "action", "function", "tool", "call"]
                has_command_field = any(
                    field.lower() in [k.lower() for k in parsed.keys()]
                    for field in command_fields
                )

                # If it has command-like structure, prioritize it
                if has_command_field:
                    # Insert at the beginning to prioritize command JSON
                    json_strings.insert(0, json_str)
                    positions.insert(0, (start, end))
                else:
                    json_strings.append(json_str)
                    positions.append((start, end))
            else:
                json_strings.append(json_str)
                positions.append((start, end))

            return True
        except json.JSONDecodeError as e:
            logger.debug(f"Invalid JSON found: {json_str[:50]}... Error: {e}")
            return False

    def llm_json_parser(self, llm_response):
        """
        Function to extract the commands and JSON strings from the LLM response.
        :param llm_response: is the response from an LLM request. Accepts JSON strings only.
        :return: returns the JSON strings in the response.
        """

        # Handle both OpenAI and Ollama response formats
        try:
            # Try OpenAI format first
            if "choices" in llm_response and llm_response["choices"]:
                response_content = llm_response["choices"][0]["message"]["content"]
            # Try Ollama format
            elif "response" in llm_response:
                response_content = llm_response["response"]
            else:
                # Fallback to webgpthandler
                response_content = llm_response

        except (KeyError, IndexError) as e:
            logger.error(f"Could not extract content from LLM response: {e}")
            logger.error(f"Response structure: {list(llm_response.keys())}")
            return {"error": "Invalid response format"}

        return self.extract_json_str(response_content)

    def sanitize_response_text(self, text):
        """
        Enhanced sanitization to remove all thinking patterns and unwanted content.
        """
        if not text:
            return text

        # Remove think tags and their content (case insensitive, handles variations)
        patterns_to_remove = [
            r"<think\s*>.*?</think\s*>",  # Standard think tags
            r"<thinking\s*>.*?</thinking\s*>",  # Thinking tags
            r"<thought\s*>.*?</thought\s*>",  # Thought tags
            r"<analysis\s*>.*?</analysis\s*>",  # Analysis tags
            r"<reasoning\s*>.*?</reasoning\s*>",  # Reasoning tags
            r"\*\*Think:\*\*.*?(?=\n\n|\n[A-Z]|$)",  # Markdown style thinking
            r"\*Think:.*?(?=\n\n|\n[A-Z]|$)",  # Simple think markers
            r"Let me think.*?(?=\n\n|\n[A-Z]|$)",  # Natural thinking expressions
        ]

        cleaned_text = text
        for pattern in patterns_to_remove:
            cleaned_text = re.sub(
                pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE
            )

        # Clean up extra whitespace and newlines
        cleaned_text = re.sub(
            r"\n\s*\n\s*\n", "\n\n", cleaned_text
        )  # Multiple newlines to double
        cleaned_text = re.sub(
            r"^\s+|\s+$", "", cleaned_text
        )  # Leading/trailing whitespace

        return cleaned_text

    def extract_image_references(self, text):
        """
        Extract image file paths or URLs from the response text.
        """
        # Pattern to match common image references
        image_patterns = [
            r"!\[.*?\]\((.*?\.(?:jpg|jpeg|png|gif|bmp|webp))\)",  # Markdown images
            r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>',  # HTML img tags
            r"(?:image|screenshot|photo):\s*([^\s]+\.(?:jpg|jpeg|png|gif|bmp|webp))",  # Simple format
            r"file://([^\s]+\.(?:jpg|jpeg|png|gif|bmp|webp))",  # File URLs
        ]

        images = []
        for pattern in image_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                images.append(match.group(1))

        return images

    def validate_command_structure(self, json_string):
        """
        Validate that a JSON command has the required structure.
        """
        try:
            command_data = json.loads(json_string)

            # Check if it's a valid command structure
            if not isinstance(command_data, dict):
                return False, "Command must be a JSON object"

            # Check for required COMMAND field
            if "COMMAND" not in command_data:
                return False, "Missing required 'COMMAND' field"

            # Validate command format
            command = command_data["COMMAND"]
            if not isinstance(command, str) or not command.strip():
                return False, "COMMAND field must be a non-empty string"

            return True, "Valid command structure"

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"

    def extract_command_parameters(self, json_string):
        """
        Extract parameters from a command JSON string.
        """
        try:
            command_data = json.loads(json_string)

            # Extract command and parameters
            command = command_data.get("COMMAND", "")
            location = command_data.get("LOCATION", "")

            # Get all other fields as parameters
            parameters = {
                k: v
                for k, v in command_data.items()
                if k not in ["COMMAND", "LOCATION"]
            }

            return {
                "command": command,
                "location": location,
                "parameters": parameters,
                "raw_data": command_data,
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse command JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error extracting command parameters: {e}")
            return None

    def format_response_for_output(self, command_results, text_content):
        """
        Format the final response combining command results and text content.
        """
        formatted_response = []

        # Add text content if present
        if text_content and text_content.strip():
            formatted_response.append(text_content.strip())

        # Add command results
        for i, result in enumerate(command_results):
            if result.get("success", True):
                # Format successful results
                if "messages" in result:
                    # Topic query results
                    topic = result.get("topic", "Unknown")
                    count = result.get("count", 0)
                    if count > 0:
                        formatted_response.append(
                            f"\n📡 Topic '{topic}' - {count} messages:"
                        )
                        for j, msg in enumerate(
                            result["messages"][-3:], 1
                        ):  # Show last 3 messages
                            timestamp = msg.get("timestamp", "Unknown time")
                            if isinstance(timestamp, str):
                                try:
                                    from datetime import datetime

                                    dt = datetime.fromisoformat(
                                        timestamp.replace("Z", "+00:00")
                                    )
                                    timestamp = dt.strftime("%H:%M:%S")
                                except:
                                    pass
                            data = str(msg.get("data", "No data"))[:100] + (
                                "..." if len(str(msg.get("data", ""))) > 100 else ""
                            )
                            formatted_response.append(f"  {j}. [{timestamp}] {data}")
                    else:
                        formatted_response.append(
                            f"\n📡 Topic '{topic}' - No messages available"
                        )

                elif "available_topics" in result:
                    # Available topics result
                    topics = result["available_topics"]
                    if topics:
                        formatted_response.append(
                            f"\n📋 Available topics ({len(topics)}):"
                        )
                        for topic in topics[:10]:  # Show first 10 topics
                            formatted_response.append(f"  • {topic}")
                        if len(topics) > 10:
                            formatted_response.append(
                                f"  ... and {len(topics) - 10} more"
                            )
                    else:
                        formatted_response.append("\n📋 No topics currently available")

                elif "subscribed" in result:
                    # Subscription result
                    topic = result.get("topic", "Unknown")
                    success = result.get("subscribed", False)
                    if success:
                        formatted_response.append(
                            f"\n✅ Successfully subscribed to topic: {topic}"
                        )
                    else:
                        formatted_response.append(
                            f"\n❌ Failed to subscribe to topic: {topic}"
                        )

                else:
                    # Generic successful result
                    formatted_response.append(
                        f"\n✅ Command {i + 1} executed successfully"
                    )
                    if "message" in result:
                        formatted_response.append(f"   {result['message']}")

            else:
                # Format error results
                error = result.get("error", "Unknown error")
                command = result.get("command", f"Command {i + 1}")
                formatted_response.append(f"\n❌ Error in {command}: {error}")

        return (
            "\n".join(formatted_response)
            if formatted_response
            else "No response generated."
        )

    def process_topic_response(self, topic_result):
        """
        Process and format topic query responses specifically.
        """
        if "error" in topic_result:
            return f"❌ Topic Error: {topic_result['error']}"

        topic_name = topic_result.get("topic", "Unknown")
        query_type = topic_result.get("query_type", "query")

        if query_type == "latest" or query_type == "history":
            messages = topic_result.get("messages", [])
            if not messages:
                return f"📡 No messages found in topic '{topic_name}'"

            response_parts = [
                f"📡 Topic '{topic_name}' ({query_type}) - {len(messages)} messages:"
            ]

            for i, msg in enumerate(messages[-5:], 1):  # Show last 5 messages
                timestamp = msg.get("timestamp", "Unknown")
                if hasattr(timestamp, "strftime"):
                    timestamp = timestamp.strftime("%H:%M:%S")
                elif isinstance(timestamp, str):
                    try:
                        from datetime import datetime

                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        timestamp = dt.strftime("%H:%M:%S")
                    except:
                        timestamp = str(timestamp)[:8]

                data = msg.get("data", {})
                if isinstance(data, dict):
                    # Format dict data nicely
                    key_vals = [f"{k}: {v}" for k, v in list(data.items())[:3]]
                    data_str = ", ".join(key_vals)
                    if len(data) > 3:
                        data_str += f" (+ {len(data) - 3} more fields)"
                else:
                    data_str = str(data)[:100] + ("..." if len(str(data)) > 100 else "")

                response_parts.append(f"  {i}. [{timestamp}] {data_str}")

            return "\n".join(response_parts)

        elif query_type == "wait_for_new":
            if "message" in topic_result:
                msg = topic_result["message"]
                timestamp = msg.get("timestamp", "now")
                if hasattr(timestamp, "strftime"):
                    timestamp = timestamp.strftime("%H:%M:%S")

                data = msg.get("data", "No data")
                return f"📡 New message on '{topic_name}' at {timestamp}: {data}"
            else:
                return f"⏰ Timeout waiting for new message on '{topic_name}'"

        else:
            return f"📡 Topic '{topic_name}' query completed: {query_type}"


if __name__ == "__main__":
    # Load JSON data from file correctly
    with open("../../tests/test.json", "r", encoding="utf-8") as file:
        response = json.load(file)

    module = LLMResponseHandler()
    response = module.llm_json_parser(response)

    for json_obj in response["json_strings"]:
        # Process each JSON object and extract commands
        logger.info(json_obj)
        logger.info(response["text_content"])
