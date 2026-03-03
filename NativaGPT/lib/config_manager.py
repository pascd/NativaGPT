import json
import os
import glob
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
from pathlib import Path


class ConfigManager:
    """
    ConfigManager v2.0 - Performance Optimized
    - Loads main configuration
    - Loads tools from database folder
    - Caches tools for performance
    - Thread-safe operations
    - Parallel JSON file loading
    - Hot reload capability
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_json: Optional[Dict[str, Any]] = None
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._lock = threading.RLock()  # Reentrant lock
        self.executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="config_mgr"
        )

        # Load config on initialization
        self.config_json = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load main configuration file."""
        try:
            # If config_path is already a dict, return it directly
            if isinstance(self.config_path, dict):
                return self.config_path
            # Otherwise
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")

    def get(self) -> Dict[str, Any]:
        """Get the main configuration."""
        if self.config_json is None:
            with self._lock:
                if self.config_json is None:
                    self.config_json = self._load_config()
        return self.config_json

    def reload(self) -> Dict[str, Any]:
        """Reload configuration from file."""
        with self._lock:
            self.config_json = self._load_config()
            self._tools_cache = None  # Invalidate tools cache
        return self.config_json

    # ---------------------- Tools Loading ----------------------

    def get_tools_json(self) -> List[Dict[str, Any]]:
        """
        Load all JSON files from database folder and treat each object as a tool.
        Results are cached for performance.
        """
        # Return cached tools if available
        if self._tools_cache is not None:
            return self._tools_cache

        with self._lock:
            # Double-check after acquiring lock
            if self._tools_cache is not None:
                return self._tools_cache

            # Load tools
            tools = self._load_tools_from_database()
            self._tools_cache = tools
            return tools

    def _load_tools_from_database(self) -> List[Dict[str, Any]]:
        """Internal method to load tools from database folder."""
        if self.config_json is None:
            return []
        database_folder = self.config_json.get("nativa_gpt", {}).get(
            "database_folder", ""
        )

        if not database_folder:
            print("WARNING: No database_folder configured")
            return []

        # Expand paths (handle ~, environment variables)
        database_folder = os.path.expanduser(os.path.expandvars(database_folder))

        if not os.path.exists(database_folder):
            print(f"WARNING: Database folder not found: {database_folder}")
            return []

        try:
            # Find all JSON files recursively
            json_files = self._find_json_files(database_folder)

            if not json_files:
                print(f"WARNING: No JSON files found in {database_folder}")
                return []

            print(f"Loading tools from {len(json_files)} JSON file(s)...")

            # Load files (parallel if many files)
            if len(json_files) > 5:
                tools = self._load_tools_parallel(json_files)
            else:
                tools = self._load_tools_sequential(json_files)

            print(f"Loaded {len(tools)} tool(s) from database")
            return tools

        except Exception as e:
            print(f"ERROR loading tools from database: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _find_json_files(self, folder: str, recursive: bool = True) -> List[str]:
        """Find all JSON files in folder."""
        if recursive:
            # Recursive search
            pattern = os.path.join(folder, "**", "*.json")
            return glob.glob(pattern, recursive=True)
        else:
            # Non-recursive search
            pattern = os.path.join(folder, "*.json")
            return glob.glob(pattern)

    def _load_tools_sequential(self, json_files: List[str]) -> List[Dict[str, Any]]:
        """Load JSON files sequentially."""
        tools = []

        for json_file in json_files:
            try:
                file_tools = self._load_single_json_file(json_file)
                tools.extend(file_tools)
            except Exception as e:
                print(f"ERROR loading {json_file}: {e}")

        return tools

    def _load_tools_parallel(self, json_files: List[str]) -> List[Dict[str, Any]]:
        """Load JSON files in parallel for better performance."""
        tools = []

        futures = [
            self.executor.submit(self._load_single_json_file, f) for f in json_files
        ]

        for future in as_completed(futures):
            try:
                file_tools = future.result()
                tools.extend(file_tools)
            except Exception as e:
                print(f"ERROR loading JSON file: {e}")

        return tools

    def _load_single_json_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load a single JSON file and return list of tool objects.
        Handles both single objects and arrays.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Add metadata
            source_file = os.path.basename(filepath)

            # Handle config_path being a dict vs file path
            if isinstance(self.config_path, dict):
                rel_path = filepath  # Just use full path if config is a dict
            else:
                rel_path = os.path.relpath(
                    filepath, start=os.path.dirname(self.config_path)
                )

            # Handle array of tools
            if isinstance(data, list):
                tools = []
                for item in data:
                    if isinstance(item, dict):
                        item["_source_file"] = source_file
                        item["_source_path"] = rel_path
                        tools.append(item)
                return tools

            # Handle single tool object
            elif isinstance(data, dict):
                data["_source_file"] = source_file
                data["_source_path"] = rel_path
                return [data]

            else:
                print(f"WARNING: Unexpected JSON structure in {filepath}")
                return []

        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in {filepath}: {e}")
            return []
        except Exception as e:
            print(f"ERROR reading {filepath}: {e}")
            return []

    def validate_tool(self, tool: Dict[str, Any]) -> bool:
        """
        Validate that a tool object has the minimum required fields.
        """
        required_fields = ["name", "command"]

        for field in required_fields:
            if field not in tool:
                source = tool.get("_source_file", "unknown")
                print(f"WARNING: Tool missing required field '{field}' in {source}")
                return False

        return True

    def normalize_tool(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize tool object to standard format.
        Ensures all tools have consistent structure.
        """
        normalized = {
            "name": tool.get("name", "unnamed_tool"),
            "command": tool.get("command", ""),
            "description": tool.get("description", tool.get("desc", "No description")),
            "execution": tool.get("execution", "shell"),
            "location": tool.get("location", ""),
            "parameters": tool.get("parameters", tool.get("params", {})),
            "examples": tool.get("examples", []),
            "category": tool.get("category", "Custom Tools"),
            "tags": tool.get("tags", []),
            "_source_file": tool.get("_source_file", "unknown"),
            "_source_path": tool.get("_source_path", "unknown"),
        }

        return normalized

    def get_validated_tools(self) -> List[Dict[str, Any]]:
        """
        Get all tools from database, validated and normalized.
        Returns only valid tools in standard format.
        """
        raw_tools = self.get_tools_json()

        validated_tools = []
        for tool in raw_tools:
            if self.validate_tool(tool):
                normalized = self.normalize_tool(tool)
                validated_tools.append(normalized)

        invalid_count = len(raw_tools) - len(validated_tools)
        if invalid_count > 0:
            print(f"INFO: {invalid_count} invalid tool(s) filtered out")

        return validated_tools

    def get_tools_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get tools grouped by category.
        Returns: {category_name: [tool1, tool2, ...]}
        """
        tools = self.get_validated_tools()

        categorized = {}
        for tool in tools:
            category = tool.get("category", "Uncategorized")
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(tool)

        return categorized

    def get_tool_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Find a tool by its name.
        Returns None if not found.
        """
        tools = self.get_validated_tools()

        for tool in tools:
            if tool.get("name") == name:
                return tool

        return None

    def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """
        Search tools by name, description, or tags.
        Case-insensitive search.
        """
        query_lower = query.lower()
        tools = self.get_validated_tools()

        matching_tools = []
        for tool in tools:
            # Check name
            if query_lower in tool.get("name", "").lower():
                matching_tools.append(tool)
                continue

            # Check description
            if query_lower in tool.get("description", "").lower():
                matching_tools.append(tool)
                continue

            # Check tags
            tags = tool.get("tags", [])
            if any(query_lower in tag.lower() for tag in tags):
                matching_tools.append(tool)
                continue

        return matching_tools

    def reload_tools(self) -> List[Dict[str, Any]]:
        """
        Reload tools from database folder.
        Invalidates cache and loads fresh data.
        """
        with self._lock:
            self._tools_cache = None
            return self.get_tools_json()
