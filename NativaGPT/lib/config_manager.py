"""Configuration and tool-database loading for NativaGPT.

Provides :class:`ConfigManager`, which loads the main JSON application
configuration (resolving ``${REPO_ROOT}`` path placeholders) and the
collection of "tool" JSON files that describe the functions/commands
exposed to the LLM, with validation, normalization, caching, and search
helpers on top.
"""

import json
import os
import glob
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional
from pathlib import Path

# Absolute path to the repository root (the directory containing this
# package's outer "NativaGPT/" folder, "config/", "README.md", etc).
# Computed once from this file's own location so config values can refer to
# "${REPO_ROOT}" instead of a hardcoded, developer-specific absolute path.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _substitute_repo_root(value: Any) -> Any:
    """Recursively replace the literal token "${REPO_ROOT}" in config values.

    Walks dicts/lists/strings loaded from a config JSON file and replaces
    every occurrence of the placeholder "${REPO_ROOT}" with the actual
    repository root path, so config files can ship portable, relative-style
    paths (e.g. "${REPO_ROOT}/config/functions") instead of a contributor's
    personal absolute path.

    Args:
        value: A JSON-decoded value (dict, list, str, or other scalar).

    Returns:
        The same structure with every "${REPO_ROOT}" substring replaced.
    """
    if isinstance(value, str):
        return value.replace("${REPO_ROOT}", str(REPO_ROOT))
    if isinstance(value, dict):
        return {k: _substitute_repo_root(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_repo_root(v) for v in value]
    return value


class ConfigManager:
    """Loads, validates, and caches the application configuration and tool database.

    Reads the main JSON config file (or accepts an already-parsed dict),
    resolving any "${REPO_ROOT}" placeholders in string values, and
    separately loads the "tool" JSON files referenced by
    ``nativa_gpt.database_folder`` (the function/command definitions exposed
    to the LLM). Tool loading is parallelized for folders with many files
    and the results are cached until ``reload_tools()`` is called.

    All public methods are safe to call from multiple threads.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config_json: Optional[Dict[str, Any]] = None
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._lock = threading.RLock()  # Reentrant lock
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="config_mgr")

        # Load config on initialization
        self.config_json = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load the main configuration file and resolve "${REPO_ROOT}" placeholders."""
        try:
            # If config_path is already a dict, return it directly (still
            # resolving placeholders, in case it was built programmatically).
            if isinstance(self.config_path, dict):
                return _substitute_repo_root(self.config_path)
            # Otherwise, load it from disk.
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return _substitute_repo_root(config)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")

    def get(self) -> Dict[str, Any]:
        """Returns the main configuration, loading it first if necessary.

        Thread-safe: if the configuration hasn't been loaded yet (or was
        cleared), it is loaded under `self._lock` before returning.

        Returns:
            Dict[str, Any]: The parsed configuration dict, with
            ``${REPO_ROOT}`` placeholders already resolved.
        """
        if self.config_json is None:
            with self._lock:
                if self.config_json is None:
                    self.config_json = self._load_config()
        return self.config_json

    def reload(self) -> Dict[str, Any]:
        """Reloads the configuration from disk (or the original dict) and invalidates the tools cache.

        Returns:
            Dict[str, Any]: The freshly loaded configuration dict.

        Raises:
            FileNotFoundError: If `self.config_path` is a file path that
                does not exist.
            ValueError: If the configuration file contains invalid JSON.
            RuntimeError: For any other error encountered while loading.
        """
        with self._lock:
            self.config_json = self._load_config()
            self._tools_cache = None  # Invalidate tools cache
        return self.config_json

    # ---------------------- Tools Loading ----------------------

    def get_tools_json(self) -> List[Dict[str, Any]]:
        """Returns the raw (unvalidated) tool objects loaded from the database folder.

        Results are cached in `self._tools_cache` after the first call;
        subsequent calls return the cached list until `reload_tools()` (or
        `reload()`) is called. Loading is thread-safe via a double-checked
        lock.

        Returns:
            List[Dict[str, Any]]: One dict per tool object found across
            all JSON files in the configured database folder, each
            tagged with `_source_file` and `_source_path`. Empty if the
            database folder is not configured or not found.
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
        """Finds and loads all tool JSON files under ``nativa_gpt.database_folder``.

        Expands ``~`` and environment variables in the configured folder
        path, then dispatches to `_load_tools_parallel` (more than 5
        files) or `_load_tools_sequential`. Logs a warning and returns an
        empty list if the folder isn't configured, doesn't exist, or
        contains no JSON files; logs and swallows any other error.

        Returns:
            List[Dict[str, Any]]: All tool objects found, or an empty
            list on failure.
        """
        database_folder = self.config_json.get("nativa_gpt", {}).get("database_folder", "")

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
        """Globs for `*.json` files under `folder`, recursively by default."""
        if recursive:
            # Recursive search
            pattern = os.path.join(folder, "**", "*.json")
            return glob.glob(pattern, recursive=True)
        else:
            # Non-recursive search
            pattern = os.path.join(folder, "*.json")
            return glob.glob(pattern)

    def _load_tools_sequential(self, json_files: List[str]) -> List[Dict[str, Any]]:
        """Loads `json_files` one at a time via `_load_single_json_file`, skipping files that error."""
        tools = []

        for json_file in json_files:
            try:
                file_tools = self._load_single_json_file(json_file)
                tools.extend(file_tools)
            except Exception as e:
                print(f"ERROR loading {json_file}: {e}")

        return tools

    def _load_tools_parallel(self, json_files: List[str]) -> List[Dict[str, Any]]:
        """Loads `json_files` concurrently via `self.executor`, skipping files that error."""
        tools = []

        futures = [self.executor.submit(self._load_single_json_file, f) for f in json_files]

        for future in as_completed(futures):
            try:
                file_tools = future.result()
                tools.extend(file_tools)
            except Exception as e:
                print(f"ERROR loading JSON file: {e}")

        return tools

    def _load_single_json_file(self, filepath: str) -> List[Dict[str, Any]]:
        """Loads one JSON file and returns its contents as a list of tool dicts.

        Handles both a top-level JSON array (one tool per element) and a
        single top-level JSON object (treated as one tool). Each returned
        tool dict gains `_source_file` (basename) and `_source_path`
        (path relative to the config file's directory, or the raw path
        if the config was supplied as a dict) keys. Invalid JSON, read
        errors, or an unexpected top-level type are logged and result in
        an empty list rather than raising.

        Returns:
            List[Dict[str, Any]]: The tool object(s) found in the file,
            or an empty list on error / unexpected structure.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Add metadata
            source_file = os.path.basename(filepath)

            # Handle config_path being a dict vs file path
            if isinstance(self.config_path, dict):
                rel_path = filepath  # Just use full path if config is a dict
            else:
                rel_path = os.path.relpath(filepath, start=os.path.dirname(self.config_path))

            # Handle array of tools
            if isinstance(data, list):
                tools = []
                for item in data:
                    if isinstance(item, dict):
                        item['_source_file'] = source_file
                        item['_source_path'] = rel_path
                        tools.append(item)
                return tools

            # Handle single tool object
            elif isinstance(data, dict):
                data['_source_file'] = source_file
                data['_source_path'] = rel_path
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
        """Checks that a tool object has the minimum required fields (`name`, `command`).

        Logs a warning (including the tool's `_source_file`, if any) for
        the first missing required field found.

        Args:
            tool: A raw tool dict as returned by `get_tools_json`.

        Returns:
            bool: True if all required fields are present, False
            otherwise.
        """
        required_fields = ['name', 'command']

        for field in required_fields:
            if field not in tool:
                source = tool.get('_source_file', 'unknown')
                print(f"WARNING: Tool missing required field '{field}' in {source}")
                return False

        return True

    def normalize_tool(self, tool: Dict[str, Any]) -> Dict[str, Any]:
        """Normalizes a raw tool dict into a consistent standard shape.

        Fills in defaults for missing fields (e.g. falls back from
        `desc`/`params` to `description`/`parameters`) so downstream code
        can rely on a fixed set of keys regardless of how the source JSON
        was authored.

        Args:
            tool: A raw tool dict, typically one that has already passed
                `validate_tool`.

        Returns:
            Dict[str, Any]: A new dict with exactly the keys `name`,
            `command`, `description`, `execution`, `location`,
            `parameters`, `examples`, `category`, `tags`,
            `_source_file`, and `_source_path`.
        """
        normalized = {
            'name': tool.get('name', 'unnamed_tool'),
            'command': tool.get('command', ''),
            'description': tool.get('description', tool.get('desc', 'No description')),
            'execution': tool.get('execution', 'shell'),
            'location': tool.get('location', ''),
            'parameters': tool.get('parameters', tool.get('params', {})),
            'examples': tool.get('examples', []),
            'category': tool.get('category', 'Custom Tools'),
            'tags': tool.get('tags', []),
            '_source_file': tool.get('_source_file', 'unknown'),
            '_source_path': tool.get('_source_path', 'unknown')
        }

        return normalized

    def get_validated_tools(self) -> List[Dict[str, Any]]:
        """Returns all cached tools, filtered by `validate_tool` and normalized by `normalize_tool`.

        Logs an info message with the count of tools filtered out for
        failing validation, if any.

        Returns:
            List[Dict[str, Any]]: Normalized tool dicts for every raw
            tool that passed validation.
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
        """Groups validated, normalized tools by their `category` field.

        Returns:
            Dict[str, List[Dict[str, Any]]]: A mapping from category name
            (defaulting to `"Uncategorized"` when unset) to the list of
            tool dicts in that category.
        """
        tools = self.get_validated_tools()

        categorized = {}
        for tool in tools:
            category = tool.get('category', 'Uncategorized')
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(tool)

        return categorized

    def get_tool_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Finds a validated, normalized tool by its exact `name`.

        Args:
            name: The exact tool name to look up (case-sensitive).

        Returns:
            Optional[Dict[str, Any]]: The matching normalized tool dict,
            or `None` if no tool with that name exists.
        """
        tools = self.get_validated_tools()

        for tool in tools:
            if tool.get('name') == name:
                return tool

        return None

    def search_tools(self, query: str) -> List[Dict[str, Any]]:
        """Searches validated tools whose name, description, or tags contain `query`.

        Matching is case-insensitive; a tool is included at most once
        even if `query` matches multiple fields.

        Args:
            query: The substring to search for.

        Returns:
            List[Dict[str, Any]]: Normalized tool dicts matching the
            query, in their original order.
        """
        query_lower = query.lower()
        tools = self.get_validated_tools()

        matching_tools = []
        for tool in tools:
            # Check name
            if query_lower in tool.get('name', '').lower():
                matching_tools.append(tool)
                continue

            # Check description
            if query_lower in tool.get('description', '').lower():
                matching_tools.append(tool)
                continue

            # Check tags
            tags = tool.get('tags', [])
            if any(query_lower in tag.lower() for tag in tags):
                matching_tools.append(tool)
                continue

        return matching_tools

    def reload_tools(self) -> List[Dict[str, Any]]:
        """Invalidates the tools cache and reloads raw tool data from the database folder.

        Returns:
            List[Dict[str, Any]]: The freshly loaded raw tool list (same
            shape as `get_tools_json`).
        """
        with self._lock:
            self._tools_cache = None
            return self.get_tools_json()