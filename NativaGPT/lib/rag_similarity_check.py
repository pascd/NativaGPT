"""RAG (retrieval-augmented) similarity search over the tool/function database.

Provides :class:`RAGSimilarityCheck`, which loads JSON/text entries from a
configured "functions" folder, embeds them via a local Ollama embedding
model, and retrieves the most semantically similar entries to a query
using cosine similarity. Also provides `convert_json_to_toon`, a helper
that renders a "function" JSON tool definition as a compact,
LLM-friendly Token Object Oriented Notation (TOON) string.
"""

import os
import sys
import json
import ollama
import pathlib
from typing import List, Tuple, Union, Any

# Assuming these are in your project structure
from NativaGPT.lib.config_manager import ConfigManager
from NativaGPT.lib.coloring_logger import logger

class RAGSimilarityCheck():
    """Embeds and searches a local database of function/tool JSON definitions.

    On construction, ensures the configured Ollama embedding model is
    available locally (downloading it if necessary), loads every file
    under `database_folder` (JSON files are parsed into one or more
    entries; other files are read as newline-separated text entries),
    and computes an embedding for each entry to build an in-memory
    vector database. Queries are answered by embedding the query text
    and ranking stored entries by cosine similarity.
    """

    def __init__(self, config):
        """Initializes the RAG database: ensures the embedding model is available and builds the index.

        Args:
            config: The application configuration dict (as returned by
                `ConfigManager.get()`); must contain
                `config["nativa_gpt"]["embedding_model"]`.
        """
        self.config = config
        self.loaded_entries = []
        self.vector_db = []
        self.database_folder = str(pathlib.Path(__file__).parent.parent.parent / "config" / "functions")
        self.embedding_model = self.config["nativa_gpt"]["embedding_model"]

        # Ensure the embedding model is available
        self._ensure_model_available()

        if self.read_database_files():
            self.add_chunks_to_database()

    def _ensure_model_available(self):
        """Check if the embedding model exists, download if not."""
        try:
            models = ollama.list()
            model_names = [model['name'] for model in models.get('models', [])]
            if self.embedding_model not in model_names:
                logger.info(f"Embedding model '{self.embedding_model}' not found locally.")
                self._download_model()
            else:
                logger.info(f"Embedding model '{self.embedding_model}' is available locally.")
        except Exception as e:
            logger.warning(f"Could not check available models: {e}")
            logger.info("Attempting to download model anyway...")
            self._download_model()

    def _download_model(self):
        """Downloads the embedding model via `ollama.pull`, re-raising on failure."""
        try:
            logger.info(f"Downloading embedding model: {self.embedding_model}")
            logger.info("This may take a few minutes...")
            ollama.pull(self.embedding_model)
            logger.info(f"Successfully downloaded model: {self.embedding_model}")
        except Exception as e:
            logger.error(f"Failed to download model '{self.embedding_model}': {e}")
            raise

    def _test_model_embedding(self, test_text: str = "test") -> bool:
        """Test if the model can generate embeddings successfully."""
        try:
            ollama.embed(model=self.embedding_model, input=test_text)
            return True
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return False

    def read_database_files(self):
        """Loads all entries from `self.database_folder` into `self.loaded_entries`.

        Walks the folder recursively. JSON files are parsed and their
        contents (a list of entries, or a single entry) are appended to
        `self.loaded_entries`; any other file is read as
        newline-separated plain text entries (blank lines are skipped).
        Logs progress via `logger.info`.

        Returns:
            bool: True if the folder exists and every file was read
            successfully. False if the folder doesn't exist, or a file
            could not be parsed/read (loading stops early in that case,
            without raising).
        """
        self.loaded_entries = []
        if not os.path.exists(self.database_folder):
            logger.warning(f"Database folder '{self.database_folder}' does not exist.")
            return False
        for root, dirs, files in os.walk(self.database_folder):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        if filename.endswith('.json'):
                            data = json.load(file)
                            if isinstance(data, list):
                                self.loaded_entries.extend(data)
                                logger.info(f"Loaded {len(data)} JSON entries from {file_path}")
                            else:
                                self.loaded_entries.append(data)
                                logger.info(f"Loaded 1 JSON entry from {file_path}")
                        else:
                            lines = file.readlines()
                            valid_lines = [line.strip() for line in lines if line.strip()]
                            self.loaded_entries.extend(valid_lines)
                            logger.info(f"Loaded {len(valid_lines)} text entries from {file_path}")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON in {file_path}: {e}")
                    return False
                except Exception as e:
                    logger.error(f"Failed to read {file_path}: {e}")
                    return False
        logger.info(f"Total loaded entries: {len(self.loaded_entries)}")
        return True

    def add_chunks_to_database(self):
        """Embeds every entry in `self.loaded_entries` and appends it to `self.vector_db`.

        For each entry, extracts searchable text (via
        `_extract_searchable_text` for dicts, or `str()` otherwise),
        generates its embedding via `ollama.embed`, and appends
        `(entry, embedding, searchable_text)` to `self.vector_db`. Does
        nothing if `self.loaded_entries` is empty or a quick embedding
        model test (`_test_model_embedding`) fails. Individual entry
        failures are logged and skipped rather than raised. Logs
        progress every 10 entries.
        """
        if not self.loaded_entries:
            logger.warning("No entries to add to database.")
            return
        if not self._test_model_embedding():
            logger.error("Model embedding test failed. Cannot proceed.")
            return
        for i, entry in enumerate(self.loaded_entries):
            try:
                if isinstance(entry, dict):
                    searchable_text = self._extract_searchable_text(entry)
                else:
                    searchable_text = str(entry)

                embedding_response = ollama.embed(model=self.embedding_model, input=searchable_text)
                # Corrected: 'embeddings' is the key, not 'embedding'
                embedding = embedding_response['embeddings'][0]
                self.vector_db.append((entry, embedding, searchable_text))
                if (i + 1) % 10 == 0 or i + 1 == len(self.loaded_entries):
                    logger.info(f'Added entry {i+1}/{len(self.loaded_entries)} to the database')
            except Exception as e:
                logger.error(f"Failed to process entry {i+1}: {e}")
                continue

    def _extract_searchable_text(self, json_obj: dict) -> str:
        """Recursively flattens a JSON object into a single space-joined string of its text.

        Special-cases the `{"function": {...}}` tool shape by pulling
        just `name`, `description`, and `command`; otherwise
        concatenates every string value found in the dict, recursing
        into nested dicts and lists.
        """
        if isinstance(json_obj, dict):
            if 'function' in json_obj:
                func = json_obj['function']
                parts = [
                    func.get('name', ''),
                    func.get('description', ''),
                    func.get('command', '')
                ]
                return ' '.join(filter(None, parts))
            else:
                text_parts = []
                for key, value in json_obj.items():
                    if isinstance(value, str):
                        text_parts.append(value)
                    elif isinstance(value, dict):
                        text_parts.append(self._extract_searchable_text(value))
                    elif isinstance(value, (list, tuple)):
                        for item in value:
                            if isinstance(item, (str, dict)):
                                if isinstance(item, dict):
                                    text_parts.append(self._extract_searchable_text(item))
                                else:
                                    text_parts.append(str(item))
                return ' '.join(filter(None, text_parts))
        else:
            return str(json_obj)

    def _get_cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculates cosine similarity between two equal-length vectors.

        Returns 0.0 if either vector has zero magnitude.

        Raises:
            ValueError: If `a` and `b` have different lengths.
        """
        if len(a) != len(b):
            raise ValueError("Vectors must have the same length")
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def retrieve(self, query: str, top_n: int = 3) -> List[Tuple[Any, float, str]]:
        """Finds the entries most similar to `query` by cosine similarity of their embeddings.

        Embeds `query` via Ollama, computes cosine similarity against
        every stored embedding in `self.vector_db`, and returns the
        `top_n` highest-scoring entries. For each entry considered, also
        logs a debug rendering (original JSON plus a TOON conversion for
        dict entries, or the raw text otherwise).

        Args:
            query: The natural-language query to search for.
            top_n: Maximum number of results to return.

        Returns:
            List[Tuple[Any, float, str]]: Up to `top_n` tuples of
            `(entry, similarity, searchable_text)`, sorted by descending
            similarity. Returns an empty list if the vector database is
            empty or the query embedding could not be generated.
        """
        if not self.vector_db:
            logger.warning("Vector database is empty. No entries to retrieve.")
            return []
        try:
            query_embedding_response = ollama.embed(model=self.embedding_model, input=query)
            # Corrected: 'embeddings' is the key, not 'embedding'
            query_embedding = query_embedding_response['embeddings'][0]
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return []
        similarities = []
        for entry, embedding, searchable_text in self.vector_db:
            try:
                similarity = self._get_cosine_similarity(query_embedding, embedding)

                if isinstance(entry, dict):
                    logger.info('ORIGINAL JSON')
                    # Use logger.info for multi-line output
                    for line in json.dumps(entry, indent=2).splitlines():
                        logger.info(line)

                    logger.info('\n--- TOON CONVERSION ---')
                    toon_output = convert_json_to_toon(entry)
                    for line in toon_output.splitlines():
                        logger.info(line)
                else:
                    # Handle non-dict entries (like plain text)
                    logger.info(f'--- Text Entry ---')
                    logger.info(entry)

                similarities.append((entry, similarity, searchable_text))
            except Exception as e:
                logger.error(f"Error calculating similarity: {e}")
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    def get_stats(self) -> dict:
        """Returns basic statistics about the loaded entries and vector database.

        Returns:
            dict: A dict with `total_entries`, `vector_db_size`,
            `embedding_model`, and `database_folder`.
        """
        return {
            "total_entries": len(self.loaded_entries),
            "vector_db_size": len(self.vector_db),
            "embedding_model": self.embedding_model,
            "database_folder": self.database_folder
        }

def convert_json_to_toon(data: dict) -> str:
    """Renders a "function" JSON tool definition as a Token Object Oriented Notation (TOON) string.

    Produces a compact, indented, class-instantiation-like
    representation of a `{"function": {...}}` tool definition (name,
    description, command, parameters, and required fields), intended as
    a more token-efficient alternative to pretty-printed JSON when shown
    to an LLM.

    Args:
        data: A dict expected to contain a `"function"` key whose value
            is itself a dict with `name`, `description`, `command`, and
            `parameters` sub-fields.

    Returns:
        str: The TOON-formatted string, or, if `data` doesn't have the
        expected `"function"` shape, a fallback string containing a raw
        JSON dump prefixed with a note that it wasn't a function JSON.
    """

    # Check if this is the function format we expect
    if "function" not in data or not isinstance(data["function"], dict):
        # Not the expected format, just return a simple non-TOON dump
        return f"--- (Not a function JSON, raw data) ---\n{json.dumps(data, indent=2)}"

    func = data["function"]
    indent = "  "
    toon = []

    # Function Header (Class Instantiation)
    name = func.get("name", "unknown_function")
    desc = func.get("description", "")
    toon.append(f'@Function(name="{name}", description="{desc}") {{')

    # Simple top-level properties (like command)
    if "command" in func:
        # Use json.dumps to correctly format string values
        toon.append(f'{indent}command: {json.dumps(func["command"])}')

    # Parameters Block
    params = func.get("parameters", {})
    if params and isinstance(params, dict):
        toon.append(f'{indent}@Parameters {{')

        # Properties
        props = params.get("properties", {})
        if props:
            for prop_name, details in props.items():
                toon.append(f'{indent}{indent}@Property(name="{prop_name}") {{')
                for key, val in details.items():
                    # Use json.dumps for all values to handle strings, lists, bools
                    toon.append(f'{indent}{indent}{indent}{key}: {json.dumps(val)}')
                toon.append(f'{indent}{indent}}}') # Close @Property

        # Required
        required = params.get("required", [])
        if required:
            toon.append(f'{indent}{indent}required: {json.dumps(required)}')

        toon.append(f'{indent}}}') # Close @Parameters

    toon.append('}') # Close @Function

    # Join all lines with a newline
    return "\n".join(toon)

if __name__ == "__main__":
    # Default config path, resolved relative to this file's location.
    config_path = str(pathlib.Path(__file__).resolve().parent.parent.parent / "config" / "config_default.json")

    # Check if config file exists
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at: {config_path}")
        logger.error("Please update the 'config_path' variable in the script.")
        sys.exit(1)

    config_manager = ConfigManager(config_path=config_path)
    config = config_manager.get()

    try:
        rag = RAGSimilarityCheck(config)

        stats = rag.get_stats()
        logger.info(f"RAG Database Stats: {json.dumps(stats, indent=2)}")

        while True:
            input_query = input('\nAsk me a question (or "quit" to exit): ')
            if input_query.lower() in ['quit', 'exit', 'q']:
                break

            retrieved_knowledge = rag.retrieve(input_query)

            if not retrieved_knowledge:
                logger.info('No relevant knowledge found.')
                continue

            logger.info('Retrieved knowledge:')

            for i, (entry, similarity, searchable_text) in enumerate(retrieved_knowledge, 1):

                if isinstance(entry, dict):
                    logger.info('ORIGINAL JSON')
                    # Use logger.info for multi-line output
                    for line in json.dumps(entry, indent=2).splitlines():
                        logger.info(line)

                    logger.info('\n--- TOON CONVERSION ---')
                    toon_output = convert_json_to_toon(entry)
                    for line in toon_output.splitlines():
                        logger.info(line)

                else:
                    # Handle non-dict entries (like plain text)
                    logger.info(f'--- Text Entry ---')
                    logger.info(entry)

                # Show what text was used for the similarity match
                searchable_text_snippet = (searchable_text[:120] + '...') if len(searchable_text) > 120 else searchable_text
                logger.info(f'\n   (Matched on: "{searchable_text_snippet}")')
            # <--- END OF MODIFIED LOOP ---

    except KeyboardInterrupt:
        logger.info("\nExiting...")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)