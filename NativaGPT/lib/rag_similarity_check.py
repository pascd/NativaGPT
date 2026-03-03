"""
RAG Similarity Check v2.0 - Configurable Embedding Provider

Supports:
- Ollama embeddings (default)
- OpenAI embeddings (optional)
- Local sentence transformers (optional)
"""

import os
import sys
import json
import pathlib
from typing import List, Tuple, Any, Optional
from abc import ABC, abstractmethod

from NativaGPT.lib.config_manager import ConfigManager
from NativaGPT.lib.coloring_logger import logger


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text."""
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embedding vectors for multiple texts."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama embedding provider using ollama Python library."""

    def __init__(self, model: str):
        self.model = model
        self._ollama = None
        self._available = False
        self._check_available()

    def _check_available(self):
        try:
            import ollama

            self._ollama = ollama
            self._available = True
            logger.info(
                f"[OllamaEmbedding] Provider available with model: {self.model}"
            )
        except ImportError:
            logger.warning("[OllamaEmbedding] ollama library not installed")
            self._available = False

    def is_available(self) -> bool:
        return self._available

    def get_embedding(self, text: str) -> List[float]:
        if not self._available:
            raise RuntimeError("Ollama not available")
        try:
            response = self._ollama.embed(model=self.model, input=text)
            return response.get("embeddings", [[]])[0]
        except Exception as e:
            logger.error(f"[OllamaEmbedding] Error: {e}")
            raise

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self._available:
            raise RuntimeError("Ollama not available")
        try:
            response = self._ollama.embed(model=self.model, input=texts)
            return response.get("embeddings", [])
        except Exception as e:
            logger.error(f"[OllamaEmbedding] Error: {e}")
            raise

    def ensure_model_available(self):
        """Download model if not available locally."""
        if not self._available:
            return False

        try:
            models = self._ollama.list()
            model_names = [m.get("name", "") for m in models.get("models", [])]
            if self.model not in model_names:
                logger.info(f"[OllamaEmbedding] Downloading model: {self.model}")
                self._ollama.pull(self.model)
                logger.info(f"[OllamaEmbedding] Model downloaded: {self.model}")
            return True
        except Exception as e:
            logger.error(f"[OllamaEmbedding] Failed to ensure model: {e}")
            return False


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""

    def __init__(
        self, model: str = "text-embedding-3-small", api_key: Optional[str] = None
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
        self._available = False
        self._check_available()

    def _check_available(self):
        if not self.api_key:
            logger.warning("[OpenAIEmbedding] No API key provided")
            return

        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key)
            self._available = True
            logger.info(
                f"[OpenAIEmbedding] Provider available with model: {self.model}"
            )
        except ImportError:
            logger.warning("[OpenAIEmbedding] openai library not installed")
        except Exception as e:
            logger.warning(f"[OpenAIEmbedding] Init failed: {e}")

    def is_available(self) -> bool:
        return self._available

    def get_embedding(self, text: str) -> List[float]:
        if not self._available:
            raise RuntimeError("OpenAI not available")
        try:
            response = self._client.embeddings.create(input=text, model=self.model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"[OpenAIEmbedding] Error: {e}")
            raise

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self._available:
            raise RuntimeError("OpenAI not available")
        try:
            response = self._client.embeddings.create(input=texts, model=self.model)
            return [d.embedding for d in response.data]
        except Exception as e:
            logger.error(f"[OpenAIEmbedding] Error: {e}")
            raise


class SentenceTransformerProvider(EmbeddingProvider):
    """Local sentence transformer embedding provider."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.model_name = model
        self._model = None
        self._available = False
        self._check_available()

    def _check_available(self):
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._available = True
            logger.info(f"[SentenceTransformer] Provider available: {self.model_name}")
        except ImportError:
            logger.warning("[SentenceTransformer] sentence-transformers not installed")
        except Exception as e:
            logger.warning(f"[SentenceTransformer] Init failed: {e}")

    def is_available(self) -> bool:
        return self._available

    def get_embedding(self, text: str) -> List[float]:
        if not self._available:
            raise RuntimeError("SentenceTransformer not available")
        return self._model.encode(text).tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self._available:
            raise RuntimeError("SentenceTransformer not available")
        return self._model.encode(texts).tolist()


class RAGSimilarityCheck:
    """
    RAG Similarity Check v2.0 - Configurable Embedding Provider

    Features:
    - Multiple embedding backends (Ollama, OpenAI, SentenceTransformers)
    - Automatic model download for Ollama
    - Efficient similarity search with cosine similarity
    - JSON and text database loading
    """

    def __init__(self, config):
        self.config = config
        self.loaded_entries = []
        self.vector_db = []

        nativa_cfg = config.get("nativa_gpt", {})
        llm_cfg = config.get("llm_config", {})

        self.database_folder = nativa_cfg.get(
            "database_folder",
            str(pathlib.Path(__file__).parent.parent.parent / "config" / "functions"),
        )

        embedding_backend = llm_cfg.get("embedding_backend", "ollama")
        embedding_model = llm_cfg.get("embedding_model", "nomic-embed-text")

        self.provider = self._create_provider(embedding_backend, embedding_model)

        if self.provider and self.provider.is_available():
            if hasattr(self.provider, "ensure_model_available"):
                self.provider.ensure_model_available()

            if self.read_database_files():
                self.add_chunks_to_database()
        else:
            logger.warning("No embedding provider available - RAG disabled")

    def _create_provider(self, backend: str, model: str) -> Optional[EmbeddingProvider]:
        """Create embedding provider based on backend type."""
        if backend == "ollama":
            return OllamaEmbeddingProvider(model)
        elif backend == "openai":
            return OpenAIEmbeddingProvider(model)
        elif backend == "sentence_transformers":
            return SentenceTransformerProvider(model)
        else:
            logger.warning(
                f"Unknown embedding backend: {backend}, defaulting to ollama"
            )
            return OllamaEmbeddingProvider(model)

    def read_database_files(self) -> bool:
        """Load all database files."""
        self.loaded_entries = []

        if not self.database_folder:
            logger.warning("No database_folder configured")
            return False

        if not os.path.exists(self.database_folder):
            logger.warning(f"Database folder not found: {self.database_folder}")
            return False

        json_count = 0
        for root, dirs, files in os.walk(self.database_folder):
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    if filename.endswith(".json"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            if isinstance(data, list):
                                self.loaded_entries.extend(data)
                            else:
                                self.loaded_entries.append(data)
                            json_count += 1
                    elif filename.endswith(".txt"):
                        with open(file_path, "r", encoding="utf-8") as f:
                            lines = [l.strip() for l in f if l.strip()]
                            self.loaded_entries.extend(lines)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error in {file_path}: {e}")
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")

        logger.info(
            f"Loaded {len(self.loaded_entries)} entries from {json_count} JSON files"
        )
        return len(self.loaded_entries) > 0

    def add_chunks_to_database(self):
        """Add entries to vector database with embeddings."""
        if not self.loaded_entries:
            logger.warning("No entries to add to database")
            return

        if not self.provider or not self.provider.is_available():
            logger.warning("No embedding provider available")
            return

        logger.info(f"Creating embeddings for {len(self.loaded_entries)} entries...")

        for i, entry in enumerate(self.loaded_entries):
            try:
                searchable_text = self._extract_searchable_text(entry)
                embedding = self.provider.get_embedding(searchable_text)
                self.vector_db.append((entry, embedding, searchable_text))

                if (i + 1) % 10 == 0 or i + 1 == len(self.loaded_entries):
                    logger.info(f"Embedded {i + 1}/{len(self.loaded_entries)} entries")

            except Exception as e:
                logger.error(f"Failed to embed entry {i + 1}: {e}")
                continue

        logger.info(f"Vector database ready with {len(self.vector_db)} vectors")

    def _extract_searchable_text(self, obj) -> str:
        """Extract searchable text from object."""
        if isinstance(obj, str):
            return obj

        if isinstance(obj, dict):
            if "function" in obj:
                func = obj["function"]
                parts = [
                    func.get("name", ""),
                    func.get("description", ""),
                    func.get("command", ""),
                ]
                return " ".join(filter(None, parts))

            text_parts = []
            for key, value in obj.items():
                if isinstance(value, str):
                    text_parts.append(value)
                elif isinstance(value, dict):
                    text_parts.append(self._extract_searchable_text(value))
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif isinstance(item, dict):
                            text_parts.append(self._extract_searchable_text(item))
            return " ".join(filter(None, text_parts))

        return str(obj)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_n: int = 3) -> List[Tuple[Any, float, str]]:
        """
        Retrieve most similar entries to query.

        Returns:
            List of (entry, similarity_score, searchable_text) tuples
        """
        if not self.vector_db:
            logger.warning("Vector database is empty")
            return []

        if not self.provider or not self.provider.is_available():
            logger.warning("No embedding provider available")
            return []

        try:
            query_embedding = self.provider.get_embedding(query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            return []

        similarities = []
        for entry, embedding, searchable_text in self.vector_db:
            try:
                sim = self._cosine_similarity(query_embedding, embedding)
                similarities.append((entry, sim, searchable_text))
            except Exception as e:
                logger.error(f"Error calculating similarity: {e}")
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]

    def get_stats(self) -> dict:
        """Get RAG database statistics."""
        provider_name = self.provider.__class__.__name__ if self.provider else "None"
        model_name = ""
        if self.provider:
            if hasattr(self.provider, "model"):
                model_name = self.provider.model
            elif hasattr(self.provider, "model_name"):
                model_name = self.provider.model_name

        return {
            "total_entries": len(self.loaded_entries),
            "vector_db_size": len(self.vector_db),
            "provider": provider_name,
            "model": model_name,
            "database_folder": self.database_folder,
        }


if __name__ == "__main__":
    config_path = (
        "/home/pedro/Documents/uv-projects/NativaGPT/config/config_default.json"
    )

    if not os.path.exists(config_path):
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    config_manager = ConfigManager(config_path=config_path)
    config = config_manager.get()

    try:
        rag = RAGSimilarityCheck(config)
        stats = rag.get_stats()
        logger.info(f"RAG Stats: {json.dumps(stats, indent=2)}")

        while True:
            query = input('\nQuery (or "quit"): ')
            if query.lower() in ["quit", "exit", "q"]:
                break

            results = rag.retrieve(query)
            if not results:
                logger.info("No results found")
                continue

            for i, (entry, sim, text) in enumerate(results, 1):
                logger.info(f"\n--- Result {i} (similarity: {sim:.3f}) ---")
                if isinstance(entry, dict):
                    logger.info(json.dumps(entry, indent=2)[:500])
                else:
                    logger.info(str(entry)[:500])

    except KeyboardInterrupt:
        logger.info("Exiting...")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
