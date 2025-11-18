"""Embedding model management."""

from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingModel:
    """Wrapper for Sentence Transformers embedding model."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize embedding model.

        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embedding vectors
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
