"""Parakeet Semantic Search - Intelligent podcast discovery engine."""

__version__ = "0.1.0"
__author__ = "Julien Pequegnot"

from .search import SearchEngine
from .embeddings import EmbeddingModel
from .vectorstore import VectorStore

__all__ = ["SearchEngine", "EmbeddingModel", "VectorStore"]
