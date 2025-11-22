"""Parakeet Semantic Search - Intelligent podcast discovery engine."""

__version__ = "0.1.0"
__author__ = "Julien Pequegnot"

from .search import SearchEngine
from .embeddings import EmbeddingModel
from .vectorstore import VectorStore
from .models import Episode, Transcript, SearchResult, Config
from .evaluation import (
    EvaluationFramework,
    RelevanceEvaluator,
    CoverageEvaluator,
    DiversityEvaluator,
    EvaluationMetrics,
)
from .clustering import ClusteringAnalyzer
from .optimization import (
    QueryCache,
    CachingSearchEngine,
    PerformanceProfiler,
    MemoryOptimizer,
    BatchSearchEngine,
)

__all__ = [
    "SearchEngine",
    "EmbeddingModel",
    "VectorStore",
    "Episode",
    "Transcript",
    "SearchResult",
    "Config",
    "EvaluationFramework",
    "RelevanceEvaluator",
    "CoverageEvaluator",
    "DiversityEvaluator",
    "EvaluationMetrics",
    "ClusteringAnalyzer",
    "QueryCache",
    "CachingSearchEngine",
    "PerformanceProfiler",
    "MemoryOptimizer",
    "BatchSearchEngine",
]
