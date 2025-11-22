"""Performance optimization and caching for semantic search."""

import time
from typing import Dict, List, Optional, Any, Callable, Tuple
from functools import wraps
import hashlib
import json
import threading
from collections import OrderedDict
import numpy as np


class QueryCache:
    """LRU cache for search query results."""

    def __init__(self, max_size: int = 1000, ttl_seconds: Optional[int] = None):
        """Initialize query cache.

        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cache entries (None = no expiry)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, query: str, **kwargs) -> str:
        """Create cache key from query and parameters.

        Args:
            query: Search query
            **kwargs: Additional parameters (limit, threshold, etc.)

        Returns:
            Unique cache key
        """
        params = {"query": query, **kwargs}
        # Sort params for consistent key generation
        key_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, query: str, **kwargs) -> Optional[List[Dict]]:
        """Get cached results.

        Args:
            query: Search query
            **kwargs: Additional parameters

        Returns:
            Cached results or None if not found/expired
        """
        key = self._make_key(query, **kwargs)

        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None

            results, timestamp = self.cache[key]

            # Check TTL
            if self.ttl_seconds:
                if time.time() - timestamp > self.ttl_seconds:
                    del self.cache[key]
                    self.misses += 1
                    return None

            # Move to end (LRU)
            self.cache.move_to_end(key)
            self.hits += 1
            return results

    def set(self, query: str, results: List[Dict], **kwargs) -> None:
        """Cache search results.

        Args:
            query: Search query
            results: Search results to cache
            **kwargs: Additional parameters
        """
        key = self._make_key(query, **kwargs)

        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    self.cache.popitem(last=False)

            self.cache[key] = (results, time.time())

    def clear(self) -> None:
        """Clear all cached results."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache stats
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "usage_percent": (len(self.cache) / self.max_size * 100) if self.max_size > 0 else 0,
        }


class CachingSearchEngine:
    """Search engine wrapper with result caching."""

    def __init__(self, search_engine, cache_size: int = 1000, ttl_seconds: Optional[int] = None):
        """Initialize caching search engine.

        Args:
            search_engine: Underlying SearchEngine instance
            cache_size: Maximum cached queries
            ttl_seconds: Cache TTL in seconds
        """
        self.search_engine = search_engine
        self.cache = QueryCache(max_size=cache_size, ttl_seconds=ttl_seconds)

    def search(self, query: str, limit: int = 10, threshold: Optional[float] = None):
        """Search with caching.

        Args:
            query: Search query
            limit: Number of results
            threshold: Similarity threshold

        Returns:
            Search results (from cache or computed)
        """
        # Try cache first
        cached = self.cache.get(query, limit=limit, threshold=threshold)
        if cached is not None:
            return cached

        # Compute and cache
        results = self.search_engine.search(query, limit=limit, threshold=threshold)
        self.cache.set(query, results, limit=limit, threshold=threshold)
        return results

    def get_recommendations(
        self,
        episode_id: str,
        limit: int = 5,
        podcast_id: str = None,
        exclude_episode: bool = True,
    ):
        """Get recommendations with caching.

        Args:
            episode_id: Episode ID
            limit: Number of recommendations
            podcast_id: Optional podcast filter
            exclude_episode: Whether to exclude source episode

        Returns:
            Recommendations (from cache or computed)
        """
        cached = self.cache.get(
            f"rec:{episode_id}",
            limit=limit,
            podcast_id=podcast_id,
            exclude_episode=exclude_episode,
        )
        if cached is not None:
            return cached

        results = self.search_engine.get_recommendations(
            episode_id=episode_id,
            limit=limit,
            podcast_id=podcast_id,
            exclude_episode=exclude_episode,
        )
        self.cache.set(
            f"rec:{episode_id}",
            results,
            limit=limit,
            podcast_id=podcast_id,
            exclude_episode=exclude_episode,
        )
        return results

    def get_hybrid_recommendations(
        self,
        episode_ids: List[str],
        limit: int = 10,
        diversity_boost: float = 0.0,
        podcast_id: Optional[str] = None,
    ):
        """Get hybrid recommendations for multiple episodes with caching.

        Args:
            episode_ids: List of episode IDs
            limit: Number of recommendations
            diversity_boost: Diversity boost factor (0-1)
            podcast_id: Optional podcast filter

        Returns:
            Hybrid recommendations (from cache or computed)
        """
        # Create cache key from episode IDs
        episodes_key = ":".join(sorted(episode_ids))
        cached = self.cache.get(
            f"hybrid_rec:{episodes_key}",
            limit=limit,
            diversity_boost=diversity_boost,
            podcast_id=podcast_id,
        )
        if cached is not None:
            return cached

        results = self.search_engine.get_hybrid_recommendations(
            episode_ids=episode_ids,
            limit=limit,
            diversity_boost=diversity_boost,
            podcast_id=podcast_id,
        )
        self.cache.set(
            f"hybrid_rec:{episodes_key}",
            results,
            limit=limit,
            diversity_boost=diversity_boost,
            podcast_id=podcast_id,
        )
        return results

    def clear_cache(self) -> None:
        """Clear search cache."""
        self.cache.clear()

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()


class PerformanceProfiler:
    """Profile search engine performance."""

    def __init__(self):
        """Initialize profiler."""
        self.timings: Dict[str, List[float]] = {}

    def profile_search(self, search_engine, queries: List[str], limit: int = 10) -> Dict[str, Any]:
        """Profile search performance.

        Args:
            search_engine: SearchEngine instance
            queries: List of queries to profile
            limit: Number of results per query

        Returns:
            Performance statistics
        """
        if not queries:
            return {}

        query_times = []
        embedding_times = []
        vectorstore_times = []

        for query in queries:
            # Time full search
            start = time.perf_counter()
            results = search_engine.search(query, limit=limit)
            elapsed = time.perf_counter() - start
            query_times.append(elapsed)

        return {
            "num_queries": len(queries),
            "total_time": sum(query_times),
            "mean_time": np.mean(query_times),
            "median_time": np.median(query_times),
            "min_time": np.min(query_times),
            "max_time": np.max(query_times),
            "std_dev": np.std(query_times),
            "qps": len(queries) / sum(query_times) if sum(query_times) > 0 else 0,  # Queries per second
            "percentile_95": np.percentile(query_times, 95),
            "percentile_99": np.percentile(query_times, 99),
        }

    def profile_batch_search(
        self,
        search_engine,
        queries: List[str],
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Profile batch search performance.

        Args:
            search_engine: SearchEngine instance
            queries: List of queries
            limit: Results per query

        Returns:
            Batch performance stats
        """
        if not hasattr(search_engine, "batch_search"):
            raise NotImplementedError("Search engine doesn't support batch_search")

        times = []
        for _ in range(3):  # Run 3 times for consistency
            start = time.perf_counter()
            search_engine.batch_search(queries, limit=limit)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return {
            "num_queries": len(queries),
            "mean_time": np.mean(times),
            "median_time": np.median(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "qps": len(queries) / np.mean(times) if times else 0,
        }


class MemoryOptimizer:
    """Optimize memory usage of search engine."""

    @staticmethod
    def batch_embed(embedding_model, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Embed texts in batches to reduce memory.

        Args:
            embedding_model: EmbeddingModel instance
            texts: List of texts to embed
            batch_size: Batch size for embedding

        Returns:
            Embedding matrix
        """
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = embedding_model.embed_batch(batch)
            embeddings.append(batch_embeddings)

        return np.vstack(embeddings) if embeddings else np.array([])

    @staticmethod
    def quantize_embeddings(embeddings: np.ndarray, nbits: int = 8) -> np.ndarray:
        """Quantize embeddings to reduce memory (simple int8 quantization).

        Args:
            embeddings: Embedding matrix
            nbits: Number of bits (8 or 16)

        Returns:
            Quantized embeddings
        """
        # Find min/max per embedding
        min_vals = embeddings.min(axis=1, keepdims=True)
        max_vals = embeddings.max(axis=1, keepdims=True)

        # Normalize to [0, 1]
        normalized = (embeddings - min_vals) / (max_vals - min_vals + 1e-8)

        # Quantize
        if nbits == 8:
            quantized = (normalized * 255).astype(np.uint8)
        elif nbits == 16:
            quantized = (normalized * 65535).astype(np.uint16)
        else:
            raise ValueError("nbits must be 8 or 16")

        return quantized

    @staticmethod
    def dequantize_embeddings(
        quantized: np.ndarray,
        min_vals: np.ndarray,
        max_vals: np.ndarray,
        nbits: int = 8,
    ) -> np.ndarray:
        """Dequantize embeddings.

        Args:
            quantized: Quantized embeddings
            min_vals: Minimum values used during quantization
            max_vals: Maximum values used during quantization
            nbits: Number of bits used

        Returns:
            Dequantized embeddings
        """
        # Dequantize
        if nbits == 8:
            normalized = quantized.astype(np.float32) / 255.0
        elif nbits == 16:
            normalized = quantized.astype(np.float32) / 65535.0
        else:
            raise ValueError("nbits must be 8 or 16")

        # Denormalize
        embeddings = normalized * (max_vals - min_vals) + min_vals
        return embeddings


def time_function(func: Callable) -> Callable:
    """Decorator to time function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result

    return wrapper


class BatchSearchEngine:
    """Extend search engine with batch operations."""

    def __init__(self, search_engine):
        """Initialize batch search engine.

        Args:
            search_engine: Underlying SearchEngine instance
        """
        self.search_engine = search_engine

    def batch_search(
        self,
        queries: List[str],
        limit: int = 10,
        threshold: Optional[float] = None,
    ) -> List[List[Dict]]:
        """Search multiple queries efficiently.

        Args:
            queries: List of search queries
            limit: Number of results per query
            threshold: Similarity threshold

        Returns:
            List of result lists, one per query
        """
        results = []

        for query in queries:
            query_results = self.search_engine.search(
                query, limit=limit, threshold=threshold
            )
            results.append(query_results)

        return results

    def batch_recommendations(
        self,
        episode_ids: List[str],
        limit: int = 5,
        podcast_id: str = None,
    ) -> List[List[Dict]]:
        """Get recommendations for multiple episodes.

        Args:
            episode_ids: List of episode IDs
            limit: Recommendations per episode
            podcast_id: Optional podcast filter

        Returns:
            List of recommendation lists
        """
        results = []

        for episode_id in episode_ids:
            recs = self.search_engine.get_recommendations(
                episode_id=episode_id,
                limit=limit,
                podcast_id=podcast_id,
            )
            results.append(recs)

        return results
