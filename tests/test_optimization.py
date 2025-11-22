"""Tests for optimization and caching functionality."""

import pytest
import time
import numpy as np
from unittest.mock import Mock, MagicMock
from src.parakeet_search.optimization import (
    QueryCache,
    CachingSearchEngine,
    PerformanceProfiler,
    MemoryOptimizer,
    BatchSearchEngine,
)


class TestQueryCache:
    """Tests for query result caching."""

    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = QueryCache(max_size=100)

        assert cache.max_size == 100
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_set_and_get(self):
        """Test setting and getting cache entries."""
        cache = QueryCache(max_size=100)

        results = [{"id": "ep_001", "_distance": 0.1}]
        cache.set("machine learning", results, limit=10)

        cached = cache.get("machine learning", limit=10)

        assert cached == results
        assert cache.hits == 1
        assert cache.misses == 0

    def test_cache_miss(self):
        """Test cache miss."""
        cache = QueryCache(max_size=100)

        cached = cache.get("unknown query")

        assert cached is None
        assert cache.hits == 0
        assert cache.misses == 1

    def test_cache_different_params(self):
        """Test that different parameters create different cache entries."""
        cache = QueryCache(max_size=100)

        results_10 = [{"id": f"ep_{i:03d}"} for i in range(10)]
        results_20 = [{"id": f"ep_{i:03d}"} for i in range(20)]

        cache.set("machine learning", results_10, limit=10)
        cache.set("machine learning", results_20, limit=20)

        cached_10 = cache.get("machine learning", limit=10)
        cached_20 = cache.get("machine learning", limit=20)

        assert cached_10 == results_10
        assert cached_20 == results_20

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = QueryCache(max_size=3)

        # Fill cache
        cache.set("query1", [{"id": "ep_001"}])
        cache.set("query2", [{"id": "ep_002"}])
        cache.set("query3", [{"id": "ep_003"}])

        # Add one more (should evict query1)
        cache.set("query4", [{"id": "ep_004"}])

        # query1 should be evicted
        assert cache.get("query1") is None

        # Others should exist
        assert cache.get("query2") is not None
        assert cache.get("query3") is not None
        assert cache.get("query4") is not None

    def test_cache_ttl_expiry(self):
        """Test cache entry expiry with TTL."""
        cache = QueryCache(max_size=100, ttl_seconds=1)

        results = [{"id": "ep_001"}]
        cache.set("test query", results)

        # Should exist immediately
        assert cache.get("test query") is not None

        # Wait for expiry
        time.sleep(1.1)

        # Should be expired
        assert cache.get("test query") is None

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = QueryCache(max_size=100)

        # Add some entries
        cache.set("query1", [{"id": "ep_001"}])
        cache.get("query1")
        cache.get("query1")
        cache.get("unknown")

        stats = cache.stats()

        assert stats["size"] == 1
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] > 0.5

    def test_cache_clear(self):
        """Test clearing cache."""
        cache = QueryCache(max_size=100)

        cache.set("query1", [{"id": "ep_001"}])
        cache.set("query2", [{"id": "ep_002"}])

        cache.clear()

        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0


class TestCachingSearchEngine:
    """Tests for caching search engine wrapper."""

    def test_caching_search_engine_initialization(self):
        """Test initialization."""
        mock_engine = Mock()
        caching_engine = CachingSearchEngine(mock_engine, cache_size=100)

        assert caching_engine.search_engine == mock_engine
        assert caching_engine.cache.max_size == 100

    def test_search_uses_cache(self):
        """Test that search results are cached."""
        mock_engine = Mock()
        mock_engine.search.return_value = [{"id": "ep_001"}]

        caching_engine = CachingSearchEngine(mock_engine)

        # First call should hit search engine
        result1 = caching_engine.search("test query", limit=10)
        assert mock_engine.search.call_count == 1

        # Second call should hit cache
        result2 = caching_engine.search("test query", limit=10)
        assert mock_engine.search.call_count == 1  # Still 1

        assert result1 == result2

    def test_search_different_params_no_cache_reuse(self):
        """Test that different parameters don't share cache."""
        mock_engine = Mock()
        mock_engine.search.side_effect = [
            [{"id": "ep_001"}],
            [{"id": "ep_001"}, {"id": "ep_002"}],
        ]

        caching_engine = CachingSearchEngine(mock_engine)

        result1 = caching_engine.search("test query", limit=5)
        result2 = caching_engine.search("test query", limit=10)

        assert mock_engine.search.call_count == 2

    def test_recommendations_cached(self):
        """Test that recommendations are cached."""
        mock_engine = Mock()
        mock_engine.get_recommendations.return_value = [{"id": "ep_001"}]

        caching_engine = CachingSearchEngine(mock_engine)

        # First call
        result1 = caching_engine.get_recommendations("ep_001", limit=5)
        assert mock_engine.get_recommendations.call_count == 1

        # Second call (cached)
        result2 = caching_engine.get_recommendations("ep_001", limit=5)
        assert mock_engine.get_recommendations.call_count == 1

    def test_clear_cache(self):
        """Test clearing cache."""
        mock_engine = Mock()
        mock_engine.search.return_value = [{"id": "ep_001"}]

        caching_engine = CachingSearchEngine(mock_engine)

        # Populate cache
        caching_engine.search("query1")
        assert len(caching_engine.cache.cache) == 1

        # Clear
        caching_engine.clear_cache()
        assert len(caching_engine.cache.cache) == 0

    def test_cache_stats(self):
        """Test getting cache statistics."""
        mock_engine = Mock()
        mock_engine.search.return_value = [{"id": "ep_001"}]

        caching_engine = CachingSearchEngine(mock_engine)

        caching_engine.search("query1")
        caching_engine.search("query1")
        caching_engine.search("query2")

        stats = caching_engine.cache_stats()

        # First call to query1 = miss, second = hit, call to query2 = miss
        assert stats["hits"] == 1
        assert stats["misses"] == 2


class TestPerformanceProfiler:
    """Tests for performance profiling."""

    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = PerformanceProfiler()

        assert profiler.timings == {}

    def test_profile_search(self):
        """Test search profiling."""
        mock_engine = Mock()
        mock_engine.search.return_value = [{"id": "ep_001"}]

        profiler = PerformanceProfiler()
        stats = profiler.profile_search(mock_engine, ["query1", "query2"], limit=10)

        assert "num_queries" in stats
        assert "total_time" in stats
        assert "mean_time" in stats
        assert "qps" in stats
        assert stats["num_queries"] == 2

    def test_profile_search_stats_accuracy(self):
        """Test that profiling stats are accurate."""
        mock_engine = Mock()
        mock_engine.search.return_value = [{"id": "ep_001"}]

        profiler = PerformanceProfiler()
        queries = ["query1", "query2", "query3"]
        stats = profiler.profile_search(mock_engine, queries, limit=10)

        # Should have called search 3 times
        assert mock_engine.search.call_count == 3

        # Stats should be present
        assert stats["num_queries"] == 3
        assert stats["mean_time"] > 0
        assert stats["std_dev"] >= 0

    def test_batch_search_profiling(self):
        """Test batch search profiling."""
        mock_engine = Mock()
        mock_engine.batch_search.return_value = [[{"id": "ep_001"}], [{"id": "ep_002"}]]

        profiler = PerformanceProfiler()
        stats = profiler.profile_batch_search(mock_engine, ["query1", "query2"])

        assert "num_queries" in stats
        assert "mean_time" in stats
        assert "qps" in stats


class TestMemoryOptimizer:
    """Tests for memory optimization."""

    def test_batch_embed(self):
        """Test batch embedding."""
        mock_model = Mock()
        mock_model.embed_batch.side_effect = [
            np.random.randn(2, 384),
            np.random.randn(1, 384),
        ]

        texts = ["text1", "text2", "text3"]
        embeddings = MemoryOptimizer.batch_embed(mock_model, texts, batch_size=2)

        assert embeddings.shape[0] == 3
        assert embeddings.shape[1] == 384
        assert mock_model.embed_batch.call_count == 2

    def test_batch_embed_single_batch(self):
        """Test batch embedding with single batch."""
        mock_model = Mock()
        mock_model.embed_batch.return_value = np.random.randn(3, 384)

        texts = ["text1", "text2", "text3"]
        embeddings = MemoryOptimizer.batch_embed(mock_model, texts, batch_size=5)

        assert embeddings.shape == (3, 384)

    def test_quantize_embeddings_int8(self):
        """Test int8 embedding quantization."""
        embeddings = np.array([
            [1.0, 0.0, 0.5],
            [0.2, 0.8, 0.1],
        ], dtype=np.float32)

        quantized = MemoryOptimizer.quantize_embeddings(embeddings, nbits=8)

        assert quantized.dtype == np.uint8
        assert quantized.shape == embeddings.shape

    def test_quantize_embeddings_int16(self):
        """Test int16 embedding quantization."""
        embeddings = np.random.randn(10, 384).astype(np.float32)

        quantized = MemoryOptimizer.quantize_embeddings(embeddings, nbits=16)

        assert quantized.dtype == np.uint16
        assert quantized.shape == embeddings.shape

    def test_dequantize_embeddings(self):
        """Test embedding dequantization."""
        embeddings = np.random.randn(5, 384).astype(np.float32)
        min_vals = embeddings.min(axis=1, keepdims=True)
        max_vals = embeddings.max(axis=1, keepdims=True)

        quantized = MemoryOptimizer.quantize_embeddings(embeddings, nbits=8)
        dequantized = MemoryOptimizer.dequantize_embeddings(
            quantized, min_vals, max_vals, nbits=8
        )

        # Dequantized should be close to original
        assert dequantized.shape == embeddings.shape
        # Quantization to int8 has limited precision, use larger tolerance
        assert np.allclose(dequantized, embeddings, atol=0.1)


class TestBatchSearchEngine:
    """Tests for batch search operations."""

    def test_batch_search(self):
        """Test batch search."""
        mock_engine = Mock()
        mock_engine.search.side_effect = [
            [{"id": "ep_001"}],
            [{"id": "ep_002"}],
            [{"id": "ep_003"}],
        ]

        batch_engine = BatchSearchEngine(mock_engine)
        results = batch_engine.batch_search(["query1", "query2", "query3"], limit=10)

        assert len(results) == 3
        assert mock_engine.search.call_count == 3

    def test_batch_search_empty(self):
        """Test batch search with empty list."""
        mock_engine = Mock()

        batch_engine = BatchSearchEngine(mock_engine)
        results = batch_engine.batch_search([], limit=10)

        assert results == []
        assert mock_engine.search.call_count == 0

    def test_batch_recommendations(self):
        """Test batch recommendations."""
        mock_engine = Mock()
        mock_engine.get_recommendations.side_effect = [
            [{"id": "ep_001"}],
            [{"id": "ep_002"}],
        ]

        batch_engine = BatchSearchEngine(mock_engine)
        results = batch_engine.batch_recommendations(["ep_001", "ep_002"], limit=5)

        assert len(results) == 2
        assert mock_engine.get_recommendations.call_count == 2


class TestOptimizationIntegration:
    """Integration tests for optimization features."""

    def test_caching_plus_batch_search(self):
        """Test combining caching with batch search."""
        mock_engine = Mock()
        mock_engine.search.return_value = [{"id": "ep_001"}]

        caching_engine = CachingSearchEngine(mock_engine)
        batch_engine = BatchSearchEngine(caching_engine)

        results = batch_engine.batch_search(["query1", "query1", "query2"], limit=10)

        # query1 appears twice, should only call underlying search twice
        assert mock_engine.search.call_count == 2

    def test_memory_optimization_workflow(self):
        """Test complete memory optimization workflow."""
        # Create sample embeddings
        embeddings = np.random.randn(100, 384).astype(np.float32)

        # Quantize
        min_vals = embeddings.min(axis=1, keepdims=True)
        max_vals = embeddings.max(axis=1, keepdims=True)
        quantized = MemoryOptimizer.quantize_embeddings(embeddings, nbits=8)

        # Should use significantly less memory
        original_size = embeddings.nbytes
        quantized_size = quantized.nbytes
        compression_ratio = original_size / quantized_size

        assert compression_ratio == 4  # float32 = 4 bytes, uint8 = 1 byte

        # Dequantize and verify
        dequantized = MemoryOptimizer.dequantize_embeddings(
            quantized, min_vals, max_vals, nbits=8
        )

        # Quantization to int8 has limited precision, use larger tolerance
        assert np.allclose(dequantized, embeddings, atol=0.1)
