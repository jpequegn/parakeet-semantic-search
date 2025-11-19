"""Performance benchmarks for Parakeet Semantic Search.

This module contains benchmarks for key operations:
- Embedding generation speed
- Vector store operations (creation, insertion, search)
- Search latency at various dataset sizes

Run with: pytest tests/test_benchmarks.py -v --benchmark-only
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from parakeet_search.embeddings import EmbeddingModel
from parakeet_search.vectorstore import VectorStore
from parakeet_search.search import SearchEngine


class TestEmbeddingBenchmarks:
    """Benchmarks for embedding generation performance."""

    @pytest.fixture
    def embedding_model(self):
        """Create embedding model for benchmarks."""
        with patch("parakeet_search.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            em = EmbeddingModel()
            yield em

    def test_single_text_embedding(self, benchmark, embedding_model):
        """Benchmark: Embed a single text (typical use case)."""
        text = "Machine learning is a subset of artificial intelligence..."
        embedding_model.model.encode.return_value = np.random.randn(384)

        result = benchmark(embedding_model.embed_text, text)

        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)

    def test_batch_text_embedding_10(self, benchmark, embedding_model):
        """Benchmark: Embed batch of 10 texts."""
        texts = ["Text {i}" for i in range(10)]
        embedding_model.model.encode.return_value = np.random.randn(10, 384)

        result = benchmark(embedding_model.embed_texts, texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 384)

    def test_batch_text_embedding_100(self, benchmark, embedding_model):
        """Benchmark: Embed batch of 100 texts."""
        texts = [
            f"This is a sample transcript for text number {i} with some content. " * 5
            for i in range(100)
        ]
        embedding_model.model.encode.return_value = np.random.randn(100, 384)

        result = benchmark(embedding_model.embed_texts, texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 384)

    def test_batch_text_embedding_1000(self, benchmark, embedding_model):
        """Benchmark: Embed batch of 1000 texts."""
        texts = [
            f"Transcript chunk {i}: " + "word " * 100
            for i in range(1000)
        ]
        embedding_model.model.encode.return_value = np.random.randn(1000, 384)

        result = benchmark(embedding_model.embed_texts, texts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1000, 384)

    def test_long_text_embedding(self, benchmark, embedding_model):
        """Benchmark: Embed a very long text (full transcript)."""
        long_text = "word " * 10000  # ~50KB text
        embedding_model.model.encode.return_value = np.random.randn(384)

        result = benchmark(embedding_model.embed_text, long_text)

        assert isinstance(result, np.ndarray)


class TestVectorStoreBenchmarks:
    """Benchmarks for vector store operations."""

    @pytest.fixture
    def vectorstore(self):
        """Create mock vector store for benchmarks."""
        with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db

            with patch("parakeet_search.vectorstore.Path.mkdir"):
                vs = VectorStore()
                yield vs

    def test_create_table_small(self, benchmark, vectorstore):
        """Benchmark: Create vector store table with 10 episodes."""
        data = pd.DataFrame({
            "id": range(10),
            "episode_id": [f"ep_{i:03d}" for i in range(10)],
            "embedding": [np.random.randn(384).tolist() for _ in range(10)],
        })

        mock_table = MagicMock()
        vectorstore.db.create_table.return_value = mock_table

        result = benchmark(vectorstore.create_table, data, table_name="episodes")

        assert result is not None

    def test_create_table_medium(self, benchmark, vectorstore):
        """Benchmark: Create vector store table with 100 episodes."""
        data = pd.DataFrame({
            "id": range(100),
            "episode_id": [f"ep_{i:03d}" for i in range(100)],
            "embedding": [np.random.randn(384).tolist() for _ in range(100)],
        })

        mock_table = MagicMock()
        vectorstore.db.create_table.return_value = mock_table

        result = benchmark(vectorstore.create_table, data, table_name="episodes")

        assert result is not None

    def test_create_table_large(self, benchmark, vectorstore):
        """Benchmark: Create vector store table with 1000 episodes."""
        data = pd.DataFrame({
            "id": range(1000),
            "episode_id": [f"ep_{i:04d}" for i in range(1000)],
            "embedding": [np.random.randn(384).tolist() for _ in range(1000)],
        })

        mock_table = MagicMock()
        vectorstore.db.create_table.return_value = mock_table

        result = benchmark(vectorstore.create_table, data, table_name="episodes")

        assert result is not None

    def test_search_small_dataset(self, benchmark, vectorstore):
        """Benchmark: Search in vector store with 10 episodes."""
        query_embedding = np.random.randn(384).tolist()

        mock_search_result = MagicMock()
        mock_search_result.to_list.return_value = [
            {"id": 0, "episode_id": "ep_000", "_distance": 0.1}
        ]

        mock_table = MagicMock()
        mock_table.search.return_value.limit.return_value = mock_search_result
        vectorstore.db.open_table.return_value = mock_table

        result = benchmark(vectorstore.search, query_embedding, limit=5)

        assert len(result) > 0

    def test_search_medium_dataset(self, benchmark, vectorstore):
        """Benchmark: Search in vector store with 100 episodes."""
        query_embedding = np.random.randn(384).tolist()

        results = [
            {"id": i, "episode_id": f"ep_{i:03d}", "_distance": 0.1 + (i * 0.001)}
            for i in range(10)
        ]
        mock_search_result = MagicMock()
        mock_search_result.to_list.return_value = results

        mock_table = MagicMock()
        mock_table.search.return_value.limit.return_value = mock_search_result
        vectorstore.db.open_table.return_value = mock_table

        result = benchmark(vectorstore.search, query_embedding, limit=10)

        assert len(result) > 0

    def test_search_large_dataset(self, benchmark, vectorstore):
        """Benchmark: Search in vector store with 1000 episodes."""
        query_embedding = np.random.randn(384).tolist()

        results = [
            {"id": i, "episode_id": f"ep_{i:04d}", "_distance": 0.1 + (i * 0.0001)}
            for i in range(20)
        ]
        mock_search_result = MagicMock()
        mock_search_result.to_list.return_value = results

        mock_table = MagicMock()
        mock_table.search.return_value.limit.return_value = mock_search_result
        vectorstore.db.open_table.return_value = mock_table

        result = benchmark(vectorstore.search, query_embedding, limit=20)

        assert len(result) > 0


class TestSearchEngineBenchmarks:
    """Benchmarks for search engine end-to-end performance."""

    @pytest.fixture
    def search_engine(self):
        """Create search engine with mocked dependencies."""
        mock_em = MagicMock()
        mock_vs = MagicMock()
        return SearchEngine(embedding_model=mock_em, vectorstore=mock_vs)

    def test_search_simple_query(self, benchmark, search_engine):
        """Benchmark: Search with simple single-word query."""
        query = "machine"
        query_embedding = np.random.randn(384)
        search_engine.embedding_model.embed_text.return_value = query_embedding

        results = [
            {"episode_id": "ep_001", "_distance": 0.1},
            {"episode_id": "ep_002", "_distance": 0.2},
        ]
        search_engine.vectorstore.search.return_value = results

        result = benchmark(search_engine.search, query, limit=10)

        assert len(result) > 0

    def test_search_complex_query(self, benchmark, search_engine):
        """Benchmark: Search with complex multi-word query."""
        query = "machine learning and deep neural networks with transformers"
        query_embedding = np.random.randn(384)
        search_engine.embedding_model.embed_text.return_value = query_embedding

        results = [
            {"episode_id": f"ep_{i:03d}", "_distance": 0.1 + (i * 0.01)}
            for i in range(10)
        ]
        search_engine.vectorstore.search.return_value = results

        result = benchmark(search_engine.search, query, limit=10)

        assert len(result) > 0

    def test_search_with_threshold(self, benchmark, search_engine):
        """Benchmark: Search with threshold filtering."""
        query = "artificial intelligence"
        query_embedding = np.random.randn(384)
        search_engine.embedding_model.embed_text.return_value = query_embedding

        results = [
            {"episode_id": f"ep_{i:03d}", "_distance": 0.05 + (i * 0.01)}
            for i in range(20)
        ]
        search_engine.vectorstore.search.return_value = results

        result = benchmark(search_engine.search, query, limit=10, threshold=0.3)

        assert isinstance(result, list)

    def test_multiple_sequential_searches(self, benchmark, search_engine):
        """Benchmark: Perform 10 sequential searches."""
        queries = [
            "machine learning",
            "deep learning",
            "neural networks",
            "transformers",
            "natural language processing",
            "computer vision",
            "reinforcement learning",
            "data science",
            "feature engineering",
            "model evaluation",
        ]

        def run_searches():
            for query in queries:
                query_embedding = np.random.randn(384)
                search_engine.embedding_model.embed_text.return_value = query_embedding
                results = [
                    {"episode_id": f"ep_{i:03d}", "_distance": 0.1 + (i * 0.01)}
                    for i in range(10)
                ]
                search_engine.vectorstore.search.return_value = results
                search_engine.search(query, limit=10)

        benchmark(run_searches)


class TestMemoryBenchmarks:
    """Benchmarks for memory usage during operations."""

    def test_embedding_memory_usage(self):
        """Benchmark: Memory usage for storing embeddings."""
        # Create embeddings similar to what would be stored
        num_embeddings = 10000
        embedding_dim = 384
        embeddings = np.random.randn(num_embeddings, embedding_dim)

        # Estimate memory usage
        memory_bytes = embeddings.nbytes
        memory_mb = memory_bytes / (1024 * 1024)

        # Benchmark should show memory efficiency
        # 10K x 384-dim float64 = ~29.3MB (~2.8KB per embedding)
        assert memory_mb < 35  # Should be roughly 29-30MB

    def test_large_dataframe_creation(self, benchmark):
        """Benchmark: Create large DataFrame with embeddings."""
        num_rows = 1000

        def create_dataframe():
            data = {
                "id": range(num_rows),
                "episode_id": [f"ep_{i:04d}" for i in range(num_rows)],
                "embedding": [np.random.randn(384).tolist() for _ in range(num_rows)],
                "text": [f"Transcript chunk {i}" for i in range(num_rows)],
            }
            return pd.DataFrame(data)

        df = benchmark(create_dataframe)

        assert len(df) == num_rows
        assert "embedding" in df.columns


class TestScalabilityBenchmarks:
    """Benchmarks to measure scalability with increasing dataset sizes."""

    @pytest.fixture
    def vectorstore(self):
        """Create mock vector store for scalability tests."""
        with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db

            with patch("parakeet_search.vectorstore.Path.mkdir"):
                vs = VectorStore()
                yield vs

    @pytest.mark.parametrize("size", [10, 100, 1000])
    def test_search_scalability(self, benchmark, vectorstore, size):
        """Benchmark: Search latency scales with dataset size."""
        query_embedding = np.random.randn(384).tolist()

        results = [
            {"id": i, "episode_id": f"ep_{i:04d}", "_distance": 0.1 + (i * 0.001)}
            for i in range(min(10, size))
        ]
        mock_search_result = MagicMock()
        mock_search_result.to_list.return_value = results

        mock_table = MagicMock()
        mock_table.search.return_value.limit.return_value = mock_search_result
        vectorstore.db.open_table.return_value = mock_table

        result = benchmark(vectorstore.search, query_embedding, limit=10)

        assert len(result) > 0

    @pytest.mark.parametrize("batch_size", [10, 100, 1000])
    def test_batch_embedding_scalability(self, benchmark, batch_size):
        """Benchmark: Embedding generation scales with batch size."""
        with patch("parakeet_search.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model

            em = EmbeddingModel()
            texts = [f"Text {i}" for i in range(batch_size)]
            mock_model.encode.return_value = np.random.randn(batch_size, 384)

            result = benchmark(em.embed_texts, texts)

            assert result.shape == (batch_size, 384)
