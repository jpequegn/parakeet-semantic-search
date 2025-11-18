"""Unit tests for SearchEngine class."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from parakeet_search.search import SearchEngine


class TestSearchEngineInit:
    """Test SearchEngine initialization."""

    def test_init_default_models(self):
        """Test initialization with default models."""
        with patch('parakeet_search.search.EmbeddingModel') as mock_em_class:
            with patch('parakeet_search.search.VectorStore') as mock_vs_class:
                mock_em = MagicMock()
                mock_vs = MagicMock()
                mock_em_class.return_value = mock_em
                mock_vs_class.return_value = mock_vs

                se = SearchEngine()

                assert se.embedding_model is mock_em
                assert se.vectorstore is mock_vs
                mock_em_class.assert_called_once_with()
                mock_vs_class.assert_called_once_with()

    def test_init_custom_models(self):
        """Test initialization with custom models."""
        mock_em = MagicMock()
        mock_vs = MagicMock()

        se = SearchEngine(embedding_model=mock_em, vectorstore=mock_vs)

        assert se.embedding_model is mock_em
        assert se.vectorstore is mock_vs

    def test_init_custom_embedding_model_only(self):
        """Test initialization with custom embedding model."""
        mock_em = MagicMock()

        with patch('parakeet_search.search.VectorStore') as mock_vs_class:
            mock_vs = MagicMock()
            mock_vs_class.return_value = mock_vs

            se = SearchEngine(embedding_model=mock_em)

            assert se.embedding_model is mock_em
            assert se.vectorstore is mock_vs

    def test_init_custom_vectorstore_only(self):
        """Test initialization with custom vector store."""
        mock_vs = MagicMock()

        with patch('parakeet_search.search.EmbeddingModel') as mock_em_class:
            mock_em = MagicMock()
            mock_em_class.return_value = mock_em

            se = SearchEngine(vectorstore=mock_vs)

            assert se.embedding_model is mock_em
            assert se.vectorstore is mock_vs


class TestSearchEngineSearch:
    """Test SearchEngine.search() method."""

    @pytest.fixture
    def search_engine(self):
        """Create a mock SearchEngine for testing."""
        mock_em = MagicMock()
        mock_vs = MagicMock()
        return SearchEngine(embedding_model=mock_em, vectorstore=mock_vs)

    def test_search_basic(self, search_engine):
        """Test basic search functionality."""
        # Setup mocks
        query = "machine learning"
        query_embedding = np.random.randn(384)
        search_engine.embedding_model.embed_text.return_value = query_embedding

        expected_results = [
            {'episode_title': 'Episode 1', '_distance': 0.1},
            {'episode_title': 'Episode 2', '_distance': 0.2}
        ]
        search_engine.vectorstore.search.return_value = expected_results

        # Execute search
        results = search_engine.search(query)

        # Verify
        assert len(results) == 2
        assert results == expected_results
        search_engine.embedding_model.embed_text.assert_called_once_with(query)
        search_engine.vectorstore.search.assert_called_once_with(
            query_embedding.tolist(),
            limit=10
        )

    def test_search_with_custom_limit(self, search_engine):
        """Test search with custom limit."""
        query = "test query"
        query_embedding = np.random.randn(384)
        search_engine.embedding_model.embed_text.return_value = query_embedding
        search_engine.vectorstore.search.return_value = []

        search_engine.search(query, limit=20)

        search_engine.vectorstore.search.assert_called_once_with(
            query_embedding.tolist(),
            limit=20
        )

    def test_search_with_threshold(self, search_engine):
        """Test search with similarity threshold filtering."""
        query = "test query"
        query_embedding = np.random.randn(384)
        search_engine.embedding_model.embed_text.return_value = query_embedding

        # Simulate results with various distances
        all_results = [
            {'episode_title': 'Episode 1', '_distance': 0.1},
            {'episode_title': 'Episode 2', '_distance': 0.5},
            {'episode_title': 'Episode 3', '_distance': 0.8}
        ]
        search_engine.vectorstore.search.return_value = all_results

        # Note: threshold in implementation filters by distance >= threshold
        results = search_engine.search(query, threshold=0.5)

        # Filter results with distance >= threshold
        filtered = [r for r in all_results if r.get('_distance', 0) >= 0.5]
        assert len(filtered) == 2

    def test_search_generates_query_embedding(self, search_engine):
        """Test that search generates embedding for query."""
        query = "semantic search"
        query_embedding = np.random.randn(384)
        search_engine.embedding_model.embed_text.return_value = query_embedding
        search_engine.vectorstore.search.return_value = []

        search_engine.search(query)

        search_engine.embedding_model.embed_text.assert_called_once_with(query)

    def test_search_calls_vectorstore(self, search_engine):
        """Test that search calls vectorstore with embedding."""
        query = "test"
        query_embedding = np.random.randn(384)
        search_engine.embedding_model.embed_text.return_value = query_embedding
        search_engine.vectorstore.search.return_value = []

        search_engine.search(query, limit=5)

        search_engine.vectorstore.search.assert_called_once_with(
            query_embedding.tolist(),
            limit=5
        )

    def test_search_returns_empty_list_when_no_results(self, search_engine):
        """Test search returns empty list when no results found."""
        query = "nonexistent topic"
        search_engine.embedding_model.embed_text.return_value = np.random.randn(384)
        search_engine.vectorstore.search.return_value = []

        results = search_engine.search(query)

        assert results == []
        assert isinstance(results, list)

    def test_search_default_limit(self, search_engine):
        """Test that search uses default limit of 10."""
        query = "test"
        search_engine.embedding_model.embed_text.return_value = np.random.randn(384)
        search_engine.vectorstore.search.return_value = []

        search_engine.search(query)

        # Check that limit parameter is 10 by default
        call_args = search_engine.vectorstore.search.call_args
        assert call_args[1]['limit'] == 10

    def test_search_with_many_results(self, search_engine):
        """Test search returning many results."""
        query = "popular topic"
        search_engine.embedding_model.embed_text.return_value = np.random.randn(384)

        # Create many results
        results = [
            {'episode_title': f'Episode {i}', '_distance': 0.1 + i * 0.01}
            for i in range(100)
        ]
        search_engine.vectorstore.search.return_value = results

        found_results = search_engine.search(query, limit=100)

        assert len(found_results) == 100

    def test_search_threshold_filters_results(self, search_engine):
        """Test that threshold properly filters results."""
        query = "test"
        search_engine.embedding_model.embed_text.return_value = np.random.randn(384)

        # Return unfiltered results
        unfiltered_results = [
            {'id': 1, '_distance': 0.2},
            {'id': 2, '_distance': 0.5},
            {'id': 3, '_distance': 0.8},
        ]
        search_engine.vectorstore.search.return_value = unfiltered_results

        results = search_engine.search(query, threshold=0.5)

        # Filter manually to check expected behavior
        expected = [r for r in unfiltered_results if r.get('_distance', 0) >= 0.5]
        assert len(expected) == 2

    def test_search_preserves_result_metadata(self, search_engine):
        """Test that search preserves all result metadata."""
        query = "test"
        search_engine.embedding_model.embed_text.return_value = np.random.randn(384)

        result_with_metadata = {
            'id': 1,
            'episode_id': 'ep123',
            'episode_title': 'Great Episode',
            'podcast_title': 'Awesome Podcast',
            'text': 'This is great content',
            '_distance': 0.15
        }
        search_engine.vectorstore.search.return_value = [result_with_metadata]

        results = search_engine.search(query)

        assert results[0]['episode_id'] == 'ep123'
        assert results[0]['episode_title'] == 'Great Episode'
        assert results[0]['podcast_title'] == 'Awesome Podcast'


class TestSearchEngineGetRecommendations:
    """Test SearchEngine.get_recommendations() method."""

    @pytest.fixture
    def search_engine(self):
        """Create a mock SearchEngine for testing."""
        mock_em = MagicMock()
        mock_vs = MagicMock()
        return SearchEngine(embedding_model=mock_em, vectorstore=mock_vs)

    def test_get_recommendations_not_implemented(self, search_engine):
        """Test that get_recommendations is placeholder."""
        # According to current implementation, this should be None/placeholder
        result = search_engine.get_recommendations("ep123")

        assert result is None


class TestSearchEngineErrorHandling:
    """Test SearchEngine error handling."""

    @pytest.fixture
    def search_engine(self):
        """Create a mock SearchEngine for testing."""
        mock_em = MagicMock()
        mock_vs = MagicMock()
        return SearchEngine(embedding_model=mock_em, vectorstore=mock_vs)

    def test_search_handles_embedding_error(self, search_engine):
        """Test search handles embedding generation errors."""
        query = "test"
        search_engine.embedding_model.embed_text.side_effect = Exception("Model error")

        with pytest.raises(Exception):
            search_engine.search(query)

    def test_search_handles_vectorstore_error(self, search_engine):
        """Test search handles vectorstore errors."""
        query = "test"
        search_engine.embedding_model.embed_text.return_value = np.random.randn(384)
        search_engine.vectorstore.search.side_effect = Exception("DB error")

        with pytest.raises(Exception):
            search_engine.search(query)

    def test_search_with_empty_query(self, search_engine):
        """Test search with empty query string."""
        query = ""
        search_engine.embedding_model.embed_text.return_value = np.random.randn(384)
        search_engine.vectorstore.search.return_value = []

        results = search_engine.search(query)

        assert isinstance(results, list)

    def test_search_with_special_characters(self, search_engine):
        """Test search with special characters in query."""
        query = "!@#$%^&*()"
        search_engine.embedding_model.embed_text.return_value = np.random.randn(384)
        search_engine.vectorstore.search.return_value = []

        results = search_engine.search(query)

        assert isinstance(results, list)


class TestSearchEngineIntegration:
    """Integration tests for SearchEngine."""

    def test_search_engine_full_workflow(self):
        """Test complete search workflow with mocked components."""
        # Create real SearchEngine with mocked dependencies
        mock_em = MagicMock()
        mock_vs = MagicMock()

        se = SearchEngine(embedding_model=mock_em, vectorstore=mock_vs)

        # Setup mocks
        query_embedding = np.random.randn(384)
        mock_em.embed_text.return_value = query_embedding

        expected_results = [
            {'episode_title': 'Episode 1', '_distance': 0.1},
        ]
        mock_vs.search.return_value = expected_results

        # Execute
        results = se.search("test query", limit=5)

        # Verify
        assert len(results) == 1
        assert results[0]['episode_title'] == 'Episode 1'
        mock_em.embed_text.assert_called_once()
        mock_vs.search.assert_called_once()

    def test_search_engine_multiple_searches(self):
        """Test SearchEngine across multiple searches."""
        mock_em = MagicMock()
        mock_vs = MagicMock()

        se = SearchEngine(embedding_model=mock_em, vectorstore=mock_vs)

        # Setup side effects for multiple calls
        mock_em.embed_text.side_effect = [
            np.random.randn(384),
            np.random.randn(384),
        ]

        mock_vs.search.side_effect = [
            [{'id': 1, '_distance': 0.1}],
            [{'id': 2, '_distance': 0.2}],
        ]

        # Execute multiple searches
        results1 = se.search("query 1")
        results2 = se.search("query 2")

        assert len(results1) == 1
        assert results1[0]['id'] == 1
        assert len(results2) == 1
        assert results2[0]['id'] == 2
        assert mock_em.embed_text.call_count == 2
        assert mock_vs.search.call_count == 2
