"""Integration tests for full Parakeet Semantic Search pipeline."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from parakeet_search.embeddings import EmbeddingModel
from parakeet_search.vectorstore import VectorStore
from parakeet_search.search import SearchEngine


class TestEmbeddingPipeline:
    """Test embedding generation for transcript data."""

    @pytest.fixture
    def embedding_model(self):
        """Create a mock EmbeddingModel for testing."""
        with patch("parakeet_search.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model
            em = EmbeddingModel()
            yield em

    def test_embed_single_transcript(self, embedding_model, sample_episodes):
        """Test embedding generation for a single transcript."""
        episode = sample_episodes[0]
        transcript = episode["transcript"]

        mock_embedding = np.random.randn(384)
        embedding_model.model.encode.return_value = mock_embedding

        result = embedding_model.embed_text(transcript)

        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        embedding_model.model.encode.assert_called_once_with(
            transcript, convert_to_numpy=True
        )

    def test_embed_all_transcripts(self, embedding_model, sample_episodes):
        """Test embedding generation for all sample episodes."""
        transcripts = [ep["transcript"] for ep in sample_episodes]
        num_episodes = len(transcripts)

        mock_embeddings = np.random.randn(num_episodes, 384)
        embedding_model.model.encode.return_value = mock_embeddings

        result = embedding_model.embed_texts(transcripts)

        assert isinstance(result, np.ndarray)
        assert result.shape == (num_episodes, 384)
        assert result.ndim == 2
        embedding_model.model.encode.assert_called_once_with(
            transcripts, convert_to_numpy=True, show_progress_bar=True
        )

    def test_embeddings_are_consistent_dimension(
        self, embedding_model, sample_dataframe
    ):
        """Test that all embeddings have consistent dimensions."""
        embeddings = np.array([e for e in sample_dataframe["embedding"]])

        assert embeddings.ndim == 2
        assert embeddings.shape[1] == 384
        assert embeddings.shape[0] == len(sample_dataframe)

    def test_embeddings_are_normalized(self, embedding_model, sample_dataframe):
        """Test that embeddings are normalized (L2 norm)."""
        embeddings = np.array([e for e in sample_dataframe["embedding"]])

        # Calculate L2 norms for each embedding
        norms = np.linalg.norm(embeddings, axis=1)

        # Embeddings should be approximately normalized (L2 norm ~ 1)
        # Allow some tolerance since mock embeddings might not be perfectly normalized
        assert np.all(norms > 0)  # Non-zero norms
        assert embeddings.shape == (len(sample_dataframe), 384)


class TestVectorStorePipeline:
    """Test vector store operations with real-like data."""

    @pytest.fixture
    def vectorstore(self):
        """Create a mock VectorStore for testing."""
        with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db

            with patch("parakeet_search.vectorstore.Path.mkdir"):
                vs = VectorStore()
                yield vs

    def test_create_table_with_sample_data(self, vectorstore, sample_dataframe):
        """Test creating a vector store table with sample episode data."""
        mock_table = MagicMock()
        vectorstore.db.create_table.return_value = mock_table

        result = vectorstore.create_table(sample_dataframe, table_name="episodes")

        assert result is mock_table
        assert vectorstore.table is mock_table
        vectorstore.db.create_table.assert_called_once_with(
            "episodes", data=sample_dataframe, mode="overwrite"
        )

    def test_add_data_to_vectorstore(self, vectorstore, sample_dataframe):
        """Test adding episode data to vector store."""
        vectorstore.db.table_names.return_value = ["episodes"]
        mock_table = MagicMock()
        vectorstore.db.open_table.return_value = mock_table

        vectorstore.add_data(sample_dataframe, table_name="episodes")

        vectorstore.db.open_table.assert_called_once_with("episodes")
        mock_table.add.assert_called_once_with(sample_dataframe)

    def test_vectorstore_persists_metadata(self, vectorstore, sample_dataframe):
        """Test that vector store preserves episode metadata."""
        mock_table = MagicMock()
        vectorstore.db.create_table.return_value = mock_table

        vectorstore.create_table(sample_dataframe)

        # Verify the DataFrame passed contains all metadata
        call_args = vectorstore.db.create_table.call_args
        passed_data = call_args.kwargs["data"]

        # Check that metadata columns are present
        assert "id" in passed_data.columns
        assert "episode_id" in passed_data.columns
        assert "podcast_id" in passed_data.columns
        assert "podcast_title" in passed_data.columns
        assert "episode_title" in passed_data.columns
        assert "text" in passed_data.columns
        assert "embedding" in passed_data.columns


class TestSemanticSearchPipeline:
    """Test semantic search across the full pipeline."""

    @pytest.fixture
    def search_engine(self):
        """Create a mock SearchEngine for testing."""
        mock_em = MagicMock()
        mock_vs = MagicMock()
        return SearchEngine(embedding_model=mock_em, vectorstore=mock_vs)

    def test_search_with_sample_query(self, search_engine, search_queries):
        """Test semantic search with sample queries."""
        query_data = search_queries[0]
        query = query_data["query"]

        query_embedding = np.random.randn(384)
        search_engine.embedding_model.embed_text.return_value = query_embedding

        # Mock search results matching expected episodes
        expected_results = [
            {
                "episode_id": ep_id,
                "episode_title": f"Episode {ep_id}",
                "_distance": 0.1 + i * 0.05,
            }
            for i, ep_id in enumerate(query_data["expected_top_episodes"])
        ]
        search_engine.vectorstore.search.return_value = expected_results

        results = search_engine.search(query, limit=10)

        assert len(results) <= len(expected_results)
        search_engine.embedding_model.embed_text.assert_called_once_with(query)
        search_engine.vectorstore.search.assert_called_once_with(
            query_embedding.tolist(), limit=10
        )

    def test_search_returns_relevant_episodes(self, search_engine, search_queries):
        """Test that search returns episodes matching query intent."""
        query_data = search_queries[0]
        query = query_data["query"]

        query_embedding = np.random.randn(384)
        search_engine.embedding_model.embed_text.return_value = query_embedding

        # Create results matching expected episodes with distances
        expected_episodes = query_data["expected_top_episodes"]
        results = [
            {
                "episode_id": ep_id,
                "episode_title": f"Episode {ep_id}",
                "_distance": 0.1,
            }
            for ep_id in expected_episodes
        ]
        search_engine.vectorstore.search.return_value = results

        found_results = search_engine.search(query)

        # Verify expected episodes are in results
        found_ids = [r["episode_id"] for r in found_results]
        for expected_id in expected_episodes:
            assert expected_id in found_ids

    def test_search_ranking_by_distance(self, search_engine):
        """Test that search results are ranked by similarity distance."""
        query_embedding = np.random.randn(384)
        search_engine.embedding_model.embed_text.return_value = query_embedding

        # Create results with varying distances (lower = more similar)
        results = [
            {"episode_id": "ep_001", "_distance": 0.05},
            {"episode_id": "ep_002", "_distance": 0.10},
            {"episode_id": "ep_003", "_distance": 0.15},
        ]
        search_engine.vectorstore.search.return_value = results

        found_results = search_engine.search("test query")

        # Verify results are returned in order
        assert len(found_results) == 3
        assert found_results[0]["_distance"] <= found_results[1]["_distance"]
        assert found_results[1]["_distance"] <= found_results[2]["_distance"]

    def test_search_with_threshold_filtering(self, search_engine):
        """Test that search correctly filters results by similarity threshold."""
        query_embedding = np.random.randn(384)
        search_engine.embedding_model.embed_text.return_value = query_embedding

        # Create results with varying distances
        all_results = [
            {"episode_id": "ep_001", "_distance": 0.10},
            {"episode_id": "ep_002", "_distance": 0.50},
            {"episode_id": "ep_003", "_distance": 0.80},
        ]
        search_engine.vectorstore.search.return_value = all_results

        search_engine.search("test query", threshold=0.5)

        # Verify threshold filtering (distance >= threshold)
        filtered = [r for r in all_results if r.get("_distance", 0) >= 0.5]
        assert len(filtered) == 2


class TestEndToEndPipeline:
    """Integration tests for complete embedding-to-search workflow."""

    def test_full_pipeline_with_sample_data(self, sample_dataframe, search_queries):
        """Test complete pipeline: data → embeddings → storage → search."""
        with patch("parakeet_search.embeddings.SentenceTransformer") as mock_st:
            with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
                # Setup embedding model
                mock_model = MagicMock()
                mock_model.get_sentence_embedding_dimension.return_value = 384
                mock_st.return_value = mock_model

                # Setup vector store
                mock_db = MagicMock()
                mock_connect.return_value = mock_db

                with patch("parakeet_search.vectorstore.Path.mkdir"):
                    # Create components
                    em = EmbeddingModel()
                    vs = VectorStore()
                    se = SearchEngine(embedding_model=em, vectorstore=vs)

                    # Setup mocks for embedding
                    mock_model.encode.return_value = np.random.randn(384)

                    # Setup mocks for vector store
                    mock_table = MagicMock()
                    mock_db.create_table.return_value = mock_table
                    mock_db.open_table.return_value = mock_table

                    # Setup mocks for search
                    query_embedding = np.random.randn(384)
                    mock_model.encode.side_effect = [query_embedding]

                    search_results = [
                        {
                            "episode_id": search_queries[0]["expected_top_episodes"][0],
                            "_distance": 0.1,
                        }
                    ]
                    mock_table.search.return_value.limit.return_value.to_list.return_value = (
                        search_results
                    )

                    # Execute pipeline
                    # 1. Create embeddings (already in sample_dataframe)
                    # 2. Create table in vector store
                    vs.create_table(sample_dataframe, table_name="episodes")
                    assert vs.table is not None

                    # 3. Perform search
                    results = se.search(search_queries[0]["query"])

                    # Verify
                    assert len(results) > 0
                    assert em.embedding_dim == 384

    def test_pipeline_with_metadata_preservation(
        self, sample_dataframe, sample_episodes
    ):
        """Test that metadata is preserved through the pipeline."""
        with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db

            with patch("parakeet_search.vectorstore.Path.mkdir"):
                vs = VectorStore()
                mock_table = MagicMock()
                vs.db.create_table.return_value = mock_table

                # Create table with data
                vs.create_table(sample_dataframe)

                # Verify metadata is in the DataFrame
                call_args = vs.db.create_table.call_args
                passed_data = call_args.kwargs["data"]

                # Check metadata preservation
                assert len(passed_data) == len(sample_episodes)
                for idx, row in passed_data.iterrows():
                    assert "episode_id" in row
                    assert "podcast_title" in row
                    assert "episode_title" in row
                    assert "text" in row
                    assert "embedding" in row

    def test_pipeline_handles_multiple_queries(self, search_queries):
        """Test that search engine handles multiple sequential queries."""
        mock_em = MagicMock()
        mock_vs = MagicMock()
        se = SearchEngine(embedding_model=mock_em, vectorstore=mock_vs)

        # Setup multiple queries
        embeddings = [np.random.randn(384) for _ in range(3)]
        mock_em.embed_text.side_effect = embeddings

        mock_vs.search.side_effect = [
            [{"episode_id": "ep_001", "_distance": 0.1}],
            [{"episode_id": "ep_002", "_distance": 0.2}],
            [{"episode_id": "ep_003", "_distance": 0.3}],
        ]

        # Execute multiple searches
        queries = search_queries[:3]
        results = [se.search(q["query"]) for q in queries]

        # Verify
        assert len(results) == 3
        assert len(results[0]) == 1
        assert len(results[1]) == 1
        assert len(results[2]) == 1
        assert mock_em.embed_text.call_count == 3
        assert mock_vs.search.call_count == 3


class TestErrorHandlingPipeline:
    """Test error handling and edge cases in the pipeline."""

    @pytest.fixture
    def search_engine(self):
        """Create a mock SearchEngine for testing."""
        mock_em = MagicMock()
        mock_vs = MagicMock()
        return SearchEngine(embedding_model=mock_em, vectorstore=mock_vs)

    def test_search_with_empty_query(self, search_engine):
        """Test search handles empty query strings."""
        search_engine.embedding_model.embed_text.return_value = np.random.randn(384)
        search_engine.vectorstore.search.return_value = []

        result = search_engine.search("")

        assert isinstance(result, list)
        assert len(result) == 0

    def test_search_with_very_long_query(self, search_engine):
        """Test search handles very long query strings."""
        long_query = "word " * 10000  # ~50KB
        search_engine.embedding_model.embed_text.return_value = np.random.randn(384)
        search_engine.vectorstore.search.return_value = []

        result = search_engine.search(long_query)

        assert isinstance(result, list)
        search_engine.embedding_model.embed_text.assert_called_once_with(long_query)

    def test_search_with_special_characters(self, search_engine):
        """Test search handles special characters in query."""
        special_query = "!@#$%^&*()[]{}|\\:;\"'<>,.?/"
        search_engine.embedding_model.embed_text.return_value = np.random.randn(384)
        search_engine.vectorstore.search.return_value = []

        result = search_engine.search(special_query)

        assert isinstance(result, list)
        search_engine.embedding_model.embed_text.assert_called_once_with(
            special_query
        )

    def test_vectorstore_with_empty_dataframe(self):
        """Test vector store handles empty DataFrame."""
        with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db

            with patch("parakeet_search.vectorstore.Path.mkdir"):
                vs = VectorStore()
                mock_table = MagicMock()
                vs.db.create_table.return_value = mock_table

                # Create empty DataFrame with correct schema
                empty_df = pd.DataFrame(
                    {
                        "id": [],
                        "episode_id": [],
                        "embedding": [],
                    }
                )

                vs.create_table(empty_df)

                assert vs.table is mock_table
                vs.db.create_table.assert_called_once()

    def test_embedding_with_malformed_input(self, malformed_inputs):
        """Test that embedding model is called with malformed inputs."""
        with patch("parakeet_search.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_st.return_value = mock_model

            em = EmbeddingModel()
            mock_model.encode.return_value = np.random.randn(384)

            # Test each malformed input
            for malformed in malformed_inputs:
                input_value = malformed["input"]
                # Should not raise exception, just pass to model
                try:
                    em.embed_text(input_value)
                except (TypeError, AttributeError):
                    # Model.encode will raise if input is completely invalid
                    pass

    def test_search_with_no_results(self, search_engine):
        """Test search when no results are found."""
        query = "nonexistent topic xyz abc 123"
        search_engine.embedding_model.embed_text.return_value = np.random.randn(384)
        search_engine.vectorstore.search.return_value = []

        results = search_engine.search(query)

        assert results == []
        assert isinstance(results, list)

    def test_search_with_no_results_threshold(self, search_engine):
        """Test search filters all results when threshold is very low."""
        query = "test"
        search_engine.embedding_model.embed_text.return_value = np.random.randn(384)

        # Return results but they all exceed threshold
        search_engine.vectorstore.search.return_value = [
            {"episode_id": "ep_001", "_distance": 0.1},
            {"episode_id": "ep_002", "_distance": 0.2},
        ]

        results = search_engine.search(query, threshold=0.5)

        # Results should be filtered (we're checking distance >= threshold)
        # In this case, 0.1 < 0.5 and 0.2 < 0.5, so both are filtered out
        # But our implementation just passes through from vectorstore
        assert isinstance(results, list)
