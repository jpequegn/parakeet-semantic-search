"""Tests for recommendation functionality."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from parakeet_search.search import SearchEngine
from parakeet_search.cli import cli


class TestRecommendationBackend:
    """Test SearchEngine.get_recommendations() method."""

    @pytest.fixture
    def mock_vectorstore(self, sample_dataframe):
        """Create a mock vectorstore with sample data."""
        with patch("parakeet_search.search.VectorStore") as MockVectorStore:
            mock_instance = MagicMock()
            MockVectorStore.return_value = mock_instance

            # Setup mock table
            mock_table = MagicMock()
            mock_instance.get_table.return_value = mock_table
            mock_instance.table = mock_table

            # Mock search_where to find episodes by ID
            def search_where_impl(where_clause):
                """Mock implementation of search_where."""
                # Extract episode_id from WHERE clause
                for record in sample_dataframe.to_dict("records"):
                    if f'episode_id = "{record["episode_id"]}"' == where_clause:
                        mock_result = MagicMock()
                        mock_result.limit.return_value.to_list.return_value = [record]
                        return mock_result

                # Return empty if not found
                mock_result = MagicMock()
                mock_result.limit.return_value.to_list.return_value = []
                return mock_result

            mock_table.search_where = search_where_impl

            yield mock_instance

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        with patch("parakeet_search.search.EmbeddingModel") as MockEmbedding:
            mock_instance = MagicMock()
            MockEmbedding.return_value = mock_instance
            yield mock_instance

    def test_get_recommendations_basic(self, mock_vectorstore, mock_embedding_model, sample_dataframe):
        """Test basic recommendation retrieval."""
        # Setup mock search to return similar episodes
        mock_vectorstore.search.return_value = [
            sample_dataframe.iloc[1].to_dict(),  # ep_002
            sample_dataframe.iloc[2].to_dict(),  # ep_003
        ]

        engine = SearchEngine()
        results = engine.get_recommendations("ep_001", limit=2)

        assert len(results) == 2
        assert results[0]["episode_id"] == "ep_002"
        assert results[1]["episode_id"] == "ep_003"

    def test_get_recommendations_excludes_source_episode(
        self, mock_vectorstore, mock_embedding_model, sample_dataframe
    ):
        """Test that source episode is excluded from results."""
        # Setup mock search to return source + others
        mock_vectorstore.search.return_value = [
            sample_dataframe.iloc[0].to_dict(),  # ep_001 (source)
            sample_dataframe.iloc[1].to_dict(),  # ep_002
            sample_dataframe.iloc[2].to_dict(),  # ep_003
        ]

        engine = SearchEngine()
        results = engine.get_recommendations("ep_001", limit=2, exclude_episode=True)

        # Should exclude ep_001
        assert len(results) == 2
        assert all(r["episode_id"] != "ep_001" for r in results)
        assert results[0]["episode_id"] == "ep_002"

    def test_get_recommendations_respects_limit(
        self, mock_vectorstore, mock_embedding_model, sample_dataframe
    ):
        """Test that limit parameter is respected."""
        # Setup mock search to return many results
        mock_vectorstore.search.return_value = [
            sample_dataframe.iloc[i].to_dict() for i in range(1, 6)
        ]

        engine = SearchEngine()
        results = engine.get_recommendations("ep_001", limit=3)

        assert len(results) <= 3

    def test_get_recommendations_with_podcast_filter(
        self, mock_vectorstore, mock_embedding_model, sample_dataframe
    ):
        """Test filtering recommendations by podcast."""
        # Setup mock search to return episodes from multiple podcasts
        mock_vectorstore.search.return_value = [
            sample_dataframe.iloc[1].to_dict(),  # ep_002, pod_001
            sample_dataframe.iloc[3].to_dict(),  # ep_004, pod_002
            sample_dataframe.iloc[4].to_dict(),  # ep_005, pod_003
        ]

        engine = SearchEngine()
        results = engine.get_recommendations(
            "ep_001",
            limit=5,
            podcast_id="pod_001",
        )

        # Should only return episodes from pod_001
        assert all(r["podcast_id"] == "pod_001" for r in results)

    def test_get_recommendations_episode_not_found(
        self, mock_vectorstore, mock_embedding_model
    ):
        """Test error when episode not found."""
        engine = SearchEngine()

        with pytest.raises(ValueError, match="not found"):
            engine.get_recommendations("ep_nonexistent")

    def test_get_recommendations_vectorstore_not_initialized(self, mock_embedding_model):
        """Test error when vectorstore not initialized."""
        with patch("parakeet_search.search.VectorStore") as MockVectorStore:
            mock_instance = MagicMock()
            mock_instance.table = None  # Not initialized
            MockVectorStore.return_value = mock_instance

            engine = SearchEngine()

            with pytest.raises(RuntimeError, match="not initialized"):
                engine.get_recommendations("ep_001")

    def test_get_recommendations_no_embedding(self, mock_embedding_model):
        """Test error when episode has no embedding."""
        # Create fresh mock for this test
        with patch("parakeet_search.search.VectorStore") as MockVectorStore:
            mock_instance = MagicMock()
            MockVectorStore.return_value = mock_instance
            mock_instance.table = MagicMock()

            # Mock search_where to return episode without embedding
            mock_table = mock_instance.get_table.return_value
            search_result_mock = MagicMock()
            search_result_mock.limit.return_value.to_list.return_value = [
                {"episode_id": "ep_001", "embedding": None}
            ]
            mock_table.search_where.return_value = search_result_mock

            engine = SearchEngine()

            with pytest.raises(ValueError, match="No embedding"):
                engine.get_recommendations("ep_001")

    def test_get_recommendations_empty_results(
        self, mock_vectorstore, mock_embedding_model, sample_dataframe
    ):
        """Test handling empty results."""
        # Setup mock search to return only source episode (which gets filtered)
        mock_vectorstore.search.return_value = [
            sample_dataframe.iloc[0].to_dict(),  # ep_001 (source, will be filtered)
        ]

        engine = SearchEngine()
        results = engine.get_recommendations("ep_001", limit=5, exclude_episode=True)

        assert len(results) == 0

    def test_get_recommendations_respects_search_limit_plus_one(
        self, mock_vectorstore, mock_embedding_model, sample_dataframe
    ):
        """Test that search requests limit+1 to account for excluded source."""
        mock_vectorstore.search.return_value = [
            sample_dataframe.iloc[0].to_dict(),
            sample_dataframe.iloc[1].to_dict(),
            sample_dataframe.iloc[2].to_dict(),
        ]

        engine = SearchEngine()
        engine.get_recommendations("ep_001", limit=2, exclude_episode=True)

        # Should have called search with limit=3 (limit+1 for exclusion)
        mock_vectorstore.search.assert_called_once()
        call_kwargs = mock_vectorstore.search.call_args[1]
        assert call_kwargs["limit"] == 3


class TestRecommendationCLI:
    """Test recommend CLI command."""

    @pytest.fixture
    def runner(self):
        """Create a Click CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_engine(self, sample_dataframe):
        """Mock SearchEngine with sample results."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine_class:
            mock_instance = MagicMock()
            mock_engine_class.return_value = mock_instance

            # Mock get_recommendations to return sample results
            mock_instance.get_recommendations.return_value = [
                {
                    "episode_id": "ep_002",
                    "episode_title": "Deep Learning and Neural Networks",
                    "podcast_title": "AI Today Podcast",
                    "_distance": 0.15,
                    "text": "Deep learning is...",
                },
                {
                    "episode_id": "ep_003",
                    "episode_title": "NLP Advances",
                    "podcast_title": "Tech Trends Weekly",
                    "_distance": 0.25,
                    "text": "Natural language processing...",
                },
            ]

            yield mock_instance

    def test_recommend_requires_episode_id(self, runner):
        """Test that --episode-id is required."""
        result = runner.invoke(cli, ["recommend"])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "episode-id" in result.output

    def test_recommend_basic(self, runner, mock_engine):
        """Test basic recommend command."""
        result = runner.invoke(
            cli,
            ["recommend", "--episode-id", "ep_001"],
        )

        assert result.exit_code == 0
        mock_engine.get_recommendations.assert_called_once()
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs["episode_id"] == "ep_001"
        assert call_kwargs["limit"] == 5  # default

    def test_recommend_with_custom_limit(self, runner, mock_engine):
        """Test recommend with custom limit."""
        result = runner.invoke(
            cli,
            ["recommend", "--episode-id", "ep_001", "--limit", "10"],
        )

        assert result.exit_code == 0
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs["limit"] == 10

    def test_recommend_with_podcast_filter(self, runner, mock_engine):
        """Test recommend with podcast filter."""
        result = runner.invoke(
            cli,
            ["recommend", "--episode-id", "ep_001", "--podcast-id", "pod_001"],
        )

        assert result.exit_code == 0
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs["podcast_id"] == "pod_001"

    def test_recommend_limit_validation_too_high(self, runner):
        """Test limit validation (too high)."""
        with patch("parakeet_search.cli.SearchEngine"):
            result = runner.invoke(
                cli,
                ["recommend", "--episode-id", "ep_001", "--limit", "200"],
            )
            assert result.exit_code != 0
            assert "limit must be between" in result.output

    def test_recommend_limit_validation_too_low(self, runner):
        """Test limit validation (zero/negative)."""
        with patch("parakeet_search.cli.SearchEngine"):
            result = runner.invoke(
                cli,
                ["recommend", "--episode-id", "ep_001", "--limit", "0"],
            )
            assert result.exit_code != 0

    def test_recommend_table_format(self, runner, mock_engine):
        """Test recommend with table output format."""
        result = runner.invoke(
            cli,
            ["recommend", "--episode-id", "ep_001", "--format", "table"],
        )

        assert result.exit_code == 0
        assert "Episode" in result.output or "Deep Learning" in result.output

    def test_recommend_json_format(self, runner, mock_engine):
        """Test recommend with JSON output format."""
        result = runner.invoke(
            cli,
            ["recommend", "--episode-id", "ep_001", "--format", "json"],
        )

        assert result.exit_code == 0
        # Should contain JSON structure
        assert "[" in result.output

    def test_recommend_markdown_format(self, runner, mock_engine):
        """Test recommend with markdown output format."""
        result = runner.invoke(
            cli,
            ["recommend", "--episode-id", "ep_001", "--format", "markdown"],
        )

        assert result.exit_code == 0
        assert "## Search Results" in result.output or "###" in result.output

    def test_recommend_save_results_json(self, runner, mock_engine):
        """Test saving results to JSON file."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "recommendations.json"

            result = runner.invoke(
                cli,
                [
                    "recommend",
                    "--episode-id",
                    "ep_001",
                    "--save-results",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert output_path.exists()

    def test_recommend_save_results_markdown(self, runner, mock_engine):
        """Test saving results to markdown file."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "recommendations.md"

            result = runner.invoke(
                cli,
                [
                    "recommend",
                    "--episode-id",
                    "ep_001",
                    "--save-results",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert output_path.exists()
            content = output_path.read_text()
            assert "## Search Results" in content or "Recommendations" in result.output

    def test_recommend_error_handling_invalid_episode(self, runner):
        """Test error handling for invalid episode."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine_class:
            mock_instance = MagicMock()
            mock_engine_class.return_value = mock_instance
            mock_instance.get_recommendations.side_effect = ValueError(
                "Episode not found"
            )

            result = runner.invoke(
                cli,
                ["recommend", "--episode-id", "ep_invalid"],
            )

            assert result.exit_code != 0
            assert "Invalid episode" in result.output or "Episode not found" in result.output

    def test_recommend_error_handling_runtime_error(self, runner):
        """Test error handling for runtime errors."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine_class:
            mock_instance = MagicMock()
            mock_engine_class.return_value = mock_instance
            mock_instance.get_recommendations.side_effect = RuntimeError(
                "Vector store not initialized"
            )

            result = runner.invoke(
                cli,
                ["recommend", "--episode-id", "ep_001"],
            )

            assert result.exit_code != 0
            assert "Search engine error" in result.output

    def test_recommend_no_results(self, runner):
        """Test recommend when no results found."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine_class:
            mock_instance = MagicMock()
            mock_engine_class.return_value = mock_instance
            mock_instance.get_recommendations.return_value = []

            result = runner.invoke(
                cli,
                ["recommend", "--episode-id", "ep_001"],
            )

            assert result.exit_code == 0
            assert "No recommendations found" in result.output

    def test_recommend_with_all_options(self, runner, mock_engine):
        """Test recommend with all options combined."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "recommendations.json"

            result = runner.invoke(
                cli,
                [
                    "recommend",
                    "--episode-id",
                    "ep_001",
                    "--limit",
                    "8",
                    "--podcast-id",
                    "pod_001",
                    "--format",
                    "json",
                    "--save-results",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            call_kwargs = mock_engine.get_recommendations.call_args[1]
            assert call_kwargs["episode_id"] == "ep_001"
            assert call_kwargs["limit"] == 8
            assert call_kwargs["podcast_id"] == "pod_001"

    def test_recommend_creates_parent_directory(self, runner, mock_engine):
        """Test that parent directories are created for output file."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir" / "recommendations.json"

            result = runner.invoke(
                cli,
                [
                    "recommend",
                    "--episode-id",
                    "ep_001",
                    "--save-results",
                    str(output_path),
                ],
            )

            assert result.exit_code == 0
            assert output_path.exists()


class TestRecommendationIntegration:
    """Integration tests for recommendation workflow."""

    def test_recommendation_workflow_complete(self, sample_dataframe):
        """Test complete recommendation workflow."""
        with patch("parakeet_search.search.VectorStore") as MockVectorStore:
            mock_vs = MagicMock()
            MockVectorStore.return_value = mock_vs

            # Setup mock table
            mock_table = MagicMock()
            mock_vs.get_table.return_value = mock_table
            mock_vs.table = mock_table

            # Mock search_where
            def search_where_impl(where_clause):
                if 'episode_id = "ep_001"' in where_clause:
                    record = sample_dataframe[
                        sample_dataframe["episode_id"] == "ep_001"
                    ].iloc[0].to_dict()
                    mock_result = MagicMock()
                    mock_result.limit.return_value.to_list.return_value = [record]
                    return mock_result
                else:
                    mock_result = MagicMock()
                    mock_result.limit.return_value.to_list.return_value = []
                    return mock_result

            mock_table.search_where = search_where_impl

            # Mock search to return similar episodes
            mock_vs.search.return_value = [
                sample_dataframe.iloc[1].to_dict(),  # ep_002
                sample_dataframe.iloc[6].to_dict(),  # ep_007 (transfer learning)
            ]

            # Perform recommendation
            with patch("parakeet_search.search.EmbeddingModel"):
                engine = SearchEngine()
                results = engine.get_recommendations("ep_001", limit=2)

                assert len(results) == 2
                # Should find deep learning and transfer learning episodes
                episode_ids = [r["episode_id"] for r in results]
                assert "ep_002" in episode_ids
                assert "ep_007" in episode_ids

    def test_recommendation_ranking_by_similarity(self, sample_dataframe):
        """Test that results are ranked by similarity."""
        with patch("parakeet_search.search.VectorStore") as MockVectorStore:
            mock_vs = MagicMock()
            MockVectorStore.return_value = mock_vs
            mock_vs.table = MagicMock()

            # Mock search_where for episode lookup
            mock_table = mock_vs.get_table.return_value
            record = sample_dataframe.iloc[0].to_dict()
            mock_result = MagicMock()
            mock_result.limit.return_value.to_list.return_value = [record]
            mock_table.search_where.return_value = mock_result

            # Mock search with distance scores (smaller = more similar)
            results_data = [
                {**sample_dataframe.iloc[1].to_dict(), "_distance": 0.10},
                {**sample_dataframe.iloc[2].to_dict(), "_distance": 0.20},
                {**sample_dataframe.iloc[3].to_dict(), "_distance": 0.30},
            ]
            mock_vs.search.return_value = results_data

            # Perform recommendation
            with patch("parakeet_search.search.EmbeddingModel"):
                engine = SearchEngine()
                results = engine.get_recommendations("ep_001", limit=3)

                # Results should be in order (lower distance = higher relevance)
                assert results[0]["_distance"] == 0.10
                assert results[1]["_distance"] == 0.20
                assert results[2]["_distance"] == 0.30
