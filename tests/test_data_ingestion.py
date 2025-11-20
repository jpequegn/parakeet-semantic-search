"""Tests for data ingestion from P³ DuckDB to vector store."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

# Import the ingester - this will be in scripts directory
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from ingest_from_duckdb import P3DataIngester, IngestionReport


class TestIngestionReport:
    """Test IngestionReport generation and formatting."""

    def test_report_creation(self):
        """Test creating an ingestion report."""
        start = datetime.now()
        end = start + timedelta(seconds=5)

        report = IngestionReport(
            start_time=start,
            end_time=end,
            episodes_processed=10,
            transcripts_processed=50,
            embeddings_created=10,
            errors=[],
            warnings=[],
        )

        assert report.episodes_processed == 10
        assert report.embeddings_created == 10
        assert report.duration_seconds == pytest.approx(5.0, abs=0.1)

    def test_report_success_rate(self):
        """Test success rate calculation."""
        start = datetime.now()
        end = start + timedelta(seconds=1)

        # All successful
        report = IngestionReport(
            start_time=start,
            end_time=end,
            episodes_processed=10,
            transcripts_processed=50,
            embeddings_created=10,
            errors=[],
            warnings=[],
        )
        assert report.success_rate == 100.0

        # Some errors
        report = IngestionReport(
            start_time=start,
            end_time=end,
            episodes_processed=10,
            transcripts_processed=50,
            embeddings_created=5,
            errors=["Error 1", "Error 2", "Error 3"],
            warnings=[],
        )
        # 60 total - 3 errors = 57 successful = 95%
        assert report.success_rate == pytest.approx(95.0, abs=1)

    def test_report_string_representation(self):
        """Test report string formatting."""
        report = IngestionReport(
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=10),
            episodes_processed=100,
            transcripts_processed=500,
            embeddings_created=100,
            errors=["Error 1"],
            warnings=["Warning 1"],
        )

        report_str = str(report)
        assert "INGESTION REPORT" in report_str
        assert "100" in report_str  # Episodes
        assert "500" in report_str  # Transcripts


class TestP3DataIngesterConnection:
    """Test P³ database connection."""

    def test_connection_success(self):
        """Test successful connection to P³ database."""
        with patch("pathlib.Path.exists", return_value=True):
            with patch("ingest_from_duckdb.duckdb") as mock_duckdb:
                mock_conn = MagicMock()
                mock_duckdb.connect.return_value = mock_conn

                ingester = P3DataIngester(p3_db_path="test.duckdb")
                conn = ingester.connect_to_p3()

                assert conn is not None
                mock_duckdb.connect.assert_called_once()

    def test_connection_file_not_found(self):
        """Test connection fails when file not found."""
        ingester = P3DataIngester(p3_db_path="/nonexistent/path/test.duckdb")

        with pytest.raises(FileNotFoundError):
            ingester.connect_to_p3()

    def test_connection_read_only(self):
        """Test that connection is read-only."""
        with patch("ingest_from_duckdb.duckdb") as mock_duckdb:
            with patch("pathlib.Path.exists", return_value=True):
                mock_conn = MagicMock()
                mock_duckdb.connect.return_value = mock_conn

                ingester = P3DataIngester(p3_db_path="test.duckdb")
                ingester.connect_to_p3()

                # Check that read_only=True was passed
                call_kwargs = mock_duckdb.connect.call_args[1]
                assert call_kwargs.get("read_only") is True


class TestP3DataIngesterValidation:
    """Test data validation."""

    @pytest.fixture
    def ingester(self):
        """Create an ingester instance."""
        with patch("ingest_from_duckdb.EmbeddingModel"):
            with patch("ingest_from_duckdb.VectorStore"):
                return P3DataIngester()

    def test_validate_episodes_valid(self, ingester):
        """Test validation of valid episodes."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "title": ["Episode 1", "Episode 2", "Episode 3"],
            "duration_seconds": [3600, 3600, 3600],
        })

        validated, errors = ingester.validate_data(df, "episodes")

        assert len(validated) == 3
        assert len(errors) == 0

    def test_validate_episodes_missing_columns(self, ingester):
        """Test validation fails with missing columns."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            # Missing 'title' column
        })

        validated, errors = ingester.validate_data(df, "episodes")

        assert len(errors) > 0
        assert "required columns" in errors[0]

    def test_validate_episodes_missing_values(self, ingester):
        """Test validation removes rows with missing critical fields."""
        df = pd.DataFrame({
            "id": [1, 2, None],
            "title": ["Episode 1", None, "Episode 3"],
        })

        validated, errors = ingester.validate_data(df, "episodes")

        assert len(validated) == 1  # Only first row is valid
        assert "Episode 1" in validated["title"].values

    def test_validate_episodes_invalid_duration(self, ingester):
        """Test validation removes episodes with invalid duration."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "title": ["Ep1", "Ep2", "Ep3"],
            "duration_seconds": [3600, -100, 0],  # Invalid durations
        })

        validated, errors = ingester.validate_data(df, "episodes")

        assert len(validated) == 1  # Only first is valid
        assert validated["id"].values[0] == 1

    def test_validate_transcripts_valid(self, ingester):
        """Test validation of valid transcripts."""
        df = pd.DataFrame({
            "episode_id": [1, 1, 2],
            "text": ["Text 1", "Text 2", "Text 3"],
        })

        validated, errors = ingester.validate_data(df, "transcripts")

        assert len(validated) == 3
        assert len(errors) == 0

    def test_validate_transcripts_empty_text(self, ingester):
        """Test validation removes transcripts with empty text."""
        df = pd.DataFrame({
            "episode_id": [1, 2, 3],
            "text": ["Valid text", "  ", ""],  # Some empty
        })

        validated, errors = ingester.validate_data(df, "transcripts")

        assert len(validated) == 1
        assert validated["text"].values[0] == "Valid text"

    def test_validate_transcripts_invalid_episode_id(self, ingester):
        """Test validation handles non-integer episode IDs."""
        df = pd.DataFrame({
            "episode_id": ["abc", "def", "ghi"],
            "text": ["Text 1", "Text 2", "Text 3"],
        })

        validated, errors = ingester.validate_data(df, "transcripts")

        assert len(errors) > 0
        assert "episode_id" in errors[0]


class TestP3DataIngesterQuerying:
    """Test database querying."""

    @pytest.fixture
    def ingester(self):
        """Create an ingester instance."""
        with patch("ingest_from_duckdb.EmbeddingModel"):
            with patch("ingest_from_duckdb.VectorStore"):
                return P3DataIngester()

    def test_query_episodes(self, ingester):
        """Test querying episodes."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_df = pd.DataFrame({
            "id": [1, 2],
            "episode_title": ["Ep1", "Ep2"],
            "duration_seconds": [3600, 3600],
        })
        mock_result.df.return_value = mock_df
        mock_conn.execute.return_value = mock_result

        df = ingester.query_episodes(mock_conn)

        assert len(df) == 2
        assert "episode_title" in df.columns
        mock_conn.execute.assert_called_once()

    def test_query_episodes_empty(self, ingester):
        """Test querying episodes when none exist."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.df.return_value = pd.DataFrame()
        mock_conn.execute.return_value = mock_result

        df = ingester.query_episodes(mock_conn)

        assert len(df) == 0

    def test_query_transcripts(self, ingester):
        """Test querying transcripts."""
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_df = pd.DataFrame({
            "episode_id": [1, 1, 2],
            "text": ["Text 1", "Text 2", "Text 3"],
        })
        mock_result.df.return_value = mock_df
        mock_conn.execute.return_value = mock_result

        df = ingester.query_transcripts(mock_conn)

        assert len(df) == 3
        assert "episode_id" in df.columns
        assert "text" in df.columns


class TestP3DataIngesterPreparation:
    """Test data preparation."""

    @pytest.fixture
    def ingester(self):
        """Create an ingester instance."""
        with patch("ingest_from_duckdb.EmbeddingModel"):
            with patch("ingest_from_duckdb.VectorStore"):
                return P3DataIngester()

    def test_prepare_data_basic(self, ingester):
        """Test basic data preparation."""
        episodes = pd.DataFrame({
            "id": [1, 2],
            "episode_title": ["Ep1", "Ep2"],
            "podcast_id": ["pod1", "pod2"],
        })

        transcripts = pd.DataFrame({
            "episode_id": [1, 1, 2],
            "text": ["Text 1", "Text 2", "Text 3"],
            "speaker": ["Speaker A", "Speaker B", "Speaker A"],
            "confidence": [0.95, 0.92, 0.98],
        })

        result = ingester.prepare_data_for_vectorstore(episodes, transcripts)

        assert len(result) == 2  # Two episodes
        assert "episode_id" in result.columns
        assert "text" in result.columns
        assert result["episode_id"].iloc[0].startswith("ep_")

    def test_prepare_data_empty_transcripts(self, ingester):
        """Test preparation with empty transcripts."""
        episodes = pd.DataFrame({"id": [1], "episode_title": ["Ep1"]})
        transcripts = pd.DataFrame()

        result = ingester.prepare_data_for_vectorstore(episodes, transcripts)

        assert len(result) == 0

    def test_prepare_data_no_matching_episodes(self, ingester):
        """Test preparation when episodes don't match transcripts."""
        episodes = pd.DataFrame({
            "id": [1],
            "episode_title": ["Ep1"],
        })

        transcripts = pd.DataFrame({
            "episode_id": [999],  # Non-matching
            "text": ["Text"],
            "speaker": ["Speaker"],
            "confidence": [0.95],
        })

        result = ingester.prepare_data_for_vectorstore(episodes, transcripts)

        assert len(result) == 0


class TestP3DataIngesterEmbedding:
    """Test embedding generation."""

    @pytest.fixture
    def ingester(self):
        """Create an ingester instance."""
        with patch("ingest_from_duckdb.EmbeddingModel") as mock_em:
            with patch("ingest_from_duckdb.VectorStore"):
                mock_em.return_value.embed_text.return_value = np.random.randn(384)
                ingester = P3DataIngester()
                return ingester

    def test_generate_embeddings(self, ingester):
        """Test embedding generation."""
        df = pd.DataFrame({
            "episode_id": ["ep_001", "ep_002"],
            "text": ["Text 1", "Text 2"],
            "embedding": [None, None],
        })

        result = ingester.generate_embeddings(df)

        assert (result["embedding"].notna()).sum() == 2
        assert result["embedding"].iloc[0] is not None

    def test_generate_embeddings_empty_text(self, ingester):
        """Test embedding generation with empty text."""
        df = pd.DataFrame({
            "episode_id": ["ep_001", "ep_002"],
            "text": ["", "Valid text"],
            "embedding": [None, None],
        })

        result = ingester.generate_embeddings(df)

        # Second should have embedding, first should not
        assert result["embedding"].iloc[0] is None
        assert result["embedding"].iloc[1] is not None

    def test_generate_embeddings_error_handling(self, ingester):
        """Test error handling during embedding."""
        # Make embed_text raise an exception
        ingester.embedding_model.embed_text.side_effect = Exception("Embedding failed")

        df = pd.DataFrame({
            "episode_id": ["ep_001"],
            "text": ["Text"],
            "embedding": [None],
        })

        result = ingester.generate_embeddings(df)

        # Should have None embedding due to error
        assert result["embedding"].iloc[0] is None


class TestP3DataIngesterVectorStore:
    """Test vector store population."""

    @pytest.fixture
    def ingester(self):
        """Create an ingester instance."""
        with patch("ingest_from_duckdb.EmbeddingModel"):
            with patch("ingest_from_duckdb.VectorStore") as mock_vs:
                ingester = P3DataIngester()
                return ingester

    def test_populate_vectorstore_success(self, ingester):
        """Test successful vector store population."""
        df = pd.DataFrame({
            "id": [1, 2],
            "episode_id": ["ep_001", "ep_002"],
            "embedding": [np.random.randn(384), np.random.randn(384)],
        })

        count, errors = ingester.populate_vectorstore(df)

        assert count == 2
        assert len(errors) == 0
        ingester.vectorstore.create_table.assert_called_once()

    def test_populate_vectorstore_with_none_embeddings(self, ingester):
        """Test that rows with None embeddings are filtered."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "episode_id": ["ep_001", "ep_002", "ep_003"],
            "embedding": [np.random.randn(384), None, np.random.randn(384)],
        })

        count, errors = ingester.populate_vectorstore(df)

        # Should only store 2 valid embeddings
        assert count == 2
        # Verify only 2 rows were passed
        call_df = ingester.vectorstore.create_table.call_args[0][0]
        assert len(call_df) == 2

    def test_populate_vectorstore_no_valid_embeddings(self, ingester):
        """Test handling when no valid embeddings exist."""
        df = pd.DataFrame({
            "id": [1, 2],
            "episode_id": ["ep_001", "ep_002"],
            "embedding": [None, None],
        })

        count, errors = ingester.populate_vectorstore(df)

        assert count == 0
        assert len(errors) > 0


class TestP3DataIngesterIntegration:
    """Integration tests for complete ingestion workflow."""

    def test_ingest_workflow_complete(self):
        """Test complete ingestion workflow."""
        with patch("ingest_from_duckdb.duckdb") as mock_duckdb:
            with patch("ingest_from_duckdb.EmbeddingModel") as mock_em:
                with patch("ingest_from_duckdb.VectorStore") as mock_vs:
                    with patch("pathlib.Path.exists", return_value=True):
                        # Setup mocks
                        mock_conn = MagicMock()
                        mock_duckdb.connect.return_value = mock_conn

                        mock_em.return_value.embed_text.return_value = np.random.randn(384)

                        # Mock query results
                        episodes_df = pd.DataFrame({
                            "id": [1],
                            "podcast_id": ["pod1"],
                            "episode_title": ["Episode 1"],
                            "duration_seconds": [3600],
                            "title": ["Episode 1"],  # Required by validation
                        })

                        transcripts_df = pd.DataFrame({
                            "episode_id": [1],
                            "text": ["Sample transcript text"],
                            "speaker": ["Speaker"],
                            "confidence": [0.95],
                        })

                        mock_execute = MagicMock()
                        mock_execute.df.side_effect = [episodes_df, transcripts_df]
                        mock_conn.execute.return_value = mock_execute

                        # Run ingestion
                        ingester = P3DataIngester()
                        report = ingester.ingest()

                        # Verify results
                        assert report is not None
                        assert report.episodes_processed == 1
                        assert report.duration_seconds > 0

    def test_ingest_workflow_error_handling(self):
        """Test error handling in ingestion workflow."""
        with patch("ingest_from_duckdb.duckdb") as mock_duckdb:
            with patch("ingest_from_duckdb.EmbeddingModel"):
                with patch("ingest_from_duckdb.VectorStore"):
                    # Make connection fail
                    mock_duckdb.connect.side_effect = RuntimeError("Connection failed")

                    ingester = P3DataIngester()
                    report = ingester.ingest()

                    # Should have errors in report
                    assert len(report.errors) > 0
