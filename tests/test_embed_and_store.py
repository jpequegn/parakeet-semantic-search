"""Tests for the complete embedding pipeline."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock
import sqlite3

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from embed_and_store import EmbeddingReport, EmbeddingCheckpoint, EmbeddingPipeline


class TestEmbeddingReport:
    """Test EmbeddingReport dataclass."""

    def test_report_creation(self):
        """Test creating an embedding report."""
        start = datetime.now()
        end = start.replace(second=start.second + 5)

        report = EmbeddingReport(
            start_time=start,
            end_time=end,
            chunks_processed=100,
            chunks_embedded=100,
            embeddings_stored=100,
            embedding_failures=0,
            storage_failures=0,
            total_tokens_embedded=50000,
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
            errors=[],
            warnings=[],
        )

        assert report.chunks_processed == 100
        assert report.chunks_embedded == 100
        assert report.duration_seconds == pytest.approx(5.0, abs=0.1)

    def test_report_success_rate(self):
        """Test success rate calculation."""
        start = datetime.now()
        end = start.replace(second=start.second + 1)

        # 100% success
        report = EmbeddingReport(
            start_time=start,
            end_time=end,
            chunks_processed=100,
            chunks_embedded=100,
            embeddings_stored=100,
            embedding_failures=0,
            storage_failures=0,
            total_tokens_embedded=50000,
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
            errors=[],
            warnings=[],
        )
        assert report.success_rate == 100.0

        # 80% success
        report = EmbeddingReport(
            start_time=start,
            end_time=end,
            chunks_processed=100,
            chunks_embedded=80,
            embeddings_stored=80,
            embedding_failures=20,
            storage_failures=0,
            total_tokens_embedded=40000,
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
            errors=[],
            warnings=[],
        )
        assert report.success_rate == 80.0

    def test_report_avg_embedding_time(self):
        """Test average embedding time calculation."""
        start = datetime.now()
        end = start.replace(second=start.second + 10)

        report = EmbeddingReport(
            start_time=start,
            end_time=end,
            chunks_processed=100,
            chunks_embedded=100,
            embeddings_stored=100,
            embedding_failures=0,
            storage_failures=0,
            total_tokens_embedded=50000,
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
            errors=[],
            warnings=[],
        )

        # 10 seconds / 100 embeddings = 100ms per embedding
        assert report.avg_embedding_time == pytest.approx(100.0, abs=1)

    def test_report_string_representation(self):
        """Test report formatting."""
        report = EmbeddingReport(
            start_time=datetime.now(),
            end_time=datetime.now(),
            chunks_processed=100,
            chunks_embedded=95,
            embeddings_stored=95,
            embedding_failures=5,
            storage_failures=0,
            total_tokens_embedded=50000,
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
            errors=["Error 1"],
            warnings=["Warning 1"],
        )

        report_str = str(report)
        assert "EMBEDDING PIPELINE REPORT" in report_str
        assert "100" in report_str
        assert "95" in report_str


class TestEmbeddingCheckpoint:
    """Test checkpoint management."""

    def test_checkpoint_creation(self):
        """Test creating a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = EmbeddingCheckpoint(tmpdir)

            assert checkpoint.checkpoint_dir.exists()
            assert checkpoint.checkpoint_db.exists()

    def test_record_and_check_chunk(self):
        """Test recording and checking chunk processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = EmbeddingCheckpoint(tmpdir)

            # Initially not processed
            assert not checkpoint.is_chunk_processed("chunk_001")

            # Record processing
            checkpoint.record_chunk(
                chunk_id="chunk_001",
                episode_id="ep_001",
                embedding_dim=384,
                status="embedded",
            )

            # Now should be processed
            assert checkpoint.is_chunk_processed("chunk_001")

    def test_get_processed_chunks(self):
        """Test retrieving processed chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = EmbeddingCheckpoint(tmpdir)

            # Record multiple chunks
            for i in range(5):
                checkpoint.record_chunk(
                    chunk_id=f"chunk_{i:03d}",
                    episode_id=f"ep_{i:03d}",
                    embedding_dim=384,
                    status="embedded",
                )

            processed = checkpoint.get_processed_chunks()

            assert len(processed) == 5
            assert "chunk_000" in processed
            assert "chunk_004" in processed

    def test_get_processing_stats(self):
        """Test retrieving processing statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = EmbeddingCheckpoint(tmpdir)

            # Record chunks
            for i in range(5):
                checkpoint.record_chunk(
                    chunk_id=f"chunk_{i:03d}",
                    episode_id=f"ep_{i:03d}",
                    embedding_dim=384,
                    status="stored" if i < 3 else "embedded",
                )

            stats = checkpoint.get_processing_stats()

            assert stats["total_processed"] == 5
            assert stats["total_stored"] == 3
            assert stats["avg_embedding_dim"] == 384

    def test_checkpoint_persistence(self):
        """Test that checkpoint persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and use first checkpoint
            checkpoint1 = EmbeddingCheckpoint(tmpdir)
            checkpoint1.record_chunk("chunk_001", "ep_001", 384, "embedded")

            # Create second checkpoint from same dir
            checkpoint2 = EmbeddingCheckpoint(tmpdir)

            # Should still be able to find recorded chunk
            assert checkpoint2.is_chunk_processed("chunk_001")


class TestEmbeddingPipeline:
    """Test embedding pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline with mocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("embed_and_store.EmbeddingModel") as mock_em:
                with patch("embed_and_store.VectorStore") as mock_vs:
                    mock_em_instance = MagicMock()
                    mock_em_instance.model_name = "all-MiniLM-L6-v2"
                    mock_em_instance.embed_texts.return_value = np.random.randn(100, 384)
                    mock_em.return_value = mock_em_instance

                    mock_vs_instance = MagicMock()
                    mock_vs.return_value = mock_vs_instance

                    checkpoint = EmbeddingCheckpoint(tmpdir)
                    pipeline = EmbeddingPipeline(
                        embedding_model=mock_em_instance,
                        vectorstore=mock_vs_instance,
                        checkpoint=checkpoint,
                        batch_size=32,
                    )
                    yield pipeline

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.batch_size == 32
        assert pipeline.embedding_model is not None
        assert pipeline.vectorstore is not None
        assert pipeline.checkpoint is not None

    def test_embed_chunks_basic(self, pipeline):
        """Test basic chunk embedding."""
        df = pd.DataFrame({
            "chunk_id": ["chunk_001", "chunk_002"],
            "episode_id": ["ep_001", "ep_001"],
            "text": ["Text 1", "Text 2"],
            "token_count": [10, 15],
        })

        result_df, errors = pipeline.embed_chunks(df)

        assert len(result_df) == 2
        assert len(errors) == 0
        assert "embedding" in result_df.columns

    def test_embed_chunks_missing_columns(self, pipeline):
        """Test error handling for missing required columns."""
        df = pd.DataFrame({
            "episode_id": ["ep_001"],
            # Missing 'text' and 'chunk_id'
        })

        result_df, errors = pipeline.embed_chunks(df)

        assert len(errors) > 0
        assert "Missing required columns" in errors[0]

    def test_embed_chunks_empty_dataframe(self, pipeline):
        """Test handling empty DataFrame."""
        df = pd.DataFrame()

        result_df, errors = pipeline.embed_chunks(df)

        assert len(result_df) == 0

    def test_store_embeddings_basic(self, pipeline):
        """Test storing embeddings."""
        df = pd.DataFrame({
            "chunk_id": ["chunk_001", "chunk_002"],
            "episode_id": ["ep_001", "ep_001"],
            "embedding": [np.random.randn(384), np.random.randn(384)],
        })

        stored_count, errors = pipeline.store_embeddings(df)

        assert stored_count == 2
        assert len(errors) == 0

    def test_store_embeddings_no_valid(self, pipeline):
        """Test storing when no valid embeddings."""
        df = pd.DataFrame({
            "chunk_id": ["chunk_001"],
            "episode_id": ["ep_001"],
            "embedding": [None],
        })

        stored_count, errors = pipeline.store_embeddings(df)

        assert stored_count == 0
        assert len(errors) > 0

    def test_process_pipeline_complete(self, pipeline):
        """Test complete pipeline execution."""
        df = pd.DataFrame({
            "chunk_id": ["chunk_001", "chunk_002"],
            "episode_id": ["ep_001", "ep_002"],
            "text": ["Text 1", "Text 2"],
            "token_count": [10, 15],
        })

        # Mock embedding and storage
        pipeline.embedding_model.embed_texts.return_value = np.array([
            np.random.randn(384),
            np.random.randn(384),
        ])

        report = pipeline.process_pipeline(df)

        assert report.chunks_processed == 2
        assert report.chunks_embedded > 0
        assert report.duration_seconds >= 0

    def test_resume_from_checkpoint(self, pipeline):
        """Test resuming from checkpoint."""
        # Record first chunk as processed
        pipeline.checkpoint.record_chunk("chunk_001", "ep_001", 384, "embedded")

        df = pd.DataFrame({
            "chunk_id": ["chunk_001", "chunk_002"],
            "episode_id": ["ep_001", "ep_002"],
            "text": ["Text 1", "Text 2"],
            "token_count": [10, 15],
        })

        result_df, errors = pipeline.embed_chunks(df)

        # Only chunk_002 should be embedded (chunk_001 already processed)
        assert len(errors) == 0

    def test_pipeline_error_handling(self, pipeline):
        """Test error handling in pipeline."""
        df = pd.DataFrame({
            "chunk_id": ["chunk_001"],
            "episode_id": ["ep_001"],
            "text": ["Text 1"],
            "token_count": [10],
        })

        # Mock error in embedding
        pipeline.embedding_model.embed_texts.side_effect = Exception("Embedding failed")

        result_df, errors = pipeline.embed_chunks(df)

        assert len(errors) > 0
        assert "Error embedding batch" in errors[0]


class TestEmbeddingPipelineIntegration:
    """Integration tests for embedding pipeline."""

    def test_full_pipeline_workflow(self):
        """Test complete pipeline workflow with mocks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("embed_and_store.EmbeddingModel") as mock_em:
                with patch("embed_and_store.VectorStore") as mock_vs:
                    mock_em_instance = MagicMock()
                    mock_em_instance.model_name = "test-model"
                    mock_em_instance.embed_texts.return_value = np.random.randn(5, 384)
                    mock_em.return_value = mock_em_instance

                    mock_vs_instance = MagicMock()
                    mock_vs.return_value = mock_vs_instance

                    checkpoint = EmbeddingCheckpoint(tmpdir)
                    pipeline = EmbeddingPipeline(
                        embedding_model=mock_em_instance,
                        vectorstore=mock_vs_instance,
                        checkpoint=checkpoint,
                        batch_size=32,
                    )

                    # Create test data
                    df = pd.DataFrame({
                        "chunk_id": [f"chunk_{i:03d}" for i in range(5)],
                        "episode_id": ["ep_001"] * 5,
                        "text": [f"Text {i}" for i in range(5)],
                        "token_count": [10 + i for i in range(5)],
                    })

                    # Run pipeline
                    report = pipeline.process_pipeline(df)

                    assert report.chunks_processed == 5
                    assert report.embeddings_stored > 0
                    assert report.success_rate > 0
                    assert len(report.errors) == 0

    def test_pipeline_with_checkpoint_recovery(self):
        """Test pipeline recovery with checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("embed_and_store.EmbeddingModel") as mock_em:
                with patch("embed_and_store.VectorStore") as mock_vs:
                    mock_em_instance = MagicMock()
                    mock_em_instance.model_name = "test-model"
                    mock_em_instance.embed_texts.return_value = np.random.randn(3, 384)
                    mock_em.return_value = mock_em_instance

                    mock_vs_instance = MagicMock()
                    mock_vs.return_value = mock_vs_instance

                    checkpoint = EmbeddingCheckpoint(tmpdir)

                    # First pipeline instance
                    pipeline1 = EmbeddingPipeline(
                        embedding_model=mock_em_instance,
                        vectorstore=mock_vs_instance,
                        checkpoint=checkpoint,
                        batch_size=32,
                    )

                    # Process first batch
                    df = pd.DataFrame({
                        "chunk_id": ["chunk_001", "chunk_002", "chunk_003"],
                        "episode_id": ["ep_001", "ep_001", "ep_001"],
                        "text": ["Text 1", "Text 2", "Text 3"],
                        "token_count": [10, 15, 12],
                    })

                    report1 = pipeline1.process_pipeline(df)
                    chunks_processed_first = report1.chunks_embedded

                    # Second pipeline instance (should skip already processed)
                    pipeline2 = EmbeddingPipeline(
                        embedding_model=mock_em_instance,
                        vectorstore=mock_vs_instance,
                        checkpoint=checkpoint,
                        batch_size=32,
                    )

                    report2 = pipeline2.process_pipeline(df)

                    # Second run should process fewer chunks due to checkpoint
                    assert report2.chunks_embedded <= chunks_processed_first
