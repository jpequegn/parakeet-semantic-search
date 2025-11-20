#!/usr/bin/env python3
"""Complete embedding pipeline: fetch, chunk, embed, and store.

This script orchestrates the complete pipeline:
1. Ingest chunks from chunking script
2. Generate embeddings in batches
3. Store embeddings in LanceDB
4. Resume from interruptions using checkpoints
5. Report progress and statistics

Features:
- Batch embedding generation for efficiency
- Progress tracking with resumable checkpoints
- Comprehensive error handling and logging
- Performance metrics and statistics
- Flexible configuration options
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
import sqlite3

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from parakeet_search.embeddings import EmbeddingModel
from parakeet_search.vectorstore import VectorStore


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingReport:
    """Report on embedding pipeline execution."""

    start_time: datetime
    end_time: datetime
    chunks_processed: int
    chunks_embedded: int
    embeddings_stored: int
    embedding_failures: int
    storage_failures: int
    total_tokens_embedded: int
    model_name: str
    batch_size: int
    errors: List[str]
    warnings: List[str]

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.chunks_processed == 0:
            return 0.0
        return (self.chunks_embedded / self.chunks_processed) * 100

    @property
    def avg_embedding_time(self) -> float:
        """Average time per embedding in milliseconds."""
        if self.chunks_embedded == 0:
            return 0.0
        return (self.duration_seconds * 1000) / self.chunks_embedded

    def __str__(self) -> str:
        """Generate human-readable report."""
        return f"""
╔════════════════════════════════════════════════════════════╗
║           EMBEDDING PIPELINE REPORT                        ║
╠════════════════════════════════════════════════════════════╣
║ Duration:              {self.duration_seconds:.2f}s                       ║
║ Chunks Processed:      {self.chunks_processed:<26} ║
║ Chunks Embedded:       {self.chunks_embedded:<26} ║
║ Embeddings Stored:     {self.embeddings_stored:<26} ║
║ Success Rate:          {self.success_rate:.1f}%                        ║
║ Embedding Failures:    {self.embedding_failures:<26} ║
║ Storage Failures:      {self.storage_failures:<26} ║
║ Model:                 {self.model_name:<26} ║
║ Batch Size:            {self.batch_size:<26} ║
║ Avg Time/Embedding:    {self.avg_embedding_time:.2f}ms                    ║
║ Total Tokens Embedded: {self.total_tokens_embedded:<26} ║
╚════════════════════════════════════════════════════════════╝

{"ERRORS:" if self.errors else ""}
{chr(10).join("  - " + err for err in self.errors[:10])}
{f"  ... and {len(self.errors) - 10} more" if len(self.errors) > 10 else ""}

{"WARNINGS:" if self.warnings else ""}
{chr(10).join("  - " + warn for warn in self.warnings[:10])}
{f"  ... and {len(self.warnings) - 10} more" if len(self.warnings) > 10 else ""}
"""


class EmbeddingCheckpoint:
    """Manages resumable checkpoints for embedding pipeline."""

    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_db = self.checkpoint_dir / "embeddings.db"
        self._init_db()

    def _init_db(self):
        """Initialize checkpoint database."""
        conn = sqlite3.connect(str(self.checkpoint_db))
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processed_chunks (
                chunk_id TEXT PRIMARY KEY,
                episode_id TEXT,
                processed_at TIMESTAMP,
                embedding_dim INT,
                storage_status TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                chunks_processed INT,
                chunks_embedded INT,
                status TEXT
            )
        """)

        conn.commit()
        conn.close()

    def record_chunk(self, chunk_id: str, episode_id: str, embedding_dim: int, status: str = "embedded"):
        """Record that a chunk has been processed.

        Args:
            chunk_id: Unique chunk identifier
            episode_id: Associated episode ID
            embedding_dim: Embedding dimension
            status: Processing status (embedded, stored, failed)
        """
        conn = sqlite3.connect(str(self.checkpoint_db))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO processed_chunks
            (chunk_id, episode_id, processed_at, embedding_dim, storage_status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chunk_id, episode_id, datetime.now(), embedding_dim, status),
        )

        conn.commit()
        conn.close()

    def is_chunk_processed(self, chunk_id: str) -> bool:
        """Check if chunk has been processed.

        Args:
            chunk_id: Chunk identifier

        Returns:
            True if chunk is in checkpoint
        """
        conn = sqlite3.connect(str(self.checkpoint_db))
        cursor = conn.cursor()

        cursor.execute("SELECT 1 FROM processed_chunks WHERE chunk_id = ?", (chunk_id,))
        result = cursor.fetchone()
        conn.close()

        return result is not None

    def get_processed_chunks(self) -> set:
        """Get set of all processed chunk IDs.

        Returns:
            Set of chunk IDs that have been processed
        """
        conn = sqlite3.connect(str(self.checkpoint_db))
        cursor = conn.cursor()

        cursor.execute("SELECT chunk_id FROM processed_chunks")
        chunks = {row[0] for row in cursor.fetchall()}
        conn.close()

        return chunks

    def get_processing_stats(self) -> Dict:
        """Get processing statistics from checkpoint.

        Returns:
            Dictionary with processing stats
        """
        conn = sqlite3.connect(str(self.checkpoint_db))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM processed_chunks")
        total_processed = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM processed_chunks WHERE storage_status = 'stored'")
        total_stored = cursor.fetchone()[0]

        cursor.execute("SELECT AVG(embedding_dim) FROM processed_chunks")
        avg_embedding_dim = cursor.fetchone()[0]

        conn.close()

        return {
            "total_processed": total_processed,
            "total_stored": total_stored,
            "avg_embedding_dim": avg_embedding_dim,
        }


class EmbeddingPipeline:
    """Complete embedding pipeline orchestrator."""

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        vectorstore: Optional[VectorStore] = None,
        checkpoint: Optional[EmbeddingCheckpoint] = None,
        batch_size: int = 32,
        resume: bool = True,
    ):
        """Initialize pipeline.

        Args:
            embedding_model: EmbeddingModel instance
            vectorstore: VectorStore instance
            checkpoint: EmbeddingCheckpoint instance
            batch_size: Batch size for embedding
            resume: Whether to resume from checkpoint
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vectorstore = vectorstore or VectorStore()
        self.checkpoint = checkpoint or EmbeddingCheckpoint()
        self.batch_size = batch_size
        self.resume = resume

        logger.info(
            f"Initialized EmbeddingPipeline: model={self.embedding_model.model_name}, "
            f"batch_size={batch_size}, resume={resume}"
        )

    def embed_chunks(self, chunks_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Embed chunk texts.

        Args:
            chunks_df: DataFrame with chunks (must have 'text' and 'chunk_id' columns)

        Returns:
            Tuple of (DataFrame with embeddings, error list)
        """
        if chunks_df.empty:
            logger.warning("Empty DataFrame provided")
            return pd.DataFrame(), []

        # Validate required columns
        required_cols = {"text", "chunk_id"}
        if not required_cols.issubset(chunks_df.columns):
            missing = required_cols - set(chunks_df.columns)
            error_msg = f"Missing required columns: {missing}"
            logger.error(error_msg)
            return chunks_df, [error_msg]

        errors = []
        embeddings_list = []
        chunk_ids_list = []

        logger.info(f"Starting embedding for {len(chunks_df)} chunks")

        # Process in batches
        for batch_start in tqdm(range(0, len(chunks_df), self.batch_size), desc="Embedding batches"):
            batch_end = min(batch_start + self.batch_size, len(chunks_df))
            batch_df = chunks_df.iloc[batch_start:batch_end]

            try:
                # Filter out already processed chunks
                batch_df = batch_df[~batch_df["chunk_id"].isin(self.checkpoint.get_processed_chunks())]

                if len(batch_df) == 0:
                    continue

                texts = batch_df["text"].tolist()
                chunk_ids = batch_df["chunk_id"].tolist()

                # Generate embeddings
                embeddings = self.embedding_model.embed_texts(texts)

                embeddings_list.extend(embeddings.tolist())
                chunk_ids_list.extend(chunk_ids)

                # Record in checkpoint
                for chunk_id, embedding in zip(chunk_ids, embeddings):
                    self.checkpoint.record_chunk(
                        chunk_id=chunk_id,
                        episode_id=batch_df[batch_df["chunk_id"] == chunk_id]["episode_id"].iloc[0],
                        embedding_dim=len(embedding),
                        status="embedded",
                    )

            except Exception as e:
                error_msg = f"Error embedding batch [{batch_start}:{batch_end}]: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)

        # Create result DataFrame
        if embeddings_list:
            result_df = chunks_df.copy()
            result_df["embedding"] = None  # Will be filled selectively

            # Only add embeddings for successfully embedded chunks
            for chunk_id, embedding in zip(chunk_ids_list, embeddings_list):
                result_df.loc[result_df["chunk_id"] == chunk_id, "embedding"] = [embedding]

            logger.info(f"Successfully embedded {len(embeddings_list)} chunks")
            return result_df, errors
        else:
            logger.warning("No embeddings were generated")
            return chunks_df, errors

    def store_embeddings(self, chunks_df: pd.DataFrame, table_name: str = "transcripts") -> Tuple[int, List[str]]:
        """Store embeddings in vector store.

        Args:
            chunks_df: DataFrame with embeddings (must have 'embedding' column)
            table_name: Name of table in vector store

        Returns:
            Tuple of (count stored, error list)
        """
        if chunks_df.empty:
            logger.warning("Empty DataFrame to store")
            return 0, []

        # Filter chunks with valid embeddings
        valid_df = chunks_df[chunks_df["embedding"].notna()].copy()

        if valid_df.empty:
            error_msg = "No valid embeddings to store"
            logger.error(error_msg)
            return 0, [error_msg]

        logger.info(f"Storing {len(valid_df)} embeddings in vector store")

        try:
            # Convert embeddings to list format for LanceDB
            valid_df["embedding"] = valid_df["embedding"].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else x
            )

            # Store in vector store
            self.vectorstore.create_table(valid_df, table_name=table_name)

            # Update checkpoint
            for chunk_id in valid_df["chunk_id"]:
                self.checkpoint.record_chunk(
                    chunk_id=chunk_id,
                    episode_id=valid_df[valid_df["chunk_id"] == chunk_id]["episode_id"].iloc[0],
                    embedding_dim=len(valid_df[valid_df["chunk_id"] == chunk_id]["embedding"].iloc[0]),
                    status="stored",
                )

            logger.info(f"Successfully stored {len(valid_df)} embeddings")
            return len(valid_df), []

        except Exception as e:
            error_msg = f"Failed to store embeddings: {str(e)}"
            logger.error(error_msg)
            return 0, [error_msg]

    def process_pipeline(
        self,
        chunks_df: pd.DataFrame,
        table_name: str = "transcripts",
    ) -> EmbeddingReport:
        """Run complete embedding pipeline.

        Args:
            chunks_df: DataFrame with chunks to process
            table_name: Vector store table name

        Returns:
            EmbeddingReport with statistics
        """
        start_time = datetime.now()
        errors = []
        warnings = []

        try:
            # Step 1: Embed chunks
            embedded_df, embed_errors = self.embed_chunks(chunks_df)
            errors.extend(embed_errors)

            chunks_embedded = (embedded_df["embedding"].notna()).sum()

            if chunks_embedded == 0:
                error_msg = "No chunks were successfully embedded"
                logger.error(error_msg)
                errors.append(error_msg)
                end_time = datetime.now()

                return EmbeddingReport(
                    start_time=start_time,
                    end_time=end_time,
                    chunks_processed=len(chunks_df),
                    chunks_embedded=0,
                    embeddings_stored=0,
                    embedding_failures=len(chunks_df),
                    storage_failures=0,
                    total_tokens_embedded=chunks_df["token_count"].sum() if "token_count" in chunks_df.columns else 0,
                    model_name=self.embedding_model.model_name,
                    batch_size=self.batch_size,
                    errors=errors,
                    warnings=warnings,
                )

            # Step 2: Store embeddings
            stored_count, store_errors = self.store_embeddings(embedded_df, table_name=table_name)
            errors.extend(store_errors)

            end_time = datetime.now()

            # Create report
            report = EmbeddingReport(
                start_time=start_time,
                end_time=end_time,
                chunks_processed=len(chunks_df),
                chunks_embedded=chunks_embedded,
                embeddings_stored=stored_count,
                embedding_failures=len(chunks_df) - chunks_embedded,
                storage_failures=chunks_embedded - stored_count,
                total_tokens_embedded=chunks_df["token_count"].sum() if "token_count" in chunks_df.columns else 0,
                model_name=self.embedding_model.model_name,
                batch_size=self.batch_size,
                errors=errors,
                warnings=warnings,
            )

            logger.info(f"Pipeline completed: {report.chunks_embedded} embeddings, {report.embeddings_stored} stored")
            return report

        except Exception as e:
            logger.exception("Unhandled exception in pipeline")
            errors.append(f"Unhandled exception: {str(e)}")
            end_time = datetime.now()

            return EmbeddingReport(
                start_time=start_time,
                end_time=end_time,
                chunks_processed=len(chunks_df) if not chunks_df.empty else 0,
                chunks_embedded=0,
                embeddings_stored=0,
                embedding_failures=len(chunks_df) if not chunks_df.empty else 0,
                storage_failures=0,
                total_tokens_embedded=0,
                model_name=self.embedding_model.model_name,
                batch_size=self.batch_size,
                errors=errors,
                warnings=warnings,
            )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Complete embedding pipeline for semantic search")
    parser.add_argument(
        "--chunks-file",
        default="data/chunks.parquet",
        help="Path to chunks file (parquet or CSV)",
    )
    parser.add_argument(
        "--output-db",
        default="data/vectors.db",
        help="Path to LanceDB database",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="data/checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Embedding model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint",
    )
    parser.add_argument(
        "--table-name",
        default="transcripts",
        help="Vector store table name",
    )

    args = parser.parse_args()

    # Load chunks
    chunks_path = Path(args.chunks_file)
    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {args.chunks_file}")
        sys.exit(1)

    logger.info(f"Loading chunks from {args.chunks_file}")

    if chunks_path.suffix == ".parquet":
        chunks_df = pd.read_parquet(args.chunks_file)
    else:
        chunks_df = pd.read_csv(args.chunks_file)

    logger.info(f"Loaded {len(chunks_df)} chunks")

    # Create pipeline
    checkpoint = EmbeddingCheckpoint(args.checkpoint_dir)
    embedding_model = EmbeddingModel(args.model)
    vectorstore = VectorStore(args.output_db)

    pipeline = EmbeddingPipeline(
        embedding_model=embedding_model,
        vectorstore=vectorstore,
        checkpoint=checkpoint,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )

    # Run pipeline
    report = pipeline.process_pipeline(chunks_df, table_name=args.table_name)
    print(report)

    # Exit with appropriate code
    sys.exit(0 if len(report.errors) == 0 else 1)


if __name__ == "__main__":
    main()
