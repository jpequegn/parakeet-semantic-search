#!/usr/bin/env python3
"""Ingest podcast data from P³ DuckDB to Parakeet Semantic Search vector store.

This script:
1. Connects to the P³ DuckDB database
2. Queries episodes and transcripts
3. Validates and cleans data
4. Generates embeddings for transcripts
5. Populates the vector store
6. Generates an ingestion report
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

import pandas as pd
import duckdb
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
class IngestionReport:
    """Report of ingestion results."""

    start_time: datetime
    end_time: datetime
    episodes_processed: int
    transcripts_processed: int
    embeddings_created: int
    errors: List[str]
    warnings: List[str]

    @property
    def duration_seconds(self) -> float:
        """Duration of ingestion in seconds."""
        return (self.end_time - self.start_time).total_seconds()

    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        total = self.episodes_processed + self.transcripts_processed
        if total == 0:
            return 0.0
        successful = total - len(self.errors)
        return (successful / total) * 100

    def __str__(self) -> str:
        """Generate human-readable report."""
        return f"""
╔═══════════════════════════════════════════════════════════╗
║           INGESTION REPORT                                ║
╠═══════════════════════════════════════════════════════════╣
║ Duration:              {self.duration_seconds:.2f}s                      ║
║ Episodes Processed:    {self.episodes_processed:<25} ║
║ Transcripts Processed: {self.transcripts_processed:<25} ║
║ Embeddings Created:    {self.embeddings_created:<25} ║
║ Success Rate:          {self.success_rate:.1f}%                         ║
║ Errors:                {len(self.errors):<25} ║
║ Warnings:              {len(self.warnings):<25} ║
╚═══════════════════════════════════════════════════════════╝

{"ERRORS:" if self.errors else ""}
{chr(10).join("  - " + err for err in self.errors[:10])}
{f"  ... and {len(self.errors) - 10} more" if len(self.errors) > 10 else ""}

{"WARNINGS:" if self.warnings else ""}
{chr(10).join("  - " + warn for warn in self.warnings[:10])}
{f"  ... and {len(self.warnings) - 10} more" if len(self.warnings) > 10 else ""}
"""


class P3DataIngester:
    """Ingests data from P³ DuckDB to vector store."""

    def __init__(
        self,
        p3_db_path: str = "/Users/julienpequegnot/Code/parakeet-podcast-processor/data/p3.duckdb",
        vector_db_path: str = "data/vectors.db",
        batch_size: int = 32,
    ):
        """Initialize ingester.

        Args:
            p3_db_path: Path to P³ DuckDB database
            vector_db_path: Path to vector store
            batch_size: Batch size for embedding generation
        """
        self.p3_db_path = p3_db_path
        self.vector_db_path = vector_db_path
        self.batch_size = batch_size

        # Initialize components
        self.embedding_model = EmbeddingModel()
        self.vectorstore = VectorStore(vector_db_path)

        # Tracking
        self.report = None

    def connect_to_p3(self) -> duckdb.DuckDBPyConnection:
        """Connect to P³ DuckDB database.

        Returns:
            DuckDB connection

        Raises:
            FileNotFoundError: If database file not found
            RuntimeError: If connection fails
        """
        db_path = Path(self.p3_db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"P³ database not found: {self.p3_db_path}")

        try:
            conn = duckdb.connect(str(db_path), read_only=True)
            logger.info(f"Connected to P³ database: {self.p3_db_path}")
            return conn
        except Exception as e:
            raise RuntimeError(f"Failed to connect to P³ database: {str(e)}")

    def validate_data(self, df: pd.DataFrame, data_type: str) -> Tuple[pd.DataFrame, List[str]]:
        """Validate and clean data.

        Args:
            df: DataFrame to validate
            data_type: Type of data (episodes/transcripts)

        Returns:
            Tuple of (cleaned DataFrame, list of error messages)
        """
        errors = []
        original_count = len(df)

        if data_type == "episodes":
            # Validate episodes
            if "id" not in df.columns or "title" not in df.columns:
                errors.append("Episodes missing required columns (id, title)")
                return df, errors

            # Remove rows with missing critical fields
            df = df.dropna(subset=["id", "title"])

            # Validate numeric fields
            if "duration_seconds" in df.columns:
                df = df[df["duration_seconds"] > 0]

        elif data_type == "transcripts":
            # Validate transcripts
            if "episode_id" not in df.columns or "text" not in df.columns:
                errors.append("Transcripts missing required columns (episode_id, text)")
                return df, errors

            # Remove rows with missing critical fields
            df = df.dropna(subset=["episode_id", "text"])

            # Remove empty text
            df = df[df["text"].str.strip().str.len() > 0]

            # Validate episode_id is integer
            try:
                df["episode_id"] = df["episode_id"].astype(int)
            except Exception as e:
                errors.append(f"Failed to convert episode_id to integer: {str(e)}")
                return df, errors

        # Log validation results
        removed_count = original_count - len(df)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} invalid {data_type} records")

        return df, errors

    def query_episodes(self, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Query episodes from P³ database.

        Args:
            conn: DuckDB connection

        Returns:
            DataFrame with episodes
        """
        try:
            query = """
            SELECT
                id,
                podcast_id,
                title AS episode_title,
                date,
                url,
                duration_seconds
            FROM episodes
            WHERE status IN ('transcribed', 'processed', 'summarized')
            ORDER BY id
            """
            df = conn.execute(query).df()
            logger.info(f"Queried {len(df)} episodes from P³")
            return df
        except Exception as e:
            logger.error(f"Failed to query episodes: {str(e)}")
            return pd.DataFrame()

    def query_transcripts(self, conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
        """Query transcripts from P³ database.

        Args:
            conn: DuckDB connection

        Returns:
            DataFrame with transcripts
        """
        try:
            query = """
            SELECT
                episode_id,
                text,
                speaker,
                timestamp_start,
                timestamp_end,
                confidence
            FROM transcripts
            ORDER BY episode_id, timestamp_start
            """
            df = conn.execute(query).df()
            logger.info(f"Queried {len(df)} transcript segments from P³")
            return df
        except Exception as e:
            logger.error(f"Failed to query transcripts: {str(e)}")
            return pd.DataFrame()

    def prepare_data_for_vectorstore(
        self,
        episodes_df: pd.DataFrame,
        transcripts_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare data for vector store ingestion.

        Args:
            episodes_df: Episodes DataFrame
            transcripts_df: Transcripts DataFrame

        Returns:
            Combined DataFrame ready for vector store
        """
        if transcripts_df.empty:
            logger.warning("No transcripts to process")
            return pd.DataFrame()

        # Build aggregation dict based on available columns
        agg_dict = {"text": " ".join}
        if "speaker" in transcripts_df.columns:
            agg_dict["speaker"] = lambda x: list(x.unique())
        if "confidence" in transcripts_df.columns:
            agg_dict["confidence"] = "mean"

        # Group transcripts by episode
        transcript_groups = transcripts_df.groupby("episode_id").agg(agg_dict).reset_index()

        # Merge with episodes
        merged = transcript_groups.merge(
            episodes_df,
            left_on="episode_id",
            right_on="id",
            how="inner",
        )

        if merged.empty:
            logger.warning("No matching episodes found for transcripts")
            return pd.DataFrame()

        # Prepare final dataset
        result = pd.DataFrame({
            "id": merged["id"],
            "episode_id": merged["id"].astype(str).apply(lambda x: f"ep_{int(x):03d}"),
            "episode_title": merged["episode_title"],
            "podcast_id": merged.get("podcast_id", "unknown").fillna("unknown").astype(str),
            "text": merged["text"],
            "embedding": None,  # Will be populated
        })

        logger.info(f"Prepared {len(result)} episodes for vector store")
        return result

    def generate_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate embeddings for text data.

        Args:
            df: DataFrame with 'text' column

        Returns:
            DataFrame with 'embedding' column populated
        """
        if df.empty:
            return df

        logger.info(f"Generating embeddings for {len(df)} episodes...")

        embeddings_list = []
        errors = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):
            try:
                text = row["text"]
                if not isinstance(text, str) or len(text.strip()) == 0:
                    errors.append(f"Episode {row['episode_id']}: Empty or invalid text")
                    embeddings_list.append(None)
                    continue

                embedding = self.embedding_model.embed_text(text)
                embeddings_list.append(embedding)

            except Exception as e:
                errors.append(f"Episode {row['episode_id']}: {str(e)}")
                embeddings_list.append(None)

        df["embedding"] = embeddings_list

        if errors:
            logger.warning(f"Encountered {len(errors)} errors during embedding:")
            for err in errors[:5]:
                logger.warning(f"  {err}")

        return df

    def populate_vectorstore(self, df: pd.DataFrame) -> Tuple[int, List[str]]:
        """Populate vector store with data.

        Args:
            df: DataFrame with embeddings

        Returns:
            Tuple of (count of successful inserts, list of errors)
        """
        if df.empty:
            logger.warning("No data to populate")
            return 0, []

        # Filter out rows with None embeddings
        valid_df = df[df["embedding"].notna()].copy()

        if valid_df.empty:
            logger.error("No valid embeddings to store")
            return 0, [f"No valid embeddings generated for {len(df)} episodes"]

        logger.info(f"Storing {len(valid_df)} episodes in vector store...")

        try:
            self.vectorstore.create_table(valid_df, table_name="transcripts")
            logger.info(f"Successfully stored {len(valid_df)} episodes")
            return len(valid_df), []
        except Exception as e:
            error_msg = f"Failed to populate vector store: {str(e)}"
            logger.error(error_msg)
            return 0, [error_msg]

    def ingest(self) -> IngestionReport:
        """Run complete ingestion pipeline.

        Returns:
            IngestionReport with results
        """
        start_time = datetime.now()
        errors = []
        warnings = []

        try:
            # Connect to P³ database
            conn = self.connect_to_p3()

            # Query data
            episodes_df = self.query_episodes(conn)
            transcripts_df = self.query_transcripts(conn)

            episodes_count = len(episodes_df)
            transcripts_count = len(transcripts_df)

            if episodes_df.empty or transcripts_df.empty:
                error_msg = "Failed to query data from P³"
                logger.error(error_msg)
                errors.append(error_msg)

            # Validate data
            episodes_df, ep_errors = self.validate_data(episodes_df, "episodes")
            transcripts_df, tr_errors = self.validate_data(transcripts_df, "transcripts")
            errors.extend(ep_errors + tr_errors)

            # Prepare data
            data_df = self.prepare_data_for_vectorstore(episodes_df, transcripts_df)

            if data_df.empty:
                error_msg = "No valid data after preparation"
                logger.error(error_msg)
                errors.append(error_msg)
                end_time = datetime.now()
                self.report = IngestionReport(
                    start_time=start_time,
                    end_time=end_time,
                    episodes_processed=episodes_count,
                    transcripts_processed=transcripts_count,
                    embeddings_created=0,
                    errors=errors,
                    warnings=warnings,
                )
                return self.report

            # Generate embeddings
            data_df = self.generate_embeddings(data_df)
            embeddings_created = (data_df["embedding"].notna()).sum()

            # Populate vector store
            stored_count, store_errors = self.populate_vectorstore(data_df)
            errors.extend(store_errors)

            # Close connection
            conn.close()

            end_time = datetime.now()

            # Create report
            self.report = IngestionReport(
                start_time=start_time,
                end_time=end_time,
                episodes_processed=episodes_count,
                transcripts_processed=transcripts_count,
                embeddings_created=embeddings_created,
                errors=errors,
                warnings=warnings,
            )

            return self.report

        except Exception as e:
            logger.exception("Unhandled exception during ingestion")
            errors.append(f"Unhandled exception: {str(e)}")
            end_time = datetime.now()

            self.report = IngestionReport(
                start_time=start_time,
                end_time=end_time,
                episodes_processed=0,
                transcripts_processed=0,
                embeddings_created=0,
                errors=errors,
                warnings=warnings,
            )

            return self.report


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Ingest P³ data to vector store")
    parser.add_argument(
        "--p3-db",
        default="/Users/julienpequegnot/Code/parakeet-podcast-processor/data/p3.duckdb",
        help="Path to P³ DuckDB database",
    )
    parser.add_argument(
        "--vector-db",
        default="data/vectors.db",
        help="Path to vector store",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )

    args = parser.parse_args()

    # Run ingestion
    ingester = P3DataIngester(
        p3_db_path=args.p3_db,
        vector_db_path=args.vector_db,
        batch_size=args.batch_size,
    )

    report = ingester.ingest()
    print(report)

    # Exit with appropriate code
    sys.exit(0 if len(report.errors) == 0 else 1)


if __name__ == "__main__":
    main()
