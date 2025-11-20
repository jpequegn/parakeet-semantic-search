#!/usr/bin/env python3
"""Chunk long transcripts for semantic search.

This script implements a sliding window chunking strategy with:
- Token-based window sizing (~512 tokens)
- Configurable overlap (default 20%)
- Sentence-aware boundaries
- Full metadata preservation
- Comprehensive edge case handling

Strategy Rationale:
- Token-based sizing ensures consistent embedding dimensionality
- Overlap prevents semantic breaks at chunk boundaries
- Sentence awareness maintains readability and context
- Metadata preservation enables episode/podcast filtering in search
- Configurable parameters allow tuning for different use cases
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import re

import pandas as pd

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptChunk:
    """A chunk of transcript with metadata."""

    episode_id: str
    podcast_id: str
    episode_title: str
    podcast_title: Optional[str]
    episode_date: Optional[str]
    chunk_index: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    speaker: Optional[str] = None
    confidence: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert chunk to dictionary."""
        return {
            "episode_id": self.episode_id,
            "podcast_id": self.podcast_id,
            "episode_title": self.episode_title,
            "podcast_title": self.podcast_title,
            "episode_date": self.episode_date,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "token_count": self.token_count,
            "speaker": self.speaker,
            "confidence": self.confidence,
        }


class TranscriptChunker:
    """Chunks transcripts using sliding window strategy."""

    def __init__(
        self,
        token_window_size: int = 512,
        overlap_ratio: float = 0.2,
        min_chunk_size: int = 50,
        max_chunk_size: int = 2048,
    ):
        """Initialize chunker.

        Args:
            token_window_size: Target tokens per chunk (default 512)
            overlap_ratio: Fraction of window to overlap (0.0-0.5, default 0.2)
            min_chunk_size: Minimum chunk size in tokens (default 50)
            max_chunk_size: Maximum chunk size in tokens (default 2048)
        """
        self.token_window_size = token_window_size
        self.overlap_ratio = overlap_ratio
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Validate parameters
        if not 0.0 <= overlap_ratio < 0.5:
            raise ValueError("overlap_ratio must be between 0.0 and 0.5")
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size must be positive")
        if max_chunk_size < min_chunk_size:
            raise ValueError("max_chunk_size must be >= min_chunk_size")

        logger.info(
            f"Initialized TranscriptChunker: window={token_window_size}, "
            f"overlap={overlap_ratio:.1%}, min={min_chunk_size}, max={max_chunk_size}"
        )

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return []
        # Split on whitespace and punctuation boundaries
        tokens = re.findall(r"\S+", text)
        return tokens

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximate, ~1 token per 4 chars).

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        if not isinstance(text, str):
            return 0
        return max(1, len(text) // 4)

    def find_sentence_boundary(self, text: str, position: int, backward: bool = False) -> int:
        """Find nearest sentence boundary.

        Args:
            text: Text to search
            position: Starting position
            backward: Search backward (True) or forward (False)

        Returns:
            Position of sentence boundary
        """
        sentence_endings = {'.', '!', '?', '\n'}

        if backward:
            # Search backward from position
            search_range = range(position, -1, -1)
            for i in search_range:
                if i < len(text) and text[i] in sentence_endings:
                    # Return position after sentence ending
                    return min(i + 2, len(text))
            return 0
        else:
            # Search forward from position
            search_range = range(position, len(text))
            for i in search_range:
                if text[i] in sentence_endings:
                    # Return position after sentence ending
                    return min(i + 2, len(text))
            return len(text)

    def chunk(
        self,
        text: str,
        episode_id: str,
        podcast_id: str,
        episode_title: str,
        podcast_title: Optional[str] = None,
        episode_date: Optional[str] = None,
        speaker: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> List[TranscriptChunk]:
        """Chunk a transcript.

        Args:
            text: Transcript text
            episode_id: Episode ID
            podcast_id: Podcast ID
            episode_title: Episode title
            podcast_title: Podcast title (optional)
            episode_date: Episode date (optional)
            speaker: Speaker name (optional)
            confidence: Transcription confidence (optional)

        Returns:
            List of TranscriptChunk objects
        """
        # Validate inputs
        if not isinstance(text, str):
            logger.warning(f"Episode {episode_id}: text is not string, skipping")
            return []

        text = text.strip()
        if len(text) == 0:
            logger.warning(f"Episode {episode_id}: empty text, skipping")
            return []

        chunks = []
        text_len = len(text)
        tokens = self.tokenize(text)
        total_tokens = len(tokens)

        # Single chunk if small transcript
        if total_tokens <= self.token_window_size:
            chunk = TranscriptChunk(
                episode_id=episode_id,
                podcast_id=podcast_id,
                episode_title=episode_title,
                podcast_title=podcast_title,
                episode_date=episode_date,
                chunk_index=0,
                text=text,
                char_start=0,
                char_end=text_len,
                token_count=total_tokens,
                speaker=speaker,
                confidence=confidence,
            )
            chunks.append(chunk)
            logger.info(
                f"Episode {episode_id}: single chunk ({total_tokens} tokens)"
            )
            return chunks

        # Multiple chunks with sliding window
        step_size = int(self.token_window_size * (1 - self.overlap_ratio))
        step_size = max(1, step_size)  # Ensure at least 1 token step

        chunk_index = 0
        pos = 0

        while pos < text_len:
            # Calculate window end position (character-based)
            target_end_token = min(
                total_tokens,
                self.tokenize(text[:pos]).__len__() + self.token_window_size
            )

            # Estimate character position for target tokens
            # Approximate: 1 token ≈ 4 characters
            estimated_char_end = pos + (target_end_token - self.tokenize(text[:pos]).__len__()) * 4
            estimated_char_end = min(estimated_char_end, text_len)

            # Find sentence boundary
            chunk_end = self.find_sentence_boundary(text, min(int(estimated_char_end), text_len - 1), backward=False)
            chunk_end = min(chunk_end, text_len)

            # Ensure we make progress
            if chunk_end <= pos:
                chunk_end = min(pos + self.token_window_size * 4, text_len)

            chunk_text = text[pos:chunk_end]
            chunk_tokens = len(self.tokenize(chunk_text))

            # Skip very small chunks (except last)
            if chunk_tokens >= self.min_chunk_size or chunk_end >= text_len:
                chunk = TranscriptChunk(
                    episode_id=episode_id,
                    podcast_id=podcast_id,
                    episode_title=episode_title,
                    podcast_title=podcast_title,
                    episode_date=episode_date,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    char_start=pos,
                    char_end=chunk_end,
                    token_count=chunk_tokens,
                    speaker=speaker,
                    confidence=confidence,
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move position with overlap
            if chunk_end >= text_len:
                break

            pos = chunk_end - int(self.token_window_size * self.overlap_ratio * 4)
            pos = max(pos, chunk_end - int(self.token_window_size * self.overlap_ratio * 4))
            pos = min(pos, chunk_end)

        logger.info(
            f"Episode {episode_id}: {len(chunks)} chunks from {total_tokens} tokens"
        )
        return chunks

    def chunk_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        episode_id_column: str = "episode_id",
        podcast_id_column: str = "podcast_id",
        episode_title_column: str = "episode_title",
        podcast_title_column: Optional[str] = None,
        episode_date_column: Optional[str] = None,
        speaker_column: Optional[str] = None,
        confidence_column: Optional[str] = None,
    ) -> pd.DataFrame:
        """Chunk transcripts in a DataFrame.

        Args:
            df: DataFrame with transcript data
            text_column: Name of text column
            episode_id_column: Name of episode ID column
            podcast_id_column: Name of podcast ID column
            episode_title_column: Name of episode title column
            podcast_title_column: Name of podcast title column (optional)
            episode_date_column: Name of episode date column (optional)
            speaker_column: Name of speaker column (optional)
            confidence_column: Name of confidence column (optional)

        Returns:
            DataFrame with chunked transcripts
        """
        if df.empty:
            logger.warning("Empty DataFrame, returning empty result")
            return pd.DataFrame()

        all_chunks = []
        for idx, row in df.iterrows():
            chunks = self.chunk(
                text=row[text_column],
                episode_id=str(row[episode_id_column]),
                podcast_id=str(row[podcast_id_column]),
                episode_title=str(row[episode_title_column]),
                podcast_title=str(row[podcast_title_column]) if podcast_title_column and podcast_title_column in row else None,
                episode_date=str(row[episode_date_column]) if episode_date_column and episode_date_column in row else None,
                speaker=str(row[speaker_column]) if speaker_column and speaker_column in row else None,
                confidence=float(row[confidence_column]) if confidence_column and confidence_column in row else None,
            )
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No valid chunks generated")
            return pd.DataFrame()

        # Convert to DataFrame
        chunk_dicts = [chunk.to_dict() for chunk in all_chunks]
        result_df = pd.DataFrame(chunk_dicts)

        logger.info(f"Generated {len(result_df)} chunks from {len(df)} transcripts")
        return result_df


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Chunk transcripts for semantic search")
    parser.add_argument(
        "--input-db",
        default="/Users/julienpequegnot/Code/parakeet-podcast-processor/data/p3.duckdb",
        help="Path to P³ DuckDB database",
    )
    parser.add_argument(
        "--output-db",
        default="data/chunks.db",
        help="Path to output chunks database",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="Token window size (default 512)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Overlap ratio 0.0-0.5 (default 0.2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default 32)",
    )

    args = parser.parse_args()

    # Validate paths
    input_path = Path(args.input_db)
    if not input_path.exists():
        logger.error(f"Input database not found: {args.input_db}")
        sys.exit(1)

    # Create chunker
    chunker = TranscriptChunker(
        token_window_size=args.window_size,
        overlap_ratio=args.overlap,
    )

    logger.info(f"Chunking transcripts from {args.input_db}")
    logger.info(f"Output will be saved to {args.output_db}")

    # This is a template for actual implementation
    # In practice, would read from P³ DuckDB, chunk, and write to output
    logger.info("Template script - actual chunking would integrate with data pipeline")

    sys.exit(0)


if __name__ == "__main__":
    main()
