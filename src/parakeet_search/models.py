"""Pydantic models for data schema definition and validation."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
import numpy as np


class Episode(BaseModel):
    """Podcast episode metadata and content.

    Attributes:
        id: Unique identifier within database
        episode_id: Unique external episode identifier
        podcast_id: Unique external podcast identifier
        podcast_title: Title of the podcast series
        episode_title: Title of the individual episode
        transcript: Full transcript text of the episode
        duration_seconds: Duration of the episode in seconds (optional)
        published_at: Publication date of the episode (optional)
        url: URL to the episode (optional)

    Examples:
        >>> episode = Episode(
        ...     id=1,
        ...     episode_id="ep_001",
        ...     podcast_id="pod_001",
        ...     podcast_title="AI Today Podcast",
        ...     episode_title="Introduction to Machine Learning",
        ...     transcript="Machine learning is a subset of artificial intelligence...",
        ...     duration_seconds=3600,
        ...     published_at="2024-01-15T10:00:00Z"
        ... )
    """

    id: int = Field(..., description="Unique database identifier")
    episode_id: str = Field(..., min_length=1, description="External episode ID")
    podcast_id: str = Field(..., min_length=1, description="External podcast ID")
    podcast_title: str = Field(..., min_length=1, description="Podcast series title")
    episode_title: str = Field(..., min_length=1, description="Episode title")
    transcript: str = Field(..., min_length=1, description="Full episode transcript")
    duration_seconds: Optional[int] = Field(None, ge=0, description="Duration in seconds")
    published_at: Optional[str] = Field(None, description="ISO 8601 publication timestamp")
    url: Optional[str] = Field(None, description="URL to episode")

    @field_validator("episode_id", "podcast_id")
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate that IDs are non-empty and don't contain invalid characters."""
        if not v or not v.strip():
            raise ValueError("ID cannot be empty or whitespace")
        if len(v) > 255:
            raise ValueError("ID cannot exceed 255 characters")
        return v.strip()

    @field_validator("transcript")
    @classmethod
    def validate_transcript(cls, v: str) -> str:
        """Validate that transcript is substantial (at least 10 characters)."""
        if not v or len(v.strip()) < 10:
            raise ValueError("Transcript must contain at least 10 characters")
        return v

    @field_validator("duration_seconds")
    @classmethod
    def validate_duration(cls, v: Optional[int]) -> Optional[int]:
        """Validate that duration is positive if provided."""
        if v is not None and v < 0:
            raise ValueError("Duration must be non-negative")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "episode_id": "ep_001",
                "podcast_id": "pod_001",
                "podcast_title": "AI Today Podcast",
                "episode_title": "Introduction to Machine Learning",
                "transcript": "Machine learning is a subset of artificial intelligence...",
                "duration_seconds": 3600,
                "published_at": "2024-01-15T10:00:00Z",
                "url": "https://example.com/episodes/ep_001",
            }
        }
    )


class Transcript(BaseModel):
    """Episode transcript with embedding and metadata.

    This model represents a transcript chunk or full transcript
    with its vector embedding and associated metadata.

    Attributes:
        id: Unique identifier within database
        episode_id: Reference to the episode
        text: Transcript text content
        embedding: 384-dimensional vector embedding from Sentence Transformers
        chunk_index: Index if this is a chunked transcript (optional)
        metadata: Additional metadata dictionary

    Examples:
        >>> embedding = np.random.randn(384).tolist()
        >>> transcript = Transcript(
        ...     id=1,
        ...     episode_id="ep_001",
        ...     text="Machine learning is a subset of artificial intelligence...",
        ...     embedding=embedding,
        ...     metadata={"source": "Parakeet", "model": "all-MiniLM-L6-v2"}
        ... )
    """

    id: int = Field(..., description="Unique database identifier")
    episode_id: str = Field(..., min_length=1, description="Reference to episode")
    text: str = Field(..., min_length=1, description="Transcript text content")
    embedding: List[float] = Field(
        ...,
        description="384-dimensional vector embedding",
        min_length=384,
        max_length=384,
    )
    chunk_index: Optional[int] = Field(None, ge=0, description="Index if chunked")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Validate that text is substantial."""
        if not v or len(v.strip()) < 5:
            raise ValueError("Text must contain at least 5 characters")
        return v

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        """Validate embedding dimension and values."""
        if len(v) != 384:
            raise ValueError(f"Embedding must have exactly 384 dimensions, got {len(v)}")
        # Ensure all values are finite (not NaN or inf)
        if not all(isinstance(x, (int, float)) and np.isfinite(x) for x in v):
            raise ValueError("Embedding contains NaN or infinite values")
        return v

    @field_validator("chunk_index")
    @classmethod
    def validate_chunk_index(cls, v: Optional[int]) -> Optional[int]:
        """Validate that chunk index is non-negative if provided."""
        if v is not None and v < 0:
            raise ValueError("Chunk index must be non-negative")
        return v

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "episode_id": "ep_001",
                "text": "Machine learning is a subset of artificial intelligence...",
                "embedding": [0.1, 0.2, 0.3, -0.1],  # Abbreviated for example
                "chunk_index": 0,
                "metadata": {"source": "Parakeet", "model": "all-MiniLM-L6-v2"},
            }
        },
    )


class SearchResult(BaseModel):
    """Search result with episode information and relevance score.

    This model represents a single result from semantic search,
    including the episode information and similarity scoring.

    Attributes:
        id: Database identifier
        episode_id: Episode identifier
        podcast_title: Podcast title
        episode_title: Episode title
        distance: Vector similarity distance (lower = more similar)
        similarity_score: Calculated similarity score (0-1, higher = more similar)
        transcript_excerpt: Optional excerpt from transcript showing relevance

    Examples:
        >>> result = SearchResult(
        ...     id=1,
        ...     episode_id="ep_001",
        ...     podcast_title="AI Today Podcast",
        ...     episode_title="Introduction to Machine Learning",
        ...     distance=0.15,
        ...     similarity_score=0.85
        ... )
    """

    id: int = Field(..., description="Database identifier")
    episode_id: str = Field(..., min_length=1, description="Episode identifier")
    podcast_title: str = Field(..., min_length=1, description="Podcast title")
    episode_title: str = Field(..., min_length=1, description="Episode title")
    distance: float = Field(..., ge=0.0, description="Vector distance (lower=better)")
    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Similarity score 0-1"
    )
    transcript_excerpt: Optional[str] = Field(
        None, description="Excerpt showing relevance"
    )
    url: Optional[str] = Field(None, description="URL to episode")

    @field_validator("distance")
    @classmethod
    def validate_distance(cls, v: float) -> float:
        """Validate distance is finite."""
        if not np.isfinite(v):
            raise ValueError("Distance must be a finite number")
        return v

    @field_validator("similarity_score")
    @classmethod
    def validate_similarity_score(cls, v: float) -> float:
        """Validate similarity score is between 0 and 1."""
        if not (0.0 <= v <= 1.0):
            raise ValueError("Similarity score must be between 0 and 1")
        return v

    @classmethod
    def from_search_result(
        cls,
        search_result: Dict[str, Any],
        similarity_score: Optional[float] = None,
    ) -> "SearchResult":
        """Create SearchResult from raw search output.

        Args:
            search_result: Dictionary from vectorstore.search()
            similarity_score: Optional calculated similarity score.
                            If not provided, calculated from distance.

        Returns:
            SearchResult instance
        """
        distance = search_result.get("_distance", 0.5)
        # Convert distance to similarity (0-1 scale, higher = better)
        # Using 1 / (1 + distance) normalization
        if similarity_score is None:
            similarity_score = 1.0 / (1.0 + distance)

        return cls(
            id=search_result.get("id"),
            episode_id=search_result.get("episode_id"),
            podcast_title=search_result.get("podcast_title", ""),
            episode_title=search_result.get("episode_title", ""),
            distance=distance,
            similarity_score=similarity_score,
            transcript_excerpt=search_result.get("text", "")[:200],
            url=search_result.get("url"),
        )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "episode_id": "ep_001",
                "podcast_title": "AI Today Podcast",
                "episode_title": "Introduction to Machine Learning",
                "distance": 0.15,
                "similarity_score": 0.85,
                "transcript_excerpt": "Machine learning is a subset of artificial intelligence...",
                "url": "https://example.com/episodes/ep_001",
            }
        }
    )


class Config(BaseModel):
    """Application configuration settings.

    Attributes:
        embedding_model: Name of the embedding model to use
        embedding_dimension: Dimension of embeddings produced by the model
        vector_db_path: Path to the vector database
        batch_size: Batch size for embedding generation
        max_transcript_length: Maximum transcript length to process
        search_top_k: Default number of results to return from search
        min_similarity_threshold: Minimum similarity score for results

    Examples:
        >>> config = Config(
        ...     embedding_model="all-MiniLM-L6-v2",
        ...     embedding_dimension=384,
        ...     vector_db_path="data/vectors.db",
        ...     batch_size=32,
        ...     search_top_k=10
        ... )
    """

    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Name of the embedding model to use",
    )
    embedding_dimension: int = Field(
        default=384, ge=1, description="Dimension of model embeddings"
    )
    vector_db_path: str = Field(
        default="data/vectors.db", description="Path to vector database"
    )
    batch_size: int = Field(
        default=32, ge=1, le=1024, description="Batch size for processing"
    )
    max_transcript_length: int = Field(
        default=1000000,
        ge=1000,
        description="Maximum transcript length in characters",
    )
    search_top_k: int = Field(
        default=10, ge=1, le=1000, description="Default number of search results"
    )
    min_similarity_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for results",
    )
    chunk_size: Optional[int] = Field(
        default=None, ge=100, description="Transcript chunk size in characters"
    )
    chunk_overlap: Optional[int] = Field(
        default=None, ge=0, description="Overlap between chunks"
    )

    @field_validator("embedding_dimension")
    @classmethod
    def validate_dimension(cls, v: int) -> int:
        """Validate embedding dimension is reasonable."""
        if v not in (96, 192, 384, 768, 1024):
            raise ValueError(
                f"Embedding dimension {v} not supported. "
                "Common dimensions: 96, 192, 384, 768, 1024"
            )
        return v

    @field_validator("chunk_size", "chunk_overlap")
    @classmethod
    def validate_chunk_params(cls, v: Optional[int]) -> Optional[int]:
        """Validate chunk parameters if provided."""
        if v is not None and v <= 0:
            raise ValueError("Chunk parameters must be positive if provided")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "embedding_model": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "vector_db_path": "data/vectors.db",
                "batch_size": 32,
                "max_transcript_length": 1000000,
                "search_top_k": 10,
                "min_similarity_threshold": 0.0,
                "chunk_size": 500,
                "chunk_overlap": 50,
            }
        }
    )
