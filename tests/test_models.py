"""Tests for Pydantic data models."""

import numpy as np
import pytest
from pydantic import ValidationError

from parakeet_search.models import Episode, Transcript, SearchResult, Config


class TestEpisodeModel:
    """Test Episode Pydantic model."""

    def test_episode_creation_minimal(self):
        """Test creating episode with required fields only."""
        episode = Episode(
            id=1,
            episode_id="ep_001",
            podcast_id="pod_001",
            podcast_title="Podcast Title",
            episode_title="Episode Title",
            transcript="This is a valid transcript text.",
        )

        assert episode.id == 1
        assert episode.episode_id == "ep_001"
        assert episode.podcast_id == "pod_001"
        assert episode.duration_seconds is None
        assert episode.published_at is None

    def test_episode_creation_full(self):
        """Test creating episode with all fields."""
        episode = Episode(
            id=1,
            episode_id="ep_001",
            podcast_id="pod_001",
            podcast_title="Podcast Title",
            episode_title="Episode Title",
            transcript="This is a valid transcript text.",
            duration_seconds=3600,
            published_at="2024-01-15T10:00:00Z",
            url="https://example.com/ep_001",
        )

        assert episode.duration_seconds == 3600
        assert episode.published_at == "2024-01-15T10:00:00Z"
        assert episode.url == "https://example.com/ep_001"

    def test_episode_validation_empty_episode_id(self):
        """Test that empty episode_id is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Episode(
                id=1,
                episode_id="",
                podcast_id="pod_001",
                podcast_title="Title",
                episode_title="Title",
                transcript="Valid transcript",
            )
        assert "episode_id" in str(exc_info.value)

    def test_episode_validation_empty_transcript(self):
        """Test that empty transcript is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Episode(
                id=1,
                episode_id="ep_001",
                podcast_id="pod_001",
                podcast_title="Title",
                episode_title="Title",
                transcript="",
            )
        assert "transcript" in str(exc_info.value).lower()

    def test_episode_validation_short_transcript(self):
        """Test that transcript under 10 characters is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Episode(
                id=1,
                episode_id="ep_001",
                podcast_id="pod_001",
                podcast_title="Title",
                episode_title="Title",
                transcript="short",
            )
        assert "transcript" in str(exc_info.value).lower()

    def test_episode_validation_negative_duration(self):
        """Test that negative duration is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Episode(
                id=1,
                episode_id="ep_001",
                podcast_id="pod_001",
                podcast_title="Title",
                episode_title="Title",
                transcript="Valid transcript text here.",
                duration_seconds=-100,
            )
        assert "duration" in str(exc_info.value).lower()

    def test_episode_validation_long_id(self):
        """Test that very long ID is rejected."""
        long_id = "a" * 300
        with pytest.raises(ValidationError) as exc_info:
            Episode(
                id=1,
                episode_id=long_id,
                podcast_id="pod_001",
                podcast_title="Title",
                episode_title="Title",
                transcript="Valid transcript",
            )
        assert "episode_id" in str(exc_info.value).lower()

    def test_episode_json_schema(self):
        """Test that episode can be serialized to JSON."""
        episode = Episode(
            id=1,
            episode_id="ep_001",
            podcast_id="pod_001",
            podcast_title="Podcast",
            episode_title="Episode",
            transcript="Valid transcript text.",
            duration_seconds=3600,
        )

        json_data = episode.model_dump_json()
        assert "ep_001" in json_data
        assert "3600" in json_data


class TestTranscriptModel:
    """Test Transcript Pydantic model."""

    def test_transcript_creation_minimal(self):
        """Test creating transcript with required fields."""
        embedding = np.random.randn(384).tolist()
        transcript = Transcript(
            id=1, episode_id="ep_001", text="Valid transcript text.", embedding=embedding
        )

        assert transcript.id == 1
        assert transcript.episode_id == "ep_001"
        assert len(transcript.embedding) == 384
        assert transcript.chunk_index is None

    def test_transcript_creation_with_metadata(self):
        """Test creating transcript with metadata."""
        embedding = np.random.randn(384).tolist()
        transcript = Transcript(
            id=1,
            episode_id="ep_001",
            text="Valid transcript text.",
            embedding=embedding,
            chunk_index=0,
            metadata={"source": "Parakeet", "model": "all-MiniLM-L6-v2"},
        )

        assert transcript.chunk_index == 0
        assert transcript.metadata["source"] == "Parakeet"

    def test_transcript_validation_wrong_embedding_size(self):
        """Test that wrong embedding size is rejected."""
        embedding = np.random.randn(256).tolist()  # Wrong size
        with pytest.raises(ValidationError) as exc_info:
            Transcript(
                id=1,
                episode_id="ep_001",
                text="Valid text.",
                embedding=embedding,
            )
        assert "embedding" in str(exc_info.value).lower()

    def test_transcript_validation_nan_in_embedding(self):
        """Test that NaN values in embedding are rejected."""
        embedding = np.random.randn(384).tolist()
        embedding[0] = float("nan")
        with pytest.raises(ValidationError) as exc_info:
            Transcript(
                id=1,
                episode_id="ep_001",
                text="Valid text.",
                embedding=embedding,
            )
        assert "embedding" in str(exc_info.value).lower()

    def test_transcript_validation_inf_in_embedding(self):
        """Test that infinite values in embedding are rejected."""
        embedding = np.random.randn(384).tolist()
        embedding[0] = float("inf")
        with pytest.raises(ValidationError) as exc_info:
            Transcript(
                id=1,
                episode_id="ep_001",
                text="Valid text.",
                embedding=embedding,
            )
        assert "embedding" in str(exc_info.value).lower()

    def test_transcript_validation_short_text(self):
        """Test that text under 5 characters is rejected."""
        embedding = np.random.randn(384).tolist()
        with pytest.raises(ValidationError) as exc_info:
            Transcript(
                id=1, episode_id="ep_001", text="abc", embedding=embedding
            )
        assert "text" in str(exc_info.value).lower()

    def test_transcript_validation_negative_chunk_index(self):
        """Test that negative chunk index is rejected."""
        embedding = np.random.randn(384).tolist()
        with pytest.raises(ValidationError) as exc_info:
            Transcript(
                id=1,
                episode_id="ep_001",
                text="Valid text.",
                embedding=embedding,
                chunk_index=-1,
            )
        assert "chunk_index" in str(exc_info.value).lower()

    def test_transcript_json_roundtrip(self):
        """Test that transcript can be serialized and deserialized."""
        embedding = np.random.randn(384).tolist()
        transcript = Transcript(
            id=1,
            episode_id="ep_001",
            text="Valid text content.",
            embedding=embedding,
            metadata={"key": "value"},
        )

        # Serialize to dict and back
        data = transcript.model_dump()
        new_transcript = Transcript(**data)
        assert new_transcript.id == transcript.id
        assert new_transcript.episode_id == transcript.episode_id
        assert len(new_transcript.embedding) == 384


class TestSearchResultModel:
    """Test SearchResult Pydantic model."""

    def test_search_result_creation(self):
        """Test creating search result with required fields."""
        result = SearchResult(
            id=1,
            episode_id="ep_001",
            podcast_title="Podcast",
            episode_title="Episode",
            distance=0.15,
            similarity_score=0.85,
        )

        assert result.id == 1
        assert result.distance == 0.15
        assert result.similarity_score == 0.85

    def test_search_result_with_excerpt(self):
        """Test creating search result with optional fields."""
        result = SearchResult(
            id=1,
            episode_id="ep_001",
            podcast_title="Podcast",
            episode_title="Episode",
            distance=0.15,
            similarity_score=0.85,
            transcript_excerpt="This is a relevant excerpt.",
            url="https://example.com/ep_001",
        )

        assert result.transcript_excerpt == "This is a relevant excerpt."
        assert result.url == "https://example.com/ep_001"

    def test_search_result_validation_invalid_distance(self):
        """Test that negative distance is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SearchResult(
                id=1,
                episode_id="ep_001",
                podcast_title="Podcast",
                episode_title="Episode",
                distance=-0.1,
                similarity_score=0.85,
            )
        assert "distance" in str(exc_info.value).lower()

    def test_search_result_validation_invalid_similarity(self):
        """Test that similarity outside 0-1 range is rejected."""
        with pytest.raises(ValidationError):
            SearchResult(
                id=1,
                episode_id="ep_001",
                podcast_title="Podcast",
                episode_title="Episode",
                distance=0.15,
                similarity_score=1.5,
            )

    def test_search_result_validation_nan_distance(self):
        """Test that NaN distance is rejected."""
        with pytest.raises(ValidationError):
            SearchResult(
                id=1,
                episode_id="ep_001",
                podcast_title="Podcast",
                episode_title="Episode",
                distance=float("nan"),
                similarity_score=0.85,
            )

    def test_search_result_from_search_result(self):
        """Test creating SearchResult from raw search output."""
        raw_result = {
            "id": 1,
            "episode_id": "ep_001",
            "podcast_title": "Podcast",
            "episode_title": "Episode",
            "_distance": 0.2,
            "text": "This is a long transcript excerpt that should be truncated...",
            "url": "https://example.com/ep_001",
        }

        result = SearchResult.from_search_result(raw_result)

        assert result.id == 1
        assert result.distance == 0.2
        # Similarity should be calculated from distance
        assert 0.0 <= result.similarity_score <= 1.0
        # Excerpt should be truncated to 200 chars
        assert len(result.transcript_excerpt) <= 200

    def test_search_result_from_search_result_with_custom_score(self):
        """Test creating SearchResult with custom similarity score."""
        raw_result = {
            "id": 1,
            "episode_id": "ep_001",
            "podcast_title": "Podcast",
            "episode_title": "Episode",
            "_distance": 0.2,
        }

        result = SearchResult.from_search_result(raw_result, similarity_score=0.95)

        assert result.similarity_score == 0.95


class TestConfigModel:
    """Test Config Pydantic model."""

    def test_config_creation_default(self):
        """Test creating config with default values."""
        config = Config()

        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.embedding_dimension == 384
        assert config.vector_db_path == "data/vectors.db"
        assert config.batch_size == 32
        assert config.search_top_k == 10

    def test_config_creation_custom(self):
        """Test creating config with custom values."""
        config = Config(
            embedding_model="all-mpnet-base-v2",
            embedding_dimension=768,
            vector_db_path="custom/path.db",
            batch_size=64,
            search_top_k=20,
            min_similarity_threshold=0.5,
        )

        assert config.embedding_model == "all-mpnet-base-v2"
        assert config.embedding_dimension == 768
        assert config.batch_size == 64
        assert config.search_top_k == 20
        assert config.min_similarity_threshold == 0.5

    def test_config_validation_invalid_batch_size(self):
        """Test that invalid batch size is rejected."""
        with pytest.raises(ValidationError):
            Config(batch_size=0)

    def test_config_validation_invalid_search_top_k(self):
        """Test that invalid search_top_k is rejected."""
        with pytest.raises(ValidationError):
            Config(search_top_k=0)

    def test_config_validation_invalid_similarity_threshold(self):
        """Test that similarity threshold outside 0-1 is rejected."""
        with pytest.raises(ValidationError):
            Config(min_similarity_threshold=1.5)

    def test_config_validation_invalid_dimension(self):
        """Test that invalid embedding dimension is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            Config(embedding_dimension=256)
        assert "dimension" in str(exc_info.value).lower()

    def test_config_validation_valid_dimensions(self):
        """Test that valid embedding dimensions are accepted."""
        for dim in [96, 192, 384, 768, 1024]:
            config = Config(embedding_dimension=dim)
            assert config.embedding_dimension == dim

    def test_config_chunk_parameters(self):
        """Test config with chunk parameters."""
        config = Config(chunk_size=500, chunk_overlap=50)

        assert config.chunk_size == 500
        assert config.chunk_overlap == 50

    def test_config_validation_invalid_chunk_size(self):
        """Test that invalid chunk size is rejected."""
        with pytest.raises(ValidationError):
            Config(chunk_size=50)  # Too small (must be >= 100)

    def test_config_json_schema(self):
        """Test that config can be serialized to JSON."""
        config = Config(batch_size=64, search_top_k=20)

        json_data = config.model_dump_json()
        assert "64" in json_data
        assert "20" in json_data


class TestModelIntegration:
    """Integration tests for models working together."""

    def test_episode_to_transcript_flow(self):
        """Test flow from Episode to Transcript."""
        # Create an episode
        episode = Episode(
            id=1,
            episode_id="ep_001",
            podcast_id="pod_001",
            podcast_title="Podcast",
            episode_title="Episode",
            transcript="This is a valid transcript with enough content.",
        )

        # Create transcript from episode content
        embedding = np.random.randn(384).tolist()
        transcript = Transcript(
            id=1,
            episode_id=episode.episode_id,
            text=episode.transcript,
            embedding=embedding,
            metadata={"source": "episode", "podcast_id": episode.podcast_id},
        )

        assert transcript.episode_id == episode.episode_id
        assert transcript.text == episode.transcript

    def test_transcript_to_searchresult_flow(self):
        """Test flow from raw search result to SearchResult model."""
        # Simulate raw search result
        raw_result = {
            "id": 1,
            "episode_id": "ep_001",
            "podcast_title": "Podcast Title",
            "episode_title": "Episode Title",
            "_distance": 0.15,
            "text": "This is the transcript text that was found in search.",
        }

        # Convert to SearchResult
        search_result = SearchResult.from_search_result(raw_result)

        assert search_result.id == raw_result["id"]
        assert search_result.episode_id == raw_result["episode_id"]
        assert search_result.distance == raw_result["_distance"]
        assert 0.0 <= search_result.similarity_score <= 1.0

    def test_config_in_search_context(self):
        """Test using config in search context."""
        config = Config(
            search_top_k=5,
            min_similarity_threshold=0.5,
            batch_size=16,
        )

        # Simulate multiple search results
        results = []
        for i in range(10):
            result = {
                "id": i,
                "episode_id": f"ep_{i:03d}",
                "podcast_title": "Podcast",
                "episode_title": f"Episode {i}",
                "_distance": 0.1 + (i * 0.05),
            }
            search_result = SearchResult.from_search_result(result)
            # Filter by threshold
            if search_result.distance <= (1.0 - config.min_similarity_threshold):
                results.append(search_result)

        # Should respect threshold filtering
        assert len(results) > 0
        for result in results:
            assert result.distance >= 0.0
