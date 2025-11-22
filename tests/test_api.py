"""Tests for REST API endpoints - FastAPI.

Tests all API endpoints including search, recommendations, cache management,
and error handling.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from unittest.mock import Mock, MagicMock

from parakeet_search.api import (
    create_app,
    SearchRequest,
    RecommendationRequest,
    HybridRecommendationRequest,
    SearchResult,
    SearchResponse,
    RecommendationResponse,
    CacheStats,
    HealthResponse,
    ErrorResponse,
)


def create_mock_search_engine():
    """Create a mock search engine for testing."""
    engine = Mock()

    # Mock search method
    def mock_search(query: str, limit: int = 10, threshold: float = 0.0):
        return [
            {
                "episode_id": "ep_001",
                "episode_title": "Test Episode 1",
                "podcast_id": "pod_001",
                "podcast_title": "Test Podcast 1",
                "similarity": 0.95,
                "distance": 0.05,
            },
            {
                "episode_id": "ep_002",
                "episode_title": "Test Episode 2",
                "podcast_id": "pod_001",
                "podcast_title": "Test Podcast 1",
                "similarity": 0.87,
                "distance": 0.13,
            },
        ][:limit]

    # Mock get_recommendations method
    def mock_recommendations(episode_id: str, limit: int = 5, podcast_id=None, exclude_episode=True):
        results = [
            {
                "episode_id": "ep_003",
                "episode_title": "Recommended Episode 1",
                "podcast_id": podcast_id or "pod_001",
                "podcast_title": "Test Podcast 1",
                "similarity": 0.92,
                "distance": 0.08,
            },
            {
                "episode_id": "ep_004",
                "episode_title": "Recommended Episode 2",
                "podcast_id": podcast_id or "pod_002",
                "podcast_title": "Test Podcast 2",
                "similarity": 0.88,
                "distance": 0.12,
            },
        ]
        if podcast_id:
            results = [r for r in results if r["podcast_id"] == podcast_id]
        return results[:limit]

    # Mock get_hybrid_recommendations method
    def mock_hybrid_recommendations(episode_ids, limit: int = 5, diversity_boost=0.0, podcast_id=None):
        results = [
            {
                "episode_id": "ep_005",
                "episode_title": "Hybrid Recommendation 1",
                "podcast_id": podcast_id or "pod_001",
                "podcast_title": "Test Podcast 1",
                "similarity": 0.91,
                "distance": 0.09,
            },
            {
                "episode_id": "ep_006",
                "episode_title": "Hybrid Recommendation 2",
                "podcast_id": podcast_id or "pod_002",
                "podcast_title": "Test Podcast 2",
                "similarity": 0.85,
                "distance": 0.15,
            },
        ]
        if podcast_id:
            results = [r for r in results if r["podcast_id"] == podcast_id]
        return results[:limit]

    engine.search = mock_search
    engine.get_recommendations = mock_recommendations
    engine.get_hybrid_recommendations = mock_hybrid_recommendations

    return engine


@pytest.fixture
def search_engine():
    """Create mock search engine."""
    return create_mock_search_engine()


@pytest.fixture
def app_client(search_engine):
    """Create FastAPI test client."""
    app = create_app(search_engine=search_engine, cache_enabled=True, cache_ttl=3600)
    return TestClient(app)


# ============================================================================
# Health Check Tests
# ============================================================================


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_success(self, app_client):
        """Test health check returns healthy status."""
        response = app_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
        assert data["uptime_seconds"] >= 0

    def test_health_response_schema(self, app_client):
        """Test health response matches schema."""
        response = app_client.get("/health")
        assert response.status_code == 200
        health = HealthResponse(**response.json())
        assert health.status == "healthy"
        assert health.version == "1.0.0"


# ============================================================================
# Search Endpoint Tests
# ============================================================================


class TestSearchEndpoint:
    """Test semantic search endpoint."""

    def test_search_basic(self, app_client):
        """Test basic search request."""
        response = app_client.post(
            "/search",
            json={
                "query": "machine learning",
                "limit": 5,
                "threshold": 0.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "machine learning"
        assert isinstance(data["results"], list)
        assert "execution_time_ms" in data
        assert "timestamp" in data

    def test_search_with_limit(self, app_client):
        """Test search respects limit parameter."""
        response = app_client.post(
            "/search",
            json={
                "query": "test query",
                "limit": 3,
                "threshold": 0.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 3

    def test_search_with_threshold(self, app_client):
        """Test search respects similarity threshold."""
        response = app_client.post(
            "/search",
            json={
                "query": "test",
                "limit": 10,
                "threshold": 0.5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        # All results should have similarity >= threshold
        for result in data["results"]:
            assert result["similarity"] >= 0.5

    def test_search_empty_query(self, app_client):
        """Test search rejects empty query."""
        response = app_client.post(
            "/search",
            json={
                "query": "",
                "limit": 10,
                "threshold": 0.0,
            },
        )
        assert response.status_code == 422  # Validation error

    def test_search_response_schema(self, app_client):
        """Test search response matches schema."""
        response = app_client.post(
            "/search",
            json={
                "query": "test",
                "limit": 5,
                "threshold": 0.0,
            },
        )
        assert response.status_code == 200
        search_resp = SearchResponse(**response.json())
        assert search_resp.query == "test"
        assert isinstance(search_resp.results, list)
        assert search_resp.result_count == len(search_resp.results)

    def test_search_result_schema(self, app_client):
        """Test search result items have correct schema."""
        response = app_client.post(
            "/search",
            json={
                "query": "test",
                "limit": 1,
                "threshold": 0.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        if data["results"]:
            result = SearchResult(**data["results"][0])
            assert result.episode_id
            assert result.episode_title
            assert result.podcast_id
            assert result.podcast_title
            assert 0 <= result.similarity <= 1
            assert result.distance >= 0

    def test_search_execution_time(self, app_client):
        """Test search returns realistic execution time."""
        response = app_client.post(
            "/search",
            json={
                "query": "test",
                "limit": 5,
                "threshold": 0.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["execution_time_ms"] >= 0
        assert data["execution_time_ms"] < 60000  # Less than 60 seconds

    def test_search_limit_bounds(self, app_client):
        """Test search limit parameter bounds."""
        # Min: 1
        response = app_client.post(
            "/search",
            json={
                "query": "test",
                "limit": 1,
                "threshold": 0.0,
            },
        )
        assert response.status_code == 200

        # Max: 100
        response = app_client.post(
            "/search",
            json={
                "query": "test",
                "limit": 100,
                "threshold": 0.0,
            },
        )
        assert response.status_code == 200

        # Over max
        response = app_client.post(
            "/search",
            json={
                "query": "test",
                "limit": 101,
                "threshold": 0.0,
            },
        )
        assert response.status_code == 422

    def test_search_threshold_bounds(self, app_client):
        """Test search threshold parameter bounds."""
        # Min: 0
        response = app_client.post(
            "/search",
            json={
                "query": "test",
                "limit": 5,
                "threshold": 0.0,
            },
        )
        assert response.status_code == 200

        # Max: 1.0
        response = app_client.post(
            "/search",
            json={
                "query": "test",
                "limit": 5,
                "threshold": 1.0,
            },
        )
        assert response.status_code == 200

        # Over max
        response = app_client.post(
            "/search",
            json={
                "query": "test",
                "limit": 5,
                "threshold": 1.1,
            },
        )
        assert response.status_code == 422


# ============================================================================
# Single Recommendation Tests
# ============================================================================


class TestSingleRecommendationEndpoint:
    """Test single episode recommendation endpoint."""

    def test_single_recommendation_basic(self, app_client):
        """Test basic single recommendation request."""
        response = app_client.post(
            "/recommendations/single",
            json={
                "episode_id": "ep_001",
                "limit": 5,
                "podcast_id": None,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["input_episodes"] == ["ep_001"]
        assert isinstance(data["recommendations"], list)
        assert "execution_time_ms" in data
        assert "timestamp" in data

    def test_single_recommendation_with_podcast_filter(self, app_client):
        """Test recommendation with podcast filter."""
        response = app_client.post(
            "/recommendations/single",
            json={
                "episode_id": "ep_001",
                "limit": 5,
                "podcast_id": "pod_001",
            },
        )
        assert response.status_code == 200
        data = response.json()
        # All recommendations should be from filtered podcast (or empty)
        for rec in data["recommendations"]:
            if rec["podcast_id"]:
                assert rec["podcast_id"] == "pod_001"

    def test_single_recommendation_response_schema(self, app_client):
        """Test recommendation response matches schema."""
        response = app_client.post(
            "/recommendations/single",
            json={
                "episode_id": "ep_001",
                "limit": 5,
                "podcast_id": None,
            },
        )
        assert response.status_code == 200
        rec_resp = RecommendationResponse(**response.json())
        assert rec_resp.input_episodes == ["ep_001"]
        assert isinstance(rec_resp.recommendations, list)
        assert rec_resp.recommendation_count == len(rec_resp.recommendations)

    def test_single_recommendation_limit_bounds(self, app_client):
        """Test recommendation limit bounds."""
        # Min: 1
        response = app_client.post(
            "/recommendations/single",
            json={
                "episode_id": "ep_001",
                "limit": 1,
                "podcast_id": None,
            },
        )
        assert response.status_code == 200

        # Max: 100
        response = app_client.post(
            "/recommendations/single",
            json={
                "episode_id": "ep_001",
                "limit": 100,
                "podcast_id": None,
            },
        )
        assert response.status_code == 200

        # Over max
        response = app_client.post(
            "/recommendations/single",
            json={
                "episode_id": "ep_001",
                "limit": 101,
                "podcast_id": None,
            },
        )
        assert response.status_code == 422


# ============================================================================
# Hybrid Recommendation Tests
# ============================================================================


class TestHybridRecommendationEndpoint:
    """Test hybrid multi-episode recommendation endpoint."""

    def test_hybrid_recommendation_basic(self, app_client):
        """Test basic hybrid recommendation request."""
        response = app_client.post(
            "/recommendations/hybrid",
            json={
                "episode_ids": ["ep_001", "ep_002"],
                "limit": 5,
                "diversity_boost": 0.0,
                "podcast_id": None,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["input_episodes"] == ["ep_001", "ep_002"]
        assert isinstance(data["recommendations"], list)

    def test_hybrid_recommendation_with_diversity(self, app_client):
        """Test hybrid recommendation with diversity boost."""
        response = app_client.post(
            "/recommendations/hybrid",
            json={
                "episode_ids": ["ep_001", "ep_002"],
                "limit": 5,
                "diversity_boost": 0.5,
                "podcast_id": None,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["input_episodes"] == ["ep_001", "ep_002"]

    def test_hybrid_recommendation_multiple_episodes(self, app_client):
        """Test hybrid recommendation with multiple episodes."""
        response = app_client.post(
            "/recommendations/hybrid",
            json={
                "episode_ids": ["ep_001", "ep_002", "ep_003", "ep_004"],
                "limit": 10,
                "diversity_boost": 0.3,
                "podcast_id": None,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["input_episodes"]) == 4

    def test_hybrid_recommendation_response_schema(self, app_client):
        """Test hybrid recommendation response matches schema."""
        response = app_client.post(
            "/recommendations/hybrid",
            json={
                "episode_ids": ["ep_001", "ep_002"],
                "limit": 5,
                "diversity_boost": 0.0,
                "podcast_id": None,
            },
        )
        assert response.status_code == 200
        rec_resp = RecommendationResponse(**response.json())
        assert len(rec_resp.input_episodes) == 2
        assert isinstance(rec_resp.recommendations, list)

    def test_hybrid_recommendation_empty_episodes(self, app_client):
        """Test hybrid recommendation rejects empty episode list."""
        response = app_client.post(
            "/recommendations/hybrid",
            json={
                "episode_ids": [],
                "limit": 5,
                "diversity_boost": 0.0,
                "podcast_id": None,
            },
        )
        assert response.status_code == 422

    def test_hybrid_recommendation_too_many_episodes(self, app_client):
        """Test hybrid recommendation rejects too many episodes."""
        response = app_client.post(
            "/recommendations/hybrid",
            json={
                "episode_ids": [f"ep_{i:03d}" for i in range(11)],
                "limit": 5,
                "diversity_boost": 0.0,
                "podcast_id": None,
            },
        )
        assert response.status_code == 422

    def test_hybrid_recommendation_diversity_bounds(self, app_client):
        """Test diversity boost parameter bounds."""
        # Min: 0
        response = app_client.post(
            "/recommendations/hybrid",
            json={
                "episode_ids": ["ep_001"],
                "limit": 5,
                "diversity_boost": 0.0,
                "podcast_id": None,
            },
        )
        assert response.status_code == 200

        # Max: 1.0
        response = app_client.post(
            "/recommendations/hybrid",
            json={
                "episode_ids": ["ep_001"],
                "limit": 5,
                "diversity_boost": 1.0,
                "podcast_id": None,
            },
        )
        assert response.status_code == 200

        # Over max
        response = app_client.post(
            "/recommendations/hybrid",
            json={
                "episode_ids": ["ep_001"],
                "limit": 5,
                "diversity_boost": 1.1,
                "podcast_id": None,
            },
        )
        assert response.status_code == 422


# ============================================================================
# Cache Endpoint Tests
# ============================================================================


class TestCacheEndpoints:
    """Test cache management endpoints."""

    def test_cache_stats_basic(self, app_client):
        """Test cache stats endpoint returns valid data."""
        response = app_client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "size" in data
        assert "hits" in data
        assert "misses" in data
        assert "hit_rate" in data
        assert "memory_bytes" in data
        assert "timestamp" in data

    def test_cache_stats_schema(self, app_client):
        """Test cache stats response matches schema."""
        response = app_client.get("/cache/stats")
        assert response.status_code == 200
        cache_stats = CacheStats(**response.json())
        assert cache_stats.size >= 0
        assert cache_stats.hits >= 0
        assert cache_stats.misses >= 0
        assert 0 <= cache_stats.hit_rate <= 1
        assert cache_stats.memory_bytes >= 0

    def test_cache_clear_success(self, app_client):
        """Test cache clear endpoint."""
        response = app_client.post("/cache/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "message" in data
        assert "timestamp" in data

    def test_cache_clear_and_verify(self, app_client):
        """Test cache is actually cleared."""
        # Perform a search to populate cache
        app_client.post(
            "/search",
            json={
                "query": "test",
                "limit": 5,
                "threshold": 0.0,
            },
        )

        # Clear cache
        response = app_client.post("/cache/clear")
        assert response.status_code == 200

        # Verify cache is empty
        stats_response = app_client.get("/cache/stats")
        stats = stats_response.json()
        assert stats["size"] == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestAPIIntegration:
    """Integration tests for complete workflows."""

    def test_search_and_cache_workflow(self, app_client):
        """Test search populates cache correctly."""
        # Initial cache should be empty
        stats_resp = app_client.get("/cache/stats")
        initial_stats = stats_resp.json()
        initial_size = initial_stats["size"]

        # Perform search
        app_client.post(
            "/search",
            json={
                "query": "test query",
                "limit": 5,
                "threshold": 0.0,
            },
        )

        # Cache should now have entries
        stats_resp = app_client.get("/cache/stats")
        new_stats = stats_resp.json()
        assert new_stats["size"] > initial_size

    def test_multiple_searches_same_query(self, app_client):
        """Test cache hit for identical queries."""
        query = {
            "query": "cache hit test",
            "limit": 5,
            "threshold": 0.0,
        }

        # First search (cache miss)
        app_client.post("/search", json=query)
        stats1 = app_client.get("/cache/stats").json()

        # Second search (cache hit)
        app_client.post("/search", json=query)
        stats2 = app_client.get("/cache/stats").json()

        # Hit count should increase
        assert stats2["hits"] >= stats1["hits"]

    def test_search_and_recommendation_workflow(self, app_client):
        """Test combined search and recommendation workflow."""
        # Search for episodes
        search_resp = app_client.post(
            "/search",
            json={
                "query": "test topic",
                "limit": 5,
                "threshold": 0.0,
            },
        )
        assert search_resp.status_code == 200

        # Get recommendations for first result
        search_data = search_resp.json()
        if search_data["results"]:
            first_episode = search_data["results"][0]["episode_id"]

            rec_resp = app_client.post(
                "/recommendations/single",
                json={
                    "episode_id": first_episode,
                    "limit": 5,
                    "podcast_id": None,
                },
            )
            assert rec_resp.status_code == 200
            assert rec_resp.json()["input_episodes"][0] == first_episode

    def test_health_endpoint_always_available(self, app_client):
        """Test health endpoint is always available."""
        for _ in range(5):
            response = app_client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"


# ============================================================================
# API Documentation Tests
# ============================================================================


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_schema_available(self, app_client):
        """Test OpenAPI schema is available."""
        response = app_client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "Parakeet Semantic Search API"

    def test_swagger_ui_available(self, app_client):
        """Test Swagger UI documentation is available."""
        response = app_client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()

    def test_redoc_available(self, app_client):
        """Test ReDoc documentation is available."""
        response = app_client.get("/redoc")
        assert response.status_code == 200


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling across endpoints."""

    def test_invalid_json_request(self, app_client):
        """Test endpoint handles invalid JSON."""
        response = app_client.post(
            "/search",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code in [400, 422]

    def test_missing_required_field(self, app_client):
        """Test endpoint validates required fields."""
        response = app_client.post(
            "/search",
            json={
                "limit": 5,
                # Missing 'query'
            },
        )
        assert response.status_code == 422

    def test_invalid_field_type(self, app_client):
        """Test endpoint validates field types."""
        response = app_client.post(
            "/search",
            json={
                "query": "test",
                "limit": "not a number",  # Invalid type
                "threshold": 0.0,
            },
        )
        assert response.status_code == 422

    def test_404_not_found(self, app_client):
        """Test 404 for non-existent endpoint."""
        response = app_client.get("/nonexistent")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
