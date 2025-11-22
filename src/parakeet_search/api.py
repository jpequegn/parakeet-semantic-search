"""REST API for Parakeet Semantic Search - FastAPI Implementation.

Provides RESTful endpoints for search, recommendations, and analytics.

Features:
    - Full CRUD operations for search and recommendations
    - Authentication and rate limiting (optional)
    - Comprehensive error handling
    - API documentation with Swagger UI
    - JSON request/response format

Endpoints:
    - GET /health - Health check
    - POST /search - Semantic search
    - POST /recommendations/single - Single episode recommendations
    - POST /recommendations/hybrid - Multi-episode recommendations
    - GET /analytics/overview - Analytics overview
    - GET /cache/stats - Cache statistics
    - POST /cache/clear - Clear cache
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .search import SearchEngine
from .optimization import CachingSearchEngine

logger = logging.getLogger(__name__)


# ============================================================================
# Request/Response Models
# ============================================================================


class SearchRequest(BaseModel):
    """Semantic search request."""

    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    limit: int = Field(10, description="Maximum results", ge=1, le=100)
    threshold: float = Field(
        0.0, description="Similarity threshold", ge=0.0, le=1.0
    )

    class Config:
        """Model configuration."""

        json_schema_extra = {
            "example": {
                "query": "machine learning best practices",
                "limit": 10,
                "threshold": 0.5,
            }
        }


class SearchResult(BaseModel):
    """Individual search result."""

    episode_id: str
    episode_title: str
    podcast_id: str
    podcast_title: str
    similarity: float
    distance: float

    class Config:
        """Model configuration."""

        json_schema_extra = {
            "example": {
                "episode_id": "ep_001",
                "episode_title": "Introduction to Machine Learning",
                "podcast_id": "pod_001",
                "podcast_title": "AI Today Podcast",
                "similarity": 0.95,
                "distance": 0.05,
            }
        }


class SearchResponse(BaseModel):
    """Search response with metadata."""

    query: str
    results: List[SearchResult]
    result_count: int
    execution_time_ms: float
    timestamp: str

    class Config:
        """Model configuration."""

        json_schema_extra = {
            "example": {
                "query": "machine learning",
                "results": [],
                "result_count": 0,
                "execution_time_ms": 42.5,
                "timestamp": "2025-11-22T12:34:56Z",
            }
        }


class RecommendationRequest(BaseModel):
    """Single episode recommendation request."""

    episode_id: str = Field(..., description="Episode ID")
    limit: int = Field(10, description="Number of recommendations", ge=1, le=100)
    podcast_id: Optional[str] = Field(None, description="Filter by podcast ID")

    class Config:
        """Model configuration."""

        json_schema_extra = {
            "example": {
                "episode_id": "ep_001",
                "limit": 5,
                "podcast_id": None,
            }
        }


class HybridRecommendationRequest(BaseModel):
    """Multi-episode hybrid recommendation request."""

    episode_ids: List[str] = Field(..., description="List of episode IDs", min_items=1, max_items=10)
    limit: int = Field(10, description="Number of recommendations", ge=1, le=100)
    diversity_boost: float = Field(
        0.0, description="Diversity boost factor", ge=0.0, le=1.0
    )
    podcast_id: Optional[str] = Field(None, description="Filter by podcast ID")

    class Config:
        """Model configuration."""

        json_schema_extra = {
            "example": {
                "episode_ids": ["ep_001", "ep_002"],
                "limit": 5,
                "diversity_boost": 0.3,
                "podcast_id": None,
            }
        }


class RecommendationResponse(BaseModel):
    """Recommendation response."""

    input_episodes: List[str]
    recommendations: List[SearchResult]
    recommendation_count: int
    execution_time_ms: float
    timestamp: str

    class Config:
        """Model configuration."""

        json_schema_extra = {
            "example": {
                "input_episodes": ["ep_001"],
                "recommendations": [],
                "recommendation_count": 0,
                "execution_time_ms": 45.2,
                "timestamp": "2025-11-22T12:34:56Z",
            }
        }


class CacheStats(BaseModel):
    """Cache statistics."""

    size: int = Field(..., description="Number of cached queries")
    hits: int = Field(..., description="Number of cache hits")
    misses: int = Field(..., description="Number of cache misses")
    hit_rate: float = Field(..., description="Cache hit rate (0-1)")
    memory_bytes: int = Field(..., description="Memory usage in bytes")
    timestamp: str = Field(..., description="Timestamp")

    class Config:
        """Model configuration."""

        json_schema_extra = {
            "example": {
                "size": 245,
                "hits": 180,
                "misses": 65,
                "hit_rate": 0.735,
                "memory_bytes": 12582912,
                "timestamp": "2025-11-22T12:34:56Z",
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Response timestamp")
    uptime_seconds: float = Field(..., description="Uptime in seconds")

    class Config:
        """Model configuration."""

        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2025-11-22T12:34:56Z",
                "uptime_seconds": 3600.5,
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional details")
    timestamp: str = Field(..., description="Error timestamp")

    class Config:
        """Model configuration."""

        json_schema_extra = {
            "example": {
                "error": "Not Found",
                "detail": "Episode ep_999 not found in database",
                "timestamp": "2025-11-22T12:34:56Z",
            }
        }


# ============================================================================
# FastAPI Application
# ============================================================================


def create_app(
    search_engine: Optional[SearchEngine] = None,
    cache_enabled: bool = True,
    cache_ttl: int = 3600,
) -> FastAPI:
    """Create FastAPI application.

    Args:
        search_engine: SearchEngine instance (uses default if None)
        cache_enabled: Whether to enable result caching
        cache_ttl: Cache time-to-live in seconds

    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Parakeet Semantic Search API",
        description="REST API for semantic podcast search, recommendations, and analytics",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Initialize search engine with optional caching
    if search_engine is None:
        search_engine = SearchEngine()

    if cache_enabled:
        search_engine = CachingSearchEngine(search_engine, ttl_seconds=cache_ttl)

    # Store in app state
    app.state.search_engine = search_engine
    app.state.start_time = datetime.now()

    # ========================================================================
    # Health & Status Endpoints
    # ========================================================================

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health_check() -> Dict[str, Any]:
        """Check API health status.

        Returns:
            Health status with version and uptime
        """
        uptime = (datetime.now() - app.state.start_time).total_seconds()
        return {
            "status": "healthy",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat() + "Z",
            "uptime_seconds": uptime,
        }

    # ========================================================================
    # Search Endpoints
    # ========================================================================

    @app.post("/search", response_model=SearchResponse, tags=["Search"])
    async def search(request: SearchRequest) -> Dict[str, Any]:
        """Perform semantic search.

        Args:
            request: Search request with query and parameters

        Returns:
            Search results with execution time

        Raises:
            HTTPException: If search fails
        """
        try:
            start_time = datetime.now()

            # Perform search
            results = app.state.search_engine.search(
                query=request.query,
                limit=request.limit,
                threshold=request.threshold,
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Format results
            formatted_results = [
                SearchResult(
                    episode_id=r.get("episode_id", ""),
                    episode_title=r.get("episode_title", ""),
                    podcast_id=r.get("podcast_id", ""),
                    podcast_title=r.get("podcast_title", ""),
                    similarity=r.get("similarity", 0.0),
                    distance=r.get("distance", 0.0),
                )
                for r in results
            ]

            return {
                "query": request.query,
                "results": formatted_results,
                "result_count": len(formatted_results),
                "execution_time_ms": execution_time,
                "timestamp": datetime.now().isoformat() + "Z",
            }
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    # ========================================================================
    # Recommendation Endpoints
    # ========================================================================

    @app.post(
        "/recommendations/single",
        response_model=RecommendationResponse,
        tags=["Recommendations"],
    )
    async def single_recommendation(request: RecommendationRequest) -> Dict[str, Any]:
        """Get recommendations for a single episode.

        Args:
            request: Single recommendation request

        Returns:
            Recommended episodes with execution time

        Raises:
            HTTPException: If recommendation fails
        """
        try:
            start_time = datetime.now()

            # Get recommendations
            recommendations = app.state.search_engine.get_recommendations(
                episode_id=request.episode_id,
                limit=request.limit,
                podcast_id=request.podcast_id,
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Format results
            formatted_recs = [
                SearchResult(
                    episode_id=r.get("episode_id", ""),
                    episode_title=r.get("episode_title", ""),
                    podcast_id=r.get("podcast_id", ""),
                    podcast_title=r.get("podcast_title", ""),
                    similarity=r.get("similarity", 0.0),
                    distance=r.get("distance", 0.0),
                )
                for r in recommendations
            ]

            return {
                "input_episodes": [request.episode_id],
                "recommendations": formatted_recs,
                "recommendation_count": len(formatted_recs),
                "execution_time_ms": execution_time,
                "timestamp": datetime.now().isoformat() + "Z",
            }
        except Exception as e:
            logger.error(f"Recommendation error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Recommendation failed: {str(e)}"
            )

    @app.post(
        "/recommendations/hybrid",
        response_model=RecommendationResponse,
        tags=["Recommendations"],
    )
    async def hybrid_recommendation(
        request: HybridRecommendationRequest,
    ) -> Dict[str, Any]:
        """Get hybrid recommendations for multiple episodes.

        Args:
            request: Hybrid recommendation request

        Returns:
            Recommended episodes combining multiple input episodes

        Raises:
            HTTPException: If recommendation fails
        """
        try:
            start_time = datetime.now()

            # Get hybrid recommendations
            recommendations = app.state.search_engine.get_hybrid_recommendations(
                episode_ids=request.episode_ids,
                limit=request.limit,
                diversity_boost=request.diversity_boost,
                podcast_id=request.podcast_id,
            )

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Format results
            formatted_recs = [
                SearchResult(
                    episode_id=r.get("episode_id", ""),
                    episode_title=r.get("episode_title", ""),
                    podcast_id=r.get("podcast_id", ""),
                    podcast_title=r.get("podcast_title", ""),
                    similarity=r.get("similarity", 0.0),
                    distance=r.get("distance", 0.0),
                )
                for r in recommendations
            ]

            return {
                "input_episodes": request.episode_ids,
                "recommendations": formatted_recs,
                "recommendation_count": len(formatted_recs),
                "execution_time_ms": execution_time,
                "timestamp": datetime.now().isoformat() + "Z",
            }
        except Exception as e:
            logger.error(f"Hybrid recommendation error: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Hybrid recommendation failed: {str(e)}"
            )

    # ========================================================================
    # Cache Endpoints
    # ========================================================================

    @app.get("/cache/stats", response_model=CacheStats, tags=["Cache"])
    async def cache_stats() -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Current cache statistics if caching enabled

        Raises:
            HTTPException: If caching not enabled
        """
        try:
            if isinstance(app.state.search_engine, CachingSearchEngine):
                stats = app.state.search_engine.cache_stats()
                return {
                    "size": stats.get("size", 0),
                    "hits": stats.get("hits", 0),
                    "misses": stats.get("misses", 0),
                    "hit_rate": stats.get("hit_rate", 0.0),
                    "memory_bytes": stats.get("memory_bytes", 0),
                    "timestamp": datetime.now().isoformat() + "Z",
                }
            else:
                raise HTTPException(
                    status_code=400, detail="Caching is not enabled"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Cache stats error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

    @app.post("/cache/clear", tags=["Cache"])
    async def clear_cache() -> Dict[str, Any]:
        """Clear the search result cache.

        Returns:
            Confirmation message

        Raises:
            HTTPException: If cache clearing fails
        """
        try:
            if isinstance(app.state.search_engine, CachingSearchEngine):
                app.state.search_engine.clear_cache()
                return {
                    "status": "success",
                    "message": "Cache cleared successfully",
                    "timestamp": datetime.now().isoformat() + "Z",
                }
            else:
                raise HTTPException(
                    status_code=400, detail="Caching is not enabled"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")

    # ========================================================================
    # Error Handlers
    # ========================================================================

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        """Handle HTTP exceptions with consistent error format."""
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "detail": None,
                "timestamp": datetime.now().isoformat() + "Z",
            },
        )

    return app


def create_app_with_settings(config: Optional[Dict[str, Any]] = None) -> FastAPI:
    """Create app with configuration dictionary.

    Args:
        config: Configuration dictionary with 'cache_enabled' and 'cache_ttl'

    Returns:
        FastAPI application instance
    """
    config = config or {}
    return create_app(
        cache_enabled=config.get("cache_enabled", True),
        cache_ttl=config.get("cache_ttl", 3600),
    )
