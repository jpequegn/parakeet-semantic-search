"""FastAPI application entry point for Parakeet Semantic Search.

Run with: uvicorn apps.fastapi_app:app --reload --port 8000

Features:
    - RESTful API for search and recommendations
    - Interactive API documentation at /docs
    - Cache management
    - Health monitoring
"""

from parakeet_search.api import create_app_with_settings

# Create FastAPI application with default settings
app = create_app_with_settings({
    "cache_enabled": True,
    "cache_ttl": 3600,
})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
