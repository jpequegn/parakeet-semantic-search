"""Search utility functions for Streamlit app."""

from typing import List, Dict, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.parakeet_search import (
    SearchEngine,
    EmbeddingModel,
    VectorStore,
    CachingSearchEngine,
)


class SearchManager:
    """Manages search operations with caching."""

    _instance = None

    def __new__(cls):
        """Singleton pattern for search engine."""
        if cls._instance is None:
            cls._instance = super(SearchManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize search manager."""
        if self._initialized:
            return

        try:
            # Initialize components
            self.embedding_model = EmbeddingModel()
            self.vectorstore = VectorStore()
            self.search_engine = SearchEngine(self.embedding_model, self.vectorstore)

            # Wrap with caching
            self.cached_engine = CachingSearchEngine(self.search_engine, cache_size=500)

            self._initialized = True
        except Exception as e:
            print(f"Error initializing SearchManager: {e}")
            self._initialized = False

    def search(
        self,
        query: str,
        limit: int = 10,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """Execute search query.

        Args:
            query: Search query
            limit: Number of results
            threshold: Similarity threshold

        Returns:
            List of results
        """
        if not self._initialized:
            return []

        try:
            results = self.cached_engine.search(query, limit=limit, threshold=threshold)
            return results if results else []
        except Exception as e:
            print(f"Search error: {e}")
            return []

    def get_recommendations(
        self,
        episode_id: str,
        limit: int = 5,
        podcast_id: Optional[str] = None,
    ) -> List[Dict]:
        """Get recommendations for an episode.

        Args:
            episode_id: Episode ID
            limit: Number of recommendations
            podcast_id: Optional podcast filter

        Returns:
            List of recommendations
        """
        if not self._initialized:
            return []

        try:
            results = self.cached_engine.get_recommendations(
                episode_id=episode_id,
                limit=limit,
                podcast_id=podcast_id,
            )
            return results if results else []
        except Exception as e:
            print(f"Recommendation error: {e}")
            return []

    def get_hybrid_recommendations(
        self,
        episode_ids: List[str],
        limit: int = 5,
        podcast_id: Optional[str] = None,
    ) -> List[Dict]:
        """Get hybrid recommendations from multiple episodes.

        Args:
            episode_ids: List of episode IDs
            limit: Number of recommendations
            podcast_id: Optional podcast filter

        Returns:
            List of recommendations
        """
        if not self._initialized:
            return []

        try:
            results = self.search_engine.get_hybrid_recommendations(
                episode_ids=episode_ids,
                limit=limit,
                podcast_id=podcast_id,
            )
            return results if results else []
        except Exception as e:
            print(f"Hybrid recommendation error: {e}")
            return []

    def clear_cache(self) -> None:
        """Clear search cache."""
        if self._initialized:
            self.cached_engine.clear_cache()

    def cache_stats(self) -> Dict:
        """Get cache statistics."""
        if self._initialized:
            return self.cached_engine.cache_stats()
        return {}


def format_result(result: Dict) -> Dict:
    """Format search result for display.

    Args:
        result: Raw search result

    Returns:
        Formatted result
    """
    return {
        "episode_id": result.get("episode_id", "N/A"),
        "episode_title": result.get("episode_title", "Untitled"),
        "podcast_title": result.get("podcast_title", "Unknown Podcast"),
        "podcast_id": result.get("podcast_id", "N/A"),
        "similarity": 1 - result.get("_distance", 0),  # Convert distance to similarity
        "distance": result.get("_distance", 0),
    }


def format_results_table(results: List[Dict]) -> List[Dict]:
    """Format results for table display.

    Args:
        results: List of raw results

    Returns:
        List of formatted results
    """
    return [format_result(r) for r in results]
