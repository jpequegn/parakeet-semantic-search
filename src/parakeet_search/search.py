"""Search engine implementation."""

from typing import List, Optional, Union
import numpy as np
from .embeddings import EmbeddingModel
from .vectorstore import VectorStore


class SearchEngine:
    """Semantic search engine for podcasts."""

    def __init__(self, embedding_model: EmbeddingModel = None, vectorstore: VectorStore = None):
        """Initialize search engine.

        Args:
            embedding_model: EmbeddingModel instance
            vectorstore: VectorStore instance
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.vectorstore = vectorstore or VectorStore()

    def search(self, query: str, limit: int = 10, threshold: float = None):
        """Search for relevant episodes.

        Args:
            query: Search query (natural language)
            limit: Number of results to return
            threshold: Minimum similarity score (optional)

        Returns:
            List of search results with scores
        """
        # Generate embedding for query
        query_embedding = self.embedding_model.embed_text(query)

        # Search vector store
        results = self.vectorstore.search(query_embedding.tolist(), limit=limit)

        # Filter by threshold if provided
        if threshold:
            results = [r for r in results if r.get("_distance", 0) >= threshold]

        return results

    def get_recommendations(
        self,
        episode_id: str,
        limit: int = 5,
        podcast_id: str = None,
        exclude_episode: bool = True,
    ):
        """Get recommendations similar to a given episode.

        Args:
            episode_id: Episode ID to find similar episodes for
            limit: Number of recommendations
            podcast_id: Optional filter by podcast ID
            exclude_episode: Whether to exclude the source episode (default: True)

        Returns:
            List of similar episodes with metadata

        Raises:
            ValueError: If episode_id not found in vector store
            RuntimeError: If vector store is not initialized
        """
        if not self.vectorstore.table:
            raise RuntimeError("Vector store not initialized. Call create_table() first.")

        # Get the episode's embedding from the vector store
        table = self.vectorstore.get_table()

        # Query for the source episode
        try:
            source_records = table.search_where(
                f'episode_id = "{episode_id}"'
            ).limit(1).to_list()
        except Exception as e:
            raise ValueError(f"Episode {episode_id} not found: {str(e)}")

        if not source_records:
            raise ValueError(f"Episode {episode_id} not found in vector store")

        source_embedding = source_records[0].get("embedding")
        if source_embedding is None:
            raise ValueError(f"No embedding found for episode {episode_id}")

        # Search for similar episodes
        # Request limit+1 in case we need to exclude the source episode
        search_limit = limit + 1 if exclude_episode else limit
        results = self.vectorstore.search(
            source_embedding,
            table_name="transcripts",
            limit=search_limit,
        )

        # Filter out the source episode if requested
        if exclude_episode:
            results = [
                r for r in results
                if r.get("episode_id") != episode_id
            ]

        # Apply podcast filter if provided
        if podcast_id:
            results = [
                r for r in results
                if r.get("podcast_id") == podcast_id
            ]

        # Trim to requested limit
        results = results[:limit]

        return results

    def get_hybrid_recommendations(
        self,
        episode_ids: Union[str, List[str]],
        limit: int = 5,
        podcast_id: str = None,
        exclude_episodes: bool = True,
        diversity_boost: float = 0.0,
    ) -> List[dict]:
        """Get recommendations based on multiple episodes (hybrid approach).

        Combines embeddings from multiple episodes to find recommendations
        that are similar to the collective input.

        Args:
            episode_ids: Single episode ID (str) or list of episode IDs
            limit: Number of recommendations to return
            podcast_id: Optional filter by podcast ID
            exclude_episodes: Whether to exclude source episodes (default: True)
            diversity_boost: Score to promote diversity (0.0-1.0, default 0.0)

        Returns:
            List of similar episodes with metadata

        Raises:
            ValueError: If any episode_id not found or embedding issues
            RuntimeError: If vector store is not initialized
        """
        if not self.vectorstore.table:
            raise RuntimeError("Vector store not initialized. Call create_table() first.")

        # Normalize input to list
        if isinstance(episode_ids, str):
            episode_ids = [episode_ids]

        if not episode_ids:
            raise ValueError("episode_ids cannot be empty")

        if len(episode_ids) > 10:
            raise ValueError("Maximum 10 episodes supported for hybrid recommendations")

        # Get embeddings for all source episodes
        table = self.vectorstore.get_table()
        embeddings = []
        excluded_ids = set()

        for episode_id in episode_ids:
            try:
                source_records = table.search_where(
                    f'episode_id = "{episode_id}"'
                ).limit(1).to_list()
            except Exception as e:
                raise ValueError(f"Episode {episode_id} not found: {str(e)}")

            if not source_records:
                raise ValueError(f"Episode {episode_id} not found in vector store")

            embedding = source_records[0].get("embedding")
            if embedding is None:
                raise ValueError(f"No embedding found for episode {episode_id}")

            embeddings.append(embedding)
            excluded_ids.add(episode_id)

        # Compute average embedding (simple hybrid approach)
        if isinstance(embeddings[0], list):
            # Convert to numpy for averaging
            embeddings_array = np.array(embeddings)
            hybrid_embedding = embeddings_array.mean(axis=0).tolist()
        else:
            hybrid_embedding = embeddings[0]

        # Search for similar episodes
        search_limit = limit + len(episode_ids) if exclude_episodes else limit
        results = self.vectorstore.search(
            hybrid_embedding,
            table_name="transcripts",
            limit=search_limit,
        )

        # Filter out source episodes
        if exclude_episodes:
            results = [
                r for r in results
                if r.get("episode_id") not in excluded_ids
            ]

        # Apply podcast filter
        if podcast_id:
            results = [
                r for r in results
                if r.get("podcast_id") == podcast_id
            ]

        # Apply diversity boosting if requested
        if diversity_boost > 0.0:
            results = self._apply_diversity_scoring(results, diversity_boost)

        # Trim to requested limit
        results = results[:limit]

        return results

    def get_recommendations_with_date_filter(
        self,
        episode_id: str,
        limit: int = 5,
        podcast_id: str = None,
        exclude_episode: bool = True,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> List[dict]:
        """Get recommendations with optional date range filtering.

        Args:
            episode_id: Episode ID to find similar episodes for
            limit: Number of recommendations
            podcast_id: Optional filter by podcast ID
            exclude_episode: Whether to exclude source episode
            min_date: Minimum date (ISO format: YYYY-MM-DD), optional
            max_date: Maximum date (ISO format: YYYY-MM-DD), optional

        Returns:
            List of similar episodes matching date criteria

        Raises:
            ValueError: If episode_id not found or date format invalid
        """
        # Get base recommendations
        results = self.get_recommendations(
            episode_id=episode_id,
            limit=limit * 2,  # Request more to account for filtering
            podcast_id=podcast_id,
            exclude_episode=exclude_episode,
        )

        # Apply date filtering
        if min_date or max_date:
            results = self._filter_by_date_range(results, min_date, max_date)

        # Trim to requested limit
        results = results[:limit]

        return results

    def _apply_diversity_scoring(self, results: List[dict], boost_factor: float) -> List[dict]:
        """Apply diversity scoring to promote varied results.

        Penalizes results that are too similar to each other based on
        podcast or episode characteristics.

        Args:
            results: Search results to diversify
            boost_factor: Strength of diversity boost (0.0-1.0)

        Returns:
            Reordered results with diversity consideration
        """
        if not results or len(results) <= 1:
            return results

        # Track which podcasts/episode groups we've selected
        selected_podcasts = set()
        selected_episodes = set()
        diversified = []

        # First pass: select most similar (already sorted)
        for result in results:
            podcast_id = result.get("podcast_id")
            episode_id = result.get("episode_id")

            # Check diversity penalty
            podcast_count = selected_podcasts.count(podcast_id) if isinstance(selected_podcasts, list) else (1 if podcast_id in selected_podcasts else 0)
            penalty = podcast_count * boost_factor

            # Apply penalty to distance score
            original_distance = result.get("_distance", 0)
            penalized_distance = original_distance + penalty

            result_copy = result.copy()
            result_copy["_diversity_score"] = penalized_distance

            diversified.append(result_copy)
            selected_podcasts.add(podcast_id)
            selected_episodes.add(episode_id)

        # Re-sort by diversity-adjusted score
        diversified.sort(key=lambda x: x.get("_diversity_score", x.get("_distance", 0)))

        # Remove the temporary score field
        for result in diversified:
            result.pop("_diversity_score", None)

        return diversified

    def _filter_by_date_range(
        self,
        results: List[dict],
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
    ) -> List[dict]:
        """Filter results by date range.

        Args:
            results: Search results to filter
            min_date: Minimum date (ISO format)
            max_date: Maximum date (ISO format)

        Returns:
            Filtered results within date range
        """
        filtered = []

        for result in results:
            episode_date = result.get("episode_date")

            # Skip if no date
            if not episode_date:
                continue

            # Check min date
            if min_date and episode_date < min_date:
                continue

            # Check max date
            if max_date and episode_date > max_date:
                continue

            filtered.append(result)

        return filtered
