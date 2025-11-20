"""Search engine implementation."""

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
