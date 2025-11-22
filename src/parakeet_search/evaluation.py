"""Quality metrics and evaluation framework for semantic search."""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    relevance_score: float
    coverage_score: float
    diversity_score: float
    precision: float
    recall: float
    ndcg: float
    mean_reciprocal_rank: float


class RelevanceEvaluator:
    """Evaluates relevance of search results to queries."""

    @staticmethod
    def evaluate_relevance(
        query: str,
        results: List[Dict],
        relevance_judgments: Dict[str, int],
    ) -> float:
        """Evaluate relevance of results based on human judgments.

        Args:
            query: Search query
            results: List of search results with metadata
            relevance_judgments: Dict mapping result_id to relevance score (0-3)
                0: Not relevant
                1: Slightly relevant
                2: Relevant
                3: Highly relevant

        Returns:
            Relevance score (0-1)
        """
        if not results:
            return 0.0

        total_relevance = 0.0
        count = 0

        for result in results:
            result_id = result.get("episode_id")
            if result_id in relevance_judgments:
                relevance = relevance_judgments[result_id]
                total_relevance += relevance / 3.0  # Normalize to 0-1
                count += 1

        return total_relevance / len(results) if results else 0.0

    @staticmethod
    def precision_at_k(
        results: List[Dict],
        relevance_judgments: Dict[str, int],
        k: int = 10,
        threshold: int = 2,
    ) -> float:
        """Calculate precision@k (fraction of top-k results that are relevant).

        Args:
            results: List of search results
            relevance_judgments: Dict mapping result_id to relevance score (0-3)
            k: Cutoff position
            threshold: Minimum relevance score to consider as relevant

        Returns:
            Precision@k score (0-1)
        """
        if not results or k == 0:
            return 0.0

        top_k = results[:k]
        relevant_count = 0

        for result in top_k:
            result_id = result.get("episode_id")
            if result_id in relevance_judgments:
                if relevance_judgments[result_id] >= threshold:
                    relevant_count += 1

        return relevant_count / k

    @staticmethod
    def recall_at_k(
        results: List[Dict],
        relevance_judgments: Dict[str, int],
        k: int = 10,
        threshold: int = 2,
    ) -> float:
        """Calculate recall@k (fraction of all relevant items in top-k).

        Args:
            results: List of search results
            relevance_judgments: Dict mapping result_id to relevance score (0-3)
            k: Cutoff position
            threshold: Minimum relevance score to consider as relevant

        Returns:
            Recall@k score (0-1)
        """
        # Count total relevant items
        total_relevant = sum(
            1 for score in relevance_judgments.values()
            if score >= threshold
        )

        if total_relevant == 0:
            return 0.0

        # Count relevant items in top-k
        top_k = results[:k]
        relevant_in_top_k = 0

        for result in top_k:
            result_id = result.get("episode_id")
            if result_id in relevance_judgments:
                if relevance_judgments[result_id] >= threshold:
                    relevant_in_top_k += 1

        return relevant_in_top_k / total_relevant

    @staticmethod
    def ndcg(
        results: List[Dict],
        relevance_judgments: Dict[str, int],
        k: int = 10,
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain (NDCG).

        Measures ranking quality, giving higher weight to relevant results
        that appear earlier in the ranking.

        Args:
            results: List of search results
            relevance_judgments: Dict mapping result_id to relevance score (0-3)
            k: Cutoff position

        Returns:
            NDCG@k score (0-1)
        """
        def dcg(results_list, judgments, cutoff):
            """Calculate Discounted Cumulative Gain."""
            dcg_score = 0.0
            for i, result in enumerate(results_list[:cutoff], 1):
                result_id = result.get("episode_id")
                relevance = judgments.get(result_id, 0)
                dcg_score += relevance / np.log2(i + 1)
            return dcg_score

        # Calculate DCG for actual results
        actual_dcg = dcg(results, relevance_judgments, k)

        # Calculate ideal DCG (all relevant items ranked first)
        relevant_scores = sorted(
            [score for score in relevance_judgments.values()],
            reverse=True
        )[:k]
        ideal_dcg = sum(
            score / np.log2(i + 2) for i, score in enumerate(relevant_scores)
        )

        if ideal_dcg == 0:
            return 0.0

        return actual_dcg / ideal_dcg

    @staticmethod
    def mean_reciprocal_rank(
        results: List[Dict],
        relevance_judgments: Dict[str, int],
        threshold: int = 2,
    ) -> float:
        """Calculate Mean Reciprocal Rank (MRR).

        Measures how early the first relevant result appears.

        Args:
            results: List of search results
            relevance_judgments: Dict mapping result_id to relevance score (0-3)
            threshold: Minimum relevance score to consider as relevant

        Returns:
            MRR score (0-1)
        """
        for i, result in enumerate(results, 1):
            result_id = result.get("episode_id")
            if result_id in relevance_judgments:
                if relevance_judgments[result_id] >= threshold:
                    return 1.0 / i

        return 0.0


class CoverageEvaluator:
    """Evaluates coverage of search results (how comprehensively results span content)."""

    @staticmethod
    def topic_coverage(
        results: List[Dict],
        topic_assignments: Dict[str, Set[str]],
    ) -> float:
        """Evaluate coverage of topics in search results.

        Args:
            results: List of search results
            topic_assignments: Dict mapping episode_id to set of topic tags

        Returns:
            Topic coverage score (0-1)
        """
        if not results:
            return 0.0

        covered_topics = set()

        for result in results:
            result_id = result.get("episode_id")
            if result_id in topic_assignments:
                covered_topics.update(topic_assignments[result_id])

        # Calculate as fraction of unique topics
        all_topics = set()
        for topics in topic_assignments.values():
            all_topics.update(topics)

        if not all_topics:
            return 0.0

        return len(covered_topics) / len(all_topics)

    @staticmethod
    def podcast_coverage(results: List[Dict]) -> float:
        """Evaluate coverage of different podcasts in results.

        Args:
            results: List of search results

        Returns:
            Podcast coverage score (0-1)
        """
        if not results:
            return 0.0

        podcasts = set()
        for result in results:
            podcast_id = result.get("podcast_id")
            if podcast_id:
                podcasts.add(podcast_id)

        # Coverage is number of unique podcasts / total results
        # (higher diversity in sources is better)
        return len(podcasts) / len(results) if results else 0.0

    @staticmethod
    def temporal_coverage(
        results: List[Dict],
    ) -> Tuple[float, Dict]:
        """Evaluate temporal distribution of results.

        Args:
            results: List of search results with 'published_at' field

        Returns:
            Tuple of (temporal_score, distribution_stats)
        """
        if not results:
            return 0.0, {}

        dates = []
        for result in results:
            date = result.get("published_at")
            if date:
                dates.append(date)

        if not dates:
            return 0.0, {}

        # Calculate date range span
        min_date = min(dates)
        max_date = max(dates)
        date_range = max_date > min_date

        # Calculate uniformity of distribution
        dates_sorted = sorted(dates)
        gaps = []
        for i in range(len(dates_sorted) - 1):
            # This is a simplistic gap calculation
            gap = 1 if dates_sorted[i] != dates_sorted[i + 1] else 0
            gaps.append(gap)

        temporal_score = sum(gaps) / len(gaps) if gaps else 0.0

        return temporal_score, {
            "min_date": min_date,
            "max_date": max_date,
            "unique_dates": len(set(dates)),
            "temporal_uniformity": temporal_score,
        }


class DiversityEvaluator:
    """Evaluates diversity of search results."""

    @staticmethod
    def content_diversity(
        results: List[Dict],
        similarity_matrix: Optional[np.ndarray] = None,
    ) -> float:
        """Evaluate content diversity using embeddings or similarity.

        Args:
            results: List of search results
            similarity_matrix: Pre-computed similarity matrix (optional)

        Returns:
            Diversity score (0-1)
        """
        if len(results) <= 1:
            return 0.0

        # If similarity matrix provided, use it
        if similarity_matrix is not None:
            # Calculate average pairwise dissimilarity
            n = len(results)
            if n < 2:
                return 0.0

            total_dissimilarity = 0.0
            count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    dissimilarity = 1.0 - similarity_matrix[i][j]
                    total_dissimilarity += dissimilarity
                    count += 1

            return total_dissimilarity / count if count > 0 else 0.0

        # Otherwise, use heuristic: different podcasts + different dates
        podcasts = set()
        dates = set()

        for result in results:
            podcasts.add(result.get("podcast_id"))
            date = result.get("published_at")
            if date:
                dates.add(date)

        # Diversity = average uniqueness across dimensions
        podcast_diversity = len(podcasts) / len(results)
        temporal_diversity = len(dates) / len(results) if dates else 0.0

        return (podcast_diversity + temporal_diversity) / 2.0

    @staticmethod
    def result_uniqueness(results: List[Dict]) -> float:
        """Evaluate uniqueness of results (no duplicates).

        Args:
            results: List of search results

        Returns:
            Uniqueness score (0-1, where 1 = all unique)
        """
        if not results:
            return 0.0

        episode_ids = [r.get("episode_id") for r in results]
        unique_episodes = len(set(episode_ids))

        return unique_episodes / len(results)

    @staticmethod
    def semantic_diversity(
        results: List[Dict],
        embeddings: np.ndarray,
    ) -> float:
        """Evaluate semantic diversity of results using embeddings.

        Args:
            results: List of search results
            embeddings: 2D numpy array of embeddings

        Returns:
            Semantic diversity score (0-1)
        """
        if len(embeddings) <= 1:
            return 0.0

        # Calculate pairwise cosine distances
        normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = np.dot(normalized, normalized.T)

        # Calculate average distance (dissimilarity)
        n = len(embeddings)
        total_distance = 0.0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                distance = 1.0 - similarity_matrix[i][j]
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else 0.0


class EvaluationFramework:
    """Complete evaluation framework combining all metrics."""

    def __init__(self):
        """Initialize evaluation framework."""
        self.relevance_evaluator = RelevanceEvaluator()
        self.coverage_evaluator = CoverageEvaluator()
        self.diversity_evaluator = DiversityEvaluator()

    def evaluate_search_results(
        self,
        query: str,
        results: List[Dict],
        relevance_judgments: Optional[Dict[str, int]] = None,
        topic_assignments: Optional[Dict[str, Set[str]]] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> EvaluationMetrics:
        """Comprehensive evaluation of search results.

        Args:
            query: Search query
            results: List of search results
            relevance_judgments: Optional human relevance judgments
            topic_assignments: Optional topic assignments per episode
            embeddings: Optional embeddings for semantic diversity evaluation

        Returns:
            EvaluationMetrics with all scores
        """
        # Relevance metrics
        relevance_score = (
            self.relevance_evaluator.evaluate_relevance(
                query, results, relevance_judgments or {}
            )
            if relevance_judgments
            else 0.0
        )

        precision = (
            self.relevance_evaluator.precision_at_k(
                results, relevance_judgments or {}, k=10
            )
            if relevance_judgments
            else 0.0
        )

        recall = (
            self.relevance_evaluator.recall_at_k(
                results, relevance_judgments or {}, k=10
            )
            if relevance_judgments
            else 0.0
        )

        ndcg = (
            self.relevance_evaluator.ndcg(
                results, relevance_judgments or {}, k=10
            )
            if relevance_judgments
            else 0.0
        )

        mrr = (
            self.relevance_evaluator.mean_reciprocal_rank(
                results, relevance_judgments or {}
            )
            if relevance_judgments
            else 0.0
        )

        # Coverage metrics
        coverage_score = (
            self.coverage_evaluator.topic_coverage(
                results, topic_assignments or {}
            )
            if topic_assignments
            else 0.0
        )

        # Diversity metrics
        diversity_score = (
            self.diversity_evaluator.semantic_diversity(results, embeddings)
            if embeddings is not None
            else self.diversity_evaluator.content_diversity(results)
        )

        return EvaluationMetrics(
            relevance_score=relevance_score,
            coverage_score=coverage_score,
            diversity_score=diversity_score,
            precision=precision,
            recall=recall,
            ndcg=ndcg,
            mean_reciprocal_rank=mrr,
        )

    @staticmethod
    def aggregate_metrics(
        metrics_list: List[EvaluationMetrics],
    ) -> Dict[str, float]:
        """Aggregate evaluation metrics across multiple evaluations.

        Args:
            metrics_list: List of EvaluationMetrics

        Returns:
            Dict with averaged metrics
        """
        if not metrics_list:
            return {}

        n = len(metrics_list)
        return {
            "avg_relevance": sum(m.relevance_score for m in metrics_list) / n,
            "avg_coverage": sum(m.coverage_score for m in metrics_list) / n,
            "avg_diversity": sum(m.diversity_score for m in metrics_list) / n,
            "avg_precision": sum(m.precision for m in metrics_list) / n,
            "avg_recall": sum(m.recall for m in metrics_list) / n,
            "avg_ndcg": sum(m.ndcg for m in metrics_list) / n,
            "avg_mrr": sum(m.mean_reciprocal_rank for m in metrics_list) / n,
        }
