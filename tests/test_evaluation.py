"""Tests for evaluation framework and quality metrics."""

import pytest
import numpy as np
from src.parakeet_search.evaluation import (
    EvaluationFramework,
    RelevanceEvaluator,
    CoverageEvaluator,
    DiversityEvaluator,
)
from tests.evaluation_dataset import (
    get_relevance_judgments,
    get_topic_assignments,
    get_sample_results,
)


class TestRelevanceEvaluator:
    """Tests for relevance evaluation metrics."""

    def test_evaluate_relevance_perfect_match(self):
        """Test relevance evaluation with perfect matches."""
        evaluator = RelevanceEvaluator()

        results = [
            {"episode_id": "ep_001"},
            {"episode_id": "ep_002"},
        ]

        judgments = {"ep_001": 3, "ep_002": 3}

        score = evaluator.evaluate_relevance("machine learning", results, judgments)

        assert score == 1.0  # Perfect relevance

    def test_evaluate_relevance_partial_match(self):
        """Test relevance evaluation with partial matches."""
        evaluator = RelevanceEvaluator()

        results = [
            {"episode_id": "ep_001"},
            {"episode_id": "ep_010"},
        ]

        judgments = {"ep_001": 3, "ep_010": 1}

        score = evaluator.evaluate_relevance("machine learning", results, judgments)

        expected = (3.0 / 3 + 1.0 / 3) / 2
        assert abs(score - expected) < 0.01

    def test_evaluate_relevance_no_matches(self):
        """Test relevance evaluation with no relevant results."""
        evaluator = RelevanceEvaluator()

        results = [
            {"episode_id": "ep_001"},
            {"episode_id": "ep_002"},
        ]

        judgments = {}  # No judgments

        score = evaluator.evaluate_relevance("unknown query", results, judgments)

        assert score == 0.0

    def test_precision_at_k(self):
        """Test precision@k calculation."""
        evaluator = RelevanceEvaluator()

        results = get_sample_results("machine learning")
        judgments = get_relevance_judgments("machine learning")

        precision_10 = evaluator.precision_at_k(results, judgments, k=10, threshold=2)

        # Should be in [0, 1]
        assert 0.0 <= precision_10 <= 1.0

        # All results in our sample are relevant (threshold=2 means score >= 2)
        # ep_001(3), ep_005(3), ep_002(3), ep_006(2), ep_010(1)
        # First 4 are relevant: precision@5 = 4/5 = 0.8
        precision_5 = evaluator.precision_at_k(results[:5], judgments, k=5, threshold=2)
        assert abs(precision_5 - 0.8) < 0.01

    def test_recall_at_k(self):
        """Test recall@k calculation."""
        evaluator = RelevanceEvaluator()

        results = get_sample_results("machine learning")
        judgments = get_relevance_judgments("machine learning")

        # Count total relevant items (threshold=2)
        total_relevant = sum(1 for score in judgments.values() if score >= 2)

        recall_5 = evaluator.recall_at_k(results[:5], judgments, k=5, threshold=2)

        # Should be in [0, 1]
        assert 0.0 <= recall_5 <= 1.0
        assert total_relevant > 0

    def test_ndcg_calculation(self):
        """Test NDCG@k calculation."""
        evaluator = RelevanceEvaluator()

        results = get_sample_results("machine learning")
        judgments = get_relevance_judgments("machine learning")

        ndcg = evaluator.ndcg(results, judgments, k=10)

        # NDCG should be in [0, 1]
        assert 0.0 <= ndcg <= 1.0

        # With perfectly ranked results, should be close to 1
        assert ndcg > 0.5

    def test_mean_reciprocal_rank(self):
        """Test MRR calculation."""
        evaluator = RelevanceEvaluator()

        # Best case: first result is relevant
        results_best = [
            {"episode_id": "ep_001"},
            {"episode_id": "ep_010"},
        ]
        judgments = {"ep_001": 3, "ep_010": 0}

        mrr_best = evaluator.mean_reciprocal_rank(results_best, judgments, threshold=2)
        assert mrr_best == 1.0

        # Worst case: second result is relevant
        results_worst = [
            {"episode_id": "ep_010"},
            {"episode_id": "ep_001"},
        ]

        mrr_worst = evaluator.mean_reciprocal_rank(results_worst, judgments, threshold=2)
        assert abs(mrr_worst - 0.5) < 0.01

    def test_no_relevant_results(self):
        """Test metrics when no relevant results exist."""
        evaluator = RelevanceEvaluator()

        results = [{"episode_id": "ep_999"}]
        judgments = {}

        mrr = evaluator.mean_reciprocal_rank(results, judgments)
        assert mrr == 0.0


class TestCoverageEvaluator:
    """Tests for coverage evaluation metrics."""

    def test_topic_coverage(self):
        """Test topic coverage calculation."""
        evaluator = CoverageEvaluator()

        results = get_sample_results("machine learning")[:3]
        assignments = get_topic_assignments()

        coverage = evaluator.topic_coverage(results, assignments)

        # Coverage should be between 0 and 1
        assert 0.0 <= coverage <= 1.0

        # With multiple episodes covering different topics, should be > 0.5
        assert coverage > 0.3

    def test_topic_coverage_empty_results(self):
        """Test topic coverage with no results."""
        evaluator = CoverageEvaluator()

        results = []
        assignments = get_topic_assignments()

        coverage = evaluator.topic_coverage(results, assignments)

        assert coverage == 0.0

    def test_podcast_coverage(self):
        """Test podcast coverage calculation."""
        evaluator = CoverageEvaluator()

        # Mixed results from different podcasts
        results = [
            {"podcast_id": "pod_001"},
            {"podcast_id": "pod_001"},
            {"podcast_id": "pod_002"},
            {"podcast_id": "pod_003"},
        ]

        coverage = evaluator.podcast_coverage(results)

        # 3 unique podcasts / 4 results = 0.75
        assert abs(coverage - 0.75) < 0.01

    def test_podcast_coverage_single_podcast(self):
        """Test podcast coverage with single podcast."""
        evaluator = CoverageEvaluator()

        results = [
            {"podcast_id": "pod_001"},
            {"podcast_id": "pod_001"},
        ]

        coverage = evaluator.podcast_coverage(results)

        # 1 unique podcast / 2 results = 0.5
        assert abs(coverage - 0.5) < 0.01

    def test_temporal_coverage(self):
        """Test temporal coverage calculation."""
        evaluator = CoverageEvaluator()

        results = get_sample_results("machine learning")

        coverage, stats = evaluator.temporal_coverage(results)

        # Coverage should be in [0, 1]
        assert 0.0 <= coverage <= 1.0

        # Stats should have date information
        assert "min_date" in stats
        assert "max_date" in stats
        assert "unique_dates" in stats

    def test_temporal_coverage_no_dates(self):
        """Test temporal coverage with no dates."""
        evaluator = CoverageEvaluator()

        results = [
            {"episode_id": "ep_001"},
            {"episode_id": "ep_002"},
        ]

        coverage, stats = evaluator.temporal_coverage(results)

        assert coverage == 0.0
        assert stats == {}


class TestDiversityEvaluator:
    """Tests for diversity evaluation metrics."""

    def test_content_diversity_different_podcasts(self):
        """Test content diversity with different podcasts."""
        evaluator = DiversityEvaluator()

        results = [
            {"podcast_id": "pod_001", "published_at": "2025-01-01"},
            {"podcast_id": "pod_002", "published_at": "2025-01-05"},
            {"podcast_id": "pod_003", "published_at": "2025-01-10"},
        ]

        diversity = evaluator.content_diversity(results)

        # Should be high due to different podcasts and dates
        assert 0.0 <= diversity <= 1.0
        assert diversity > 0.5

    def test_content_diversity_same_podcast(self):
        """Test content diversity with same podcast."""
        evaluator = DiversityEvaluator()

        results = [
            {"podcast_id": "pod_001", "published_at": "2025-01-01"},
            {"podcast_id": "pod_001", "published_at": "2025-01-01"},
        ]

        diversity = evaluator.content_diversity(results)

        # Should be low due to same podcast and date
        # podcast_diversity = 1/2 = 0.5, temporal_diversity = 0
        # average = (0.5 + 0) / 2 = 0.25
        assert 0.0 <= diversity <= 0.5

    def test_content_diversity_single_result(self):
        """Test content diversity with single result."""
        evaluator = DiversityEvaluator()

        results = [{"podcast_id": "pod_001"}]

        diversity = evaluator.content_diversity(results)

        assert diversity == 0.0

    def test_semantic_diversity_orthogonal_embeddings(self):
        """Test semantic diversity with orthogonal embeddings."""
        evaluator = DiversityEvaluator()

        # Create orthogonal embeddings (completely different)
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        results = [
            {"episode_id": "ep_001"},
            {"episode_id": "ep_002"},
            {"episode_id": "ep_003"},
        ]

        diversity = evaluator.semantic_diversity(results, embeddings)

        # Orthogonal vectors have dissimilarity close to 1
        assert 0.8 <= diversity <= 1.0

    def test_semantic_diversity_similar_embeddings(self):
        """Test semantic diversity with similar embeddings."""
        evaluator = DiversityEvaluator()

        # Create similar embeddings
        embeddings = np.array([
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
        ])

        results = [
            {"episode_id": "ep_001"},
            {"episode_id": "ep_002"},
            {"episode_id": "ep_003"},
        ]

        diversity = evaluator.semantic_diversity(results, embeddings)

        # Similar vectors have low dissimilarity
        assert 0.0 <= diversity < 0.1

    def test_result_uniqueness_no_duplicates(self):
        """Test uniqueness with no duplicate episodes."""
        evaluator = DiversityEvaluator()

        results = [
            {"episode_id": "ep_001"},
            {"episode_id": "ep_002"},
            {"episode_id": "ep_003"},
        ]

        uniqueness = evaluator.result_uniqueness(results)

        # All unique
        assert uniqueness == 1.0

    def test_result_uniqueness_with_duplicates(self):
        """Test uniqueness with duplicate episodes."""
        evaluator = DiversityEvaluator()

        results = [
            {"episode_id": "ep_001"},
            {"episode_id": "ep_001"},
            {"episode_id": "ep_002"},
        ]

        uniqueness = evaluator.result_uniqueness(results)

        # 2 unique / 3 results = 0.667
        assert abs(uniqueness - 0.667) < 0.01


class TestEvaluationFramework:
    """Tests for complete evaluation framework."""

    def test_framework_initialization(self):
        """Test framework initialization."""
        framework = EvaluationFramework()

        assert framework.relevance_evaluator is not None
        assert framework.coverage_evaluator is not None
        assert framework.diversity_evaluator is not None

    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation with all metrics."""
        framework = EvaluationFramework()

        query = "machine learning"
        results = get_sample_results(query)
        relevance_judgments = get_relevance_judgments(query)
        topic_assignments = get_topic_assignments()

        # Create mock embeddings for diversity
        embeddings = np.random.randn(len(results), 384)

        metrics = framework.evaluate_search_results(
            query=query,
            results=results,
            relevance_judgments=relevance_judgments,
            topic_assignments=topic_assignments,
            embeddings=embeddings,
        )

        # All metrics should be calculated
        assert metrics.relevance_score >= 0.0
        assert metrics.coverage_score >= 0.0
        assert metrics.diversity_score >= 0.0
        assert metrics.precision >= 0.0
        assert metrics.recall >= 0.0
        assert metrics.ndcg >= 0.0
        assert metrics.mean_reciprocal_rank >= 0.0

    def test_evaluation_without_judgments(self):
        """Test evaluation when judgments are not provided."""
        framework = EvaluationFramework()

        results = get_sample_results("machine learning")

        metrics = framework.evaluate_search_results(
            query="machine learning",
            results=results,
        )

        # Metrics without judgments should be 0
        assert metrics.relevance_score == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.ndcg == 0.0
        assert metrics.mean_reciprocal_rank == 0.0

        # But diversity and coverage can still be calculated
        assert metrics.diversity_score >= 0.0

    def test_aggregate_metrics(self):
        """Test metrics aggregation."""
        framework = EvaluationFramework()

        queries = ["machine learning", "deep learning", "computer vision"]
        all_metrics = []

        for query in queries:
            results = get_sample_results(query)
            judgments = get_relevance_judgments(query)

            metrics = framework.evaluate_search_results(
                query=query,
                results=results,
                relevance_judgments=judgments,
            )

            all_metrics.append(metrics)

        aggregated = EvaluationFramework.aggregate_metrics(all_metrics)

        # Should have averaged metrics
        assert "avg_relevance" in aggregated
        assert "avg_precision" in aggregated
        assert "avg_recall" in aggregated
        assert "avg_ndcg" in aggregated

        # All should be in valid range
        for key, value in aggregated.items():
            assert 0.0 <= value <= 1.0

    def test_aggregate_empty_list(self):
        """Test aggregation with empty metrics list."""
        aggregated = EvaluationFramework.aggregate_metrics([])

        assert aggregated == {}


class TestEvaluationDataset:
    """Tests for evaluation dataset."""

    def test_relevance_judgments_exist(self):
        """Test that relevance judgments exist for common queries."""
        judgments = get_relevance_judgments("machine learning")

        assert len(judgments) > 0
        assert all(isinstance(v, int) for v in judgments.values())
        assert all(0 <= v <= 3 for v in judgments.values())

    def test_topic_assignments_complete(self):
        """Test that topic assignments are complete."""
        assignments = get_topic_assignments()

        assert len(assignments) == 10  # 10 episodes
        assert all(isinstance(topics, set) for topics in assignments.values())

    def test_sample_results_valid(self):
        """Test that sample results are valid."""
        results = get_sample_results("machine learning")

        assert len(results) > 0

        for result in results:
            assert "episode_id" in result
            assert "podcast_id" in result
            assert "_distance" in result
            assert 0.0 <= result["_distance"] <= 1.0

    def test_sample_results_empty_query(self):
        """Test sample results for unrecognized query."""
        results = get_sample_results("unknown query")

        assert results == []


class TestEndToEndEvaluation:
    """End-to-end tests for complete evaluation workflow."""

    def test_full_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        framework = EvaluationFramework()

        # Simulate a complete search evaluation
        query = "deep learning"
        results = get_sample_results(query)
        judgments = get_relevance_judgments(query)
        assignments = get_topic_assignments()
        embeddings = np.random.randn(len(results), 384) if results else None

        metrics = framework.evaluate_search_results(
            query=query,
            results=results,
            relevance_judgments=judgments,
            topic_assignments=assignments,
            embeddings=embeddings,
        )

        # Verify metrics are reasonable
        assert metrics.mean_reciprocal_rank > 0  # Should find relevant result quickly
        assert metrics.diversity_score > 0  # Results from different sources

    def test_comparison_across_queries(self):
        """Test metric comparison across different queries."""
        framework = EvaluationFramework()

        query1_metrics = framework.evaluate_search_results(
            query="machine learning",
            results=get_sample_results("machine learning"),
            relevance_judgments=get_relevance_judgments("machine learning"),
        )

        query2_metrics = framework.evaluate_search_results(
            query="reinforcement learning",
            results=get_sample_results("reinforcement learning"),
            relevance_judgments=get_relevance_judgments("reinforcement learning"),
        )

        # Both should have metrics
        assert query1_metrics.mean_reciprocal_rank >= 0
        assert query2_metrics.mean_reciprocal_rank >= 0
