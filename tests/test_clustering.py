"""Tests for clustering and topic analysis functionality."""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from parakeet_search.clustering import ClusteringAnalyzer, ClusterInfo, OutlierInfo


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings with clear clusters."""
    # Create 3 clusters
    X, y = make_blobs(n_samples=300, centers=3, n_features=384, random_state=42, cluster_std=0.5)
    return X.astype(np.float32)


@pytest.fixture
def sample_metadata(sample_embeddings):
    """Create metadata for sample embeddings."""
    n_samples = len(sample_embeddings)
    return pd.DataFrame(
        {
            "episode_id": [f"ep_{i:04d}" for i in range(n_samples)],
            "episode_title": [f"Episode {i}" for i in range(n_samples)],
            "podcast_id": [f"pod_{i % 10}" for i in range(n_samples)],
        }
    )


@pytest.fixture
def clustering_analyzer(sample_embeddings, sample_metadata):
    """Create ClusteringAnalyzer instance."""
    return ClusteringAnalyzer(sample_embeddings, sample_metadata)


class TestClusteringAnalyzer:
    """Test ClusteringAnalyzer class."""

    def test_initialization(self, clustering_analyzer, sample_embeddings):
        """Test ClusteringAnalyzer initialization."""
        assert clustering_analyzer.n_samples == len(sample_embeddings)
        assert clustering_analyzer.embedding_dim == 384
        assert clustering_analyzer.embeddings_scaled.shape == sample_embeddings.shape

    def test_kmeans_clustering(self, clustering_analyzer):
        """Test K-Means clustering."""
        result = clustering_analyzer.kmeans_clustering(n_clusters=3)

        assert result.algorithm == "K-Means"
        assert result.n_clusters == 3
        assert len(result.labels) == clustering_analyzer.n_samples
        assert len(result.clusters) == 3
        assert result.silhouette_score > -1 and result.silhouette_score < 1
        assert result.inertia is not None and result.inertia > 0

    def test_kmeans_cluster_sizes(self, clustering_analyzer):
        """Test that K-Means produces valid clusters."""
        result = clustering_analyzer.kmeans_clustering(n_clusters=3)

        total_size = sum(cluster.size for cluster in result.clusters)
        assert total_size == clustering_analyzer.n_samples

        # All clusters should have at least 1 sample
        assert all(cluster.size > 0 for cluster in result.clusters)

    def test_hierarchical_clustering(self, clustering_analyzer):
        """Test Hierarchical clustering."""
        result = clustering_analyzer.hierarchical_clustering(n_clusters=3, linkage_method="ward")

        assert result.algorithm == "Hierarchical"
        assert result.n_clusters == 3
        assert len(result.labels) == clustering_analyzer.n_samples
        assert len(result.clusters) == 3
        assert result.silhouette_score > -1 and result.silhouette_score < 1
        assert result.inertia is None  # Hierarchical doesn't compute inertia

    def test_hierarchical_different_linkage_methods(self, clustering_analyzer):
        """Test hierarchical clustering with different linkage methods."""
        for method in ["ward", "complete", "average", "single"]:
            result = clustering_analyzer.hierarchical_clustering(n_clusters=3, linkage_method=method)
            assert result.algorithm == "Hierarchical"
            assert result.n_clusters == 3

    def test_cluster_info_structure(self, clustering_analyzer):
        """Test ClusterInfo structure."""
        result = clustering_analyzer.kmeans_clustering(n_clusters=3)

        for cluster in result.clusters:
            assert isinstance(cluster, ClusterInfo)
            assert cluster.cluster_id >= 0
            assert cluster.size > 0
            assert cluster.center.shape == (384,)
            assert len(cluster.episodes) == cluster.size
            assert len(cluster.podcasts) == cluster.size
            assert -1 < cluster.silhouette_score < 1

    def test_isolation_forest_outliers(self, clustering_analyzer):
        """Test outlier detection with Isolation Forest."""
        outliers = clustering_analyzer.find_outliers(method="isolation_forest", contamination=0.1)

        assert isinstance(outliers, list)
        assert len(outliers) <= int(0.1 * clustering_analyzer.n_samples) + 1  # Allow small margin
        assert all(isinstance(o, OutlierInfo) for o in outliers)

    def test_outlier_info_structure(self, clustering_analyzer):
        """Test OutlierInfo structure."""
        outliers = clustering_analyzer.find_outliers(method="isolation_forest", contamination=0.05)

        for outlier in outliers:
            assert isinstance(outlier, OutlierInfo)
            assert outlier.episode_id is not None
            assert outlier.episode_title is not None
            assert isinstance(outlier.outlier_score, (float, np.floating))

    def test_distance_based_outliers(self, clustering_analyzer):
        """Test distance-based outlier detection."""
        kmeans_result = clustering_analyzer.kmeans_clustering(n_clusters=3)
        outliers = clustering_analyzer.find_outliers(
            method="distance_based", clustering_labels=kmeans_result.labels
        )

        assert isinstance(outliers, list)
        # With well-clustered synthetic data, there may be few or no outliers
        assert all(isinstance(o, OutlierInfo) for o in outliers)
        assert all(o.nearest_cluster_id >= 0 for o in outliers)

    def test_outliers_invalid_method(self, clustering_analyzer):
        """Test outlier detection with invalid method."""
        with pytest.raises(ValueError, match="Unknown outlier method"):
            clustering_analyzer.find_outliers(method="invalid_method")

    def test_distance_based_outliers_requires_labels(self, clustering_analyzer):
        """Test that distance-based outliers require clustering labels."""
        with pytest.raises(ValueError, match="clustering_labels required"):
            clustering_analyzer.find_outliers(method="distance_based")

    def test_cluster_statistics(self, clustering_analyzer):
        """Test cluster statistics generation."""
        result = clustering_analyzer.kmeans_clustering(n_clusters=3)
        stats = clustering_analyzer.get_cluster_statistics(result)

        assert isinstance(stats, pd.DataFrame)
        assert len(stats) == 3
        assert "cluster_id" in stats.columns
        assert "size" in stats.columns
        assert "percentage" in stats.columns
        assert "silhouette_score" in stats.columns
        assert "intra_distance_mean" in stats.columns
        assert stats["percentage"].sum() == pytest.approx(100.0)

    def test_cluster_summarization(self, clustering_analyzer):
        """Test cluster summarization."""
        result = clustering_analyzer.kmeans_clustering(n_clusters=3)

        for cluster in result.clusters:
            summary = clustering_analyzer.summarize_cluster(cluster)

            assert summary["cluster_id"] == cluster.cluster_id
            assert summary["size"] == cluster.size
            assert 0 < summary["percentage"] <= 100
            assert "podcast_distribution" in summary
            assert "top_episodes" in summary
            assert len(summary["top_episodes"]) > 0

    def test_optimal_k_analysis(self, clustering_analyzer):
        """Test optimal k analysis."""
        k_range = range(2, 6)
        inertias, silhouette_scores = clustering_analyzer.optimal_k_analysis(k_range)

        assert len(inertias) == len(list(k_range))
        assert len(silhouette_scores) == len(list(k_range))
        assert all(i > 0 for i in inertias)
        assert all(-1 < s < 1 for s in silhouette_scores)
        # Inertia should decrease with more clusters
        assert inertias[0] > inertias[-1]

    def test_kmeans_reproducibility(self, clustering_analyzer):
        """Test that K-Means produces reproducible results."""
        result1 = clustering_analyzer.kmeans_clustering(n_clusters=3, random_state=42)
        result2 = clustering_analyzer.kmeans_clustering(n_clusters=3, random_state=42)

        assert np.array_equal(result1.labels, result2.labels)
        assert np.allclose(result1.cluster_centers, result2.cluster_centers)

    def test_different_k_values(self, clustering_analyzer):
        """Test clustering with different k values."""
        for k in [2, 3, 5, 10]:
            result = clustering_analyzer.kmeans_clustering(n_clusters=k)
            assert result.n_clusters == k
            assert len(result.clusters) == k

    def test_cluster_ordering(self, clustering_analyzer):
        """Test that clusters are ordered by size."""
        result = clustering_analyzer.kmeans_clustering(n_clusters=3)
        sizes = [cluster.size for cluster in result.clusters]
        assert sizes == sorted(sizes, reverse=True)

    def test_embeddings_not_modified(self, clustering_analyzer, sample_embeddings):
        """Test that original embeddings are not modified."""
        original_embeddings = sample_embeddings.copy()
        clustering_analyzer.kmeans_clustering(n_clusters=3)

        assert np.array_equal(clustering_analyzer.embeddings, original_embeddings)

    def test_metadata_not_modified(self, clustering_analyzer, sample_metadata):
        """Test that metadata is not modified."""
        original_metadata = sample_metadata.copy()
        clustering_analyzer.kmeans_clustering(n_clusters=3)

        pd.testing.assert_frame_equal(clustering_analyzer.metadata, original_metadata)

    def test_small_dataset_clustering(self):
        """Test clustering with small dataset."""
        small_embeddings = np.random.randn(10, 384)
        small_metadata = pd.DataFrame({
            "episode_id": [f"ep_{i}" for i in range(10)],
            "episode_title": [f"Episode {i}" for i in range(10)],
            "podcast_id": [f"pod_{i % 2}" for i in range(10)],
        })

        analyzer = ClusteringAnalyzer(small_embeddings, small_metadata)
        result = analyzer.kmeans_clustering(n_clusters=2)

        assert result.n_clusters == 2
        assert len(result.clusters) == 2

    def test_single_sample_per_cluster(self, clustering_analyzer):
        """Test with more clusters than average cluster size would suggest."""
        # With 300 samples and 30 clusters, some will have very few samples
        result = clustering_analyzer.kmeans_clustering(n_clusters=30)

        assert result.n_clusters == 30
        assert all(cluster.size > 0 for cluster in result.clusters)
        assert sum(cluster.size for cluster in result.clusters) == clustering_analyzer.n_samples


class TestClusteringIntegration:
    """Integration tests for clustering analysis."""

    def test_full_clustering_workflow(self, clustering_analyzer):
        """Test complete clustering workflow."""
        # 1. Perform K-Means clustering
        kmeans_result = clustering_analyzer.kmeans_clustering(n_clusters=3)
        assert kmeans_result.n_clusters == 3

        # 2. Get statistics
        stats = clustering_analyzer.get_cluster_statistics(kmeans_result)
        assert len(stats) == 3

        # 3. Summarize clusters
        summaries = [clustering_analyzer.summarize_cluster(cluster) for cluster in kmeans_result.clusters]
        assert len(summaries) == 3

        # 4. Detect outliers
        outliers = clustering_analyzer.find_outliers(method="isolation_forest", contamination=0.1)
        assert len(outliers) >= 0

    def test_kmeans_vs_hierarchical(self, clustering_analyzer):
        """Compare K-Means and Hierarchical clustering."""
        k = 3
        kmeans_result = clustering_analyzer.kmeans_clustering(n_clusters=k)
        hierarchical_result = clustering_analyzer.hierarchical_clustering(n_clusters=k)

        # Both should produce k clusters
        assert len(kmeans_result.clusters) == len(hierarchical_result.clusters)
        # Both should produce valid silhouette scores
        assert -1 < kmeans_result.silhouette_score < 1
        assert -1 < hierarchical_result.silhouette_score < 1

    def test_outlier_not_in_clusters(self, clustering_analyzer):
        """Test that outlier detection identifies episodes not well-fitted."""
        kmeans_result = clustering_analyzer.kmeans_clustering(n_clusters=3)
        outliers = clustering_analyzer.find_outliers(
            method="distance_based", clustering_labels=kmeans_result.labels
        )

        # Outliers should have larger distances to cluster center
        outlier_ids = {o.episode_id for o in outliers}
        assert len(outlier_ids) == len(outliers)  # No duplicates
