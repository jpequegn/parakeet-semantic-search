"""Clustering and topic analysis for podcast embeddings."""

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


@dataclass
class ClusterInfo:
    """Information about a cluster."""

    cluster_id: int
    size: int
    center: np.ndarray
    episodes: List[str]
    podcasts: List[str]
    silhouette_score: float
    intra_distance_mean: float
    intra_distance_std: float


@dataclass
class ClusteringResult:
    """Results from clustering analysis."""

    algorithm: str
    n_clusters: int
    labels: np.ndarray
    cluster_centers: Optional[np.ndarray]
    silhouette_score: float
    inertia: Optional[float]
    clusters: List[ClusterInfo]


@dataclass
class OutlierInfo:
    """Information about an outlier episode."""

    episode_id: str
    episode_title: str
    outlier_score: float
    distance_to_nearest_cluster: float
    nearest_cluster_id: int


class ClusteringAnalyzer:
    """Analyze podcast embeddings using clustering techniques."""

    def __init__(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        """Initialize analyzer with embeddings and metadata.

        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            metadata: DataFrame with episode metadata (must include 'episode_id', 'episode_title', 'podcast_id')
        """
        self.embeddings = embeddings
        self.metadata = metadata
        self.n_samples = len(embeddings)
        self.embedding_dim = embeddings.shape[1]

        # Standardize embeddings for clustering
        self.scaler = StandardScaler()
        self.embeddings_scaled = self.scaler.fit_transform(embeddings)

        # Cache results
        self._kmeans_result: Optional[ClusteringResult] = None
        self._hierarchical_result: Optional[ClusteringResult] = None
        self._outliers: Optional[List[OutlierInfo]] = None

    def kmeans_clustering(self, n_clusters: int, random_state: int = 42) -> ClusteringResult:
        """Perform K-Means clustering.

        Args:
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility

        Returns:
            ClusteringResult with clustering information
        """
        # Fit K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(self.embeddings_scaled)

        # Calculate silhouette score
        from sklearn.metrics import silhouette_score

        silhouette = silhouette_score(self.embeddings_scaled, labels)

        # Build cluster info
        clusters = self._build_cluster_info(labels, kmeans.cluster_centers_)

        result = ClusteringResult(
            algorithm="K-Means",
            n_clusters=n_clusters,
            labels=labels,
            cluster_centers=kmeans.cluster_centers_,
            silhouette_score=silhouette,
            inertia=kmeans.inertia_,
            clusters=clusters,
        )

        self._kmeans_result = result
        return result

    def hierarchical_clustering(self, n_clusters: int, linkage_method: str = "ward") -> ClusteringResult:
        """Perform Hierarchical clustering.

        Args:
            n_clusters: Number of clusters
            linkage_method: Method for calculating distances ('ward', 'complete', 'average', 'single')

        Returns:
            ClusteringResult with clustering information
        """
        # Fit Hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        labels = hierarchical.fit_predict(self.embeddings_scaled)

        # Calculate silhouette score
        from sklearn.metrics import silhouette_score

        silhouette = silhouette_score(self.embeddings_scaled, labels)

        # Compute cluster centers as mean of samples in each cluster
        cluster_centers = np.array([self.embeddings_scaled[labels == i].mean(axis=0) for i in range(n_clusters)])

        # Build cluster info
        clusters = self._build_cluster_info(labels, cluster_centers)

        result = ClusteringResult(
            algorithm="Hierarchical",
            n_clusters=n_clusters,
            labels=labels,
            cluster_centers=cluster_centers,
            silhouette_score=silhouette,
            inertia=None,
            clusters=clusters,
        )

        self._hierarchical_result = result
        return result

    def find_outliers(
        self,
        method: str = "isolation_forest",
        contamination: float = 0.1,
        clustering_labels: Optional[np.ndarray] = None,
    ) -> List[OutlierInfo]:
        """Identify outlier episodes.

        Args:
            method: 'isolation_forest' or 'distance_based'
            contamination: Expected fraction of outliers
            clustering_labels: Cluster labels from previous clustering (for distance-based method)

        Returns:
            List of OutlierInfo objects
        """
        if method == "isolation_forest":
            outliers = self._outliers_isolation_forest(contamination)
        elif method == "distance_based":
            if clustering_labels is None:
                raise ValueError("clustering_labels required for distance_based method")
            outliers = self._outliers_distance_based(clustering_labels)
        else:
            raise ValueError(f"Unknown outlier method: {method}")

        self._outliers = outliers
        return outliers

    def _outliers_isolation_forest(self, contamination: float) -> List[OutlierInfo]:
        """Detect outliers using Isolation Forest."""
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(self.embeddings_scaled)

        outliers = []
        for idx in np.where(outlier_labels == -1)[0]:
            outlier = OutlierInfo(
                episode_id=self.metadata.iloc[idx]["episode_id"],
                episode_title=self.metadata.iloc[idx].get("episode_title", "Unknown"),
                outlier_score=iso_forest.score_samples(self.embeddings_scaled[idx : idx + 1])[0],
                distance_to_nearest_cluster=0.0,
                nearest_cluster_id=-1,
            )
            outliers.append(outlier)

        return sorted(outliers, key=lambda x: x.outlier_score)

    def _outliers_distance_based(self, clustering_labels: np.ndarray) -> List[OutlierInfo]:
        """Detect outliers based on distance to cluster center."""
        cluster_centers = np.array(
            [self.embeddings_scaled[clustering_labels == i].mean(axis=0) for i in np.unique(clustering_labels)]
        )

        outliers = []
        threshold_percentile = 90  # Top 10% as potential outliers

        for idx in range(len(self.embeddings_scaled)):
            distances = np.linalg.norm(cluster_centers - self.embeddings_scaled[idx], axis=1)
            min_distance = distances.min()
            nearest_cluster = np.argmin(distances)

            if min_distance > np.percentile(
                np.linalg.norm(cluster_centers[clustering_labels[idx]] - self.embeddings_scaled, axis=1),
                threshold_percentile,
            ):
                outlier = OutlierInfo(
                    episode_id=self.metadata.iloc[idx]["episode_id"],
                    episode_title=self.metadata.iloc[idx].get("episode_title", "Unknown"),
                    outlier_score=min_distance,
                    distance_to_nearest_cluster=min_distance,
                    nearest_cluster_id=int(nearest_cluster),
                )
                outliers.append(outlier)

        return sorted(outliers, key=lambda x: x.outlier_score, reverse=True)

    def _build_cluster_info(self, labels: np.ndarray, cluster_centers: np.ndarray) -> List[ClusterInfo]:
        """Build ClusterInfo for all clusters."""
        clusters = []

        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_embeddings = self.embeddings_scaled[mask]
            cluster_metadata = self.metadata[mask]

            # Calculate intra-cluster distances
            if len(cluster_embeddings) > 1:
                distances = pdist(cluster_embeddings, metric="euclidean")
                intra_distance_mean = distances.mean()
                intra_distance_std = distances.std()
            else:
                intra_distance_mean = 0.0
                intra_distance_std = 0.0

            # Calculate silhouette score for this cluster
            from sklearn.metrics import silhouette_samples

            all_silhouettes = silhouette_samples(self.embeddings_scaled, labels)
            cluster_silhouette = all_silhouettes[mask].mean()

            cluster_info = ClusterInfo(
                cluster_id=int(cluster_id),
                size=int(mask.sum()),
                center=cluster_centers[cluster_id],
                episodes=cluster_metadata["episode_id"].tolist(),
                podcasts=cluster_metadata["podcast_id"].tolist(),
                silhouette_score=cluster_silhouette,
                intra_distance_mean=intra_distance_mean,
                intra_distance_std=intra_distance_std,
            )
            clusters.append(cluster_info)

        return sorted(clusters, key=lambda x: x.size, reverse=True)

    def get_cluster_statistics(self, clustering_result: ClusteringResult) -> pd.DataFrame:
        """Get detailed statistics for each cluster.

        Args:
            clustering_result: Result from clustering

        Returns:
            DataFrame with cluster statistics
        """
        stats = []

        for cluster in clustering_result.clusters:
            stats.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "size": cluster.size,
                    "percentage": 100 * cluster.size / self.n_samples,
                    "silhouette_score": cluster.silhouette_score,
                    "intra_distance_mean": cluster.intra_distance_mean,
                    "intra_distance_std": cluster.intra_distance_std,
                    "unique_podcasts": len(set(cluster.podcasts)),
                    "dominant_podcast": max(
                        set(cluster.podcasts), key=cluster.podcasts.count, default="Unknown"
                    ),
                }
            )

        return pd.DataFrame(stats)

    def summarize_cluster(self, cluster_info: ClusterInfo) -> Dict:
        """Generate summary for a cluster.

        Args:
            cluster_info: Information about the cluster

        Returns:
            Dictionary with cluster summary
        """
        # Get podcast distribution
        podcast_counts = {}
        for podcast_id in cluster_info.podcasts:
            podcast_counts[podcast_id] = podcast_counts.get(podcast_id, 0) + 1

        # Get top episodes (closest to cluster center)
        cluster_mask = self.metadata["episode_id"].isin(cluster_info.episodes)
        cluster_embeddings = self.embeddings_scaled[cluster_mask]
        cluster_metadata = self.metadata[cluster_mask]

        distances = np.linalg.norm(cluster_embeddings - cluster_info.center, axis=1)
        top_indices = np.argsort(distances)[:5]

        top_episodes = [
            {
                "episode_id": cluster_metadata.iloc[idx]["episode_id"],
                "episode_title": cluster_metadata.iloc[idx].get("episode_title", "Unknown"),
                "distance_to_center": float(distances[idx]),
            }
            for idx in top_indices
        ]

        return {
            "cluster_id": cluster_info.cluster_id,
            "size": cluster_info.size,
            "percentage": 100 * cluster_info.size / self.n_samples,
            "silhouette_score": cluster_info.silhouette_score,
            "podcast_distribution": podcast_counts,
            "dominant_podcast": max(podcast_counts, key=podcast_counts.get, default="Unknown"),
            "unique_podcasts": len(podcast_counts),
            "intra_distance_mean": cluster_info.intra_distance_mean,
            "top_episodes": top_episodes,
        }

    def optimal_k_analysis(self, k_range: range = range(2, 11)) -> Tuple[List[float], List[float]]:
        """Analyze optimal number of clusters using elbow method.

        Args:
            k_range: Range of k values to test

        Returns:
            Tuple of (inertias, silhouette_scores)
        """
        inertias = []
        silhouette_scores = []

        from sklearn.metrics import silhouette_score

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.embeddings_scaled, labels))

        return inertias, silhouette_scores
