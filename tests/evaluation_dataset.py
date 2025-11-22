"""Evaluation dataset with human relevance judgments for quality metrics."""

from typing import Dict, Set


# Human relevance judgments for common search queries
# Relevance scale: 0=Not relevant, 1=Slightly relevant, 2=Relevant, 3=Highly relevant
RELEVANCE_JUDGMENTS = {
    "machine learning": {
        "ep_001": 3,  # Introduction to Machine Learning (highly relevant)
        "ep_002": 3,  # Deep Learning and Neural Networks (highly relevant)
        "ep_003": 2,  # NLP Advances (relevant - uses ML)
        "ep_004": 2,  # Computer Vision Applications (relevant - uses ML)
        "ep_005": 3,  # Building Production ML Systems (highly relevant)
        "ep_006": 2,  # Feature Engineering (relevant - part of ML)
        "ep_007": 2,  # Transfer Learning (relevant - part of ML)
        "ep_008": 2,  # Reinforcement Learning (relevant - type of ML)
        "ep_009": 2,  # Model Evaluation (relevant - ML evaluation)
        "ep_010": 1,  # Ethics in AI (slightly relevant)
    },
    "deep learning": {
        "ep_001": 1,  # Introduction to ML (mentions it)
        "ep_002": 3,  # Deep Learning and Neural Networks (highly relevant)
        "ep_003": 2,  # NLP (uses deep learning)
        "ep_004": 2,  # Computer Vision (uses deep learning)
        "ep_005": 2,  # Production ML (includes deep learning)
        "ep_006": 0,  # Feature Engineering (not about deep learning)
        "ep_007": 3,  # Transfer Learning (deep learning focused)
        "ep_008": 1,  # RL (different paradigm)
        "ep_009": 1,  # Model Evaluation (mentions deep learning)
        "ep_010": 0,  # Ethics in AI (not about deep learning)
    },
    "natural language processing": {
        "ep_001": 0,  # Introduction to ML (general)
        "ep_002": 1,  # Deep Learning (mentions it)
        "ep_003": 3,  # NLP Advances (highly relevant)
        "ep_004": 0,  # Computer Vision (different domain)
        "ep_005": 1,  # Production ML (might include NLP)
        "ep_006": 0,  # Feature Engineering (general)
        "ep_007": 1,  # Transfer Learning (mentions NLP apps)
        "ep_008": 0,  # Reinforcement Learning (different)
        "ep_009": 0,  # Model Evaluation (general)
        "ep_010": 0,  # Ethics in AI (general)
    },
    "computer vision": {
        "ep_001": 0,  # Introduction to ML
        "ep_002": 1,  # Deep Learning (mentions CNNs for vision)
        "ep_003": 0,  # NLP (different)
        "ep_004": 3,  # Computer Vision Applications (highly relevant)
        "ep_005": 1,  # Production ML (might include vision)
        "ep_006": 0,  # Feature Engineering (general)
        "ep_007": 1,  # Transfer Learning (mentions vision)
        "ep_008": 0,  # Reinforcement Learning (different)
        "ep_009": 0,  # Model Evaluation (general)
        "ep_010": 0,  # Ethics in AI
    },
    "neural networks": {
        "ep_001": 1,  # Mentions neural networks
        "ep_002": 3,  # Deep Learning - neural networks (highly relevant)
        "ep_003": 2,  # NLP - transformers are neural networks
        "ep_004": 2,  # Computer Vision - CNNs are neural networks
        "ep_005": 1,  # Production ML (mentions neural networks)
        "ep_006": 0,  # Feature Engineering (not neural networks)
        "ep_007": 2,  # Transfer Learning - mentions neural networks
        "ep_008": 1,  # RL - uses neural networks
        "ep_009": 0,  # Model Evaluation (general)
        "ep_010": 0,  # Ethics in AI
    },
}

# Topic assignments for coverage evaluation
# Maps episode_id to set of topics covered
TOPIC_ASSIGNMENTS = {
    "ep_001": {"machine learning", "supervised learning", "unsupervised learning"},
    "ep_002": {
        "machine learning",
        "deep learning",
        "neural networks",
        "CNN",
        "RNN",
        "transformers",
    },
    "ep_003": {"machine learning", "NLP", "transformers", "BERT", "GPT"},
    "ep_004": {"machine learning", "computer vision", "CNN", "object detection"},
    "ep_005": {"machine learning", "MLOps", "data pipelines", "model deployment"},
    "ep_006": {"machine learning", "feature engineering", "data preprocessing"},
    "ep_007": {"machine learning", "deep learning", "transfer learning", "fine-tuning"},
    "ep_008": {"machine learning", "reinforcement learning", "policy learning"},
    "ep_009": {"machine learning", "evaluation metrics", "model assessment"},
    "ep_010": {"AI", "ethics", "fairness", "bias", "transparency"},
}

# Sample search results with metadata for evaluation
SAMPLE_SEARCH_RESULTS = {
    "machine learning query": [
        {
            "episode_id": "ep_001",
            "podcast_id": "pod_001",
            "podcast_title": "AI Today Podcast",
            "episode_title": "Introduction to Machine Learning",
            "published_at": "2025-01-15",
            "_distance": 0.05,
        },
        {
            "episode_id": "ep_005",
            "podcast_id": "pod_003",
            "podcast_title": "Data Science Daily",
            "episode_title": "Building Production ML Systems",
            "published_at": "2025-01-10",
            "_distance": 0.12,
        },
        {
            "episode_id": "ep_002",
            "podcast_id": "pod_001",
            "podcast_title": "AI Today Podcast",
            "episode_title": "Deep Learning and Neural Networks",
            "published_at": "2025-01-08",
            "_distance": 0.18,
        },
        {
            "episode_id": "ep_006",
            "podcast_id": "pod_003",
            "podcast_title": "Data Science Daily",
            "episode_title": "Feature Engineering Techniques",
            "published_at": "2025-01-05",
            "_distance": 0.25,
        },
        {
            "episode_id": "ep_010",
            "podcast_id": "pod_001",
            "podcast_title": "AI Today Podcast",
            "episode_title": "Ethics in Artificial Intelligence",
            "published_at": "2024-12-28",
            "_distance": 0.65,
        },
    ],
    "deep learning query": [
        {
            "episode_id": "ep_002",
            "podcast_id": "pod_001",
            "podcast_title": "AI Today Podcast",
            "episode_title": "Deep Learning and Neural Networks",
            "published_at": "2025-01-08",
            "_distance": 0.03,
        },
        {
            "episode_id": "ep_007",
            "podcast_id": "pod_001",
            "podcast_title": "AI Today Podcast",
            "episode_title": "Transfer Learning and Fine-tuning",
            "published_at": "2024-12-20",
            "_distance": 0.15,
        },
        {
            "episode_id": "ep_004",
            "podcast_id": "pod_002",
            "podcast_title": "Tech Trends Weekly",
            "episode_title": "Computer Vision Applications",
            "published_at": "2025-01-12",
            "_distance": 0.22,
        },
        {
            "episode_id": "ep_003",
            "podcast_id": "pod_002",
            "podcast_title": "Tech Trends Weekly",
            "episode_title": "Natural Language Processing Advances",
            "published_at": "2025-01-18",
            "_distance": 0.28,
        },
        {
            "episode_id": "ep_001",
            "podcast_id": "pod_001",
            "podcast_title": "AI Today Podcast",
            "episode_title": "Introduction to Machine Learning",
            "published_at": "2025-01-15",
            "_distance": 0.35,
        },
    ],
    "computer vision query": [
        {
            "episode_id": "ep_004",
            "podcast_id": "pod_002",
            "podcast_title": "Tech Trends Weekly",
            "episode_title": "Computer Vision Applications",
            "published_at": "2025-01-12",
            "_distance": 0.02,
        },
        {
            "episode_id": "ep_002",
            "podcast_id": "pod_001",
            "podcast_title": "AI Today Podcast",
            "episode_title": "Deep Learning and Neural Networks",
            "published_at": "2025-01-08",
            "_distance": 0.18,
        },
        {
            "episode_id": "ep_007",
            "podcast_id": "pod_001",
            "podcast_title": "AI Today Podcast",
            "episode_title": "Transfer Learning and Fine-tuning",
            "published_at": "2024-12-20",
            "_distance": 0.32,
        },
        {
            "episode_id": "ep_005",
            "podcast_id": "pod_003",
            "podcast_title": "Data Science Daily",
            "episode_title": "Building Production ML Systems",
            "published_at": "2025-01-10",
            "_distance": 0.45,
        },
        {
            "episode_id": "ep_009",
            "podcast_id": "pod_003",
            "podcast_title": "Data Science Daily",
            "episode_title": "Model Evaluation and Metrics",
            "published_at": "2025-01-02",
            "_distance": 0.58,
        },
    ],
    "reinforcement learning query": [
        {
            "episode_id": "ep_008",
            "podcast_id": "pod_002",
            "podcast_title": "Tech Trends Weekly",
            "episode_title": "Reinforcement Learning Fundamentals",
            "published_at": "2025-01-17",
            "_distance": 0.04,
        },
        {
            "episode_id": "ep_002",
            "podcast_id": "pod_001",
            "podcast_title": "AI Today Podcast",
            "episode_title": "Deep Learning and Neural Networks",
            "published_at": "2025-01-08",
            "_distance": 0.35,
        },
        {
            "episode_id": "ep_001",
            "podcast_id": "pod_001",
            "podcast_title": "AI Today Podcast",
            "episode_title": "Introduction to Machine Learning",
            "published_at": "2025-01-15",
            "_distance": 0.42,
        },
        {
            "episode_id": "ep_005",
            "podcast_id": "pod_003",
            "podcast_title": "Data Science Daily",
            "episode_title": "Building Production ML Systems",
            "published_at": "2025-01-10",
            "_distance": 0.51,
        },
        {
            "episode_id": "ep_003",
            "podcast_id": "pod_002",
            "podcast_title": "Tech Trends Weekly",
            "episode_title": "Natural Language Processing Advances",
            "published_at": "2025-01-18",
            "_distance": 0.72,
        },
    ],
}


def get_relevance_judgments(query: str) -> Dict[str, int]:
    """Get human relevance judgments for a query.

    Args:
        query: Search query

    Returns:
        Dict mapping episode_id to relevance score
    """
    # Normalize query to match our dataset
    query_lower = query.lower().strip()

    for key in RELEVANCE_JUDGMENTS.keys():
        if key in query_lower:
            return RELEVANCE_JUDGMENTS[key]

    # If no exact match, return empty
    return {}


def get_topic_assignments() -> Dict[str, Set[str]]:
    """Get topic assignments for all episodes.

    Returns:
        Dict mapping episode_id to set of topics
    """
    return TOPIC_ASSIGNMENTS.copy()


def get_sample_results(query: str) -> list:
    """Get sample search results for a query.

    Args:
        query: Search query

    Returns:
        List of search results
    """
    query_lower = query.lower().strip()

    for key in SAMPLE_SEARCH_RESULTS.keys():
        if key.replace(" query", "") in query_lower:
            return SAMPLE_SEARCH_RESULTS[key].copy()

    # Default to empty results
    return []
