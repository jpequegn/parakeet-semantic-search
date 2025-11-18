"""Test fixtures and sample data for integration tests."""

import pandas as pd
import numpy as np
import pytest


# Sample episodes for integration testing
SAMPLE_EPISODES = [
    {
        "id": 1,
        "episode_id": "ep_001",
        "podcast_id": "pod_001",
        "podcast_title": "AI Today Podcast",
        "episode_title": "Introduction to Machine Learning",
        "transcript": (
            "Machine learning is a subset of artificial intelligence that enables systems "
            "to learn and improve from experience without being explicitly programmed. "
            "In this episode, we explore the fundamentals of machine learning, including "
            "supervised learning, unsupervised learning, and reinforcement learning."
        ),
    },
    {
        "id": 2,
        "episode_id": "ep_002",
        "podcast_id": "pod_001",
        "podcast_title": "AI Today Podcast",
        "episode_title": "Deep Learning and Neural Networks",
        "transcript": (
            "Deep learning is a powerful subset of machine learning that uses neural networks "
            "with multiple layers to extract progressively higher-level features. "
            "We discuss convolutional neural networks, recurrent neural networks, and transformers "
            "in this comprehensive episode about deep learning architectures."
        ),
    },
    {
        "id": 3,
        "episode_id": "ep_003",
        "podcast_id": "pod_002",
        "podcast_title": "Tech Trends Weekly",
        "episode_title": "Natural Language Processing Advances",
        "transcript": (
            "Natural language processing enables computers to understand and generate human language. "
            "Recent advances in NLP using transformer models have led to breakthroughs in machine translation, "
            "sentiment analysis, and question answering systems. We explore BERT, GPT, and other state-of-the-art models."
        ),
    },
    {
        "id": 4,
        "episode_id": "ep_004",
        "podcast_id": "pod_002",
        "podcast_title": "Tech Trends Weekly",
        "episode_title": "Computer Vision Applications",
        "transcript": (
            "Computer vision is the field of artificial intelligence that trains computers to interpret "
            "and understand visual information from the world. Applications include object detection, "
            "image classification, facial recognition, and autonomous vehicles. We discuss how CNNs revolutionized vision tasks."
        ),
    },
    {
        "id": 5,
        "episode_id": "ep_005",
        "podcast_id": "pod_003",
        "podcast_title": "Data Science Daily",
        "episode_title": "Building Production ML Systems",
        "transcript": (
            "Deploying machine learning models to production involves many challenges beyond model training. "
            "We discuss data pipelines, model versioning, monitoring, and retraining strategies. "
            "Learn about MLOps best practices and tools for building reliable, scalable ML systems."
        ),
    },
    {
        "id": 6,
        "episode_id": "ep_006",
        "podcast_id": "pod_003",
        "podcast_title": "Data Science Daily",
        "episode_title": "Feature Engineering Techniques",
        "transcript": (
            "Feature engineering is the art and science of extracting meaningful features from raw data. "
            "Good features can make simple models work well, while poor features require complex models. "
            "We explore techniques like normalization, encoding, dimensionality reduction, and domain-specific feature creation."
        ),
    },
    {
        "id": 7,
        "episode_id": "ep_007",
        "podcast_id": "pod_001",
        "podcast_title": "AI Today Podcast",
        "episode_title": "Transfer Learning and Fine-tuning",
        "transcript": (
            "Transfer learning allows us to leverage knowledge from pre-trained models on large datasets "
            "and adapt them to new tasks with limited data. Fine-tuning pre-trained models has become "
            "a standard approach in modern deep learning, enabling faster training and better performance."
        ),
    },
    {
        "id": 8,
        "episode_id": "ep_008",
        "podcast_id": "pod_002",
        "podcast_title": "Tech Trends Weekly",
        "episode_title": "Reinforcement Learning Fundamentals",
        "transcript": (
            "Reinforcement learning is a paradigm where agents learn by interacting with an environment. "
            "The agent receives rewards for good actions and penalties for bad actions. Applications include "
            "game playing, robotics, autonomous driving, and recommendation systems using RL techniques."
        ),
    },
    {
        "id": 9,
        "episode_id": "ep_009",
        "podcast_id": "pod_003",
        "podcast_title": "Data Science Daily",
        "episode_title": "Model Evaluation and Metrics",
        "transcript": (
            "Choosing the right evaluation metrics is critical for assessing model performance. "
            "Different tasks require different metrics: classification uses accuracy, precision, recall; "
            "regression uses MSE, RMSE, R-squared. We discuss confusion matrices, ROC curves, and AUC."
        ),
    },
    {
        "id": 10,
        "episode_id": "ep_010",
        "podcast_id": "pod_001",
        "podcast_title": "AI Today Podcast",
        "episode_title": "Ethics in Artificial Intelligence",
        "transcript": (
            "As AI systems become more powerful and widespread, ethical considerations become increasingly important. "
            "Topics include bias in training data, fairness, transparency, accountability, and privacy. "
            "We discuss the importance of responsible AI development and deployment in society."
        ),
    },
]


@pytest.fixture
def sample_episodes():
    """Provide sample episodes for testing."""
    return SAMPLE_EPISODES


@pytest.fixture
def sample_dataframe():
    """Create a pandas DataFrame from sample episodes."""
    data = []
    for episode in SAMPLE_EPISODES:
        # In real usage, transcript would be chunked and embedded
        # For testing, we create mock embeddings
        data.append(
            {
                "id": episode["id"],
                "episode_id": episode["episode_id"],
                "podcast_id": episode["podcast_id"],
                "podcast_title": episode["podcast_title"],
                "episode_title": episode["episode_title"],
                "text": episode["transcript"],
                # Mock embedding (384-dimensional vector matching Sentence Transformers output)
                "embedding": np.random.randn(384),
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def search_queries():
    """Provide test queries for semantic search validation."""
    return [
        {
            "query": "machine learning fundamentals",
            "expected_top_episodes": ["ep_001", "ep_007"],
            "description": "Should find episodes about ML basics",
        },
        {
            "query": "deep neural networks",
            "expected_top_episodes": ["ep_002", "ep_003"],
            "description": "Should find episodes about deep learning and NLP",
        },
        {
            "query": "computer vision and object detection",
            "expected_top_episodes": ["ep_004"],
            "description": "Should find computer vision episode",
        },
        {
            "query": "production machine learning systems",
            "expected_top_episodes": ["ep_005"],
            "description": "Should find MLOps episode",
        },
        {
            "query": "reinforcement learning and game playing",
            "expected_top_episodes": ["ep_008"],
            "description": "Should find RL episode",
        },
        {
            "query": "model evaluation metrics and performance",
            "expected_top_episodes": ["ep_009"],
            "description": "Should find evaluation metrics episode",
        },
        {
            "query": "AI ethics and bias",
            "expected_top_episodes": ["ep_010"],
            "description": "Should find ethics episode",
        },
    ]


@pytest.fixture
def malformed_inputs():
    """Provide malformed inputs for error handling tests."""
    return [
        {"input": "", "description": "Empty string"},
        {"input": None, "description": "None value"},
        {"input": 12345, "description": "Integer instead of string"},
        {"input": [], "description": "Empty list"},
        {"input": "a" * 1000000, "description": "Very long string (1MB)"},
    ]
