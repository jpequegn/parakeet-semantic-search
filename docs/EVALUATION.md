# Quality Metrics & Evaluation Framework

**Phase 3.4 Deliverable** - Comprehensive quality evaluation framework for semantic search results

## Overview

The evaluation framework provides comprehensive metrics to assess the quality, relevance, coverage, and diversity of semantic search results. It enables data-driven analysis of search performance and supports continuous improvement of the search system.

## Metrics Categories

### 1. Relevance Metrics

Measure how well search results match user intent and information needs.

#### Relevance Score
- **Scale**: 0-1 (0 = no relevant results, 1 = all results highly relevant)
- **Calculation**: Average of normalized relevance judgments (0-3 scale)
- **Use Case**: Quick assessment of overall result quality
- **Implementation**: `RelevanceEvaluator.evaluate_relevance()`

```python
relevance = evaluator.evaluate_relevance(
    query="machine learning",
    results=search_results,
    relevance_judgments={"ep_001": 3, "ep_002": 2, ...}
)
```

#### Precision@K
- **Scale**: 0-1 (fraction of top-k results that are relevant)
- **Common Values**: P@5, P@10, P@20
- **Threshold**: Configurable minimum relevance score
- **Use Case**: Assess result quality in top positions
- **Implementation**: `RelevanceEvaluator.precision_at_k()`

```python
# What fraction of top-10 results are relevant?
p10 = evaluator.precision_at_k(results, judgments, k=10, threshold=2)
```

**Interpretation**:
- 0.8: 8 out of 10 top results are relevant
- 0.5: 5 out of 10 top results are relevant
- Good target: >0.6 for top-10 results

#### Recall@K
- **Scale**: 0-1 (fraction of all relevant items appearing in top-k)
- **Common Values**: R@5, R@10, R@20
- **Use Case**: Measure coverage of relevant items
- **Implementation**: `RelevanceEvaluator.recall_at_k()`

```python
# What fraction of all relevant items appear in top-10?
r10 = evaluator.recall_at_k(results, judgments, k=10, threshold=2)
```

**Interpretation**:
- 1.0: All relevant items in top-10
- 0.5: Half of relevant items in top-10
- Trade-off: Precision vs Recall (usually inverse)

#### NDCG (Normalized Discounted Cumulative Gain)
- **Scale**: 0-1 (ranking quality measure)
- **Formula**: DCG / IDCG (position-weighted relevance)
- **Key Feature**: Rewards relevant items in early positions
- **Use Case**: Comprehensive ranking quality assessment
- **Implementation**: `RelevanceEvaluator.ndcg()`

```python
# How good is the ranking order?
ndcg = evaluator.ndcg(results, judgments, k=10)
```

**Interpretation**:
- 1.0: Perfect ranking (all relevant items first)
- 0.8+: Excellent ranking
- 0.6-0.8: Good ranking
- <0.6: Needs improvement

**Position Weighting**:
```
Rank 1: weight = log2(2) ≈ 1.0
Rank 2: weight = log2(3) ≈ 1.58
Rank 5: weight = log2(6) ≈ 2.58
Rank 10: weight = log2(11) ≈ 3.46
```

#### Mean Reciprocal Rank (MRR)
- **Scale**: 0-1 (1/rank of first relevant item)
- **Formula**: 1 / (rank of first relevant result)
- **Use Case**: Speed of finding relevant information
- **Implementation**: `RelevanceEvaluator.mean_reciprocal_rank()`

```python
# How quickly do we find a relevant result?
mrr = evaluator.mean_reciprocal_rank(results, judgments, threshold=2)
```

**Interpretation**:
- 1.0: First result is relevant
- 0.5: Second result is relevant
- 0.33: Third result is relevant
- 0.0: No relevant results

**Target**: >0.7 (relevant result in top 2-3 positions)

### 2. Coverage Metrics

Measure how comprehensively results span the content space.

#### Topic Coverage
- **Scale**: 0-1 (fraction of unique topics covered)
- **Calculation**: Unique topics in results / Total unique topics
- **Use Case**: Assess breadth of result diversity
- **Implementation**: `CoverageEvaluator.topic_coverage()`

```python
coverage = evaluator.topic_coverage(
    results=search_results,
    topic_assignments={
        "ep_001": {"ML", "supervised"},
        "ep_002": {"ML", "deep learning", "CNN"},
        ...
    }
)
```

**Interpretation**:
- 0.8: Results cover 80% of topic space
- 0.5: Results cover 50% of topic space
- High coverage indicates diverse results

#### Podcast Coverage
- **Scale**: 0-1 (fraction of unique podcasts in results)
- **Calculation**: Unique podcasts / Number of results
- **Use Case**: Assess source diversity
- **Implementation**: `CoverageEvaluator.podcast_coverage()`

```python
coverage = evaluator.podcast_coverage(results)
# Returns 0.75 if 3 unique podcasts in 4 results
```

**Interpretation**:
- 1.0: One result per podcast (maximum diversity)
- 0.5: Two results per podcast
- 0.1: Many results from same podcast (low diversity)

#### Temporal Coverage
- **Scale**: 0-1 (temporal distribution quality)
- **Returns**: Coverage score + distribution statistics
- **Use Case**: Assess recency and temporal spread
- **Implementation**: `CoverageEvaluator.temporal_coverage()`

```python
coverage, stats = evaluator.temporal_coverage(results)
# stats: {min_date, max_date, unique_dates, temporal_uniformity}
```

### 3. Diversity Metrics

Measure variety and dissimilarity of results.

#### Content Diversity
- **Scale**: 0-1 (heuristic diversity based on podcasts and dates)
- **Calculation**: Average of podcast and temporal diversity
- **Use Case**: Quick assessment of result variety
- **Implementation**: `DiversityEvaluator.content_diversity()`

```python
diversity = evaluator.content_diversity(results)
```

#### Semantic Diversity
- **Scale**: 0-1 (embedding-based dissimilarity)
- **Calculation**: Average pairwise cosine distance
- **Use Case**: Assess semantic difference in results
- **Implementation**: `DiversityEvaluator.semantic_diversity()`

```python
diversity = evaluator.semantic_diversity(
    results=search_results,
    embeddings=embedding_vectors  # Shape: (n_results, 384)
)
```

**Interpretation**:
- 0.9+: Very different semantic content
- 0.6-0.8: Diverse content
- 0.3-0.6: Somewhat similar content
- <0.3: Very similar content

**Calculation Detail**:
```python
# Normalized embeddings
norm_embeddings = embeddings / ||embeddings||

# Cosine similarity matrix
similarities = norm_embeddings @ norm_embeddings.T

# Distance = 1 - similarity
distances = 1 - similarities

# Diversity = average pairwise distance
```

#### Result Uniqueness
- **Scale**: 0-1 (fraction of unique episodes)
- **Calculation**: Unique episodes / Total results
- **Use Case**: Detect duplicate results
- **Implementation**: `DiversityEvaluator.result_uniqueness()`

```python
uniqueness = evaluator.result_uniqueness(results)
# 1.0 = no duplicates, 0.8 = 20% duplicates
```

## Complete Evaluation Workflow

### Single Query Evaluation

```python
from parakeet_search import EvaluationFramework
from tests.evaluation_dataset import (
    get_relevance_judgments,
    get_topic_assignments,
    get_sample_results,
)

# Initialize framework
framework = EvaluationFramework()

# Prepare evaluation data
query = "machine learning"
results = search_engine.search(query, limit=10)
relevance_judgments = get_relevance_judgments(query)
topic_assignments = get_topic_assignments()
embeddings = np.array([result["embedding"] for result in results])

# Evaluate
metrics = framework.evaluate_search_results(
    query=query,
    results=results,
    relevance_judgments=relevance_judgments,
    topic_assignments=topic_assignments,
    embeddings=embeddings,
)

# Access metrics
print(f"Relevance: {metrics.relevance_score:.3f}")
print(f"NDCG@10: {metrics.ndcg:.3f}")
print(f"Diversity: {metrics.diversity_score:.3f}")
print(f"Coverage: {metrics.coverage_score:.3f}")
```

### Batch Evaluation

```python
# Evaluate multiple queries
queries = [
    "machine learning",
    "deep learning",
    "computer vision",
]

all_metrics = []
for query in queries:
    results = search_engine.search(query)
    judgments = get_relevance_judgments(query)

    metrics = framework.evaluate_search_results(
        query=query,
        results=results,
        relevance_judgments=judgments,
    )
    all_metrics.append(metrics)

# Aggregate results
aggregated = EvaluationFramework.aggregate_metrics(all_metrics)

print(f"Avg NDCG: {aggregated['avg_ndcg']:.3f}")
print(f"Avg Precision: {aggregated['avg_precision']:.3f}")
print(f"Avg Recall: {aggregated['avg_recall']:.3f}")
```

## Interpretation Guide

### Result Quality Levels

| NDCG | Precision@10 | Interpretation |
|------|-------------|---|
| 0.9+ | 0.8+ | Excellent - Release ready |
| 0.8-0.9 | 0.6-0.8 | Good - Minor improvements |
| 0.7-0.8 | 0.5-0.6 | Fair - Needs improvement |
| <0.7 | <0.5 | Poor - Significant work needed |

### Diversity Recommendations

| Metric | Target | Status |
|--------|--------|--------|
| Semantic Diversity | >0.6 | ✓ Good |
| Podcast Coverage | >0.5 | ✓ Good |
| Topic Coverage | >0.7 | ✓ Good |
| Result Uniqueness | 1.0 | ✓ Excellent |

## Human Relevance Judgments

The framework uses a 4-point relevance scale:

| Score | Definition | Example |
|-------|-----------|---------|
| 3 | Highly Relevant | Episode directly answers query |
| 2 | Relevant | Episode contains useful information |
| 1 | Slightly Relevant | Episode mentions topic tangentially |
| 0 | Not Relevant | Episode has no relation to query |

### Creating Judgments

```python
relevance_judgments = {
    "ep_001": 3,  # Highly relevant
    "ep_002": 2,  # Relevant
    "ep_003": 1,  # Slightly relevant
    "ep_004": 0,  # Not relevant
}
```

## Evaluation Dataset

Pre-built evaluation dataset in `tests/evaluation_dataset.py`:

- **10 Sample Episodes**: Covering AI/ML topics
- **5 Search Queries**: machine learning, deep learning, computer vision, NLP, reinforcement learning
- **~50 Relevance Judgments**: Human-evaluated for each query
- **Topic Assignments**: ~60 topic tags across episodes

### Using the Dataset

```python
from tests.evaluation_dataset import (
    get_relevance_judgments,
    get_topic_assignments,
    get_sample_results,
)

judgments = get_relevance_judgments("machine learning")
assignments = get_topic_assignments()
results = get_sample_results("machine learning")
```

## Test Coverage

**32 Tests** covering:

✓ Relevance metrics (8 tests)
✓ Coverage metrics (6 tests)
✓ Diversity metrics (7 tests)
✓ Framework integration (5 tests)
✓ Dataset validation (4 tests)
✓ End-to-end evaluation (2 tests)

Run tests:
```bash
python3 -m pytest tests/test_evaluation.py -v
```

## Best Practices

### 1. Regular Evaluation
- Evaluate search quality weekly
- Track metrics trends over time
- Set target thresholds for each metric

### 2. Human Judgments
- Include diverse raters
- Use multiple queries per domain
- Document judgment criteria
- Maintain consistency across raters

### 3. Interpretation
- Don't optimize single metrics
- Consider trade-offs (precision vs recall)
- Use context-appropriate thresholds
- Validate with user feedback

### 4. Performance Monitoring
- Baseline metrics on production
- Alert on significant drops
- A/B test improvements
- Document changes and impacts

## Metrics Quick Reference

| Metric | Range | Higher | Lower | Use Case |
|--------|-------|--------|-------|----------|
| Relevance | 0-1 | Better | Worse | Overall quality |
| Precision@K | 0-1 | Better | Worse | Top-k quality |
| Recall@K | 0-1 | Better | Worse | Item coverage |
| NDCG | 0-1 | Better | Worse | Ranking quality |
| MRR | 0-1 | Better | Worse | Speed to relevant |
| Coverage | 0-1 | Better | Worse | Breadth of results |
| Diversity | 0-1 | Better | Worse | Result variety |

## API Reference

### EvaluationFramework

```python
class EvaluationFramework:
    def evaluate_search_results(
        query: str,
        results: List[Dict],
        relevance_judgments: Optional[Dict[str, int]] = None,
        topic_assignments: Optional[Dict[str, Set[str]]] = None,
        embeddings: Optional[np.ndarray] = None,
    ) -> EvaluationMetrics

    @staticmethod
    def aggregate_metrics(
        metrics_list: List[EvaluationMetrics],
    ) -> Dict[str, float]
```

### Individual Evaluators

```python
class RelevanceEvaluator:
    - evaluate_relevance()
    - precision_at_k()
    - recall_at_k()
    - ndcg()
    - mean_reciprocal_rank()

class CoverageEvaluator:
    - topic_coverage()
    - podcast_coverage()
    - temporal_coverage()

class DiversityEvaluator:
    - content_diversity()
    - semantic_diversity()
    - result_uniqueness()
```

## Related Documentation

- **Search Engine**: `docs/SEARCH.md`
- **Data Models**: `src/parakeet_search/models.py`
- **Test Fixtures**: `tests/fixtures.py`
- **Evaluation Dataset**: `tests/evaluation_dataset.py`

## Future Enhancements

- [ ] Automated relevance judgment using LLMs
- [ ] User engagement metrics (clicks, dwell time)
- [ ] A/B testing framework for algorithm comparison
- [ ] Metric visualization dashboard
- [ ] Performance profiling and optimization
- [ ] Query-specific metric targets
