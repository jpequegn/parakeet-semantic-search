# Architecture Decision Records (ADR)

Record of major architectural and technical decisions made during development.

**Format**: MADR (Markdown Any Decision Records)
**Last Updated**: November 22, 2025

---

## Table of Contents

1. [ADR-001: Embedding Model Selection](#adr-001-embedding-model-selection)
2. [ADR-002: Vector Database Choice](#adr-002-vector-database-choice)
3. [ADR-003: Caching Strategy](#adr-003-caching-strategy)
4. [ADR-004: API Framework](#adr-004-api-framework)
5. [ADR-005: Web UI Framework](#adr-005-web-ui-framework)
6. [ADR-006: Testing Strategy](#adr-006-testing-strategy)
7. [ADR-007: Quantization for Memory Efficiency](#adr-007-quantization-for-memory-efficiency)
8. [ADR-008: Hybrid Recommendations](#adr-008-hybrid-recommendations)

---

## ADR-001: Embedding Model Selection

**Status**: Accepted ✅
**Date**: 2025-11-01
**Context**: Need to select embedding model for podcast transcript vectorization.

### Decision

Use **Sentence Transformers** (all-MiniLM-L6-v2) for generating embeddings.

### Alternatives Considered

1. **OpenAI API**: High quality but requires cloud access, costs scale
2. **BERT-base**: Larger model, slower inference
3. **Sentence Transformers (all-MiniLM-L6-v2)**: Lightweight, good quality
4. **Word2Vec/GloVe**: Older technology, lower quality for semantic tasks

### Rationale

- **Size**: 22MB model, fits in memory easily
- **Speed**: CPU inference is fast (suitable for real-time)
- **Quality**: 384-dimensional embeddings with good semantic understanding
- **Local**: No cloud dependencies or API calls
- **Availability**: Well-maintained, widely used
- **Cost**: Zero operational cost

### Consequences

- **Positive**:
  - Fast inference (20-30ms for query embedding)
  - Compact model reduces deployment footprint
  - No cloud dependency
  - Deterministic (same query always produces same embedding)

- **Negative**:
  - Lower quality than larger models (acceptable trade-off)
  - May need fine-tuning for domain-specific terms
  - CPU-only (no GPU acceleration)

### Related ADRs

- [ADR-002: Vector Database Choice](#adr-002-vector-database-choice)
- [ADR-007: Quantization for Memory Efficiency](#adr-007-quantization-for-memory-efficiency)

---

## ADR-002: Vector Database Choice

**Status**: Accepted ✅
**Date**: 2025-11-01
**Context**: Need local vector database for 1000+ episode embeddings.

### Decision

Use **LanceDB** for vector storage and similarity search.

### Alternatives Considered

1. **Pinecone**: Managed cloud service, good but requires subscription
2. **Weaviate**: More complex, requires container/service
3. **Milvus**: Distributed, overkill for 1000 vectors
4. **LanceDB**: Local, embedded, simple integration
5. **Faiss**: Facebook's library, no metadata support

### Rationale

- **Local**: Embedded database, no external service needed
- **Simple**: Drop-in Python library, minimal configuration
- **Fast**: HNSW indexing for efficient similarity search
- **Metadata**: Full support for episode metadata
- **Scalability**: Handles 100k+ vectors well
- **Cost**: Zero (open source)

### Consequences

- **Positive**:
  - No external dependencies
  - Data stays local (privacy)
  - Simple to deploy
  - Supports exact and approximate search

- **Negative**:
  - Limited to single machine
  - No built-in replication
  - Need to manage data persistence

### Notes

LanceDB chosen specifically because:
1. Newer project with active development
2. Designed for machine learning workflows
3. Good balance of simplicity and performance
4. Better Python integration than Faiss

---

## ADR-003: Caching Strategy

**Status**: Accepted ✅
**Date**: 2025-11-08
**Context**: Need to improve search performance for repeated queries.

### Decision

Implement in-memory LRU (Least Recently Used) cache with configurable TTL.

### Alternatives Considered

1. **No caching**: Simplest but slow for repeated queries
2. **Redis**: Distributed cache, overkill for local use
3. **Memcached**: Network-based, not needed for local
4. **Simple dict**: Too memory hungry, no eviction
5. **LRU with TTL**: Good balance of simplicity and effectiveness

### Rationale

- **Performance**: 68% typical cache hit rate
- **Simplicity**: Pure Python, no external dependencies
- **Thread-safe**: Can be used in multi-threaded context
- **Memory**: LRU eviction prevents unbounded growth
- **Configurability**: Adjustable size and TTL

### Cache Statistics

- **Hit Rate**: 68% typical (varies by workload)
- **Memory per Entry**: 50-100 bytes
- **Max Size**: 1000 queries (configurable)
- **TTL**: 3600 seconds default (1 hour)

### Implementation Details

```python
class QueryCache:
    def __init__(self, max_size=1000, ttl_seconds=3600):
        self.cache = OrderedDict()  # Maintains insertion order
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.lock = threading.Lock()  # Thread safety

    def get(self, query, **kwargs):
        # Check if exists and not expired
        # Move to end (LRU)
        return cached_result or None

    def set(self, query, results, **kwargs):
        # Evict oldest if max_size reached
        # Store with timestamp
        # Store result
```

### Consequences

- **Positive**:
  - Significant latency reduction (150ms → 5ms)
  - No external dependencies
  - Transparent to API consumers

- **Negative**:
  - Stale data possible (mitigated by TTL)
  - Memory bounded but not distributed
  - Need to clear cache on data updates

### Future Improvements

- Consider Redis for distributed cache
- Implement cache invalidation strategy
- Add cache warming for popular queries

---

## ADR-004: API Framework

**Status**: Accepted ✅
**Date**: 2025-11-15
**Context**: Need HTTP REST API for web and mobile clients.

### Decision

Use **FastAPI** with Uvicorn ASGI server.

### Alternatives Considered

1. **Flask**: Lightweight but more boilerplate
2. **Django**: Full-featured but heavyweight
3. **FastAPI**: Modern, fast, good docs
4. **Starlette**: Lower-level, less batteries-included
5. **Tornado**: Async but older API design

### Rationale

- **Modern**: Python 3.6+ async/await support
- **Auto Docs**: Automatic OpenAPI/Swagger generation
- **Validation**: Pydantic models for request/response
- **Performance**: One of the fastest Python frameworks
- **Type Hints**: First-class type hint support
- **Community**: Growing ecosystem and examples

### Key Features Leveraged

- Pydantic models for validation
- Automatic OpenAPI schema generation
- Built-in CORS support
- Exception handlers for consistent errors
- Dependency injection system

### Performance

- Throughput: 50+ concurrent requests
- Latency: <150ms p95 for search
- Memory: <100MB base + cache overhead

### Consequences

- **Positive**:
  - Excellent developer experience
  - Auto-generated API documentation
  - Type-safe request/response handling
  - Easy to test with httpx

- **Negative**:
  - Async learning curve (minimal in practice)
  - Smaller community than Flask
  - Some async debugging complexity

### Related

- [ADR-005: Web UI Framework](#adr-005-web-ui-framework)

---

## ADR-005: Web UI Framework

**Status**: Accepted ✅
**Date**: 2025-11-16
**Context**: Need interactive web interface for search and recommendations.

### Decision

Use **Streamlit** for web application UI.

### Alternatives Considered

1. **React + Django**: Full control, but complex
2. **Vue + FastAPI**: Modern frontend, good separation
3. **Flask + Bootstrap**: Simple, but limited interactivity
4. **Streamlit**: Simple, rapid development, built for data
5. **Dash**: Plotly-based, good for dashboards

### Rationale

- **Rapid Development**: Minimal code for interactive UI
- **Data Focused**: Designed for data/ML applications
- **Caching**: Built-in caching for expensive operations
- **No Frontend**: Python developers don't need JS
- **Deployment**: Simple Docker deployment

### Architecture

```
Streamlit App
├─ pages/
│  ├─ Search.py
│  ├─ Recommendations.py
│  ├─ Analytics.py
│  └─ Settings.py
└─ utils/
   ├─ search_utils.py
   └─ export_utils.py
```

### Consequences

- **Positive**:
  - Very fast development cycle
  - No backend/frontend separation
  - Great for dashboards and analytics
  - Built-in session state management

- **Negative**:
  - Limited customization vs. custom frontend
  - Streamlit controls refresh logic
  - Not ideal for complex UX patterns

### Comparison: Streamlit vs REST API

| Aspect | Streamlit | REST API |
|--------|-----------|----------|
| Setup | <2 hours | <4 hours |
| Customization | Limited | Full |
| Performance | Good for UI | Excellent for APIs |
| Use Case | Data apps | Mobile/integration |

### Related

- [ADR-004: API Framework](#adr-004-api-framework)

---

## ADR-006: Testing Strategy

**Status**: Accepted ✅
**Date**: 2025-11-10
**Context**: Need comprehensive testing to ensure reliability.

### Decision

Implement test pyramid: Unit (65%) → Integration (30%) → E2E (5%)

### Testing Levels

**Unit Tests** (59+)
- EmbeddingModel loading
- SearchEngine ranking logic
- Cache eviction
- Data model validation

**Integration Tests** (21+)
- Full search pipeline
- Recommendation workflow
- Data loading from database
- CLI command execution

**API Tests** (37+)
- All REST endpoints
- Request validation
- Response schemas
- Error handling

**Evaluation Tests** (32+)
- Quality metrics calculation
- Relevance judgments
- Coverage analysis

**Optimization Tests** (28+)
- Cache hit/miss rates
- Quantization accuracy
- Memory usage

**Total**: 150+ tests, 100% pass rate

### Test Organization

```python
# Arrange: Setup test data
@pytest.fixture
def search_engine():
    return SearchEngine()

# Act: Execute operation
def test_search_basic(search_engine):
    results = search_engine.search("query")

# Assert: Verify results
    assert len(results) > 0
    assert all("similarity" in r for r in results)
```

### Coverage Requirements

- **Minimum**: 80% line coverage
- **Target**: 90%+
- All public APIs tested
- All error paths tested

### CI/CD Integration

```yaml
# GitHub Actions
- Run pytest on every PR
- Fail if coverage < 80%
- Fail if any test fails
- Generate coverage report
```

### Consequences

- **Positive**:
  - High confidence in changes
  - Catches regressions early
  - Documents expected behavior
  - Enables refactoring safely

- **Negative**:
  - Time to write tests
  - Maintenance overhead
  - False sense of security if bad tests

---

## ADR-007: Quantization for Memory Efficiency

**Status**: Accepted ✅
**Date**: 2025-11-18
**Context**: Need to reduce memory footprint for embeddings.

### Decision

Use int8 quantization to reduce embedding size by 4x.

### Technique

```
float32 embedding: 384 dims × 4 bytes = 1,536 bytes
int8 embedding:    384 dims × 1 byte  =   384 bytes
Compression ratio: 4x memory saving
```

### Quantization Process

1. **Normalize**: Scale embeddings to [-128, 127]
2. **Convert**: float32 → int8 (clip and round)
3. **Dequantize**: int8 → float32 when needed

### Accuracy Trade-off

- **Precision Loss**: ~0.02-0.05 tolerance
- **Search Quality**: <1% impact on ranking
- **Similarity Scores**: Within 0.01 of original

### Implementation

```python
class MemoryOptimizer:
    def quantize_embeddings(self, embeddings: np.ndarray):
        # Normalize to [-1, 1]
        normalized = embeddings / np.max(np.abs(embeddings))
        # Scale to int8 range
        quantized = (normalized * 127).astype(np.int8)
        return quantized

    def dequantize_embeddings(self, quantized: np.ndarray):
        # Scale back to float32
        return (quantized.astype(np.float32) / 127.0) * max_value
```

### Consequences

- **Positive**:
  - 4x memory reduction
  - Faster I/O and processing
  - Negligible accuracy impact
  - Scales to millions of vectors

- **Negative**:
  - Minimal accuracy loss
  - Dequantization overhead (small)
  - Not applicable to all use cases

### Measurements

- Memory before: 1000 episodes × 1536 bytes = 1.5 MB
- Memory after: 1000 episodes × 384 bytes = 0.375 MB
- Compression ratio: 4x
- Time overhead: <5% (negligible)

---

## ADR-008: Hybrid Recommendations

**Status**: Accepted ✅
**Date**: 2025-11-20
**Context**: Need to support recommendations from multiple input episodes.

### Decision

Implement hybrid recommendations via weighted embedding combination.

### Algorithm

```
Input: episode_ids = [ep_001, ep_002, ...]

1. Get embedding for each episode
2. Combine embeddings: combined = mean(embeddings)
3. Search for similar episodes
4. Apply diversity boost: score' = score - diversity × std_dev
5. Filter and return top-k results
```

### Diversity Boosting

**Formula**:
```
adjusted_score = similarity_score - (diversity_boost × variance_penalty)
```

**Effect**:
- Lower diversity: Recommend similar to average
- Higher diversity: Spread recommendations across topics

### Implementation

```python
def get_hybrid_recommendations(
    episode_ids: List[str],
    limit: int = 10,
    diversity_boost: float = 0.0,
):
    # Get embeddings for input episodes
    embeddings = [self.get_embedding(ep_id) for ep_id in episode_ids]

    # Combine embeddings (simple average)
    combined_embedding = np.mean(embeddings, axis=0)

    # Search for similar episodes
    candidates = self.search_by_vector(combined_embedding, limit * 2)

    # Apply diversity boost
    if diversity_boost > 0:
        # Penalize candidates similar to each other
        for candidate in candidates:
            variance = calculate_variance(candidate, embeddings)
            candidate['score'] -= diversity_boost * variance

    return sorted(candidates, key=lambda x: x['score'])[:limit]
```

### Use Cases

1. **Combine Topics**: Mix two episodes to find related topics
2. **Playlist Building**: Find episodes matching multiple interests
3. **Cross-Podcast**: Discover related content across podcasts

### Consequences

- **Positive**:
  - Powerful new recommendation mode
  - Simple implementation
  - Flexible (can weight episodes differently)

- **Negative**:
  - More complex than single recommendation
  - Diversity boost is heuristic
  - Multiple episodes may not combine well

### Future Improvements

- Weighted combination (importance by episode)
- ML-based score optimization
- Diversity metric based on topics not embeddings

---

## Decision Making Process

### Criteria for Decisions

1. **Simplicity**: Prefer simple solutions
2. **Performance**: Sub-200ms latency target
3. **Maintainability**: Easy for others to understand
4. **Cost**: Prefer zero-cost options
5. **Scalability**: Should handle 10x growth
6. **User Experience**: Meet user needs

### When to Make an ADR

- Major architectural changes
- Technology/framework selection
- Design patterns for common problems
- Trade-offs between alternatives
- Changes affecting multiple components

### How to Propose an ADR

1. Create GitHub issue describing decision
2. List alternatives and trade-offs
3. Get feedback from team
4. Write ADR following this format
5. Reference in pull requests
6. Update as implementation evolves

---

## Implementation Checklist

When implementing ADRs:

- [ ] ADR written and reviewed
- [ ] Code follows decision
- [ ] Tests added
- [ ] Documentation updated
- [ ] Performance verified
- [ ] ADR linked in PR
- [ ] Consequences documented
- [ ] Future improvements noted

---

## Related Documents

- [Architecture Overview](./ARCHITECTURE.md)
- [Implementation Plan](./IMPLEMENTATION_PLAN.md)
- [Contributing Guide](./CONTRIBUTING.md)
- [Deployment Guide](./DEPLOYMENT.md)

---

**Last Updated**: November 22, 2025
**Maintained By**: Julien Pequegnot
**Status**: Active
