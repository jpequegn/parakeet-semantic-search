# Architecture Documentation

Complete system design, component breakdown, and technical decisions for Parakeet Semantic Search.

**Version**: 1.0.0
**Last Updated**: November 22, 2025

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Parakeet Semantic Search                        │
│                      Production System v1.0                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  USER INTERFACES LAYER                                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │  CLI Interface   │  │  Streamlit WebUI │  │  REST API        │ │
│  │  (Click)         │  │  (Multi-page)    │  │  (FastAPI)       │ │
│  │  - Search        │  │  - Search        │  │  - JSON I/O      │ │
│  │  - Recommend     │  │  - Recommend     │  │  - Auth-ready    │ │
│  │  - Analytics     │  │  - Analytics     │  │  - OpenAPI docs  │ │
│  │  - Export        │  │  - Settings      │  │  - Monitoring    │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  APPLICATION LOGIC LAYER                                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌──────────────────────┐  ┌──────────────┐  │
│  │  SearchEngine   │  │ RecommendationEngine │  │  Analytics   │  │
│  │                 │  │                      │  │              │  │
│  │ - search()      │  │ - get_recommend()    │  │ - cluster()  │  │
│  │ - vectorize()   │  │ - hybrid_recommend() │  │ - evaluate() │  │
│  │ - rank()        │  │ - filter()           │  │ - profile()  │  │
│  └────────┬────────┘  └──────────┬───────────┘  └──────┬───────┘  │
│           │                       │                      │          │
│  ┌────────▼──────────────────────▼──────────────────────▼────────┐ │
│  │         Optimization & Caching Layer                          │ │
│  ├───────────────────────────────────────────────────────────────┤ │
│  │                                                               │ │
│  │  ┌─────────────────────┐  ┌──────────────────────────────┐  │ │
│  │  │  CachingSearch      │  │  MemoryOptimizer             │  │ │
│  │  │  Engine             │  │  - Quantization (int8)       │  │ │
│  │  │                     │  │  - Batch operations          │  │ │
│  │  │  - LRU Cache        │  │  - Compression               │  │ │
│  │  │  - TTL expiry       │  │  - Thread safety             │  │ │
│  │  │  - Stats tracking   │  │  - 4x memory savings         │  │ │
│  │  └─────────────────────┘  └──────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│  DATA LAYER                                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐  ┌──────────────────────┐  ┌───────────┐ │
│  │  VectorStore        │  │ EmbeddingModel       │  │  DataProc │ │
│  │  (LanceDB)          │  │ (Sent. Transform.)   │  │  Pipeline │ │
│  │                     │  │                      │  │           │ │
│  │ - 384-dim vecs      │  │ - all-MiniLM-L6-v2   │  │ - Chunk   │ │
│  │ - HNSW indexing     │  │ - 384 dimensions     │  │ - Embed   │ │
│  │ - 1000+ episodes    │  │ - GPU-optimized      │  │ - Store   │ │
│  │ - Fast search       │  │ - Low memory         │  │ - Export  │ │
│  └─────────────────────┘  └──────────────────────┘  └───────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Search & Recommendation Engine

#### SearchEngine (`src/parakeet_search/search.py`)
- **Purpose**: Core semantic search functionality
- **Responsibilities**:
  - Load embeddings from vector store
  - Generate query embeddings
  - Perform similarity search
  - Rank results by relevance
  - Filter by thresholds

**Key Methods**:
```python
search(query: str, limit: int, threshold: float) → List[Dict]
get_recommendations(episode_id: str, limit: int) → List[Dict]
get_hybrid_recommendations(episode_ids: List[str], ...) → List[Dict]
```

**Performance**:
- Search latency: 100-150ms
- Throughput: 50+ concurrent
- Memory per query: <1MB

---

#### RecommendationEngine (`src/parakeet_search/search.py`)
- **Purpose**: Episode recommendation logic
- **Modes**:
  - **Single**: Find similar episodes to one input
  - **Hybrid**: Combine multiple episodes for collective recommendations
  - **Filtered**: Restrict to specific podcasts
  - **Diverse**: Boost diversity with penalty term

**Algorithm**:
```
1. Get embedding for input episode(s)
2. Search for similar episodes in vector store
3. Apply diversity boost if requested
4. Filter by podcast if specified
5. Rank and return top-k results
```

---

### 2. Optimization & Performance

#### CachingSearchEngine (`src/parakeet_search/optimization.py`)
- **Purpose**: Transparent caching wrapper
- **Strategy**: LRU (Least Recently Used) with TTL

**Cache Behavior**:
```
Query → Hash(query + params) → Cache lookup
        ├─ HIT → Return cached result (5ms)
        └─ MISS → Compute & store → Return result (100-150ms)
```

**Stats**:
- Typical hit rate: 68%
- Cache size: 500 queries max
- Memory overhead: 50-100 bytes per entry
- TTL: Configurable (default: 3600s)

#### MemoryOptimizer (`src/parakeet_search/optimization.py`)
- **Purpose**: Reduce memory footprint
- **Techniques**:
  - **Quantization**: float32 → int8 (4x compression)
  - **Batch Processing**: Efficient embedding generation
  - **Streaming**: Process large datasets incrementally

**Memory Savings**:
```
Original (float32): 384 dims × 4 bytes = 1,536 bytes
Quantized (int8):   384 dims × 1 byte  =   384 bytes
Compression Ratio: 4x
```

---

### 3. Analytics & Evaluation

#### ClusteringAnalyzer (`src/parakeet_search/clustering.py`)
- **Algorithms**:
  - K-Means clustering
  - Hierarchical clustering (Ward linkage)
  - Silhouette analysis
  - Outlier detection

#### EvaluationFramework (`src/parakeet_search/evaluation.py`)
- **Relevance Metrics**:
  - Precision@k
  - Recall@k
  - NDCG (Normalized Discounted Cumulative Gain)
  - MRR (Mean Reciprocal Rank)

- **Coverage Metrics**:
  - Topic coverage
  - Podcast coverage
  - Temporal coverage

- **Diversity Metrics**:
  - Content diversity
  - Semantic diversity
  - Result uniqueness

---

### 4. REST API

#### FastAPI Application (`src/parakeet_search/api.py`)
- **Framework**: FastAPI with Uvicorn
- **Features**:
  - Automatic OpenAPI/Swagger docs
  - Pydantic request validation
  - CORS support
  - Error handling
  - Response compression

**Endpoints**:
| Method | Path | Purpose |
|--------|------|---------|
| GET | /health | Health check |
| POST | /search | Semantic search |
| POST | /recommendations/single | Single recommendations |
| POST | /recommendations/hybrid | Hybrid recommendations |
| GET | /cache/stats | Cache statistics |
| POST | /cache/clear | Clear cache |

---

### 5. User Interfaces

#### CLI (`src/parakeet_search/cli.py`)
- **Framework**: Click
- **Commands**:
  - `search`: Semantic search with filtering
  - `recommend`: Get recommendations
  - `analyze`: Show analytics

#### Streamlit App (`apps/streamlit_app.py`)
- **Pages**:
  1. Search - Query interface with history
  2. Recommendations - Single & hybrid modes
  3. Analytics - Dashboard with visualizations
  4. Settings - Configuration management

#### Jupyter Notebooks (`notebooks/`)
- Exploratory analysis
- Visualization samples
- Integration examples

---

## Data Flow

### Search Request Flow

```
User Input (CLI/API/Web)
     ↓
QueryParser
     ↓
CachingSearchEngine
     ├─→ Cache Hit → Return cached result
     └─→ Cache Miss:
           ↓
           EmbeddingModel (Sentence Transformers)
           ├─ Generate query embedding (384-dim)
           ├─ Vectorize query
           ↓
           SearchEngine
           ├─ Load from VectorStore (LanceDB)
           ├─ HNSW similarity search
           ├─ Compute distances
           ├─ Apply threshold filter
           ├─ Rank results
           ↓
           ResultFormatter
           ├─ Add metadata
           ├─ Format output
           ↓
           CachingSearchEngine
           ├─ Store in cache
           ↓
           User Output (CLI/API/Web)
```

**Execution Time Breakdown**:
- Embedding: ~20-30ms
- Vector search: ~30-50ms
- Formatting: ~5-10ms
- Cache overhead: ~5ms
- **Total**: 100-150ms (cache miss) or <5ms (cache hit)

---

### Recommendation Flow

```
Input: Episode ID(s)
     ↓
RecommendationEngine
     ├─ Load input episode embedding(s)
     ├─ For hybrid: Combine embeddings (weighted average)
     ├─ Search for similar episodes
     ├─ Apply filters (podcast, exclusion)
     ├─ Apply diversity boost (if requested)
     ├─ Rank by similarity + diversity score
     ↓
Output: Similar episodes
```

---

## Data Models

### Core Models (`src/parakeet_search/models.py`)

```python
class Episode(BaseModel):
    episode_id: str              # Unique identifier
    episode_title: str           # Title
    podcast_id: str              # Podcast identifier
    podcast_title: str           # Podcast name
    transcript: str              # Full transcript (optional)
    duration_minutes: float      # Episode length
    release_date: datetime       # Publication date
    embedding: Optional[np.ndarray]  # 384-dim vector

class SearchResult(BaseModel):
    episode_id: str              # Matched episode
    episode_title: str
    podcast_id: str
    podcast_title: str
    similarity: float            # 0-1 score
    distance: float              # 0-1 distance metric

class Recommendation(SearchResult):
    reason: str                  # Why recommended
```

---

## Scalability & Performance

### Current Metrics
- **Database Size**: 1000+ episodes
- **Vector Dimension**: 384 (compact model)
- **Index Type**: HNSW (approximate nearest neighbor)
- **Search Latency**: 100-150ms p50, 200ms p95
- **Cache Hit Rate**: 68% typical
- **Concurrent Capacity**: 50+ users

### Scaling Strategies

**For More Data**:
1. Use larger HNSW index
2. Implement sharding by podcast
3. Add read replicas for vector store

**For More Throughput**:
1. Increase cache TTL
2. Add API load balancer
3. Deploy multiple instances
4. Use connection pooling

**For Better Latency**:
1. Optimize HNSW parameters
2. Implement query caching
3. Add result pre-computation
4. Use GPU acceleration (optional)

---

## Technology Decisions

### Embedding Model
- **Choice**: Sentence Transformers (all-MiniLM-L6-v2)
- **Why**:
  - Lightweight (384 dimensions)
  - Fast (CPU-compatible)
  - Good quality for podcast text
  - Widely available
  - Smaller model = faster inference

### Vector Database
- **Choice**: LanceDB
- **Why**:
  - Local/embedded (no cloud deps)
  - Fast HNSW indexing
  - Simple integration
  - Good for 1000-100k vectors
  - Python-native

### API Framework
- **Choice**: FastAPI
- **Why**:
  - Auto OpenAPI docs
  - Pydantic validation
  - High performance
  - Modern Python
  - Easy to test

### Web UI
- **Choice**: Streamlit
- **Why**:
  - Rapid development
  - Interactive widgets
  - Built-in caching
  - No frontend needed
  - Great for dashboards

### Caching Strategy
- **Choice**: In-memory LRU with TTL
- **Why**:
  - Simple implementation
  - Fast access
  - Suitable for 500-1000 queries
  - Thread-safe

---

## Deployment Architecture

### Development
```
Local Machine
├─ Python virtual environment
├─ SQLite/LanceDB database
├─ Jupyter notebooks
└─ CLI + Streamlit
```

### Production
```
Server / Cloud Instance
├─ Docker container
├─ FastAPI + Uvicorn (port 8000)
├─ Streamlit (port 8501)
├─ LanceDB (persistent volume)
├─ Monitoring & logging
└─ Health checks
```

---

## Security Considerations

### Current Implementation
- No authentication required (local use)
- Input validation via Pydantic
- SQL injection: Not applicable (no SQL)
- XSS: Not applicable (no frontend templates)

### For Production
- Add API authentication (JWT/OAuth)
- Implement rate limiting
- HTTPS/TLS encryption
- Input sanitization
- Access logging

---

## Testing Architecture

### Test Pyramid

```
                    ▲
                   /E2E\                 (5% coverage)
                  /─────\
                 /  Integration\        (30% coverage)
                /──────────────\
               / Unit Tests      \     (65% coverage)
              /──────────────────\
```

**Test Coverage**:
- **Unit Tests**: 59+ (model loading, embedding, search logic)
- **Integration Tests**: 21+ (full pipeline, data loading)
- **Evaluation Tests**: 32+ (quality metrics validation)
- **Optimization Tests**: 28+ (cache, quantization)
- **API Tests**: 37+ (all endpoints, validation)
- **Total**: 150+ tests

---

## Monitoring & Observability

### Metrics to Track
- Query latency (p50, p95, p99)
- Cache hit rate
- Error rates by endpoint
- Throughput (requests/sec)
- Memory usage
- CPU utilization

### Health Checks
```
GET /health → {status: "healthy", uptime: 3600.5}
```

### Logging
```python
# Structured logging recommended
logger.info("Search executed", extra={
    "query": "machine learning",
    "duration_ms": 125,
    "results": 5,
    "cache_hit": False
})
```

---

## Future Improvements

### Short Term (6 months)
- [ ] Add authentication/authorization
- [ ] Implement query analytics
- [ ] Add user preferences
- [ ] Performance monitoring dashboard

### Medium Term (1 year)
- [ ] Multi-language support
- [ ] Advanced filtering (date ranges, speaker)
- [ ] User profiles & bookmarks
- [ ] Batch processing API

### Long Term (1+ years)
- [ ] Machine learning for ranking
- [ ] Automatic topic extraction
- [ ] Real-time indexing
- [ ] Mobile app
- [ ] Browser extension

---

## References

- [System Design Document](./IMPLEMENTATION_PLAN.md)
- [Performance Benchmarks](./BENCHMARKS.md)
- [Data Pipeline](./DATA_INGESTION.md)
- [REST API](../REST_API.md)
- [Evaluation Metrics](./EVALUATION.md)

---

**Last Updated**: November 22, 2025
**Maintainer**: Julien Pequegnot
**Status**: Production v1.0
