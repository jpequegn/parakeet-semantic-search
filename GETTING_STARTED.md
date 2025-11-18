# Getting Started with Parakeet Semantic Search

This guide will help you get the project set up and running.

## Quick Start

### 1. Clone & Setup

```bash
cd /Users/julienpequegnot/Code/parakeet-semantic-search

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
pip install -r requirements.txt
```

### 2. Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/parakeet_search --cov-report=html
```

### 3. Try the CLI

```bash
# Search for episodes
parakeet-search search "machine learning"

# Get recommendations
parakeet-search recommend --episode-id 42

# Export results
parakeet-search search "AI" --format json --save-results results.json
```

## Development Workflow

### Code Quality

Before committing, run:

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type checking
mypy src/
```

### Running Specific Tests

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Single test
pytest tests/test_embeddings.py::test_embed_text -v
```

## Project Structure

```
parakeet-semantic-search/
├── src/parakeet_search/          # Main source code
│   ├── embeddings.py            # Sentence Transformers wrapper
│   ├── vectorstore.py           # LanceDB interface
│   ├── search.py                # Search engine logic
│   └── cli.py                   # CLI commands
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   └── integration/             # Integration tests
├── scripts/                      # Data processing scripts
│   └── create_github_issues.sh   # Auto-create GitHub issues
├── docs/                        # Documentation
│   ├── IMPLEMENTATION_PLAN.md   # Detailed roadmap
│   └── GITHUB_ISSUES.md         # Issue templates
├── notebooks/                   # Jupyter notebooks
├── data/                        # Vector store (git-ignored)
└── apps/                        # Applications (CLI, Streamlit, API)
```

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- ✅ Core classes (Embedding, VectorStore, SearchEngine)
- ⏳ Unit & integration tests
- ⏳ Performance benchmarks

### Phase 2: Search Interface (Weeks 3-4)
- ⏳ Full CLI implementation
- ⏳ Data ingestion pipeline
- ⏳ Transcript chunking

### Phase 3: Advanced Features (Weeks 5-6)
- ⏳ Recommendation engine
- ⏳ Clustering & topic analysis
- ⏳ Quality metrics

### Phase 4: Polish & Release (Weeks 7-8)
- ⏳ Streamlit web app
- ⏳ REST API (optional)
- ⏳ Docker containerization

## Creating GitHub Issues

Once you push to GitHub, create all issues automatically:

```bash
# First, push repository to GitHub
git remote add origin https://github.com/USERNAME/parakeet-semantic-search.git
git branch -M main
git push -u origin main

# Then create all issues
./scripts/create_github_issues.sh
```

This will create:
- 18 issues across 4 phases
- Labels for organization
- Milestones for tracking

## Data Ingestion

To populate the vector store with P³ podcast data:

```bash
# 1. Ingest data from P³
python3 scripts/ingest_from_duckdb.py

# 2. Chunk transcripts
python3 scripts/chunk_transcripts.py

# 3. Generate embeddings
python3 scripts/embed_and_store.py
```

## Architecture

```
P³ DuckDB
    ↓
Ingest Episodes & Transcripts
    ↓
Chunk Transcripts (sliding window)
    ↓
Generate Embeddings (Sentence Transformers)
    ↓
Store in LanceDB
    ↓
┌──────────────────────────────────┐
│    Semantic Search Engine        │
├──────────────────────────────────┤
│  CLI  │  Jupyter  │  Streamlit   │
└──────────────────────────────────┘
```

## Configuration

### Environment Variables

Create `.env` file:

```bash
# P³ Database path
P3_DATABASE_PATH=~/.claude/data/p3.duckdb

# LanceDB path
VECTORSTORE_PATH=data/vectors.db

# Embedding model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Search settings
DEFAULT_RESULT_LIMIT=10
DEFAULT_SIMILARITY_THRESHOLD=0.5
```

## Performance Expectations

- **Embedding Generation**: >100 texts/second (batch)
- **Search Latency**: <500ms per query (target)
- **Memory Usage**: <2GB for 10K episodes
- **Storage**: ~200MB for 10K episodes + embeddings

## Troubleshooting

### Import Errors

```bash
# Make sure you installed in development mode
pip install -e .

# Or verify installation
python3 -c "from parakeet_search import SearchEngine; print('OK')"
```

### Test Failures

```bash
# Check test dependencies
pip install pytest pytest-cov

# Run with verbose output
pytest tests/ -vv --tb=long
```

### Missing P³ Database

The system expects P³ database at `~/.claude/data/p3.duckdb`. If not found:
```bash
# Create placeholder or check path
export P3_DATABASE_PATH=/path/to/database.duckdb
```

## Next Steps

1. **Read** `docs/IMPLEMENTATION_PLAN.md` for detailed roadmap
2. **Review** `docs/GITHUB_ISSUES.md` for task breakdown
3. **Start** Phase 1 issues (testing & benchmarks)
4. **Track** progress using GitHub milestones

## Additional Resources

- **Sentence Transformers**: https://www.sbert.net/
- **LanceDB**: https://lancedb.com/
- **Click CLI**: https://click.palletsprojects.com/
- **Streamlit**: https://streamlit.io/

## Getting Help

For questions about:
- **Architecture**: See `docs/ARCHITECTURE.md`
- **Implementation**: See `docs/IMPLEMENTATION_PLAN.md`
- **Usage**: See `docs/USAGE.md` (Phase 2)
- **API**: See `docs/API.md` (Phase 4)

---

**Happy coding!** 🚀

Start with Phase 1: Setting up comprehensive tests and benchmarks.
