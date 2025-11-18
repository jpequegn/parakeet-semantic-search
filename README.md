# Parakeet Semantic Search

Building an intelligent semantic search and recommendation engine for podcast transcripts using embeddings and vector similarity search.

## Vision

Transform your podcast archive into a discoverable knowledge base:
- **Semantic Search**: "Find episodes about AI regulation" (not just keyword matching)
- **Recommendations**: Get similar episodes based on content
- **Analytics**: Understand topics, trends, and speaker patterns
- **Portability**: Works locally, no cloud dependencies

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/parakeet-semantic-search.git
cd parakeet-semantic-search
python -m venv venv
source venv/bin/activate
pip install -e .

# Generate embeddings from your P³ podcasts
python scripts/ingest_transcripts.py --source /path/to/p3/duckdb.db
python scripts/generate_embeddings.py
python scripts/build_index.py

# Search
python -m parakeet_search.cli search "AI and regulation"
```

## Project Status

🚧 **In Development** - Phase 1 starting

See [GitHub Issues](https://github.com/yourusername/parakeet-semantic-search/issues) for current work.

## Architecture

```
P³ Podcast Data (DuckDB)
    ↓
Extract Transcripts
    ↓
Generate Embeddings (Sentence Transformers)
    ↓
Vector Store (LanceDB)
    ↓
Search Engine + Recommendations
    ↓
UI: CLI, Jupyter, Streamlit
```

## Tech Stack

- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Store**: LanceDB (local, embedded)
- **Data Source**: P³ DuckDB transcripts
- **Interfaces**: CLI, Jupyter notebooks, Streamlit (Phase 4)
- **Language**: Python 3.9+

## Phases

### Phase 1: Foundation (Weeks 1-2)
- ✅ Project setup
- Embedding model selection and testing
- Data ingestion from P³
- Vector store initialization

### Phase 2: Search (Weeks 3-4)
- Semantic search implementation
- Query processing
- Result ranking and filtering
- CLI interface

### Phase 3: Recommendations (Weeks 5-6)
- Content-based recommendations
- Similarity metrics
- Quality evaluation

### Phase 4: Polish (Weeks 7-8)
- Streamlit web interface
- Analytics dashboard
- Performance optimization
- Documentation

## Resources

- [Architecture Document](docs/ARCHITECTURE.md)
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [Design Decisions](docs/DESIGN_DECISIONS.md)

## Contributing

This is a personal learning project. See [GitHub Issues](https://github.com/yourusername/parakeet-semantic-search/issues) for current tasks.

## License

MIT
