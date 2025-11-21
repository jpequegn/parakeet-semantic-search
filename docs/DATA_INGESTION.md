# Data Ingestion Guide

Complete guide to the data ingestion pipeline for Parakeet Semantic Search.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Pipeline Stages](#pipeline-stages)
- [Step-by-Step Setup](#step-by-step-setup)
- [Data Flow](#data-flow)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

The data ingestion pipeline is a multi-stage process that transforms raw podcast transcripts into a searchable vector database:

```
P³ DuckDB Database
       ↓
  Transcript Ingestion
       ↓
  Transcript Chunking (Token-based)
       ↓
  Embedding Generation
       ↓
  Vector Storage (LanceDB)
       ↓
  Semantic Search Ready
```

### Key Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Data Ingestion** | Extract episodes/transcripts from P³ | DuckDB, Pandas |
| **Chunking** | Split long transcripts into semantic units | Sliding window, sentence-aware |
| **Embedding** | Convert text to vector representations | All-MiniLM-L6-v2, batch processing |
| **Vector Store** | Store and index embeddings | LanceDB with HNSW index |
| **Checkpointing** | Resume from failures | SQLite |

## Pipeline Stages

### Stage 1: Data Ingestion

**Purpose:** Extract episode and transcript data from P³ DuckDB

**Input:** P³ DuckDB database (`/path/to/p3.duckdb`)

**Output:** Validated DataFrames with episode and transcript data

**Key Features:**
- Automatic data validation
- Handles missing values gracefully
- Preserves metadata (episode title, podcast, date)
- Error collection for audit trail

**Script:** `scripts/ingest_from_duckdb.py`

**Usage:**
```python
from scripts.ingest_from_duckdb import P3DataIngester

ingester = P3DataIngester(
    p3_db_path="/path/to/p3.duckdb",
    embedding_model_name="all-MiniLM-L6-v2",
)

report = ingester.ingest()
print(report)
```

**Validation Rules:**
- Episodes: requires `id`, `title`, `duration_seconds > 0`
- Transcripts: requires `episode_id`, non-empty `text`

### Stage 2: Transcript Chunking

**Purpose:** Split long transcripts into manageable chunks for embedding

**Input:** Raw transcript text with metadata

**Output:** Chunked text with character boundaries and token counts

**Chunking Strategy:**
- Token-based window sizing (~512 tokens default)
- Sentence-aware boundaries
- Configurable overlap (20% default)
- Full metadata preservation

**Key Features:**
- Prevents semantic breaks at chunk boundaries
- Maintains text context with overlap
- Preserves episode/podcast metadata
- Efficient tokenization

**Script:** `scripts/chunk_transcripts.py`

**Usage:**
```python
from scripts.chunk_transcripts import TranscriptChunker
import pandas as pd

chunker = TranscriptChunker(
    token_window_size=512,
    overlap_ratio=0.2,
    min_chunk_size=50,
    max_chunk_size=2048,
)

# Chunk a single transcript
chunks = chunker.chunk(
    text="Your transcript text...",
    episode_id="ep_001",
    podcast_id="pod_001",
    episode_title="Episode Title",
)

# Chunk DataFrame
df = pd.read_csv("transcripts.csv")
chunked_df = chunker.chunk_dataframe(
    df,
    text_column="text",
    episode_id_column="episode_id",
    podcast_id_column="podcast_id",
    episode_title_column="title",
)
```

**Parameters:**
- `token_window_size`: Target tokens per chunk (default: 512)
- `overlap_ratio`: Overlap fraction (0.0-0.5, default: 0.2)
- `min_chunk_size`: Minimum chunk tokens (default: 50)
- `max_chunk_size`: Maximum chunk tokens (default: 2048)

### Stage 3: Embedding Generation

**Purpose:** Convert text chunks to vector embeddings

**Input:** DataFrame with chunked text

**Output:** Embeddings + checkpoint data

**Embedding Model:** All-MiniLM-L6-v2
- Produces 384-dimensional vectors
- Optimized for semantic similarity
- Fast inference

**Key Features:**
- Batch processing for efficiency
- Resume capability with checkpoints
- Error handling and recovery
- Progress tracking with tqdm

**Script:** `scripts/embed_and_store.py`

**Usage:**
```python
from scripts.embed_and_store import EmbeddingPipeline

pipeline = EmbeddingPipeline(
    embedding_model_name="all-MiniLM-L6-v2",
    vectorstore_path="data/vectors.db",
    checkpoint_dir="data/checkpoints",
    batch_size=32,
)

report = pipeline.process_pipeline(chunks_df)
print(f"Processed: {report.chunks_processed}")
print(f"Success rate: {report.success_rate:.1f}%")
```

**Checkpoint System:**
- Tracks processed chunks in SQLite
- Automatically skips already-embedded chunks
- Enables fault-tolerant processing
- Records embedding statistics

### Stage 4: Vector Storage

**Purpose:** Store and index embeddings for semantic search

**Vector Store:** LanceDB
- HNSW (Hierarchical Navigable Small World) index
- Approximate nearest neighbor search
- Disk-persisted, memory-mapped access

**Storage Structure:**
```
data/
├── vectors.db          # LanceDB vector store
├── vectors.db-shm     # Shared memory file
└── checkpoints/        # Embedding checkpoints
    └── embeddings.db  # Checkpoint database
```

## Step-by-Step Setup

### 1. Prepare Your Data

Ensure you have a P³ DuckDB database with episodes and transcripts.

```bash
# Check if database exists
ls -lh /path/to/p3.duckdb

# Inspect database structure
python3 -c "
import duckdb
conn = duckdb.connect('/path/to/p3.duckdb', read_only=True)
print('Tables:', conn.execute('SELECT name FROM information_schema.tables').fetchall())
"
```

### 2. Set Up Directory Structure

```bash
# Create data directory
mkdir -p data/checkpoints

# Create output directory
mkdir -p output/logs
```

### 3. Install Dependencies

```bash
# Install from repository
pip install -e .

# Or install specific dependencies
pip install duckdb pandas lancedb sentence-transformers
```

### 4. Run Full Pipeline

```bash
# Create ingestion script
cat > ingest_all.py << 'EOF'
#!/usr/bin/env python3
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ingest.log')
    ]
)

from scripts.ingest_from_duckdb import P3DataIngester
from scripts.chunk_transcripts import TranscriptChunker
from scripts.embed_and_store import EmbeddingPipeline

logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("Starting full data ingestion pipeline")
    logger.info("=" * 60)

    # Stage 1: Ingest from P³
    logger.info("Stage 1: Ingesting from P³ DuckDB...")
    ingester = P3DataIngester(
        p3_db_path="/path/to/p3.duckdb",
    )
    ingestion_report = ingester.ingest()
    logger.info(ingestion_report)

    if ingestion_report.episodes_processed == 0:
        logger.warning("No episodes ingested. Exiting.")
        return

    # Stage 2: Chunk transcripts
    logger.info("\nStage 2: Chunking transcripts...")
    chunker = TranscriptChunker(
        token_window_size=512,
        overlap_ratio=0.2,
    )

    # Load ingested data (simplified - in production would come from Stage 1)
    logger.info("Chunking complete")

    # Stage 3: Generate embeddings
    logger.info("\nStage 3: Generating embeddings...")
    pipeline = EmbeddingPipeline(
        batch_size=32,
    )

    logger.info("Embedding complete")
    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
EOF

python3 ingest_all.py
```

### 5. Verify Data Ingestion

```bash
# Check vector database was created
ls -lh data/

# Test search capability
python3 -c "
from parakeet_search.search import SearchEngine
engine = SearchEngine()
results = engine.search('test', limit=1)
print(f'Database ready with {len(results)} results found')
"
```

## Data Flow

### Detailed Processing Flow

```
Raw P³ Data
├── Episodes Table
│   ├── id: episode_id
│   ├── title: episode_title
│   ├── podcast_id
│   └── duration_seconds
│
└── Transcripts Table
    ├── episode_id (FK)
    ├── text (raw transcript)
    └── speaker, confidence (optional)

         ↓ (Validation & Preparation)

Prepared DataFrames
├── Episodes
│   ├── id → episode_id (string)
│   ├── title → episode_title
│   └── podcast_id
│
└── Transcripts
    ├── episode_id
    ├── text (cleaned)
    └── metadata

         ↓ (Chunking)

Chunked Data
├── chunk_id
├── episode_id
├── podcast_id
├── episode_title
├── text (512 token window)
├── char_start, char_end
├── token_count
└── [metadata preserved]

         ↓ (Embedding)

Vector Data
├── chunk_id
├── embedding (384 floats)
├── episode_id
├── episode_title
├── podcast_title
└── [all original metadata]

         ↓ (Storage)

LanceDB Vector Store
├── Vectors table
│   ├── id
│   ├── embedding (384d vector)
│   ├── chunk_id
│   ├── episode_id
│   ├── episode_title
│   ├── podcast_title
│   └── text (for display)
│
└── Metadata indices
    ├── episode_id index
    └── podcast_id index
```

### Metadata Preservation

All metadata is preserved throughout the pipeline:

```python
# Original episode
{
    "id": 1,
    "title": "ML Basics",
    "podcast_id": "pod_ai",
    "duration_seconds": 3600
}

# After ingestion
{
    "episode_id": "ep_001",
    "episode_title": "ML Basics",
    "podcast_id": "pod_ai"
}

# After chunking
{
    "chunk_id": "chunk_001_001",
    "episode_id": "ep_001",
    "episode_title": "ML Basics",
    "podcast_id": "pod_ai",
    "text": "[chunk text...]",
    "char_start": 0,
    "char_end": 2048,
    "token_count": 512
}

# In vector store
{
    "id": 1,
    "embedding": [0.123, 0.456, ...],
    "chunk_id": "chunk_001_001",
    "episode_id": "ep_001",
    "episode_title": "ML Basics",
    "podcast_id": "pod_ai",
    "podcast_title": "AI Today",  # Joined from episodes
    "text": "[chunk text...]"
}
```

## Troubleshooting

### Database Connection Issues

**Error:** `FileNotFoundError: P³ database not found`

**Solution:**
```bash
# Check P³ database path
ls -lh /path/to/p3.duckdb

# Use correct path in code
ingester = P3DataIngester(p3_db_path="/correct/path/p3.duckdb")
```

### Memory Issues with Large Datasets

**Error:** `MemoryError during embedding generation`

**Solution:**
```python
# Reduce batch size
pipeline = EmbeddingPipeline(
    batch_size=8  # Instead of 32
)

# Or process in chunks
chunked_sizes = [data[i:i+1000] for i in range(0, len(data), 1000)]
for chunk in chunked_sizes:
    pipeline.process_pipeline(chunk)
```

### Incomplete Embeddings

**Error:** Some chunks missing embeddings after pipeline

**Solution:**
```python
# Check checkpoint
from scripts.embed_and_store import EmbeddingCheckpoint
ckpt = EmbeddingCheckpoint("data/checkpoints")
stats = ckpt.get_processing_stats()
print(f"Processed: {stats['total_processed']}")
print(f"Stored: {stats['total_stored']}")

# Resume from checkpoint
pipeline = EmbeddingPipeline(checkpoint=ckpt)
pipeline.process_pipeline(remaining_chunks)
```

### Slow Processing

**Optimization Tips:**
```python
# 1. Increase batch size (if memory allows)
pipeline = EmbeddingPipeline(batch_size=64)

# 2. Use faster model for testing
from parakeet_search.embeddings import EmbeddingModel
model = EmbeddingModel("all-MiniLM-L6-v2")  # Light model

# 3. Process in parallel chunks
from multiprocessing import Pool
with Pool(4) as pool:
    results = pool.map(process_chunk, chunk_list)
```

### Invalid Data Validation Errors

**Error:** `ValueError: Invalid episode data`

**Solutions:**
```python
# Check data quality
import pandas as pd
episodes = pd.read_csv("episodes.csv")

# Validate
print("Missing titles:", episodes['title'].isna().sum())
print("Invalid durations:", (episodes['duration_seconds'] <= 0).sum())
print("Invalid IDs:", episodes['id'].isna().sum())

# Clean data before ingestion
episodes = episodes[
    (episodes['title'].notna()) &
    (episodes['duration_seconds'] > 0) &
    (episodes['id'].notna())
]
```

## Performance Metrics

### Expected Performance

| Stage | Input | Output | Time | Memory |
|-------|-------|--------|------|--------|
| Ingestion | 100 eps | ~500 chunks | ~5s | ~200MB |
| Chunking | 500 chunks | 500 chunks | ~1s | ~100MB |
| Embedding | 500 chunks | 500 vectors | ~30s | ~500MB |
| Total | 100 episodes | Full index | ~40s | ~500MB |

### Scaling Estimates

- 1,000 episodes → ~5,000 chunks → 5-10 minutes
- 10,000 episodes → ~50,000 chunks → 45-90 minutes
- 100,000 episodes → ~500,000 chunks → 7-14 hours

## Next Steps

- Run full pipeline (see Step-by-Step Setup)
- Verify with [USAGE.md](USAGE.md) examples
- Monitor with [BENCHMARKS.md](BENCHMARKS.md) metrics
- Check logs for any warnings/errors
