# Parakeet Semantic Search - Usage Guide

A comprehensive guide to using the Parakeet Semantic Search CLI for finding and discovering podcast episodes.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Search Command](#search-command)
- [Recommend Command](#recommend-command)
- [Output Formats](#output-formats)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.9+
- pip or conda package manager
- Access to populated vector database (see [DATA_INGESTION.md](DATA_INGESTION.md))

### Install from Source

```bash
git clone https://github.com/jpequegn/parakeet-semantic-search.git
cd parakeet-semantic-search
pip install -e .
```

### Verify Installation

```bash
parakeet-search --help
```

You should see help text for the CLI with available commands.

## Quick Start

### Search for Episodes

Find episodes matching a query:

```bash
parakeet-search search "machine learning"
```

### Get Recommendations

Find similar episodes to a known episode:

```bash
parakeet-search recommend --episode-id ep_001
```

## Search Command

Search for episodes matching a natural language query.

### Basic Usage

```bash
parakeet-search search <QUERY>
```

**Parameters:**
- `QUERY` (required): Your search query in natural language

### Examples

**Simple search:**
```bash
parakeet-search search "artificial intelligence"
parakeet-search search "deep learning algorithms"
parakeet-search search "transformer models and attention"
```

**Search with special characters:**
```bash
parakeet-search search "C++ & Python programming"
parakeet-search search "GPU acceleration!"
parakeet-search search "What is machine learning?"
```

### Options

#### `--limit` (default: 10)

Control how many results to return (1-1000).

```bash
# Get top 5 results
parakeet-search search "AI" --limit 5

# Get top 50 results
parakeet-search search "machine learning" --limit 50
```

#### `--threshold` (default: None)

Filter results by minimum similarity score (0.0-1.0).

```bash
# Only highly relevant results
parakeet-search search "neural networks" --threshold 0.7

# Include lower relevance results
parakeet-search search "AI" --threshold 0.3

# No filtering
parakeet-search search "deep learning" --threshold 0
```

#### `--format` (default: table)

Choose output format: `table`, `json`, or `markdown`.

```bash
# Table format (default)
parakeet-search search "AI" --format table

# JSON format (for scripting)
parakeet-search search "AI" --format json

# Markdown format (for documentation)
parakeet-search search "AI" --format markdown
```

#### `--save-results`

Save results to a file. Format is auto-detected by file extension.

```bash
# Save to JSON
parakeet-search search "AI" --save-results results.json

# Save to Markdown
parakeet-search search "AI" --save-results results.md

# Save to nested directory
parakeet-search search "AI" --save-results ./output/search_results.json
```

### Complete Example

```bash
parakeet-search search "transformer architecture" \
  --limit 20 \
  --threshold 0.5 \
  --format json \
  --save-results results.json
```

**Output:**
```json
[
  {
    "episode_id": "ep_001",
    "episode_title": "Attention is All You Need",
    "podcast_title": "AI Research",
    "relevance": 95.2,
    "distance": 0.048
  },
  {
    "episode_id": "ep_002",
    "episode_title": "Transformers in Practice",
    "podcast_title": "ML Weekly",
    "relevance": 89.7,
    "distance": 0.103
  }
]
```

## Recommend Command

Find episodes similar to a known episode.

### Basic Usage

```bash
parakeet-search recommend --episode-id <EPISODE_ID>
```

**Parameters:**
- `--episode-id` (required): The episode ID to find recommendations for

### Examples

**Simple recommendation:**
```bash
parakeet-search recommend --episode-id ep_001
```

**Get more recommendations:**
```bash
parakeet-search recommend --episode-id ep_001 --limit 20
```

**Filter by podcast:**
```bash
parakeet-search recommend --episode-id ep_001 --podcast-id pod_ai
```

**Save recommendations:**
```bash
parakeet-search recommend --episode-id ep_001 --save-results recommendations.json
```

### Options

#### `--limit` (default: 5)

Control how many recommendations to return (1-100).

```bash
# Get top 3 similar episodes
parakeet-search recommend --episode-id ep_001 --limit 3

# Get top 20 similar episodes
parakeet-search recommend --episode-id ep_001 --limit 20
```

#### `--podcast-id` (optional)

Filter recommendations by podcast.

```bash
# Only recommend from the same podcast
parakeet-search recommend --episode-id ep_001 --podcast-id pod_ai

# Only recommend from a specific podcast
parakeet-search recommend --episode-id ep_001 --podcast-id pod_ml
```

#### `--format` (default: table)

Choose output format: `table`, `json`, or `markdown`.

```bash
parakeet-search recommend --episode-id ep_001 --format json
parakeet-search recommend --episode-id ep_001 --format markdown
```

#### `--save-results`

Save recommendations to a file.

```bash
parakeet-search recommend --episode-id ep_001 --save-results recs.json
```

### Complete Example

```bash
parakeet-search recommend --episode-id ep_001 \
  --limit 10 \
  --podcast-id pod_ai \
  --format markdown \
  --save-results podcast_recommendations.md
```

## Output Formats

### Table Format (Default)

Human-readable ASCII table:

```
# | Episode              | Podcast           | Relevance
--+----------------------+-------------------+-----------
1 | ML Basics            | AI Today          | 95%
2 | Deep Learning        | Tech News         | 88%
3 | Neural Networks      | ML Weekly         | 82%
```

**Best for:** Terminal viewing, quick inspection

### JSON Format

Machine-readable JSON with full details:

```json
[
  {
    "episode_id": "ep_001",
    "episode_title": "ML Basics",
    "podcast_title": "AI Today",
    "relevance": 95.0,
    "distance": 0.05
  }
]
```

**Best for:** Scripts, automation, integration, data processing

**Using JSON in scripts:**
```bash
# Count results
parakeet-search search "AI" --format json | jq 'length'

# Extract titles
parakeet-search search "AI" --format json | jq '.[].episode_title'

# Filter by relevance
parakeet-search search "AI" --format json | jq '.[] | select(.relevance > 90)'
```

### Markdown Format

Formatted for documentation and reports:

```markdown
## Search Results

### 1. ML Basics
**Podcast**: AI Today
**Relevance**: 95.0%
**Excerpt**: Machine learning is a subset of artificial intelligence...

### 2. Deep Learning
**Podcast**: Tech News
**Relevance**: 88.0%
**Excerpt**: Deep learning uses neural networks with multiple layers...
```

**Best for:** Documentation, reports, sharing results

## Advanced Usage

### Batch Processing

Process multiple queries in a loop:

```bash
for query in "AI" "machine learning" "neural networks"; do
  parakeet-search search "$query" \
    --limit 5 \
    --format json \
    --save-results "results_${query// /_}.json"
done
```

### Pipeline Integration

Use results in other commands:

```bash
# Get first episode ID from search and get recommendations
EPISODE_ID=$(parakeet-search search "AI" --format json | jq -r '.[0].episode_id')
parakeet-search recommend --episode-id "$EPISODE_ID" --format markdown
```

### Threshold Tuning

Find optimal threshold for relevance:

```bash
# Test different thresholds
for threshold in 0.3 0.5 0.7 0.9; do
  echo "Threshold: $threshold"
  parakeet-search search "machine learning" --threshold $threshold | head -1
done
```

### Bulk Recommendations

Get recommendations for multiple episodes:

```bash
# Create batch processing script
for episode_id in ep_001 ep_002 ep_003; do
  parakeet-search recommend --episode-id "$episode_id" \
    --limit 5 \
    --format json \
    --save-results "recs_${episode_id}.json"
  echo "Completed: $episode_id"
done
```

### Database Status

Check available data:

```bash
# Test connectivity
parakeet-search search "test" --limit 1

# Get episode count (approximate)
parakeet-search search "." --limit 1000 | wc -l
```

## Troubleshooting

### Database Connection Errors

**Error:** `RuntimeError: Vector database not initialized`

**Solutions:**
1. Verify database file exists: Check `data/` directory
2. Run data ingestion: See [DATA_INGESTION.md](DATA_INGESTION.md)
3. Check database permissions: `ls -la data/`

```bash
# Debug command
python -c "from parakeet_search.search import SearchEngine; \
           e = SearchEngine(); \
           print(e.vectorstore)"
```

### No Results Found

**Problem:** Search returns no results even for broad queries

**Solutions:**
1. **Lower the threshold:**
   ```bash
   parakeet-search search "AI" --threshold 0
   ```

2. **Increase limit:**
   ```bash
   parakeet-search search "AI" --limit 100
   ```

3. **Try simpler query:**
   ```bash
   parakeet-search search "learning"  # Instead of "machine learning algorithms"
   ```

4. **Verify data is loaded:**
   ```bash
   parakeet-search search "." --limit 1
   ```

### Invalid Episode ID

**Error:** `ValueError: Episode not found`

**Solutions:**
1. **Verify episode ID format:**
   - Should be like: `ep_001`, `ep_123`, etc.
   - Check format in search results

2. **Find valid episode ID:**
   ```bash
   parakeet-search search "any topic" --format json | jq '.[0].episode_id'
   ```

3. **Use search first:**
   ```bash
   # Find episode then get recommendations
   EPISODE=$(parakeet-search search "AI" --format json | jq -r '.[0]')
   echo "$EPISODE" | jq '.episode_id'
   ```

### Output Format Issues

**Problem:** JSON output not parsing correctly

**Solutions:**
1. **Strip progress output:**
   ```bash
   parakeet-search search "AI" --format json 2>/dev/null | jq '.'
   ```

2. **Use explicit output redirection:**
   ```bash
   parakeet-search search "AI" --format json > results.json 2>&1
   ```

3. **Validate JSON:**
   ```bash
   parakeet-search search "AI" --format json | python -m json.tool
   ```

### Performance Issues

**Problem:** Search is slow

**Solutions:**
1. **Reduce limit:**
   ```bash
   # Instead of --limit 1000, use:
   parakeet-search search "AI" --limit 10
   ```

2. **Use higher threshold:**
   ```bash
   # Skip low-relevance results
   parakeet-search search "AI" --threshold 0.5
   ```

3. **Check system resources:**
   ```bash
   # Monitor memory/CPU
   top  # Press 'q' to exit
   ```

### File Save Errors

**Error:** `Error: Failed to save results`

**Solutions:**
1. **Check directory exists:**
   ```bash
   mkdir -p results/
   parakeet-search search "AI" --save-results results/output.json
   ```

2. **Check write permissions:**
   ```bash
   touch results/test.txt
   parakeet-search search "AI" --save-results results/output.json
   ```

3. **Use absolute paths:**
   ```bash
   parakeet-search search "AI" --save-results /tmp/results.json
   ```

### Command Not Found

**Error:** `parakeet-search: command not found`

**Solutions:**
1. **Reinstall package:**
   ```bash
   pip install -e .
   ```

2. **Check Python path:**
   ```bash
   python3 -m parakeet_search.cli search "AI"
   ```

3. **Use full module path:**
   ```bash
   python3 -c "from parakeet_search.cli import main; main()" search "AI"
   ```

## Getting Help

### View Help Text

```bash
# General help
parakeet-search --help

# Search command help
parakeet-search search --help

# Recommend command help
parakeet-search recommend --help
```

### Example Commands

```bash
# Simple search
parakeet-search search "AI"

# Search with all options
parakeet-search search "machine learning" \
  --limit 20 \
  --threshold 0.5 \
  --format json \
  --save-results ml_results.json

# Get recommendations
parakeet-search recommend --episode-id ep_001 --limit 10

# Find similar episodes from same podcast
parakeet-search recommend --episode-id ep_001 \
  --podcast-id pod_ai \
  --format markdown \
  --save-results similar.md
```

## Next Steps

- Review [DATA_INGESTION.md](DATA_INGESTION.md) for data pipeline details
- Check [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for architecture overview
- See [BENCHMARKS.md](BENCHMARKS.md) for performance metrics
