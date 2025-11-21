# Jupyter Notebooks

Interactive analysis and exploration notebooks for Parakeet Semantic Search.

## Available Notebooks

### `exploratory_analysis.ipynb`

Comprehensive exploratory data analysis of the podcast embeddings and search system.

**Contents**:
1. **Setup and Imports** - Initialize Jupyter environment and load dependencies
2. **Data Loading** - Load embeddings from LanceDB vector store
3. **Embedding Distribution Analysis** - Statistics and visualization of embedding space
4. **Dimensionality Reduction** - t-SNE visualization of embeddings (2D projection)
5. **Clustering Analysis** - K-Means clustering with silhouette analysis
6. **Search Query Examples** - Demonstrate semantic search with sample queries
7. **Recommendation Engine** - Show content-based recommendations
8. **Similarity Analysis** - Pairwise distance statistics and distribution
9. **Performance Analysis** - Benchmark search and embedding generation
10. **Hybrid Recommendations** - Multi-episode recommendation demonstration
11. **Key Findings** - Summary of insights and observations
12. **Conclusions** - Final analysis and recommendations

**Key Visualizations**:
- Embedding magnitude distribution histogram
- t-SNE scatter plot colored by podcast (shows semantic clustering)
- Elbow method for optimal cluster detection
- Silhouette score analysis
- K-Means clustering visualization
- Pairwise distance distribution

**Expected Output**:
- Dataset statistics (samples, dimensions, uniqueness)
- Clustering metrics (inertia, silhouette scores)
- Search latency measurements
- Quality examples of search and recommendation results
- Detailed findings and improvement recommendations

## Prerequisites

### Environment

```bash
# Activate virtual environment
source venv/bin/activate

# Install notebook support
pip install jupyter notebook ipykernel
```

### Dependencies

The notebooks require:
- `jupyter` - Interactive notebook environment
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Static visualizations
- `plotly` - Interactive visualizations
- `scikit-learn` - Machine learning (t-SNE, KMeans, metrics)
- `scipy` - Scientific computing (distance metrics)

All dependencies are in `requirements.txt` (except notebook/jupyter which is optional).

## Running the Notebooks

### Option 1: Jupyter Notebook (GUI)

```bash
# From project root
jupyter notebook notebooks/exploratory_analysis.ipynb
```

This opens the notebook in your default browser at `http://localhost:8888`

### Option 2: Jupyter Lab (Enhanced Interface)

```bash
pip install jupyterlab
jupyter lab notebooks/exploratory_analysis.ipynb
```

### Option 3: Command Line (VSCode, etc.)

Most IDEs support Jupyter notebooks natively:
- **VSCode**: Open `.ipynb` file directly
- **PyCharm**: Open notebook in IDE
- **JetBrains**: Native notebook support

## Running Cells

1. **Run single cell**: Click cell and press `Shift+Enter`
2. **Run all cells**: Menu → Cell → Run All
3. **Run up to current**: Menu → Cell → Run All Above

## Notes

### Dataset Size
- The t-SNE visualization uses a 1,000-sample subset for performance (edit `n_samples` to adjust)
- Clustering analysis uses full dataset (may be slow for very large datasets)
- Adjust sample sizes based on your hardware

### Dependencies
- `umap-learn` is optional but recommended for faster dimensionality reduction (alternative to t-SNE)
- `plotly` requires internet connection for rendering (works offline in Jupyter Lab with extension)

### Performance
- t-SNE computation can take 1-5 minutes depending on dataset size
- KMeans clustering is O(n*k) - adjust `k_range` if slow
- Full notebook runtime: 5-15 minutes depending on dataset size

## Customization

Edit notebook cells to:
- Change sample size: `n_samples = min(1000, len(embeddings))`
- Adjust clustering k range: `k_range = range(2, 11)`
- Modify query examples: `example_queries = [...]`
- Change visualization parameters: Modify `fig.update_layout()`

## Output

The notebook generates:
- Interactive visualizations (Plotly)
- Statistical summaries
- Performance metrics
- Example search and recommendation results
- Insights and recommendations for system improvement

## Troubleshooting

### Import Errors
```bash
# Reinstall dependencies
pip install -e .
pip install plotly scikit-learn scipy
```

### Slow t-SNE
- Reduce `n_samples` value
- Use UMAP instead: `from umap import UMAP; UMAP(n_components=2).fit_transform(...)`
- Skip t-SNE if dataset is very large

### Memory Issues
- Reduce sample size for analysis
- Process in batches
- Use smaller clustering k values

### LanceDB Connection
- Verify `data/vectors.db` exists
- Check file permissions: `ls -la data/`
- Reingest data if corrupted: See `docs/DATA_INGESTION.md`

## Tips for Use

1. **Interactive Exploration**: Modify cells and re-run to explore different questions
2. **Save Results**: Export notebook as HTML/PDF for sharing
3. **Version Control**: Use `git` to track notebook changes
4. **Reproducibility**: Set `random_state` parameters for consistent results

## See Also

- **[DATA_INGESTION.md](../docs/DATA_INGESTION.md)** - How data is processed
- **[USAGE.md](../docs/USAGE.md)** - CLI usage guide
- **[ARCHITECTURE.md](../docs/ARCHITECTURE.md)** - System architecture
- **[README.md](../README.md)** - Project overview

## Creating New Notebooks

To add analysis notebooks:
1. Create new `.ipynb` file in `notebooks/`
2. Start with template cell structure
3. Document purpose and expected outputs
4. Update this README with description
5. Commit to git

Example template:
```python
# 1. Setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))

# 2. Imports
import pandas as pd
import numpy as np
from parakeet_search.search import SearchEngine

# 3. Initialize
search_engine = SearchEngine()

# 4. Analysis...
```
