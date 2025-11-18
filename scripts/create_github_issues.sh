#!/bin/bash

# Parakeet Semantic Search - GitHub Issues Creation Script
# This script creates all GitHub issues for the project
# Usage: ./scripts/create_github_issues.sh

set -e

echo "🚀 Parakeet Semantic Search - Issue Creation Script"
echo "=================================================="

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo "Install from: https://cli.github.com/"
    exit 1
fi

# Check if we're in a git repository with a remote
if ! git remote get-url origin &> /dev/null; then
    echo "❌ No git remote found. Please push to GitHub first:"
    echo "   git remote add origin https://github.com/username/parakeet-semantic-search.git"
    echo "   git push -u origin main"
    exit 1
fi

REPO=$(gh repo view --json nameWithOwner -q)
echo "📦 Repository: $REPO"

# Create labels
echo ""
echo "📌 Creating labels..."
gh label create "phase-1" --description "Phase 1: Foundation & Setup" --color "0052cc" 2>/dev/null || true
gh label create "phase-2" --description "Phase 2: Search Interface & CLI" --color "0075ca" 2>/dev/null || true
gh label create "phase-3" --description "Phase 3: Advanced Features" --color "009800" 2>/dev/null || true
gh label create "phase-4" --description "Phase 4: Polish & Deployment" --color "fbca04" 2>/dev/null || true
gh label create "testing" --description "Testing related" --color "ffd700" 2>/dev/null || true
gh label create "infrastructure" --description "Infrastructure setup" --color "d4c5f9" 2>/dev/null || true
gh label create "documentation" --description "Documentation" --color "0075ca" 2>/dev/null || true
gh label create "feature" --description "New feature" --color "a2eeef" 2>/dev/null || true
gh label create "cli" --description "CLI related" --color "cccccc" 2>/dev/null || true
gh label create "data-ingestion" --description "Data ingestion pipeline" --color "5319e7" 2>/dev/null || true
gh label create "ml" --description "ML/AI related" --color "c2e0c6" 2>/dev/null || true
gh label create "performance" --description "Performance optimization" --color "f29513" 2>/dev/null || true
gh label create "optimization" --description "Code optimization" --color "fbca04" 2>/dev/null || true
gh label create "ui" --description "User interface" --color "c5def5" 2>/dev/null || true
gh label create "api" --description "API related" --color "0969da" 2>/dev/null || true
gh label create "deployment" --description "Deployment related" --color "ff6b6b" 2>/dev/null || true
gh label create "release" --description "Release related" --color "0e8a16" 2>/dev/null || true
gh label create "analysis" --description "Data analysis" --color "1f883d" 2>/dev/null || true
echo "✅ Labels created"

# Create milestones
echo ""
echo "🎯 Creating milestones..."
gh milestone create "Phase 1 - Foundation" --description "Weeks 1-2: Core infrastructure and tests" 2>/dev/null || true
gh milestone create "Phase 2 - Search Interface" --description "Weeks 3-4: Search CLI and data ingestion" 2>/dev/null || true
gh milestone create "Phase 3 - Advanced Features" --description "Weeks 5-6: Recommendations and analysis" 2>/dev/null || true
gh milestone create "Phase 4 - Polish & Release" --description "Weeks 7-8: Web UI and final polish" 2>/dev/null || true
echo "✅ Milestones created"

# Phase 1 Issues
echo ""
echo "📋 Creating Phase 1 issues..."

gh issue create \
  --title "Phase 1.1: Unit tests for EmbeddingModel and VectorStore" \
  --body "Implement comprehensive unit tests for core components with ≥80% coverage.

## Tasks
- [ ] Test \`EmbeddingModel.embed_text()\` with sample strings
- [ ] Test \`EmbeddingModel.embed_texts()\` with batch processing
- [ ] Test \`VectorStore.create_table()\` and \`add_data()\`
- [ ] Test \`VectorStore.search()\` with mock queries
- [ ] Test \`SearchEngine.search()\` with mock embeddings
- [ ] Verify error handling for invalid inputs
- [ ] Generate coverage report

## Acceptance Criteria
- All tests pass
- Coverage report shows ≥80%
- Mocks properly isolate units under test

## Effort Estimate
2-3 days" \
  --label "phase-1,testing,infrastructure" \
  --milestone "Phase 1 - Foundation"

gh issue create \
  --title "Phase 1.2: Integration tests - Full embedding to search pipeline" \
  --body "Test end-to-end workflow from transcript text to search results.

## Tasks
- [ ] Load small test dataset (5-10 episodes)
- [ ] Test embedding generation for all episodes
- [ ] Test vector store population
- [ ] Test semantic search with various queries
- [ ] Validate result accuracy and ranking
- [ ] Test error handling (malformed input, edge cases)

## Acceptance Criteria
- Full pipeline executes without errors
- Search results are semantically relevant
- Error handling is graceful
- Integration tests pass

## Effort Estimate
2-3 days" \
  --label "phase-1,testing,infrastructure" \
  --milestone "Phase 1 - Foundation"

gh issue create \
  --title "Phase 1.3: Data schema documentation & Pydantic models" \
  --body "Define and document data models using Pydantic with validation.

## Tasks
- [ ] Create \`Episode\` Pydantic model with validation
- [ ] Create \`Transcript\` model (text, embedding, metadata)
- [ ] Create \`SearchResult\` model with scoring
- [ ] Create \`Config\` model for settings
- [ ] Document all fields and constraints
- [ ] Add examples in docstrings

## Acceptance Criteria
- All models properly validate input
- Type hints are complete
- Documentation explains each model
- Models integrate with existing code

## Effort Estimate
1-2 days" \
  --label "phase-1,documentation,infrastructure" \
  --milestone "Phase 1 - Foundation"

gh issue create \
  --title "Phase 1.4: Performance benchmarks & profiling" \
  --body "Establish baseline performance metrics and profiling infrastructure.

## Tasks
- [ ] Benchmark embedding generation speed (texts/second)
- [ ] Profile memory usage during embedding
- [ ] Benchmark vector store creation and search
- [ ] Measure search latency with various dataset sizes
- [ ] Create \`docs/BENCHMARKS.md\` with results
- [ ] Set up benchmark suite in pytest

## Acceptance Criteria
- Benchmarks documented with baseline numbers
- Benchmarks are reproducible
- Performance targets defined
- Profiling code included for future optimization

## Effort Estimate
2-3 days" \
  --label "phase-1,performance,infrastructure" \
  --milestone "Phase 1 - Foundation"

# Phase 2 Issues
echo "📋 Creating Phase 2 issues..."

gh issue create \
  --title "Phase 2.1: Enhance CLI - Full search command implementation" \
  --body "Implement complete \`search\` command with all options and features.

## Tasks
- [ ] Implement \`--limit\` option (results count)
- [ ] Implement \`--threshold\` option (similarity minimum)
- [ ] Implement \`--format\` option (json/markdown/table)
- [ ] Implement \`--save-results\` option (export to file)
- [ ] Add progress indicators for operations
- [ ] Improve error messages and user feedback
- [ ] Test all command combinations

## Acceptance Criteria
- \`parakeet-search search \"query\"\` returns results
- All options work correctly
- Output is formatted properly
- Error messages are helpful
- CLI tests pass

## Effort Estimate
3-4 days" \
  --label "phase-2,cli,feature" \
  --milestone "Phase 2 - Search Interface"

gh issue create \
  --title "Phase 2.2: Implement recommend command" \
  --body "Implement \`recommend\` command to find similar episodes.

## Tasks
- [ ] Implement \`SearchEngine.get_recommendations(episode_id, limit)\`
- [ ] Handle episode lookup in vector store
- [ ] Find N nearest neighbors
- [ ] Return ranked results with metadata
- [ ] Add optional filtering (by podcast, date range)
- [ ] CLI command with proper options
- [ ] Test with various episode IDs

## Acceptance Criteria
- Command returns relevant similar episodes
- Results are ranked by similarity
- Handles missing episode IDs gracefully
- Tests pass

## Effort Estimate
3-4 days" \
  --label "phase-2,cli,feature" \
  --milestone "Phase 2 - Search Interface"

gh issue create \
  --title "Phase 2.3: Data ingestion - DuckDB to vector store pipeline" \
  --body "Create scripts to ingest P³ podcast data and populate vector store.

## Tasks
- [ ] Create \`scripts/ingest_from_duckdb.py\` - read P³ database
- [ ] Handle connection to P³ DuckDB
- [ ] Query episodes and transcripts
- [ ] Validate and clean data
- [ ] Error handling for malformed entries
- [ ] Batch processing support

## Acceptance Criteria
- Script successfully reads P³ database
- Handles all data gracefully (no crashes)
- Creates ingestion report (count, errors, duration)
- Integrates with embedding pipeline

## Effort Estimate
4-5 days" \
  --label "phase-2,data-ingestion,infrastructure" \
  --milestone "Phase 2 - Search Interface"

gh issue create \
  --title "Phase 2.4: Transcript chunking strategy implementation" \
  --body "Implement sliding window chunking for long transcripts.

## Tasks
- [ ] Design chunking strategy (window size, overlap)
- [ ] Implement \`scripts/chunk_transcripts.py\`
- [ ] Handle edge cases (very long, very short transcripts)
- [ ] Preserve episode/podcast metadata in chunks
- [ ] Test with various transcript lengths
- [ ] Document strategy rationale

## Acceptance Criteria
- Chunks properly preserve context
- Metadata is accurate for all chunks
- Edge cases handled correctly
- Documentation explains approach

## Effort Estimate
2-3 days" \
  --label "phase-2,data-ingestion,infrastructure" \
  --milestone "Phase 2 - Search Interface"

gh issue create \
  --title "Phase 2.5: Complete embedding pipeline script" \
  --body "Create script to generate embeddings and populate vector store.

## Tasks
- [ ] Create \`scripts/embed_and_store.py\`
- [ ] Batch embedding generation with progress
- [ ] Store embeddings in LanceDB
- [ ] Resume capability for interrupted runs
- [ ] Error handling and logging
- [ ] Performance reporting

## Acceptance Criteria
- Script processes all chunks successfully
- Can resume from interruptions
- Reports progress and statistics
- Integration test passes

## Effort Estimate
3-4 days" \
  --label "phase-2,data-ingestion,infrastructure" \
  --milestone "Phase 2 - Search Interface"

gh issue create \
  --title "Phase 2.6: CLI integration tests & documentation" \
  --body "Test CLI thoroughly and document usage.

## Tasks
- [ ] Test \`search\` command with various queries
- [ ] Test \`recommend\` command with episode IDs
- [ ] Test error handling (invalid inputs)
- [ ] Test output formatting options
- [ ] Create \`docs/USAGE.md\` with examples
- [ ] Create \`docs/DATA_INGESTION.md\` guide
- [ ] Add CLI examples to README

## Acceptance Criteria
- All CLI tests pass (≥80% coverage)
- Documentation is clear with examples
- Troubleshooting section included
- New users can follow guide successfully

## Effort Estimate
2-3 days" \
  --label "phase-2,testing,documentation" \
  --milestone "Phase 2 - Search Interface"

# Phase 3 Issues
echo "📋 Creating Phase 3 issues..."

gh issue create \
  --title "Phase 3.1: Recommendation engine implementation" \
  --body "Implement content-based recommendation system.

## Tasks
- [ ] Implement \`SearchEngine.get_recommendations()\` fully
- [ ] Support hybrid recommendations (multiple episodes)
- [ ] Add optional filtering (podcast, date range)
- [ ] Return diverse results
- [ ] Test with various inputs

## Acceptance Criteria
- Recommendations are relevant
- Results are diverse
- Filtering works correctly
- Tests pass

## Effort Estimate
3-4 days" \
  --label "phase-3,feature,ml" \
  --milestone "Phase 3 - Advanced Features"

gh issue create \
  --title "Phase 3.2: Exploratory analysis Jupyter notebook" \
  --body "Create comprehensive Jupyter notebook for data exploration.

## Tasks
- [ ] Load embeddings and explore distribution
- [ ] Visualize embedding space (t-SNE, UMAP)
- [ ] Analyze clustering patterns
- [ ] Show query examples and results
- [ ] Demonstrate recommendations
- [ ] Performance analysis
- [ ] Document findings

## Acceptance Criteria
- Notebook runs without errors
- Visualizations are informative
- Examples show system capabilities
- Well-documented with markdown

## Effort Estimate
3-4 days" \
  --label "phase-3,documentation,analysis" \
  --milestone "Phase 3 - Advanced Features"

gh issue create \
  --title "Phase 3.3: Episode clustering & topic analysis" \
  --body "Implement clustering to identify topics and patterns.

## Tasks
- [ ] Implement K-means clustering
- [ ] Implement hierarchical clustering
- [ ] Generate topic summaries
- [ ] Identify outlier episodes
- [ ] Visualize clustering results
- [ ] Document findings

## Acceptance Criteria
- Clusters are meaningful
- Visualizations are clear
- Analysis is documented
- Code is tested

## Effort Estimate
3-4 days" \
  --label "phase-3,ml,analysis" \
  --milestone "Phase 3 - Advanced Features"

gh issue create \
  --title "Phase 3.4: Quality metrics & evaluation framework" \
  --body "Define and implement quality evaluation metrics.

## Tasks
- [ ] Define relevance metric (human evaluation)
- [ ] Define coverage metric
- [ ] Define diversity metric
- [ ] Create evaluation dataset
- [ ] Implement automated tests
- [ ] Document metrics in \`docs/EVALUATION.md\`

## Acceptance Criteria
- Metrics are well-defined
- Evaluation framework is implemented
- Results documented
- Tests pass

## Effort Estimate
2-3 days" \
  --label "phase-3,testing,analysis" \
  --milestone "Phase 3 - Advanced Features"

gh issue create \
  --title "Phase 3.5: Performance optimization & caching" \
  --body "Optimize search latency and implement caching.

## Tasks
- [ ] Profile search bottlenecks
- [ ] Implement query result caching
- [ ] Optimize vector store indexing
- [ ] Batch search support
- [ ] Memory optimization
- [ ] Target: <500ms per query

## Acceptance Criteria
- Search latency <500ms (target)
- Caching reduces repeated queries
- Benchmarks show improvement
- Performance regression tests added

## Effort Estimate
3-4 days" \
  --label "phase-3,performance,optimization" \
  --milestone "Phase 3 - Advanced Features"

# Phase 4 Issues
echo "📋 Creating Phase 4 issues..."

gh issue create \
  --title "Phase 4.1: Streamlit web application" \
  --body "Create interactive web interface using Streamlit.

## Tasks
- [ ] Create \`apps/streamlit_app.py\`
- [ ] Implement search interface
- [ ] Implement recommendations view
- [ ] Add trending topics section
- [ ] Add statistics dashboard
- [ ] Settings/configuration page
- [ ] Export functionality (CSV/JSON)
- [ ] Session state management

## Acceptance Criteria
- App launches successfully
- All features work correctly
- UI is responsive and intuitive
- Export works properly

## Effort Estimate
4-5 days" \
  --label "phase-4,ui,deployment" \
  --milestone "Phase 4 - Polish & Release"

gh issue create \
  --title "Phase 4.2: Optional REST API with FastAPI" \
  --body "Create REST API for programmatic access (optional).

## Tasks
- [ ] Create \`apps/api.py\` with FastAPI
- [ ] Implement \`/search\` endpoint
- [ ] Implement \`/recommend/:episode_id\` endpoint
- [ ] Implement \`/episodes\` endpoint
- [ ] Implement \`/stats\` endpoint
- [ ] Add authentication (API key)
- [ ] Rate limiting
- [ ] OpenAPI documentation

## Acceptance Criteria
- All endpoints work correctly
- API documentation is complete
- Authentication works
- Tests pass

## Effort Estimate
3-4 days" \
  --label "phase-4,api,deployment" \
  --milestone "Phase 4 - Polish & Release"

gh issue create \
  --title "Phase 4.3: Comprehensive documentation" \
  --body "Complete all documentation for release.

## Tasks
- [ ] Update README with full feature overview
- [ ] Create \`docs/ARCHITECTURE.md\` with diagrams
- [ ] Create \`docs/API.md\` with endpoint details
- [ ] Create \`docs/DEPLOYMENT.md\` with installation steps
- [ ] Create \`docs/CONTRIBUTING.md\` for developers
- [ ] Create \`docs/ADR.md\` (Architecture Decision Records)
- [ ] Add examples throughout

## Acceptance Criteria
- Documentation is complete and clear
- All features are explained
- Examples are provided
- No broken links

## Effort Estimate
2-3 days" \
  --label "phase-4,documentation,deployment" \
  --milestone "Phase 4 - Polish & Release"

gh issue create \
  --title "Phase 4.4: Docker containerization (optional)" \
  --body "Create Docker setup for consistent deployment.

## Tasks
- [ ] Create \`Dockerfile\` for application
- [ ] Optimize image size
- [ ] Create \`docker-compose.yml\` (optional multi-service)
- [ ] Document Docker usage in \`docs/DOCKER.md\`
- [ ] Test image build and run

## Acceptance Criteria
- Docker image builds successfully
- Container runs all features
- Instructions are clear

## Effort Estimate
1-2 days" \
  --label "phase-4,deployment,infrastructure" \
  --milestone "Phase 4 - Polish & Release"

gh issue create \
  --title "Phase 4.5: Release preparation & QA" \
  --body "Final testing and release preparation.

## Tasks
- [ ] End-to-end testing of all features
- [ ] Load testing (concurrent queries)
- [ ] Edge case testing
- [ ] Security review (no secrets in code)
- [ ] Cross-platform testing (macOS, Linux)
- [ ] Version bump to v0.1.0
- [ ] Create \`CHANGELOG.md\`
- [ ] Prepare GitHub release notes

## Acceptance Criteria
- All tests pass
- No security issues
- Changelog is comprehensive
- Release ready for publication

## Effort Estimate
2-3 days" \
  --label "phase-4,testing,release" \
  --milestone "Phase 4 - Polish & Release"

echo ""
echo "✅ All 18 GitHub issues created successfully!"
echo ""
echo "📊 Summary:"
echo "  Phase 1: 4 issues"
echo "  Phase 2: 6 issues"
echo "  Phase 3: 5 issues"
echo "  Phase 4: 5 issues"
echo ""
echo "🎯 Next steps:"
echo "  1. Visit: https://github.com/$(gh repo view --json nameWithOwner -q)/issues"
echo "  2. Start with Phase 1 issues"
echo "  3. Track progress with milestones"
