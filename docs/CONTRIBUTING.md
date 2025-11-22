# Contributing Guide

Guidelines for contributing to Parakeet Semantic Search.

**Version**: 1.0.0
**Last Updated**: November 22, 2025

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Standards](#code-standards)
4. [Testing Guidelines](#testing-guidelines)
5. [Git Workflow](#git-workflow)
6. [Pull Request Process](#pull-request-process)
7. [Issue Triage](#issue-triage)
8. [Documentation](#documentation)

---

## Getting Started

### Prerequisites

- Python 3.12+
- Git
- Virtual environment (venv or conda)
- GitHub account

### Repository Setup

```bash
# Fork the repository on GitHub

# Clone your fork
git clone https://github.com/<your-username>/parakeet-semantic-search.git
cd parakeet-semantic-search

# Add upstream remote
git remote add upstream https://github.com/jpequegn/parakeet-semantic-search.git

# Verify remotes
git remote -v
# origin    https://github.com/<your-username>/... (fetch)
# origin    https://github.com/<your-username>/... (push)
# upstream  https://github.com/jpequegn/... (fetch)
# upstream  https://github.com/jpequegn/... (push)
```

---

## Development Setup

### Environment Setup

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install black ruff mypy pytest pytest-cov

# Install in editable mode
pip install -e .
```

### Verify Installation

```bash
# Check CLI works
parakeet-search --version

# Run a simple test
pytest tests/test_search.py::TestSemanticSearch::test_search_returns_results -v
```

---

## Code Standards

### Style Guide

**PEP 8 Compliance**:
- 88-character line limit (Black default)
- 4-space indentation
- Use type hints for all functions
- Docstrings for all public functions

**Formatting**:
```bash
# Format code with Black
black src/ tests/ apps/

# Check with Ruff
ruff check src/ tests/ apps/

# Type checking
mypy src/parakeet_search/
```

### Code Quality Checklist

- [ ] All functions have type hints
- [ ] All public functions have docstrings
- [ ] No unused imports
- [ ] No hardcoded values (use constants)
- [ ] Code follows Black formatting
- [ ] No Ruff violations
- [ ] MyPy type checking passes
- [ ] Tests cover new functionality

### Example Code

```python
"""Module docstring explaining purpose."""

from typing import Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Represents a search result.

    Attributes:
        episode_id: Unique episode identifier
        similarity: Similarity score (0-1)
        title: Episode title
    """
    episode_id: str
    similarity: float
    title: str

def search_episodes(
    query: str,
    limit: int = 10,
    threshold: float = 0.0,
) -> List[SearchResult]:
    """Search for similar episodes.

    Performs semantic search across episode transcripts and returns
    results ranked by similarity score.

    Args:
        query: Natural language search query
        limit: Maximum number of results (1-100)
        threshold: Minimum similarity threshold (0-1)

    Returns:
        List of SearchResult objects ranked by similarity

    Raises:
        ValueError: If parameters out of valid range
        DatabaseError: If database connection fails

    Examples:
        >>> results = search_episodes("machine learning")
        >>> len(results)
        10
        >>> results[0].similarity
        0.95
    """
    if not 1 <= limit <= 100:
        raise ValueError(f"limit must be 1-100, got {limit}")

    logger.info(f"Searching for: {query}")

    # Implementation here
    return []
```

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `SearchEngine`, `CachingWrapper` |
| Functions | snake_case | `search_episodes`, `get_recommendations` |
| Constants | UPPER_SNAKE_CASE | `MAX_CACHE_SIZE`, `DEFAULT_LIMIT` |
| Private | Leading underscore | `_internal_method`, `_private_var` |
| Modules | lowercase with underscores | `search.py`, `optimization.py` |

---

## Testing Guidelines

### Test Structure

```
tests/
├── test_search.py           # Search functionality
├── test_optimization.py     # Caching & performance
├── test_api.py              # REST API endpoints
├── test_evaluation.py       # Quality metrics
├── conftest.py              # Shared fixtures
└── fixtures/                # Test data
    ├── test_episodes.json
    └── sample_embeddings.npy
```

### Writing Tests

```python
import pytest
from parakeet_search.search import SearchEngine

class TestSemanticSearch:
    """Tests for semantic search functionality."""

    @pytest.fixture
    def engine(self):
        """Create test search engine."""
        return SearchEngine()

    def test_search_returns_results(self, engine):
        """Test that search returns valid results."""
        results = engine.search("test query", limit=5)

        assert len(results) <= 5
        assert all("similarity" in r for r in results)
        assert all(0 <= r["similarity"] <= 1 for r in results)

    def test_search_empty_query_raises_error(self, engine):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError):
            engine.search("", limit=10)

    @pytest.mark.parametrize("limit", [1, 5, 10, 50])
    def test_search_limit_respected(self, engine, limit):
        """Test that search respects limit parameter."""
        results = engine.search("test", limit=limit)
        assert len(results) <= limit
```

### Test Execution

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_search.py -v

# Run specific test class
pytest tests/test_search.py::TestSemanticSearch -v

# Run specific test
pytest tests/test_search.py::TestSemanticSearch::test_search_returns_results -v

# Run with coverage
pytest tests/ --cov=parakeet_search --cov-report=html

# Run only fast tests
pytest tests/ -m "not slow" -v

# Run with debugging
pytest tests/ -v --pdb  # Drop to debugger on failure
```

### Test Coverage Requirements

- **Minimum**: 80% line coverage
- **Target**: 90%+ coverage
- All public APIs must be tested
- All error paths must be tested

---

## Git Workflow

### Branch Naming

```
feature/<issue-number>-<description>      # New features
bugfix/<issue-number>-<description>       # Bug fixes
docs/<issue-number>-<description>         # Documentation
refactor/<issue-number>-<description>     # Refactoring
test/<issue-number>-<description>         # Tests
```

**Examples**:
```
feature/17-rest-api
bugfix/42-fix-cache-ttl
docs/18-deployment-guide
refactor/25-optimize-search
test/30-add-api-tests
```

### Creating a Feature Branch

```bash
# Update local main
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/18-documentation

# Make changes, commit, and push
git add .
git commit -m "Clear commit message"
git push origin feature/18-documentation
```

### Commit Messages

**Format**:
```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Build/tooling

**Example**:
```
feat: Add hybrid recommendations endpoint

Implement /recommendations/hybrid endpoint that combines
multiple episode embeddings for collective recommendations.

- Add HybridRecommendationRequest model
- Add endpoint handler with validation
- Add comprehensive tests
- Update API documentation

Closes #17
```

### Code Review Checklist

- [ ] Code follows style guide
- [ ] All tests pass
- [ ] Type hints present
- [ ] Docstrings complete
- [ ] No breaking changes
- [ ] Documentation updated
- [ ] Commit messages clear
- [ ] No debug code/prints

---

## Pull Request Process

### Before Creating PR

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests locally**:
   ```bash
   pytest tests/ -v
   pytest tests/ --cov=parakeet_search
   ```

3. **Check code quality**:
   ```bash
   black src/ tests/ apps/
   ruff check src/ tests/ apps/
   mypy src/parakeet_search/
   ```

4. **Verify no conflicts**:
   ```bash
   git status
   git diff upstream/main
   ```

### Creating the PR

1. **Push to your fork**:
   ```bash
   git push origin feature/18-documentation
   ```

2. **Open PR on GitHub**:
   - Go to https://github.com/jpequegn/parakeet-semantic-search/pull/new
   - Select your branch
   - Fill in PR template

3. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] New feature
   - [ ] Bug fix
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing Done
   - [ ] Unit tests added
   - [ ] Integration tests added
   - [ ] Manual testing done

   ## Checklist
   - [ ] Code follows style guide
   - [ ] Tests added and passing
   - [ ] Documentation updated
   - [ ] No breaking changes

   Closes #18
   ```

### PR Review Process

1. **Automated checks**:
   - GitHub Actions run tests
   - Code quality checks
   - Coverage reports

2. **Code review**:
   - Maintainer reviews code
   - Provides feedback/suggestions
   - Requests changes if needed

3. **Approval & Merge**:
   - PR approved after review
   - Squash commits and merge
   - Delete feature branch

---

## Issue Triage

### Creating Issues

Use appropriate template:
- **Feature**: What, Why, Example usage
- **Bug**: Description, Steps, Expected vs. Actual
- **Documentation**: What's missing, where

### Labels

```
priority/high          # Critical issues
priority/medium        # Important features
priority/low           # Nice-to-have
type/bug               # Defects
type/feature           # New features
type/documentation     # Doc updates
type/question          # Questions/discussions
phase-1                # Core infrastructure
phase-2                # Advanced features
phase-3                # Analytics
phase-4                # Deployment
```

---

## Documentation

### Documentation Standards

- All public APIs documented
- Examples for common use cases
- Clear parameter descriptions
- Return value documentation
- Exception documentation

### Docstring Format

```python
def function(param1: str, param2: int = 10) -> List[str]:
    """Brief one-line description.

    Longer description if needed, explaining:
    - What the function does
    - How to use it
    - Important notes

    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)

    Returns:
        Description of return value

    Raises:
        ValueError: When param2 is negative
        TypeError: When param1 is not a string

    Examples:
        >>> result = function("test")
        >>> len(result)
        0
    """
    pass
```

### Updating Documentation

When making changes:
1. Update relevant docstrings
2. Update README.md if needed
3. Update docs/ARCHITECTURE.md if structural
4. Add examples to docs/
5. Update changelog if releasing

---

## Development Tools

### Recommended IDE Setup

**Visual Studio Code**:
```json
{
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.python",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

**PyCharm**:
- Configure Black as formatter
- Enable Ruff inspection
- Set Python 3.12 as interpreter
- Run tests through pytest

### Useful Commands

```bash
# Format all code
black src/ tests/ apps/

# Check style
ruff check src/ tests/ apps/ --fix

# Type checking
mypy src/parakeet_search/ --strict

# Run tests with coverage
pytest tests/ --cov=parakeet_search --cov-report=term --cov-report=html

# Profile performance
python -m cProfile -s cumtime -m pytest tests/test_search.py

# Generate documentation
pdoc src/parakeet_search --html -o docs/api
```

---

## Code Review Guidelines

### As an Author
- Respond to all feedback
- Make requested changes
- Re-request review when done
- Keep PRs focused and small
- Add context and explanation

### As a Reviewer
- Be constructive and respectful
- Suggest improvements
- Ask questions if unclear
- Approve when satisfied
- Acknowledge good work

---

## Community

- **Issues**: [GitHub Issues](https://github.com/jpequegn/parakeet-semantic-search/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jpequegn/parakeet-semantic-search/discussions)
- **Email**: julien@example.com

---

## Code of Conduct

Be respectful and inclusive. We welcome all contributors and will not tolerate harassment.

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Last Updated**: November 22, 2025
**Status**: Active Development
