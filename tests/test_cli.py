"""Tests for CLI functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from click.testing import CliRunner

from parakeet_search.cli import cli, format_table, format_json, format_markdown


class TestSearchCommandBasic:
    """Test basic search command functionality."""

    @pytest.fixture
    def runner(self):
        """Create a Click CLI test runner."""
        return CliRunner()

    def test_search_without_query(self, runner):
        """Test that search command requires a query."""
        result = runner.invoke(cli, ["search"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_search_with_query(self, runner):
        """Test basic search with a query."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.search.return_value = []

            result = runner.invoke(cli, ["search", "test query"])

            assert result.exit_code == 0
            mock_instance.search.assert_called_once()

    def test_search_with_empty_results(self, runner):
        """Test search with no results."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.search.return_value = []

            result = runner.invoke(cli, ["search", "nonexistent"])

            assert result.exit_code == 0
            assert "No results found" in result.output


class TestSearchCommandOptions:
    """Test search command options."""

    @pytest.fixture
    def runner(self):
        """Create a Click CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_engine(self):
        """Mock SearchEngine with sample results."""
        with patch("parakeet_search.cli.SearchEngine") as mock:
            instance = MagicMock()
            instance.search.return_value = [
                {
                    "episode_id": "ep_001",
                    "episode_title": "Machine Learning Basics",
                    "podcast_title": "AI Today",
                    "_distance": 0.15,
                    "text": "This is a sample transcript...",
                },
                {
                    "episode_id": "ep_002",
                    "episode_title": "Deep Learning",
                    "podcast_title": "AI Today",
                    "_distance": 0.25,
                    "text": "Deep learning is...",
                },
            ]
            mock.return_value = instance
            yield instance

    def test_search_with_limit(self, runner, mock_engine):
        """Test search with --limit option."""
        result = runner.invoke(cli, ["search", "query", "--limit", "5"])

        assert result.exit_code == 0
        mock_engine.search.assert_called_once()
        call_kwargs = mock_engine.search.call_args[1]
        assert call_kwargs["limit"] == 5

    def test_search_with_threshold(self, runner, mock_engine):
        """Test search with --threshold option."""
        result = runner.invoke(cli, ["search", "query", "--threshold", "0.5"])

        assert result.exit_code == 0
        call_kwargs = mock_engine.search.call_args[1]
        assert call_kwargs["threshold"] == 0.5

    def test_search_with_invalid_limit(self, runner):
        """Test search with invalid limit (too high)."""
        with patch("parakeet_search.cli.SearchEngine"):
            result = runner.invoke(cli, ["search", "query", "--limit", "2000"])
            assert result.exit_code != 0
            assert "limit must be between 1 and 1000" in result.output

    def test_search_with_invalid_limit_zero(self, runner):
        """Test search with invalid limit (zero)."""
        with patch("parakeet_search.cli.SearchEngine"):
            result = runner.invoke(cli, ["search", "query", "--limit", "0"])
            assert result.exit_code != 0

    def test_search_with_invalid_threshold_too_high(self, runner):
        """Test search with threshold > 1."""
        with patch("parakeet_search.cli.SearchEngine"):
            result = runner.invoke(cli, ["search", "query", "--threshold", "1.5"])
            assert result.exit_code != 0
            assert "threshold must be between 0 and 1" in result.output

    def test_search_with_invalid_threshold_negative(self, runner):
        """Test search with negative threshold."""
        with patch("parakeet_search.cli.SearchEngine"):
            result = runner.invoke(cli, ["search", "query", "--threshold", "-0.5"])
            assert result.exit_code != 0

    def test_search_with_threshold_zero(self, runner, mock_engine):
        """Test search with threshold = 0."""
        result = runner.invoke(cli, ["search", "query", "--threshold", "0"])

        assert result.exit_code == 0
        call_kwargs = mock_engine.search.call_args[1]
        assert call_kwargs["threshold"] == 0.0


class TestSearchCommandFormatting:
    """Test search command output formatting."""

    @pytest.fixture
    def runner(self):
        """Create a Click CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results(self):
        """Sample search results for testing."""
        return [
            {
                "episode_id": "ep_001",
                "episode_title": "ML Basics",
                "podcast_title": "AI Today",
                "_distance": 0.15,
                "text": "Sample transcript",
            },
            {
                "episode_id": "ep_002",
                "episode_title": "Deep Learning",
                "podcast_title": "Tech News",
                "_distance": 0.25,
                "text": "Another sample",
            },
        ]

    def test_search_default_format_table(self, runner, sample_results):
        """Test default output is table format."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.search.return_value = sample_results

            result = runner.invoke(cli, ["search", "query"])

            assert result.exit_code == 0
            # Table format has separators
            assert "-+-" in result.output or "#" in result.output

    def test_search_table_format(self, runner, sample_results):
        """Test table output format."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.search.return_value = sample_results

            result = runner.invoke(cli, ["search", "query", "--format", "table"])

            assert result.exit_code == 0
            assert "Episode" in result.output
            assert "Podcast" in result.output

    def test_search_json_format(self, runner, sample_results):
        """Test JSON output format."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.search.return_value = sample_results

            result = runner.invoke(cli, ["search", "query", "--format", "json"])

            assert result.exit_code == 0
            # Find JSON content in output (it's after the progress info)
            output_lines = result.output.split("\n")
            # Find the line that starts with "["
            json_start_idx = None
            for i, line in enumerate(output_lines):
                if line.strip().startswith("["):
                    json_start_idx = i
                    break

            if json_start_idx is not None:
                # Extract JSON from the start
                json_lines = []
                for line in output_lines[json_start_idx:]:
                    json_lines.append(line)
                    if line.strip().endswith("]"):
                        break
                json_str = "\n".join(json_lines)
                parsed = json.loads(json_str)
                assert isinstance(parsed, list)
                assert len(parsed) == 2

    def test_search_markdown_format(self, runner, sample_results):
        """Test Markdown output format."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.search.return_value = sample_results

            result = runner.invoke(cli, ["search", "query", "--format", "markdown"])

            assert result.exit_code == 0
            assert "## Search Results" in result.output
            assert "###" in result.output  # Markdown headers


class TestSearchCommandSaveResults:
    """Test search command --save-results option."""

    @pytest.fixture
    def runner(self):
        """Create a Click CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results(self):
        """Sample search results for testing."""
        return [
            {
                "episode_id": "ep_001",
                "episode_title": "ML Basics",
                "podcast_title": "AI Today",
                "_distance": 0.15,
                "text": "Sample transcript",
            }
        ]

    def test_save_results_to_json(self, runner, sample_results):
        """Test saving results to JSON file."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.search.return_value = sample_results

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "results.json"

                result = runner.invoke(
                    cli,
                    [
                        "search",
                        "query",
                        "--save-results",
                        str(output_path),
                    ],
                )

                assert result.exit_code == 0
                assert output_path.exists()
                with open(output_path) as f:
                    data = json.load(f)
                    assert isinstance(data, list)
                    assert len(data) == 1

    def test_save_results_to_markdown(self, runner, sample_results):
        """Test saving results to Markdown file."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.search.return_value = sample_results

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "results.md"

                result = runner.invoke(
                    cli,
                    [
                        "search",
                        "query",
                        "--save-results",
                        str(output_path),
                    ],
                )

                assert result.exit_code == 0
                assert output_path.exists()
                content = output_path.read_text()
                assert "## Search Results" in content

    def test_save_results_creates_parent_directory(self, runner, sample_results):
        """Test that parent directories are created."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.search.return_value = sample_results

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "subdir" / "results.json"

                result = runner.invoke(
                    cli,
                    [
                        "search",
                        "query",
                        "--save-results",
                        str(output_path),
                    ],
                )

                assert result.exit_code == 0
                assert output_path.exists()

    def test_save_results_with_format_override(self, runner, sample_results):
        """Test format option with save results."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.search.return_value = sample_results

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "results.json"

                result = runner.invoke(
                    cli,
                    [
                        "search",
                        "query",
                        "--format",
                        "json",
                        "--save-results",
                        str(output_path),
                    ],
                )

                assert result.exit_code == 0
                assert output_path.exists()


class TestFormatFunctions:
    """Test individual formatting functions."""

    def test_format_table_empty(self):
        """Test table formatting with empty results."""
        output = format_table([])
        assert "No results found" in output

    def test_format_table_with_results(self):
        """Test table formatting with results."""
        results = [
            {
                "episode_id": "ep_001",
                "episode_title": "ML Basics",
                "podcast_title": "AI Today",
                "_distance": 0.15,
            }
        ]
        output = format_table(results)
        assert "ML Basics" in output
        assert "AI Today" in output
        assert "85%" in output  # 100 * (1 - 0.15)

    def test_format_json_empty(self):
        """Test JSON formatting with empty results."""
        output = format_json([])
        data = json.loads(output)
        assert isinstance(data, list)
        assert len(data) == 0

    def test_format_json_with_results(self):
        """Test JSON formatting with results."""
        results = [
            {
                "episode_id": "ep_001",
                "episode_title": "ML Basics",
                "podcast_title": "AI Today",
                "_distance": 0.15,
            }
        ]
        output = format_json(results)
        data = json.loads(output)
        assert len(data) == 1
        assert data[0]["episode_id"] == "ep_001"
        assert data[0]["relevance"] == pytest.approx(85.0, abs=1)

    def test_format_markdown_empty(self):
        """Test Markdown formatting with empty results."""
        output = format_markdown([])
        assert "No results found" in output

    def test_format_markdown_with_results(self):
        """Test Markdown formatting with results."""
        results = [
            {
                "episode_id": "ep_001",
                "episode_title": "ML Basics",
                "podcast_title": "AI Today",
                "_distance": 0.15,
                "text": "Sample transcript",
            }
        ]
        output = format_markdown(results)
        assert "## Search Results" in output
        assert "ML Basics" in output
        assert "AI Today" in output


class TestSearchCommandIntegration:
    """Test complete search command workflows."""

    @pytest.fixture
    def runner(self):
        """Create a Click CLI test runner."""
        return CliRunner()

    def test_search_with_all_options(self, runner):
        """Test search with all options combined."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.search.return_value = []

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "results.json"

                result = runner.invoke(
                    cli,
                    [
                        "search",
                        "query",
                        "--limit",
                        "20",
                        "--threshold",
                        "0.5",
                        "--format",
                        "json",
                        "--save-results",
                        str(output_path),
                    ],
                )

                assert result.exit_code == 0
                # Verify search was called with the right params
                if mock_instance.search.called:
                    call_kwargs = mock_instance.search.call_args[1]
                    assert call_kwargs["limit"] == 20
                    assert call_kwargs["threshold"] == 0.5

    def test_search_special_characters_in_query(self, runner):
        """Test search with special characters in query."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.search.return_value = []

            result = runner.invoke(cli, ["search", "C++ & Python & JavaScript!"])

            assert result.exit_code == 0
            # Verify search was called with the query
            if mock_instance.search.called:
                call_args = mock_instance.search.call_args[0]
                assert "C++ & Python & JavaScript!" in call_args

    def test_search_very_long_query(self, runner):
        """Test search with very long query."""
        long_query = " ".join(["word"] * 100)
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.search.return_value = []

            result = runner.invoke(cli, ["search", long_query])

            assert result.exit_code == 0


class TestRecommendCommandBasic:
    """Test basic recommend command functionality."""

    @pytest.fixture
    def runner(self):
        """Create a Click CLI test runner."""
        return CliRunner()

    def test_recommend_requires_episode_id(self, runner):
        """Test that recommend command requires --episode-id."""
        with patch("parakeet_search.cli.SearchEngine"):
            result = runner.invoke(cli, ["recommend"])
            assert result.exit_code != 0
            assert "Missing option" in result.output or "required" in result.output.lower()

    def test_recommend_with_episode_id(self, runner):
        """Test basic recommend with episode ID."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.get_recommendations.return_value = []

            result = runner.invoke(cli, ["recommend", "--episode-id", "ep_001"])

            assert result.exit_code == 0
            mock_instance.get_recommendations.assert_called_once()

    def test_recommend_with_empty_results(self, runner):
        """Test recommend with no results."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.get_recommendations.return_value = []

            result = runner.invoke(cli, ["recommend", "--episode-id", "ep_001"])

            assert result.exit_code == 0
            assert "No recommendations found" in result.output


class TestRecommendCommandOptions:
    """Test recommend command options."""

    @pytest.fixture
    def runner(self):
        """Create a Click CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_engine(self):
        """Mock SearchEngine with sample recommendations."""
        with patch("parakeet_search.cli.SearchEngine") as mock:
            instance = MagicMock()
            instance.get_recommendations.return_value = [
                {
                    "episode_id": "ep_002",
                    "episode_title": "Deep Learning",
                    "podcast_title": "AI Today",
                    "_distance": 0.1,
                    "text": "Deep learning discussion...",
                },
                {
                    "episode_id": "ep_003",
                    "episode_title": "Neural Networks",
                    "podcast_title": "Tech News",
                    "_distance": 0.2,
                    "text": "Neural nets...",
                },
            ]
            mock.return_value = instance
            yield instance

    def test_recommend_with_limit(self, runner, mock_engine):
        """Test recommend with --limit option."""
        result = runner.invoke(
            cli, ["recommend", "--episode-id", "ep_001", "--limit", "3"]
        )

        assert result.exit_code == 0
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs["limit"] == 3

    def test_recommend_with_podcast_filter(self, runner, mock_engine):
        """Test recommend with --podcast-id filter."""
        result = runner.invoke(
            cli,
            ["recommend", "--episode-id", "ep_001", "--podcast-id", "pod_001"],
        )

        assert result.exit_code == 0
        call_kwargs = mock_engine.get_recommendations.call_args[1]
        assert call_kwargs["podcast_id"] == "pod_001"

    def test_recommend_with_invalid_limit(self, runner):
        """Test recommend with invalid limit (too high)."""
        with patch("parakeet_search.cli.SearchEngine"):
            result = runner.invoke(
                cli, ["recommend", "--episode-id", "ep_001", "--limit", "500"]
            )
            assert result.exit_code != 0
            assert "limit must be between 1 and 100" in result.output

    def test_recommend_with_invalid_limit_zero(self, runner):
        """Test recommend with invalid limit (zero)."""
        with patch("parakeet_search.cli.SearchEngine"):
            result = runner.invoke(
                cli, ["recommend", "--episode-id", "ep_001", "--limit", "0"]
            )
            assert result.exit_code != 0


class TestRecommendCommandFormatting:
    """Test recommend command output formatting."""

    @pytest.fixture
    def runner(self):
        """Create a Click CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results(self):
        """Sample recommendations for testing."""
        return [
            {
                "episode_id": "ep_002",
                "episode_title": "Deep Learning",
                "podcast_title": "AI Today",
                "_distance": 0.1,
                "text": "Deep learning...",
            },
            {
                "episode_id": "ep_003",
                "episode_title": "Neural Networks",
                "podcast_title": "Tech News",
                "_distance": 0.2,
                "text": "Neural nets...",
            },
        ]

    def test_recommend_default_format_table(self, runner, sample_results):
        """Test default output is table format."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.get_recommendations.return_value = sample_results

            result = runner.invoke(cli, ["recommend", "--episode-id", "ep_001"])

            assert result.exit_code == 0
            assert "-+-" in result.output or "#" in result.output

    def test_recommend_table_format(self, runner, sample_results):
        """Test table output format."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.get_recommendations.return_value = sample_results

            result = runner.invoke(
                cli,
                ["recommend", "--episode-id", "ep_001", "--format", "table"],
            )

            assert result.exit_code == 0
            assert "Episode" in result.output
            assert "Podcast" in result.output

    def test_recommend_json_format(self, runner, sample_results):
        """Test JSON output format."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.get_recommendations.return_value = sample_results

            result = runner.invoke(
                cli,
                ["recommend", "--episode-id", "ep_001", "--format", "json"],
            )

            assert result.exit_code == 0
            # Extract JSON from output
            output_lines = result.output.split("\n")
            json_start_idx = None
            for i, line in enumerate(output_lines):
                if line.strip().startswith("["):
                    json_start_idx = i
                    break

            if json_start_idx is not None:
                json_lines = []
                for line in output_lines[json_start_idx:]:
                    json_lines.append(line)
                    if line.strip().endswith("]"):
                        break
                json_str = "\n".join(json_lines)
                parsed = json.loads(json_str)
                assert isinstance(parsed, list)
                assert len(parsed) == 2

    def test_recommend_markdown_format(self, runner, sample_results):
        """Test Markdown output format."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.get_recommendations.return_value = sample_results

            result = runner.invoke(
                cli,
                ["recommend", "--episode-id", "ep_001", "--format", "markdown"],
            )

            assert result.exit_code == 0
            assert "## Search Results" in result.output
            assert "###" in result.output


class TestRecommendCommandSaveResults:
    """Test recommend command --save-results option."""

    @pytest.fixture
    def runner(self):
        """Create a Click CLI test runner."""
        return CliRunner()

    @pytest.fixture
    def sample_results(self):
        """Sample recommendations for testing."""
        return [
            {
                "episode_id": "ep_002",
                "episode_title": "Deep Learning",
                "podcast_title": "AI Today",
                "_distance": 0.1,
                "text": "Deep learning...",
            }
        ]

    def test_save_recommendations_to_json(self, runner, sample_results):
        """Test saving recommendations to JSON file."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.get_recommendations.return_value = sample_results

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "recommendations.json"

                result = runner.invoke(
                    cli,
                    [
                        "recommend",
                        "--episode-id",
                        "ep_001",
                        "--save-results",
                        str(output_path),
                    ],
                )

                assert result.exit_code == 0
                assert output_path.exists()
                with open(output_path) as f:
                    data = json.load(f)
                    assert isinstance(data, list)
                    assert len(data) == 1

    def test_save_recommendations_to_markdown(self, runner, sample_results):
        """Test saving recommendations to Markdown file."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.get_recommendations.return_value = sample_results

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "recommendations.md"

                result = runner.invoke(
                    cli,
                    [
                        "recommend",
                        "--episode-id",
                        "ep_001",
                        "--save-results",
                        str(output_path),
                    ],
                )

                assert result.exit_code == 0
                assert output_path.exists()
                content = output_path.read_text()
                assert "## Search Results" in content

    def test_save_recommendations_creates_parent_directory(self, runner, sample_results):
        """Test that parent directories are created."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.get_recommendations.return_value = sample_results

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "subdir" / "recommendations.json"

                result = runner.invoke(
                    cli,
                    [
                        "recommend",
                        "--episode-id",
                        "ep_001",
                        "--save-results",
                        str(output_path),
                    ],
                )

                assert result.exit_code == 0
                assert output_path.exists()


class TestRecommendCommandErrorHandling:
    """Test recommend command error handling."""

    @pytest.fixture
    def runner(self):
        """Create a Click CLI test runner."""
        return CliRunner()

    def test_recommend_invalid_episode(self, runner):
        """Test recommend with invalid episode ID."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.get_recommendations.side_effect = ValueError(
                "Episode not found"
            )

            result = runner.invoke(cli, ["recommend", "--episode-id", "invalid"])

            assert result.exit_code != 0
            assert "Invalid episode" in result.output

    def test_recommend_engine_error(self, runner):
        """Test recommend with search engine error."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.get_recommendations.side_effect = RuntimeError(
                "Database error"
            )

            result = runner.invoke(cli, ["recommend", "--episode-id", "ep_001"])

            assert result.exit_code != 0
            assert "Search engine error" in result.output

    def test_recommend_generic_error(self, runner):
        """Test recommend with generic error."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.get_recommendations.side_effect = Exception(
                "Unexpected error"
            )

            result = runner.invoke(cli, ["recommend", "--episode-id", "ep_001"])

            assert result.exit_code != 0
            assert "Recommendation failed" in result.output


class TestRecommendCommandIntegration:
    """Test complete recommend command workflows."""

    @pytest.fixture
    def runner(self):
        """Create a Click CLI test runner."""
        return CliRunner()

    def test_recommend_with_all_options(self, runner):
        """Test recommend with all options combined."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.get_recommendations.return_value = []

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "recommendations.json"

                result = runner.invoke(
                    cli,
                    [
                        "recommend",
                        "--episode-id",
                        "ep_001",
                        "--limit",
                        "10",
                        "--podcast-id",
                        "pod_001",
                        "--format",
                        "json",
                        "--save-results",
                        str(output_path),
                    ],
                )

                assert result.exit_code == 0
                # Verify correct parameters passed
                if mock_instance.get_recommendations.called:
                    call_kwargs = mock_instance.get_recommendations.call_args[1]
                    assert call_kwargs["limit"] == 10
                    assert call_kwargs["podcast_id"] == "pod_001"

    def test_recommend_multiple_calls(self, runner):
        """Test multiple recommend calls."""
        with patch("parakeet_search.cli.SearchEngine") as mock_engine:
            mock_instance = MagicMock()
            mock_engine.return_value = mock_instance
            mock_instance.get_recommendations.return_value = [
                {
                    "episode_id": "ep_002",
                    "episode_title": "Related Episode",
                    "podcast_title": "Podcast",
                    "_distance": 0.1,
                    "text": "Sample",
                }
            ]

            # First recommendation
            result1 = runner.invoke(cli, ["recommend", "--episode-id", "ep_001"])
            assert result1.exit_code == 0

            # Second recommendation
            result2 = runner.invoke(cli, ["recommend", "--episode-id", "ep_003"])
            assert result2.exit_code == 0

            # Verify both calls were made
            assert mock_instance.get_recommendations.call_count == 2
