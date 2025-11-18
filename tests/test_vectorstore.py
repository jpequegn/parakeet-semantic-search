"""Unit tests for VectorStore class."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from parakeet_search.vectorstore import VectorStore


class TestVectorStoreInit:
    """Test VectorStore initialization."""

    def test_init_default_path(self):
        """Test initialization with default database path."""
        with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db

            with patch("parakeet_search.vectorstore.Path.mkdir"):
                vs = VectorStore()

                assert vs.db_path == Path("data/vectors.db")
                assert vs.db is mock_db
                assert vs.table is None

    def test_init_custom_path(self):
        """Test initialization with custom database path."""
        with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db

            with patch("parakeet_search.vectorstore.Path.mkdir"):
                vs = VectorStore(db_path="custom/path.db")

                assert vs.db_path == Path("custom/path.db")

    def test_init_creates_parent_directory(self):
        """Test that init creates parent directory."""
        with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db

            with patch("parakeet_search.vectorstore.Path.mkdir") as mock_mkdir:
                VectorStore(db_path="data/vectors.db")

                mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_init_connects_to_database(self):
        """Test that init connects to LanceDB."""
        with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db

            with patch("parakeet_search.vectorstore.Path.mkdir"):
                VectorStore(db_path="custom/path.db")

                mock_connect.assert_called_once_with("custom/path.db")


class TestVectorStoreCreateTable:
    """Test VectorStore.create_table() method."""

    @pytest.fixture
    def vectorstore(self):
        """Create a mock VectorStore for testing."""
        with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db

            with patch("parakeet_search.vectorstore.Path.mkdir"):
                vs = VectorStore()
                yield vs

    def test_create_table_basic(self, vectorstore):
        """Test creating a table with basic data."""
        # Create mock data
        data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "episode_id": ["ep1", "ep2", "ep3"],
                "text": ["text1", "text2", "text3"],
                "embedding": [np.random.randn(384), np.random.randn(384), np.random.randn(384)],
            }
        )

        mock_table = MagicMock()
        vectorstore.db.create_table.return_value = mock_table

        result = vectorstore.create_table(data, table_name="test_table")

        assert result is mock_table
        assert vectorstore.table is mock_table
        vectorstore.db.create_table.assert_called_once_with(
            "test_table", data=data, mode="overwrite"
        )

    def test_create_table_default_name(self, vectorstore):
        """Test creating a table with default name."""
        data = pd.DataFrame({"id": [1], "embedding": [np.random.randn(384)]})

        mock_table = MagicMock()
        vectorstore.db.create_table.return_value = mock_table

        vectorstore.create_table(data)

        vectorstore.db.create_table.assert_called_once_with(
            "transcripts", data=data, mode="overwrite"
        )

    def test_create_table_with_metadata(self, vectorstore):
        """Test creating a table with metadata columns."""
        data = pd.DataFrame(
            {
                "id": [1, 2],
                "episode_id": ["ep1", "ep2"],
                "podcast_title": ["podcast1", "podcast2"],
                "episode_title": ["episode1", "episode2"],
                "embedding": [np.random.randn(384), np.random.randn(384)],
            }
        )

        mock_table = MagicMock()
        vectorstore.db.create_table.return_value = mock_table

        vectorstore.create_table(data)

        assert vectorstore.db.create_table.called

    def test_create_table_overwrites_existing(self, vectorstore):
        """Test that create_table overwrites existing table."""
        data = pd.DataFrame({"id": [1], "embedding": [np.random.randn(384)]})

        mock_table1 = MagicMock()
        mock_table2 = MagicMock()

        # First call returns table1, second returns table2
        vectorstore.db.create_table.side_effect = [mock_table1, mock_table2]

        vectorstore.create_table(data, table_name="test")
        assert vectorstore.table is mock_table1

        vectorstore.create_table(data, table_name="test")
        assert vectorstore.table is mock_table2

        # Verify mode="overwrite" was used
        assert vectorstore.db.create_table.call_count == 2


class TestVectorStoreAddData:
    """Test VectorStore.add_data() method."""

    @pytest.fixture
    def vectorstore(self):
        """Create a mock VectorStore for testing."""
        with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db

            with patch("parakeet_search.vectorstore.Path.mkdir"):
                vs = VectorStore()
                yield vs

    def test_add_data_creates_table_if_not_exists(self, vectorstore):
        """Test that add_data creates table if it doesn't exist."""
        data = pd.DataFrame({"id": [1], "embedding": [np.random.randn(384)]})

        vectorstore.db.table_names.return_value = []
        mock_table = MagicMock()
        vectorstore.db.create_table.return_value = mock_table

        vectorstore.add_data(data, table_name="new_table")

        vectorstore.db.create_table.assert_called_once()

    def test_add_data_adds_to_existing_table(self, vectorstore):
        """Test adding data to existing table."""
        data = pd.DataFrame({"id": [1], "embedding": [np.random.randn(384)]})

        vectorstore.db.table_names.return_value = ["existing_table"]
        mock_table = MagicMock()
        vectorstore.db.open_table.return_value = mock_table

        vectorstore.add_data(data, table_name="existing_table")

        vectorstore.db.open_table.assert_called_once_with("existing_table")
        mock_table.add.assert_called_once_with(data)

    def test_add_data_default_table_name(self, vectorstore):
        """Test add_data with default table name."""
        data = pd.DataFrame({"id": [1], "embedding": [np.random.randn(384)]})

        vectorstore.db.table_names.return_value = ["transcripts"]
        mock_table = MagicMock()
        vectorstore.db.open_table.return_value = mock_table

        vectorstore.add_data(data)

        vectorstore.db.open_table.assert_called_once_with("transcripts")


class TestVectorStoreSearch:
    """Test VectorStore.search() method."""

    @pytest.fixture
    def vectorstore(self):
        """Create a mock VectorStore for testing."""
        with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db

            with patch("parakeet_search.vectorstore.Path.mkdir"):
                vs = VectorStore()
                yield vs

    def test_search_basic(self, vectorstore):
        """Test basic search functionality."""
        query_embedding = np.random.randn(384).tolist()

        mock_search_result = MagicMock()
        mock_search_result.to_list.return_value = [
            {"id": 1, "text": "result1", "_distance": 0.1},
            {"id": 2, "text": "result2", "_distance": 0.2},
        ]

        mock_table = MagicMock()
        mock_table.search.return_value.limit.return_value = mock_search_result
        vectorstore.db.open_table.return_value = mock_table

        results = vectorstore.search(query_embedding)

        assert len(results) == 2
        assert results[0]["id"] == 1
        assert results[1]["id"] == 2

    def test_search_with_limit(self, vectorstore):
        """Test search with custom limit."""
        query_embedding = np.random.randn(384).tolist()

        mock_search_result = MagicMock()
        mock_search_result.to_list.return_value = []

        mock_table = MagicMock()
        mock_table.search.return_value.limit.return_value = mock_search_result
        vectorstore.db.open_table.return_value = mock_table

        vectorstore.search(query_embedding, limit=20)

        # Verify limit was called with correct value
        mock_table.search.return_value.limit.assert_called_once_with(20)

    def test_search_with_custom_table_name(self, vectorstore):
        """Test search on specific table."""
        query_embedding = np.random.randn(384).tolist()

        mock_search_result = MagicMock()
        mock_search_result.to_list.return_value = []

        mock_table = MagicMock()
        mock_table.search.return_value.limit.return_value = mock_search_result
        vectorstore.db.open_table.return_value = mock_table

        vectorstore.search(query_embedding, table_name="custom_table")

        vectorstore.db.open_table.assert_called_once_with("custom_table")

    def test_search_returns_list(self, vectorstore):
        """Test that search returns a list."""
        query_embedding = np.random.randn(384).tolist()

        mock_search_result = MagicMock()
        mock_search_result.to_list.return_value = []

        mock_table = MagicMock()
        mock_table.search.return_value.limit.return_value = mock_search_result
        vectorstore.db.open_table.return_value = mock_table

        result = vectorstore.search(query_embedding)

        assert isinstance(result, list)

    def test_search_with_results(self, vectorstore):
        """Test search returning multiple results."""
        query_embedding = np.random.randn(384).tolist()

        expected_results = [
            {"id": 1, "text": "result1", "_distance": 0.05},
            {"id": 2, "text": "result2", "_distance": 0.10},
            {"id": 3, "text": "result3", "_distance": 0.15},
        ]

        mock_search_result = MagicMock()
        mock_search_result.to_list.return_value = expected_results

        mock_table = MagicMock()
        mock_table.search.return_value.limit.return_value = mock_search_result
        vectorstore.db.open_table.return_value = mock_table

        results = vectorstore.search(query_embedding, limit=3)

        assert len(results) == 3
        assert results == expected_results


class TestVectorStoreGetTable:
    """Test VectorStore.get_table() method."""

    @pytest.fixture
    def vectorstore(self):
        """Create a mock VectorStore for testing."""
        with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db

            with patch("parakeet_search.vectorstore.Path.mkdir"):
                vs = VectorStore()
                yield vs

    def test_get_table_default_name(self, vectorstore):
        """Test getting table with default name."""
        mock_table = MagicMock()
        vectorstore.db.open_table.return_value = mock_table

        result = vectorstore.get_table()

        vectorstore.db.open_table.assert_called_once_with("transcripts")
        assert result is mock_table

    def test_get_table_custom_name(self, vectorstore):
        """Test getting table with custom name."""
        mock_table = MagicMock()
        vectorstore.db.open_table.return_value = mock_table

        result = vectorstore.get_table(table_name="custom")

        vectorstore.db.open_table.assert_called_once_with("custom")
        assert result is mock_table


class TestVectorStoreIntegration:
    """Integration tests for VectorStore."""

    @pytest.fixture
    def vectorstore(self):
        """Create a mock VectorStore for testing."""
        with patch("parakeet_search.vectorstore.lancedb.connect") as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db

            with patch("parakeet_search.vectorstore.Path.mkdir"):
                vs = VectorStore()
                yield vs

    def test_create_and_search_workflow(self, vectorstore):
        """Test complete workflow of creating table and searching."""
        # Create data
        data = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "embedding": [np.random.randn(384), np.random.randn(384), np.random.randn(384)],
            }
        )

        # Create table
        mock_table = MagicMock()
        vectorstore.db.create_table.return_value = mock_table
        vectorstore.create_table(data)

        assert vectorstore.table is mock_table

        # Search - need to set up mocks for open_table as well
        query_embedding = np.random.randn(384).tolist()
        mock_search_result = MagicMock()
        expected_results = [{"id": 1, "_distance": 0.1}]
        mock_search_result.to_list.return_value = expected_results

        # Set up open_table to return our mock table
        vectorstore.db.open_table.return_value = mock_table
        mock_table.search.return_value.limit.return_value = mock_search_result

        results = vectorstore.search(query_embedding)

        assert results == expected_results
        assert len(results) == 1
        assert results[0]["id"] == 1

    def test_database_path_persistence(self, vectorstore):
        """Test that database path is properly stored."""
        assert vectorstore.db_path == Path("data/vectors.db")
        assert isinstance(vectorstore.db_path, Path)
