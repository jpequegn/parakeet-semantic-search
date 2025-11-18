"""Vector store management using LanceDB."""

import lancedb
import pandas as pd
from pathlib import Path


class VectorStore:
    """Interface to LanceDB vector store."""

    def __init__(self, db_path: str = "data/vectors.db"):
        """Initialize vector store.

        Args:
            db_path: Path to LanceDB database directory
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self.table = None

    def create_table(self, data: pd.DataFrame, table_name: str = "transcripts"):
        """Create or overwrite a table in the vector store.

        Args:
            data: DataFrame with columns: id, episode_id, text, embedding, ...
            table_name: Name of the table to create
        """
        self.table = self.db.create_table(table_name, data=data, mode="overwrite")
        return self.table

    def add_data(self, data: pd.DataFrame, table_name: str = "transcripts"):
        """Add data to existing table.

        Args:
            data: DataFrame to add
            table_name: Name of the table
        """
        if table_name not in self.db.table_names():
            self.create_table(data, table_name)
        else:
            table = self.db.open_table(table_name)
            table.add(data)

    def search(self, query_embedding: list, table_name: str = "transcripts", limit: int = 10):
        """Search for similar vectors.

        Args:
            query_embedding: Query vector
            table_name: Name of the table to search
            limit: Number of results to return

        Returns:
            List of similar records
        """
        table = self.db.open_table(table_name)
        results = table.search(query_embedding).limit(limit).to_list()
        return results

    def get_table(self, table_name: str = "transcripts"):
        """Get reference to a table.

        Args:
            table_name: Name of the table

        Returns:
            LanceDB table object
        """
        return self.db.open_table(table_name)
