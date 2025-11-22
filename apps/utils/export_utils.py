"""Export utility functions for Streamlit app."""

import json
import csv
from io import StringIO, BytesIO
from typing import List, Dict
from datetime import datetime


def export_to_csv(results: List[Dict], filename: str = "results.csv") -> bytes:
    """Export results to CSV format.

    Args:
        results: List of result dictionaries
        filename: Output filename

    Returns:
        CSV data as bytes
    """
    if not results:
        return b""

    output = StringIO()
    fieldnames = list(results[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerows(results)

    return output.getvalue().encode("utf-8")


def export_to_json(
    results: List[Dict],
    metadata: Dict = None,
    filename: str = "results.json",
) -> bytes:
    """Export results to JSON format.

    Args:
        results: List of result dictionaries
        metadata: Optional metadata to include
        filename: Output filename

    Returns:
        JSON data as bytes
    """
    data = {
        "metadata": {
            "export_date": datetime.now().isoformat(),
            "result_count": len(results),
            **(metadata or {}),
        },
        "results": results,
    }

    json_str = json.dumps(data, indent=2, default=str)
    return json_str.encode("utf-8")


def create_download_button_data(
    results: List[Dict],
    format: str = "csv",
    filename: str = None,
    metadata: Dict = None,
) -> tuple:
    """Create download data for Streamlit download_button.

    Args:
        results: List of results
        format: Export format ("csv" or "json")
        filename: Output filename
        metadata: Optional metadata

    Returns:
        Tuple of (data, filename, mime_type)
    """
    if format == "csv":
        data = export_to_csv(results)
        mime_type = "text/csv"
        default_filename = "parakeet_results.csv"
    elif format == "json":
        data = export_to_json(results, metadata=metadata)
        mime_type = "application/json"
        default_filename = "parakeet_results.json"
    else:
        raise ValueError(f"Unsupported format: {format}")

    final_filename = filename or default_filename
    return data, final_filename, mime_type


def format_metadata(
    query: str = None,
    result_count: int = 0,
    search_time: float = 0.0,
) -> Dict:
    """Create metadata dictionary for export.

    Args:
        query: Search query
        result_count: Number of results
        search_time: Search execution time in seconds

    Returns:
        Metadata dictionary
    """
    return {
        "query": query or "N/A",
        "result_count": result_count,
        "search_time_seconds": round(search_time, 3),
    }
