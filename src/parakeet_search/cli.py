"""Command-line interface for semantic search."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any

import click
from click import ClickException

from .search import SearchEngine
from .models import SearchResult


# Custom exception class for CLI errors
class CLIError(ClickException):
    """Custom CLI error with better formatting."""

    def show(self):
        click.secho(f"Error: {self.format_message()}", fg="red", err=True)


@click.group()
def cli():
    """Parakeet Semantic Search CLI.

    Semantic search and discovery engine for podcast transcripts.
    Use 'parakeet-search search <query>' to search episodes.
    """
    pass


def format_table(results: List[Dict[str, Any]]) -> str:
    """Format results as an ASCII table.

    Args:
        results: List of search results

    Returns:
        Formatted table string
    """
    if not results:
        return "No results found."

    # Calculate column widths
    headers = ["#", "Episode", "Podcast", "Relevance"]
    col_widths = [3, 35, 30, 12]

    # Create header
    header_line = " | ".join(
        f"{h:<{w}}" for h, w in zip(headers, col_widths)
    )
    separator = "-+-".join("-" * w for w in col_widths)

    lines = [header_line, separator]

    # Add rows
    for i, result in enumerate(results, 1):
        episode_title = (
            result.get("episode_title", "Unknown")[:col_widths[1] - 2]
        )
        podcast_title = (
            result.get("podcast_title", "Unknown")[:col_widths[2] - 2]
        )
        relevance = f"{100 * (1 - result.get('_distance', 1)):.0f}%"

        row = " | ".join(
            f"{str(v):<{w}}"
            for v, w in zip([i, episode_title, podcast_title, relevance], col_widths)
        )
        lines.append(row)

    return "\n".join(lines)


def format_markdown(results: List[Dict[str, Any]]) -> str:
    """Format results as Markdown.

    Args:
        results: List of search results

    Returns:
        Formatted markdown string
    """
    if not results:
        return "No results found."

    lines = ["## Search Results\n"]
    for i, result in enumerate(results, 1):
        relevance = 100 * (1 - result.get("_distance", 1))
        lines.append(f"### {i}. {result.get('episode_title', 'Unknown')}")
        lines.append(f"**Podcast**: {result.get('podcast_title', 'Unknown')}")
        lines.append(f"**Relevance**: {relevance:.1f}%")
        if result.get("text"):
            excerpt = result.get("text", "")[:200]
            lines.append(f"**Excerpt**: {excerpt}...")
        lines.append("")

    return "\n".join(lines)


def format_json(results: List[Dict[str, Any]]) -> str:
    """Format results as JSON.

    Args:
        results: List of search results

    Returns:
        Formatted JSON string
    """
    formatted_results = []
    for result in results:
        formatted_results.append({
            "episode_id": result.get("episode_id"),
            "episode_title": result.get("episode_title"),
            "podcast_title": result.get("podcast_title"),
            "relevance": 100 * (1 - result.get("_distance", 1)),
            "distance": result.get("_distance"),
        })

    return json.dumps(formatted_results, indent=2)


@cli.command()
@click.argument("query")
@click.option(
    "--limit",
    default=10,
    type=int,
    help="Number of results to return (default: 10)",
    show_default=True,
)
@click.option(
    "--threshold",
    default=None,
    type=float,
    help="Minimum similarity score (0-1)",
)
@click.option(
    "--format",
    type=click.Choice(["table", "json", "markdown"]),
    default="table",
    help="Output format (default: table)",
    show_default=True,
)
@click.option(
    "--save-results",
    type=click.Path(),
    default=None,
    help="Save results to file (json/md auto-detected by extension)",
)
def search(
    query: str,
    limit: int,
    threshold: float,
    format: str,
    save_results: str,
):
    """Search for episodes matching the query.

    Examples:
        parakeet-search search "machine learning"
        parakeet-search search "AI" --limit 20 --format json
        parakeet-search search "deep learning" --threshold 0.5 --save-results results.json
    """
    # Validate options
    if limit < 1 or limit > 1000:
        raise CLIError("--limit must be between 1 and 1000")

    if threshold is not None and (threshold < 0 or threshold > 1):
        raise CLIError("--threshold must be between 0 and 1")

    try:
        # Show progress
        with click.progressbar(
            length=100,
            label=f"Searching for '{query}'",
            show_pos=True,
        ) as bar:
            # Initialize search engine
            engine = SearchEngine()
            bar.update(20)

            # Perform search
            results = engine.search(query, limit=limit, threshold=threshold)
            bar.update(80)

    except Exception as e:
        raise CLIError(f"Search failed: {str(e)}")

    # Format results
    if format == "json":
        output = format_json(results)
    elif format == "markdown":
        output = format_markdown(results)
    else:  # table
        output = format_table(results)

    # Display results
    click.echo()
    if not results:
        click.secho("No results found.", fg="yellow")
    else:
        click.secho(
            f"Found {len(results)} result{'s' if len(results) != 1 else ''}",
            fg="green",
        )
        click.echo()
        click.echo(output)

    # Save to file if requested
    if save_results:
        try:
            output_path = Path(save_results)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Detect format from file extension if not explicitly set
            if format == "table":
                if save_results.endswith(".json"):
                    output = format_json(results)
                elif save_results.endswith(".md"):
                    output = format_markdown(results)

            with open(output_path, "w") as f:
                f.write(output)

            click.secho(
                f"\n✓ Results saved to {output_path}",
                fg="green",
            )
        except Exception as e:
            raise CLIError(f"Failed to save results: {str(e)}")


@cli.command()
@click.option(
    "--episode-id",
    required=True,
    help="Episode ID to find recommendations for",
)
@click.option(
    "--limit",
    default=5,
    type=int,
    help="Number of recommendations (default: 5)",
    show_default=True,
)
@click.option(
    "--podcast-id",
    default=None,
    help="Filter recommendations by podcast ID (optional)",
)
@click.option(
    "--format",
    type=click.Choice(["table", "json", "markdown"]),
    default="table",
    help="Output format (default: table)",
    show_default=True,
)
@click.option(
    "--save-results",
    type=click.Path(),
    default=None,
    help="Save results to file (json/md auto-detected by extension)",
)
def recommend(
    episode_id: str,
    limit: int,
    podcast_id: str,
    format: str,
    save_results: str,
):
    """Get recommendations for an episode.

    Examples:
        parakeet-search recommend --episode-id ep_001
        parakeet-search recommend --episode-id ep_001 --limit 10 --format json
        parakeet-search recommend --episode-id ep_001 --podcast-id pod_001 --save-results recommendations.json
    """
    # Validate options
    if limit < 1 or limit > 100:
        raise CLIError("--limit must be between 1 and 100")

    try:
        # Show progress
        with click.progressbar(
            length=100,
            label=f"Finding recommendations for episode '{episode_id}'",
            show_pos=True,
        ) as bar:
            # Initialize search engine
            engine = SearchEngine()
            bar.update(30)

            # Get recommendations
            results = engine.get_recommendations(
                episode_id=episode_id,
                limit=limit,
                podcast_id=podcast_id,
            )
            bar.update(70)

    except ValueError as e:
        raise CLIError(f"Invalid episode: {str(e)}")
    except RuntimeError as e:
        raise CLIError(f"Search engine error: {str(e)}")
    except Exception as e:
        raise CLIError(f"Recommendation failed: {str(e)}")

    # Format results
    if format == "json":
        output = format_json(results)
    elif format == "markdown":
        output = format_markdown(results)
    else:  # table
        output = format_table(results)

    # Display results
    click.echo()
    if not results:
        click.secho("No recommendations found.", fg="yellow")
    else:
        click.secho(
            f"Found {len(results)} recommendation{'s' if len(results) != 1 else ''}",
            fg="green",
        )
        click.echo()
        click.echo(output)

    # Save to file if requested
    if save_results:
        try:
            output_path = Path(save_results)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Detect format from file extension if not explicitly set
            if format == "table":
                if save_results.endswith(".json"):
                    output = format_json(results)
                elif save_results.endswith(".md"):
                    output = format_markdown(results)

            with open(output_path, "w") as f:
                f.write(output)

            click.secho(
                f"\n✓ Results saved to {output_path}",
                fg="green",
            )
        except Exception as e:
            raise CLIError(f"Failed to save results: {str(e)}")


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
