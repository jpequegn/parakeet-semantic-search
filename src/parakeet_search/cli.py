"""Command-line interface for semantic search."""

import click
from .search import SearchEngine


@click.group()
def cli():
    """Parakeet Semantic Search CLI."""
    pass


@cli.command()
@click.argument("query")
@click.option("--limit", default=10, help="Number of results to return")
def search(query: str, limit: int):
    """Search for episodes matching the query."""
    engine = SearchEngine()
    results = engine.search(query, limit=limit)

    click.echo(f"\nSearching for: {query}\n")

    if not results:
        click.echo("No results found.")
        return

    for i, result in enumerate(results, 1):
        click.echo(f"{i}. Episode: {result.get('episode_title', 'Unknown')}")
        click.echo(f"   Podcast: {result.get('podcast_title', 'Unknown')}")
        click.echo(f"   Relevance: {100 * (1 - result.get('_distance', 1)):.1f}%")
        click.echo()


@cli.command()
@click.option("--episode-id", help="Episode ID for recommendations")
@click.option("--limit", default=5, help="Number of recommendations")
def recommend(episode_id: str, limit: int):
    """Get recommendations based on an episode."""
    click.echo(f"Recommendations for episode {episode_id}...")
    click.echo("(Coming in Phase 3)")


def main():
    """Entry point for CLI."""
    cli()


if __name__ == "__main__":
    main()
