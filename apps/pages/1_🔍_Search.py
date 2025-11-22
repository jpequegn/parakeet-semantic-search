"""Search Interface - Streamlit Page."""

import streamlit as st
import time
from typing import List, Dict
from apps.utils.search_utils import SearchManager, format_results_table
from apps.utils.export_utils import create_download_button_data, format_metadata

# Configure page
st.set_page_config(
    page_title="Search - Parakeet",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Semantic Search")
st.markdown("Find episodes using natural language search")

# Initialize search manager
search_mgr = SearchManager()

# Sidebar Controls
with st.sidebar:
    st.subheader("Search Settings")

    limit = st.slider(
        "Results per page",
        min_value=1,
        max_value=50,
        value=10,
        help="Number of results to display",
    )

    threshold = st.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Minimum similarity score (0 = no filter)",
    )

    cache_stats = search_mgr.cache_stats()
    if cache_stats:
        st.divider()
        st.subheader("Cache Statistics")
        st.metric("Cache Size", f"{cache_stats.get('size', 0)}/{cache_stats.get('max_size', 0)}")
        st.metric("Hit Rate", f"{cache_stats.get('hit_rate', 0):.1%}")

        if st.button("Clear Cache", key="clear_cache"):
            search_mgr.clear_cache()
            st.success("Cache cleared!")

# Main Search Interface
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Enter your search query",
        placeholder="e.g., 'machine learning applications' or 'natural language processing'",
        help="Describe what you're looking for",
    )

with col2:
    search_button = st.button("🔍 Search", use_container_width=True)

# Initialize session state
if "last_query" not in st.session_state:
    st.session_state.last_query = None
if "last_results" not in st.session_state:
    st.session_state.last_results = []
if "search_history" not in st.session_state:
    st.session_state.search_history = []

# Execute search
results = []
search_time = 0

if search_button and query:
    with st.spinner("Searching..."):
        start_time = time.time()
        results = search_mgr.search(query, limit=limit, threshold=threshold)
        search_time = time.time() - start_time

    # Update session state
    st.session_state.last_query = query
    st.session_state.last_results = results
    if query not in st.session_state.search_history:
        st.session_state.search_history.insert(0, query)

# Display Results
if st.session_state.last_results:
    results_count = len(st.session_state.last_results)

    # Results summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Results Found", results_count)
    with col2:
        st.metric("Search Time", f"{search_time:.3f}s" if search_time > 0 else "Cached")
    with col3:
        st.metric("Query", st.session_state.last_query[:30] + "..." if len(st.session_state.last_query) > 30 else st.session_state.last_query)

    st.divider()

    # Results display
    if results_count > 0:
        # Format results for display
        formatted_results = format_results_table(st.session_state.last_results)

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Table View", "Detailed View", "Export"])

        with tab1:
            # Table view
            st.dataframe(
                formatted_results,
                use_container_width=True,
                column_config={
                    "similarity": st.column_config.ProgressColumn(
                        "Similarity",
                        min_value=0,
                        max_value=1,
                    ),
                    "distance": st.column_config.NumberColumn(
                        "Distance",
                        format="%.4f",
                    ),
                },
            )

        with tab2:
            # Detailed view
            for idx, result in enumerate(st.session_state.last_results, 1):
                with st.expander(
                    f"{idx}. {result.get('episode_title', 'Untitled')} "
                    f"({result.get('podcast_title', 'Unknown')})",
                    expanded=(idx == 1),
                ):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**Episode:** {result.get('episode_title', 'N/A')}")
                        st.markdown(f"**Podcast:** {result.get('podcast_title', 'N/A')}")
                        st.markdown(f"**Episode ID:** `{result.get('episode_id', 'N/A')}`")
                        st.markdown(f"**Podcast ID:** `{result.get('podcast_id', 'N/A')}`")

                    with col2:
                        similarity = 1 - result.get("_distance", 0)
                        st.metric("Similarity", f"{similarity:.1%}")

        with tab3:
            # Export options
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Export as CSV")
                csv_data, csv_filename, csv_mime = create_download_button_data(
                    formatted_results,
                    format="csv",
                    filename=f"search_{st.session_state.last_query.replace(' ', '_')}.csv",
                )

                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=csv_filename,
                    mime=csv_mime,
                    use_container_width=True,
                )

            with col2:
                st.subheader("Export as JSON")
                metadata = format_metadata(
                    query=st.session_state.last_query,
                    result_count=results_count,
                    search_time=search_time,
                )
                json_data, json_filename, json_mime = create_download_button_data(
                    formatted_results,
                    format="json",
                    filename=f"search_{st.session_state.last_query.replace(' ', '_')}.json",
                    metadata=metadata,
                )

                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=json_filename,
                    mime=json_mime,
                    use_container_width=True,
                )

    else:
        st.warning("No results found. Try a different search query.")

elif st.session_state.last_query and not search_button:
    # Show cached results
    if st.session_state.last_results:
        st.info(f"Showing cached results for: {st.session_state.last_query}")
        formatted_results = format_results_table(st.session_state.last_results)
        st.dataframe(formatted_results, use_container_width=True)

# Search History
if st.session_state.search_history:
    st.divider()
    st.subheader("Recent Searches")

    cols = st.columns(min(len(st.session_state.search_history), 5))
    for col, history_query in zip(cols, st.session_state.search_history[:5]):
        with col:
            if st.button(f"🔄 {history_query[:20]}...", key=f"history_{history_query}"):
                st.session_state.last_query = history_query
                with st.spinner("Searching..."):
                    start_time = time.time()
                    results = search_mgr.search(history_query, limit=limit, threshold=threshold)
                    search_time = time.time() - start_time
                st.session_state.last_results = results
                st.rerun()
