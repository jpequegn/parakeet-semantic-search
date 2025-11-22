"""Recommendations Interface - Streamlit Page."""

import streamlit as st
from typing import List
from apps.utils.search_utils import SearchManager, format_results_table
from apps.utils.export_utils import create_download_button_data, format_metadata

# Configure page
st.set_page_config(
    page_title="Recommendations - Parakeet",
    page_icon="💡",
    layout="wide",
)

st.title("💡 Smart Recommendations")
st.markdown("Discover episodes similar to your favorites")

# Initialize search manager
search_mgr = SearchManager()

# Sidebar Controls
with st.sidebar:
    st.subheader("Recommendation Settings")

    rec_mode = st.radio(
        "Recommendation Mode",
        ["Single Episode", "Hybrid (Multi-Episode)"],
        help="Choose how to generate recommendations",
    )

    limit = st.slider(
        "Number of recommendations",
        min_value=1,
        max_value=20,
        value=5,
    )

    diversity = st.slider(
        "Diversity boost",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Boost diversity of results (0 = disabled)",
    )

# Main Content
tab1, tab2 = st.tabs(["Get Recommendations", "History"])

with tab1:
    if rec_mode == "Single Episode":
        st.subheader("🎯 Find Similar Episodes")
        st.markdown("Select an episode to find similar content")

        col1, col2 = st.columns([3, 1])

        with col1:
            episode_id = st.text_input(
                "Episode ID",
                placeholder="e.g., ep_001",
                help="Enter the episode ID",
            )

        with col2:
            podcast_filter = st.text_input(
                "Filter by Podcast (optional)",
                placeholder="pod_001",
                help="Leave empty to show all podcasts",
            )

        if st.button("🔍 Get Recommendations", use_container_width=True):
            if episode_id:
                with st.spinner("Finding similar episodes..."):
                    try:
                        results = search_mgr.get_recommendations(
                            episode_id=episode_id,
                            limit=limit,
                            podcast_id=podcast_filter if podcast_filter else None,
                        )

                        if results:
                            st.session_state.last_recommendations = results
                            st.session_state.rec_episode = episode_id
                            st.success(f"Found {len(results)} recommendations!")
                        else:
                            st.error(f"Episode {episode_id} not found or no recommendations available.")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter an Episode ID")

    else:  # Hybrid Mode
        st.subheader("🎯 Hybrid Recommendations")
        st.markdown("Combine multiple episodes to find tailored recommendations")

        col1, col2 = st.columns([3, 1])

        with col1:
            episode_list = st.text_area(
                "Episode IDs (one per line)",
                placeholder="ep_001\nep_002\nep_003",
                height=100,
                help="Enter episode IDs separated by newlines",
            )

        with col2:
            podcast_filter = st.text_input(
                "Filter by Podcast (optional)",
                placeholder="pod_001",
                help="Leave empty to show all podcasts",
            )

        if st.button("🔍 Get Hybrid Recommendations", use_container_width=True):
            if episode_list.strip():
                episode_ids = [e.strip() for e in episode_list.strip().split("\n") if e.strip()]

                if len(episode_ids) > 10:
                    st.warning("Maximum 10 episodes supported for hybrid recommendations")
                else:
                    with st.spinner("Generating hybrid recommendations..."):
                        try:
                            results = search_mgr.get_hybrid_recommendations(
                                episode_ids=episode_ids,
                                limit=limit,
                                podcast_id=podcast_filter if podcast_filter else None,
                            )

                            if results:
                                st.session_state.last_recommendations = results
                                st.session_state.rec_episodes = episode_ids
                                st.success(f"Found {len(results)} recommendations!")
                            else:
                                st.error("No recommendations found. Check episode IDs.")

                        except Exception as e:
                            st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter at least one Episode ID")

    # Display Recommendations
    if "last_recommendations" in st.session_state and st.session_state.last_recommendations:
        results = st.session_state.last_recommendations
        st.divider()
        st.subheader("Recommended Episodes")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.metric("Recommendations", len(results))
        with col2:
            if st.button("Clear", key="clear_rec"):
                del st.session_state.last_recommendations
                st.rerun()

        # Tabs for different views
        tab_table, tab_detail, tab_export = st.tabs(["List", "Details", "Export"])

        with tab_table:
            formatted_results = format_results_table(results)
            st.dataframe(
                formatted_results,
                use_container_width=True,
                column_config={
                    "similarity": st.column_config.ProgressColumn(
                        "Similarity",
                        min_value=0,
                        max_value=1,
                    ),
                },
            )

        with tab_detail:
            for idx, result in enumerate(results, 1):
                with st.expander(
                    f"{idx}. {result.get('episode_title', 'Untitled')}",
                    expanded=(idx == 1),
                ):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**Episode:** {result.get('episode_title')}")
                        st.markdown(f"**Podcast:** {result.get('podcast_title')}")
                        st.markdown(f"**Episode ID:** `{result.get('episode_id')}`")
                        st.markdown(f"**Podcast ID:** `{result.get('podcast_id')}`")

                    with col2:
                        similarity = 1 - result.get("_distance", 0)
                        st.metric("Similarity", f"{similarity:.1%}")

        with tab_export:
            col1, col2 = st.columns(2)

            with col1:
                formatted_results = format_results_table(results)
                csv_data, csv_filename, csv_mime = create_download_button_data(
                    formatted_results,
                    format="csv",
                    filename="recommendations.csv",
                )
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=csv_filename,
                    mime=csv_mime,
                    use_container_width=True,
                )

            with col2:
                metadata = format_metadata(
                    query="recommendations",
                    result_count=len(results),
                )
                json_data, json_filename, json_mime = create_download_button_data(
                    formatted_results,
                    format="json",
                    filename="recommendations.json",
                    metadata=metadata,
                )
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=json_filename,
                    mime=json_mime,
                    use_container_width=True,
                )

with tab2:
    st.subheader("Recommendation History")
    st.info("History of previously generated recommendations will appear here")
    st.markdown(
        """
        - Shows your recent recommendation requests
        - Allows quick re-runs of previous queries
        - Displays trending recommendations
        """
    )
