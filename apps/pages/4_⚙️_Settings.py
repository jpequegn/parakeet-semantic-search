"""Settings & Configuration - Streamlit Page."""

import streamlit as st

# Configure page
st.set_page_config(
    page_title="Settings - Parakeet",
    page_icon="⚙️",
    layout="wide",
)

st.title("⚙️ Settings & Configuration")
st.markdown("Customize your Parakeet experience")

# Create tabs for different setting categories
tab1, tab2, tab3, tab4 = st.tabs(
    ["General", "Cache", "Display", "Export"]
)

# Initialize settings in session state
if "settings" not in st.session_state:
    st.session_state.settings = {
        "dark_mode": True,
        "enable_notifications": True,
        "auto_save": True,
        "cache_enabled": True,
        "cache_ttl": 3600,
        "default_result_limit": 10,
        "default_similarity_threshold": 0.0,
        "export_format": "json",
        "include_metadata": True,
    }

with tab1:
    st.subheader("General Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Interface")

        dark_mode = st.toggle(
            "Dark Mode",
            value=st.session_state.settings["dark_mode"],
            help="Enable dark theme (requires page reload)",
        )
        st.session_state.settings["dark_mode"] = dark_mode

        notifications = st.toggle(
            "Enable Notifications",
            value=st.session_state.settings["enable_notifications"],
            help="Show search and recommendation notifications",
        )
        st.session_state.settings["enable_notifications"] = notifications

        auto_save = st.toggle(
            "Auto-Save Results",
            value=st.session_state.settings["auto_save"],
            help="Automatically save search results",
        )
        st.session_state.settings["auto_save"] = auto_save

    with col2:
        st.markdown("### Search Defaults")

        default_limit = st.slider(
            "Default Result Limit",
            min_value=1,
            max_value=50,
            value=st.session_state.settings["default_result_limit"],
            help="Default number of results to display",
        )
        st.session_state.settings["default_result_limit"] = default_limit

        default_threshold = st.slider(
            "Default Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.settings["default_similarity_threshold"],
            step=0.05,
            help="Default minimum similarity score",
        )
        st.session_state.settings["default_similarity_threshold"] = default_threshold

        language = st.selectbox(
            "Language",
            ["English", "Spanish", "French", "German"],
            help="Display language",
        )

with tab2:
    st.subheader("Cache Settings")

    col1, col2 = st.columns(2)

    with col1:
        cache_enabled = st.toggle(
            "Enable Caching",
            value=st.session_state.settings["cache_enabled"],
            help="Cache search results for faster access",
        )
        st.session_state.settings["cache_enabled"] = cache_enabled

    with col2:
        if cache_enabled:
            cache_ttl = st.slider(
                "Cache TTL (seconds)",
                min_value=60,
                max_value=86400,
                value=st.session_state.settings["cache_ttl"],
                step=300,
                help="How long to keep cached results",
            )
            st.session_state.settings["cache_ttl"] = cache_ttl

    st.divider()

    # Cache management
    st.subheader("Cache Management")

    col1, col2, col3 = st.columns(3)

    with col1:
        cache_size = 245  # Mock value
        st.metric("Cache Size", f"{cache_size} queries")

    with col2:
        hit_rate = 68
        st.metric("Hit Rate", f"{hit_rate}%")

    with col3:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.success("Cache cleared successfully!")

    # Cache status
    with st.expander("📊 Cache Details"):
        st.markdown("""
        - **Max Size**: 500 queries
        - **Current Size**: 245 queries (49%)
        - **Memory Usage**: ~12 MB
        - **Hit Rate**: 68%
        - **Miss Rate**: 32%
        - **Last Cleared**: 2 hours ago
        """)

with tab3:
    st.subheader("Display Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Table Display")

        rows_per_page = st.slider(
            "Rows per page",
            min_value=5,
            max_value=50,
            value=10,
            help="Number of rows to display in tables",
        )

        enable_search_highlight = st.toggle(
            "Highlight Search Terms",
            value=True,
            help="Highlight query terms in results",
        )

        show_scores = st.toggle(
            "Show Similarity Scores",
            value=True,
            help="Display similarity/distance scores",
        )

    with col2:
        st.markdown("### Visualization")

        chart_type = st.selectbox(
            "Default Chart Type",
            ["Line", "Bar", "Area", "Scatter"],
            help="Default chart style for analytics",
        )

        enable_animations = st.toggle(
            "Enable Animations",
            value=True,
            help="Animate charts and transitions",
        )

        color_theme = st.selectbox(
            "Color Theme",
            ["Purple", "Blue", "Green", "Orange"],
            help="Color scheme for visualizations",
        )

with tab4:
    st.subheader("Export Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Default Format")

        export_format = st.radio(
            "Default Export Format",
            ["JSON", "CSV"],
            help="Default format for exporting results",
        )
        st.session_state.settings["export_format"] = export_format.lower()

        include_metadata = st.toggle(
            "Include Metadata",
            value=st.session_state.settings["include_metadata"],
            help="Include query metadata in exports",
        )
        st.session_state.settings["include_metadata"] = include_metadata

    with col2:
        st.markdown("### Export Options")

        include_scores = st.toggle(
            "Include Similarity Scores",
            value=True,
            help="Include distance/similarity scores",
        )

        include_timestamps = st.toggle(
            "Include Timestamps",
            value=True,
            help="Include export timestamp",
        )

        compression = st.selectbox(
            "Compression",
            ["None", "GZIP"],
            help="Compress exported files",
        )

    st.divider()

    st.subheader("Export Preview")

    with st.expander("📋 Sample Export (JSON)"):
        st.json({
            "metadata": {
                "export_date": "2025-11-22T12:34:56",
                "query": "machine learning",
                "result_count": 10,
                "include_metadata": True,
            },
            "results": [
                {
                    "episode_id": "ep_001",
                    "episode_title": "Introduction to Machine Learning",
                    "podcast_title": "AI Today Podcast",
                    "similarity": 0.95,
                }
            ]
        })

# Save & Reset buttons
st.divider()

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("### Settings Management")

with col2:
    if st.button("💾 Save Settings", use_container_width=True):
        st.success("Settings saved successfully!")

with col3:
    if st.button("🔄 Reset to Default", use_container_width=True):
        st.session_state.settings = {
            "dark_mode": True,
            "enable_notifications": True,
            "auto_save": True,
            "cache_enabled": True,
            "cache_ttl": 3600,
            "default_result_limit": 10,
            "default_similarity_threshold": 0.0,
            "export_format": "json",
            "include_metadata": True,
        }
        st.success("Settings reset to defaults!")

# Settings info
with st.expander("ℹ️ About Settings"):
    st.markdown("""
    ### Your Settings
    - Settings are saved locally in your browser
    - You can export your settings and import them elsewhere
    - Reset to defaults at any time
    - All settings take effect immediately

    ### Privacy
    - No settings data is sent to our servers
    - Your preferences are stored locally only
    - You can clear all settings from your browser
    """)
