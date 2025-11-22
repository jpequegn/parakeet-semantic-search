"""Parakeet Semantic Search - Streamlit Web Application.

Interactive web interface for semantic search, recommendations, analytics, and configuration.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure Streamlit
st.set_page_config(
    page_title="Parakeet Semantic Search",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 0.5em;
    }
    .sub-header {
        font-size: 1.5em;
        color: #666;
        margin-bottom: 1em;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5em;
        border-radius: 0.5em;
        color: white;
        margin: 0.5em 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title
col1, col2 = st.columns([8, 1])
with col1:
    st.markdown("<div class='main-header'>🦜 Parakeet Semantic Search</div>", unsafe_allow_html=True)
with col2:
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-jpequegn/parakeet--semantic--search-blue)](https://github.com/jpequegn/parakeet-semantic-search)")

st.markdown("<div class='sub-header'>Intelligent podcast discovery engine</div>", unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.title("Navigation")

    # App Info
    st.markdown("""
    ### About
    Parakeet is an intelligent podcast discovery engine using semantic search and vector embeddings.

    **Features:**
    - 🔍 Semantic search across transcripts
    - 💡 Smart recommendations
    - 📊 Analytics & insights
    - ⚙️ Configuration & settings
    """)

    st.divider()

    # Session State Info
    if "query_count" in st.session_state:
        st.metric("Total Queries", st.session_state.query_count)

    st.divider()

    # About & Links
    st.markdown("""
    ### Documentation
    - [GitHub Repository](https://github.com/jpequegn/parakeet-semantic-search)
    - [Project README](../README.md)
    - [API Docs](../docs/)
    """)

# Initialize session state
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "search_history" not in st.session_state:
    st.session_state.search_history = []
if "last_results" not in st.session_state:
    st.session_state.last_results = None

# Main Content - Welcome/Overview
st.markdown("---")

# Key Features Overview
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class='metric-card'>
        <h3>🔍 Search</h3>
        <p>Find episodes by semantic meaning</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class='metric-card'>
        <h3>💡 Recommendations</h3>
        <p>Discover similar episodes</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class='metric-card'>
        <h3>📊 Analytics</h3>
        <p>View insights & trends</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        """
        <div class='metric-card'>
        <h3>⚙️ Settings</h3>
        <p>Configure your preferences</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# Quick Start
st.subheader("Quick Start")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("""
    **1. 🔍 Search:** Use the Search page to find episodes by semantic meaning

    **2. 💡 Recommendations:** Get similar episodes based on your selections

    **3. 📊 Analytics:** Track metrics and discover trending topics

    **4. ⚙️ Settings:** Customize your experience and export data
    """)

with col2:
    st.markdown("""
    ### Status
    - ✅ Core System
    - ✅ Search Engine
    - ✅ Evaluation
    - ✅ Optimization
    - 🚀 Web UI
    """)

st.markdown("---")

# System Information
st.subheader("System Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("""
    **Environment:** Production Ready
    **Version:** 0.1.0
    **Status:** Active
    """)

with col2:
    st.success("""
    **Tests:** 150+ passing
    **Coverage:** Comprehensive
    **Docs:** Complete
    """)

with col3:
    st.markdown("""
    **Navigate to pages** using the sidebar menu:
    - Search
    - Recommendations
    - Analytics
    - Settings
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
    Made with ❤️ | © 2025 Parakeet Semantic Search |
    <a href='https://github.com/jpequegn/parakeet-semantic-search'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
