"""Analytics Dashboard - Streamlit Page."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random

# Configure page
st.set_page_config(
    page_title="Analytics - Parakeet",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Analytics Dashboard")
st.markdown("Track search metrics and system performance")

# Initialize session state for analytics
if "query_log" not in st.session_state:
    st.session_state.query_log = []

if "search_times" not in st.session_state:
    st.session_state.search_times = []

# Sample data generation for demo
@st.cache_data
def generate_sample_data():
    """Generate sample analytics data for demo."""
    queries = [
        "machine learning",
        "deep learning",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "neural networks",
        "transformers",
        "embeddings",
    ]

    topics = {
        "Machine Learning": 35,
        "Deep Learning": 28,
        "NLP": 22,
        "Computer Vision": 18,
        "Reinforcement Learning": 12,
        "Transfer Learning": 8,
        "Other": 5,
    }

    podcasts = {
        "AI Today Podcast": 25,
        "Tech Trends Weekly": 22,
        "Data Science Daily": 18,
        "ML Engineering": 15,
        "AI Research": 12,
    }

    # Search performance data
    search_times = [
        random.uniform(0.05, 0.2) for _ in range(100)
    ]

    return queries, topics, podcasts, search_times


queries, topics, podcasts, search_times = generate_sample_data()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Trending Topics", "System Performance", "Podcast Analytics"]
)

with tab1:
    st.subheader("System Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Queries",
            "1,247",
            "+12%",
            help="Total search queries executed",
        )

    with col2:
        st.metric(
            "Avg Response Time",
            "120ms",
            "-5%",
            help="Average search latency",
        )

    with col3:
        st.metric(
            "Cache Hit Rate",
            "68%",
            "+3%",
            help="Percentage of cached results",
        )

    with col4:
        st.metric(
            "System Uptime",
            "99.9%",
            "stable",
            help="System availability",
        )

    st.divider()

    # Query volume over time
    st.subheader("Query Volume Trend")

    dates = pd.date_range(start="2025-11-01", end="2025-11-22", freq="D")
    volumes = [random.randint(30, 100) for _ in dates]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=volumes,
            mode="lines+markers",
            name="Queries",
            line=dict(color="#667eea", width=3),
            fill="tozeroy",
        )
    )

    fig.update_layout(
        title="Daily Query Volume",
        xaxis_title="Date",
        yaxis_title="Number of Queries",
        hovermode="x unified",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Trending Topics")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Topic distribution pie chart
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(topics.keys()),
                    values=list(topics.values()),
                    hole=0.3,
                )
            ]
        )

        fig.update_layout(
            title="Topic Distribution",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Top Topics")
        for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True):
            st.metric(topic, count)

    st.divider()

    # Topic trend
    st.subheader("Topic Trends Over Time")

    selected_topics = st.multiselect(
        "Select topics to compare",
        list(topics.keys()),
        default=["Machine Learning", "Deep Learning", "NLP"],
    )

    if selected_topics:
        trend_data = {}
        for topic in selected_topics:
            values = [
                random.randint(5, 20) for _ in range(22)
            ]
            trend_data[topic] = values

        trend_df = pd.DataFrame(
            trend_data,
            index=pd.date_range(start="2025-11-01", periods=22, freq="D"),
        )

        fig = go.Figure()
        for column in trend_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=trend_df.index,
                    y=trend_df[column],
                    mode="lines+markers",
                    name=column,
                )
            )

        fig.update_layout(
            title="Topic Trends",
            xaxis_title="Date",
            yaxis_title="Query Count",
            hovermode="x unified",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("System Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:
        # Response time distribution
        fig = go.Figure(
            data=[go.Histogram(x=search_times, nbinsx=30, name="Response Time")]
        )

        fig.update_layout(
            title="Response Time Distribution",
            xaxis_title="Time (seconds)",
            yaxis_title="Frequency",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Performance metrics table
        st.markdown("### Performance Statistics")

        import numpy as np

        stats = {
            "Metric": [
                "Mean",
                "Median",
                "Min",
                "Max",
                "Std Dev",
                "P95",
                "P99",
            ],
            "Value (ms)": [
                f"{np.mean(search_times) * 1000:.1f}",
                f"{np.median(search_times) * 1000:.1f}",
                f"{np.min(search_times) * 1000:.1f}",
                f"{np.max(search_times) * 1000:.1f}",
                f"{np.std(search_times) * 1000:.1f}",
                f"{np.percentile(search_times, 95) * 1000:.1f}",
                f"{np.percentile(search_times, 99) * 1000:.1f}",
            ],
        }

        st.dataframe(
            pd.DataFrame(stats),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # Cache performance
    st.subheader("Cache Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Requests", "2,847")

    with col2:
        st.metric("Cache Hits", "1,935 (68%)")

    with col3:
        st.metric("Cache Misses", "912 (32%)")

    # Cache hit rate over time
    cache_dates = pd.date_range(start="2025-11-15", end="2025-11-22", freq="D")
    hit_rates = [
        random.randint(50, 80) for _ in cache_dates
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cache_dates,
            y=hit_rates,
            name="Hit Rate (%)",
            marker_color="#667eea",
        )
    )

    fig.update_layout(
        title="Cache Hit Rate Trend",
        xaxis_title="Date",
        yaxis_title="Hit Rate (%)",
        height=300,
    )

    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Podcast Analytics")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Podcast bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(podcasts.keys()),
                    y=list(podcasts.values()),
                    marker_color="#667eea",
                )
            ]
        )

        fig.update_layout(
            title="Results by Podcast",
            xaxis_title="Podcast",
            yaxis_title="Number of Results",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Podcast Distribution")
        total = sum(podcasts.values())
        for podcast, count in sorted(
            podcasts.items(), key=lambda x: x[1], reverse=True
        ):
            percentage = (count / total) * 100
            st.metric(podcast, f"{count} ({percentage:.1f}%)")

    st.divider()

    # Podcast statistics
    st.subheader("Podcast Performance")

    podcast_stats = {
        "Podcast": list(podcasts.keys()),
        "Results": list(podcasts.values()),
        "Avg Relevance": [f"{random.uniform(0.75, 0.95):.2%}" for _ in podcasts],
        "Avg Search Time (ms)": [f"{random.randint(80, 150)}" for _ in podcasts],
    }

    st.dataframe(
        pd.DataFrame(podcast_stats),
        use_container_width=True,
        hide_index=True,
    )

# Footer
st.divider()
st.markdown(
    """
    **Note:** Analytics data is cached and updated periodically.
    Last update: Just now
    """
)
